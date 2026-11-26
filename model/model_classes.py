import torch
import torch.nn as nn


class FourierFeatures(nn.Module):
    def __init__(self, n_freq):
        super().__init__()
        """Fixed frequencies for expanding lat/lon into sin/cos."""
        freqs = 2.0**torch.arange(n_freq)
        self.register_buffer("freqs", freqs)

    def forward(self, latlon):
        """(batch, 2) raw lat/lon -> (batch, 4 * n_freq) sin/cos features."""
        x = torch.deg2rad(latlon)
        x = x.unsqueeze(-1) * self.freqs
        x = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        x = x.flatten(start_dim=1)
        return x

class MLP(nn.Module):
    def __init__(self, n_in, n_out, n_layers, n_hidden, dropout):
        """Small stack of Linear + SiLU layers (reused by every encoder and head)."""
        super().__init__()
        layers, d, = [], n_in
        for _ in range(max(1, n_layers)):
            layers += [nn.Linear(d, n_hidden), nn.LayerNorm(n_hidden), nn.SiLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = n_hidden
        layers.append(nn.Linear(d, n_out))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class PCLModel(nn.Module):
    def __init__(self, obs_cols, loc_cols, proxy_names: list, n_freq, n_hidden, n_hidden_proxy, n_layers_enc, n_latent, n_layers_pred, dropout):
        """Build the Fourier features, both encoders, the O3 head, and one head per proxy."""
        super().__init__()
        n_obs = len(obs_cols)
        n_loc = ((len(loc_cols)-2) + 4 * n_freq)

        self.proxy_names = proxy_names
        self.latlon_idx = [loc_cols.index('latitude'), loc_cols.index('longitude')]
        self.other_idxs = [loc_cols.index(col) for col in loc_cols if col not in ['latitude','longitude']]

        self.ff_ft = FourierFeatures(n_freq=n_freq)
        self.e_obs = MLP(n_in=n_obs, n_out=n_latent, n_layers=n_layers_enc, n_hidden=n_hidden, dropout=dropout)
        self.e_loc = MLP(n_in=n_loc, n_out=n_latent, n_layers=n_layers_enc, n_hidden=n_hidden, dropout=dropout)
        self.pred  = MLP(n_in=n_latent*2, n_out=1, n_layers=n_layers_pred, n_hidden=n_hidden, dropout=dropout)
        self.proxy = nn.ModuleDict()
        for p in self.proxy_names:
            self.proxy[p] = MLP(n_in=n_latent, n_out=1, n_layers=1, n_hidden=n_hidden_proxy, dropout=dropout)
    
    def embed_loc(self, x_loc):
        """lat/lon -> Fourier features, joined with the rest of x_loc -> h_loc."""
        latlon = x_loc[:, [self.latlon_idx[0], self.latlon_idx[1]]]
        latlon_enc = self.ff_ft.forward(latlon=latlon)

        other_cols = x_loc[:, self.other_idxs]
        full_loc   = torch.cat([latlon_enc, other_cols], dim=1)

        return self.e_loc.forward(full_loc)

    def forward(self, x_obs, x_loc):
        """Site rows: h_obs + h_loc -> predicted target (standardized)."""
        obs_emb = self.e_obs(x_obs)
        loc_emb = self.embed_loc(x_loc)
        tot_emb = torch.cat([obs_emb, loc_emb], dim=1)

        return self.pred(tot_emb).squeeze(1)

    def forward_proxy(self, x_loc, name):
        """Proxy rows: h_loc only -> predicted proxy (standardized)."""
        loc_emb = self.embed_loc(x_loc)
        return self.proxy[name].forward(loc_emb).squeeze(1)

if __name__ == "__main__":
    from configs.test_config import CONFIG

    # column lists built the same way PCLDataModule builds them
    obs_cols = list(CONFIG["OBS_VARS"])
    loc_cols = list(CONFIG["LOC_VARS"])
    if CONFIG["EMBEDDINGS"]:
        loc_cols += [f"emb_{i}" for i in range(64)]
    if CONFIG["TERRAIN"]:
        loc_cols += ["terrain_height"]
    proxy_names = CONFIG["PCL_VARS"] or []

    model = PCLModel(obs_cols=obs_cols, loc_cols=loc_cols, proxy_names=proxy_names, **CONFIG["MODEL"])
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")

    # random inputs with realistic lat/lon
    B = 256
    x_obs = torch.randn(B, len(obs_cols))
    x_loc = torch.randn(B, len(loc_cols))
    x_loc[:, loc_cols.index("latitude")]  = torch.empty(B).uniform_(24, 50)
    x_loc[:, loc_cols.index("longitude")] = torch.empty(B).uniform_(-125, -66.5)

    # forward passes
    pred = model.forward(x_obs, x_loc)
    print(f"site prediction: {tuple(pred.shape)}  (expected ({B},))")
    loss = (pred ** 2).mean()
    for name in proxy_names:
        p = model.forward_proxy(x_loc, name)
        print(f"proxy {name}: {tuple(p.shape)}  (expected ({B},))")
        loss = loss + (p ** 2).mean()

    # every weight should receive a gradient
    loss.backward()
    no_grad = [n for n, p in model.named_parameters() if p.grad is None]
    print("parameters without gradients:", no_grad if no_grad else "none")

    # dropout off -> identical predictions on repeat
    model.eval()
    with torch.no_grad():
        same = torch.allclose(model.forward(x_obs, x_loc), model.forward(x_obs, x_loc))
    print("eval mode deterministic:", same)