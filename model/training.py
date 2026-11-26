import json
import time
from pathlib import Path
from datetime import datetime

import numpy as np
import torch

from data_loader import PCLDataModule
from model_classes import PCLModel

def next_batch(iterators, loaders, name):
    """Next batch from one proxy loader; restart that loader when it runs out."""
    try:
        batch = next(iterators[name])
    except StopIteration:
        iterators[name] = iter(loaders[name]) 
        batch = next(iterators[name]) 
    return batch

def train_one_epoch(model, loaders, iterators, optimizer, lambdas):
    model.train()                                   # dropout ON
    mse = torch.nn.MSELoss()
    totals = {"site": 0.0, **{name: 0.0 for name in lambdas}}
    n_steps = 0

    for x_obs, x_loc, y, idx in loaders["train"]:
        optimizer.zero_grad()                       # 1. clear last step's gradients

        loss_site = mse(model.forward(x_obs, x_loc), y)          # 2. site prediction vs target
        loss = loss_site

        for name in lambdas:                        # 3. one batch per proxy
            px, py = next_batch(iterators, loaders["proxy"], name)
            loss_p = mse(model.forward_proxy(px, name), py)        #    that proxy's prediction vs its target
            loss = loss + lambdas[name] * loss_p
            totals[name] += loss_p.item()

        loss.backward()                             # 4. backpropagate the total loss
        optimizer.step()                            # 5. update the weights

        totals["site"] += loss_site.item()
        n_steps += 1

    return {k: v / n_steps for k, v in totals.items()}   # average losses this epoch

@torch.no_grad()
def evaluate(model, loader, target_stats):
    # 1. dropout off
    model.eval()

    # 2. loop over the loader: predict each batch, collect predictions, targets, and idx in lists
    preds, ys, idxs = [], [], []
    for x_obs, x_loc, y, idx in loader:
        pred = model.forward(x_obs, x_loc)
        preds.append(pred); ys.append(y); idxs.append(idx)

    # 3. join each list into one tensor
    preds, ys, idxs = torch.cat(preds), torch.cat(ys), torch.cat(idxs)

    # 4. validation loss: MSE between predictions and targets (still standardized)
    mse = torch.nn.MSELoss()
    val_loss = mse(preds, ys).item()

    # 5. convert predictions and targets back to ppb using target_stats["mean"] and ["std"]
    pred_ppb = preds*target_stats['std'] + target_stats['mean']
    y_ppb    = ys*target_stats['std'] + target_stats['mean']

    # 6. RMSE and R2 in ppb
    rmse = torch.sqrt(mse(pred_ppb, y_ppb)).item()
    r2 = (1 - ((pred_ppb - y_ppb) ** 2).sum() / ((y_ppb - y_ppb.mean()) ** 2).sum()).item()

    metrics = {"val_loss": val_loss, "RMSE": rmse, "R2": r2}
    # 7. return the metrics (dict), the ppb predictions, and idx
    return metrics, pred_ppb, idxs

def save_checkpoint(path, model, data, config, metrics):
    """Weights + everything needed to use the model later (columns, stats, config, metrics)."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)                      # make the run folder

    torch.save(model.state_dict(), path / "model.pt")             # the weights

    def to_float(stats):                                          # numpy floats -> plain floats (JSON can't write numpy)
        return {col: {k: float(v) for k, v in s.items()} for col, s in stats.items()}

    meta = {
        "config":      config,
        "obs_cols":    data.obs_cols,
        "loc_cols":    data.loc_cols,
        "site_stats":  to_float(data.table_train_stats),
        "proxy_stats": {name: to_float(s) for name, s in (getattr(data, "proxy_table_stats", None) or {}).items()},
        "metrics":     metrics,
    }
    with open(path / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

def train(config, run_dir=None):
    if run_dir is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M")
        run_dir = f"runs/{config['TARG_VAR'].lower()}_{config['FOLD_TYPE'].lower()}_fold{config['FOLD_SEL']}_{stamp}"

    # 1. data: build the PCLDataModule and run its four steps (attach, fold, transform, loaders)
    data = PCLDataModule(config)
    data.attach_statics()
    data.select_fold()
    data.fit_and_transform()
    loaders = data.dataloaders()

    # 2. model: build PCLModel from data.obs_cols, data.loc_cols, the proxy names, and config["MODEL"]
    model = PCLModel(obs_cols=data.obs_cols, loc_cols=data.loc_cols, proxy_names=config["PCL_VARS"] or [], **config["MODEL"])

    # 3. optimizer: Adam on the model's parameters, learning rate from config["TRAIN"]
    optimizer = torch.optim.Adam(model.parameters(), lr=config["TRAIN"]["LR"])

    # 4. proxy bookmarks: one iterator per proxy loader (the dict next_batch uses)
    iterators = {name: iter(loaders['proxy'][name]) for name in loaders['proxy'].keys()}

    # 5. epoch loop (up to max_epochs)
    max_epochs = config['TRAIN']['EPOCHS']
    
    counter = 0
    for epoch in range(max_epochs):
        # a. train one epoch -> average training losses
        avg_losses = train_one_epoch(model, loaders, iterators, optimizer, config['LAMBDAS'] or {})

        # b. evaluate on the test loader -> metrics
        metrics, _, _ = evaluate(model, loaders['test'], data.table_train_stats[config['TARG_VAR']])

        # c. print a one-line summary
        proxy_str = "".join(f" | {name} {avg_losses[name]:.4f}" for name in (config["LAMBDAS"] or {}))
        print(f"epoch {epoch + 1:>3} | train site {avg_losses['site']:.4f}{proxy_str} | "
              f"val loss {metrics['val_loss']:.4f} | RMSE {metrics['RMSE']:.2f} ppb | R2 {metrics['R2']:.3f}")

        # d. if val_loss improved: remember it, save a checkpoint, reset the patience counter

        if epoch == 0:
            best_val_loss = metrics['val_loss']
            save_checkpoint(run_dir, model, data, config, metrics)
        elif metrics['val_loss'] < best_val_loss:
            best_val_loss = metrics['val_loss']
            save_checkpoint(run_dir, model, data, config, metrics)
            counter = 0
        else:
            counter += 1

        #    otherwise: add one to the counter; stop if it reaches patience
        if counter == config['TRAIN']['PATIENCE']:
            break
    # 6. return the best val_loss
    return best_val_loss

if __name__ == "__main__":
    from configs.test_config import CONFIG
    train(CONFIG)