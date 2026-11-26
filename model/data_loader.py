"""
PCL data module: turns the site and proxy tables into normalized train / validation sets.

Order of use:
    data = PCLDataModule(config)   # load tables, drop NaNs, fixed transforms
    data.attach_statics()          # AlphaEarth + terrain (before stats, so terrain gets standardized)
    data.select_fold()             # A) train / validation split
    data.fit_and_transform()       # B) standardize with TRAIN statistics
    data.dataloaders(batch_size)   # C) PyTorch DataLoaders
"""

import numpy as np
import pandas as pd
import xarray as xr
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.cluster import AgglomerativeClustering

# Fixed transforms, applied right after loading (same formula for every row, no stats needed).
# Remove an entry to switch that transform off.
FIXED_TRANSFORMS = {
    "tempo_no2":  lambda x: np.arcsinh(x / 1e15),   # compress the city tail; handles negatives
    "tempo_hcho": lambda x: x / 1e16,               # scale only; keeps negatives
    "hrrr_blh":   np.log,                           # BLH spans ~50 m to several km
}

# Proxy targets are all called 'vertical_column_troposphere', so their transform goes by table name
PROXY_TRANSFORMS = {
    "NO2":  lambda x: np.arcsinh(x / 1e15),
    "HCHO": lambda x: x / 1e16,
    # O3 and RATIO: none (RATIO is already a log), just standardized later
}

# Columns that get (x - mean) / std with TRAIN statistics, if they are present in a table.
# Left alone on purpose: site, time, latitude, longitude, sin/cos time features,
# direct flag, AlphaEarth bands (already unit vectors).
STANDARDIZE = ["hrrr_blh", "hrrr_ssrd", "hrrr_sp", "hrrr_gust", "hrrr_mslp", "hrrr_t2m",
               "hrrr_d2m", "hrrr_rh", "hrrr_u10", "hrrr_v10", "hrrr_tcc",
               "tempo_no2", "tempo_hcho", "tempo_o3", "tempo_ratio",
               "no2_offset_min", "hcho_offset_min", "o3_offset_min", "ratio_offset_min",
               "O3", "NO2", "terrain_height", "vertical_column_troposphere"]

class PCLDataModule:
    def __init__(self, config: dict):
        """Load the tables once and keep the fold choice.

        config keys used here:
            TARG_VAR : target column and table name, e.g. "O3"
            OBS_VARS : columns for the observation encoder
            LOC_VARS : columns for the location-time encoder (only those stored in the table)
            PCL_VARS : proxy tables to load, e.g. ["O3", "RATIO"]
        """
        self.config = config

        # ---- site table: only the columns this run needs, then drop incomplete rows ----
        self.training_columns = [self.config['TARG_VAR']] + self.config['OBS_VARS'] \
                                 + self.config['LOC_VARS'] + ['site', 'time']

        training_table = pd.read_parquet(rf"../master_datasets/training_tables/{self.config['TARG_VAR']}.parquet", \
                                         columns=self.training_columns)

        self.training_table = training_table.dropna().reset_index(drop=True)

        # ---- columns for each encoder (new lists, so CONFIG itself is never modified) ----
        self.obs_cols = list(self.config['OBS_VARS'])
        self.loc_cols = list(self.config['LOC_VARS'])
        if self.config['EMBEDDINGS']:
            self.loc_cols += [f'emb_{i}' for i in range(64)]       # attached in attach_statics()
        if self.config['TERRAIN']:
            self.loc_cols += ['terrain_height']                            # attached in attach_statics()

        # Fixed transforms (only for columns loaded in this run)
        for col, f in FIXED_TRANSFORMS.items():
            if col in self.training_table.columns:
                self.training_table[col] = f(self.training_table[col])

        # ---- proxy tables: one per proxy, each with its own fixed transform ----
        self.proxy_tables = None
        if self.config['PCL_VARS'] is not None:
            self.proxy_tables = {f"{var}": None for var in self.config['PCL_VARS']}
            for var in self.config['PCL_VARS']:
                self.proxy_tables[var] = pd.read_parquet(rf"../master_datasets/proxy_tables/{var}.parquet")
                self.proxy_tables[var] = self.proxy_tables[var].dropna().reset_index(drop=True)
                if var in PROXY_TRANSFORMS:
                    col = "vertical_column_troposphere"
                    self.proxy_tables[var][col] = PROXY_TRANSFORMS[var](self.proxy_tables[var][col])

    def attach_statics(self):
        """Join site-year AlphaEarth + terrain onto the site rows;
        load the static grids for the proxy lookups."""

        if self.config['EMBEDDINGS']:
            embedding_arrays = {2023: None, 2024: None, 2025:None}
            for year in embedding_arrays.keys():
                with xr.open_zarr(rf"../master_datasets/alpha_earth/alphaearth_conus_{year}_0.02.zarr") as ds:
                    values = ds.drop_vars("emb_heterogeneity").to_array().values
                    embedding_arrays[year] = values
            
            #TRAINING TABLE
            self.training_table['i'] = np.rint((self.training_table['latitude'] - 24.01) / 0.02).astype(int)
            self.training_table['j'] = np.rint((self.training_table['longitude'] - (-124.99)) / 0.02).astype(int)
            self.training_table['year'] = self.training_table['time'].dt.year

            sites = self.training_table[['site','latitude','longitude','year','i','j']].drop_duplicates(['site','year']).reset_index(drop=True)
            embeddings = np.zeros((len(sites), 64), dtype=np.float32)
            for i, row in sites.iterrows():
                embeddings[i, :] = embedding_arrays[row['year']][:, row['i'], row['j']]

            emb_cols = [f'emb_{b}' for b in range(64)]
            sites[emb_cols] = embeddings                      # creates the columns as floats
            self.training_table = self.training_table.merge(sites[['site', 'year'] + emb_cols], on=['site', 'year'], how='left')
            self.training_table = self.training_table.dropna().reset_index(drop=True)

            if self.proxy_tables is not None:
                emb_cols = [f"emb_{b}" for b in range(64)]
                for key in self.proxy_tables.keys():
                    proxy_table = self.proxy_tables[key]
                    proxy_table['i']    = np.rint((proxy_table['latitude'] - 24.01) / 0.02).astype(int)
                    proxy_table['j']    = np.rint((proxy_table['longitude'] - (-124.99)) / 0.02).astype(int)
                    proxy_table['year'] = proxy_table['pass_time'].dt.year
                    proxy_table[emb_cols] = np.full((len(proxy_table), 64), np.nan, dtype=np.float32)
                    for year, grid in embedding_arrays.items():
                        rows = proxy_table["year"] == year
                        i = proxy_table.loc[rows, "i"].values
                        j = proxy_table.loc[rows, "j"].values
                        proxy_table.loc[rows, emb_cols] = grid[:, i, j].T
                    self.proxy_tables[key] = proxy_table.dropna().reset_index(drop=True)
                
        if self.config['TERRAIN']:
            with xr.open_dataset(r"../master_datasets/tempo_terrain_height.nc") as ds:
                embedding_array = ds.to_array().values

            self.training_table['i'] = np.rint((self.training_table['latitude'] - 24.01) / 0.02).astype(int)
            self.training_table['j'] = np.rint((self.training_table['longitude'] - (-124.99)) / 0.02).astype(int)
            sites = self.training_table[['site','latitude','longitude','i','j']].drop_duplicates(['site']).reset_index(drop=True)

            embedding = np.zeros((len(sites), 1), dtype=np.float32)
            for i, row in sites.iterrows():
                embedding[i, :] = embedding_array[:, row['i'], row['j']]

            sites['terrain_height'] = embedding.reshape(-1)                  
            self.training_table = self.training_table.merge(sites[['site', 'terrain_height']], on=['site'], how='left')
            self.training_table = self.training_table.dropna().reset_index(drop=True)

            if self.proxy_tables is not None:
                for key in self.proxy_tables.keys():
                    proxy_table = self.proxy_tables[key]
                    proxy_table['i']    = np.rint((proxy_table['latitude'] - 24.01) / 0.02).astype(int)
                    proxy_table['j']    = np.rint((proxy_table['longitude'] - (-124.99)) / 0.02).astype(int)
                    proxy_table['year'] = proxy_table['pass_time'].dt.year
                    proxy_table['terrain_height'] = np.full(len(proxy_table), np.nan, dtype=np.float32)

                    i = proxy_table["i"].values
                    j = proxy_table["j"].values
                    proxy_table['terrain_height'] = embedding_array[0, i, j]
                    self.proxy_tables[key] = proxy_table.dropna().reset_index(drop=True)
            
        else:
            pass

    def select_fold(self):
        """A) Split rows into train / val, by held-out sites (LOSO) or a held-out time block.

        Sets self.table_train and self.table_test (site rows), and filters the proxy tables.
        """
        fold      = self.config['FOLD_SEL']      # which fold is held out (int, 0-4)
        fold_type = self.config['FOLD_TYPE']     # "TEMPORAL" or "SPATIAL"

        if fold_type == "TEMPORAL":
            # Temporal fold selection, for held-out fold k:
            #
            #   k-1 = fold(time - 1 day),  k = fold(time),  k+1 = fold(time + 1 day)
            #
            #   k-1     k       k+1     ->  row is
            #   ----    ----    ----        ----------------------------------------------
            #   any     == k    any         VALIDATION
            #   != k    != k    != k        TRAINING
            #   == k    != k    != k        BUFFER (just after a held-out week)  -> drop
            #   != k    != k    == k        BUFFER (just before a held-out week) -> drop

            def add_fold_ids(table, time_column, day0='2023-01-01', n_folds=5):
                """Week-block fold of each row, and of the same row shifted -1 / +1 day."""
                day0 = pd.to_datetime(day0)

                table['k-1'] = ((pd.to_datetime(table[time_column]) - pd.Timedelta(days=1)) - day0).dt.days // 7 % n_folds
                table['k']   = (pd.to_datetime(table[time_column]) - day0).dt.days // 7 % n_folds
                table['k+1'] = ((pd.to_datetime(table[time_column]) + pd.Timedelta(days=1)) - day0).dt.days // 7 % n_folds

                return table

            # site rows are labelled by the EPA hour; proxy rows by the TEMPO pass time
            self.training_table = add_fold_ids(self.training_table, 'time')

            if self.proxy_tables is not None:
                for key in self.proxy_tables.keys():
                    self.proxy_tables[key] = add_fold_ids(self.proxy_tables[key], 'pass_time')

            def train_test_split(table, train: bool):
                """Training rows (held-out weeks and buffer days removed) or validation rows."""
                if train:
                    table = table[(table['k'] != fold) & (table['k-1'] != fold) & (table['k+1'] != fold)]
                else:
                    table = table[table['k'] == fold]

                return table.copy()

            self.table_train = train_test_split(self.training_table, train=True)
            self.table_test  = train_test_split(self.training_table, train=False)

            # Remove validation weeks (and buffer rows) from the proxy tables so the model doesn't train on them
            if self.proxy_tables is not None:
                for key in self.proxy_tables.keys():
                    self.proxy_tables[key] = train_test_split(self.proxy_tables[key], train=True)

        elif fold_type == "SPATIAL":
            
            def add_fold_ids(table, threshold_km=25, n_folds=5, seed=42):

                sites = table[["site", "latitude", "longitude"]].drop_duplicates("site").reset_index(drop=True)

                xy    = np.column_stack([sites["longitude"] * 111 * np.cos(np.deg2rad(sites["latitude"])), sites["latitude"] * 111])

                sites["group"] = AgglomerativeClustering(n_clusters=None, distance_threshold=threshold_km, linkage="single").fit_predict(xy)

                sizes = sites["group"].value_counts()
                order = np.random.default_rng(seed).permutation(sizes.index)
                fold_size = np.zeros(n_folds, int)
                fold_of_group = {}
                for g in order:
                    f = fold_size.argmin()
                    fold_of_group[g] = f
                    fold_size[f] += sizes[g]
                sites["k"] = sites["group"].map(fold_of_group)

                return sites[["site", "k"]]
            
            site_folds = add_fold_ids(pd.read_parquet(rf"../master_datasets/training_tables/{self.config['TARG_VAR']}.parquet", \
                                         columns=self.training_columns))

            self.training_table = self.training_table.merge(site_folds, how='left', on='site')
            
            self.table_train = self.training_table[self.training_table['k'] != fold].copy()
            self.table_test  = self.training_table[self.training_table['k'] == fold].copy()

        elif fold_type == "NONE":
            self.table_train = self.training_table
            self.table_test = None

    def fit_and_transform(self):
        """B) Compute means/stds from the TRAIN rows only, then standardize every table with them.

        Stores the statistics (needed later to turn predictions back into ppb):
            self.table_train_stats   : {column: {"mean", "std"}} for the site table
            self.proxy_table_stats   : {proxy name: {column: {"mean", "std"}}}
        """

        def get_stats(table):
            """Mean and std of every column of this table that is in STANDARDIZE."""
            var_stats = {f"{col}": {"mean": None, "std": None} for col in table.columns if col in STANDARDIZE}
            for col in var_stats.keys():
                var_stats[col]['mean'] = table[col].astype("float64").mean()
                var_stats[col]['std']  = table[col].astype("float64").std()

            return var_stats

        def transform(table, var_stats):
            """(x - mean) / std for every column that has stats."""
            for col in var_stats.keys():
                table[col] = (table[col] - var_stats[col]['mean']) / var_stats[col]['std']
            return table

        # site rows: stats from train, applied to train and test
        self.table_train_stats = get_stats(self.table_train)
        self.table_train = transform(self.table_train, self.table_train_stats)
        if self.table_test is not None:
            self.table_test  = transform(self.table_test, self.table_train_stats)

        # proxy rows: each proxy standardized with its own stats
        if self.proxy_tables is not None:
            self.proxy_table_stats = {f"{var}": None for var in self.config['PCL_VARS']}
            for key in self.proxy_tables.keys():
                self.proxy_table_stats[key] = get_stats(self.proxy_tables[key])
                self.proxy_tables[key] = transform(self.proxy_tables[key], self.proxy_table_stats[key])

    def dataloaders(self):
        """C) Wrap train/val site rows and proxy rows in PyTorch DataLoaders."""
        batch_size = self.config["TRAIN"]["BATCH_SIZE"]
        proxy_batch_size = self.config["TRAIN"]["PROXY_BATCH_SIZE"]

        def train_test_loaders(table):
            x_obs = torch.tensor(table[self.obs_cols].values, dtype=torch.float32)
            x_loc = torch.tensor(table[self.loc_cols].values, dtype=torch.float32)
            y     = torch.tensor(table[self.config["TARG_VAR"]].values, dtype=torch.float32)
            idx   = torch.arange(len(table))
            loader = DataLoader(TensorDataset(x_obs, x_loc, y, idx), batch_size=batch_size, shuffle=True)

            return loader

        data_loaders = {"train": None, "test": None, "proxy": None}
        
        train_loader = train_test_loaders(self.table_train)
        data_loaders['train'] = train_loader

        if self.table_test is not None:
            test_loader  = train_test_loaders(self.table_test)
        else:
            test_loader = None
        data_loaders['test'] = test_loader

        if self.proxy_tables is not None:
            data_loaders["proxy"] = {}
            for name, table in self.proxy_tables.items():
                x = torch.tensor(table[self.loc_cols].values, dtype=torch.float32)
                y = torch.tensor(table["vertical_column_troposphere"].values, dtype=torch.float32)
                data_loaders["proxy"][name] = DataLoader(TensorDataset(x, y), batch_size=proxy_batch_size, shuffle=True)

        return data_loaders

if __name__ == "__main__":
    import copy
    import time
    from configs.test_config import CONFIG

    config = copy.deepcopy(CONFIG)                     # never modify the shared test config
    config["TRAIN"]["BATCH_SIZE"] = config["TRAIN"]["BATCH_SIZE"] or 1024   # test config leaves it as None

    def step(label, fn):
        t = time.time(); out = fn(); print(f"{label:<16}{time.time() - t:6.1f} s"); return out

    data = step("load",           lambda: PCLDataModule(config))
    step("attach_statics",        data.attach_statics)
    step("select_fold",           data.select_fold)
    step("fit_and_transform",     data.fit_and_transform)
    loaders = step("dataloaders", data.dataloaders)

    n_test = 0 if data.table_test is None else len(data.table_test)
    print(f"\nsite rows: train {len(data.table_train):,} / test {n_test:,}")

    # one site batch: shapes and NaNs
    x_obs, x_loc, y, idx = next(iter(loaders["train"]))
    print(f"site batch: x_obs {tuple(x_obs.shape)}, x_loc {tuple(x_loc.shape)}, y {tuple(y.shape)}")
    print(f"  expected: x_obs (B, {len(data.obs_cols)}), x_loc (B, {len(data.loc_cols)})")
    n_nan = int(torch.isnan(x_obs).sum() + torch.isnan(x_loc).sum() + torch.isnan(y).sum())
    print(f"  NaNs in batch: {n_nan}")

    # one batch per proxy
    if loaders["proxy"] is not None:
        for name, loader in loaders["proxy"].items():
            px, py = next(iter(loader))
            print(f"proxy {name}: x {tuple(px.shape)}, y {tuple(py.shape)}, {len(data.proxy_tables[name]):,} rows")

    # standardized columns should be mean ~0, std ~1 on the training rows
    std_cols = list(data.table_train_stats)
    stds = data.table_train[std_cols].std()
    print(f"\nstandardized mean range: {data.table_train[std_cols].mean().agg(['min', 'max']).round(3).tolist()}")
    print(f"standardized std range:  {stds.agg(['min', 'max']).round(3).tolist()}")
    if (abs(stds - 1) > 0.01).any():
        print("  columns with std far from 1:", stds[abs(stds - 1) > 0.01].round(3).to_dict())

    # embeddings should be unit vectors
    if config["EMBEDDINGS"]:
        emb = data.table_train[[f"emb_{b}" for b in range(64)]].values
        print(f"embedding |v| mean: {np.linalg.norm(emb, axis=1).mean():.4f} (should be ~1)")