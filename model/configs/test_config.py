"""
Test configuration: small and fast, for checking that the pipeline runs end to end.
Used by the name-guard tests in data_loader.py and model.py, and for short trial training runs.
Copy this file to make a config for a real run (e.g. configs/no2_spatial_fold0.py).
"""

CONFIG = {
    # ---- data ----
    "TARG_VAR": "O3",
    "OBS_VARS": ['hrrr_blh', 'hrrr_ssrd', 'hrrr_sp', 'hrrr_gust', 'hrrr_mslp', 'hrrr_t2m', 'hrrr_d2m', 'hrrr_rh',
                 'hrrr_u10', 'hrrr_v10', 'hrrr_tcc', 'tempo_no2', 'no2_offset_min', 'tempo_hcho', 'hcho_offset_min',
                 'tempo_o3', 'o3_offset_min', 'tempo_ratio', 'ratio_offset_min'],
    "LOC_VARS": ['latitude', 'longitude', 'solar_sin', 'solar_cos', 'dow_sin', 'dow_cos', 'doy_sin', 'doy_cos'],
    "PCL_VARS": ["O3", "RATIO"],          # proxy tables to load (must match the LAMBDAS keys)
    "EMBEDDINGS": True,
    "TERRAIN": True,

    # ---- fold ----
    "FOLD_TYPE": "SPATIAL",               # "SPATIAL", "TEMPORAL", or "NONE"
    "FOLD_SEL": 0,                        # held-out fold, 0-4

    # ---- model ----
    "MODEL": {"n_freq": 12, "n_hidden": 128, "n_hidden_proxy": 32, "n_layers_enc": 3,
              "n_latent": 64, "n_layers_pred": 2, "dropout": 0.1},

    # ---- proxy loss weights ----
    "LAMBDAS": {"O3": 0, "RATIO":0},

    # ---- training (small, for a quick check) ----
    "TRAIN": {"LR": 1e-3, "BATCH_SIZE": 1024, "PROXY_BATCH_SIZE": 4096, "EPOCHS": 200, 
              "PATIENCE": 10, "LR_FACTOR": 0.5, "LR_PATIENCE": 3},
}

assert set(CONFIG["PCL_VARS"] or []) == set(CONFIG["LAMBDAS"]), "PCL_VARS and LAMBDAS keys must match"