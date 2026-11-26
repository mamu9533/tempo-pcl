CONFIG = {
    "TARG_VAR": "NO2",
    "OBS_VARS": ['hrrr_blh', 'hrrr_ssrd', 'hrrr_sp', 'hrrr_gust', 'hrrr_mslp', 'hrrr_t2m', 'hrrr_d2m', 'hrrr_rh',
                 'hrrr_u10', 'hrrr_v10', 'hrrr_tcc', 'tempo_no2', 'no2_offset_min', 'tempo_hcho', 'hcho_offset_min',
                 'tempo_o3', 'o3_offset_min', 'tempo_ratio', 'ratio_offset_min'],
    "LOC_VARS": ['latitude', 'longitude', 'solar_sin', 'solar_cos', 'dow_sin', 'dow_cos', 'doy_sin', 'doy_cos'],
    "PCL_VARS": ['NO2'],
    "EMBEDDINGS": True,
    "TERRAIN": True,
    "FOLD_TYPE": "SPATIAL",
    "FOLD_SEL": 0,
    "MODEL": {"n_freq": 12, "n_hidden": 128, "n_hidden_proxy": 32, "n_layers_enc": 3,
              "n_latent": 64, "n_layers_pred": 2, "dropout": 0.1},
    }
