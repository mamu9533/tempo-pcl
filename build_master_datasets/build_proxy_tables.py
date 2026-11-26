import warnings

import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

# =============================================================================
# TEMPO master stores (one QC'd column per scan, plus pass_time per longitude)
# =============================================================================
TEMPO_DIR = r"D:\TEMPO_CONUS\MASTERS"
no2   = xr.open_zarr(fr"{TEMPO_DIR}\NO2_MASTER.zarr")
hcho  = xr.open_zarr(fr"{TEMPO_DIR}\HCHO_MASTER.zarr").rename({'vertical_column': 'vertical_column_troposphere'})
o3    = xr.open_zarr(fr"{TEMPO_DIR}\O3_MASTER.zarr").rename({'troposphere_ozone_column': 'vertical_column_troposphere'})
ratio = xr.open_zarr(fr"{TEMPO_DIR}\RATIO_MASTER.zarr")        # log(HCHO/NO2), already filtered

# =============================================================================
# Sample N random valid pixels from one scan
# =============================================================================
def tempo_random_samples(tempo_slice, seed, var=None, no_samples=500):
    # pass_time is per longitude (2,925 values): convert it here, not per pixel
    pt = tempo_slice["pass_time"].values
    if pt.dtype == object:
        pt = pd.to_datetime(pt.astype(str), format="ISO8601", errors="coerce")
    tempo_slice = tempo_slice.assign(pass_time=("longitude", pt))

    df = tempo_slice.to_dataframe().reset_index()              # one row per pixel of this scan
    df = df.dropna(subset=['vertical_column_troposphere'])     # keep pixels that passed QC
    if var == 'HCHO':
        df = df[df['vertical_column_troposphere'] > 0]         # drop negative HCHO retrievals
    df = df[(df["pass_time"] > "2023-01-01") & (df["pass_time"] < "2027-01-01")]   # bad pass times

    # seed = scan index: different pixels per scan, same pixels on every rerun
    df = df.sample(min(no_samples, len(df)), random_state=seed).reset_index(drop=True)

    solar_hour = df["pass_time"].dt.hour + df["pass_time"].dt.minute / 60 + df["longitude"] / 15
    df["solar_sin"] = np.sin(2 * np.pi * solar_hour / 24)
    df["solar_cos"] = np.cos(2 * np.pi * solar_hour / 24)

    solar_time = df["pass_time"] + pd.to_timedelta(df["longitude"] / 15, unit="h")
    dow = solar_time.dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    doy = df["pass_time"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


# =============================================================================
# Sample every scan in parallel, stack, save
# =============================================================================
def build_proxy_table(tempo_ds, name, n_jobs=12):
    n = tempo_ds.sizes["time"]

    def one_scan(i):
        warnings.filterwarnings("ignore")                      # workers are separate processes
        return tempo_random_samples(tempo_ds.isel(time=i), var=name, seed=i)

    print(f"Sampling for proxy table: {name}")
    results = Parallel(n_jobs=n_jobs, return_as="generator")(delayed(one_scan)(i) for i in range(n))
    table = pd.concat(tqdm(results, total=n), ignore_index=True)

    table.to_parquet(fr"../master_datasets/proxy_tables/{name}.parquet", index=False)
    print(f"  {len(table):,} rows -> {name}.parquet")
    return table


if __name__ == "__main__":
    #build_proxy_table(no2,  "NO2")
    #build_proxy_table(hcho, "HCHO")
    #build_proxy_table(o3,   "O3")
    build_proxy_table(ratio, "RATIO")