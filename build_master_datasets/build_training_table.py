"""
Build the site training tables: one row per (monitor, hour), holding
    EPA target  +  HRRR at the monitor  +  TEMPO over the monitor
Hours are labelled by their start, the way EPA does it.
"""
import warnings
import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")

# =============================================================================
# Settings
# =============================================================================
TEMPO_DIR  = r"D:\TEMPO_CONUS\MASTERS"
HRRR_FILE  = r"D:\HRRR_SITES\hrrr_master.nc"

# =============================================================================
# Helpers
# =============================================================================
def to_table(ds, fill_gaps=False):
    """(time, site) dataset -> flat table, rows with any missing value dropped.
    fill_gaps: fill gaps of up to 2 h first (targets only)."""
    if fill_gaps:
        ds = ds.interpolate_na(dim="time", max_gap="2h")
    return ds.to_dataframe().reset_index().dropna()

def hrrr_at_sites(hrrr, sites):
    """Nearest HRRR cell for each monitor -> (time, site)."""
    dlat = hrrr.cell_lat - sites.latitude
    dlon = (hrrr.cell_lon - sites.longitude) * np.cos(np.deg2rad(sites.latitude))
    nearest = (dlat**2 + dlon**2).argmin("cell")
    return hrrr.isel(cell=nearest).drop_vars(["cell", "cell_lat", "cell_lon"])

def tempo_at_sites(tempo, sites, name):
    """TEMPO pixel over each monitor -> table with site, time, tempo_<name>, <name>_offset_min."""
    picked = tempo.sel(latitude=sites.latitude, longitude=sites.longitude, method="nearest").load()
    df = picked.to_dataframe().reset_index().drop(columns=["latitude", "longitude", "time"])
    df = df.dropna(subset=["vertical_column_troposphere"])  # cloud, night, bad QC

    # pass_time: cftime -> datetime, and drop the garbage ones
    df["pass_time"] = pd.to_datetime(df["pass_time"].astype(str))
    df = df[(df["pass_time"] > "2023-01-01") & (df["pass_time"] < "2027-01-01")]

    # the monitor hour it falls in (14:37 -> 14:00) and minutes into that hour
    df["time"] = df["pass_time"].dt.floor("h")
    df[f"{name}_offset_min"] = (df["pass_time"] - df["time"]).dt.total_seconds() / 60

    # two scans in one hour (special scans): keep the first
    df = df.sort_values("pass_time").drop_duplicates(["site", "time"]).drop(columns="pass_time")
    return df.rename(columns={"vertical_column_troposphere": f"tempo_{name}"})

def add_time_features(df):
    """Solar hour, day of week, day of year as sin/cos (same formulas as the proxy tables).
    Uses the hour label 'time' (UTC) and the monitor's longitude."""
    solar_hour = df["time"].dt.hour + df["time"].dt.minute / 60 + df["longitude"] / 15
    df["solar_sin"] = np.sin(2 * np.pi * solar_hour / 24)
    df["solar_cos"] = np.cos(2 * np.pi * solar_hour / 24)

    solar_time = df["time"] + pd.to_timedelta(df["longitude"] / 15, unit="h")
    dow = solar_time.dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    doy = df["time"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df

def build_site_table(epa, hrrr, tempo_products, name):
    """EPA target + HRRR (inner join) + time features + every TEMPO product (left join)
    -> <name>_sites.parquet."""
    table = to_table(epa, fill_gaps=True)
    table = table.merge(to_table(hrrr_at_sites(hrrr, epa)), on=["site", "time"], how="inner")
    table = add_time_features(table)

    for product, tempo in tempo_products.items():
        print(f"  TEMPO {product}")
        n_rows = len(table)
        table = table.merge(tempo_at_sites(tempo, epa, product), on=["site", "time"], how="left")
        assert len(table) == n_rows, f"TEMPO {product} merge changed the row count"

    out = fr"..\master_datasets\training_tables\{name}.parquet"
    table.to_parquet(out, index=False)
    print(f"  wrote {out} ({len(table):,} rows)")
    return table

# =============================================================================
# Load EPA and HRRR
# =============================================================================
epa_no2 = xr.open_dataset(r"..\master_datasets\NO2_master.nc")
epa_o3  = xr.open_dataset(r"..\master_datasets\O3_master.nc")

hrrr = xr.open_dataset(HRRR_FILE)
print(hrrr.sizes, f"{hrrr.nbytes / 1e9:.1f} GB")
hrrr = hrrr.load()                     # read once; picking cells straight from disk is very slow

# fill values: ssrd -9999/-10000 only at night -> 0; unphysical dew points -> NaN
hrrr["hrrr_ssrd"] = hrrr["hrrr_ssrd"].where(hrrr["hrrr_ssrd"] > -9000, 0)
hrrr["hrrr_d2m"]  = hrrr["hrrr_d2m"].where(hrrr["hrrr_d2m"] >= 223)

# =============================================================================
# Load TEMPO (every product's column renamed to vertical_column_troposphere)
# =============================================================================
no2   = xr.open_zarr(fr"{TEMPO_DIR}\NO2_MASTER.zarr")
hcho  = xr.open_zarr(fr"{TEMPO_DIR}\HCHO_MASTER.zarr").rename({"vertical_column": "vertical_column_troposphere"})
o3    = xr.open_zarr(fr"{TEMPO_DIR}\O3_MASTER.zarr").rename({"troposphere_ozone_column": "vertical_column_troposphere"})
ratio = xr.open_zarr(fr"{TEMPO_DIR}\RATIO_MASTER.zarr")

TEMPO = {"no2": no2, "hcho": hcho, "o3": o3, "ratio": ratio}
# =============================================================================
# Build
# =============================================================================
if __name__ == "__main__":
    print("NO2 table")
    no2_df = build_site_table(epa_no2, hrrr, TEMPO, "no2")

    print("O3 table")
    o3_df = build_site_table(epa_o3, hrrr, TEMPO, "o3")