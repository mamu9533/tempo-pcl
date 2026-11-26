from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")
pd.set_option("future.infer_string", False)

# =============================================================================
# 1. Load the EPA hourly NO2 files and combine them into one table
# =============================================================================
print("Loading EPA NO2 files...")
files = Path(r"..\epa_data\no2")
files = files.iterdir()
dfs = []
for f in files:
    print(f"  reading {f.name}")
    dfs.append(pd.read_csv(f))

print("Combining files into one table...")
master_df = pd.concat(dfs, axis=0)
master_df = master_df.reset_index(drop=True)

# Keep CONUS only (drops AK, HI, PR, VI, and any Canada/Mexico sites)
print("Keeping CONUS monitors only...")
conus = (master_df["Latitude"].between(24, 50) & master_df["Longitude"].between(-125, -66.5))
print(f"  dropping {(~conus).sum()} rows")
master_df = master_df[conus].reset_index(drop=True)

# =============================================================================
# 2. Add derived columns: UTC datetime, instrument type, instrument ID
# =============================================================================
# Combine the UTC date and time columns into one datetime column (whole column at once)
print("Building datetime column...")
master_df["datetime"] = pd.to_datetime(master_df["Date GMT"] + " " + master_df["Time GMT"], format="%Y-%m-%d %H:%M")

# Method codes for near-true NO2: CAPS (212, 256) and photolytic (200); everything else is converter chemiluminescence
print("Assigning instrument types...")
direct = [200, 212, 256]
master_df['instrument_type'] = np.where(master_df['Method Code'].isin(direct), "direct", "converter")

# Unique ID per instrument: state_county_site_POC
print("Creating instrument IDs...")
master_df['site_ID'] = (master_df['State Code'].astype(str) + "_" + master_df['County Code'].astype(str) + "_"
                        + master_df['Site Num'].astype(str) + "_" + master_df['POC'].astype(str))

# =============================================================================
# 3. Split the table into one dictionary entry per instrument
# =============================================================================
print("Splitting data by instrument...")
sites = master_df['site_ID'].unique()
site_data = {f"site_{s}": {"direct?": None, "Latitude": None, "Longitude": None, "datetime": None, "NO2": None} for s in sites}
for s in sites:
    sub = master_df[master_df['site_ID'] == s]   # all rows for this instrument

    # Instrument type, taken from the instrument's first row
    if sub['instrument_type'].iloc[0] == 'direct':
        site_data[f"site_{s}"]['direct?'] = True
    else:
        site_data[f"site_{s}"]['direct?'] = False

    # Location (constant for an instrument)
    lat = sub['Latitude'].unique()[0]
    lon = sub['Longitude'].unique()[0]

    # Hourly time series
    dat = np.array(sub['datetime'])
    no2 = np.array(sub['Sample Measurement'])

    site_data[f"site_{s}"]["Latitude"]  = lat
    site_data[f"site_{s}"]["Longitude"] = lon

    site_data[f"site_{s}"]["datetime"]  = dat
    site_data[f"site_{s}"]["NO2"]       = no2

# =============================================================================
# 4. Convert the dictionary to an xarray Dataset with NO2 on (site, time)
# =============================================================================
keys = list(site_data.keys())

# 4a. Common hourly time axis (UTC) covering every instrument
print("Building common time axis...")
t0 = min(site_data[k]["datetime"].min() for k in keys)
t1 = max(site_data[k]["datetime"].max() for k in keys)
times = pd.date_range(t0, t1, freq="h")
tindex = pd.Index(times)

# 4b. Empty site x time grid; hours with no measurement stay NaN
print("Creating empty site x time grid...")
no2 = np.full((len(keys), len(times)), np.nan, dtype=np.float32)

# 4c. Fill each instrument's row: find where its timestamps fall on the common axis
print("Filling grid...")
for i, k in enumerate(keys):
    pos = tindex.get_indexer(site_data[k]["datetime"])
    ok = pos >= 0
    no2[i, pos[ok]] = site_data[k]["NO2"][ok]

# 4d. Per-instrument information, in the same order as keys
print("Collecting per-instrument information...")
lat = np.array([site_data[k]["Latitude"] for k in keys], dtype=np.float32)
lon = np.array([site_data[k]["Longitude"] for k in keys], dtype=np.float32)
direct = np.array([site_data[k]["direct?"] for k in keys], dtype=np.int8)

# 4e. Assemble the Dataset
print("Assembling Dataset...")
ds = xr.Dataset(
    {
        "NO2": (("site", "time"), no2),
        "latitude": ("site", lat),
        "longitude": ("site", lon),
        "direct": ("site", direct), 
    },
        coords={"site": np.array(keys, dtype=str), "time": times},
)
ds["NO2"].attrs["units"] = "ppb"
ds["direct"].attrs["description"] = "1 = CAPS or photolytic (near-true NO2), 0 = converter chemiluminescence"

# =============================================================================
# 5. Check the layout and save (compressed)
# =============================================================================
print(ds["NO2"].dims)   # must be ('site', 'time')
print(ds)
print("Saving...")
ds.to_netcdf(r"..\master_datasets\no2_master.nc", encoding={"NO2": {"zlib": True, "complevel": 4}})
print("Done.")