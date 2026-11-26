from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")
pd.set_option("future.infer_string", False)

# =============================================================================
# 1. Load the EPA hourly O3 files and combine them into one table
# =============================================================================
print("Loading EPA O3 files...")
files = Path(r"..\epa_data\o3")
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
# 2. Add derived columns: UTC datetime, O3 in ppb, instrument ID
# =============================================================================
# Combine the UTC date and time columns into one datetime column (whole column at once)
print("Building datetime column...")
master_df["datetime"] = pd.to_datetime(master_df["Date GMT"] + " " + master_df["Time GMT"], format="%Y-%m-%d %H:%M")

# EPA reports O3 in ppm; convert to ppb
print("Converting O3 from ppm to ppb...")
master_df["O3"] = master_df["Sample Measurement"] * 1000

# Unique ID per instrument: state_county_site_POC
print("Creating instrument IDs...")
master_df['site_ID'] = (master_df['State Code'].astype(str) + "_" + master_df['County Code'].astype(str) + "_"
                        + master_df['Site Num'].astype(str) + "_" + master_df['POC'].astype(str))

# UV photometry and chemiluminescence agree to ~0.5 ppb, so no instrument tag is needed.
# Where a site runs both, drop the chemiluminescence instrument so the site isn't counted twice.
print("Dropping chemiluminescence instruments at sites that also have a UV instrument...")
chemi_name = "Instrumental - Chemiluminescence API Model 265E and T265"
master_df["location"] = (master_df['State Code'].astype(str) + "_" + master_df['County Code'].astype(str) + "_"
                         + master_df['Site Num'].astype(str))
is_chemi = master_df["Method Name"] == chemi_name
uv_locations = set(master_df.loc[~is_chemi, "location"])
drop = is_chemi & master_df["location"].isin(uv_locations)
print(f"  dropping {master_df.loc[drop, 'site_ID'].nunique()} instruments ({drop.sum()} rows)")
master_df = master_df[~drop]

# =============================================================================
# 3. Split the table into one dictionary entry per instrument
# =============================================================================
print("Splitting data by instrument...")
sites = master_df['site_ID'].unique()
site_data = {f"site_{s}": {"Latitude": None, "Longitude": None, "datetime": None, "O3": None} for s in sites}
for s in sites:
    sub = master_df[master_df['site_ID'] == s]   # all rows for this instrument

    # Location (constant for an instrument)
    lat = sub['Latitude'].unique()[0]
    lon = sub['Longitude'].unique()[0]

    # Hourly time series
    dat = np.array(sub['datetime'])
    o3 = np.array(sub['O3'])

    site_data[f"site_{s}"]["Latitude"]  = lat
    site_data[f"site_{s}"]["Longitude"] = lon

    site_data[f"site_{s}"]["datetime"]  = dat
    site_data[f"site_{s}"]["O3"]        = o3

# =============================================================================
# 4. Convert the dictionary to an xarray Dataset with O3 on (site, time)
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
o3 = np.full((len(keys), len(times)), np.nan, dtype=np.float32)

# 4c. Fill each instrument's row: find where its timestamps fall on the common axis
print("Filling grid...")
for i, k in enumerate(keys):
    pos = tindex.get_indexer(site_data[k]["datetime"])
    ok = pos >= 0
    o3[i, pos[ok]] = site_data[k]["O3"][ok]

# 4d. Per-instrument information, in the same order as keys
print("Collecting per-instrument information...")
lat = np.array([site_data[k]["Latitude"] for k in keys], dtype=np.float32)
lon = np.array([site_data[k]["Longitude"] for k in keys], dtype=np.float32)

# 4e. Assemble the Dataset
print("Assembling Dataset...")
ds = xr.Dataset(
    {
        "O3": (("site", "time"), o3),
        "latitude": ("site", lat),
        "longitude": ("site", lon),
    },
        coords={"site": np.array(keys, dtype=str), "time": times},
)
ds["O3"].attrs["units"] = "ppb"

# =============================================================================
# 5. Check the layout and save (compressed)
# =============================================================================
print(ds["O3"].dims)   # must be ('site', 'time')
print(ds)
print("Saving...")
ds.to_netcdf(r"..\master_datasets\o3_master.nc", encoding={"O3": {"zlib": True, "complevel": 4}})
print("Done.")