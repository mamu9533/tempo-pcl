"""
Process TEMPO L3 V04 downloads into one Zarr store per product (NO2, HCHO, or O3).

Steps (each one can be rerun safely; finished work is skipped):
  1. check   - delete raw files that are corrupt (incomplete downloads)
  2. process - quality-filter each scan, keep the scan start/end, add the pass time per longitude,
               write one small file per scan to the PROCESSED folder
  3. merge   - stack all processed scans along time into one Zarr store

Set PRODUCT below, then run the whole file (or the steps you need at the bottom).
"""
import os
import re
from pathlib import Path

import h5py
import pandas as pd
import xarray as xr
from tqdm.auto import tqdm

# =============================================================================
# Settings
# =============================================================================
PRODUCT = "O3"                          # "NO2", "HCHO" or "O3"

ROOT = Path(r"D:\TEMPO_CONUS")
RAW_DIR = ROOT / f"L3_{PRODUCT}_V04"                 # Harmony downloads
PROCESSED_DIR = ROOT / f"L3_{PRODUCT}_V04_PROCESSED"  # one QC'd file per scan
STORE = ROOT / "MASTERS" / f"{PRODUCT}_MASTER.zarr"   # final merged store

# Column variable in each product's 'product' group
# (check the HCHO name on one raw file before the first HCHO run)
VARIABLES = {
    "NO2": "vertical_column_troposphere",
    "HCHO": "vertical_column",
    "O3": "troposphere_ozone_column",
}
VAR = VARIABLES[PRODUCT]

# Quality filter: keep pixels with flag 0 (normal) and cloud fraction below this.
# O3PROF has no quality flag, so for O3 only the cloud fraction is used.
MAX_CLOUD = 0.2

# Edge longitudes of the CONUS download box (pixel centres), for the pass-time estimate
WEST_EDGE, EAST_EDGE = -124.99, -66.51

MERGE_BATCH = 24                         # scans stacked per write to the Zarr store
SCAN_STAMP = re.compile(r"_(\d{8}T\d{6})Z_")   # e.g. _20230802T151249Z_ in the file name


# =============================================================================
# 1. Check: remove corrupt raw files
# =============================================================================
def check_raw_files():
    bad = []
    for f in tqdm(sorted(RAW_DIR.glob("*.nc4")), desc="check"):
        try:
            with h5py.File(f, "r"):
                pass
        except Exception as e:
            print(f"problem with file {f.name}: {e}")
            bad.append(f)
    for f in bad:
        f.unlink(missing_ok=True)
    print(f"removed {len(bad)} corrupt files (re-download them)")


# =============================================================================
# 2. Process: QC, keep scan times, add pass time
# =============================================================================
def rough_pass_time(longitude, scan_start, scan_stop):
    """When the scan reached each longitude: it sweeps east -> west at a roughly steady rate."""
    fraction = (EAST_EDGE - longitude) / (EAST_EDGE - WEST_EDGE)
    return scan_start + (scan_stop - scan_start) * fraction


def process_scan(f):
    with xr.open_dataset(f) as root, \
         xr.open_dataset(f, group="product") as prod, \
         xr.open_dataset(f, group="support_data") as sup:
        if PRODUCT == "O3":                           # O3PROF has no quality flag
            good = sup["eff_cloud_fraction"] < MAX_CLOUD
        else:
            good = (prod["main_data_quality_flag"] == 0) & (sup["eff_cloud_fraction"] < MAX_CLOUD)
        da = prod[VAR].where(good).assign_coords(root.coords)
        ds = da.astype("float32").to_dataset(name=VAR).load()
        start = pd.Timestamp(root.attrs["time_coverage_start"]).tz_localize(None)
        stop = pd.Timestamp(root.attrs["time_coverage_end"]).tz_localize(None)

    ds.attrs = {"time_coverage_start": str(start), "time_coverage_end": str(stop)}
    pt = rough_pass_time(ds.longitude.values, start, stop)
    ds["pass_time"] = ("longitude", pd.DatetimeIndex(pt).values)
    return ds


def process_all():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    for f in tqdm(sorted(RAW_DIR.glob("*.nc4")), desc="process"):
        out = PROCESSED_DIR / f.name
        tmp = out.with_suffix(".tmp")
        if out.exists():
            continue
        try:
            process_scan(f).to_netcdf(tmp, encoding={VAR: {"zlib": True, "complevel": 4}})
            with xr.open_dataset(tmp) as check:          # verify it reads back
                check[VAR].load()
                check["pass_time"].load()
            os.replace(tmp, out)
        except Exception as e:
            print(f"{f.name}: {e}")
            tmp.unlink(missing_ok=True)


def add_missing_pass_times():
    """For processed files made before pass_time was part of step 2: add it in place."""
    files = sorted(PROCESSED_DIR.glob("*.nc4"))
    fixed = 0
    for f in tqdm(files, desc="pass_time"):
        tmp = f.with_suffix(".tmp")
        try:
            with xr.open_dataset(f) as d:
                if "pass_time" in d:                     # already has it -> skip
                    continue
                ds = d.load()                            # read everything before the file closes
            start = pd.Timestamp(ds.attrs["time_coverage_start"]).tz_localize(None)
            stop = pd.Timestamp(ds.attrs["time_coverage_end"]).tz_localize(None)
            pt = rough_pass_time(ds.longitude.values, start, stop)
            ds["pass_time"] = ("longitude", pd.DatetimeIndex(pt).values)
            ds.to_netcdf(tmp, encoding={VAR: {"zlib": True, "complevel": 4}})
            with xr.open_dataset(tmp) as check:          # verify before replacing the original
                check["pass_time"].load()
            os.replace(tmp, f)
            fixed += 1
        except Exception as e:
            print(f"{f.name}: {e}")
            tmp.unlink(missing_ok=True)
    print(f"added pass_time to {fixed} files")


# =============================================================================
# 3. Merge: all processed scans -> one Zarr store
# =============================================================================
def merge_to_zarr():
    STORE.parent.mkdir(parents=True, exist_ok=True)
    
    files = sorted(PROCESSED_DIR.glob("*.nc4"), key=lambda f: SCAN_STAMP.search(f.name).group(1))

    for i in tqdm(range(0, len(files), MERGE_BATCH), desc="merge"):
        batch = []
        for f in files[i:i + MERGE_BATCH]:
            ds = xr.open_dataset(f).load()
            ds["pass_time"] = ds["pass_time"].expand_dims(time=ds["time"])   # one row of pass times per scan
            ds.attrs = {}                   
            batch.append(ds)
        batch = xr.concat(batch, dim="time")

        if i == 0:
            batch.to_zarr(STORE, mode="w")
        else:
            batch.to_zarr(STORE, append_dim="time")
    print(f"wrote {STORE}")


# =============================================================================
# Run
# =============================================================================
if __name__ == "__main__":
    check_raw_files()
    process_all()
    add_missing_pass_times()     # only does anything for files processed before pass_time was added in step 2
    merge_to_zarr()