"""
Pull HRRR 3 km hourly analyses at every monitor location, laid out exactly like the master datasets.

For each master file (no2_master.nc, o3_master.nc) the output has the same `site` and `time`
coordinates, with every HRRR variable on (site, time), so it lines up one-to-one with NO2 / O3.

How it works:
  1. Read the monitor lat/lon from the master files and find the nearest HRRR grid cell for each.
     Instruments at the same spot share a cell, so each HRRR cell is read only once.
  2. Download month by month (one small file per month in --directory/months). Months already
     downloaded are skipped, so the script can be stopped and restarted.
  3. Combine the months and write one file per master: hrrr_<master name>.nc.

Source: MesoWest / University of Utah Zarr mirror (s3://hrrrzarr, anonymous, free). Only the
150 x 150 chunks that contain a monitor are read.

Winds: HRRR stores u/v relative to its own Lambert grid, not east/north. They are rotated to
true east/north here, so wind direction means the same thing everywhere in CONUS.

Monitors outside HRRR (Alaska, Hawaii, Puerto Rico) are kept but get NaN.

    python hrrr_sites.py                                   # time range taken from the master files
    python hrrr_sites.py --start 2023-08-01 --end 2025-12-31
    python hrrr_sites.py --combine-only                    # just rebuild the final files from the months
"""
import argparse
import io
import os
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")

BUCKET = "hrrrzarr"
# (HRRR level, HRRR name, output name)
VARS = [
    ("surface",           "HPBL",  "hrrr_blh"),   # boundary layer height
    ("surface",           "DSWRF", "hrrr_ssrd"),
    ("surface",           "PRES",  "hrrr_sp"),    # station-level pressure, NOT sea-level reduced
    ("surface",           "GUST",  "hrrr_gust"),
    ("mean_sea_level",    "MSLMA", "hrrr_mslp"),
    ("2m_above_ground",   "TMP",   "hrrr_t2m"),
    ("2m_above_ground",   "DPT",   "hrrr_d2m"),
    ("2m_above_ground",   "RH",    "hrrr_rh"),
    ("10m_above_ground",  "UGRD",  "hrrr_u10"),
    ("10m_above_ground",  "VGRD",  "hrrr_v10"),
    ("entire_atmosphere", "TCDC",  "hrrr_tcc"),
]
NAMES = [name for _, _, name in VARS]

# HRRR Lambert conformal projection constants (for rotating winds to east/north)
HRRR_LAT_TAN = 38.5     # standard parallel
HRRR_LON_CEN = -97.5    # central meridian

DEFAULT_MASTERS = [r"../master_datasets/no2_master.nc",
                   r"../master_datasets/o3_master.nc"]
DEFAULT_OUTPUT_DIR = r"D:/HRRR_SITES"


# =============================================================================
# 1. Monitor locations -> nearest HRRR cell
# =============================================================================
def load_hrrr_grid(fs):
    """2-D latitude/longitude of the native HRRR grid."""
    import h5py
    with fs.open(f"{BUCKET}/grid/HRRR_latlon.h5") as f:
        h = h5py.File(io.BytesIO(f.read()), "r")
        return np.asarray(h["latitude"][:], float), np.asarray(h["longitude"][:], float)


def nearest_cells(grid_lat, grid_lon, lat, lon, max_km=4.0):
    """
    Flat index of the nearest HRRR cell for each point, or -1 if the point is outside HRRR.
    Distances are measured in approximate km so they mean the same thing at every latitude.
    """
    from scipy.spatial import cKDTree
    def km(la, lo):
        return np.column_stack([la * 110.57, lo * 111.32 * np.cos(np.deg2rad(la))])
    tree = cKDTree(km(grid_lat.ravel(), grid_lon.ravel()))
    dist, idx = tree.query(km(lat, lon), k=1)
    idx[dist > max_km] = -1
    return idx


def read_masters(paths):
    """Site lists, lat/lon and time ranges of each master file."""
    masters = []
    for p in paths:
        with xr.open_dataset(p) as ds:
            masters.append({
                "path": Path(p),
                "site": ds["site"].values,
                "lat": ds["latitude"].values.astype(float),
                "lon": ds["longitude"].values.astype(float),
                "time": pd.DatetimeIndex(ds["time"].values),
            })
        print(f"  {Path(p).name}: {len(masters[-1]['site'])} sites, "
              f"{masters[-1]['time'][0]} to {masters[-1]['time'][-1]}")
    return masters


# =============================================================================
# 2. Download: one (hour, variable) at all needed cells
# =============================================================================
def read_points(fs, stamp, level, var, iy, ix):
    """Values at the given HRRR rows/columns for one hour, or None if the archive has a gap."""
    import zarr
    import s3fs
    d, hh = stamp.strftime("%Y%m%d"), stamp.strftime("%H")
    p = f"{BUCKET}/sfc/{d}/{d}_{hh}z_anl.zarr/{level}/{var}/{level}/{var}"
    try:
        z = zarr.open_array(s3fs.S3Map(p, s3=fs), mode="r", zarr_format=2)
        return np.asarray(z.vindex[iy, ix], dtype=np.float32)   # reads only chunks with a monitor
    except Exception:
        return None


def rotate_winds(u, v, cell_lon):
    """Grid-relative u/v -> true east/north u/v (standard Lambert conformal rotation)."""
    angle = np.sin(np.deg2rad(HRRR_LAT_TAN)) * np.deg2rad(cell_lon - HRRR_LON_CEN)
    c, s = np.cos(angle), np.sin(angle)
    return c * u + s * v, -s * u + c * v


def fetch_month(fs, hours, iy, ix, cell_lon, workers):
    out = {name: np.full((len(hours), len(iy)), np.nan, np.float32) for name in NAMES}
    jobs = [(h, lev, v, name) for h in range(len(hours)) for lev, v, name in VARS]

    t0 = time.time()
    counter = {"done": 0, "missing": 0}
    lock = threading.Lock()

    def work(job):
        h, lev, v, name = job
        a = read_points(fs, hours[h], lev, v, iy, ix)
        if a is not None:
            out[name][h] = a
        with lock:
            counter["done"] += 1
            counter["missing"] += a is None
            n = counter["done"]
            if n % 200 == 0 or n == len(jobs):
                rate = n / (time.time() - t0)
                left = (len(jobs) - n) / rate / 60
                print(f"  {n}/{len(jobs)} reads, {rate:.1f} reads/s, ~{left:.0f} min left in this month"
                      f" ({counter['missing']} missing)", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, jobs))

    out["hrrr_u10"], out["hrrr_v10"] = rotate_winds(out["hrrr_u10"], out["hrrr_v10"], cell_lon)
    return out


# =============================================================================
# 3. Combine the months into one (site, time) file per master
# =============================================================================
def combine(masters, site_cells, cells, month_dir, out_dir):
    files = sorted(month_dir.glob("hrrr_sites_*.nc"))
    if not files:
        raise SystemExit(f"no monthly files in {month_dir}")
    print(f"Combining {len(files)} monthly files...")
    allm = xr.concat([xr.open_dataset(f).load() for f in files], dim="time")
    allm = allm.sel(cell=cells)                          # same cell order as the lookup

    for m, sc in zip(masters, site_cells):
        print(f"  building {m['path'].stem}...")
        on_grid = sc >= 0
        col = np.searchsorted(cells, sc[on_grid])        # column in allm for each on-grid site
        data = {}
        for name in NAMES:
            arr = allm[name].reindex(time=m["time"]).values          # (time, cell), master's hours
            site_arr = np.full((len(m["site"]), len(m["time"])), np.nan, np.float32)
            site_arr[on_grid] = arr[:, col].T
            data[name] = (("site", "time"), site_arr)
        ds = xr.Dataset(data, coords={"site": m["site"], "time": m["time"]})
        ds["latitude"] = ("site", m["lat"].astype(np.float32))
        ds["longitude"] = ("site", m["lon"].astype(np.float32))
        for lev, v, name in VARS:
            ds[name].attrs = {"hrrr_level": lev, "hrrr_name": v}
        ds["hrrr_u10"].attrs["note"] = "rotated to true east-west"
        ds["hrrr_v10"].attrs["note"] = "rotated to true north-south"
        ds.attrs["source"] = "NOAA HRRR 3 km analyses via s3://hrrrzarr, nearest grid cell to each monitor"

        target = out_dir / f"hrrr_{m['path'].stem}.nc"
        ds.to_netcdf(target, encoding={n: {"zlib": True, "complevel": 4} for n in NAMES})
        print(f"  wrote {target} ({target.stat().st_size / 1e6:.0f} MB), "
              f"{(~on_grid).sum()} sites outside HRRR left as NaN")


# =============================================================================
# 4. Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--masters", nargs="*", default=[str(p) for p in DEFAULT_MASTERS],
                    help="master files whose sites to extract at")
    ap.add_argument("--start", default=None, help="first UTC day; default = earliest hour in the masters")
    ap.add_argument("--end", default=None, help="last UTC day; default = latest hour in the masters")
    ap.add_argument("--directory", default=DEFAULT_OUTPUT_DIR, help="output folder (created if missing)")
    ap.add_argument("--workers", type=int, default=32, help="parallel (hour, variable) reads")
    ap.add_argument("--combine-only", action="store_true", help="skip downloading, just combine months")
    args = ap.parse_args()

    import s3fs
    fs = s3fs.S3FileSystem(anon=True)

    out_dir = Path(args.directory)
    month_dir = out_dir / "months"
    month_dir.mkdir(parents=True, exist_ok=True)

    print("Reading master files...")
    masters = read_masters(args.masters)

    print("Loading HRRR grid and matching monitors to cells...")
    grid_lat, grid_lon = load_hrrr_grid(fs)
    site_cells = [nearest_cells(grid_lat, grid_lon, m["lat"], m["lon"]) for m in masters]
    cells = np.unique(np.concatenate([sc[sc >= 0] for sc in site_cells]))
    iy, ix = np.unravel_index(cells, grid_lat.shape)
    cell_lon = grid_lon.ravel()[cells]
    print(f"  {sum(len(m['site']) for m in masters)} instruments -> {len(cells)} distinct HRRR cells")

    if not args.combine_only:
        start = pd.Timestamp(args.start) if args.start else min(m["time"][0] for m in masters)
        end = (pd.Timestamp(args.end) + pd.Timedelta(hours=23)) if args.end else max(m["time"][-1] for m in masters)
        hours = pd.date_range(start.floor("h"), end, freq="h")
        months = hours.to_period("M").unique()
        print(f"Downloading {len(hours)} hours ({len(months)} months), {len(VARS)} variables\n")

        for mo in months:
            final = month_dir / f"hrrr_sites_{mo.strftime('%Y%m')}.nc"
            if final.exists():
                print(f"{mo}: already done, skipping")
                continue
            mh = hours[hours.to_period("M") == mo]
            print(f"{mo}: fetching {len(mh)} hours")
            out = fetch_month(fs, mh, iy, ix, cell_lon, args.workers)

            ds = xr.Dataset({n: (("time", "cell"), out[n]) for n in NAMES},
                            coords={"time": mh, "cell": cells})
            tmp = month_dir / f"hrrr_sites_{mo.strftime('%Y%m')}.nc.part"
            ds.to_netcdf(tmp, encoding={n: {"zlib": True, "complevel": 4} for n in NAMES})
            os.replace(tmp, final)                       # only now is the month marked done
            print(f"  {np.isfinite(out['hrrr_t2m']).mean():.1%} of hours filled\n")

    combine(masters, site_cells, cells, month_dir, out_dir)
    print("Done.")


if __name__ == "__main__":
    main()