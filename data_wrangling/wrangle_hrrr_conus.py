#!/usr/bin/env python3
"""Pull HRRR 3 km hourly analyses over CONUS onto the TEMPO L3 grid, one NetCDF per day.

Source is the MesoWest / University of Utah Zarr mirror (s3://hrrrzarr, anonymous).
HRRR is on a Lambert conformal grid; each hour is resampled once onto a regular lat/lon
grid by nearest neighbour.  By default that grid matches TEMPO L3 (0.02 deg, cell centres
at odd hundredths, e.g. 24.01, 24.03, ...), or pass --template with a TEMPO file to copy
its latitude/longitude exactly.

CONUS is far too large to hold in memory for the whole period, so the output is written
one file per UTC day (hrrr_conus_YYYYMMDD.nc), one variable at a time.  Days whose file
already exists are skipped, so the script can be stopped and restarted freely; each day is
written to a temporary name first and renamed only once it is complete.

Missing hours (the archive has occasional gaps) are written as NaN rather than skipped,
so every daily file has 24 hours and the time axis stays regular.

Pick the dates you want to map: each CONUS day is roughly 0.5-1 GB on disk.
Winds are rotated to true east/north, matching the at-site HRRR file.

    python fetch_hrrr_conus.py --start 2024-07-01 --end 2024-07-07        # one week
    python fetch_hrrr_conus.py --start 2024-07-01 --end 2024-07-01        # one test day
    python fetch_hrrr_conus.py --template D:/TEMPO_CONUS/L3_NO2_V04/<any file>.nc
"""
from __future__ import annotations

import argparse
import io
import os
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

warnings.filterwarnings("ignore")

BUCKET = "hrrrzarr"
# level -> (HRRR name, output name)
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

# Continental US, same box as download_tempo.py: 24.0-50.0 N, 125.0-66.5 W
DEFAULT_LAT, DEFAULT_LON = 37.0, -95.75
DEFAULT_LAT_BUFFER = 13.0
DEFAULT_LON_BUFFER = 29.25
DEFAULT_RES = 0.02                     # TEMPO L3 grid spacing
DEFAULT_OUTPUT_DIR = r"D:/HRRR_CONUS"


# ---------------------------------------------------------------------------
# Grids
# ---------------------------------------------------------------------------

def load_hrrr_grid(fs):
    """2-D latitude/longitude of the native HRRR grid."""
    with fs.open(f"{BUCKET}/grid/HRRR_latlon.h5") as f:
        raw = f.read()
    import h5py
    h = h5py.File(io.BytesIO(raw), "r")
    return np.asarray(h["latitude"][:], float), np.asarray(h["longitude"][:], float)


def index_box(lat, lon, bbox, pad=0.08):
    """Row/column slices of the native HRRR grid that cover the output box."""
    lo, la0, hi, la1 = bbox
    m = (lat >= la0 - pad) & (lat <= la1 + pad) & (lon >= lo - pad) & (lon <= hi + pad)
    if not m.any():
        raise SystemExit("bounding box does not intersect the HRRR grid")
    iy, ix = np.where(m)
    return slice(iy.min(), iy.max() + 1), slice(ix.min(), ix.max() + 1)


def target_grid(bbox, res, template=None):
    """1-D latitude/longitude of the output grid: copied from a TEMPO file, or TEMPO-style."""
    lo, la0, hi, la1 = bbox
    if template:
        t = xr.open_dataset(template)
        latn = next(c for c in t.variables if c.lower() in ("lat", "latitude"))
        lonn = next(c for c in t.variables if c.lower() in ("lon", "longitude"))
        tlat, tlon = np.asarray(t[latn].values, float), np.asarray(t[lonn].values, float)
        t.close()
        tlat = np.sort(tlat[(tlat >= la0) & (tlat <= la1)])
        tlon = np.sort(tlon[(tlon >= lo) & (tlon <= hi)])
        return tlat, tlon
    # TEMPO L3 cell centres sit at odd hundredths (x.01, x.03, ...) on a 0.02 deg grid
    half = res / 2
    tlat = np.round(np.arange(la0 + half, la1, res), 4)
    tlon = np.round(np.arange(lo + half, hi, res), 4)
    return tlat, tlon


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------

def read_one(fs, stamp, level, var, sy, sx):
    """One (hour, variable) slice, or None when the archive has no such analysis."""
    import zarr
    import s3fs as _s3
    d, hh = stamp.strftime("%Y%m%d"), stamp.strftime("%H")
    p = f"{BUCKET}/sfc/{d}/{d}_{hh}z_anl.zarr/{level}/{var}/{level}/{var}"
    try:
        z = zarr.open_array(_s3.S3Map(p, s3=fs), mode="r", zarr_format=2)
        return np.asarray(z[sy, sx], dtype=np.float32)
    except Exception:
        return None


# HRRR Lambert conformal projection constants (for rotating winds to east/north)
HRRR_LAT_TAN = 38.5     # standard parallel
HRRR_LON_CEN = -97.5    # central meridian


def rotate_winds(u, v, lon):
    """Grid-relative u/v -> true east/north u/v. `lon` is the longitude of each grid column."""
    angle = np.sin(np.deg2rad(HRRR_LAT_TAN)) * np.deg2rad(np.asarray(lon) - HRRR_LON_CEN)
    c, s = np.cos(angle), np.sin(angle)                 # broadcasts along the last (longitude) axis
    return (c * u + s * v).astype(np.float32), (-s * u + c * v).astype(np.float32)


# ---------------------------------------------------------------------------
# One day
# ---------------------------------------------------------------------------

def fetch_day(fs, day, use, sy, sx, idx, far, tlat, tlon, out_dir, workers):
    final = out_dir / f"hrrr_conus_{day:%Y%m%d}.nc"
    if final.exists():
        print(f"{day.date()}: already done, skipping")
        return
    tmp = out_dir / f"hrrr_conus_{day:%Y%m%d}.nc.part"
    if tmp.exists():
        tmp.unlink()                               # leftover from an interrupted run

    hours = pd.date_range(day, periods=24, freq="h")
    ny, nx = len(tlat), len(tlon)
    print(f"{day.date()}: fetching {len(use)} variables x 24 hours")

    def write(name, lev, v, arr, first):
        ds = xr.Dataset({name: (("time", "latitude", "longitude"), arr)},
                        coords={"time": hours.values, "latitude": tlat, "longitude": tlon})
        ds[name].attrs = {"hrrr_level": lev, "hrrr_name": v}
        if name in ("hrrr_u10", "hrrr_v10"):
            ds[name].attrs["note"] = "rotated to true east/north"
        if first:
            ds.attrs["source"] = "NOAA HRRR 3 km analyses via s3://hrrrzarr (MesoWest / U. Utah)"
            ds.attrs["regridding"] = "nearest neighbour onto TEMPO-style regular lat/lon grid"
        enc = {name: {"zlib": True, "complevel": 4, "dtype": "float32",
                      "chunksizes": (1, min(256, ny), min(256, nx))}}
        ds.to_netcdf(tmp, mode="w" if first else "a", engine="netcdf4", encoding=enc)
        print(f"  {name:10s} {np.isfinite(arr).mean():6.1%} of cells filled")

    first = True
    held_u = None                                   # u10 waits here until v10 arrives
    for lev, v, name in use:
        arr = np.full((24, ny, nx), np.nan, np.float32)   # one variable for one day

        def work(h):
            a = read_one(fs, hours[h], lev, v, sy, sx)
            if a is not None:
                g = a.ravel()[idx]
                g[far] = np.nan
                arr[h] = g.reshape(ny, nx)

        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(work, range(24)))

        # Winds: HRRR u/v are relative to its Lambert grid; rotate both to true east/north
        if name == "hrrr_u10":
            held_u = (lev, v, arr)
            continue
        if name == "hrrr_v10" and held_u is not None:
            u_lev, u_v, u = held_u
            u, arr = rotate_winds(u, arr, tlon)
            write("hrrr_u10", u_lev, u_v, u, first)
            first = False
            held_u = None

        write(name, lev, v, arr, first)
        first = False

    if held_u is not None:                          # u10 requested without v10: keep it unrotated
        write("hrrr_u10", *held_u, first)

    os.replace(tmp, final)                          # only now is the day marked complete
    print(f"  wrote {final.name} ({final.stat().st_size / 1e6:.0f} MB)\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help="first UTC day (YYYY-MM-DD)")
    ap.add_argument("--end", required=True, help="last UTC day (YYYY-MM-DD)")
    ap.add_argument("--coordinate", default=f"{DEFAULT_LAT},{DEFAULT_LON}",
                    help="box centre as 'lat,lon'")
    ap.add_argument("--lat-buffer", type=float, default=DEFAULT_LAT_BUFFER)
    ap.add_argument("--lon-buffer", type=float, default=DEFAULT_LON_BUFFER)
    ap.add_argument("--res", type=float, default=DEFAULT_RES, help="output grid spacing, degrees")
    ap.add_argument("--template", default=None,
                    help="a TEMPO L3 file whose latitude/longitude grid to copy exactly")
    ap.add_argument("--directory", default=DEFAULT_OUTPUT_DIR, help="output folder (created if missing)")
    ap.add_argument("--workers", type=int, default=24, help="parallel hour reads per variable")
    ap.add_argument("--vars", nargs="*", default=None, help="subset of output names; default all")
    args = ap.parse_args()

    import s3fs
    from scipy.spatial import cKDTree
    fs = s3fs.S3FileSystem(anon=True)

    clat, clon = (float(x) for x in args.coordinate.split(","))
    bbox = (clon - args.lon_buffer, clat - args.lat_buffer, clon + args.lon_buffer, clat + args.lat_buffer)
    print(f"Box: lat {bbox[1]:.2f} to {bbox[3]:.2f}, lon {bbox[0]:.2f} to {bbox[2]:.2f}")

    print("Loading HRRR grid...")
    lat2d, lon2d = load_hrrr_grid(fs)
    sy, sx = index_box(lat2d, lon2d, bbox)
    sub_lat, sub_lon = lat2d[sy, sx], lon2d[sy, sx]
    print(f"  HRRR subset {sub_lat.shape[0]} x {sub_lat.shape[1]} native cells")

    print("Building output grid and nearest-neighbour lookup (one time)...")
    tlat, tlon = target_grid(bbox, args.res, args.template)
    LA, LO = np.meshgrid(tlat, tlon, indexing="ij")
    # Project to approximate km so distances are meaningful across the domain
    kx_src = 111.32 * np.cos(np.deg2rad(sub_lat.ravel()))
    kx_tgt = 111.32 * np.cos(np.deg2rad(LA.ravel()))
    tree = cKDTree(np.column_stack([sub_lat.ravel() * 110.57, sub_lon.ravel() * kx_src]))
    dist, idx = tree.query(np.column_stack([LA.ravel() * 110.57, LO.ravel() * kx_tgt]), k=1)
    far = dist > 4.0                                  # outside the HRRR domain -> NaN
    print(f"  output grid {len(tlat)} x {len(tlon)} at {args.res} deg; "
          f"{far.mean():.1%} of cells outside HRRR")

    use = [t for t in VARS if (args.vars is None or t[2] in args.vars)]
    if not use:
        raise SystemExit(f"no variables matched {args.vars}")

    out_dir = Path(args.directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    days = pd.date_range(args.start, args.end, freq="D")
    print(f"{len(days)} days, {len(use)} variables, output to {out_dir}\n")

    for day in days:
        fetch_day(fs, day, use, sy, sx, idx, far, tlat, tlon, out_dir, args.workers)

    print("Done.")


if __name__ == "__main__":
    main()