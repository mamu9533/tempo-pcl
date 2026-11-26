"""
Download AlphaEarth satellite embeddings for CONUS at ~500 m, written block by block to a Zarr store.

Earth Engine averages the native 10 m embeddings onto a regular lat/lon grid and sends each
block back as a numpy array (no Google Drive, no GeoTIFFs). Each block is written to disk as
soon as it arrives, so memory stays small and the download can be stopped and restarted:
finished blocks are skipped.

Grid: 0.005 deg by default (~500 m), 24-50 N, 125-66.5 W -> 5200 x 11700 cells.
      Cell edges line up with the TEMPO 0.02 deg grid: each TEMPO pixel = 4 x 4 cells here.
Output: one Zarr store per year, alphaearth_conus_<year>_<res>.zarr, with
      emb_000 .. emb_063   embedding, renormalised to unit length (float16)
      emb_heterogeneity    1 - length of the averaged vector: 0 = uniform cell, higher = mixed surface
      NaN over ocean / lakes / no coverage (skip those cells when sampling)
      Size: roughly 5-8 GB per year at 0.005 deg.

Open it later with:
    ds = xr.open_zarr(r"D:\\AlphaEarth_CONUS\\alphaearth_conus_2024_0.005.zarr")

Usage:
    python alphaearth_conus.py --project YOUR_GCP_PROJECT --years 2024
    python alphaearth_conus.py --project YOUR_GCP_PROJECT --years 2024 --res 0.01     # ~1 km
"""
import argparse
import threading
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import dask.array as da
import numpy as np
import xarray as xr
import ee

warnings.filterwarnings("ignore")

COLLECTION = "GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL"
BANDS = [f"A{i:02d}" for i in range(64)]
NAMES = [f"emb_{i:03d}" for i in range(64)]

# CONUS box (cell edges), same as the TEMPO download
LAT0, LAT1 = 24.0, 50.0
LON0, LON1 = -125.0, -66.5
BLOCK = 128        # cells per side per request (smaller blocks = lighter requests, since each cell averages sub x sub points)
NODATA = -9999.0


# =============================================================================
# 1. Fetch one block from Earth Engine as a numpy array
# =============================================================================
def fetch_block(img, res, i0, j0, h, w, tries=5):
    """
    Return (64, h, w) float32 for the block starting at row i0 (counted from the south)
    and column j0 (counted from the west). Rows come back south -> north.
    """
    north_edge = LAT0 + (i0 + h) * res          # Earth Engine grids start at the top-left corner
    west_edge = LON0 + j0 * res
    request = {
        "expression": img,
        "fileFormat": "NUMPY_NDARRAY",
        "grid": {
            "dimensions": {"width": w, "height": h},
            "affineTransform": {"scaleX": res, "shearX": 0, "translateX": west_edge,
                                "shearY": 0, "scaleY": -res, "translateY": north_edge},
            "crsCode": "EPSG:4326",
        },
    }
    for attempt in range(tries):
        try:
            arr = ee.data.computePixels(request)              # rows go north -> south
            cube = np.stack([arr[b] for b in BANDS]).astype(np.float32)
            return cube[:, ::-1, :]                           # flip to south -> north
        except Exception as e:
            if attempt == tries - 1:
                raise
            print(f"    block ({i0},{j0}) failed ({str(e)[:80]}); retrying")
            time.sleep(10 * (attempt + 1))


# =============================================================================
# 2. Per-block cleanup: heterogeneity and renormalising
# =============================================================================
def clean_block(cube):
    """
    Each 10 m embedding is a unit-length vector. Averaged over a cell, vectors pointing different
    ways partly cancel, so the average is shorter than 1. 1 - length = how mixed the cell is.
    Then rescale back to length 1, like the original embeddings.
    """
    cube[cube == NODATA] = np.nan
    norm = np.linalg.norm(cube, axis=0)
    het = 1.0 - norm
    cube = cube / np.where(norm > 1e-8, norm, 1.0)
    return cube, het


# =============================================================================
# 3. Empty Zarr store with the full grid, filled in block by block afterwards
# =============================================================================
def create_store(store, lat, lon, year, res):
    shape = (len(lat), len(lon))
    empty = lambda: da.full(shape, np.nan, dtype="float16", chunks=(BLOCK, BLOCK))
    data = {n: (("latitude", "longitude"), empty()) for n in NAMES}
    data["emb_heterogeneity"] = (("latitude", "longitude"), empty())
    ds = xr.Dataset(data, coords={"latitude": lat, "longitude": lon})
    ds.attrs = {"source": COLLECTION, "year": year, "resolution_deg": res,
                "note": "emb_* renormalised to unit length; NaN = no embedding (water / no coverage); "
                        "emb_heterogeneity = 1 - length of the averaged 10 m vectors"}
    ds.to_zarr(store, compute=False, mode="w")     # writes coordinates and layout only


# =============================================================================
# 4. Download one year
# =============================================================================
def download_year(year, res, sub, out_dir, workers):
    ny = int(round((LAT1 - LAT0) / res))
    nx = int(round((LON1 - LON0) / res))
    lat = np.round(LAT0 + res / 2 + res * np.arange(ny), 6)   # cell centres, south -> north
    lon = np.round(LON0 + res / 2 + res * np.arange(nx), 6)   # cell centres, west -> east

    store = out_dir / f"alphaearth_conus_{year}_{res}.zarr"
    done_file = out_dir / f"alphaearth_conus_{year}_{res}_done_blocks.txt"
    if not store.exists():
        print(f"  creating {store.name} ({ny} x {nx} cells)")
        create_store(store, lat, lon, year, res)
        done_file.unlink(missing_ok=True)

    # Blocks finished in an earlier run
    done = set(done_file.read_text().split()) if done_file.exists() else set()
    blocks = [(i, j) for i in range(0, ny, BLOCK) for j in range(0, nx, BLOCK)
              if f"{i}_{j}" not in done]
    total = len(blocks) + len(done)
    print(f"  {total} blocks in total, {len(done)} already done, {len(blocks)} to fetch")
    if not blocks:
        return store

    col = ee.ImageCollection(COLLECTION).filterDate(f"{year}-01-01", f"{year + 1}-01-01")
    # Left alone, Earth Engine picks ONE pixel per output cell. To get a real average:
    #   1. put the embedding on a fine sub-grid (sub x sub points per output cell, aligned with it)
    #   2. average those points into each output cell
    fine = res / sub
    img = (col.mosaic().select(BANDS)
              .reproject(crs="EPSG:4326", crsTransform=[fine, 0, LON0, 0, -fine, LAT1])
              .reduceResolution(reducer=ee.Reducer.mean(), maxPixels=sub * sub + 16)
              .toFloat()
              .unmask(NODATA))

    lock = threading.Lock()
    counter = {"n": len(done), "checked": False}

    def work(ij):
        i, j = ij
        h, w = min(BLOCK, ny - i), min(BLOCK, nx - j)
        cube, het = clean_block(fetch_block(img, res, i, j, h, w))

        data = {n: (("latitude", "longitude"), cube[k].astype(np.float16)) for k, n in enumerate(NAMES)}
        data["emb_heterogeneity"] = (("latitude", "longitude"), het.astype(np.float16))
        region = {"latitude": slice(i, i + h), "longitude": slice(j, j + w)}

        with lock:
            xr.Dataset(data).to_zarr(store, region=region)
            with open(done_file, "a") as f:
                f.write(f"{i}_{j}\n")             # recorded only after the write succeeded
            counter["n"] += 1
            # Sanity check on the first block that has land
            if not counter["checked"] and np.isfinite(het).any():
                counter["checked"] = True
                print(f"  check: median heterogeneity in first land block = {np.nanmedian(het):.3f} "
                      f"(should be clearly above 0)")
            if counter["n"] % 25 == 0 or counter["n"] == total:
                print(f"  {counter['n']}/{total} blocks done")

    with ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(work, blocks))
    return store


# =============================================================================
# 5. Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True, help="Google Cloud project registered for Earth Engine")
    ap.add_argument("--years", type=int, nargs="*", default=[2024])
    ap.add_argument("--res", type=float, default=0.005,
                    help="grid spacing in degrees: 0.02 (~2 km), 0.01 (~1 km), 0.005 (~500 m)")
    ap.add_argument("--sub", type=int, default=8,
                    help="points per side averaged inside each cell (8 -> 64 points, ~60 m apart at 0.005 deg)")
    ap.add_argument("--out-dir", default=r"D:\AlphaEarth_CONUS")
    ap.add_argument("--workers", type=int, default=8, help="parallel requests to Earth Engine")
    args = ap.parse_args()

    # The high-volume endpoint is the one Google recommends for many small computePixels requests
    ee.Initialize(project=args.project, opt_url="https://earthengine-highvolume.googleapis.com")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for year in args.years:
        print(f"{year}: downloading at {args.res} deg, averaging {args.sub}x{args.sub} points per cell")
        store = download_year(year, args.res, args.sub, out_dir, args.workers)
        print(f"  finished {store}\n")
    print("Done.")


if __name__ == "__main__":
    main()