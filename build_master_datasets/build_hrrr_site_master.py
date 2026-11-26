import io
from pathlib import Path

import numpy as np
import xarray as xr

MONTH_DIR = Path(r"D:\HRRR_SITES\months")
OUT_DIR = Path(r"D:\HRRR_SITES")


def load_hrrr_grid():
    import h5py
    import s3fs
    fs = s3fs.S3FileSystem(anon=True)
    with fs.open("hrrrzarr/grid/HRRR_latlon.h5") as f:
        h = h5py.File(io.BytesIO(f.read()), "r")
        return np.asarray(h["latitude"][:], float), np.asarray(h["longitude"][:], float)


# All months stacked along time: every HRRR variable on (time, cell)
hrrr = xr.open_mfdataset(sorted(MONTH_DIR.glob("hrrr_sites_*.nc")), combine="by_coords").load()

# Give each cell its latitude/longitude, so it can be matched to monitors later
grid_lat, grid_lon = load_hrrr_grid()
cells = hrrr["cell"].values
hrrr = hrrr.assign_coords(cell_lat=("cell", grid_lat.ravel()[cells].astype(np.float32)),
                          cell_lon=("cell", grid_lon.ravel()[cells].astype(np.float32)))

hrrr.to_netcdf(OUT_DIR / "hrrr_master.nc",
               encoding={n: {"zlib": True, "complevel": 4} for n in hrrr.data_vars})
print(hrrr)