import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

NO2_PATH  = r"D:\TEMPO_CONUS\MASTERS\NO2_MASTER.zarr"
HCHO_PATH = r"D:\TEMPO_CONUS\MASTERS\HCHO_MASTER.zarr"
OUT       = r"D:\TEMPO_CONUS\MASTERS\RATIO_MASTER.zarr"

no2  = xr.open_zarr(NO2_PATH)
hcho = xr.open_zarr(HCHO_PATH).rename({'vertical_column': 'vertical_column_troposphere'})

NO2_MIN = 1e15      # molecules/cm2; below this the NO2 retrieval is mostly noise

h = hcho['vertical_column_troposphere']
n = no2['vertical_column_troposphere']
ok = (h > 0) & (n > NO2_MIN)                                   # log needs positive values
ratio = xr.Dataset({'vertical_column_troposphere': np.log(h / n).where(ok)})   # xarray lines up the scan times

raw_pt = xr.open_zarr(NO2_PATH, decode_times=False)["pass_time"].assign_coords(time=no2.time)
ratio["pass_time"] = raw_pt.sel(time=ratio.time)

ratio = ratio.chunk({"time": 24, "latitude": -1, "longitude": -1})
for v in ratio.variables.values():                          
    v.encoding.pop("chunks", None)
    v.encoding.pop("preferred_chunks", None)

with ProgressBar():
    ratio.to_zarr(OUT, mode="w")

print(xr.open_zarr(OUT)["pass_time"].isel(time=0, longitude=[0, -1]).values)