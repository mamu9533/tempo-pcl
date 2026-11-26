from pathlib import Path
import pandas as pd
import xarray as xr
from harmony import Client, Collection, Request, BBox

out_dir = Path(r"D:\TEMPO_CONUS\TERRAIN_HEIGHT")
out_dir.mkdir(parents=True, exist_ok=True)

client = Client()
request = Request(
    collection=Collection(id="C3685896708-LARC_CLOUD"),        # TEMPO NO2 L3 V04
    spatial=BBox(-125.0, 24.0, -66.5, 50.0),                   # CONUS
    temporal={"start": pd.Timestamp("2024-07-01 17:00"), "stop": pd.Timestamp("2024-07-01 18:00")},
    variables=["support_data/terrain_height"],
)
job = client.submit(request)
client.wait_for_processing(job, show_progress=True)
f = [x.result() for x in client.download_all(job, directory=str(out_dir))][0]

# Keep just the 2-D terrain field on the TEMPO grid
with xr.open_dataset(f) as root, xr.open_dataset(f, group="support_data") as sup:
    terrain = sup["terrain_height"].isel(time=0).assign_coords(
        latitude=root["latitude"].values, longitude=root["longitude"].values).load()
terrain.to_dataset(name="terrain_height").to_netcdf(out_dir / "tempo_terrain_height.nc")
print(terrain)