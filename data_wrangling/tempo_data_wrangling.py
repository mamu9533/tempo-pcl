"""
TEMPO Data Download Script

Downloads daily TEMPO L3 granules over a bounding box for a given date range
via NASA Harmony, with logging and per-day error handling.

By default the box covers the continental US (24.0-50.0 N, 125.0-66.5 W).

Usage:
    python download_tempo.py                                  # CONUS
    python download_tempo.py --start 2023-08-02 --end 2026-06-12
    python download_tempo.py --lat-buffer 13 --lon-buffer 29.25
    python download_tempo.py --product O3 --coordinate 33.94,-118.22 --buffer 0.25   # one buffer for both
    python download_tempo.py --product O3 --variables product/tropospheric_ozone_column \
        product/main_data_quality_flag support_data/eff_cloud_fraction            # only these variables
"""

import argparse
import logging
import os
import traceback
from pathlib import Path

import pandas as pd

from harmony import Client, Collection, Request, BBox


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Map friendly product names to their Harmony/CMR collection concept IDs.
COLLECTIONS = {
    "NO2": "C3685896708-LARC_CLOUD",
    "O3": "C3685896402-LARC_CLOUD",
    "HCHO": "C3685897141-LARC_CLOUD"
}

OFFICIAL_PATHS = {
    "NO2":r"D:\TEMPO_CONUS\L3_NO2_V04",
    "O3":r"D:\TEMPO_CONUS\L3_O3_V04",
    "HCHO":r"D:\TEMPO_CONUS\L3_HCHO_V04"
} 

DEFAULT_PRODUCT = "NO2"

# Continental US: center point plus separate half-widths (degrees) in latitude and
# longitude, giving 24.0-50.0 N and 125.0-66.5 W with a small margin.
DEFAULT_LAT, DEFAULT_LON = 37.0, -95.75
DEFAULT_LAT_BUFFER = 13.0
DEFAULT_LON_BUFFER = 29.25
# Previous LA setup, for reference: --coordinate 33.86,-117.71 --buffer 1.01

DEFAULT_START = "2023-08-02"
DEFAULT_END = "2026-11-06"
LOGFILE = "TEMPO_download_log.txt"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

def setup_logging(logfile: str = LOGFILE):
    logging.basicConfig(
        filename=logfile,
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def build_bounding_box(center_lat: float, center_lon: float, lat_buffer: float, lon_buffer: float):
    min_lat, max_lat = center_lat - lat_buffer, center_lat + lat_buffer
    min_lon, max_lon = center_lon - lon_buffer, center_lon + lon_buffer
    print(f"Bounding box: lat {min_lat:.2f} to {max_lat:.2f}, lon {min_lon:.2f} to {max_lon:.2f}")
    logging.info(f"Bounding box: lat {min_lat:.2f}..{max_lat:.2f}, lon {min_lon:.2f}..{max_lon:.2f}")
    return BBox(min_lon, min_lat, max_lon, max_lat)


# ---------------------------------------------------------------------------
# Resume support: skip days that are already downloaded
# ---------------------------------------------------------------------------

def marker_path(output_dir: str, day: pd.Timestamp) -> Path:
    """Empty file written after a day finishes downloading completely."""
    return Path(output_dir) / ".completed" / f"{day:%Y-%m-%d}"


def already_downloaded(output_dir: str, day: pd.Timestamp) -> bool:
    # Finished in a run that wrote markers
    if marker_path(output_dir, day).exists():
        return True
    # Downloaded before markers existed: granule file names contain the UTC date, e.g. ..._20240101T...
    existing = list(Path(output_dir).glob(f"*{day:%Y%m%d}T*"))
    if existing:
        logging.info(f"{day.date()}: {len(existing)} files found without a completion marker; treating as done")
        return True
    return False


# ---------------------------------------------------------------------------
# Core download logic
# ---------------------------------------------------------------------------

def download_day(harmony_client: Client, collection_id: str, bounding_box: BBox, day: pd.Timestamp, output_dir: str,
                 variables=None):
    if already_downloaded(output_dir, day):
        print(f"Skipping {day.date()}: already downloaded")
        return

    start_time = day
    stop_time = day + pd.Timedelta(hours=23, minutes=59, seconds=59)

    # Only ask for specific variables if some were given; otherwise Harmony returns everything
    extra = {"variables": variables} if variables else {}
    request = Request(
        collection=Collection(id=collection_id),
        spatial=bounding_box,
        temporal={"start": start_time, "stop": stop_time},
        **extra,
    )

    if not request.is_valid():
        print(f"Invalid Request geometry/time for {day.date()}\n")
        logging.warning(f"Invalid request for {day.date()}")
        return

    try:
        print(f"Valid Request: Submitting for {day.date()}...")
        job_id = harmony_client.submit(request=request)
        print(f"Job submitted. ID: {job_id}")
        logging.info(f"Began download for data on {day.date()} (job {job_id})")

        harmony_client.wait_for_processing(job_id, show_progress=True)

        print("Downloading files...")
        futures = harmony_client.download_all(job_id, directory=output_dir)
        file_paths = [f.result() for f in futures]
        print(f"Success! Saved {len(file_paths)} files.\n")
        logging.info(f"Saved {len(file_paths)} files for {day.date()}")

        # Mark the day complete only after every file has finished downloading
        marker_path(output_dir, day).touch()

    except Exception:
        print(f"Request failed for {day.date()}. See log for details.\n")
        logging.error(f"Error for {day.date()}:\n{traceback.format_exc()}")


def run(start: str, end: str, collection_id: str, center_lat: float, center_lon: float,
        lat_buffer: float, lon_buffer: float, output_dir: str, variables=None):
    setup_logging()

    # Create the output folder (and the marker folder) if they don't exist yet
    os.makedirs(os.path.join(output_dir, ".completed"), exist_ok=True)

    harmony_client = Client()
    bounding_box = build_bounding_box(center_lat, center_lon, lat_buffer, lon_buffer)
    date_range = pd.date_range(start=f"{start} 00:00:00", end=f"{end} 00:00:00", freq="1D")
    if variables:
        print(f"Requesting only: {', '.join(variables)}")
        logging.info(f"Variables: {variables}")

    for day in date_range:
        download_day(harmony_client, collection_id, bounding_box, day, output_dir, variables)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def parse_coordinate(value: str):
    """Parse a 'lat,lon' string into a (lat, lon) float tuple."""
    try:
        lat_str, lon_str = value.split(",")
        return float(lat_str.strip()), float(lon_str.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Coordinate must be in 'lat,lon' format, got: {value!r}"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download TEMPO L3 data via NASA Harmony.")
    parser.add_argument("--start", default=DEFAULT_START, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", default=DEFAULT_END, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--product",
        default=DEFAULT_PRODUCT,
        choices=sorted(COLLECTIONS.keys()),
        help="TEMPO product to download",
    )
    parser.add_argument(
        "--coordinate",
        default=f"{DEFAULT_LAT},{DEFAULT_LON}",
        type=parse_coordinate,
        help="Center coordinate as 'lat,lon', e.g. 33.94,-118.22",
    )
    parser.add_argument(
        "--lat-buffer",
        default=DEFAULT_LAT_BUFFER,
        type=float,
        help="Half-height of the box in degrees latitude",
    )
    parser.add_argument(
        "--lon-buffer",
        default=DEFAULT_LON_BUFFER,
        type=float,
        help="Half-width of the box in degrees longitude",
    )
    parser.add_argument(
        "--buffer",
        default=None,
        type=float,
        help="One buffer for both directions (overrides --lat-buffer and --lon-buffer)",
    )
    parser.add_argument(
        "--variables",
        nargs="*",
        default=None,
        help="Variables to download, as group/name (e.g. product/tropospheric_ozone_column). "
             "Leave out to download everything",
    )
    args = parser.parse_args()
    if args.buffer is not None:
        args.lat_buffer = args.lon_buffer = args.buffer
    return args


if __name__ == "__main__":
    args = parse_args()
    center_lat, center_lon = args.coordinate
    collection_id = COLLECTIONS[args.product]
    directory = OFFICIAL_PATHS[args.product]

    run(
        start=args.start,
        end=args.end,
        collection_id=collection_id,
        center_lat=center_lat,
        center_lon=center_lon,
        lat_buffer=args.lat_buffer,
        lon_buffer=args.lon_buffer,
        output_dir=directory,
        variables=args.variables,
    )