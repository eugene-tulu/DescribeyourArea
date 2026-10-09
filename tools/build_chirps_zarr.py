"""Build and refresh the CHIRPS tensor store.

CHIRPS is published as one GeoTIFF per month, so a job that wanted a rainfall
series paid for several hundred independent object opens. This collapses the whole
record into one contiguous, chunked array inside an Icechunk repository, so a
cell-month read is a few range reads against neighbouring chunks.

    # what the source currently offers, no writes and no image reads
    python -m tools.build_chirps_zarr --plan

    # a small slice, for testing
    python -m tools.build_chirps_zarr --start 2020-01 --end 2021-12

    # the full record
    python -m tools.build_chirps_zarr

    # after DEA publishes new months
    python -m tools.build_chirps_zarr

Configuration:

    CHIRPS_STORE_URI=s3://my-bucket/geocontextualize/chirps
    RAINFALL_S3_ENDPOINT=https://nyc3.digitaloceanspaces.com
    RAINFALL_S3_REGION=nyc3

Design notes worth knowing before changing any of this:

* **The stored grid is CHIRPS as published**, EPSG:4326 at 0.05 degrees, byte
  values unchanged apart from the nodata sentinel becoming NaN. Regridding
  happens where it can be versioned and labelled, not baked into the archive --
  ``registry.py`` declares ``native_grid_degrees=0.05`` and the product caveat
  compares CHIRPS's grid with ERA5's, and a reprojected store would falsify both.
  Equal-area correctness is obtained at read time by weighting cell means by
  cos(latitude), which is exact for a regular lat/lon grid and costs nothing.
* **A month the source does not have is a month of NaN, not a shorter axis.**
  CHIRPS has genuine holes (2023-12, 2024-07 and 2024-08 are absent) and a
  trailing publication lag. Concatenating only the months that exist would shift
  every later date, which is exactly the silent wrongness the rest of this
  codebase is careful to avoid.
* **Latitude is stored ascending.** The source GeoTIFFs are north-up, so row 0 is
  the northern edge; every consumer here expects south-up. Flipping the arrays
  without flipping the coordinates would leave each pixel paired with the wrong
  hemisphere, so both are flipped together and the order is asserted.
* **The build streams.** The full record is about 5 GB, so months are staged
  straight into an on-disk memmap and released rather than accumulated in memory.
* **Re-running is cheap.** Icechunk is content-addressed, so extending the array
  with new months re-uploads only the chunks whose contents changed.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import datetime
import json
import os
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

import rainfall
from rainfall import CHIRPS_BASE_URL, CHIRPS_NODATA

# CHIRPS v2.0 monthly starts in January 1981.
CHIRPS_RECORD_START = "1981-01-01"

# Grids this tool has seen. The source is a fixed 1500x1600 global-at-0.05
# product; anything else means the archive changed shape and a single cube can no
# longer represent it.
EXPECTED_WIDTH = 1500
EXPECTED_HEIGHT = 1600
EXPECTED_RESOLUTION = 0.05


def _as_month(value: str) -> datetime.date:
    """Accept ``YYYY-MM`` or ``YYYY-MM-DD`` and return the first of that month."""
    text = value.strip()
    if len(text) == 7:
        text = f"{text}-01"
    return datetime.date.fromisoformat(text).replace(day=1)


def month_labels(start: str, end: str) -> list[str]:
    """Every ``YYYY-MM`` label from ``start`` to ``end`` inclusive."""
    first, last = _as_month(start), _as_month(end)
    if first > last:
        raise ValueError(f"start {start} is after end {end}")
    out: list[str] = []
    year, month = first.year, first.month
    while (year, month) <= (last.year, last.month):
        out.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return out


def source_url(label: str) -> str:
    return CHIRPS_BASE_URL.format(ym=label.replace("-", "."))


def probe_month(label: str) -> bool:
    """Is this month published? A HEAD costs one small request, not a full read."""
    request = urllib.request.Request(source_url(label), method="HEAD")
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return int(response.headers.get("Content-Length", 0)) > 0
    except Exception:
        return False


def plan(start: str, end: str, workers: int) -> tuple[list[str], list[str], list[str]]:
    """Decide the time axis from HEAD requests alone.

    Returns ``(axis, holes, trailing)``. The axis ends at the last published
    month; holes inside it stay on the axis as NaN; a trailing run is the
    source's publication lag and is left off entirely.
    """
    labels = month_labels(start, end)
    print(f"requested {len(labels)} months ({labels[0]} .. {labels[-1]})")

    present: dict[str, bool] = {}
    with futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for label, ok in zip(labels, pool.map(probe_month, labels)):
            present[label] = ok

    published = [label for label in labels if present[label]]
    if not published:
        raise SystemExit(f"no month between {start} and {end} is published")

    axis = month_labels(start, published[-1])
    holes = [label for label in axis if not present[label]]
    trailing = [label for label in labels if not present[label] and label not in holes]

    print(f"axis {axis[0]} .. {axis[-1]}  ({len(axis)} months, "
          f"{len(axis) - len(holes)} published)")
    print(f"holes kept as NaN: {holes if holes else 'none'}")
    if trailing:
        print(f"trailing absent, left off the axis (publication lag): {trailing}")
    return axis, holes, trailing


def read_month(label: str) -> np.ndarray:
    """One month, float32, nodata as NaN, south-up.

    Reads the whole raster rather than a window: the files are 5.9 MB, so a
    windowed read buys nothing and costs extra round trips.
    """
    import rasterio

    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="TRUE"):
        with rasterio.open(source_url(label)) as src:
            band = src.read(1)
            transform = src.transform

    if src.width != EXPECTED_WIDTH or src.height != EXPECTED_HEIGHT:
        raise SystemExit(
            f"{label} is {src.width}x{src.height}, expected "
            f"{EXPECTED_WIDTH}x{EXPECTED_HEIGHT}; the archive shape changed and a "
            "single cube can no longer represent it"
        )
    for axis_step, name in ((transform.a, "longitude"), (transform.e, "latitude")):
        if abs(abs(axis_step) - EXPECTED_RESOLUTION) > 1e-9:
            raise SystemExit(
                f"{label} has a {abs(axis_step):g} degree {name} step, expected "
                f"{EXPECTED_RESOLUTION}"
            )

    array = band.astype("float32")
    # Two tests, both needed: the sentinel marks ocean and unprocessed cells, and
    # the source carries no other negative values, so anything below zero is
    # absent data rather than a reading.
    array[band == CHIRPS_NODATA] = np.nan
    array[array < 0] = np.nan
    return np.flipud(array), transform


def build(axis: list[str], holes: list[str], workers: int, label: str) -> dict:
    """Stage every published month into the array and commit it once."""
    import xarray as xr

    target = rainfall.chirps_store_target()
    if target is None:
        raise SystemExit(
            "CHIRPS_STORE_URI is not set, so there is nowhere to write. See the "
            "module docstring for the expected form."
        )

    workdir = Path(tempfile.gettempdir()) / f"chirps-build-{os.getpid()}"
    workdir.mkdir(parents=True, exist_ok=True)
    backing = workdir / "precip.f32"
    count = len(axis)
    nbytes = count * EXPECTED_HEIGHT * EXPECTED_WIDTH * 4
    print(f"\nstaging {count} x {EXPECTED_HEIGHT} x {EXPECTED_WIDTH} float32 "
          f"({nbytes / 1e9:.2f} GB) on disk")
    cube = np.memmap(backing, dtype="float32", mode="w+",
                     shape=(count, EXPECTED_HEIGHT, EXPECTED_WIDTH))
    cube[:] = np.nan

    geometry: list = []

    def stage(position: int, month: str) -> str:
        array, month_transform = read_month(month)
        cube[position] = array
        # Every month must carry the same geometry; keep the first as the
        # authority for the coordinates and let read_month enforce the rest.
        if not geometry:
            geometry.append(month_transform)
        return month

    wanted = [(index, month) for index, month in enumerate(axis)
              if month not in holes]
    started = time.perf_counter()
    try:
        with futures.ThreadPoolExecutor(max_workers=workers) as pool:
            pending = {pool.submit(stage, index, month): index
                       for index, month in wanted}
            written = 0
            for future in futures.as_completed(pending):
                index = pending[future]
                try:
                    month = future.result()
                except Exception as exc:  # noqa: BLE001 - report, keep going
                    print(f"  {axis[index]}  ERROR {type(exc).__name__}: "
                          f"{str(exc)[:70]}", flush=True)
                    continue
                written += 1
                if written % 25 == 0 or written == len(wanted):
                    rate = written / max(time.perf_counter() - started, 1e-9)
                    print(f"  staged {written}/{len(wanted)} "
                          f"({rate:.2f} months/s)", flush=True)

        missing = [axis[index] for index, _month in wanted
                   if np.isnan(cube[index]).all()]
        if missing:
            # A month that was present but read as entirely absent is a source or
            # network fault, not a hole. Writing it as NaN would quietly shorten
            # the record, so it is refused rather than absorbed.
            raise SystemExit(
                f"these months were published but read as entirely absent: "
                f"{missing[:6]}{' ...' if len(missing) > 6 else ''}. Re-run; if it "
                "persists the source is serving empty rasters."
            )

        # Coordinates from the source transform, flipped to match the flipped
        # arrays. Deriving one from the transform and the other from the array
        # order is how a south-up raster ends up under north-up coordinates.
        if geometry:
            transform = geometry[0]
            lons = (transform.c + 0.5 * transform.a
                    + transform.a * np.arange(EXPECTED_WIDTH))
            lats = (transform.f + 0.5 * transform.e
                    + transform.e * np.arange(EXPECTED_HEIGHT))[::-1]
        else:
            # No month was readable, so fall back to the published extent.
            lons = (-20.0 + EXPECTED_RESOLUTION / 2
                    + EXPECTED_RESOLUTION * np.arange(EXPECTED_WIDTH))
            lats = (40.0 - EXPECTED_RESOLUTION / 2
                    - EXPECTED_RESOLUTION * np.arange(EXPECTED_HEIGHT))[::-1]
        if not (lats[0] < lats[-1]):
            raise SystemExit(
                f"latitude order is still descending ({lats[0]} .. {lats[-1]}); the "
                "store would be south-up data under north-up coordinates"
            )

        times = xr.date_range(axis[0], periods=count, freq="MS", use_cftime=False)
        dataset = xr.Dataset(
            {
                rainfall.CHIRPS_STORE_VARIABLE: (
                    ("time", "latitude", "longitude"), np.asarray(cube)
                )
            },
            coords={"time": times, "latitude": lats, "longitude": lons},
            attrs={
                "title": "CHIRPS v2.0 monthly precipitation, as published",
                "summary": (
                    "CHIRPS blended satellite-and-station precipitation, ingested "
                    "unmodified at its published 0.05 degree grid. The nodata "
                    "sentinel is stored as NaN. Equal-area weighting is applied at "
                    "read time, not here."
                ),
                "institution": "Climate Hazards Group, University of California "
                               "Santa Barbara",
                "source": "CHIRPS v2.0 monthly, mirrored by Digital Earth Africa",
                "source_url_template": CHIRPS_BASE_URL,
                "doi": "10.15780/G2V513",
                "license": "CC-BY-4.0",
                "keywords": "precipitation, rainfall, CHIRPS, Africa",
                "geospatial_lat_min": float(lats.min()),
                "geospatial_lat_max": float(lats.max()),
                "geospatial_lon_min": float(lons.min()),
                "geospatial_lon_max": float(lons.max()),
                "geospatial_lat_units": "degrees_north",
                "geospatial_lon_units": "degrees_east",
                "geospatial_lat_resolution": f"{EXPECTED_RESOLUTION} degree",
                "geospatial_lon_resolution": f"{EXPECTED_RESOLUTION} degree",
                "geospatial_bounds": (
                    f"POLYGON(({lons.min()} {lats.min()}, {lons.max()} {lats.min()}, "
                    f"{lons.max()} {lats.max()}, {lons.min()} {lats.max()}, "
                    f"{lons.min()} {lats.min()}))"
                ),
                "geospatial_bounds_crs": "EPSG:4326",
                "crs": "EPSG:4326",
                "time_coverage_start": axis[0],
                "time_coverage_end": axis[-1],
                "time_coverage_resolution": "P1M",
                "missing_months": json.dumps(holes),
                "processing_version": rainfall.CHIRPS_STORE_BUILD_VERSION,
                "date_created": datetime.date.today().isoformat(),
                "note": (
                    f"{label}: months with no published CHIRPS file are present as "
                    "NaN rather than omitted, so the time axis is complete."
                ),
            },
        )
        dataset[rainfall.CHIRPS_STORE_VARIABLE].encoding.update(
            chunks=rainfall.CHIRPS_STORE_CHUNKS, dtype="float32",
            fill_value=np.nan, _FillValue=np.nan,
        )

        print(f"\nopening store {target['bucket']}/{target['prefix']} "
              f"(endpoint {target['endpoint']})")
        _repo, session = rainfall._open_chirps_store(writable=True)
        group = rainfall.CHIRPS_STORE_GROUP
        try:
            existing = xr.open_zarr(session.store, group=group,
                                    consolidated=False, chunks=None)
            print(f"  existing array {dict(existing.sizes)} -> "
                  f"{dict(dataset.sizes)}")
        except Exception:
            print(f"  creating group {group}")

        dataset.to_zarr(session.store, group=group, mode="a", consolidated=False)
        snapshot = session.commit(
            f"CHIRPS {axis[0]}..{axis[-1]} ({count} months, "
            f"{len(holes)} holes, build {rainfall.CHIRPS_STORE_BUILD_VERSION})"
        )
        print(f"  committed {snapshot}")
    finally:
        del cube
        backing.unlink(missing_ok=True)
        try:
            workdir.rmdir()
        except OSError:
            pass

    return {"snapshot": snapshot, "months": count,
            "present": count - len(holes), "holes": holes}


def extend(dry_run: bool = False) -> dict:
    """Append CHIRPS months DEA has published since the store was last built.

    DEA publishes a new CHIRPS month roughly every month, so a store built once
    is always a couple of months behind. This closes that gap in the background:
    it reads the store's last month, HEAD-probes DEA for anything after it, and
    extends the array -- holes stay as NaN so the time axis stays complete and
    every later date keeps its row. It is idempotent: nothing published past the
    store is a no-op, so it is safe to run on any cadence.

    The latitude/longitude coordinates are taken from the store itself rather
    than recomputed, so they match the existing array exactly and ``to_zarr``
    appends along time without reconciling a drifting axis. ``dry_run`` reports
    what would happen without touching the store.
    """
    import xarray as xr

    target = rainfall.chirps_store_target()
    if target is None:
        return {"status": "unconfigured", "added": 0}

    _repo, session = rainfall._open_chirps_store(writable=False)
    group = rainfall.CHIRPS_STORE_GROUP
    existing = xr.open_zarr(session.store, group=group, consolidated=False, chunks=None)
    store_last = str(existing.time.dt.strftime("%Y-%m").values[-1])
    latitude = np.asarray(existing.latitude.values, dtype="float64")
    longitude = np.asarray(existing.longitude.values, dtype="float64")

    # First month after the store, through this month.
    year, month = int(store_last[:4]), int(store_last[5:7])
    month += 1
    if month == 13:
        month, year = 1, year + 1
    start = f"{year:04d}-{month:02d}"
    end = datetime.date.today().replace(day=1).isoformat()
    if start > end:
        return {"status": "current", "added": 0, "store_last": store_last}

    try:
        axis, holes, _trailing = plan(start, end, 8)
    except SystemExit:
        # No month after the store is published yet (publication lag).
        return {"status": "current", "added": 0, "store_last": store_last}
    new_axis = [label for label in axis if label > store_last]
    if not new_axis:
        return {"status": "current", "added": 0, "store_last": store_last}
    if dry_run:
        return {"status": "would_extend", "would_add": len(new_axis),
                "range": f"{new_axis[0]}..{new_axis[-1]}", "holes": holes}

    cube = np.full((len(new_axis), EXPECTED_HEIGHT, EXPECTED_WIDTH), np.nan, dtype="float32")
    for position, label in enumerate(new_axis):
        if label in holes:
            continue
        array, _transform = read_month(label)
        cube[position] = array

    times = xr.date_range(f"{new_axis[0]}-01", periods=len(new_axis), freq="MS",
                          use_cftime=False)
    dataset = xr.Dataset(
        {rainfall.CHIRPS_STORE_VARIABLE: (("time", "latitude", "longitude"), cube)},
        coords={"time": times, "latitude": latitude, "longitude": longitude},
    )
    dataset[rainfall.CHIRPS_STORE_VARIABLE].encoding.update(
        chunks=rainfall.CHIRPS_STORE_CHUNKS, dtype="float32",
        fill_value=np.nan, _FillValue=np.nan,
    )

    _repo, session = rainfall._open_chirps_store(writable=True)
    dataset.to_zarr(session.store, group=group, mode="a", append_dim="time",
                    consolidated=False)
    snapshot = session.commit(
        f"CHIRPS extend {store_last} -> {new_axis[-1]} "
        f"({len(new_axis)} months, {len(holes)} holes)")
    # The read-only handle opened this process's view of the old snapshot; drop it
    # so the next read sees the months just appended.
    rainfall._invalidate_process_caches()
    return {"status": "extended", "added": len(new_axis),
            "range": f"{new_axis[0]}..{new_axis[-1]}", "snapshot": str(snapshot)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--start", default=CHIRPS_RECORD_START,
                        help=f"first month (default {CHIRPS_RECORD_START})")
    parser.add_argument("--end", default=None,
                        help="last month to consider (default: this month)")
    parser.add_argument("--plan", action="store_true",
                        help="report what would be written and stop")
    parser.add_argument("--threads", type=int, default=8,
                        help="concurrent source reads (default 8)")
    parser.add_argument("--label", default=os.getenv("BUILD_LABEL", "build"),
                        help="short description recorded in the array note")
    parser.add_argument("--extend", action="store_true",
                        help="append only the months DEA has published since the "
                             "store was last built, instead of a full rebuild")
    parser.add_argument("--dry-run", action="store_true",
                        help="with --extend, report what would be appended and stop")
    args = parser.parse_args(argv)

    end = args.end or datetime.date.today().replace(day=1).isoformat()
    print(f"source: {CHIRPS_BASE_URL}")
    print(f"store:  {rainfall.chirps_store_target() or 'NOT CONFIGURED'}")
    print(f"chunks: {rainfall.CHIRPS_STORE_CHUNKS} (months, lat, lon)\n")

    if args.extend:
        started = time.perf_counter()
        result = extend(dry_run=args.dry_run)
        result["seconds"] = round(time.perf_counter() - started, 1)
        print(f"extend: {result}")
        return 0

    started = time.perf_counter()
    axis, holes, _trailing = plan(args.start, end, args.threads)
    print(f"\nplanned in {time.perf_counter() - started:.0f}s")

    if args.plan:
        size = len(axis) * EXPECTED_HEIGHT * EXPECTED_WIDTH * 4
        print(f"\nplan: {len(axis)} months, {len(axis) - len(holes)} present, "
              f"{len(holes)} NaN holes")
        print(f"array before compression ~{size / 1e9:.2f} GB")
        return 0

    result = build(axis, holes, args.threads, args.label)
    print(f"\ndone: {result['months']} months, {result['present']} present, "
          f"{len(result['holes'])} holes, chunks {rainfall.CHIRPS_STORE_CHUNKS}")
    print(f"snapshot {result['snapshot']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())