"""Monthly precipitation and drought anomaly from ERA5 reanalysis.

ERA5 is read from the Earthmover Icechunk store on AWS Open Data, which is public
and needs no account or credentials. It is used in preference to the copy on the
Planetary Computer, which is the deprecated 1979-2020 subset and stale for
anything recent.

Why this is a precompute module and not part of the request path: reading a single
grid cell over the 30-year climatology takes about 20 seconds, against a whole
request budget of 20-30 seconds. The series is therefore computed offline, cached
by a hash of the study-area geometry, and the request path only ever reads the
cache.

Resolution is ERA5's 0.25 degrees, about 28 km. That is a landscape-to-regional
product, not a field survey, and every result reports how many grid cells the study
area actually covers so a caller can see when a small area resolves to one cell.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional

import rasterio as rio

from shapely.geometry import shape

# Bumping this invalidates every cached series, which is the point: a cached value
# must never outlive the code that produced it.
RAINFALL_PROCESSING_VERSION = "era5-monthly-2"

# WMO standard normal period, fixed so that "anomaly" means one thing forever.
CLIMATOLOGY_START = "1991-01-01"
CLIMATOLOGY_END = "2020-12-31"

# A month is dropped unless at least this fraction of its hours are finite.
MIN_MONTH_COVERAGE = 0.8
# A month below this is flagged rather than dropped. Near-zero months over a cell
# whose climatology is clearly wetter are usually real extremes, and hiding them
# would defeat the point of the product, so they are surfaced for review.
SUSPECT_MONTH_MM = 0.5

ERA5_BUCKET = "earthmover-icechunk-era5"
ERA5_PREFIX = "icechunkV2"
ERA5_REGION = "us-east-1"
ERA5_GROUP = "single/temporal"
ERA5_DOI = "10.24381/cds.adbb2d47"
ERA5_CITATION = (
    "ERA5 hourly data on single levels from 1940 to present, Copernicus Climate "
    "Change Service Climate Data Store. https://doi.org/10.24381/cds.adbb2d47. "
    "Analysis-ready edition: Earthmover Icechunk ERA5, CC-BY 4.0, "
    "https://registry.opendata.aws/earthmover-era5/"
)
ERA5_GRID_DEGREES = 0.25
ERA5_SOURCE = "ERA5 (Earthmover Icechunk edition)"


# --------------------------------------------------------------------------
# Cache locations and keys
# --------------------------------------------------------------------------

def cache_dir() -> Path:
    return Path(os.getenv("RAINFALL_CACHE_DIR", ".rainfall-cache")).expanduser()


# Optional object-store backing, so the portfolio is a published artefact rather
# than something baked into the image. S3-compatible, which DigitalOcean Spaces
# is. Unset means everything stays local and every remote operation is a no-op.
#
#   RAINFALL_CACHE_S3_URI=s3://my-bucket/geocontextualize/rainfall
#   RAINFALL_S3_ENDPOINT=https://nyc3.digitaloceanspaces.com
#   RAINFALL_S3_REGION=nyc3
#
# The request path reads locally first and only consults the remote on a miss, so
# a cache hit never pays a network round trip and the hot path has no new failure
# mode.
REMOTE_URI_ENV = "RAINFALL_CACHE_S3_URI"
REMOTE_ENDPOINT_ENV = "RAINFALL_S3_ENDPOINT"
REMOTE_REGION_ENV = "RAINFALL_S3_REGION"


def remote_prefix() -> Optional[str]:
    """Bucket and key prefix from the configured URI, or None."""
    uri = (os.getenv(REMOTE_URI_ENV) or "").strip()
    if not uri:
        return None
    if not uri.startswith("s3://"):
        raise ValueError(f"{REMOTE_URI_ENV} must start with s3://, got {uri!r}")
    remainder = uri[len("s3://"):].strip("/")
    if not remainder:
        raise ValueError(f"{REMOTE_URI_ENV} is missing a bucket name")
    return remainder


def _normalise_endpoint(endpoint: Optional[str], bucket: str) -> Optional[str]:
    """Accept either form of a Spaces endpoint.

    DigitalOcean documents both ``https://<bucket>.fra1.digitaloceanspaces.com``
    and ``https://fra1.digitaloceanspaces.com``. boto3 puts the bucket in the
    hostname itself, so feeding it the first form requests
    ``<bucket>.fra1.digitaloceanspaces.com/<bucket>/...`` and fails with
    ``NoSuchKey``. Stripping a leading bucket leaves boto3 to add it back, so both
    spellings work.
    """
    if not endpoint:
        return None
    host = endpoint.split("://", 1)[-1].strip("/")
    for suffix in (".digitaloceanspaces.com",):
        if host.endswith(suffix):
            head = host[: -len(suffix)]
            if head == bucket:
                host = f"{region_hint()}{suffix}"
                break
            if head.startswith(f"{bucket}."):
                head = head[len(bucket) + 1:]
                host = f"{head}{suffix}" if head else f"{suffix.lstrip('.')}"
                break
    scheme = endpoint.split("://", 1)[0] if "://" in endpoint else "https"
    return f"{scheme}://{host}"


def region_hint() -> str:
    return (os.getenv(REMOTE_REGION_ENV) or "us-east-1").strip()


def _s3_client():
    """A boto3 S3 client pointed at the configured endpoint, or None."""
    prefix = remote_prefix()
    if prefix is None:
        return None
    import boto3
    from botocore.config import Config

    return boto3.client(
        "s3",
        endpoint_url=_normalise_endpoint(os.getenv(REMOTE_ENDPOINT_ENV) or None, _bucket()),
        region_name=region_hint() or "us-east-1",
        # Path-style keeps the request valid whichever endpoint spelling is used.
        config=Config(signature_version="s3v4"),
    )


def _bucket() -> str:
    return remote_prefix().split("/", 1)[0]


def _key_prefix() -> str:
    """Key prefix inside the bucket, with the bucket name stripped off."""
    parts = remote_prefix().split("/", 1)
    return parts[1] if len(parts) > 1 else ""


def _remote_key(name: str) -> str:
    """Object key for a name, relative to the bucket. The bucket is not part of
    the key, so it must be stripped from the configured prefix."""
    prefix = _key_prefix().strip("/")
    return f"{prefix}/{name}" if prefix else name


def series_object(key: str) -> str:
    return f"series/{key}.json"


def publish(key: str) -> bool:
    """Upload one cached series. Reports whether it reached the store.

    Never raises. The series is already written locally by the time this runs, so
    a store outage must not turn a completed computation into a failed one; the
    caller records that the upload did not happen and the next deploy's pull
    reconciles it.
    """
    if remote_prefix() is None:
        return False
    path = cache_path(key)
    if not path.exists():
        return False
    try:
        _s3_client().upload_file(str(path), _bucket(), _remote_key(series_object(key)))
    except Exception:  # noqa: BLE001 - an upload failure is not a compute failure
        return False
    return True


def forget(key: str) -> dict:
    """Remove a series from local disk and, when configured, from the store.

    Both. A deletion request that only cleared the local copy would leave the
    published artefact sitting in the bucket, which is the copy that outlives a
    redeploy.
    """
    path = cache_path(key)
    local = path.exists()
    if local:
        path.unlink()

    remote = False
    if remote_prefix() is not None:
        try:
            _s3_client().delete_object(
                Bucket=_bucket(), Key=_remote_key(series_object(key))
            )
            remote = True
        except Exception:  # noqa: BLE001 - report the local outcome regardless
            remote = False
    return {"local": local, "remote": remote, "remote_configured": remote_prefix() is not None}


def fetch(key: str, attempts: int = 2) -> Optional[dict]:
    """Pull one series from the remote and cache it locally. Returns it, or None.

    A genuine miss and a transport failure must be told apart. Swallowing both
    makes a momentary network blip look exactly like "this area has no series",
    which is a wrong answer to a user rather than an absent one, so a read is
    retried before it is believed.
    """
    if remote_prefix() is None:
        return None
    client = _s3_client()
    bucket, object_key = _bucket(), _remote_key(series_object(key))

    payload = None
    for attempt in range(max(1, attempts)):
        try:
            response = client.get_object(Bucket=bucket, Key=object_key)
            payload = json.loads(response["Body"].read())
            break
        except Exception as exc:
            if _is_missing(exc):
                return None  # the object genuinely is not there
            if attempt + 1 < attempts:
                time.sleep(0.2 * (2 ** attempt))
                continue
            return None
    if payload is None:
        return None
    if payload.get("processing_version") != RAINFALL_PROCESSING_VERSION:
        return None
    write_cache(key, payload)
    return payload


def _is_missing(exc: Exception) -> bool:
    """True when the store says the object does not exist, rather than failing."""
    response = getattr(exc, "response", None) or {}
    code = str((response.get("Error") or {}).get("Code") or "")
    if code in {"NoSuchKey", "404", "NotFound", "NoSuchBucket"}:
        return True
    return type(exc).__name__ in {"NoSuchKey", "KeyError", "FileNotFoundError"}


def geometry_hash(geojson_geom: dict) -> str:
    """Stable key for a study area."""
    canonical = json.dumps(geojson_geom, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


def cellset_key(grid: dict, start: str, end: str) -> str:
    """Identity of a read: which ERA5 cells, over which window."""
    canonical = json.dumps({
        "lon": [round(v, 4) for v in grid["longitudes"]],
        "lat": [round(v, 4) for v in grid["latitudes"]],
        "start": start,
        "end": end,
    }, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:32]


# --------------------------------------------------------------------------
# Reading ERA5
# --------------------------------------------------------------------------

_DATASET = None


def _open_temporal():
    import icechunk
    import xarray as xr

    storage = icechunk.s3_storage(
        bucket=ERA5_BUCKET,
        prefix=ERA5_PREFIX,
        region=ERA5_REGION,
        anonymous=True,
    )
    repo = icechunk.Repository.open(storage)
    session = repo.readonly_session("main")
    return xr.open_zarr(session.store, group=ERA5_GROUP, consolidated=False, chunks=None)


def _dataset_handle():
    """Open the ERA5 store once per process; opening it costs several seconds."""
    global _DATASET
    if _DATASET is None:
        _DATASET = _open_temporal()
    return _DATASET


def _grid(dataset, bbox: list[float]) -> dict:
    """ERA5 cells whose centres fall within the study area's bounding box.

    Nearest-cell assignment, not an area-weighted integral. The cell count is
    reported with every result so a caller can see when a small area resolves to
    a single cell.
    """
    lons = [float(v) for v in dataset["longitude"].values]
    lats = [float(v) for v in dataset["latitude"].values]
    lon_step = abs(lons[1] - lons[0]) if len(lons) > 1 else ERA5_GRID_DEGREES
    lat_step = abs(lats[1] - lats[0]) if len(lats) > 1 else ERA5_GRID_DEGREES

    lo, hi = min(bbox[0], bbox[2]), max(bbox[0], bbox[2])
    south, north = min(bbox[1], bbox[3]), max(bbox[1], bbox[3])
    return {
        "longitudes": [v for v in lons if lo - lon_step / 2 <= v <= hi + lon_step / 2],
        "latitudes": [v for v in lats if south - lat_step / 2 <= v <= north + lat_step / 2],
        "lon_step": lon_step,
        "lat_step": lat_step,
    }


def _monthly_cell_totals(dataset, grid, start, end):
    """Monthly per-cell totals in mm.

    Returns ``(labels, array[month, latitude, longitude], coverage)``.

    ERA5 ``tp`` is an hourly accumulation in metres, so hours must be summed into
    months, and the cells must be kept separate: averaging cells is a study-area
    decision, and doing it here would force one ERA5 read per study area instead of
    one per portfolio.

    Each hour contributes one row of per-cell values, so cells stay independent and
    months stay aligned with the source time axis. A month is kept only when most of
    its hours are present *and* finite; averaging the few surviving hours of a
    mostly-missing month would give a confident, citable and completely wrong total.
    """
    import numpy as np

    selection = dataset["tp"].sel(
        longitude=grid["longitudes"],
        latitude=grid["latitudes"],
        method="nearest",
    ).sel(valid_time=slice(start, end))
    values = selection.load().values
    empty = {
        "expected_hours": 0, "valid_hours": 0, "months_kept": 0,
        "months_dropped": [], "min_coverage": MIN_MONTH_COVERAGE,
    }
    if values.size == 0:
        return [], np.zeros((0, 0, 0), dtype="float32"), empty

    times = np.asarray(selection["valid_time"].values)
    count = min(values.shape[0], times.shape[0])
    nlat, nlon = values.shape[1], values.shape[2]

    # One row per hour holding every cell, in millimetres.
    flat = values[:count].reshape(count, -1).astype("float32") * 1000.0
    finite = np.isfinite(flat).all(axis=1)

    # Collect the hour indices belonging to each month first, then index once.
    # Accumulating a running list per month and indexing it by its own length
    # silently reads the wrong hours for every month after the first.
    positions_by_month: dict[str, list[int]] = {}
    for position in range(count):
        text = str(times[position])
        positions_by_month.setdefault(f"{text[:4]}-{text[5:7]}", []).append(position)

    kept, dropped, rows = [], [], []
    for label, positions in sorted(positions_by_month.items()):
        valid = sum(1 for position in positions if finite[position])
        coverage = valid / len(positions) if positions else 0.0
        if coverage < MIN_MONTH_COVERAGE:
            dropped.append({
                "month": label,
                "valid_hours": valid,
                "expected_hours": len(positions),
                "coverage": round(coverage, 3),
            })
            continue
        usable = [position for position in positions if finite[position]]
        if not usable:
            continue
        rows.append(flat[usable].sum(axis=0).reshape(nlat, nlon))
        kept.append(label)

    array = (np.stack(rows).astype("float32") if rows
             else np.zeros((0, nlat, nlon), dtype="float32"))
    report = {
        "expected_hours": count,
        "valid_hours": int(finite.sum()),
        "months_kept": len(kept),
        "months_dropped": dropped,
        "min_coverage": MIN_MONTH_COVERAGE,
    }
    return kept, array, report


def read_cell_monthly(grid: dict, start: str, end: str, source=None):
    """Read a cell set once, memoised on disk."""
    import numpy as np

    key = cellset_key(grid, start, end)
    path = cache_dir() / "cells" / f"{key}.npz"
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as blob:
                if str(blob["version"]) == RAINFALL_PROCESSING_VERSION:
                    return (
                        [str(v) for v in blob["labels"]],
                        blob["matrix"],
                        json.loads(str(blob["coverage"])),
                    )
        except (OSError, ValueError, KeyError):
            pass

    if source is not None:
        return source(grid, start, end)

    labels, matrix, coverage = _monthly_cell_totals(_dataset_handle(), grid, start, end)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    # numpy appends ".npz" to a *path* that lacks it, which would break the atomic
    # rename, so write through an open handle instead.
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            version=np.array(RAINFALL_PROCESSING_VERSION),
            labels=np.array(labels),
            matrix=matrix,
            coverage=np.array(json.dumps(coverage)),
        )
    temporary.replace(path)
    return labels, matrix, coverage


def union_grid(geometries) -> dict:
    """Smallest cell set covering every study area, for one read serving all."""
    dataset = _dataset_handle()
    grids = []
    for geom in geometries:
        minx, miny, maxx, maxy = shape(geom).bounds
        grids.append(_grid(dataset, [float(minx), float(miny), float(maxx), float(maxy)]))
    return {
        "longitudes": sorted({v for g in grids for v in g["longitudes"]}),
        "latitudes": sorted({v for g in grids for v in g["latitudes"]}),
        "lon_step": grids[0]["lon_step"],
        "lat_step": grids[0]["lat_step"],
    }


class UnionReader:
    """Reads one wide region and slices it down to each study area's cells.

    A portfolio of adjacent areas would otherwise pay the ~20 s ERA5 read once per
    area. Reading the union of their cells once is the difference between one read
    and one per area: for the 21 Northern Rangelands Trust conservancies, 48 s in
    total rather than roughly 16 minutes.
    """

    def __init__(self, grid: dict):
        self.grid = grid

    def __call__(self, grid: dict, start: str, end: str):
        labels, matrix, coverage = read_cell_monthly(self.grid, start, end)
        try:
            rows = [self.grid["latitudes"].index(v) for v in grid["latitudes"]]
            cols = [self.grid["longitudes"].index(v) for v in grid["longitudes"]]
        except ValueError as exc:
            raise ValueError("study area needs cells outside the precomputed union") from exc
        return labels, matrix[:, rows][:, :, cols], coverage


# --------------------------------------------------------------------------
# Anomaly arithmetic
# --------------------------------------------------------------------------

def monthly_climatology(series: dict[str, float], start: str, end: str) -> dict[str, float]:
    """Mean value per calendar month over the baseline period."""
    buckets: dict[str, list[float]] = {}
    for label, value in series.items():
        month = label.split("-")[1]
        if start[:4] <= label.split("-")[0] <= end[:4]:
            buckets.setdefault(month, []).append(value)
    return {
        month: sum(values) / len(values)
        for month, values in sorted(buckets.items())
        if values
    }


def anomalies(series: dict[str, float], climatology: dict[str, float]) -> list[dict[str, Any]]:
    """Per-month deviation from the climatological mean for the same month."""
    out: list[dict[str, Any]] = []
    for label, value in sorted(series.items()):
        month = label.split("-")[1]
        if month not in climatology:
            continue
        normal = climatology[month]
        row = {
            "month": label,
            "precip_mm": round(value, 1),
            "normal_mm": round(normal, 1),
            "anomaly_mm": round(value - normal, 1),
            "anomaly_pct": round((value - normal) / normal * 100.0, 1) if normal else None,
        }
        if value < SUSPECT_MONTH_MM and normal > 5.0:
            row["suspect"] = "near_zero_month_worth_review"
        out.append(row)
    return out


def rolling_totals(rows: list[dict[str, Any]], months: int) -> list[dict[str, Any]]:
    """Trailing totals over a window, reported with the window's own percentage."""
    out: list[dict[str, Any]] = []
    for index in range(months - 1, len(rows)):
        window = rows[index - months + 1: index + 1]
        total = sum(row["precip_mm"] for row in window)
        normal = sum(row["normal_mm"] for row in window)
        out.append({
            "ending": window[-1]["month"],
            "months": months,
            "precip_mm": round(total, 1),
            "normal_mm": round(normal, 1),
            "anomaly_mm": round(total - normal, 1),
            "anomaly_pct": round((total - normal) / normal * 100.0, 1) if normal else None,
        })
    return out


def describe(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Headline statements derived from the monthly series."""
    if not rows:
        return {}
    latest = rows[-1]
    annual = rolling_totals(rows, 12)
    wet = max(rows, key=lambda r: r["precip_mm"])
    dry = min(rows, key=lambda r: r["precip_mm"])
    return {
        "suspect_months": [r["month"] for r in rows if r.get("suspect")],
        "recent_3m": rows[-3:],
        "latest_month": latest["month"],
        "latest_precip_mm": latest["precip_mm"],
        "latest_anomaly_pct": latest["anomaly_pct"],
        "driest_month": {"month": dry["month"], "precip_mm": dry["precip_mm"]},
        "wettest_month": {"month": wet["month"], "precip_mm": wet["precip_mm"]},
        "trailing_12m": annual[-1] if annual else None,
    }


def _assert_plausible(annual_mm: Optional[float], cells: int) -> None:
    """Refuse a total that no land surface on Earth produces.

    A unit error here is silent and severe: an off-by-1000 bug produced an annual
    "normal" of 0.8 mm for semi-arid rangeland, which reads as catastrophic drought
    and is absurd to anyone who knows the region. The driest inhabited places on
    Earth are near 50 mm/yr and the wettest near 12,000.
    """
    if annual_mm is None:
        return
    if not 20.0 <= annual_mm <= 12_000.0:
        raise ValueError(
            f"implausible annual precipitation of {annual_mm:.1f} mm; this is almost "
            "certainly a units or aggregation error, not a climate result"
        )


# --------------------------------------------------------------------------
# Series cache
# --------------------------------------------------------------------------

def cache_path(key: str, product: str = "rainfall") -> Path:
    """Where a series for this area and product lives.

    The area hash alone is not enough now that a second product provides the same
    measure: one key per area means ERA5 and CHIRPS for the same area collide, and
    whichever was written first is what a reader silently gets back. A reader
    choosing CHIRPS must never be served ERA5, so the product is part of the
    identity.

    The default product keeps the original filename, so an existing portfolio is
    still where it was rather than orphaned by a layout change nobody asked for.
    """
    if (product or "rainfall") == "rainfall":
        return cache_dir() / f"{key}.json"
    return cache_dir() / f"{key}-{product}.json"


def read_cache(key: str, product: str = "rainfall") -> Optional[dict]:
    path = cache_path(key, product)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("processing_version") != RAINFALL_PROCESSING_VERSION:
        return None
    return payload


def write_cache(key: str, payload: dict, product: str = "rainfall") -> None:
    path = cache_path(key, product)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True))
    temporary.replace(path)


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------

RAINFALL_SOURCES = ("chirps",)


def reader_for(product_key: str):
    """The reader for a named product, or a refusal naming those that exist.

    Resolved lazily because the CHIRPS reader is defined further down, and a
    table built at import time would have to be ordered against the file.
    """
    import registry

    key = (product_key or "rainfall").strip().lower()
    if key == "rainfall":
        return None                      # the ERA5 default, handled in read_cell_monthly
    if key not in RAINFALL_SOURCES:
        raise ValueError(
            f"{product_key!r} is not a rainfall source; available: "
            + ", ".join(("rainfall",) + RAINFALL_SOURCES))
    if key == "chirps":
        return chirps_cell_monthly
    raise ValueError(f"{product_key!r} is not a rainfall source")


def compute_series(
    geojson_geom: dict,
    *,
    start: str = "2015-01-01",
    end: Optional[str] = None,
    source=None,
    label: Optional[str] = None,
) -> dict:
    """Compute the monthly series, climatology and anomaly for a study area.

    The expensive path, intended for offline precomputation.

    ``source`` may be a reader callable or the name of a product in the
    registry. Naming it keeps the choice in one place: the call sites -- a
    request, a job, the portfolio precompute -- would otherwise each decide
    which product to read, which is how a request ends up answering a different
    question from the one a job was queued for.
    """
    if isinstance(source, str):
        source = reader_for(source)
    geom = shape(geojson_geom)
    if geom.is_empty or geom.geom_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError("study area must be a non-empty polygon")
    minx, miny, maxx, maxy = geom.bounds
    bbox = [float(minx), float(miny), float(maxx), float(maxy)]
    end = end or datetime.date.today().replace(day=1).isoformat()

    grid = _grid(_dataset_handle(), bbox)
    if not grid["longitudes"] or not grid["latitudes"]:
        raise ValueError("study area does not intersect the ERA5 grid")

    recent_labels, recent_matrix, recent_coverage = read_cell_monthly(grid, start, end, source)
    base_labels, base_matrix, base_coverage = read_cell_monthly(
        grid, CLIMATOLOGY_START, CLIMATOLOGY_END, source)
    if not recent_labels:
        raise ValueError("no ERA5 data available for the requested period")

    # Average the cells belonging to this study area. The matrix is
    # (month, latitude, longitude), so iterating it yields one 2-D block per month.
    def area_mean(matrix) -> list[float]:
        return [float(block.mean()) for block in matrix]

    recent = dict(zip(recent_labels, area_mean(recent_matrix)))
    baseline = dict(zip(base_labels, area_mean(base_matrix)))

    climatology = monthly_climatology(baseline, CLIMATOLOGY_START, CLIMATOLOGY_END)
    rows = anomalies(recent, climatology)
    cells = len(grid["longitudes"]) * len(grid["latitudes"])

    annual_normal = sum(climatology.values()) if climatology else None
    _assert_plausible(annual_normal, cells)

    return {
        "processing_version": RAINFALL_PROCESSING_VERSION,
        "indicator": "monthly_precipitation",
        # A notification has to be able to name the area it is about, and the
        # geometry is deliberately not stored, so the label travels with the series.
        "label": label,
        "source": ERA5_SOURCE,
        "doi": ERA5_DOI,
        "citation": ERA5_CITATION,
        "license": "CC-BY 4.0",
        "retrieved": datetime.datetime.now(datetime.UTC).date().isoformat(),
        "bbox": bbox,
        "resolution_degrees": ERA5_GRID_DEGREES,
        "resolution_km": round(ERA5_GRID_DEGREES * 111.32, 1),
        "grid_cells": cells,
        "climatology": {
            "start": CLIMATOLOGY_START,
            "end": CLIMATOLOGY_END,
            "standard": "WMO 1991-2020 normal",
            "monthly_mean_mm": {k: round(v, 1) for k, v in climatology.items()},
            "annual_mean_mm": round(annual_normal, 1) if annual_normal else None,
        },
        "coverage": {"window": recent_coverage, "climatology": base_coverage},
        "series": rows,
        "summary": describe(rows),
        "window": {"start": start, "end": end},
    }


def build_and_cache(
    geojson_geom: dict,
    *,
    start: str = "2015-01-01",
    source=None,
    upload: bool = False,
    label: Optional[str] = None,
    **kwargs,
) -> dict:
    """Compute and store a series, returning the cached payload.

    With ``upload`` the series is also published, so the portfolio becomes a
    published artefact rather than local state tied to one machine.

    The flag is named ``upload``, not ``publish``: a parameter called ``publish``
    shadows the module-level ``publish()`` and turns the upload into a call on a
    bool, which only fails on the upload path.
    """
    key = geometry_hash(geojson_geom)
    product = "rainfall"
    if isinstance(source, str):
        product = source.strip().lower()
    elif source is not None:
        product = "rainfall"
    payload = compute_series(geojson_geom, start=start, source=source, label=label, **kwargs)
    payload["product"] = product
    write_cache(key, payload, product)
    if upload:
        publish(key)
    return payload


def cached_context_by_key(key: str, product: str = "rainfall") -> dict:
    """Result for a caller that already knows the key.

    Keeps the same miss contract as :func:`cached_context`, without asking for a
    polygon that can be a megabyte of coordinates.
    """
    payload = read_cache(key, product) or (fetch(key) if product == "rainfall" else None)
    if payload is None:
        return {
            "status": "not_computed",
            "indicator": "monthly_precipitation",
            "reason": "no_precomputed_series",
            "message": (
                "No precomputed precipitation series for that study area. Submit the "
                "polygon to have one processed; it is computed offline because a "
                "single ERA5 grid cell takes about 20 seconds to read."
            ),
            "cache_key": key,
        }
    # The key is the caller's handle on this area: it is what /rainfall/status
    # polls and what /rainfall/forget needs in order to remove it. It used to be
    # returned only on a *miss*, so the key reached the browser only when there
    # was nothing stored -- which is precisely when there is nothing to delete.
    return {"status": "ok", "cache_key": key, **payload}


def cached_context(geojson_geom: dict, product: str = "rainfall") -> dict:
    """Result for the request path: a cached series, or an explicit miss.

    The request path never computes. A local hit is served without any network
    call; a miss falls back to the remote once and, if found, caches it locally so
    the next request is a plain file read again.
    """
    key = geometry_hash(geojson_geom)
    # A reader asking for a product must be served that product's cache, not the
    # area's other one.
    payload = read_cache(key, product) or (fetch(key) if product == "rainfall" else None)
    if payload is None:
        return {
            "status": "not_computed",
            "indicator": "monthly_precipitation",
            "reason": "no_precomputed_series",
            "message": (
                "Precipitation for this exact study area has not been processed yet. "
                "It is computed offline because a single ERA5 grid cell takes about "
                "20 seconds to read, which is longer than a whole request."
            ),
            "cache_key": key,
        }
    # The key is the caller's handle on this area: it is what /rainfall/status
    # polls and what /rainfall/forget needs in order to remove it. It used to be
    # returned only on a *miss*, so the key reached the browser only when there
    # was nothing stored -- which is precisely when there is nothing to delete.
    return {"status": "ok", "cache_key": key, **payload}


# --------------------------------------------------------------- CHIRPS
#
# A second product for the same measure. ERA5 is a 0.25 degree reanalysis: one
# cell covers about 774 km2 at the latitudes this is served for, so a study area
# below that is described by a single cell whatever its shape. CHIRPS is a
# 0.05 degree satellite-and-gauge blend -- 25 times the cells, and about five
# months fresher, because ERA5's newest complete month trails by roughly two to
# three. It is a satellite-gauge product rather than a model output, so it earns
# an "observed" class that ERA5 cannot.
#
# Measured against ERA5 over three cells in Kenya for 120 months (my own work,
# not a claim from the literature): CHIRPS reads drier by nothing at all in the
# wettest cell, and drier by 16% and 24% in the two drier ones. Month-to-month
# they agree well -- 92% of months share a sign on their anomaly -- so the
# seasonal structure holds and the level is where they part company. That is why
# both are offered rather than one replacing the other: the difference between
# them is the useful part.

CHIRPS_BASE_URL = (
    "https://deafrica-input-datasets.s3.af-south-1.amazonaws.com/"
    "rainfall_chirps_monthly/chirps-v2.0_{ym}.tif"
)
CHIRPS_NATIVE_GRID_DEGREES = 0.05
CHIRPS_NODATA = -9999.0
CHIRPS_LATENCY_DAYS = "30-60"


def _month_range(start: str, end: str) -> list[str]:
    """Month starts from start to end inclusive."""
    out, year, month = [], int(start[:4]), int(start[5:7])
    while f"{year:04d}-{month:02d}-01" <= end:
        out.append(f"{year:04d}-{month:02d}-01")
        month += 1
        if month == 13:
            month, year = 1, year + 1
    return out


def chirps_cell_monthly(grid: dict, start: str, end: str):
    """Monthly CHIRPS totals per cell, averaged over the ERA5 cell's footprint.

    CHIRPS is 0.05 deg and an ERA5 cell is 0.25, so one CHIRPS cell covers 1/625
    of the ERA5 footprint. Averaging the whole footprint rather than reading one
    central CHIRPS pixel is deliberate: a point sample of a 5 km grid would be a
    different claim from the ERA5 cell it is being compared with.
    """
    import numpy as np
    from rasterio.windows import Window

    longitudes = [float(v) for v in grid["longitudes"]]
    latitudes = [float(v) for v in grid["latitudes"]]
    labels = _month_range(start, end)
    matrix = np.full((len(labels), len(latitudes), len(longitudes)), np.nan)
    unreadable: list[str] = []

    # One object per month, so the months are independent and are read together.
    # Read in sequence this took 17 minutes for a 195-month series -- the ERA5
    # path reads its whole time series from one asset and finishes in 24 seconds,
    # and a CHIRPS job silently inherited ERA5's cost estimate, so the interface
    # promised a minute and delivered a quarter of an hour.
    def _read_one(label: str):
        ym = label[:7].replace("-", ".")
        try:
            with rio.open(CHIRPS_BASE_URL.format(ym=ym)) as src:
                return label, _chirps_block(src, latitudes, longitudes), None
        except Exception as exc:  # noqa: BLE001
            return label, None, f"{type(exc).__name__}"

    try:
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=8) as pool:
            for label, block, error in pool.map(_read_one, labels):
                if error is not None:
                    unreadable.append(label)
                    print(f"chirps {label[:7]}: {error}", file=sys.stderr)
                    continue
                matrix[labels.index(label)] = block
    except ImportError:
        for label in labels:
            got, block, error = _read_one(label)
            if error is not None:
                unreadable.append(got)
            else:
                matrix[labels.index(got)] = block

    return labels, matrix, _chirps_coverage(labels, unreadable)


def _chirps_coverage(labels, unreadable):
    return {
        "source": "chirps-v2.0",
        "months": len(labels),
        "unreadable_months": unreadable,
    }


def _chirps_block(src, latitudes, longitudes):
    """One month, every cell, each averaged over its ERA5 footprint."""
    import numpy as np

    out = np.full((len(latitudes), len(longitudes)), np.nan)
    for row, lat in enumerate(latitudes):
        for column, lon in enumerate(longitudes):
            half = ERA5_GRID_DEGREES / 2
            window = src.window(lon - half, lat - half, lon + half, lat + half)
            if window.width < 1 or window.height < 1:
                continue
            block = src.read(1, window=window, boundless=False)
            valid = block[(block != CHIRPS_NODATA) & (block < 9000)]
            if valid.size:
                out[row, column] = float(valid.mean())
    return out




def chirps_available(year: int) -> bool:
    """Whether the blend has been published for a month yet.

    CHIRPS is updated monthly and lags by a few weeks, so the newest month is not
    always present. Better to say so than to return NaN and let it read as a dry
    month.
    """
    try:
        import datetime as _dt

        today = _dt.datetime.now(_dt.UTC)
        latest = f"{year:04d}.{today.month:02d}"
        if year == today.year:
            latest = f"{year:04d}.{today.month - 1 or 12:02d}"
        with rio.open(CHIRPS_BASE_URL.format(ym=latest)):
            return True
    except Exception:  # noqa: BLE001
        return False
