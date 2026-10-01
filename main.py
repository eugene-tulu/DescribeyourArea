# --------------------------------------------------
# IMPORTS
# --------------------------------------------------
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional
import rasterio as rio
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.transform import from_bounds as transform_from_bounds
from rasterio.warp import reproject
from rasterio.windows import Window, transform as window_transform
import numpy as np
import planetary_computer
import datetime
from collections import Counter
import pystac_client
import asyncio
import json
from fastapi.middleware.cors import CORSMiddleware
import functools
import ipaddress
import math
import re
import secrets
import os
import jobs
import registry
import sensors
from dotenv import load_dotenv
import sys
import warnings
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union
from affine import Affine
from pyproj import CRS, Geod, Transformer

from contract import ContextResponse  # noqa: F401
from sensors import (  # noqa: F401 - NDVI_MIN_PLAUSIBLE and SCL_REJECTED are re-exported
    DEFAULT_SENSOR,
    cloud_mask,
    plausible,
    MODIS,
    MODIS_MIN_AREA_KM2,
    NDVI_MIN_PLAUSIBLE,
    SCL_REJECTED,
    SENSORS,
    Sensor,
    get_sensor,
    select_sensor,
)
import requests

import usage


# Addresses that can be a trusted proxy in front of this service. Enumerated
# rather than derived from ipaddress.is_private, which also reports the
# documentation ranges (192.0.2.0/24, 198.51.100.0/24, 203.0.113.0/24) as private
# and would make a documentation address look like a proxy.
_PROXY_NETWORKS = tuple(
    ipaddress.ip_network(cidr) for cidr in (
        "127.0.0.0/8", "10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "169.254.0.0/16",
        "::1/128", "fc00::/7", "fe80::/10",
    )
)


def _is_trusted_proxy(address: Optional[str]) -> bool:
    """True when an address can only be our own loopback-bound Nginx."""
    if not address or address in {"localhost", "backend", "gateway"}:
        return True if address in {"localhost", "backend", "gateway"} else False
    try:
        parsed = ipaddress.ip_address(address)
    except ValueError:
        return False
    return any(parsed in network for network in _PROXY_NETWORKS)


# --------------------------------------------------
# ENVIRONMENT
# --------------------------------------------------
# Loading .env is right in production and wrong under test: a developer's own
# bucket and keys would silently enter the test process, so an assertion about a
# cache miss could be satisfied by a live remote, and a run would touch real
# storage. The test package sets this before importing the app.
if os.getenv("GEOCONTEXT_NO_DOTENV", "") != "1":
    load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    """Read a boolean environment variable, treating "unset" and "false" clearly."""
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, *, minimum: int = 1) -> int:
    """Read a positive integer setting without making a bad env value fatal."""
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _env_float(name: str, default: float, *, minimum: float = 0.0) -> float:
    """Read a bounded float setting without making a bad env value fatal."""
    try:
        return max(minimum, float(os.getenv(name, str(default))))
    except ValueError:
        return default


# These caps keep synchronous browser requests predictable. A durable worker and
# queue are required before offering larger, asynchronous study areas.
#
# Derivation (measured against Planetary Computer, 2026-09-26, one vCPU box, peak
# RSS delta per request for a fresh process at the stated bounding-box area):
#
#   nasadem 30 m        10 km2 17 MB   100 km2 18 MB   1,000 km2 31 MB   5,500 km2 69 MB
#   worldcover 10 m     10 km2 21 MB   100 km2 42 MB   1,000 km2 235 MB  5,500 km2 1,205 MB
#   sentinel-2 20 m     2 km2 402 MB   100 km2 453 MB  (400 km2 exceeds the 75 s budget)
#
# Only worldcover has a genuinely area-driven memory curve, and NDVI is bounded by
# time rather than area: its 372 MB floor is dask/odc-stac framework overhead, not
# pixels, so 100 km2 costs 16% more than 2 km2. The single synchronous cap is
# therefore set at the largest area where all three modules finish comfortably
# inside the raster (30 s) and NDVI (75 s) budgets rather than at an arbitrary
# round number.
# ---------------------------------------------------------------------------
# Resource limits. Each of these guards a specific failure with a measured cost;
# nothing here is a style preference. A limit is a claim that exceeding it is
# worse than refusing the request.
# ---------------------------------------------------------------------------

# Payload size, on every path. This is the only limit that defends against a
# hostile body, and the only one that bounds parse memory: the largest legitimate
# study area measured is a 1,358 KB NRT conservancy, and 4 MB leaves headroom
# while still being trivially cheap to parse. Nginx's client_max_body_size is set
# to the same value, because a proxy that admits less than the application
# accepts just moves the rejection somewhere less explicable.
MAX_GEOJSON_BYTES = _env_int("MAX_GEOJSON_BYTES", 4_000_000)

# Vertices, on paths that actually iterate them. A 61,035-vertex conservancy
# costs real time to mask and clip per request. A cache lookup parses and hashes
# without iterating, so it is exempt: pass max_vertices=None.
MAX_AOI_VERTICES = _env_int("MAX_AOI_VERTICES", 10_000)

# Area budgets, in square kilometres of *bounding box* rather than polygon area,
# so an irregular outline is penalised by its own rectangle. Derived from
# measurement; see README "Measured limits" and the reasoning above each value.
#
# One synchronous cap covers DEM and vegetation, and WorldCover gets a larger one
# because it is the only product whose memory genuinely grows with area.
#
# This used to read "DEM is flat to 5,500 km2 (69 MB) ... so one synchronous cap
# covers both", which argued for a 5,500 km2 cap while the line beneath it set
# 100. The measurement is right and the inference was not: the cap is a
# *whole-request* budget set by the most expensive product in the request, not a
# per-product memory claim. DEM could be read at 5,500 km2 today and is refused at
# 100 anyway, because the request may also have asked for land cover.
MAX_SYNC_BBOX_KM2 = _env_float("MAX_SYNC_BBOX_KM2", 100.0)
# 1,000 km2 measures 235 MB; 5,500 km2 measures 1,205 MB and would starve the
# vegetation path inside a 1.8 GB container.
MAX_LANDCOVER_BBOX_KM2 = _env_float("MAX_LANDCOVER_BBOX_KM2", 1000.0)

# A study area can straddle many source tiles; bound the fan-out so a pathological
# bounding box cannot issue an unbounded number of COG opens.
MAX_SOURCE_TILES = _env_int("MAX_SOURCE_TILES", 64)

# ---------------------------------------------------------------------------
# Timeouts. Not limits on input; they are how long a caller waits, and how long
# the concurrency guard is held once the work is decided.
# ---------------------------------------------------------------------------

# Neither memory nor CPU is the binding constraint; remote read latency is. The
# global guard is generous, while the vegetation path gets a second, tighter guard
# because it is the only one whose latency degrades. Aggregate throughput still
# improves with concurrency, so this trades user-facing latency for throughput
# rather than being free.
MAX_CONCURRENT_ANALYSES = _env_int("MAX_CONCURRENT_ANALYSES", 8)
MAX_CONCURRENT_NDVI = _env_int("MAX_CONCURRENT_NDVI", 3)
# A caller that cannot enter the global guard is told the service is busy. The
# vegetation guard is proportionally slower, so waiting longer is reasonable
# there; on exhaustion the rest of the analysis still returns with vegetation
# marked unavailable.
ANALYSIS_ACQUIRE_SECONDS = _env_float("ANALYSIS_ACQUIRE_SECONDS", 2.0, minimum=0.0)
NDVI_ACQUIRE_SECONDS = _env_float("NDVI_ACQUIRE_SECONDS", 20.0, minimum=0.0)
# How long a finishing request waits for a timed-out thread before releasing the
# guard. Beyond this the guard is released regardless, because holding it
# indefinitely would wedge the service.
ANALYSIS_DRAIN_SECONDS = _env_float("ANALYSIS_DRAIN_SECONDS", 20.0, minimum=0.0)
# ---------------------------------------------------------------------------
# Product defaults. These are choices, not guards: exceeding them would not break
# anything, it would only make the result worse. They are here rather than
# scattered so it is clear which numbers are limits and which are knobs.
# ---------------------------------------------------------------------------

# Scenes averaged into a median composite. Four is a quality choice balancing a
# longer window against cloud; the cost of more is latency, not failure.
MAX_PC_SCENES = _env_int("MAX_PC_SCENES", 4)

# Enforced timeouts, named so /version publishes the same numbers the request
# path uses. They were inline literals at their three call sites, which meant
# the only way to publish them honestly was to publish nothing.
NDVI_TIMEOUT_SECONDS = _env_float("NDVI_TIMEOUT_SECONDS", 75.0, minimum=0.0)
RASTER_TIMEOUT_SECONDS = _env_float("RASTER_TIMEOUT_SECONDS", 30.0, minimum=0.0)
ASSET_SEARCH_SECONDS = _env_float("ASSET_SEARCH_SECONDS", 15.0, minimum=0.0)
# Grid CRS for the vegetation composite. EPSG:6933 is equal-area in metres, so a
# pixel is the same area everywhere. Valid to about 86 degrees latitude, so polar
# areas fall back to a local UTM zone.
NDVI_TARGET_EPSG = _env_int("NDVI_TARGET_EPSG", 6933)
# Best-effort in-process execution of a submitted rainfall job. The job record is
# durable regardless, so setting this to 0 and relying on the runner is a
# supported deployment, not a degraded one.
RAINFALL_AUTORUN = _env_bool("RAINFALL_AUTORUN", True)
_RAINFALL_TASKS: set = set()

ANALYSIS_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_ANALYSES)
NDVI_SEMAPHORE = asyncio.Semaphore(MAX_CONCURRENT_NDVI)
WGS84_GEOD = Geod(ellps="WGS84")


class _InFlightWork:
    """Count background thread work so the concurrency guard outlives timeouts.

    ``asyncio.wait_for`` cancels the *await*, not the thread, so releasing the
    semaphore on the way out would let a timed-out analysis keep occupying a
    thread, memory, and bandwidth while a new request starts.
    """

    def __init__(self) -> None:
        self._count = 0
        self._idle = asyncio.Event()
        self._idle.set()

    def enter(self) -> None:
        self._count += 1
        self._idle.clear()

    def exit(self) -> None:
        self._count = max(0, self._count - 1)
        if self._count == 0:
            self._idle.set()

    @property
    def count(self) -> int:
        return self._count

    async def drain(self, timeout: float) -> None:
        """Best-effort wait for outstanding work so the guard covers its runtime."""
        if self._count == 0:
            return
        try:
            await asyncio.wait_for(self._idle.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass


INFLIGHT = _InFlightWork()


async def run_blocking(func, *args):
    """Run ``func`` in a thread while keeping it counted past a cancelled await."""
    task = asyncio.ensure_future(
        asyncio.get_running_loop().run_in_executor(None, functools.partial(func, *args))
    )
    INFLIGHT.enter()
    task.add_done_callback(lambda _: INFLIGHT.exit())
    return await asyncio.shield(task)

# --------------------------------------------------
# APP CONFIGURATION
# --------------------------------------------------
app = FastAPI(title="GeoContext Generator API")

# Explicit origins are required because the API permits credentialed requests.
# Production uses same-origin /api behind Nginx, but this also supports local dev.
origins = [
    origin.strip()
    for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CRITICAL FIX: Strip whitespace from STAC URL
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1    ".strip()

# Single source of truth: /health and /version previously each hard-coded this
# and had already drifted apart (1.2.0 vs 1.3.0).
APP_VERSION = "1.20.0"


@app.middleware("http")
async def limit_analysis_concurrency(request: Request, call_next):
    """Bound costly STAC/raster work before it reaches shared public services."""
    if request.url.path != "/generate-context":
        return await call_next(request)

    try:
        await asyncio.wait_for(ANALYSIS_SEMAPHORE.acquire(), timeout=ANALYSIS_ACQUIRE_SECONDS)
    except asyncio.TimeoutError:
        return JSONResponse(
            status_code=429,
            content={"detail": "Analysis capacity is busy. Please retry shortly."},
        )

    try:
        return await call_next(request)
    finally:
        # A timed-out analysis leaves its thread running; hold the guard until that
        # work actually drains so the next request does not stack on top of it.
        await INFLIGHT.drain(timeout=ANALYSIS_DRAIN_SECONDS)
        ANALYSIS_SEMAPHORE.release()

# --------------------------------------------------
# SCHEMAS
# --------------------------------------------------
class GeoJSONRequest(BaseModel):
    geojson: dict

class RainfallLookupRequest(BaseModel):
    """Identify a study area, by geometry or by the key it hashes to."""
    geojson: Optional[Dict[str, Any]] = None
    cache_key: Optional[str] = None
    indicator: str = "rainfall"


class SubmitRequest(RainfallLookupRequest):
    """A submission may name which indicator it wants computed."""
    indicator: str = "rainfall"


# The published response contract lives in contract.py. It used to be
# Dict[str, Any], which made /openapi.json say nothing useful and left the
# frontend's hand-written interfaces with nothing keeping them in step.

# --------------------------------------------------
# LANDCOVER LOOKUP
# --------------------------------------------------
ESA_WORLDCOVER_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up areas",
    60: "Bare or sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetlands",
    95: "Mangroves",
    100: "Moss and lichen",
}

def label_landcover(percentages: Dict[str, float]) -> Dict[str, float]:
    return {
        ESA_WORLDCOVER_CLASSES.get(int(code), f"Unknown ({code})"): pct
        for code, pct in percentages.items()
    }

# --------------------------------------------------
# AOI ADMISSION AND GEOMETRY HELPERS
# --------------------------------------------------
def _position_count(coordinates: Any) -> int:
    """Count GeoJSON positions without assuming Polygon nesting depth."""
    if not isinstance(coordinates, (list, tuple)):
        return 0
    if coordinates and all(isinstance(value, (int, float)) for value in coordinates):
        return 1
    return sum(_position_count(value) for value in coordinates)


def _polygon_geometry(value: Any) -> tuple:
    """Return ``(geometry, repair_reason)``, or raise a client error.

    ``repair_reason`` is None when the geometry was already valid. It is returned
    rather than recorded globally so the caller can report it against this request
    only, with no shared state to leak between them.
    """
    if not isinstance(value, dict):
        raise HTTPException(status_code=400, detail="Each GeoJSON feature must be an object")

    geometry = value.get("geometry") if value.get("type") == "Feature" else value
    if not isinstance(geometry, dict):
        raise HTTPException(status_code=400, detail="GeoJSON feature is missing a geometry")
    if geometry.get("type") not in {"Polygon", "MultiPolygon"}:
        raise HTTPException(
            status_code=400,
            detail="Only Polygon and MultiPolygon study areas are supported",
        )

    try:
        geom = shape(geometry)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid GeoJSON geometry") from exc

    if geom.is_empty:
        raise HTTPException(status_code=400, detail="Study-area geometry is empty or invalid")
    reason = None
    if not geom.is_valid:
        # A ring self-intersection is a defect in someone else's file, not a reason
        # to refuse a whole conservancy. The repair is reported rather than silent.
        repaired, reason = repair_geometry(geometry)
        if reason is None:
            raise HTTPException(status_code=400, detail="Study-area geometry is empty or invalid")
        try:
            geom = shape(repaired)
        except Exception as exc:
            raise HTTPException(status_code=400, detail="Study-area geometry is empty or invalid") from exc
        if geom.is_empty or not geom.is_valid:
            raise HTTPException(status_code=400, detail="Study-area geometry is empty or invalid")
    return geom, reason


def repair_geometry(geometry: dict) -> tuple[dict, Optional[str]]:
    """Repair a self-intersecting polygon, reporting whether it was needed.

    Real-world conservation boundaries carry ring self-intersections often enough
    to matter: two of the 21 published NRT conservancies have them, and refusing
    those outright loses a whole area because of a defect in someone else's file.
    The repair is reported rather than silent, and lives in one place so the
    precompute path and the lookup path produce the same hash.
    """
    from shapely.validation import make_valid

    try:
        candidate = shape(geometry)
    except Exception:
        return geometry, None
    if candidate.is_valid and not candidate.is_empty:
        return geometry, None
    try:
        fixed = make_valid(candidate)
    except Exception:
        return geometry, None
    fixed = fixed if fixed.geom_type in {"Polygon", "MultiPolygon"} else (
        largest_polygon(fixed) if fixed.geom_type == "GeometryCollection" else None
    )
    if fixed is None or fixed.is_empty or fixed.area <= 0:
        return geometry, None
    # A self-intersecting ring has no meaningful signed area, so the comparison
    # below is skipped for exactly the bow-tie case it was written for. That is
    # deliberate: the repair is the only way such a geometry becomes usable, and a
    # genuine geometry problem will fail the is_valid check on the result instead.
    original_area = candidate.area
    if original_area > 0 and not 0.5 <= fixed.area / original_area <= 1.5:
        return geometry, None  # too different to be a self-intersection artefact
    reason = "self-intersecting rings repaired"
    return mapping(fixed), reason


def largest_polygon(collection):
    polygons = [g for g in collection.geoms if g.geom_type in {"Polygon", "MultiPolygon"}]
    if not polygons:
        return None
    return max(polygons, key=lambda g: g.area)


def canonicalize_geojson(
    geojson: dict,
    *,
    max_bytes: int = MAX_GEOJSON_BYTES,
    max_vertices: Optional[int] = MAX_AOI_VERTICES,
) -> dict:
    """Canonicalize supported GeoJSON into one Feature for every raster call.

    FeatureCollections are unioned rather than silently discarding all but the
    first polygon. This also lets uploaded MultiPolygons follow the same path
    as shapes drawn in the browser.
    """
    if not isinstance(geojson, dict):
        raise HTTPException(status_code=400, detail="GeoJSON must be an object")

    try:
        payload_size = len(json.dumps(geojson, separators=(",", ":")).encode("utf-8"))
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="GeoJSON is not JSON serializable") from exc
    if payload_size > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Study-area payload exceeds the {max_bytes} byte limit",
        )

    input_type = geojson.get("type")
    if input_type == "FeatureCollection":
        features = geojson.get("features")
        if not isinstance(features, list) or not features:
            raise HTTPException(status_code=400, detail="FeatureCollection must contain polygons")
        extracted = [_polygon_geometry(feature) for feature in features]
        properties: dict[str, Any] = {}
    elif input_type in {"Feature", "Polygon", "MultiPolygon"}:
        extracted = [_polygon_geometry(geojson)]
        properties = geojson.get("properties", {}) if input_type == "Feature" else {}
        if not isinstance(properties, dict):
            properties = {}
    else:
        raise HTTPException(
            status_code=400,
            detail="GeoJSON must be a Feature, FeatureCollection, Polygon, or MultiPolygon",
        )

    geometries = [geom for geom, _reason in extracted]
    repairs = sorted({reason for _geom, reason in extracted if reason})

    if max_vertices is not None:
        vertex_count = sum(
            _position_count(mapping(geom).get("coordinates")) for geom in geometries
        )
    if max_vertices is not None and vertex_count > max_vertices:
        raise HTTPException(
            status_code=413,
            detail=f"Study area has too many vertices (limit: {max_vertices})",
        )

    merged = unary_union(geometries)
    if merged.is_empty or merged.geom_type not in {"Polygon", "MultiPolygon"} or not merged.is_valid:
        raise HTTPException(status_code=400, detail="Study-area polygons cannot be combined safely")

    if repairs:
        # Reported rather than silent: a caller should know the boundary was fixed.
        properties = {**properties, "geometry_repaired": "; ".join(repairs)}

    return {
        "type": "Feature",
        "properties": properties,
        "geometry": mapping(merged),
    }


def aoi_bbox(feature: dict) -> list[float]:
    """Derive bounds from canonical geometry; handles Polygon and MultiPolygon."""
    geom, _reason = _polygon_geometry(feature)
    minx, miny, maxx, maxy = geom.bounds
    if minx < -180 or maxx > 180 or miny < -90 or maxy > 90 or minx >= maxx or miny >= maxy:
        raise HTTPException(status_code=400, detail="Study-area coordinates must be valid WGS84 longitude/latitude")
    return [float(minx), float(miny), float(maxx), float(maxy)]


def _bbox_area_km2(bbox: list[float]) -> float:
    """Geodesic area of the bounding box used by STAC and raster reads."""
    minx, miny, maxx, maxy = bbox
    area_m2, _ = WGS84_GEOD.geometry_area_perimeter(box(minx, miny, maxx, maxy))
    return abs(area_m2) / 1_000_000


def validate_for_lookup(geojson: dict) -> dict:
    """Admission for a cached lookup rather than a raster analysis.

    Keeps the payload and geometry checks, drops the area cap entirely, and
    allows a far higher vertex count, because nothing here reads pixels. Without
    this the precomputed portfolio is unreachable for the areas it was built for:
    the largest conservancy has a 5,510 km2 bounding box and tens of thousands of
    vertices.
    """
    # One payload cap, inherited. The vertex cap is waived because a lookup parses
    # and hashes without ever iterating vertices, so a detailed boundary costs it
    # nothing; a second cap pair here only created a second thing to keep in sync.
    feature = canonicalize_geojson(geojson, max_bytes=MAX_GEOJSON_BYTES, max_vertices=None)
    bbox = aoi_bbox(feature)
    return {"feature": feature, "bbox": bbox, "bbox_area_km2": _bbox_area_km2(bbox)}


def validate_aoi(
    geojson: dict,
    *,
    max_bytes: int = MAX_GEOJSON_BYTES,
    max_vertices: int = MAX_AOI_VERTICES,
    max_bbox_km2: float = MAX_SYNC_BBOX_KM2,
) -> dict:
    """Apply payload, vertex, and synchronous bounding-box admission limits."""
    feature = canonicalize_geojson(
        geojson,
        max_bytes=max_bytes,
        max_vertices=max_vertices,
    )
    bbox = aoi_bbox(feature)
    bbox_area_km2 = _bbox_area_km2(bbox)
    if bbox_area_km2 > max_bbox_km2:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Study-area bounding box is {bbox_area_km2:.1f} km²; "
                f"the synchronous limit is {max_bbox_km2:g} km²"
            ),
        )
    return {
        "feature": feature,
        "bbox": bbox,
        "bbox_area_km2": bbox_area_km2,
    }


def normalize_geojson(geojson: dict) -> dict:
    """Backward-compatible name for callers that only need the canonical feature."""
    return canonicalize_geojson(geojson)

def _window_for_bbox(src, bbox_4326: list[float]):
    """Return the read window of ``src`` covering a WGS84 bounding box.

    ``from_bounds`` interprets its arguments in the raster's own CRS, so a
    geographic bbox has to be reprojected first. NASADEM and WorldCover are
    published in EPSG:4326, which hid this; Sentinel-2 tiles are UTM.
    """
    from rasterio.windows import from_bounds

    source_crs = src.crs
    if source_crs is None or source_crs.to_epsg() == 4326:
        target = bbox_4326
    else:
        transformer = Transformer.from_crs(CRS.from_epsg(4326), source_crs, always_xy=True)
        xs, ys = transformer.transform(
            [bbox_4326[0], bbox_4326[0], bbox_4326[2], bbox_4326[2]],
            [bbox_4326[1], bbox_4326[3], bbox_4326[1], bbox_4326[3]],
        )
        target = [min(xs), min(ys), max(xs), max(ys)]

    window = from_bounds(*target, transform=src.transform).intersection(
        rio.windows.Window(0, 0, src.width, src.height)
    )
    if window.width < 1 or window.height < 1:
        return None
    # Whole-pixel windows are required for GDAL to serve a request from a COG
    # overview instead of the full-resolution blocks, and they are the single
    # biggest lever on read latency for remote imagery.
    aligned = window.round_offsets()
    return Window(aligned.col_off, aligned.row_off,
                max(1, int(round(window.width))), max(1, int(round(window.height))))


def _iter_tile_windows(src, bbox: list[float]):
    """Deprecated shim kept for the EPSG:4326 raster path."""
    from rasterio.windows import from_bounds

    window = from_bounds(*bbox, transform=src.transform).intersection(
        rio.windows.Window(0, 0, src.width, src.height)
    )
    if window.width < 1 or window.height < 1:
        return None
    return window


def _iter_masked_tiles(asset_hrefs: list[str], geojson: dict):
    """Yield masked, in-AOI arrays for every source tile, skipping empty overlaps.

    A study area can straddle many source tiles, so the statistics are accumulated
    per tile. Mosaicking is unnecessary for a zonal mean and would multiply peak
    memory by the number of tiles. ``crop=True`` already restricts the read to the
    part of each tile inside the AOI; the explicit window check only avoids
    opening tiles that cannot contribute.
    """
    geometry = geojson["geometry"]
    bbox = _feature_bbox(geojson)
    for href in asset_hrefs:
        with rio.open(planetary_computer.sign(href)) as src:
            if _iter_tile_windows(src, bbox) is None:
                continue
            clipped, _ = mask(src, [geometry], crop=True, nodata=src.nodata)
            if clipped.size == 0:
                continue
            yield src, clipped


def _feature_bbox(geojson: dict) -> list[float]:
    minx, miny, maxx, maxy = shape(geojson["geometry"]).bounds
    return [float(minx), float(miny), float(maxx), float(maxy)]


def compute_raster_stats(asset_hrefs: list[str], geojson: dict) -> Dict[str, float]:
    """Accumulate DEM statistics across every source tile covering the AOI."""
    total = 0.0
    total_sq = 0.0
    count = 0
    vmin: Optional[float] = None
    vmax: Optional[float] = None
    sampled = 0
    try:
        for src, clipped in _iter_masked_tiles(asset_hrefs, geojson):
            sampled += int(clipped[0].size)
            arr = clipped[0].astype("float64")
            valid = arr[arr != src.nodata] if src.nodata is not None else arr
            valid = valid[np.isfinite(valid)]
            if valid.size == 0:
                continue
            total += float(valid.sum())
            total_sq += float(np.square(valid).sum())
            count += int(valid.size)
            tile_min = float(valid.min())
            tile_max = float(valid.max())
            vmin = tile_min if vmin is None else min(vmin, tile_min)
            vmax = tile_max if vmax is None else max(vmax, tile_max)
    except Exception as exc:  # noqa: BLE001 - detail stays in the server log
        print(f"DEM computation error: {type(exc).__name__}: {str(exc)[:200]}", file=sys.stderr)
        return {"error": "elevation_unavailable"}

    if count == 0 or vmin is None or vmax is None:
        # Every sampled pixel was nodata: say so instead of reporting NaN, which
        # pydantic serialises as null and the client renders as a wrong terrain type.
        return {"error": "no_valid_elevation_pixels"}

    mean = total / count
    variance = max(0.0, (total_sq / count) - (mean * mean))
    return {
        "mean": mean,
        "min": vmin,
        "max": vmax,
        "std": math.sqrt(variance),
        "valid_pixel_count": count,
        "valid_pixel_fraction": round(count / sampled, 4) if sampled else 0.0,
    }


def interpret_terrain(dem: Dict[str, float]) -> Dict[str, Any]:
    if not dem or "error" in dem or "mean" not in dem:
        return dem

    # Guard every statistic: a single non-finite value used to fall through to the
    # "highly variable or mountainous" branch and mislabel the study area.
    values = {k: dem.get(k) for k in ("mean", "min", "max", "std")}
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in values.values()):
        return {**dem, "error": "no_valid_elevation_pixels"}

    elevation_range = dem["max"] - dem["min"]

    if elevation_range < 50:
        terrain = "relatively flat"
    elif elevation_range < 300:
        terrain = "moderately undulating"
    else:
        terrain = "highly variable or mountainous"

    return {
        **dem,
        "elevation_range_m": round(elevation_range, 1),
        "terrain_type": terrain,
    }


def compute_landcover_percentages(asset_hrefs: list[str], geojson: dict) -> Dict[str, Any]:
    """Accumulate land-cover class counts across every source tile."""
    counts: Counter = Counter()
    sampled = 0
    try:
        for src, clipped in _iter_masked_tiles(asset_hrefs, geojson):
            sampled += int(clipped[0].size)
            arr = clipped[0].astype(int)
            if src.nodata is not None:
                arr = arr[arr != src.nodata]
            if arr.size == 0:
                continue
            counts.update(arr.flatten().tolist())
    except Exception as exc:  # noqa: BLE001 - detail stays in the server log
        print(f"Landcover computation error: {type(exc).__name__}: {str(exc)[:200]}", file=sys.stderr)
        return {"error": "landcover_unavailable"}

    total = sum(counts.values())
    if total == 0:
        return {"error": "no_valid_landcover_pixels"}

    percentages = {str(k): round((v / total) * 100, 2) for k, v in counts.items()}
    labeled = label_landcover(percentages)
    dominant_class = max(labeled, key=labeled.get)

    return {
        "classes": labeled,
        "dominant_class": dominant_class,
        "dominant_percentage": labeled[dominant_class],
        "valid_pixel_count": total,
        "valid_pixel_fraction": round(total / sampled, 4) if sampled else 0.0,
    }

# --------------------------------------------------
# NDVI COMPUTATION: BOUNDED PLANETARY COMPUTER WORKFLOW
# --------------------------------------------------
# Grid CRS for the composite. EPSG:6933 (WGS 84 / NSIDC EASE-Grid 2.0 Global) is
# equal-area with metre units, so a pixel is the same area everywhere and a
# reported percentage means the same thing in Kenya as in Canada. One global CRS
# also avoids the UTM zone-edge problem, where a single conservancy straddling a
# zone boundary would need two grids stitched together. It is only valid to about
# 86 degrees latitude, so polar study areas fall back to a local UTM zone.
NDVI_TARGET_EPSG = _env_int("NDVI_TARGET_EPSG", 6933)
_EASE_GRID_MAX_LAT = 85.0


def _target_crs(bbox: list[float]):
    """Analysis CRS, falling back to a local UTM zone near the poles."""
    if abs(bbox[1]) <= _EASE_GRID_MAX_LAT and abs(bbox[3]) <= _EASE_GRID_MAX_LAT:
        return CRS.from_epsg(NDVI_TARGET_EPSG)
    centre_lon = (bbox[0] + bbox[2]) / 2.0
    centre_lat = (bbox[1] + bbox[3]) / 2.0
    zone = min(60, max(1, int((centre_lon + 180.0) / 6.0) + 1))
    return CRS.from_epsg((32700 if centre_lat < 0 else 32600) + zone)


def _vegetation_target_grid(bbox: list[float], resolution_m: int) -> tuple:
    """One projected output grid covering the AOI, in metres per pixel.

    Working in a projected CRS means the metres-per-pixel request is honoured
    exactly. A degree grid cannot do that, because a degree of longitude shrinks
    with latitude and silently halves the ground resolution away from the equator.
    """
    crs = _target_crs(bbox)
    transformer = Transformer.from_crs(CRS.from_epsg(4326), crs, always_xy=True)
    minx, miny = transformer.transform(bbox[0], bbox[1])
    maxx, maxy = transformer.transform(bbox[2], bbox[3])
    minx, maxx = min(minx, maxx), max(minx, maxx)
    miny, maxy = min(miny, maxy), max(miny, maxy)
    width = max(1, int(math.ceil((maxx - minx) / resolution_m)))
    height = max(1, int(math.ceil((maxy - miny) / resolution_m)))
    transform = transform_from_bounds(minx, miny, maxx, maxy, width, height)
    return crs, transform, width, height


def _read_window(src, window, target_res: Optional[float], resampling):
    """Read band 1 over a window at roughly ``target_res`` metres.

    Returns ``(values, transform)`` for the pixels actually read.

    Planetary Computer assets are single-band COGs with no band metadata, so the
    asset href already selects the band. Where a band is published finer than the
    requested resolution, reading a COG overview fetches a fraction of the bytes,
    but only if the *window* is expressed on the overview's own grid, which is why
    this is not just an ``out_shape`` argument.
    """
    win_transform = window_transform(window, src.transform)
    native = abs(src.transform.a) or 0.0
    if not target_res or native <= 0 or not window.width or not window.height:
        return src.read(1, window=window), win_transform

    factor = target_res / native
    if factor <= 1.05:
        # Source is already coarser than requested: read it as-is.
        return src.read(1, window=window), win_transform

    overviews = src.overviews(1) or []
    usable = [level for level in overviews if level <= factor * 1.5]
    if usable:
        level = min(usable)
        index = 2 + overviews.index(level)
        sub = Window(
            window.col_off / level, window.row_off / level,
            window.width / level, window.height / level,
        )
        try:
            values = src.read(index, window=sub)
            return values, win_transform * Affine.scale(level, level)
        except Exception:
            pass

    height = max(1, int(round(window.height / factor)))
    width = max(1, int(round(window.width / factor)))
    values = src.read(1, window=window, out_shape=(height, width), resampling=resampling)
    return values, win_transform * Affine.scale(
        window.width / width, window.height / height
    )


def _reproject_to_grid(values: np.ndarray, src_transform, src_crs, target, resampling, nodata):
    """Warp a source block onto the shared output grid.

    ``target`` is (crs, transform, width, height); numpy arrays are (rows, cols).
    """
    out = np.full((target[3], target[2]), nodata, dtype="float32")
    reproject(
        source=values,
        destination=out,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=target[1],
        dst_crs=target[0],
        resampling=resampling,
        src_nodata=None,
        # GDAL must not be told the destination no-data value is NaN: it then
        # treats every output pixel as no data and writes nothing at all. The
        # buffer is pre-filled instead, and left untouched where nothing lands.
        dst_nodata=None,
    )
    return out


def _read_asset(item, asset: str, bbox, resolution_m: int, resampling, grid_shape=None):
    """Open one asset and return ``(values, transform, crs)`` for the AOI window."""
    with rio.open(planetary_computer.sign(item.assets[asset].href)) as src:
        window = _window_for_bbox(src, bbox)
        if window is None:
            return None
        values, transform = _read_window(src, window, resolution_m, resampling)
        if values is None:
            return None
        if grid_shape is not None and values.shape != grid_shape:
            values = _resize_nearest(values, grid_shape)
        return values, transform, src.crs


def _scene_index_on_grid(item, sensor: Sensor, target, bbox, resolution_m: int):
    """Per-pixel NDVI for one scene on the shared output grid.

    A product sensor reads a finished index. A band sensor computes the ratio after
    applying the sensor's scale and offset, which for Landsat is the difference
    between a plausible number and a wrong one: the -0.2 reflectance offset does
    not cancel in the ratio, and skipping it moves NDVI by tens of percent.
    """
    try:
        if sensor.computes_ndvi:
            red = _read_asset(item, sensor.red, bbox, resolution_m, Resampling.average)
            if red is None:
                return None
            grid_shape = red[0].shape
            nir = _read_asset(item, sensor.nir, bbox, resolution_m,
                              Resampling.average, grid_shape)
            if nir is None:
                return None
            # Reflectance units before any arithmetic. Sentinel-2 L2A is a plain
            # uint16 ratio; Landsat needs the published scale and offset.
            red_f = red[0].astype("float32") * sensor.scale + sensor.offset
            nir_f = nir[0].astype("float32") * sensor.scale + sensor.offset
            with np.errstate(divide="ignore", invalid="ignore"):
                index = (nir_f - red_f) / (nir_f + red_f + 1e-8)
            index = np.where(np.isfinite(index), index, np.nan)
            index_transform, index_crs = red[1], red[2]
        else:
            product = _read_asset(item, sensor.ndvi_asset, bbox, resolution_m,
                                  Resampling.average)
            if product is None:
                return None
            index = product[0].astype("float32") * sensor.scale + sensor.offset
            index = np.where(np.isfinite(index), index, np.nan)
            index_transform, index_crs = product[1], product[2]
            grid_shape = index.shape

        on_grid = _reproject_to_grid(index, index_transform, index_crs, target,
                                     Resampling.bilinear, np.nan)

        if sensor.cloud:
            cloud = _read_asset(item, sensor.cloud, bbox, resolution_m,
                                Resampling.nearest, grid_shape)
            if cloud is not None:
                classes = _reproject_to_grid(
                    cloud[0].astype("float32"), cloud[1], cloud[2], target,
                    Resampling.nearest, -1.0,
                )
                on_grid[~cloud_mask(sensor, on_grid, classes)] = np.nan

        # Discard implausible values rather than trusting a class that flags bright
        # desert as cloud. A product sensor is already masked by its producer, but
        # the guard is cheap and the fill range is common.
        on_grid[~plausible(on_grid)] = np.nan
        return on_grid
    except Exception as exc:  # noqa: BLE001 - one bad scene must not fail the request
        print(f"vegetation scene skipped ({getattr(item, 'id', '?')}): "
              f"{type(exc).__name__}: {str(exc)[:120]}", file=sys.stderr)
        return None


async def _search_items(
    bbox: list[float],
    sensor: Sensor,
    max_scenes: int,
    window_start: str = "",
    window_end: str = "",
) -> list:
    """Fetch scenes for a sensor, retrying only transient failures."""
    if not window_start or not window_end:
        window_start, window_end = _resolve_window(None, None, 90)
    time_window = f"{window_start}/{window_end}"

    def search_once() -> list:
        catalog = pystac_client.Client.open(STAC_URL)
        # Candidates must exceed the number of scenes actually used, or the
        # cloud-cover sort below has nothing to choose from. Capping the pool at
        # ``max_scenes`` returned only the most recent items, so a request for a
        # 30-year window silently produced a composite of the last six weeks.
        pool = max(4 * max_scenes, 50)
        arguments: dict[str, Any] = {
            "collections": [sensor.collection],
            "bbox": bbox,
            "datetime": time_window,
            "limit": pool,
            "max_items": pool,
        }
        # A product sensor is already cloud-masked, so there is no reason to spend
        # the scene budget on a cloudy scene.
        if sensor.cloud_mask != "product":
            arguments["query"] = {"eo:cloud_cover": {"lt": 30}}
        return list(catalog.search(**arguments).items())[:pool]

    for attempt in range(3):
        try:
            items = search_once()
            if not items:
                return []
            # Scene-level cloud cover describes the whole tile, so a scene can score
            # near zero and still be fully overcast over the study area. Taking the
            # least cloudy first is the cheapest way to avoid a composite of cloud.
            if sensor.cloud_mask != "product":
                items.sort(key=lambda item: (item.properties or {}).get("eo:cloud_cover", 100.0))
            else:
                # MODIS publishes datetime=null and only start_datetime, so the
                # default ordering is by recency using the field that exists.
                items.sort(key=_item_start_datetime, reverse=True)
            # The wider pool is for *selection*. Only the best ``max_scenes`` are
            # composited, so widening the search does not widen the read.
            return items[:max_scenes]
        except Exception:
            if attempt == 2:
                print(f"STAC search failed after 3 attempts for {sensor.id}", file=sys.stderr)
                return []
            await asyncio.sleep(0.5 * (2 ** attempt))
    return []


def _item_start_datetime(item) -> float:
    """Sort key that works whether or not an item sets ``datetime``.

    MODIS items on Planetary Computer carry ``datetime: null`` and only populate
    ``start_datetime``, so sorting on ``datetime`` raises a TypeError on them.
    """
    for value in (getattr(item, "datetime", None), getattr(item, "start_datetime", None)):
        if value is not None:
            return value.timestamp()
    return 0.0


def _resize_nearest(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    rows = (np.arange(shape[0]) * arr.shape[0] // max(1, shape[0])).clip(0, arr.shape[0] - 1)
    cols = (np.arange(shape[1]) * arr.shape[1] // max(1, shape[1])).clip(0, arr.shape[1] - 1)
    return arr[np.ix_(rows, cols)]


def _load_and_summarize_ndvi(
    items: list,
    bbox: list[float],
    geojson_geom: dict,
    resolution_m: int,
    sensor: Sensor,
    sensor_reason: str = "",
    window: Optional[dict] = None,
) -> dict:
    """Build a per-pixel median NDVI composite and summarize it.

    Sensor-parameterised: a band sensor computes the index from red and NIR after
    applying the sensor's radiometric scale and offset, while a product sensor
    reads a finished index the producer has already cloud-masked.
    """
    target = _vegetation_target_grid(bbox, resolution_m)
    geometry = shape(geojson_geom)
    in_aoi = _geometry_mask_on_grid(geometry, target, bbox)

    scenes: list[np.ndarray] = []
    scene_ids: list[str] = []
    scene_dates: list[str] = []
    for item in items:
        grid = _scene_index_on_grid(item, sensor, target, bbox, resolution_m)
        if grid is None:
            continue
        grid = np.where(in_aoi, grid, np.nan)
        if not np.isfinite(grid).any():
            continue
        scenes.append(grid)
        scene_ids.append(item.id)
        stamp = _item_start_datetime(item)
        scene_dates.append(
            datetime.datetime.fromtimestamp(stamp, datetime.UTC).date().isoformat()
            if stamp else None
        )

    provenance = sensor.provenance()

    if not scenes:
        return {
            "status": "unavailable",
            "source": "Planetary Computer",
            "sensor": provenance,
            "sensor_reason": sensor_reason,
            "scenes_examined": len(items),
            "warning": (
                f"No usable {sensor.label} pixels were available for this study area. "
                "Every candidate scene was flagged as cloud, shadow or snow over the "
                "study area, so no vegetation value is reported rather than reporting cloud."
            ),
        }

    stack = np.stack(scenes)
    with warnings.catch_warnings():
        # Pixels with no valid observation in any scene are all-NaN by design.
        warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
        warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)
        median_ndvi = np.nanmedian(stack, axis=0)
    observed = np.isfinite(median_ndvi)
    values = median_ndvi[observed]
    if values.size == 0:
        return {
            "status": "unavailable",
            "source": "Planetary Computer",
            "sensor": provenance,
            "sensor_reason": sensor_reason,
            "scenes_examined": len(items),
            "warning": (
                f"Scene-quality masking left no valid {sensor.label} pixels."
            ),
        }

    covered = float(observed.sum()) / float(in_aoi.size)
    return {
        "status": "ok",
        "source": "Planetary Computer",
        "sensor": provenance,
        "sensor_reason": sensor_reason,
        "window": window or {},
        "mean": float(values.mean()),
        "min": float(values.min()),
        "max": float(values.max()),
        "std": float(values.std()),
        "p25": float(np.percentile(values, 25)),
        "p75": float(np.percentile(values, 75)),
        "scene_count": len(scenes),
        "scenes_examined": len(items),
        "resolution_m": resolution_m,
        "valid_pixel_count": int(values.size),
        "valid_pixel_fraction": round(covered, 4),
        "method": (
            f"{sensor.id}_median_composite"
            if sensor.computes_ndvi
            else f"{sensor.id}_product_composite"
        ),
        "scene_ids": scene_ids,
        "scene_dates": [d for d in scene_dates if d],
    }


def _geometry_mask_on_grid(geometry, target, bbox: list[float]) -> np.ndarray:
    """Rasterise the study-area polygon onto the target grid.

    ``target`` is (crs, transform, width, height): numpy wants ``out_shape`` as
    (rows, cols) while ``transform_from_bounds`` wants (width, height), so the two
    are supplied in opposite order on purpose.
    """
    from rasterio.features import geometry_mask
    from rasterio.warp import transform_geom

    _crs, transform, width, height = target
    # geometry_mask does not reproject: the study area arrives in WGS84 but the
    # grid is projected, so it has to be converted before rasterising.
    projected = transform_geom(CRS.from_epsg(4326), target[0], mapping(geometry))
    return ~geometry_mask(
        [projected],
        out_shape=(height, width),
        transform=transform,
        all_touched=True,
    )


def _target_bounds(target, bbox: list[float]) -> tuple[float, float, float, float]:
    crs, transform, width, height = target
    minx, maxy = transform * (0, 0)
    maxx, miny = transform * (width, height)
    return minx, miny, maxx, maxy



async def compute_vegetation_index(
    bbox: list[float],
    geojson_geom: dict,
    max_area_km2: float = MAX_SYNC_BBOX_KM2,
    max_scenes: int = MAX_PC_SCENES,
    resolution_m: int = 20,
    sensor_id: str = "auto",
    window_days: int = 90,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> dict:
    """Compute a bounded vegetation index from Planetary Computer only.

    The sensor is chosen by :func:`sensors.select_sensor` unless one is named.
    Nothing is ever silently downgraded: a study area past the cap, or a window
    the chosen sensor does not cover, produces an explicit reason.
    """
    bbox_area_km2 = _bbox_area_km2(bbox)
    # An explicit range is what makes a historical question answerable at all: a
    # lookback from today cannot reach 1997, which is the question the conservancy
    # audience actually asked.
    try:
        window_start, window_end = _resolve_window(start, end, window_days)
        sensor, reason = select_sensor(
            sensor_id, window_start=window_start, bbox_area_km2=bbox_area_km2
        )
    except ValueError as exc:
        return {
            "status": "unavailable",
            "source": "Planetary Computer",
            "warning": str(exc),
        }

    # MODIS is a 250 m product, so the cap that guards a 10 m or 30 m source does
    # not apply to it: the whole point is that it covers areas the others cannot.
    cap = max_area_km2 if sensor.id != "modis" else float("inf")
    if bbox_area_km2 > cap:
        return {
            "status": "skipped",
            "source": "Planetary Computer",
            "sensor": sensor.provenance(),
            "sensor_reason": reason,
            "bbox_area_km2": round(bbox_area_km2, 2),
            "warning": (
                f"A {sensor.label} index is available for bounding boxes up to "
                f"{max_area_km2:g} km²; use a smaller study area for this "
                "synchronous analysis."
            ),
        }

    if window_start < sensor.archive_start:
        return {
            "status": "skipped",
            "source": "Planetary Computer",
            "sensor": sensor.provenance(),
            "sensor_reason": reason,
            "window": {"start": window_start, "end": window_end},
            "warning": (
                f"{sensor.label} begins on {sensor.archive_start}, so the requested "
                f"window from {window_start} is not covered. Narrow the window or name a "
                "sensor whose archive reaches back that far."
            ),
        }

    bounded_scenes = min(max(sensor.default_max_scenes, MAX_PC_SCENES), max(1, int(max_scenes)))
    # This is the I/O-heavy, latency-sensitive path, so it has its own guard. On
    # exhaustion the caller still gets terrain and land cover with an explicit
    # reason, rather than a 429 for the whole request.
    try:
        await asyncio.wait_for(NDVI_SEMAPHORE.acquire(), timeout=NDVI_ACQUIRE_SECONDS)
    except asyncio.TimeoutError:
        return {
            "status": "unavailable",
            "source": "Planetary Computer",
            "sensor": sensor.provenance(),
            "warning": (
                "Vegetation analysis is busy right now. Terrain and land cover are "
                "unaffected; retry shortly for a vegetation value."
            ),
        }
    try:
        items = await _search_items(bbox, sensor, bounded_scenes, window_start, window_end)
        if not items:
            return {
                "status": "unavailable",
                "source": "Planetary Computer",
                "sensor": sensor.provenance(),
                "sensor_reason": reason,
                "sensor_reason": reason,
                "window": {"start": window_start, "end": window_end},
                "warning": (
                    f"No usable {sensor.label} scenes covered this study area between "
                    f"{window_start} and {window_end}."
                ),
            }
        try:
            return await asyncio.wait_for(
                run_blocking(
                    _load_and_summarize_ndvi, items, bbox, geojson_geom, resolution_m,
                    sensor, reason, {"start": window_start, "end": window_end},
                ),
                timeout=NDVI_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            return {
                "status": "unavailable",
                "source": "Planetary Computer",
                "sensor": sensor.provenance(),
                "warning": "Vegetation processing timed out; try a smaller study area.",
            }
        except Exception as exc:
            print(f"⚠️ vegetation processing failed ({sensor.id}): "
                  f"{type(exc).__name__}: {str(exc)[:120]}", file=sys.stderr)
            return {
                "status": "unavailable",
                "source": "Planetary Computer",
                "sensor": sensor.provenance(),
                "warning": "Vegetation could not be processed for this study area.",
            }
    finally:
        NDVI_SEMAPHORE.release()


def _resolve_window(start: Optional[str], end: Optional[str], window_days: int) -> tuple[str, str]:
    """Normalise an explicit date range, or fall back to a lookback from today.

    There is deliberately no upper bound on the span. A request takes at most
    ``MAX_PC_SCENES`` scenes however long the window is, so a decade costs the
    same as a month; a span limit set above any request a person would make was
    documentation pretending to be a control.
    """
    today = datetime.datetime.now(datetime.UTC).date()
    end_date = today if not end else _parse_date(end, "end")
    if start:
        start_date = _parse_date(start, "start")
    else:
        start_date = end_date - datetime.timedelta(days=max(1, window_days))
    if start_date > end_date:
        raise ValueError("window start must not be after the end")
    return start_date.isoformat(), end_date.isoformat()


def _parse_date(value: str, label: str) -> datetime.date:
    try:
        return datetime.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"window {label} must be an ISO date (YYYY-MM-DD)") from exc


async def compute_median_ndvi(
    bbox: list[float],
    geojson_geom: dict,
    max_area_km2: float = MAX_SYNC_BBOX_KM2,
    max_scenes: int = MAX_PC_SCENES,
    resolution_m: int = 20,
) -> dict:
    """Sentinel-2 composite, kept for callers that want that sensor specifically.

    The sensor is chosen automatically by :func:`compute_vegetation_index`.
    """
    return await compute_vegetation_index(
        bbox, geojson_geom,
        max_area_km2=max_area_km2,
        max_scenes=max_scenes,
        resolution_m=resolution_m,
        sensor_id="sentinel-2",
    )


# --------------------------------------------------
# COUNTRY CONTEXT
# --------------------------------------------------

async def get_country_from_centroid(geojson_geom: dict) -> Optional[str]:
    """Return country name using Nominatim."""
    try:
        from shapely.geometry import shape
        centroid = shape(geojson_geom).centroid
        lat, lon = centroid.y, centroid.x

        resp = await asyncio.to_thread(
            requests.get,
            "https://nominatim.openstreetmap.org/reverse",
            params={
                "format": "json",
                "lat": lat,
                "lon": lon,
                "zoom": 4,
                "addressdetails": 1,
            },
            headers={"User-Agent": "GeoContextualize/1.0"},
            timeout=10.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("address", {}).get("country")
    except Exception as e:
        print(f"⚠️ Country lookup failed: {e}", file=sys.stderr)
    return None

# --------------------------------------------------
# API ENDPOINT
# --------------------------------------------------
AVAILABLE_DATASETS = {
    "dem",
    "landcover",
    "ndvi",
    "rainfall",
}


async def _timed(timer, coroutine):
    """Await a coroutine, recording how long it took, even if it raised.

    The timer has to be entered here, not merely constructed: an unentered timer
    reported the interval since the epoch, which is four hours of nonsense.
    """
    timer.__enter__()
    try:
        return await coroutine
    finally:
        timer.__exit__(None, None, None)


def _client_host(http_request) -> Optional[str]:
    """The caller's address, not the proxy's.

    The Compose file binds the API to loopback and Nginx is the only thing that can
    reach it, so ``request.client.host`` is the proxy and is useless for rate
    limiting or for noticing scraping. The forwarded chain is therefore read, but
    only when the peer can only be our own Nginx, so a client cannot forge it by
    sending its own header. The leftmost non-private entry is the closest thing to
    the caller.

    This is safe only while the API stays loopback-bound. If the port is ever
    exposed, this must stop trusting the header.
    """
    try:
        client = getattr(http_request, "client", None)
        peer = client.host if client else None
    except Exception:  # noqa: BLE001
        return None
    if not peer:
        return None
    if not _is_trusted_proxy(peer):
        return peer
    try:
        forwarded = http_request.headers.get("x-forwarded-for", "") or ""
    except Exception:  # noqa: BLE001
        return None
    parts = [part.strip() for part in str(forwarded).split(",") if part.strip()]
    for candidate in parts:
        if not _is_trusted_proxy(candidate):
            return candidate
    return parts[0] if parts else peer


def _emit_usage_event(*, requested, outcomes, module_ms, request_timer, http_request,
                      area_km2, ndvi_stats) -> None:
    """Record what this request did, never what land it was about.

    The event builder takes no geometry and the area is reduced to a band here, so
    there is no path by which a submitted polygon could reach the log. Anything
    unexpected is swallowed: analytics must not be able to fail a request that has
    already succeeded.
    """
    try:
        if not usage.analytics_enabled(http_request):
            return
        request_timer.__exit__(None, None, None)
        sensor = None
        if isinstance(ndvi_stats, dict):
            sensor = (ndvi_stats.get("sensor") or {}).get("id")
        usage.emit(usage.build_event(
            datasets_requested=requested,
            outcomes=outcomes,
            duration_ms=module_ms,
            total_ms=getattr(request_timer, "elapsed_ms", None),
            area_km2=area_km2,
            client_address=_client_host(http_request),
            sensor=sensor,
        ))
    except Exception:  # noqa: BLE001
        pass


def _requested_datasets(value: Optional[str]) -> set[str]:
    """Parse the existing comma-separated frontend selector safely."""
    if value is None:
        # Derived, not hard-coded: a literal default drifted out of step with
        # AVAILABLE_DATASETS when rainfall was added.
        return set(AVAILABLE_DATASETS)
    requested = {name.strip().lower() for name in value.split(",") if name.strip()}
    if not requested:
        raise HTTPException(
            status_code=422,
            detail="At least one dataset must be selected",
        )
    unknown = requested - AVAILABLE_DATASETS
    if unknown:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported datasets: {', '.join(sorted(unknown))}",
        )
    return requested


async def _find_core_assets(
    bbox: list[float],
    *,
    need_dem: bool,
    need_landcover: bool,
) -> dict[str, list[str]]:
    """Collect every source tile that intersects the AOI; signing happens on read.

    Returning a single tile (limit=1) silently analysed 3% of a multi-tile
    bounding box. Zonal statistics are accumulated per tile instead.
    """
    def search() -> dict[str, list[str]]:
        catalog = pystac_client.Client.open(STAC_URL)
        assets: dict[str, list[str]] = {}
        for name, collection, asset, needed, label in (
            ("dem", "nasadem", "elevation", need_dem, "elevation"),
            ("landcover", "esa-worldcover", "map", need_landcover, "land-cover"),
        ):
            if not needed:
                continue
            items = list(
                catalog.search(
                    collections=[collection],
                    bbox=bbox,
                    limit=MAX_SOURCE_TILES,
                    max_items=MAX_SOURCE_TILES,
                ).items()
            )
            hrefs = [item.assets[asset].href for item in items if asset in item.assets]
            if not hrefs:
                raise HTTPException(
                    status_code=400,
                    detail=f"No {label} data available for this area",
                )
            assets[name] = hrefs
        return assets

    try:
        return await asyncio.to_thread(search)
    except HTTPException:
        raise
    except (KeyError, IndexError) as exc:
        raise HTTPException(status_code=502, detail="A required raster asset was unavailable") from exc


@app.post("/generate-context", response_model=ContextResponse, response_model_exclude_none=True)
async def generate_context(
    request: GeoJSONRequest,
    http_request: Request = None,
    include_ndvi: bool = True,
    datasets: Optional[str] = None,
    sensor: str = "auto",
    window_days: int = 90,
    window_start: Optional[str] = None,
    window_end: Optional[str] = None,
):
    # A malformed or inverted window is a client error, not a missing result.
    # Left to the vegetation module it degrades to "unavailable", which tells an
    # API caller the area has no data when the truth is the request was wrong.
    try:
        requested_window_start, requested_window_end = _resolve_window(
            window_start, window_end, window_days
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    request_timer = usage.Timer()
    request_timer.__enter__()
    module_ms: dict[str, int] = {}
    try:
        requested = _requested_datasets(datasets)
        aoi = validate_aoi(request.geojson)
        geojson = aoi["feature"]
        geom = geojson["geometry"]
        bbox = aoi["bbox"]

        core_assets = await asyncio.wait_for(
            _find_core_assets(
                bbox,
                need_dem="dem" in requested,
                need_landcover="landcover" in requested,
            ),
            timeout=ASSET_SEARCH_SECONDS,
        )

        raster_tasks: dict[str, Any] = {}
        landcover_over_limit: Optional[dict] = None
        if "dem" in core_assets:
            raster_tasks["dem"] = run_blocking(compute_raster_stats, core_assets["dem"], geojson)
        if "landcover" in core_assets:
            if aoi["bbox_area_km2"] > MAX_LANDCOVER_BBOX_KM2:
                # Land cover is the one module whose memory grows with area, so it
                # reports an explicit skip instead of risking the whole request.
                landcover_over_limit = {
                    # The contract's declared status, which nothing used to emit. A
            # caller switching on it got a silent miss and fell through to a
            # generic error, which is how "too large" and "it broke" looked alike.
            "status": "area_exceeded",
            "error": "landcover_area_exceeded",
                    "bbox_area_km2": round(aoi["bbox_area_km2"], 2),
                    "limit_km2": MAX_LANDCOVER_BBOX_KM2,
                }
            else:
                raster_tasks["landcover"] = run_blocking(
                    compute_landcover_percentages,
                    core_assets["landcover"],
                    geojson,
                )
        raster_timer = usage.Timer()
        raster_timer.__enter__()
        try:
            raster_values = await asyncio.wait_for(
                asyncio.gather(*raster_tasks.values()),
                timeout=RASTER_TIMEOUT_SECONDS,
            ) if raster_tasks else []
        except asyncio.TimeoutError:
            raster_timer.__exit__(None, None, None)
            raise HTTPException(status_code=504, detail="Raster processing timed out; try a smaller area")
        except MemoryError:
            raster_timer.__exit__(None, None, None)
            raise HTTPException(status_code=507, detail="Raster processing exceeded available memory")
        raster_timer.__exit__(None, None, None)
        if raster_timer.elapsed_ms is not None:
            for name in raster_tasks:
                module_ms[name] = raster_timer.elapsed_ms
        raster_results = dict(zip(raster_tasks.keys(), raster_values))
        dem = interpret_terrain(raster_results["dem"]) if "dem" in raster_results else None
        landcover = raster_results.get("landcover", landcover_over_limit)

        ndvi_stats = None
        if include_ndvi and "ndvi" in requested:
            vegetation_timer = usage.Timer()
            ndvi_stats = await _timed(vegetation_timer, compute_vegetation_index(
                bbox=bbox,
                geojson_geom=geom,
                max_area_km2=MAX_SYNC_BBOX_KM2,
                max_scenes=MAX_PC_SCENES,
                resolution_m=20,
                sensor_id=sensor,
                window_days=window_days,
                start=window_start,
                end=window_end,
            ))
            module_ms["ndvi"] = vegetation_timer.elapsed_ms

        # Rainfall is read from a precomputed cache. A single ERA5 grid cell takes
        # about 20 seconds to read, which is longer than a whole request, so the
        # request path never computes it. See rainfall.py.
        rainfall_context = None
        if "rainfall" in requested:
            import rainfall

            rainfall_context = rainfall.cached_context(geom)

        # Get country
        country = await get_country_from_centroid(geom)

        # Extract scene metadata for provenance
        scene_dates = {}
        scene_ids = {}
        if ndvi_stats:
            if "scene_dates" in ndvi_stats:
                scene_dates["ndvi_composite"] = ", ".join(ndvi_stats["scene_dates"])
            if "scene_ids" in ndvi_stats:
                scene_ids["ndvi_composite"] = ", ".join(ndvi_stats["scene_ids"])

        _emit_usage_event(
            requested=requested,
            outcomes={"dem": dem, "landcover": landcover,
                      "ndvi": ndvi_stats, "rainfall": rainfall_context},
            module_ms=module_ms,
            request_timer=request_timer,
            http_request=http_request,
            area_km2=aoi["bbox_area_km2"],
            ndvi_stats=ndvi_stats,
        )
        summary = {
            "dem": _with_dem_evidence(dem),
            "ndvi": _with_vegetation_evidence(ndvi_stats),
            "landcover": _with_landcover_evidence(landcover),
            # The resolution actually read, next to the one asked for. The
            # contract has declared this field since it was added and only the
            # worker populated it, so a synchronous caller had no way to tell a
            # 20 m composite from a 250 m one. Both readers take native
            # resolution today, which is also what makes the numbers equal.
            "applied_resolution_m": {
                "dem": 30.0,
                "landcover": 10.0,
                "ndvi": (ndvi_stats or {}).get("resolution_m"),
            },
            "rainfall": _with_rainfall_evidence(rainfall_context),
            "country": country,
            "scene_dates": scene_dates,
            "scene_ids": scene_ids,
            "analysis": {
                "bbox_area_km2": round(aoi["bbox_area_km2"], 2),
                "datasets": sorted(requested),
                "mode": "synchronous",
            },
            "caveats": _caveats(aoi, dem, landcover, ndvi_stats, rainfall_context),
        }
        # Validating here means a change to a producer that breaks the contract
        # fails the request loudly, rather than shipping a shape nobody declared.
        # exclude_none, so a module that was not requested is absent rather than a
        # field of nulls the caller has to read past.
        return ContextResponse(summary=summary).model_dump(exclude_none=True, exclude_defaults=False)

    except HTTPException:
        raise
    except Exception as e:
        # Never echo the raw exception: it routinely carries signed asset URLs.
        print(f"CRITICAL ERROR in /generate-context: {type(e).__name__} - {str(e)[:200]}", file=sys.stderr)
        raise HTTPException(status_code=500, detail="Processing failed; please retry shortly")

# --------------------------------------------------
# EVIDENCE AND CAVEATS
# --------------------------------------------------
# A number without a stated kind is an assertion pretending to be a measurement.
# The distinction that matters most here is rainfall: ERA5 is a reanalysis, a model
# output that assimilates observations, and not a gauge reading even though it
# arrives looking like any other number.

def _evidence(status, source, method, note=None, **extra):
    return {"status": status, "source": source, "method": method, "note": note, **extra}


# Ordered: the first present key explains a failure most specifically, and a
# machine reason beats a restated status. "no_precomputed_series" tells a reader
# what to do; "status not_computed" only tells them what happened.
_FAILURE_KEYS = ("error", "warning", "reason", "message")
_OK_STATUSES = ("ok", "computed", "ready")


def with_evidence(product_key: str, payload: Optional[dict], *,
                  source: Optional[str] = None,
                  method: Optional[str] = None) -> Optional[dict]:
    """Attach the evidence block to any product payload, from the registry.

    The four bespoke wrappers this replaces each decided on their own what counts
    as success, and each was only ever called from one place -- so ``/rainfall``
    and every worker artefact shipped without an evidence block while
    ``contract.py`` declared one required. One function, called from every path,
    cannot be forgotten at the next one.
    """
    import registry

    if payload is None:
        return None
    product = registry.get(product_key)
    declared_status = payload.get("status")

    failure = next((payload[k] for k in _FAILURE_KEYS if payload.get(k)), None)
    if failure is None and declared_status not in (None, *_OK_STATUSES):
        failure = f"status {declared_status}"
    ok = failure is None

    # The note is the one line a reader is most likely to actually read, so it is
    # the product's leading caveat rather than a separate string. That removes
    # four copies of the same sentences that could drift from the registry.
    note = (product.caveats[0] if product and product.caveats else None) if ok else str(failure)

    # A payload that produced no number is unconfirmed whatever the product's
    # declared class is. Reporting "modelled" beside a series that does not exist
    # is the failure this block exists to prevent.
    evidence = _evidence(
        (product.evidence if product else "unconfirmed") if ok else "unconfirmed",
        source or (product.source if product else None),
        method,
        note=note,
        doi=payload.get("doi") or (product.doi if product else None),
        license=payload.get("license"),
        retrieved=payload.get("retrieved"),
    )
    if product is not None:
        evidence["caveats"] = list(product.caveats)
    return {**payload, "status": declared_status or ("ok" if ok else "error"),
            "evidence": evidence}


def _with_dem_evidence(dem):
    return with_evidence("dem", dem, method="area mean of 30 m surface elevation")


def _with_landcover_evidence(landcover):
    return with_evidence("landcover", landcover,
                         method="area share of 10 m land-cover classes")


def _with_vegetation_evidence(result):
    # The one wrapper that still has something to say the registry does not: which
    # sensor the ladder actually picked. Everything else -- the class, the
    # caveats, the failure path -- comes from the registry.
    if result is None:
        return None
    sensor = (result.get("sensor") or {}).get("label") or "a satellite source"
    return with_evidence(
        "ndvi" if "series" not in result else "vegetation_series", result,
        source=sensor,
        method=result.get("method") or "median NDVI composite")


def _with_rainfall_evidence(result):
    if result is None:
        return None
    return with_evidence(
        "rainfall", result, source=result.get("source"),
        method="monthly totals against the 1991-2020 normal")


def _caveats(aoi, dem, landcover, ndvi, rain):
    """Everything a reader should know before relying on the numbers above.

    The habit borrowed from a source worth copying: say where the data stops and
    what was left out, rather than letting a gap look like an absence.
    """
    notes = []
    repaired = (aoi.get("feature", {}).get("properties") or {}).get("geometry_repaired")
    if repaired:
        notes.append(f"Study-area boundary was modified: {repaired}.")
    for name, module in (("Elevation", dem), ("Land cover", landcover),
                         ("Vegetation", ndvi), ("Rainfall", rain)):
        if not isinstance(module, dict):
            continue
        reason = module.get("error") or module.get("reason")
        if reason:
            notes.append(f"{name}: unavailable ({reason}).")
        warning = module.get("warning")
        if warning:
            notes.append(f"{name}: {warning}")
        if module.get("suspect_months"):
            notes.append(
                f"Rainfall: {len(module['suspect_months'])} month(s) reported near-zero "
                "totals and are worth review."
            )
    if isinstance(rain, dict) and rain.get("status") == "ok":
        window = rain.get("window") or {}
        if window.get("end"):
            notes.append(f"Rainfall series ends {window['end']}.")
    return notes


# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "GeoContext Generator API",
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "version": APP_VERSION,
    }

def require_admin_key(
    x_admin_key: Optional[str] = Header(default=None, alias="X-Admin-Key"),
) -> None:
    """Gate the destructive endpoints.

    Deleting a cached series is cheap to do and impossible to undo from outside, so
    an open endpoint is a way for anyone who can reach the API to blank the
    portfolio. The pattern is a shared secret in a header, refused outright when
    unset rather than defaulted to open.
    """
    configured = (os.getenv("ADMIN_API_KEY") or "").strip()
    if not configured:
        raise HTTPException(
            status_code=503,
            detail="this endpoint is disabled: set ADMIN_API_KEY on the server",
        )
    if not x_admin_key or not secrets.compare_digest(x_admin_key.strip(), configured):
        raise HTTPException(status_code=401, detail="missing or invalid X-Admin-Key header")


class ForgetRequest(BaseModel):
    """Identify a study area to remove, by geometry or by cache key."""
    geojson: Optional[Dict[str, Any]] = None
    cache_key: Optional[str] = None


def _rainfall_by_key(key: str, indicator: str) -> dict:
    """Read a precomputed artefact by cache key, for either transport.

    Attaches evidence here, at the single place both verbs read through.
    ``contract.py`` has declared an evidence block required on every module for
    some time, and it was only ever emitted by ``/generate-context`` -- so a
    rainfall series fetched by key, or a series served from a worker artefact,
    arrived with no statement of what kind of number it was.
    """
    import jobs
    import rainfall

    if not re.fullmatch(r"[0-9a-f]{32}", key or ""):
        raise HTTPException(
            status_code=422,
            detail="cache_key must be 32 lowercase hex characters",
        )
    name = (indicator or "rainfall").strip().lower()
    if name in jobs.RUNTIME_INDICATORS:
        return jobs.read_artefact(key, name) or {
            "status": "not_computed",
            "reason": "no_computed_artefact",
            "message": f"No {name} has been computed for this area yet.",
            "cache_key": key,
        }
    return _with_rainfall_evidence(rainfall.cached_context_by_key(key))


@app.get("/rainfall")
async def rainfall_lookup_by_key(cache_key: str, indicator: str = "rainfall"):
    """Read a computed series by key, without resending the polygon.

    The client has always called this with GET and a query string, while the
    route was registered POST-only, so every vegetation-series fetch returned
    405 and the chart behind it could never render. A key-addressed read is a
    GET by any reading of the verb, so the transport follows the intent rather
    than the other way round.
    """
    result = _rainfall_by_key(cache_key.strip(), indicator)
    import rainfall

    return {
        "rainfall": result,
        "cache_key": cache_key.strip(),
        "analysis": {
            "bbox_area_km2": None,
            "mode": "cache lookup",
            "remote_configured": rainfall.remote_prefix() is not None,
        },
    }


@app.post("/rainfall")
async def rainfall_lookup(request: RainfallLookupRequest):
    """Return the precomputed rainfall series for a study area.

    Separate from ``/generate-context`` on purpose. There the admission policy is
    about bounding raster work, but rainfall is a cache lookup: reading a series
    costs a millisecond and about 1 KB. Applying the raster caps here made the
    published portfolio unreachable for the very areas it was built for.
    """
    import jobs
    import rainfall

    if request.cache_key:
        # A caller that already knows the key never resends a polygon, which for a
        # detailed conservancy is a megabyte of coordinates.
        key = request.cache_key.strip()
        result = _rainfall_by_key(key, request.indicator or "rainfall")
        bbox_area = None
    else:
        if not request.geojson:
            raise HTTPException(status_code=422, detail="supply geojson or cache_key")
        area = validate_for_lookup(request.geojson)
        key = rainfall.geometry_hash(area["feature"]["geometry"])
        result = _with_rainfall_evidence(
            rainfall.cached_context(area["feature"]["geometry"]))
        bbox_area = round(area["bbox_area_km2"], 2)
    return {
        "rainfall": result,
        "cache_key": key,
        "analysis": {
            "bbox_area_km2": bbox_area,
            "mode": "cache lookup",
            "remote_configured": rainfall.remote_prefix() is not None,
        },
    }


@app.post("/rainfall/plan")
async def plan_rainfall(
    payload: SubmitRequest,
    window_start: str = "",
    window_end: str = "",
):
    """What would be computed for this area, without queueing anything.

    Read-only, so the caller can show what it will cost in resolution and time
    before committing. The synchronous endpoint refuses a large area outright, so
    this is the honest way to tell a user what they would actually get instead.
    """
    import indicators
    import jobs

    if not payload.geojson:
        raise HTTPException(status_code=422, detail="supply geojson to plan an area")
    area = validate_for_lookup(payload.geojson)
    requested = [
        d.strip().lower()
        for d in (payload.indicator or "").split(",")
        if d.strip().lower() in jobs.INDICATORS
    ] or ["rainfall"]
    plans = []
    for indicator in requested:
        plan = indicators.plan_indicator(
            indicator, area["bbox_area_km2"],
            start=window_start or None, end=window_end or None,
        )
        plan["already_computed"] = (
            jobs.status_for(
                rainfall_hash(area["feature"]["geometry"]), indicator
            )["state"] == "ready"
        )
        plans.append(plan)
    return {
        "plans": plans,
        "analysis": {
            "bbox_area_km2": round(area["bbox_area_km2"], 2),
            "synchronous_limit_km2": MAX_SYNC_BBOX_KM2,
            "cache_key": rainfall_hash(area["feature"]["geometry"]),
        },
    }


@app.post("/rainfall/submit")
async def submit_rainfall(
    payload: SubmitRequest,
    http_request: Request = None,
    window_start: str = "",
    window_end: str = "",
):
    """Queue a study area for precomputation, and return immediately.

    Computing a series takes about 60 seconds of ERA5 reads, so this never waits
    for it. The job record is durable: it survives a restart, and the runner
    completes whatever is still pending. An area that already has a series is
    reported ready without queueing.
    """
    import jobs

    if not payload.geojson:
        raise HTTPException(status_code=422, detail="supply geojson to submit an area")
    area = validate_for_lookup(payload.geojson)
    key = rainfall_hash(area["feature"]["geometry"])
    indicator = (payload.indicator or "rainfall").strip().lower()
    if indicator not in jobs.INDICATORS:
        raise HTTPException(
            status_code=422,
            detail=f"unknown indicator {indicator!r}; choose from {list(jobs.INDICATORS)}",
        )

    state = jobs.submit(
        area["feature"]["geometry"],
        indicator=indicator,
        label=(area["feature"].get("properties") or {}).get("NAME"),
        submitted_by=_client_host(http_request),
    )

    if state.get("state") == jobs.PENDING and RAINFALL_AUTORUN:
        _autorun_jobs()
    # Say up front what the worker will do with the area, because a large area is
    # answered at a coarser resolution rather than refused, and the caller should
    # know which they are getting.
    import indicators

    planned = indicators.plan_indicator(
        indicator,
        area["bbox_area_km2"],
        start=window_start or None,
        end=window_end or None,
    )
    return {
        "submission": state,
        "cache_key": key,
        "indicator": indicator,
        "planned": planned,
        "analysis": {
            "bbox_area_km2": round(area["bbox_area_km2"], 2),
            "compute_seconds_typical": (planned or {}).get("estimated_seconds"),
        },
    }


def _autorun_jobs() -> None:
    """Run one worker sweep in its own process, best effort.

    A subprocess rather than a thread, for two reasons. The runner is async, and
    a coroutine handed to ``to_thread`` is never awaited; and the vegetation path
    takes a semaphore bound to the server's event loop, which cannot be used from
    another loop. Either way the same code path the container runs is the right
    one to run.

    The job record is what makes a submission durable, so losing this to a restart
    costs nothing but a delay: the supervised worker picks the job up.
    """
    import asyncio
    import sys

    async def spawn():
        try:
            await asyncio.create_subprocess_exec(
                sys.executable, "-m", "worker", "--once",
                stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            )
        except Exception:  # noqa: BLE001 - best effort by definition
            pass
        finally:
            _RAINFALL_TASKS.discard(spawn)

    try:
        task = asyncio.get_running_loop().create_task(spawn())
        _RAINFALL_TASKS.add(task)
    except RuntimeError:
        pass


@app.get("/rainfall/status")
async def rainfall_status(cache_key: str, indicator: str = "rainfall"):
    """State of one area's submission, without resending its polygon."""
    if not re.fullmatch(r"[0-9a-f]{32}", cache_key or ""):
        raise HTTPException(status_code=422, detail="cache_key must be 32 lowercase hex characters")
    import jobs

    return {
        # The indicator matters: the client polls once per queued module, and
        # asking for the rainfall job's state while a vegetation job is running
        # reported the wrong module -- "not_submitted" when no rainfall series
        # existed, and a false "ready" when one did.
        "submission": jobs.status_for(cache_key, indicator.strip().lower() or "rainfall"),
        "computed": rainfall_hash and rainfall_cache_present(cache_key),
    }


def rainfall_cache_present(key: str) -> bool:
    import rainfall

    return rainfall.read_cache(key) is not None


def rainfall_hash(geometry: dict) -> str:
    import rainfall

    return rainfall.geometry_hash(geometry)


@app.post("/admin/rainfall/forget")
async def forget_rainfall_series(
    payload: ForgetRequest,
    x_admin_key: Optional[str] = Header(default=None, alias="X-Admin-Key"),
):
    """Delete a cached rainfall series for one study area.

    The cache is keyed by a hash of the submitted geometry and holds that area's
    monthly series, so it is a record about a specific piece of land even though
    the polygon itself is never stored. A request to remove an area's data cannot
    be honoured without this.

    Only exact 32-character keys are removed, and only files that look like
    series, so a malformed or hostile key reaches nothing else. The per-read cell
    cache is shared between areas and is deliberately left alone.

    Requires the ``X-Admin-Key`` header. Removing a series cannot be undone from
    outside, so an open endpoint is a way to blank the portfolio.
    """
    require_admin_key(x_admin_key)
    import rainfall

    if payload.geojson:
        geom = payload.geojson.get("geometry") if payload.geojson.get("type") == "Feature" else payload.geojson
        if not isinstance(geom, dict) or geom.get("type") not in {"Polygon", "MultiPolygon"}:
            raise HTTPException(status_code=422, detail="geojson must be a polygon or Feature")
        key = rainfall.geometry_hash(geom)
    else:
        key = (payload.cache_key or "").strip()
        if not key:
            raise HTTPException(status_code=422, detail="supply geojson or cache_key")

    if not re.fullmatch(r"[0-9a-f]{32}", key):
        raise HTTPException(status_code=422, detail="cache_key must be 32 hexadecimal characters")

    if rainfall.cache_path(key).parent.resolve() != rainfall.cache_dir().resolve():
        raise HTTPException(status_code=400, detail="refusing a path outside the cache")
    # Both stores: clearing only the local copy would leave the published artefact
    # in the bucket, which is the copy that survives a redeploy.
    outcome = rainfall.forget(key)
    import jobs

    job_dropped = jobs.drop(key)
    print(f"rainfall cache: local={outcome['local']} remote={outcome['remote']} "
          f"job_dropped={job_dropped} for {key}", file=sys.stderr)
    return {"cache_key": key, "removed": bool(outcome["local"] or outcome["remote"]),
            "local": outcome["local"], "remote": outcome["remote"],
            "job_record_dropped": job_dropped, "data": "rainfall series"}


@app.post("/rainfall/forget")
async def forget_my_area(payload: ForgetRequest, http_request: Request = None):
    """Let a user remove the record their own analysis created.

    We hold a series derived from the polygon someone submitted, plus a job
    record naming it, and there is no account system to ask them through. Without
    this, a user has no way to withdraw what they handed over -- which contradicts
    the privacy position the rest of the app argues for. The admin route cannot
    do it: it is keyed on a secret a browser does not have.

    **Why the key alone is enough.** The cache key is a hash of the submitted
    geometry, so holding it means you submitted that geometry. Guessing one is a
    128-bit preimage, which is not a thing anyone does by accident.

    **What this does not give you.** If two people analyse the *same* boundary
    they get the *same* key, so one can remove the other's series. That is a real
    limitation and it is the price of having no accounts. It is bounded rather
    than free, because what is removed is a derived climate series over a
    published reanalysis: not personal, not secret, and the worker will recompute
    it on the next request. The cost is a little compute, not data loss.

    Deliberately narrow: exact 32-hex keys only, artefacts for that key only, and
    the shared per-read cell cache is left alone because it is keyed by grid
    rather than by area and other areas still read from it.
    """
    import rainfall

    key = (payload.cache_key or "").strip()
    if not key:
        raise HTTPException(
            status_code=422,
            detail="supply the cache_key shown for this area",
        )
    if not re.fullmatch(r"[0-9a-f]{32}", key):
        raise HTTPException(
            status_code=422,
            detail="cache_key must be 32 hexadecimal characters",
        )
    if rainfall.cache_path(key).parent.resolve() != rainfall.cache_dir().resolve():
        raise HTTPException(status_code=400, detail="refusing a path outside the cache")

    outcome = rainfall.forget(key)
    import jobs

    # rainfall is not in RUNTIME_INDICATORS -- it has its own job record.
    dropped = {name: jobs.drop(key, name)
               for name in (*jobs.RUNTIME_INDICATORS, "rainfall")}
    print(
        f"rainfall forget: user request removed local={outcome['local']} "
        f"remote={outcome['remote']} jobs={dropped} for {key}",
        file=sys.stderr,
    )
    return {
        "cache_key": key,
        "removed": bool(outcome["local"] or outcome["remote"] or any(dropped.values())),
        "local": outcome["local"],
        "remote": outcome["remote"],
        "job_records_dropped": dropped,
        "data": "the rainfall series and any queued jobs for this area",
    }


@app.post("/questions/compare")
async def compare_areas(request: Request):
    """Rank several areas on the same measures. The planner's question.

    Areas may be given as geometry, or by anything the resolver understands --
    an administrative id, a country and level, a point, a box -- so "these
    twenty wards" does not require twenty files. Resolution is per area and the
    resolver's own failures are reported per area rather than failing the batch:
    one misspelled ward out of twenty should not lose the other nineteen.

    The cell count travels with every area, because for precipitation two areas
    inside one grid cell are the same measurement and a ranking between them is
    noise dressed as a finding.
    """
    import areas as areas_module
    import questions as questions_module

    body = await request.json()
    specs = body.get("areas") or []
    if not isinstance(specs, list) or not specs:
        raise HTTPException(status_code=422, detail="supply a non-empty list of areas")

    try:
        window = questions_module.parse_window(start=body.get("start"),
                                               end=body.get("end"),
                                               years=body.get("years"))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    resolver = areas_module.GaulResolver()
    resolved, failures = [], []
    for index, spec in enumerate(specs):
        spec = spec if isinstance(spec, dict) else {"geojson": spec}
        try:
            if spec.get("geojson"):
                aoi = validate_for_lookup(spec["geojson"])
                area = areas_module.Area(
                    geometry={"type": "Feature", "properties": {},
                              "geometry": aoi["feature"]["geometry"]},
                    bbox=tuple(aoi["bbox"]), area_km2=aoi["bbox_area_km2"],
                    name=spec.get("name"), source="supplied")
            else:
                area = areas_module.resolve_area(
                    resolver, id=spec.get("id"), level=spec.get("level"),
                    country=spec.get("country"), admin1=spec.get("admin1"),
                    admin2=spec.get("admin2"), lat=spec.get("lat"),
                    lon=spec.get("lon"), bbox=spec.get("bbox"))
        except areas_module.ResolverError as exc:
            failures.append({"index": index, "name": spec.get("name"),
                             "reason": exc.reason, "detail": str(exc)})
            continue
        resolved.append(area)

    if not resolved:
        raise HTTPException(status_code=404,
                            detail="none of those areas could be resolved")

    plan = questions_module.plan_comparison([a.describe() for a in resolved], window)
    return {
        "plan": plan,
        "areas": [a.describe() for a in resolved],
        "unresolved": failures,
    }


@app.get("/questions")
async def list_questions():
    """The four questions, and what each one costs to answer.

    Published so a client can offer the right control for the right persona
    without hard-coding the answer: `compare` needs several areas, `history` and
    `watch` need dates, and all four differ in whether they can be answered
    inside a request.
    """
    import questions

    return {
        "questions": list(questions.QUESTIONS),
        "needs": {k: list(v) for k, v in questions.QUESTION_NEEDS.items()},
        "presets_years": list(questions.DEFAULT_PRESETS_YEARS),
        "monthly_bin_limit": questions.MONTHLY_BIN_LIMIT,
        "note": "dates are the primary input; the label, bin width, comparison "
                "normal and routing are all derived from them",
    }


@app.get("/questions/plan")
async def plan_a_question(
    question: str,
    start: Optional[str] = None,
    end: Optional[str] = None,
    years: Optional[int] = None,
    product: Optional[str] = None,
):
    """What answering this would fetch, and whether it can happen inside a request.

    The plan without the work. It is what a client calls to decide between
    showing a result now and offering the offline route, and it costs no raster
    read -- so it stays fast enough to call before deciding what to show.
    """
    import questions

    try:
        window = questions.parse_window(start=start, end=end, years=years)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    try:
        plan = questions.plan_question(
            question, window, products=(product,) if product else None)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return plan.describe()


@app.get("/areas/resolve")
async def resolve_study_area(
    id: Optional[str] = None,
    level: Optional[int] = None,
    country: Optional[str] = None,
    admin1: Optional[str] = None,
    admin2: Optional[str] = None,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    bbox: Optional[str] = None,
    simplify: Optional[float] = None,
):
    """Turn however the caller can name an area into one canonical geometry.

    Four ways in, because that is how people arrive: an administrative id from
    another system, a country and level from a report, a point from a phone, and
    a box from a spreadsheet. Until now only the fourth worked, and only if the
    caller had the file.

    Kept separate from the product registry on purpose. A product yields a
    Measure; this yields an Area. Putting an Area in the registry would blur the
    one thing that makes the registry usable -- that every entry describes
    something that produces a number.
    """
    import areas

    try:
        area = areas.resolve_area(
            areas.GaulResolver(), id=id, level=level, country=country,
            admin1=admin1, admin2=admin2, lat=lat, lon=lon, bbox=bbox,
            simplify=simplify)
    except areas.ResolverUnavailable as exc:
        # The service is down, which is not the caller's fault and is not a 500.
        raise HTTPException(status_code=503, detail=str(exc))
    except areas.ResolverError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return area.describe()


@app.get("/analytics/summary")
async def analytics_summary(limit: int = 2000):
    """Aggregate view of recent usage, for whoever is looking after the service.

    Aggregates only. The per-event rows are deliberately not returned: a row
    carries a timestamp, a duration and an area band, and a long enough tail of
    them starts to describe a specific person even without any field that does.
    """
    import usage as _usage

    events = _usage.read_events()[-max(1, min(limit, 20000)):]
    summary = _usage.summarise(events)
    per_module: dict[str, dict[str, int]] = {}
    per_dataset: dict[str, int] = {}
    for event in events:
        for module, verdict in (event.get("outcomes") or {}).items():
            per_module.setdefault(module, {})
            per_module[module][verdict] = per_module[module].get(verdict, 0) + 1
        for dataset in event.get("datasets_requested") or []:
            per_dataset[dataset] = per_dataset.get(dataset, 0) + 1
    return {
        "analytics_enabled": _usage.analytics_enabled(),
        **summary,
        "by_module": {k: dict(sorted(v.items())) for k, v in sorted(per_module.items())},
        "datasets_requested": dict(sorted(per_dataset.items())),
        "window": {
            "from": events[0]["at"] if events else None,
            "to": events[-1]["at"] if events else None,
        },
        "note": (
            "Counts and percentiles only. Individual rows are not exposed, and no "
            "submitted geometry, raw area or full address is ever recorded."
        ),
    }


@app.get("/analytics")
async def analytics_state():
    """Whether this deployment records usage, so a caller can be told plainly."""
    enabled = usage.analytics_enabled()
    return {
        "analytics_enabled": enabled,
        "records": (
            None if not enabled
            else "dataset outcomes, durations, a coarse area band and a truncated "
                 "client prefix. No submitted geometry, no raw area, no full address."
        ),
        "opt_out": "Set ANALYTICS_DISABLED=1, or send Analytics-Do-Not-Track: true.",
    }


@app.get("/version")
async def get_version():
    return {
        "version": APP_VERSION,
        "optimizations": [
            "validated_multipolygon_aoi",
            "bounded_planetary_computer_search",
            "clip_before_ndvi_reduction",
            "synchronous_capacity_guardrails",
            "per_tile_statistic_accumulation",
            "latitude_aware_ndvi_grid",
            "all_nodata_guard",
            "separate_ndvi_concurrency_guard",
        ],
        # From the registry, so a partner reads one declaration rather than a
        # list that can fall out of step with the code that serves the data. The
        # previous hand-maintained `available_datasets` said nothing about
        # resolution, latency or the area over which a measure means anything,
        # which are the three things a reader needs in order to trust a number.
        "products": registry.describe_all(),
        "measures": registry.measures(),
        "available_datasets": sorted(AVAILABLE_DATASETS),
        "available_sensors": {
            sid: {
                "label": sn.label,
                "collection": sn.collection,
                "native_resolution_m": sn.native_res_m,
                "archive_start": sn.archive_start,
                "ndvi": "computed from bands" if sn.computes_ndvi else "product",
                "cloud_mask": sn.cloud_mask,
            }
            for sid, sn in sorted(SENSORS.items())
        },
        "default_sensor": "auto",
        "max_sync_bbox_km2": MAX_SYNC_BBOX_KM2,
        "max_landcover_bbox_km2": MAX_LANDCOVER_BBOX_KM2,
        "max_source_tiles": MAX_SOURCE_TILES,
        "max_pc_scenes": MAX_PC_SCENES,
        "max_concurrent_analyses": MAX_CONCURRENT_ANALYSES,
        "max_concurrent_ndvi": MAX_CONCURRENT_NDVI,
        # The synchronous path reads the composite at 20 m. It is not a property
        # of the service: the ladder reaches 60, 100 and 250 m for larger areas,
        # and a MODIS 250 m product is read at 250 m whatever the target says,
        # because a product cannot be resampled finer without inventing detail.
        "ndvi_resolution_m": 20,
        "ndvi_resolution_steps": [
            {"max_bbox_km2": 100.0, "resolution_m": 20},
            {"max_bbox_km2": 1000.0, "resolution_m": 60},
            {"max_bbox_km2": 10000.0, "resolution_m": 100},
        ],
        "coarsest_resolution_m": sensors.COARSEST_RESOLUTION_M,
        # Previously published as "not available until a durable asynchronous
        # worker is deployed". One is deployed, and it answers areas past the
        # synchronous cap at a coarser resolution. A partner reading this file
        # would have cited it.
        "large_area_mode": {
            "available": True,
            "how": "queue the area; the worker reads it at a coarser resolution "
                   "and records the resolution actually used",
            "queue_max_pending": jobs.MAX_PENDING_JOBS,
            "job_lease_seconds": jobs.JOB_LEASE_SECONDS,
        },
        # Enforced but previously unpublished, so a client could not know what
        # the service would accept.
        "max_geojson_bytes": MAX_GEOJSON_BYTES,
        "max_aoi_vertices": MAX_AOI_VERTICES,
        "analysis_raster_timeout_seconds": RASTER_TIMEOUT_SECONDS,
        "ndvi_timeout_seconds": NDVI_TIMEOUT_SECONDS,
        "asset_search_seconds": ASSET_SEARCH_SECONDS,
        "worker_interval_seconds": float(os.getenv("RAINFALL_WORKER_INTERVAL", "15")),
    }
