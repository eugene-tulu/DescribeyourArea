# --------------------------------------------------
# IMPORTS
# --------------------------------------------------
from fastapi import FastAPI, HTTPException, Request
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
import math
import os
from dotenv import load_dotenv
import sys
import warnings
from shapely.geometry import box, mapping, shape
from shapely.ops import unary_union
from affine import Affine
from pyproj import CRS, Geod, Transformer
import requests


# --------------------------------------------------
# ENVIRONMENT
# --------------------------------------------------
load_dotenv()


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
MAX_GEOJSON_BYTES = _env_int("MAX_GEOJSON_BYTES", 500_000)
MAX_AOI_VERTICES = _env_int("MAX_AOI_VERTICES", 10_000)
MAX_SYNC_BBOX_KM2 = _env_float("MAX_SYNC_BBOX_KM2", 100.0)
# NDVI is limited by its 75 s budget, not by memory, so it is raised to match the
# synchronous cap: 100 km2 measures 453 MB and 32 s, 400 km2 times out.
MAX_NDVI_BBOX_KM2 = _env_float("MAX_NDVI_BBOX_KM2", 100.0)
# WorldCover is the one module whose memory grows with area, so it gets its own
# budget. 1,000 km2 measures 235 MB; 5,500 km2 measures 1,205 MB and would starve
# the NDVI path inside a 1.8 GB container.
MAX_LANDCOVER_BBOX_KM2 = _env_float("MAX_LANDCOVER_BBOX_KM2", 1_000.0)
MAX_PC_SCENES = _env_int("MAX_PC_SCENES", 4)
# A study area can straddle many source tiles; bound the fan-out so a pathological
# bounding box cannot issue an unbounded number of COG opens.
MAX_SOURCE_TILES = _env_int("MAX_SOURCE_TILES", 64)
# Concurrency limits, derived from measurement rather than from memory alone.
#
# Measured against Planetary Computer (2026-09-26), 1.25 vCPU, eight concurrent
# requests through the real ASGI app:
#
#   marginal memory   ~25 MB per concurrent request (peak 322 MB at N=8)
#   CPU               4-14% of one core at N=8
#   dem+landcover     wall time flat from N=1 to N=8 (8.5 s -> 7.5 s)
#   dem+landcover+ndvi p50 latency 27 s at N=1, 38 s at N=2, 52 s at N=4, 67 s at N=8
#
# Neither memory nor CPU is the binding constraint; remote read latency is. The
# global guard is therefore generous, while the NDVI path gets a second, tighter
# guard because it is the one that degrades. Aggregate throughput still improves
# with concurrency, so this trades user-facing latency for throughput rather than
# being free.
MAX_CONCURRENT_ANALYSES = _env_int("MAX_CONCURRENT_ANALYSES", 8)
MAX_CONCURRENT_NDVI = _env_int("MAX_CONCURRENT_NDVI", 3)
# A caller that cannot enter the global guard is told the service is busy. The
# NDVI guard is proportionally slower, so waiting longer is reasonable there; on
# exhaustion the rest of the analysis still returns with NDVI marked unavailable.
ANALYSIS_ACQUIRE_SECONDS = _env_float("ANALYSIS_ACQUIRE_SECONDS", 2.0, minimum=0.0)
NDVI_ACQUIRE_SECONDS = _env_float("NDVI_ACQUIRE_SECONDS", 20.0, minimum=0.0)
ANALYSIS_DRAIN_SECONDS = _env_float("ANALYSIS_DRAIN_SECONDS", 20.0, minimum=0.0)
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
APP_VERSION = "1.5.0"


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

class ContextResponse(BaseModel):
    summary: Dict[str, Any]

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


def _polygon_geometry(value: Any):
    """Return a validated Shapely Polygon/MultiPolygon or raise a client error."""
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

    if geom.is_empty or not geom.is_valid:
        raise HTTPException(status_code=400, detail="Study-area geometry is empty or invalid")
    return geom


def canonicalize_geojson(
    geojson: dict,
    *,
    max_bytes: int = MAX_GEOJSON_BYTES,
    max_vertices: int = MAX_AOI_VERTICES,
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
        geometries = [_polygon_geometry(feature) for feature in features]
        properties: dict[str, Any] = {}
    elif input_type in {"Feature", "Polygon", "MultiPolygon"}:
        geometries = [_polygon_geometry(geojson)]
        properties = geojson.get("properties", {}) if input_type == "Feature" else {}
        if not isinstance(properties, dict):
            properties = {}
    else:
        raise HTTPException(
            status_code=400,
            detail="GeoJSON must be a Feature, FeatureCollection, Polygon, or MultiPolygon",
        )

    vertex_count = sum(_position_count(mapping(geom).get("coordinates")) for geom in geometries)
    if vertex_count > max_vertices:
        raise HTTPException(
            status_code=413,
            detail=f"Study area has too many vertices (limit: {max_vertices})",
        )

    merged = unary_union(geometries)
    if merged.is_empty or merged.geom_type not in {"Polygon", "MultiPolygon"} or not merged.is_valid:
        raise HTTPException(status_code=400, detail="Study-area polygons cannot be combined safely")

    return {
        "type": "Feature",
        "properties": properties,
        "geometry": mapping(merged),
    }


def aoi_bbox(feature: dict) -> list[float]:
    """Derive bounds from canonical geometry; handles Polygon and MultiPolygon."""
    geom = _polygon_geometry(feature)
    minx, miny, maxx, maxy = geom.bounds
    if minx < -180 or maxx > 180 or miny < -90 or maxy > 90 or minx >= maxx or miny >= maxy:
        raise HTTPException(status_code=400, detail="Study-area coordinates must be valid WGS84 longitude/latitude")
    return [float(minx), float(miny), float(maxx), float(maxy)]


def _bbox_area_km2(bbox: list[float]) -> float:
    """Geodesic area of the bounding box used by STAC and raster reads."""
    minx, miny, maxx, maxy = bbox
    area_m2, _ = WGS84_GEOD.geometry_area_perimeter(box(minx, miny, maxx, maxy))
    return abs(area_m2) / 1_000_000


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
async def _search_sentinel_items(bbox: list[float], max_scenes: int) -> list:
    """Fetch the clearest available scenes, retrying only transient failures."""
    end = datetime.datetime.now(datetime.UTC)
    start = end - datetime.timedelta(days=90)
    time_window = f"{start.date().isoformat()}/{end.date().isoformat()}"

    def search_once() -> list:
        catalog = pystac_client.Client.open(STAC_URL)
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=time_window,
            query={"eo:cloud_cover": {"lt": 30}},
            limit=max_scenes,
            max_items=max_scenes,
        )
        return list(search.items())

    for attempt in range(3):
        try:
            items = search_once()
            if items:
                # Scene-level cloud cover describes the whole 110 km tile, so a
                # scene can score near zero and still be fully overcast over the
                # study area. Taking the least cloudy candidates first is the
                # cheapest way to avoid a composite made entirely of cloud.
                items.sort(key=lambda item: (item.properties or {}).get("eo:cloud_cover", 100.0))
            return items
        except Exception:
            if attempt == 2:
                print("STAC search failed after 3 attempts", file=sys.stderr)
                return []
            await asyncio.sleep(0.5 * (2 ** attempt))
    return []



# Sentinel-2 Scene Classification codes. A pixel is unusable for a vegetation
# index when it is nodata, defective, in shadow, or flagged as cirrus or probable
# cloud.
#
# Classes 4 (cloud) and 5 (bright cloud) are deliberately NOT rejected outright.
# The brightness test behind them flags bright semi-arid ground and desert as
# "bright cloud" across entire tiles: a Sahara tile reads 100% class 5 while its
# B04/B08 reflectance (0.41/0.49) and NDVI (~0.09) are plainly desert, not cloud.
# Rejecting them empties the result for exactly the rangeland this service is
# built for. Residual cloud is removed by discarding implausible NDVI instead,
# because both cloud and open water give a near-zero or negative index.
SCL_REJECTED = frozenset({0, 1, 3, 8, 9, 10, 11})
NDVI_MIN_PLAUSIBLE = 0.0
SCL_LABELS = {
    0: "nodata", 1: "saturated", 2: "dark", 3: "cloud shadow", 4: "cloud",
    5: "bright cloud", 6: "water", 7: "unclassified", 8: "cloud (medium)",
    9: "cloud (high)", 10: "thin cirrus", 11: "snow",
}

# Grid CRS for the NDVI composite. EPSG:6933 (WGS 84 / NSIDC EASE-Grid 2.0
# Global) is equal-area with metre units, so a pixel is the same area everywhere
# and a reported area percentage means the same thing in Kenya as in Canada. One
# global CRS also avoids the UTM zone-edge problem, where a single conservancy
# straddling a zone boundary would need two grids stitched together.
# It is only valid to ~86 degrees latitude, so polar study areas fall back to a
# local UTM zone.
NDVI_TARGET_EPSG = _env_int("NDVI_TARGET_EPSG", 6933)
_EASE_GRID_MAX_LAT = 85.0


def _target_crs(bbox: list[float]):
    """Pick the analysis CRS, falling back to a local UTM zone near the poles."""
    if abs(bbox[1]) <= _EASE_GRID_MAX_LAT and abs(bbox[3]) <= _EASE_GRID_MAX_LAT:
        return CRS.from_epsg(NDVI_TARGET_EPSG)
    centre_lon = (bbox[0] + bbox[2]) / 2.0
    centre_lat = (bbox[1] + bbox[3]) / 2.0
    zone = min(60, max(1, int((centre_lon + 180.0) / 6.0) + 1))
    return CRS.from_epsg((32700 if centre_lat < 0 else 32600) + zone)


def _sentinel_target_grid(bbox: list[float], resolution_m: int) -> tuple:
    """Build one projected output grid covering the AOI.

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

    Planetary Computer Sentinel-2 assets are separate single-band COGs with no
    band metadata, so the asset href already selects the band. B04/B08 are
    published at 10 m while the composite is built at 20 m, and reading the
    full-resolution blocks for that is the single largest cost in this path.
    Requesting a COG overview reads a fraction of the bytes, but only if the
    *window* is expressed on the overview's own grid, which is why this is not
    just an ``out_shape`` argument.
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


def _reproject_to_grid(
    values: np.ndarray,
    src_transform,
    src_crs,
    target,
    resampling,
    nodata,
) -> np.ndarray:
    # target is (crs, transform, width, height); numpy arrays are (rows, cols).
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


def _scene_ndvi_on_grid(item, target, bbox, resolution_m: int) -> Optional[np.ndarray]:
    """Return per-pixel NDVI for one scene on the shared target grid.

    SCL is read with nearest-neighbour sampling because it is a class map:
    averaging it would invent boundaries and pull cloud edges into clear ground.
    """
    try:
        hrefs = {
            band: item.assets[band].href
            for band in ("B04", "B08", "SCL")
            if band in item.assets
        }
    except AttributeError:
        return None
    if len(hrefs) < 3:
        return None

    try:
        with rio.open(planetary_computer.sign(hrefs["B04"])) as red_src:
            red_window = _window_for_bbox(red_src, bbox)
            if red_window is None:
                return None
            red, transform = _read_window(red_src, red_window, resolution_m, Resampling.average)
            crs = red_src.crs
        with rio.open(planetary_computer.sign(hrefs["B08"])) as nir_src:
            nir_window = _window_for_bbox(nir_src, bbox)
            if nir_window is None:
                return None
            nir, _ = _read_window(nir_src, nir_window, resolution_m, Resampling.average)
        if red is None or nir is None or nir.shape != red.shape:
            return None

        # Sentinel-2 surface reflectance is uint16. Cast before subtracting so a
        # negative difference cannot wrap around to a large unsigned value.
        red_f = red.astype("float32")
        nir_f = nir.astype("float32")
        with np.errstate(divide="ignore", invalid="ignore"):
            ndvi = (nir_f - red_f) / (nir_f + red_f + 1e-8)

        with rio.open(planetary_computer.sign(hrefs["SCL"])) as scl_src:
            scl_window = _window_for_bbox(scl_src, bbox)
            if scl_window is None:
                return None
            scl, scl_transform = _read_window(scl_src, scl_window, resolution_m, Resampling.nearest)
            scl_crs = scl_src.crs
        if scl is None:
            return None
        if scl.shape != red.shape:
            scl = _resize_nearest(scl, red.shape)

        on_grid = _reproject_to_grid(ndvi, transform, crs, target,
                                     Resampling.bilinear, np.nan)
        classes = _reproject_to_grid(scl.astype("float32"), scl_transform, scl_crs,
                                     target, Resampling.nearest, -1.0)
        rejected = np.isin(np.nan_to_num(classes, nan=-1.0).astype("int16"),
                           tuple(SCL_REJECTED))
        on_grid[rejected] = np.nan
        on_grid[~np.isfinite(on_grid)] = np.nan
        on_grid[(on_grid < NDVI_MIN_PLAUSIBLE) | (on_grid > 1.0)] = np.nan
        return on_grid
    except Exception as exc:  # noqa: BLE001 - one bad scene must not fail the request
        print(f"NDVI scene skipped ({getattr(item, 'id', '?')}): {type(exc).__name__}: {str(exc)[:120]}",
              file=sys.stderr)
        return None


def _resize_nearest(arr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    rows = (np.arange(shape[0]) * arr.shape[0] // max(1, shape[0])).clip(0, arr.shape[0] - 1)
    cols = (np.arange(shape[1]) * arr.shape[1] // max(1, shape[1])).clip(0, arr.shape[1] - 1)
    return arr[np.ix_(rows, cols)]


def _load_and_summarize_ndvi(
    items: list,
    bbox: list[float],
    geojson_geom: dict,
    resolution_m: int,
) -> dict:
    """Build a per-pixel median NDVI composite and summarize it.

    Replaces an odc-stac/dask cube with direct windowed COG reads. The old path
    cost ~372 MB of framework overhead before touching a pixel, dominated
    concurrency; this reads only the AOI window of each band.
    """
    target = _sentinel_target_grid(bbox, resolution_m)
    geometry = shape(geojson_geom)
    in_aoi = _geometry_mask_on_grid(geometry, target, bbox)

    scenes: list[np.ndarray] = []
    scene_ids: list[str] = []
    scene_dates: list[str] = []
    for item in items:
        grid = _scene_ndvi_on_grid(item, target, bbox, resolution_m)
        if grid is None:
            continue
        grid = np.where(in_aoi, grid, np.nan)
        if not np.isfinite(grid).any():
            continue
        scenes.append(grid)
        scene_ids.append(item.id)
        stamp = getattr(item, "datetime", None)
        scene_dates.append(stamp.isoformat() if stamp else None)

    if not scenes:
        return {
            "status": "unavailable",
            "source": "Planetary Computer",
            "scenes_examined": len(items),
            "warning": (
                "No usable Sentinel-2 pixels were available for this study area. "
                "Every candidate scene was flagged as cloud, shadow or snow over the "
                "study area; no vegetation value is reported rather than reporting cloud."
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
            "warning": "Cloud and scene-quality masking left no valid Sentinel-2 pixels.",
        "scenes_examined": len(items),
        }

    covered = float(observed.sum()) / float(in_aoi.size)
    return {
        "status": "ok",
        "source": "Planetary Computer",
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
        "method": "sentinel_2_median_composite",
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



async def compute_median_ndvi(
    bbox: list[float],
    geojson_geom: dict,
    max_area_km2: float = MAX_NDVI_BBOX_KM2,
    max_scenes: int = MAX_PC_SCENES,
    resolution_m: int = 20,
) -> dict:
    """Compute a small Sentinel-2 composite from Planetary Computer only.

    A skipped result is explicit: the application never falls back to an
    unbounded MODIS raster read for a large study area.
    """
    bbox_area_km2 = _bbox_area_km2(bbox)
    if bbox_area_km2 > max_area_km2:
        return {
            "status": "skipped",
            "source": "Planetary Computer",
            "bbox_area_km2": round(bbox_area_km2, 2),
            "warning": (
                f"NDVI is available for bounding boxes up to {max_area_km2:g} km²; "
                "use a smaller study area for this synchronous analysis."
            ),
        }

    bounded_scenes = min(MAX_PC_SCENES, max(1, int(max_scenes)))
    # NDVI is the I/O-heavy, latency-sensitive path, so it has its own guard. On
    # exhaustion the caller still gets its terrain and land cover with an explicit
    # reason attached, rather than a 429 for the whole request.
    try:
        await asyncio.wait_for(NDVI_SEMAPHORE.acquire(), timeout=NDVI_ACQUIRE_SECONDS)
    except asyncio.TimeoutError:
        return {
            "status": "unavailable",
            "source": "Planetary Computer",
            "warning": (
                "Vegetation analysis is busy right now. Terrain and land cover are "
                "unaffected; retry shortly for a vegetation value."
            ),
        }
    try:
        items = await _search_sentinel_items(bbox, bounded_scenes)
        if not items:
            return {
                "status": "unavailable",
                "source": "Planetary Computer",
                "warning": "No cloud-filtered Sentinel-2 scenes were available in the last 90 days.",
            }

        try:
            return await asyncio.wait_for(
                run_blocking(_load_and_summarize_ndvi, items, bbox, geojson_geom, resolution_m),
                timeout=75.0,
            )
        except asyncio.TimeoutError:
            return {
                "status": "unavailable",
                "source": "Planetary Computer",
                "warning": "NDVI processing timed out; try a smaller study area shortly.",
            }
        except Exception as exc:
            print(f"⚠️ NDVI processing failed: {type(exc).__name__}: {str(exc)[:120]}", file=sys.stderr)
            return {
                "status": "unavailable",
                "source": "Planetary Computer",
                "warning": "NDVI could not be processed for this study area right now.",
            }
    finally:
        NDVI_SEMAPHORE.release()

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


@app.post("/generate-context", response_model=ContextResponse)
async def generate_context(
    request: GeoJSONRequest,
    include_ndvi: bool = True,
    datasets: Optional[str] = None,
):
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
            timeout=15.0,
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
        try:
            raster_values = await asyncio.wait_for(
                asyncio.gather(*raster_tasks.values()),
                timeout=30.0,
            ) if raster_tasks else []
        except asyncio.TimeoutError:
            raise HTTPException(status_code=504, detail="Raster processing timed out; try a smaller area")
        except MemoryError:
            raise HTTPException(status_code=507, detail="Raster processing exceeded available memory")
        raster_results = dict(zip(raster_tasks.keys(), raster_values))
        dem = interpret_terrain(raster_results["dem"]) if "dem" in raster_results else None
        landcover = raster_results.get("landcover", landcover_over_limit)

        ndvi_stats = None
        if include_ndvi and "ndvi" in requested:
            ndvi_stats = await compute_median_ndvi(
                bbox=bbox,
                geojson_geom=geom,
                max_area_km2=MAX_NDVI_BBOX_KM2,
                max_scenes=MAX_PC_SCENES,
                resolution_m=20,
            )

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

        return {
            "summary": {
                "dem": dem,
                "ndvi": ndvi_stats,
                "landcover": landcover,
                "rainfall": rainfall_context,
                "country": country,
                "scene_dates": scene_dates,
                "scene_ids": scene_ids,
                "analysis": {
                    "bbox_area_km2": round(aoi["bbox_area_km2"], 2),
                    "datasets": sorted(requested),
                    "mode": "synchronous",
                },
            }
        }

    except HTTPException:
        raise
    except Exception as e:
        # Never echo the raw exception: it routinely carries signed asset URLs.
        print(f"CRITICAL ERROR in /generate-context: {type(e).__name__} - {str(e)[:200]}", file=sys.stderr)
        raise HTTPException(status_code=500, detail="Processing failed; please retry shortly")

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
        "available_datasets": sorted(AVAILABLE_DATASETS),
        "max_sync_bbox_km2": MAX_SYNC_BBOX_KM2,
        "max_ndvi_bbox_km2": MAX_NDVI_BBOX_KM2,
        "max_landcover_bbox_km2": MAX_LANDCOVER_BBOX_KM2,
        "max_source_tiles": MAX_SOURCE_TILES,
        "max_pc_scenes": MAX_PC_SCENES,
        "max_concurrent_analyses": MAX_CONCURRENT_ANALYSES,
        "max_concurrent_ndvi": MAX_CONCURRENT_NDVI,
        "ndvi_resolution_m": 20,
        "large_area_mode": "not available until a durable asynchronous worker is deployed",
    }
