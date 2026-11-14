"""Per-point enrichment: join field observations to the Earth-observation stack.

This is the pipeline's core. A connector (CSV, GeoJSON, ArcGIS, EarthRanger,
COPA) produces observations; this module attaches the evidence that makes them
defensible: elevation and land cover sampled at the point, NDVI and rainfall for
the observation month, and the flags that say what is missing.

The AOI-level analysis (NDVI mean, land-cover dominant class, rainfall series)
comes from the existing generate-context path. Only the point samples are new,
and they are read from the signed COG tiles the analysis already produced.
"""

from __future__ import annotations

import math
from typing import Optional

import rasterio
from rasterio.warp import transform as warp_transform
from rasterio.windows import Window

from contract import EnrichRequest, EnrichedRecord
from registry import LULC_CLASS_LABELS


def sample_point(tiles: list[str], lon: float, lat: float) -> Optional[float]:
    """Sample the first tile that contains the point, or None.

    The tiles are signed COGs covering the AOI. A point on a tile boundary is
    tried against the next tile; a nodata pixel is treated as no sample rather
    than as a real value (a 0 m DEM pixel in an inland highland AOI is a void,
    not sea level).
    """
    for url in tiles:
        try:
            with rasterio.open(url) as src:
                xs, ys = warp_transform("EPSG:4326", src.crs, [lon], [lat])
                row, col = src.index(xs[0], ys[0])
                if row < 0 or col < 0 or row >= src.height or col >= src.width:
                    continue
                value = float(src.read(1, window=Window(col, row, 1, 1))[0, 0])
                if not math.isfinite(value):
                    continue
                if src.nodata is not None and math.isclose(value, float(src.nodata)):
                    continue
                return value
        except Exception:
            # A signed URL that expired or a tile that does not cover the point
            # is a missing sample, not a failed request.
            continue
    return None


def _month(timestamp: Optional[str]) -> Optional[str]:
    if not timestamp or len(timestamp) < 7:
        return None
    head = timestamp[:7]
    return head if len(head) == 7 and head[4] == "-" else None


def enrich_records(request: EnrichRequest, ndvi_tile: Optional[str] = None) -> list[EnrichedRecord]:
    bbox = request.bbox or []
    series = request.rainfall_series or []
    records: list[EnrichedRecord] = []
    for point in request.observations:
        flags: list[str] = []
        month = _month(point.timestamp)
        if not point.timestamp:
            flags.append("no timestamp")
        if month and series and not any(row.get("month") == month for row in series):
            flags.append("no rainfall for month")
        if len(bbox) >= 4:
            min_lon, min_lat, max_lon, max_lat = bbox[:4]
            if point.lon < min_lon or point.lon > max_lon or point.lat < min_lat or point.lat > max_lat:
                flags.append("outside AOI bbox")

        elevation = sample_point(request.dem_tiles, point.lon, point.lat)
        landcover_code = sample_point(request.landcover_tiles, point.lon, point.lat)
        landcover = None
        if landcover_code is not None:
            landcover = LULC_CLASS_LABELS.get(int(round(landcover_code)))
        if elevation is None and not request.dem_tiles:
            flags.append("no DEM tile")
        if landcover is None and not request.landcover_tiles:
            flags.append("no land-cover tile")
        # Per-point NDVI when the COG is available; the AOI mean is the fallback
        # for a point outside the COG or a read without a composite.
        ndvi = sample_point([ndvi_tile], point.lon, point.lat) if ndvi_tile else None
        if ndvi is None:
            ndvi = request.ndvi_mean

        rainfall = None
        if month:
            for row in series:
                if row.get("month") == month:
                    value = row.get("precip_mm")
                    rainfall = float(value) if isinstance(value, (int, float)) else None
                    break

        records.append(EnrichedRecord(
            id=point.id,
            lon=point.lon,
            lat=point.lat,
            timestamp=point.timestamp,
            ndvi=ndvi,
            landcover=landcover or request.landcover_dominant,
            landcover_code=int(round(landcover_code)) if landcover_code is not None else None,
            rainfall_mm=rainfall,
            elevation_m=elevation,
            flags=flags,
        ))
    return records