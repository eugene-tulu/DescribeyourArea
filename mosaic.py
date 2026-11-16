"""MosaicJSON asset index for multi-tile AOIs.

A large AOI is covered by several source COGs. Without an index, every tile
request opens every source until one covers the tile. This module builds a
MosaicJSON (the cloud-native standard) that maps each z/x/y to the assets that
actually intersect it, so a tile read touches only the COGs it needs.

The index is in-memory and keyed by area/indicator/year; the source COGs are
read through their overviews, so building it is cheap.
"""

from __future__ import annotations

from typing import Optional

import rasterio as rio
import planetary_computer
from morecantile import tms
from rasterio.warp import transform_bounds

_TMS = tms.get("WebMercatorQuad")
_CACHE: dict[str, dict] = {}


def _intersects(a: list[float], b: list[float]) -> bool:
    return not (a[2] < b[0] or a[0] > b[2] or a[3] < b[1] or a[1] > b[3])


def _bounds(href: str) -> Optional[list[float]]:
    try:
        with rio.open(planetary_computer.sign(href)) as src:
            return list(transform_bounds(src.crs, "EPSG:4326", *src.bounds))
    except Exception:
        return None


def build(hrefs: list[str], bbox: list[float], maxzoom: int = 14) -> dict:
    bounds = [b for b in (_bounds(href) for href in hrefs) if b is not None]
    tiles: dict[str, list[int]] = {}
    for z in range(maxzoom + 1):
        north_west = _TMS.tile(bbox[0], bbox[3], z)
        south_east = _TMS.tile(bbox[2], bbox[1], z)
        for x in range(north_west.x, south_east.x + 1):
            for y in range(north_west.y, south_east.y + 1):
                tile_bounds = list(transform_bounds(
                    _TMS.crs, "EPSG:4326", *_TMS.xy_bounds(x, y, z)))
                index = [i for i, b in enumerate(bounds) if _intersects(b, tile_bounds)]
                if index:
                    tiles[f"{z}/{x}/{y}"] = index
    return {
        "mosaicjson": "0.0.3",
        "bounds": [float(v) for v in bbox],
        "minzoom": 0,
        "maxzoom": maxzoom,
        "tiles": tiles,
    }


def get(key: str, indicator: str, hrefs: list[str], bbox: list[float],
        year: Optional[int] = None, maxzoom: int = 14) -> dict:
    cache_key = f"{key}:{indicator}:{year}"
    if cache_key not in _CACHE:
        _CACHE[cache_key] = build(hrefs, bbox, maxzoom)
    return _CACHE[cache_key]


def tiles_for(mosaic: dict, z: int, x: int, y: int, hrefs: list[str]) -> list[str]:
    index = (mosaic.get("tiles") or {}).get(f"{z}/{x}/{y}")
    if not index:
        # No index entry: fall back to all assets (the render loop skips the
        # ones that do not intersect, so this is correct, just less efficient).
        return hrefs
    return [hrefs[i] for i in index if 0 <= i < len(hrefs)]