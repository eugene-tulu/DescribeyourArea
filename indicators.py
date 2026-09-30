"""Compute one indicator for one area, wherever it is going to run.

A submission can be answered in two places, and they must not disagree: the
synchronous request path, which refuses a large area, and the worker, which
computes it at a resolution chosen for the area rather than refused. Both call
``compute_indicator``, so a queued area and an interactive one produce the same
shape, the same provenance and the same evidence.
"""

from __future__ import annotations

import asyncio
import math
import os
import datetime
from typing import Any, Optional

import rainfall
import sensors

# How many years of vegetation history a series covers when the caller does not
# ask for a specific window. MOD13Q1 begins in 2000-02, so 2010 leaves two
# decades of record without pretending to more.
DEFAULT_SERIES_YEARS = 16


def resolution_for(bbox_area_km2: float, indicator: str) -> tuple[int, str]:
    """Metres per pixel for this indicator at this area, and the reason."""
    native = {"landcover": 10, "ndvi": 10, "dem": 30}.get(indicator, 10)
    return sensors.resolution_for_area(bbox_area_km2, native_m=native)


def _synchronous_budget_km2(indicator: str) -> float:
    """Largest area this module will answer inside a request.

    Land cover is the only module whose memory genuinely grows with area, so it
    gets a larger budget than the rest. The figures come from measurement, not
    preference: WorldCover is 235 MB at 1,000 km2 and 1,205 MB at 5,500 km2.
    """
    # Read the definitions rather than re-deriving them from the environment.
    # Both were read here independently, which meant the defaults lived in two
    # places and could drift apart without anything failing.
    import main

    return (main.MAX_LANDCOVER_BBOX_KM2 if indicator == "landcover"
            else main.MAX_SYNC_BBOX_KM2)


async def compute_indicator(
    indicator: str,
    bbox: list[float],
    geojson_geom: dict,
    *,
    area_km2: Optional[float] = None,
    resolution_m: Optional[int] = None,
    window_start: Optional[str] = None,
    window_end: Optional[str] = None,
    enforce_budget: bool = True,
) -> dict:
    """Compute one indicator and return a payload shaped like the contract's.

    ``enforce_budget=False`` is the async path: the point of a worker is to answer
    what the request path had to refuse, so the area cap does not apply there — the
    resolution policy does.
    """
    import rainfall
    from main import (
        _bbox_area_km2,
        _find_core_assets,
        _load_and_summarize_ndvi,
        _scene_index_on_grid,
        _vegetation_target_grid,
        compute_landcover_percentages,
        compute_raster_stats,
        compute_vegetation_index,
        interpret_terrain,
    )
    from shapely.geometry import shape

    area = area_km2 if area_km2 is not None else _bbox_area_km2(bbox)
    chosen, reason = resolution_for(area, indicator)
    if resolution_m:
        chosen = max(chosen, int(resolution_m))

    if indicator == "rainfall":
        return rainfall.cached_context(geojson_geom)

    if indicator == "vegetation_series":
        import vegetation_series

        series_start = window_start or None
        if series_start is None:
            end_dt = datetime.date.fromisoformat((window_end or "")[:10]) if window_end else None
            series_start = (
                f"{end_dt.year - DEFAULT_SERIES_YEARS:04d}-01-01" if end_dt
                else None
            )
        return vegetation_series.compute_monthly_series(
            geojson_geom, bbox,
            start=series_start or "2010-01-01",
            end=(window_end or None),
        )

    if indicator == "dem":
        assets = await _find_core_assets(bbox, need_dem=True, need_landcover=False)
        stats = compute_raster_stats(assets["dem"], {"type": "Feature", "properties": {},
                                                     "geometry": geojson_geom})
        return interpret_terrain(stats)

    if indicator == "landcover":
        assets = await _find_core_assets(bbox, need_dem=False, need_landcover=True)
        return compute_landcover_percentages(assets["landcover"],
                                             {"type": "Feature", "properties": {},
                                              "geometry": geojson_geom})

    if indicator == "ndvi":
        return await compute_vegetation_index(
            bbox, geojson_geom,
            max_area_km2=(float("inf") if not enforce_budget else _synchronous_budget_km2(indicator)),
            resolution_m=chosen,
            start=window_start,
            end=window_end,
        )

    raise ValueError(f"unknown indicator {indicator!r}; choose from {('dem','landcover','ndvi','rainfall')}")


def plan_indicator(
    indicator: str,
    bbox_area_km2: float,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Optional[dict]:
    """What the worker will do for this area, and how long it should take.

    Cost drivers differ per indicator, so the plan differs: the raster modules are
    bounded by pixels at a policy resolution, while a vegetation series is bounded
    by the number of months it has to read, which barely moves with area.
    """
    if indicator == "rainfall":
        # A cache read plus a known-width ERA5 aggregate; effectively instant.
        return {
            "indicator": "rainfall",
            # Derived, not restated. It was a literal here while
            # rainfall.ERA5_GRID_DEGREES held the real value, which is the class
            # of duplication the registry exists to stop.
            "resolution_km": round(rainfall.ERA5_GRID_DEGREES * 111.32, 1),
            "reason": "a cached series, or one ERA5 read per year",
            "estimated_seconds": 60,
            "estimate_basis": "one annual ERA5 read per pass; cached areas are instant",
        }
    if indicator == "vegetation_series":
        import vegetation_series

        return vegetation_series.plan(bbox_area_km2, start or "2010-01-01", end)
    resolution, reason = resolution_for(bbox_area_km2, indicator)
    pixels = pixels_for(bbox_area_km2, resolution)
    # A raster pass is a handful of window reads; cost tracks block count, not pixel
    # count, so this is deliberately coarse and labelled an estimate.
    seconds = max(15, min(600, int(20 + pixels / 4000)))
    return {
        "indicator": indicator,
        "resolution_m": resolution,
        "reason": reason,
        "pixels_analysed": pixels,
        "estimated_seconds": seconds,
        "estimate_basis": "measured raster pass; cost tracks window reads, not pixels",
    }


def pixels_for(area_km2: float, resolution_m: int) -> int:
    """Pixels an area costs at a resolution, for reporting and for the budget."""
    return int(area_km2 * 1_000_000 / (resolution_m * resolution_m))
