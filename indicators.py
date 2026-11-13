"""Compute one indicator for one area, wherever it is going to run.

A submission can be answered in two places, and they must not disagree: the
synchronous request path, which refuses a large area, and the worker, which
computes it at a resolution chosen for the area rather than refused. Both call
``compute_indicator``, so a queued area and an interactive one produce the same
shape, the same provenance and the same evidence.
"""

from __future__ import annotations

import datetime
from typing import Any, Optional

import rainfall
import sensors

# How many years of vegetation history a series covers when the caller does not
# ask for a specific window. MOD13Q1 begins in 2000-02, so 2010 leaves two
# decades of record without pretending to more.
DEFAULT_SERIES_YEARS = 16

# The io-lulc-annual-v02 published extent. A window outside it is clamped to
# the latest/earliest published year rather than producing an empty series.
# The job readiness check uses the same clamp, so "ready" and "computed" cannot
# disagree about which years a request means.
LANDCOVER_EXTENT = (2017, 2023)


def landcover_year_range(start: Optional[str], end: Optional[str]) -> tuple[int, int]:
    """The inclusive years the land-cover history worker will compute.

    A 2025-2026 request clamps to 2023-2023; a 2010-2026 request clamps to
    2017-2023; a 2017-2020 request stays 2017-2020.
    """
    try:
        start_year = int((start or "")[:4])
    except ValueError:
        start_year = int((end or "")[:4]) - 6 if end else LANDCOVER_EXTENT[0]
    try:
        end_year = int((end or "")[:4])
    except ValueError:
        end_year = start_year + 6
    start_year = max(start_year, LANDCOVER_EXTENT[0])
    end_year = min(end_year, LANDCOVER_EXTENT[1])
    if end_year < start_year:
        start_year = end_year
    if end_year - start_year > 10:
        end_year = start_year + 10
    return start_year, end_year


def resolution_for(bbox_area_km2: float, indicator: str) -> tuple[int, str]:
    """Metres per pixel for this indicator at this area, and the reason."""
    native = {"landcover": 10, "ndvi": 10, "dem": 30}.get(indicator, 10)
    return sensors.resolution_for_area(bbox_area_km2, native_m=native)


def _synchronous_budget_km2(indicator: str) -> float:
    """Largest area this module will answer inside a request.

    Land cover is the only module whose memory genuinely grows with area, so it
    gets a larger budget than the rest. The figures come from measurement, not
    preference: io-lulc-annual-v02 is 235 MB at 1,000 km2 and 1,205 MB at 5,500 km2.
    """
    # Read the definitions rather than re-deriving them from the environment.
    # Both were read here independently, which meant the defaults lived in two
    # places and could drift apart without anything failing.
    import main

    return (main.MAX_LANDCOVER_BBOX_KM2 if indicator == "landcover"
            else main.MAX_SYNC_BBOX_KM2)


async def _compute_landcover_history(
    bbox: list[float],
    geojson_geom: dict,
    window_start: Optional[str] = None,
    window_end: Optional[str] = None,
) -> dict:
    """Per-year land-cover classes for a window, as a `series` of years."""
    import datetime as dt

    from main import (
        _bbox_area_km2,
        _find_core_assets,
        _resolve_landcover_year,
        compute_landcover_percentages,
    )

    end = window_end or dt.date.today().isoformat()
    start = window_start or f"{dt.date.fromisoformat(end[:10]).year - 6:04d}-01-01"
    start_year, end_year = landcover_year_range(start, end)
    # Read each annual map at the AOI's resolution policy (250 m for a large
    # landscape) through the COG overview, not at native 10 m.
    resolution = resolution_for(_bbox_area_km2(bbox), "landcover")[0]

    series: list[dict[str, Any]] = []
    for year in range(start_year, end_year + 1):
        assets = await _find_core_assets(
            bbox, need_dem=False, need_landcover=True,
            landcover_year=_resolve_landcover_year(start=f"{year:04d}-01-01", end=f"{year:04d}-12-31"),
        )
        used_year = assets.get("_landcover_year", year)
        if used_year != year:
            continue
        if "_landcover_year" in assets and assets["_landcover_year"] != year:
            continue
        result = compute_landcover_percentages(
            assets["landcover"],
            {"type": "Feature", "properties": {}, "geometry": geojson_geom},
            year=year,
            resolution_m=resolution,
        )
        if result.get("error"):
            continue
        series.append({
            "month": f"{year:04d}-01",
            "year": year,
            "classes": result["classes"],
            "dominant_class": result["dominant_class"],
            "dominant_percentage": result["dominant_percentage"],
        })
    if not series:
        return {"status": "unavailable", "reason": "no_landcover_years",
                "message": "No Esri 10 m Land Cover maps were available for this area and period."}
    return {
        "status": "ok",
        "indicator": "landcover_history",
        "source": "Esri 10 m Land Cover (Impact Observatory, Sentinel-2)",
        "collection": "io-lulc-annual-v02",
        "resolution_m": resolution,
        "window": {"start": f"{series[0]['year']:04d}-01-01",
                   "end": f"{series[-1]['year']:04d}-12-31"},
        # What the caller asked for, so readiness can tell a series computed for
        # this window from one computed for a narrower one, even when a year is
        # absent because it had no imagery.
        "requested_window": {"start": start, "end": end},
        "series": series,
        "source_collection": "io-lulc-annual-v02",
    }


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
    year: Optional[int] = None,
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
        compute_landcover_percentages,
        compute_raster_stats,
        compute_vegetation_index,
        interpret_terrain,
    )

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
        stats = compute_raster_stats(
            assets["dem"], {"type": "Feature", "properties": {}, "geometry": geojson_geom},
            resolution_m=chosen,
        )
        return interpret_terrain(stats)

    if indicator == "landcover":
        assets = await _find_core_assets(
            bbox, need_dem=False, need_landcover=True,
            landcover_year=year,
        )
        used_year = assets.get("_landcover_year", year)
        return compute_landcover_percentages(
            assets["landcover"],
            {"type": "Feature", "properties": {}, "geometry": geojson_geom},
            year=used_year,
            resolution_m=chosen,
        )

    if indicator == "landcover_history":
        return await _compute_landcover_history(bbox, geojson_geom, window_start, window_end)

    if indicator == "ndvi":
        return await compute_vegetation_index(
            bbox, geojson_geom,
            max_area_km2=(float("inf") if not enforce_budget else _synchronous_budget_km2(indicator)),
            resolution_m=chosen,
            start=window_start,
            end=window_end,
        )

    raise ValueError(f"unknown indicator {indicator!r}; choose from {('dem','landcover','ndvi','rainfall')}")


def plan_indicator(    indicator: str,
    bbox_area_km2: float,
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    product: Optional[str] = None,
) -> Optional[dict]:
    """What the worker will do for this area, and how long it should take.

    Cost drivers differ per indicator, so the plan differs: the raster modules are
    bounded by pixels at a policy resolution, while a series is bounded by the
    number of months it has to read, which barely moves with area.

    ``product`` selects which precipitation product is being planned for, because
    the two are not remotely the same cost: ERA5 reads its whole series from one
    asset, CHIRPS is a separate object per month. A CHIRPS job planned with
    ERA5's cost promised 24 seconds and took 17 minutes.
    """
    if indicator == "rainfall":
        return _plan_rainfall(indicator, start, end, product)
    if indicator == "chirps":
        return _plan_rainfall(indicator, start, end, product)
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


def _plan_rainfall(indicator: str, start: Optional[str], end: Optional[str],
                   product: Optional[str]) -> dict:
    """The plan for a precipitation series, at the cost of the product asked for."""
    import registry

    # Two things can name the product -- the indicator and an explicit argument --
    # and they can disagree. The indicator wins when it *is* a product, because a
    # caller that asks for "chirps" plainly means CHIRPS.
    named = (product or "").strip().lower()
    if indicator in ("chirps",):
        named = indicator
    key = named or "rainfall"
    entry = registry.get(key) or registry.RAINFALL
    if entry is None:
        entry = registry.RAINFALL
    per_month = entry.seconds_per_month or 0.12
    months = len(_months_between(start or "2010-01-01", end or _this_month()))

    if key == "rainfall":
        resolution_km = round(rainfall.ERA5_GRID_DEGREES * 111.32, 1)
        reason = "a cached series, or one ERA5 read for the whole period"
    else:
        resolution_km = entry.resolution_km()
        reason = f"a cached series, or one CHIRPS object per month at 0.05°"

    return {
        "indicator": indicator,
        "product": entry.key,
        "resolution_km": resolution_km,
        "reason": reason,
        "months": months,
        "estimated_seconds": int(round(months * per_month)),
        "estimate_basis": (
            f"{months} months at {per_month:g} s each, measured for {entry.label}. "
            f"Cached areas are instant."
        ),
    }


def _this_month() -> str:
    import datetime

    today = datetime.datetime.now(datetime.UTC).date()
    return today.replace(day=1).isoformat()


def _months_between(start: str, end: str) -> list:
    import vegetation_series

    return vegetation_series._month_range(start[:7] + "-01", end)


def pixels_for(area_km2: float, resolution_m: int) -> int:
    """Pixels an area costs at a resolution, for reporting and for the budget."""
    return int(area_km2 * 1_000_000 / (resolution_m * resolution_m))
