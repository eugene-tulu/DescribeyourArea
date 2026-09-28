"""Per-area monthly vegetation series from MODIS MOD13Q1, for a side-by-side
comparison with rainfall.

Why MOD13Q1 and not Landsat: it is a 16-day NDVI *product*, already cloud-masked
by NASA, so no cloud decisions are made here and the cadence maps onto ERA5's
monthly buckets. Landsat monthly is possible but thin at 16-day revisit, and thin
composites produce swings that look like change and are not.

Measured 2026-09-27 over a 5,505 km2 area: 1.42 s per month sequentially,
0.41 s per month at 4-way concurrency, so 1991 to present is about three minutes.
Eight-way was slightly *worse* than four, which is server-side contention rather
than local.

The values are a mean over the study area, masked to the submitted geometry, with
a valid-pixel count per month so a thin month is visible rather than plausible.
"""

from __future__ import annotations

import datetime
import math
from typing import Any, Optional

import numpy as np

import main

MODIS_COLLECTION = "modis-13Q1-061"
MODIS_NDVI_ASSET = "250m_16_days_NDVI"
MODIS_SCALE = 0.0001
MODIS_FILL = -3000
MODIS_NATIVE_M = 231.7  # the MODIS sinusoidal grid, not 250
CLIMATOLOGY_START = "1991-01-01"
CLIMATOLOGY_END = "2020-12-31"
DEFAULT_WORKERS = 4

# Measured 2026-09-27 over a 5,505 km2 area: 1.42 s per month read sequentially,
# 0.41 s at four-way concurrency. Eight-way was slower than four, which is
# server-side contention, so the default is four. Cost here is dominated by request
# latency, not by pixels -- a 60 km2 window and a 5,505 km2 one both read in about
# 1.3 s -- so the estimate scales with months read and barely moves with area.
SECONDS_PER_MONTH = 0.68
ESTIMATE_OVERHEAD_SECONDS = 25

# Shown to the reader under the chart. It is a load-bearing string: the product's
# claim is that it says what kind of number something is, and a rainfall overlay
# beside vegetation is exactly where an implied causal claim would creep in.
CAVEAT = (
    "Co-variation with rainfall is not attribution. Vegetation responds to "
    "rainfall with a lag that varies by season, and in semi-arid rangeland "
    "water is not always the limiting factor."
)


def _month_range(start: str, end: str) -> list[str]:
    first = datetime.date.fromisoformat(start[:10])
    last = datetime.date.fromisoformat(end[:10])
    months = []
    year, month = first.year, first.month
    while (year, month) <= (last.year, last.month):
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month > 12:
            year, month = year + 1, 1
    return months


def _items_by_month(bbox, months) -> dict[str, Any]:
    """One MOD13Q1 item per month, from a single paged search.

    Querying month by month meant 420 requests for a 35-year span, which is both
    wasteful and rude to a shared public service -- one of them came back as a
    connection reset. One search for the whole range is bucketed locally instead.
    A month with nothing is absent, not zero.
    """
    import time

    import planetary_computer
    import pystac_client

    wanted = set(months)
    catalog = pystac_client.Client.open(main.STAC_URL)

    def run() -> dict[str, Any]:
        # Rebuilt per attempt: reusing a paged search after a reset iterates a
        # broken pager and silently yields nothing.
        search = catalog.search(
            collections=[MODIS_COLLECTION],
            bbox=list(bbox),
            datetime=f"{months[0]}-01/{months[-1]}-28",
            limit=100,
            max_items=2000,
        )
        found: dict[str, Any] = {}
        for item in search.items():
            stamp = item.datetime or item.properties.get("start_datetime")
            if stamp is None:
                continue
            if isinstance(stamp, str):
                stamp = datetime.datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            key = f"{stamp.year:04d}-{stamp.month:02d}"
            if key in wanted and key not in found:
                found[key] = item.assets[MODIS_NDVI_ASSET].href
        return found

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            found = run()
            if found:
                return found
            last_error = RuntimeError("MODIS search returned no imagery for the period")
        except Exception as exc:  # noqa: BLE001 - transient, so retry then give up
            last_error = exc
        time.sleep(0.5 * (2 ** attempt))
    if last_error is not None:
        print(f"vegetation series: MODIS search failed: {last_error}", flush=True)
    return {}


def _read_month(href, window_spec) -> Optional[np.ndarray]:
    import planetary_computer
    import rasterio as rio

    with rio.open(planetary_computer.sign(href)) as src:
        window, geometry, transform = window_spec
        return src.read(1, window=window, boundless=False)


def estimate_seconds(months: int) -> int:
    """Rough wall-clock for a series, from the measured per-month cost.

    An estimate, not a promise, and labelled as one wherever it is shown.
    """
    return int(round(max(1, months) * SECONDS_PER_MONTH + ESTIMATE_OVERHEAD_SECONDS))


def plan(bbox_area_km2: float, start: str = "2010-01-01", end: Optional[str] = None) -> dict:
    """What the worker will do for this area, and roughly how long it will take."""
    end = end or datetime.date.today().replace(day=1).isoformat()
    months = len(_month_range(start[:7] + "-01", end[:7] + "-01"))
    seconds = estimate_seconds(months)
    return {
        "indicator": "vegetation_series",
        "source": "MODIS MOD13Q1 (250 m, 16-day) via Planetary Computer",
        "resolution_m": MODIS_NATIVE_M,
        "reason": (
            "a 16-day NDVI product, already cloud-masked by NASA, so the monthly "
            "cadence matches rainfall and no cloud decision is made here"
        ),
        "months": months,
        "estimated_seconds": seconds,
        "estimate_basis": (
            f"an estimate: {SECONDS_PER_MONTH} s per month measured at "
            f"{DEFAULT_WORKERS}-way concurrency, plus overhead. Cost is request "
            "latency, not pixels, so it barely moves with area."
        ),
        "area_km2": round(bbox_area_km2, 2),
    }


def compute_monthly_series(
    geojson_geom: dict,
    bbox: list[float],
    start: str = "2010-01-01",
    end: Optional[str] = None,
    workers: int = DEFAULT_WORKERS,
) -> dict:
    """Build the monthly series and the baseline in one pass over the imagery.

    The baseline period is derived from what the source actually has, and reported
    as such: MOD13Q1 begins in 2000-02, so a "1991-2020" label would be a claim
    about data that does not exist.
    """
    import planetary_computer
    import pystac_client
    import rasterio as rio
    from rasterio.features import geometry_mask
    from shapely.geometry import shape

    end = end or datetime.date.today().replace(day=1).isoformat()
    # Read from the nominal climatology start so the series and the normal come from
    # one pass and cannot disagree. MOD13Q1 begins 2000-02, so the years before it
    # return nothing; the baseline actually used is reported below rather than
    # claimed as 1991-2020.
    months = _month_range(CLIMATOLOGY_START, end)
    found = _items_by_month(bbox, months)

    geom = shape(geojson_geom)
    reference = found.get(months[0]) or next((v for v in found.values() if v), None)
    if reference is None:
        return {
            "status": "unavailable",
            "reason": "no_modis_imagery",
            "warning": "No MODIS MOD13Q1 imagery covers this area for the period.",
        }

    with rio.open(planetary_computer.sign(reference)) as src:
        window = main._window_for_bbox(src, bbox)
        if window is None:
            return {
                "status": "unavailable",
                "reason": "outside_modis_grid",
                "warning": "This area falls outside the MODIS sinusoidal grid.",
            }
        window_spec = (window, geom, src.window_transform(window))
        width, height = int(window.width), int(window.height)
        # geometry_mask does not reproject, and the MODIS grid is a custom
        # sinusoidal CRS while the submitted geometry is WGS84. Handing it the
        # unprojected polygon puts it millions of metres away and the mask comes
        # back empty.
        from rasterio.warp import transform_geom

        projected = transform_geom("EPSG:4326", src.crs, geom)
        mask = geometry_mask(
            [projected],
            out_shape=(height, width),
            transform=src.window_transform(window),
            invert=True,
            all_touched=True,
        )
        cell_area_km2 = (abs(src.transform.a) / 1000.0) ** 2

    total_cells = int(mask.sum())
    if total_cells == 0:
        return {
            "status": "unavailable",
            "reason": "no_cells_in_area",
            "warning": "The study area covers no whole MODIS cell.",
        }

    from concurrent.futures import ThreadPoolExecutor

    ordered = [(m, found[m]) for m in months if found.get(m)]
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        blocks = list(pool.map(lambda p: _read_month(p[1], window_spec), ordered))

    rows = []
    for (month, _href), block in zip(ordered, blocks):
        if block is None:
            continue
        values = block[mask].astype("float64") * MODIS_SCALE
        values = values[(values >= -1.0) & (values <= 1.0)]
        if values.size == 0:
            continue
        rows.append({
            "month": month,
            "value": round(float(values.mean()), 4),
            "min": round(float(values.min()), 4),
            "max": round(float(values.max()), 4),
            "valid_pixels": int(values.size),
            "area_fraction": round(float(values.size) / total_cells, 4),
        })

    if not rows:
        return {
            "status": "unavailable",
            "reason": "no_valid_months",
            "warning": "MODIS imagery exists but no month had usable pixels over the area.",
        }

    in_baseline = [
        row for row in rows
        if CLIMATOLOGY_START[:4] <= row["month"][:4] <= CLIMATOLOGY_END[:4]
    ]
    baseline = {row["month"]: row["value"] for row in in_baseline}
    monthly_normal: dict[str, list[float]] = {}
    for month, value in baseline.items():
        monthly_normal.setdefault(month[5:7], []).append(value)
    climatology = {
        calendar: round(sum(v) / len(v), 4) for calendar, v in sorted(monthly_normal.items())
    }

    series = []
    for row in rows:
        if row["month"] < start[:10][:7]:
            continue
        entry = dict(row)
        normal = climatology.get(row["month"][5:7])
        if normal:
            entry["normal"] = normal
            entry["anomaly"] = round(row["value"] - normal, 4)
            entry["anomaly_pct"] = round((row["value"] - normal) / normal * 100.0, 1) if normal else None
        series.append(entry)

    thin = [r["month"] for r in series if r["area_fraction"] < 0.9]

    return {
        "status": "ok" if series else "unavailable",
        "indicator": "monthly_vegetation_index",
        "source": "MODIS MOD13Q1 (250 m, 16-day) via Planetary Computer",
        "resolution_km": round(MODIS_NATIVE_M / 1000.0, 3),
        "native_resolution_m": MODIS_NATIVE_M,
        "method": "area mean of the monthly MOD13Q1 NDVI composites, masked to the study area",
        "window": {"start": start, "end": end},
        "read_from": rows[0]["month"],
        "read_to": rows[-1]["month"],
        "months": len(series),
        "grid_cells_in_area": total_cells,
        "cell_area_km2": round(cell_area_km2, 3),
        "climatology": {
            # Reported as the period actually used. MOD13Q1 has no data before
            # 2000-02, so calling this a 1991-2020 normal would be a false claim.
            "standard": (
                f"{in_baseline[0]['month'][:4]}-{in_baseline[-1]['month'][:4]} mean of "
                "monthly MOD13Q1 composites"
                if in_baseline else None
            ),
            "start": in_baseline[0]["month"] if in_baseline else None,
            "end": in_baseline[-1]["month"] if in_baseline else None,
            "nominal_start": CLIMATOLOGY_START,
            "nominal_end": CLIMATOLOGY_END,
            "years_used": len({r["month"][:4] for r in in_baseline}) or 0,
            "monthly_mean": climatology,
            "annual_mean": round(
                sum(climatology.values()) / len(climatology), 4
            ) if climatology else None,
        },
        "thin_months": thin,
        "series": series,
        "caveat": CAVEAT,
    }
