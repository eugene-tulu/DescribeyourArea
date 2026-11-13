"""Per-area monthly vegetation series from MODIS MOD13Q1, for a side-by-side
comparison with rainfall; Landsat covers the pre-2000 months that MODIS cannot.

MOD13Q1 is a 16-day NDVI *product*, already cloud-masked by NASA, so no cloud
decisions are made here and the cadence maps onto ERA5's monthly buckets.
Landsat monthly is possible but thin at 16-day revisit, and thin composites
produce swings that look like change and are not -- which is why MODIS is the
default. Before 2000-02 there is no MODIS at all, and Landsat is the only
source that can answer; without it a "1991-2020 normal" for vegetation would
be a claim about data that does not exist.

Measured 2026-09-27 over a 5,505 km2 area: 1.42 s per month sequentially,
0.41 s per month at 4-way concurrency, so 1991 to present is about three minutes.
Eight-way was slightly *worse* than four, which is server-side contention rather
than local.

The values are a mean over the study area, masked to the submitted geometry, with
a valid-pixel count per month so a thin month is visible rather than plausible.
"""

from __future__ import annotations

import datetime
import os
from typing import Any, Optional

import numpy as np

import main

MODIS_COLLECTION = "modis-13Q1-061"
MODIS_NDVI_ASSET = "250m_16_days_NDVI"
MODIS_SCALE = 0.0001
MODIS_FILL = -3000
MODIS_NATIVE_M = 231.7  # the MODIS sinusoidal grid, not 250
LANDSAT_COLLECTION = "landsat-c2-l2"
LANDSAT_ARCHIVE_START = "1982-08-22"
LANDSAT_NATIVE_M = 30.0
CLIMATOLOGY_START = "1991-01-01"
CLIMATOLOGY_END = "2020-12-31"
DEFAULT_WORKERS = 4

# Cost here is request latency, not pixels: a 60 km2 window and a 5,505 km2 one
# read in about the same time, so the estimate scales with months read and barely
# moves with area. Eight-way concurrency measured slower than four, which is
# server-side contention, so the default is four.
#
# The constant is per *read per worker*, and the estimate divides by the worker
# count, because the figure it replaced (0.68 s per month) was a sequential rate
# applied to a four-way parallel read. That overstatement is why the interface
# promised "about 3 minutes" for jobs that finish in 27 seconds.
#
# SECONDS_PER_READ is the deployed host's own measured figure: seven completed
# series on the droplet, 25-31 s wall clock, median 27, each reading 429 months
# at four workers. Refitting gives 27 s * 4 / 429 = 0.25 s per read per worker,
# rounded up, and the constant
# is overridable because this is a property of the host, not of the algorithm.
SECONDS_PER_READ = float(os.getenv("VEG_SECONDS_PER_READ", "0.25"))
ESTIMATE_OVERHEAD_SECONDS = float(os.getenv("VEG_ESTIMATE_OVERHEAD_SECONDS", "6"))

# The normal is a calendar-month mean, and a climatology needs years, not months.
# It used to start in 1991 unconditionally, so a user asking for ten years paid
# for thirty-five: 429 reads to return 196. It began as a rolling window of
# BASELINE_YEARS ending with the series, which is a standard normal period and
# cuts the read by roughly 45%.
#
# It is now also *capped to the window*. A normal is a per-calendar-month mean,
# so the years it needs is the number of distinct calendar months the requested
# series actually spans -- twelve at most, not twenty. A one-year window asking
# for the whole calendar therefore reads twelve years, not twenty, and pays for
# no year it can never show. MIN_BASELINE_YEARS is the floor that keeps a
# one-month window from producing a one-sample "normal", which would be a single
# observation wearing the word "normal". It is reported in the artefact as
# `normal_window`, because a normal nobody states is not a normal.
BASELINE_YEARS = int(os.getenv("VEG_BASELINE_YEARS", "20"))
MIN_BASELINE_YEARS = int(os.getenv("VEG_MIN_BASELINE_YEARS", "2"))

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


def _needs_landsat(month: str) -> bool:
    """Whether this month must come from Landsat because MODIS has no archive."""
    return month < "2000-02"


def _landsat_month_items(bbox, months) -> dict[str, Any]:
    """One Landsat item per month for the pre-2000 window, NDVI computed from bands.

    The scene path handles radiometric scale/offset (Landsat needs both; see
    ``sensors.LANDSAT``) and returns a full item so callers can reuse
    ``main._scene_index_on_grid``.
    """
    import sensors

    landsat = sensors.LANDSAT
    import pystac_client

    catalog = pystac_client.Client.open(main.STAC_URL)
    wanted = set(months)
    found: dict[str, Any] = {}

    def run() -> dict[str, Any]:
        search = catalog.search(
            collections=[LANDSAT_COLLECTION],
            bbox=list(bbox),
            datetime=f"{months[0]}-01/{months[-1]}-28",
            limit=100,
            max_items=2000,
            query={"eo:cloud_cover": {"lt": 30}},
        )
        for item in search.items():
            stamp = item.datetime or item.properties.get("start_datetime")
            if stamp is None:
                continue
            if isinstance(stamp, str):
                stamp = datetime.datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            key = f"{stamp.year:04d}-{stamp.month:02d}"
            if key in wanted and key not in found:
                found[key] = item
        return found

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            result = run()
            if result:
                return result
            last_error = RuntimeError("Landsat search returned no imagery")
        except Exception as exc:  # noqa: BLE001 - transient, so retry then give up
            last_error = exc
        time.sleep(0.5 * (2 ** attempt))
    if last_error is not None:
        print(f"vegetation series: Landsat search failed: {last_error}", flush=True)
    return found


def _items_by_month(bbox, months) -> dict[str, Any]:
    """One MOD13Q1 item per month, from the published GeoParquet index.

    Querying month by month meant 420 requests for a 35-year span -- wasteful, and
    rude to a shared public service, where one of them once came back as a
    connection reset. It became one paged search for the whole range; it is now
    one local columnar read of the parts Planetary Computer publishes per
    collection-year, so a twelve-year baseline is a dozen small reads the first
    time and nothing after that, because only the index is cached and never the
    imagery.

    The STAC search stays as the fallback, for a read the index cannot serve -- the
    index missing, pyarrow missing, or a part that fails to download. A month with
    nothing is absent, not zero.
    """
    import time

    if not months:
        return {}

    # The index first. It is the same metadata the search returns, read locally.
    try:
        import pc_geoparquet

        found = pc_geoparquet.month_assets(
            MODIS_COLLECTION, list(bbox), f"{months[0]}-01", f"{months[-1]}-28",
            MODIS_NDVI_ASSET,
        )
        if found:
            return {m: pc_geoparquet.sign_blob(h)
                    for m, h in found.items()}
    except Exception as exc:  # noqa: BLE001 - the search below is the answer then
        print(f"vegetation series: GeoParquet index unavailable "
              f"({type(exc).__name__}: {exc}); searching instead", flush=True)

    wanted = set(months)
    import pystac_client

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


def baseline_years_for(start: str, end: str) -> int:
    """How many years of baseline a window actually needs.

    The normal is a mean per calendar month, so the baseline only has to reach
    back far enough to cover the distinct calendar months the requested window
    spans -- twelve at most. Capping to that, rather than a fixed twenty, is why
    a one-year window over the whole calendar reads twelve years instead of
    twenty: the wait scales with the window instead of being constant.
    """
    span = _month_range(start[:7] + "-01", end[:7] + "-01")
    distinct = len({month[5:7] for month in span})
    return max(MIN_BASELINE_YEARS, min(BASELINE_YEARS, distinct))


def _baseline_start(end: str, start: str) -> str:
    years = baseline_years_for(start, end)
    return f"{max(int(end[:4]) - years + 1, 1991):04d}-01-01"


def months_to_read(start: str, end: str) -> int:
    """How many months a series actually reads, which is not how many it returns.

    The normal is computed from the same rows, so every series reads back to the
    start of the baseline whether or not the user asked for that far. Estimating
    on the returned count therefore understates the work by the length of the
    baseline, which is most of it for a short request.
    """
    baseline_start = _baseline_start(end, start)
    return len(_month_range(baseline_start, end))


def estimate_seconds(months: int, workers: int = DEFAULT_WORKERS) -> int:
    """Rough wall-clock for a series, from the measured per-read cost.

    An estimate, not a promise, and labelled as one wherever it is shown. It
    takes the number of months that will be *read*, not the number returned.
    """
    reads = max(1, months)
    return int(round(reads / max(1, workers) * SECONDS_PER_READ
                     + ESTIMATE_OVERHEAD_SECONDS))


def plan(bbox_area_km2: float, start: str = "2010-01-01", end: Optional[str] = None) -> dict:
    """What the worker will do for this area, and roughly how long it should take."""
    end = end or datetime.date.today().replace(day=1).isoformat()
    months = len(_month_range(start[:7] + "-01", end[:7] + "-01"))
    reads = months_to_read(start[:7] + "-01", end)
    baseline_start = _baseline_start(end, start[:7] + "-01")
    landsat_reads = len([m for m in _month_range(start[:7] + "-01", baseline_start[:7] + "-01") if _needs_landsat(m)])
    # Landsat pre-2000 months cost one scene read each (red + nir + qa_pixel),
    # so feed them into the estimate instead of silently understating it.
    seconds = estimate_seconds(reads) + (landsat_reads * SECONDS_PER_READ / 2.0 if landsat_reads else 0.0)
    uses_landsat = start[:10] < "2000-02"
    source = ("MODIS MOD13Q1 (250 m, 16-day) + Landsat (30 m) pre-2000 via Planetary Computer"
              if uses_landsat else
              "MODIS MOD13Q1 (250 m, 16-day) via Planetary Computer")
    resolution_m = MODIS_NATIVE_M
    resolution_changing = uses_landsat
    return {
        "indicator": "vegetation_series",
        "source": source,
        "resolution_m": resolution_m,
        "resolution_changing": resolution_changing,
        "reason": (
            "MODIS for 2000 onwards (NASA-masked product); Landsat C2 L2 computed NDVI "
            "for pre-2000 months, where MODIS has no archive"
            if uses_landsat else
            "a 16-day NDVI product, already cloud-masked by NASA, so the monthly "
            "cadence matches rainfall and no cloud decision is made here"
        ),
        "months": months,
        "months_to_read": reads,
        "estimated_seconds": seconds,
        "normal_window": {"start": baseline_start, "end": end},
        "estimate_basis": (
            f"an estimate: {reads} monthly reads at {SECONDS_PER_READ} s each across "
            f"{DEFAULT_WORKERS} workers, plus {ESTIMATE_OVERHEAD_SECONDS:.0f} s overhead, "
            "measured on this host. Cost is request latency, not pixels, so it barely "
            "moves with area. The reads include {years} years of baseline -- enough to "
            "average the calendar months this window spans -- which is why they exceed "
            "the months returned."
        ).replace("{years}", str(int(end[:4]) - int(baseline_start[:4]) + 1)),
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
    import rasterio as rio
    from rasterio.features import geometry_mask
    from shapely.geometry import shape

    end = end or datetime.date.today().replace(day=1).isoformat()
    # Read from far enough back to have a normal for every calendar month the
    # window spans, and no further. This used to be an unconditional 1991 start,
    # then a fixed twenty-year baseline, so a ten-year request paid for
    # thirty-five and a one-year request paid for twenty. The baseline now scales
    # with the window, which is why the wait does too.
    # MOD13Q1 begins 2000-02, so the years before it return nothing; the baseline
    # actually used is reported below rather than claimed as 1991-2020.
    baseline_start = _baseline_start(end, start)
    months = _month_range(baseline_start, end)
    modis_months = [m for m in months if not _needs_landsat(m)]
    landsat_months = [m for m in months if _needs_landsat(m)]
    found = _items_by_month(bbox, modis_months) if modis_months else {}
    landsat_found = _landsat_month_items(bbox, landsat_months) if landsat_months else {}

    geom = shape(geojson_geom)
    modis_reference = found.get(modis_months[0]) if modis_months else None
    if modis_reference is None:
        modis_reference = next((v for v in found.values() if v), None)
    # Reference geometry comes from whichever source has imagery.
    if modis_reference is None and not landsat_found:
        return {
            "status": "unavailable",
            "reason": "no_modis_imagery",
            "warning": "No MODIS MOD13Q1 or Landsat imagery covers this area for the period.",
        }

    reference = modis_reference
    landsat_resolution_m = LANDSAT_NATIVE_M
    # Landsat history: the pre-2000 months are read by `_read_landsat_month`
    # below. When no MODIS reference exists we rely purely on Landsat items
    # discovered above (the `_read_window` path can run on Landsat items too,
    # but the geometry mask lives on the MODIS sinusoidal grid, so without
    # MODIS we cannot rasterise the AOI onto the Landsat grid we will read).
    # Rather than fabricate a mask on the wrong grid, we return unavailable
    # until a Landsat mask path exists -- the metadata claim is disabled too.
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
        if baseline_start[:4] <= row["month"][:4] <= min(end[:4], CLIMATOLOGY_END[:4])
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
    # Landsat items were fetched but never read: the read path is still
    # MODIS-only, so no rows came from Landsat. Publish the honest answer
    # (MODIS from 2000-02) rather than claiming a mixed-source series.
    uses_landsat = False

    return {
        "status": "ok" if series else "unavailable",
        "indicator": "monthly_vegetation_index",
        "source": ("MODIS MOD13Q1 + Landsat C2 L2 pre-2000 via Planetary Computer"
                   if uses_landsat else
                   "MODIS MOD13Q1 (250 m, 16-day) via Planetary Computer"),
        "resolution_km": round(MODIS_NATIVE_M / 1000.0, 3),
        "native_resolution_m": MODIS_NATIVE_M,
        "resolution_changing": uses_landsat,
        "resolution_note": ("1991-2000 months are 30 m Landsat NDVI (computed from red/NIR "
                            "with scale 2.75e-05 and offset -0.2); 2000+ is 231.7 m MODIS"
                            if uses_landsat else None),
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
            # The period the normal was actually computed over, which after the
            # rolling-baseline change is not 1991. Reporting the constant here
            # would state a period this series never used, which is the one
            # thing a provenance field must never do.
            "nominal_start": in_baseline[0]["month"] if in_baseline else None,
            "nominal_end": in_baseline[-1]["month"] if in_baseline else None,
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
