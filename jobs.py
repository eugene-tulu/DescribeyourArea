"""Durable jobs for self-service rainfall submission.

A submission is a request to compute a series, and computing one takes about 60
seconds of ERA5 reads. That is longer than a request should take, so the submit
path only records a job and returns; the work happens in a runner.

The job record is what makes this durable rather than a promise: it lives on disk,
so a submission survives a restart, and a runner picks up whatever is still
pending. The runner is idempotent, because an area that already has a series is
skipped rather than recomputed.

There is no supervision here. ``worker.py`` is the runner, and running it from
cron or a sidecar is the difference between "usually works" and "reliably works".
Nothing in the design depends on where it runs.
"""

from __future__ import annotations

import asyncio
import datetime
import json
import os
from pathlib import Path
from typing import Callable, Optional

import rainfall

# Job states. Terminal states are ready, failed and cancelled.
PENDING = "pending"
RUNNING = "running"
READY = "ready"
FAILED = "failed"

TERMINAL = {READY, FAILED}

# A submission queue with no worker behind it will grow without bound, and every
# queued job is a future ERA5 read. This bounds it per host.
MAX_PENDING_JOBS = int(os.getenv("RAINFALL_MAX_PENDING_JOBS", "25"))

# How long a running job may hold its claim before another runner may take it.
# Generous, because a long series is a legitimate multi-minute read: the longest
# work here is a MODIS series at roughly 0.7 s a month, so even a 30-year span
# is well inside this. It exists to recover from a dead worker, not to police a
# slow one.
JOB_LEASE_SECONDS = float(os.getenv("RAINFALL_JOB_LEASE_SECONDS", "1800"))

JOB_VERSION = 4

# What a job computes. Rainfall is a cheap cache read; the others read rasters,
# which is why they need a resolution and why they are the reason a large area is
# queued rather than refused.
INDICATORS = ("rainfall", "dem", "landcover", "landcover_history", "ndvi", "vegetation_series")
RUNTIME_INDICATORS = ("dem", "landcover", "landcover_history", "ndvi", "vegetation_series")


def artefact_path(key: str, indicator: str) -> Path:
    """Where one indicator's result for one area lives.

    One directory per indicator, so adding a second indicator cannot collide with
    the first and each can be invalidated on its own version.
    """
    return rainfall.cache_dir() / indicator / f"{key}.json"


def read_artefact(key: str, indicator: str) -> Optional[dict]:
    path = artefact_path(key, indicator)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    return payload


def write_artefact(key: str, indicator: str, payload: dict) -> None:
    path = artefact_path(key, indicator)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=1, sort_keys=True))
    temporary.replace(path)


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def _lease_expired(job: dict) -> bool:
    """Whether a running job has held its claim for longer than the lease.

    An unreadable or absent timestamp counts as expired: a record that cannot say
    when it started cannot justify holding a queue slot open forever.
    """
    stamp = job.get("started_at")
    if not stamp:
        return True
    try:
        started = datetime.datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
    except ValueError:
        return True
    if started.tzinfo is None:
        started = started.replace(tzinfo=datetime.UTC)
    return (datetime.datetime.now(datetime.UTC) - started).total_seconds() > JOB_LEASE_SECONDS


def jobs_dir() -> Path:
    return rainfall.cache_dir() / "jobs"


def job_path(key: str, indicator: str = "rainfall") -> Path:
    """One file per (area, indicator).

    Keyed by area alone, submitting dem and ndvi for the same boundary collapsed
    into a single record and the second submission silently replaced the first.
    """
    return jobs_dir() / indicator / f"{key}.json"


def read_job(key: str, indicator: str = "rainfall") -> Optional[dict]:
    path = job_path(key, indicator)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def write_job(job: dict) -> None:
    path = job_path(job["cache_key"], job.get("indicator") or "rainfall")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(job, indent=1, sort_keys=True))
    temporary.replace(path)


def list_jobs(state: Optional[str] = None) -> list[dict]:
    directory = jobs_dir()
    if not directory.exists():
        return []
    found = []
    for path in sorted(directory.rglob("*.json")):
        try:
            job = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if state is None or job.get("state") == state:
            found.append(job)
    found.sort(key=lambda j: j.get("submitted_at") or "")
    return found


def pending_count() -> int:
    return len(list_jobs(PENDING))


def status_for(key: str, indicator: str = "rainfall",
               start: Optional[str] = None, end: Optional[str] = None,
               year: Optional[int] = None) -> dict:
    """What a caller should be told about one study area.

    The cache is authoritative: if a series exists, the area is ready whatever the
    job record says, because a runner may have completed without updating the
    record. But "exists" is not "covers the requested window": a 1-year series
    must not answer for a 5-year caller, so the window is part of the check.
    """
    if indicator == "rainfall" and _product_ready(key, _job_product(key), start, end):
        return {"state": READY, "cache_key": key, "source": "cache"}
    if indicator != "rainfall" and _artefact_ready(key, indicator, start, end, year):
        return {"state": READY, "cache_key": key, "indicator": indicator,
                "source": "artefact"}
    job = read_job(key, indicator)
    if job is None:
        return {"state": "not_submitted", "cache_key": key, "indicator": indicator}
    if job.get("state") == READY and indicator == "rainfall" and rainfall.read_cache(key) is None:
        # The record says ready but the series is gone, for instance after a
        # deletion. Report what is true rather than what was recorded.
        return {"state": "not_submitted", "cache_key": key}
    # A ready record whose coverage does not span this caller's window must
    # not be reported ready for them -- it is the old window's answer, not
    # this one's. Report not_submitted so a wider read is queued.
    if job.get("state") == READY:
        if indicator == "rainfall" and not _product_ready(key, _job_product(key), start, end):
            return {"state": "not_submitted", "cache_key": key,
                    "reason": "stored window does not cover the requested window"}
        if indicator != "rainfall" and not _artefact_ready(key, indicator, start, end, year):
            return {"state": "not_submitted", "cache_key": key, "indicator": indicator,
                    "reason": "stored window does not cover the requested window"}
    return {
        "state": job.get("state", PENDING),
        "cache_key": key,
        "indicator": job.get("indicator", "rainfall"),
        "submitted_at": job.get("submitted_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "attempts": job.get("attempts", 0),
        "reason": job.get("reason"),
    }


def _window_covers_payload(payload: dict, start: Optional[str],
                            end: Optional[str]) -> bool:
    """Whether a stored payload's window covers the requested start/end.

    Rainfall and the monthly vegetation series are window-valued: a series
    computed for one window does not answer for another. ``dem`` and
    ``landcover`` have no time dimension and always cover.

    A payload with no recorded window (pre-portfolio payloads, or a window-len
    stub) is treated as covering: the records predate the window field and
    must not start recomputing on every request. Only an explicit, narrower
    stored window blocks a wider request.
    """
    if not start and not end:
        return True
    win = payload.get("window") or {}
    have_start = (win.get("start") or payload.get("read_from") or "")[:7]
    have_end = (win.get("end") or payload.get("read_to") or "")[:7]
    want_start = (start or "")[:7]
    want_end = (end or "")[:7]
    if not want_start and not want_end:
        return True
    if not have_start or not have_end:
        # No recorded window: cannot prove it does not cover, so do not
        # block ready on it. Window-valued payloads always record one;
        # this branch is the dem/landcover and legacy-payload path.
        months = [r.get("month", "")[:7] for r in (payload.get("series") or [])
                  if isinstance(r, dict) and r.get("month")]
        if not months:
            return True  # window-less payloads (dem/landcover) always cover
        # A series with months but no recorded window is ambiguous. A wider
        # request that merely contains its months must not block, but neither
        # may a stub with a single 2020 row answer for a 2010 request: the
        # requested months outside the series are unknown, so treat the stub
        # as covering only if the request is within its months. This keeps
        # legacy stubs from answering for windows they do not span.
        return (not want_start or min(months) <= want_start) \
            and (not want_end or max(months) >= want_end)
    return (not want_start or have_start <= want_start) \
        and (not want_end or have_end >= want_end)


def _product_ready(key: str, product: Optional[str],
                   start: Optional[str] = None, end: Optional[str] = None) -> bool:
    """Whether a series for this area *and product* exists for this window.

    Reading the ERA5 cache for a CHIRPS job reports ready off the wrong series,
    and the reader is then handed a reanalysis while its evidence label claims a
    satellite-gauge blend.
    """
    payload = rainfall.read_cache(key, product or "rainfall")
    if not payload:
        return False
    stored = payload.get("product")
    if stored is not None and (product or "rainfall") != stored:
        return False
    # A stored series that does not span the requested window must not answer
    # for it: a 1-year series served for a 5-year window silently returns the
    # 1-year one under the 5-year label. Report not ready so a wider read is
    # queued. Window-less checks keep the old call shape (no window = any
    # stored series answers).
    return _window_covers_payload(payload, start, end)


def _published_landcover_year(start: Optional[str], end: Optional[str],
                              year: Optional[int] = None) -> Optional[int]:
    """The land-cover map year a request should be answered with.

    A window that extends past the published extent resolves to the extent's
    latest year, not to the window end. Comparing the raw start year (2025) to
    the artefact's year (2023) was what left a ready artefact reported as
    not_submitted forever.
    """
    if year is not None:
        return year
    from indicators import LANDCOVER_EXTENT

    try:
        end_year = int((end or start or "")[:4])
    except ValueError:
        end_year = LANDCOVER_EXTENT[1]
    return max(LANDCOVER_EXTENT[0], min(LANDCOVER_EXTENT[1], end_year))


def artefact_covers(payload: Optional[dict], indicator: str,
                    start: Optional[str] = None, end: Optional[str] = None,
                    year: Optional[int] = None) -> bool:
    """Whether a stored artefact answers the requested window/year.

    Land cover is year-valued: a 2023 map answers a 2025-2026 request because
    2023 is the latest published year. History is a series: it is ready when
    the request that produced it covers the same clamped years this request
    needs. Using the raw requested start year for either made a ready artefact
    look like it needed recomputing on every poll.
    """
    if payload is None:
        return False
    if indicator == "dem":
        return True  # no time dimension
    if indicator == "landcover":
        want = _published_landcover_year(start, end, year)
        have = payload.get("year")
        if have is None:
            win = payload.get("window") or {}
            try:
                have = int((win.get("start") or "")[:4])
            except ValueError:
                have = None
        return have is not None and want is not None and have == want
    if indicator == "landcover_history":
        from indicators import landcover_year_range

        try:
            want_lo, want_hi = landcover_year_range(start, end)
        except Exception:
            return _window_covers_payload(payload, start, end)

        # A payload records the window it was asked to compute. If that request
        # already covered the years this caller needs, it is ready even when an
        # individual year had no data and is absent from the series. Falling
        # back to the series bounds alone would recompute forever for an area
        # whose last requested year has no imagery.
        requested = payload.get("requested_window") or {}
        if requested.get("start") or requested.get("end"):
            try:
                have_lo, have_hi = landcover_year_range(
                    requested.get("start"), requested.get("end"))
                if have_lo <= want_lo and have_hi >= want_hi:
                    return True
            except Exception:
                pass

        win = payload.get("window") or {}
        try:
            have_lo = int((win.get("start") or "")[:4])
            have_hi = int((win.get("end") or "")[:4])
        except ValueError:
            return False
        years = set()
        for entry in payload.get("series") or []:
            if isinstance(entry, dict):
                try:
                    years.add(int(entry.get("year")))
                except (TypeError, ValueError):
                    pass
        return have_lo <= want_lo and have_hi >= want_hi and want_hi in years
    return _window_covers_payload(payload, start, end)


def _artefact_ready(key: str, indicator: str, start: Optional[str] = None,
                    end: Optional[str] = None, year: Optional[int] = None) -> bool:
    """Whether an artefact for this area covers the requested window/year.

    A failed artefact (`status: unavailable`) is not ready even when its window
    is window-less and therefore "covers": submit used to report it ready while
    /context correctly rejected it, so the reader was told NDVI was already
    computed and then shown it missing.
    """
    artefact = read_artefact(key, indicator)
    if not artefact or artefact.get("status") not in ("ok", "ready"):
        return False
    return artefact_covers(artefact, indicator, start, end, year)


def _job_product(key: str) -> Optional[str]:
    """Which product the existing job for this area was for, if any."""
    record = read_job(key, "rainfall")
    return (record or {}).get("product")


def submit(
    geojson_geom: dict,
    *,
    indicator: str = "rainfall",
    start: str = "2010-01-01",
    end: Optional[str] = None,
    label: Optional[str] = None,
    submitted_by: Optional[str] = None,
    product: Optional[str] = None,
    year: Optional[int] = None,
) -> dict:
    """Record a request to compute one area. Idempotent, and never blocks.

    An area that already has a series is reported ready without queueing, and
    re-submitting a pending or running area returns the existing job rather than
    starting a second computation of the same thing.
    """
    if indicator not in INDICATORS:
        raise ValueError(f"unknown indicator {indicator!r}; choose from {INDICATORS}")
    key = rainfall.geometry_hash(geojson_geom)
    product = (product or "rainfall").strip().lower()
    # Window-aware readiness: a stored series that does not span this caller's
    # window is not "already done" for them -- it is the 1-year answer to a
    # 5-year question. Treat it as not ready so the current window is queued
    # and computed over the union with the stored one in the worker.
    current = status_for(key, indicator=indicator, start=start, end=end, year=year)
    # A pending/running record must be returned so we do not queue a second
    # window. But a *ready* record whose coverage does not span this window
    # must fall through and queue the wider read.
    if current["state"] in {RUNNING, PENDING}:
        if indicator != "rainfall" or _job_product(key) in (None, product):
            return current
    elif current["state"] == READY:
        if indicator != "rainfall" or _job_product(key) in (None, product):
            # ready for this window already -- coalesce onto it
            return current

    outstanding = pending_count()
    if outstanding >= MAX_PENDING_JOBS:
        return {
            "state": "rejected",
            "cache_key": key,
            "reason": (
                f"{outstanding} areas are already queued and the limit is "
                f"{MAX_PENDING_JOBS}; try again shortly"
            ),
        }

    job = {
        "v": JOB_VERSION,
        "cache_key": key,
        "indicator": indicator,
        "state": PENDING,
        "submitted_at": _now(),
        "start": start,
        "end": (end or None),
        "year": year,
        "label": label,
        "submitted_by_prefix": submitted_by,
        "product": product,
        "attempts": 0,
    }
    # The geometry is kept so a runner can compute without the caller. It is the
    # caller's own polygon, in the caller's own cache directory, and is removed
    # once the job reaches a terminal state.
    job["geometry"] = geojson_geom
    write_job(job)
    # Report the job that was just written, not the rainfall record for the same
    # area. status_for defaults to rainfall, so asking it without the indicator
    # returned "not_submitted" for a DEM job that had just been queued
    # successfully -- the record existed with state pending while the reply said
    # nothing had been submitted. Every non-rainfall indicator was affected, which
    # is the whole multi-module reading room, and it also stopped the backend's
    # autorun from firing for them because it gates on state == pending.
    return status_for(key, indicator, start=start, end=end, year=year)


def claim_next(*, retry_failed: bool = False) -> Optional[dict]:
    """Take the oldest runnable job, moving it to running.

    Written as read-then-write rather than a lock, which is safe here because a
    single runner is the intended deployment. Two runners could claim the same
    job; the work is idempotent, so the worst case is duplicated reads.

    A job left in ``running`` by a worker that died, hung or was cancelled is
    reclaimed once it is older than ``JOB_LEASE_SECONDS``. It used to be
    reclaimed never, which meant a single interrupted compute retired that
    indicator for that area permanently: the record said running, no artefact
    existed, and every later submission short-circuited on the running record
    without queueing anything. The docstring above used to promise recovery via
    ``retry_failed``, but that flag only adds FAILED to the candidate states, so
    the recovery it described did not exist.
    """
    for state in ((PENDING, FAILED) if retry_failed else (PENDING,)):
        for job in list_jobs(state):
            if state == FAILED and job.get("attempts", 0) >= 3:
                continue
            job["state"] = RUNNING
            job["started_at"] = _now()
            job["attempts"] = job.get("attempts", 0) + 1
            write_job(job)
            return job

    for job in list_jobs(RUNNING):
        if _lease_expired(job):
            job["state"] = RUNNING
            job["started_at"] = _now()
            job["attempts"] = job.get("attempts", 0) + 1
            job["reason"] = "reclaimed: previous attempt did not finish"
            write_job(job)
            return job
    return None


def complete(key: str, payload: dict, indicator: str = "rainfall") -> dict:
    job = read_job(key, indicator) or {
        "v": JOB_VERSION, "cache_key": key, "indicator": indicator}
    job.update({
        "state": READY,
        "finished_at": _now(),
        "months": len(payload.get("series", [])),
        "grid_cells": payload.get("grid_cells"),
        "reason": None,
    })
    job.pop("geometry", None)  # the polygon is not needed once the series exists
    write_job(job)
    return job


def fail(key: str, reason: str, indicator: str = "rainfall") -> dict:
    job = read_job(key, indicator) or {
        "v": JOB_VERSION, "cache_key": key, "indicator": indicator}
    job.update({"state": FAILED, "finished_at": _now(), "reason": reason[:300]})
    write_job(job)
    return job


def drop(key: str, indicator: str = "rainfall") -> bool:
    """Forget a job record, for instance after its area was deleted."""
    path = job_path(key, indicator)
    if path.exists():
        path.unlink()
        return True
    return False


def published_upload(publish: bool) -> bool:
    return bool(publish and rainfall.remote_prefix() is not None)


async def _compute_job(
    job: dict,
    *,
    source: Optional[Callable] = None,
    publish: bool = True,
    progress: Optional[Callable[[str], None]] = None,
) -> dict:
    """One claimed job, start to finish. Returns its own counters.

    Split out of ``run_pending`` so several jobs can be in flight at once. The
    claiming still happens one at a time under the worker lock, so a second
    worker cannot double-process; only the *computing* overlaps, which is where
    the time is -- a MODIS read is network wait, and a thread waiting on the
    network is a thread that could be reading a different month for a different
    module of the same reading.
    """
    import indicators

    counts = {"processed": 0, "ready": 0, "failed": 0, "skipped": 0}
    key = job["cache_key"]
    indicator = job.get("indicator") or "rainfall"
    geometry = job.get("geometry")
    if not geometry:
        # Nothing to compute from; a deletion or a pruned record.
        fail(key, "job record has no geometry; resubmit the area", indicator)
        counts.update(processed=1, failed=1)
        return counts
    # Window-aware skip: a stored series that does not span this job's window
    # must be recomputed over the union -- skipping it would serve the old,
    # narrower window for the new, wider one.
    if indicator == "rainfall" and _product_ready(
        key, _job_product(key), job.get("start"), job.get("end")
    ):
        complete(key, rainfall.read_cache(key, _job_product(key) or 'rainfall'), indicator)
        counts.update(processed=1, skipped=1)
        return counts
    # A non-rainfall indicator is already done when its own artefact covers
    # this job's window. It used to test the rainfall cache instead and, for any area that had a
    # rainfall series, mark the job complete using the rainfall payload —
    # through complete()'s default indicator, so the rainfall record was
    # rewritten and this one was left marked running forever, with no
    # artefact ever written. A vegetation series could therefore never be
    # produced for any area that also had rainfall, which is every area
    # anyone would ask for it on.
    if indicator != "rainfall" and _artefact_ready(key, indicator, job.get("start"), job.get("end"), job.get("year")):
        complete(key, read_artefact(key, indicator), indicator)
        counts.update(processed=1, skipped=1)
        return counts
    if progress:
        progress(f"computing {indicator} for {job.get('label') or key[:12]}")
    try:
        if indicator == "rainfall":
            # Which product was asked for. A job that chose CHIRPS and got
            # ERA5 would answer a different question from the one queued, and
            # nothing downstream would notice because both are "rainfall".
            product = (job.get("product") or "rainfall").strip().lower()
            if product == "rainfall":
                union_source = source or rainfall.UnionReader(rainfall.union_grid([geometry]))
            else:
                # The *name* is passed through, not the reader it resolves
                # to. Resolving here lost the identity: build_and_cache works
                # out which cache slot to write from what it is given, so a
                # callable arrived as "rainfall" and a CHIRPS series was
                # written into the ERA5 slot -- the job reported ready and the
                # reader found nothing.
                union_source = source
            payload = rainfall.build_and_cache(
                geometry,
                start=job.get("start", "2010-01-01"),
                end=job.get("end"),
                source=union_source,
                product=product,
                upload=publish,
                label=job.get("label"),
            )
        else:
            # The async path exists to answer what the request path had to
            # refuse, so the area budget does not apply; the resolution policy does.
            from main import _bbox_area_km2, aoi_bbox, canonicalize_geojson

            canonical = canonicalize_geojson(
                {"type": "Feature", "properties": {}, "geometry": geometry},
                max_bytes=4_000_000, max_vertices=None,
            )
            bbox = aoi_bbox(canonical)
            area = _bbox_area_km2(bbox)
            resolution, reason = indicators.resolution_for(area, indicator)
            payload = await indicators.compute_indicator(
                indicator, bbox, geometry,
                area_km2=area, resolution_m=resolution,
                window_start=job.get("start"),
                window_end=job.get("end"), enforce_budget=False,
                year=job.get("year"),
                cog_path=(f"/app/.rainfall-cache/tiles/{key}/ndvi.tif"
                          if indicator == "ndvi" else None),
            )
            # The resolution to report is the one actually read, not the policy
            # target. A MODIS 250 m composite computed because the policy asked
            # for 100 m must say 250 m, or the provenance is a lie.
            applied = resolution
            note = None
            if indicator == "ndvi":
                native = ((payload.get("sensor") or {}).get("native_resolution_m")
                          or payload.get("resolution_m"))
                if native:
                    applied = max(resolution, int(native))
                    if applied > resolution:
                        note = (
                            f"{resolution} m was requested, but "
                            f"{(payload.get('sensor') or {}).get('label') or 'the chosen source'} "
                            f"publishes a finished {applied} m product, so that is what was read. "
                            "A product cannot be resampled finer without inventing detail."
                        )
            import main

            payload = main.with_evidence(indicator, {
                **payload,
                "indicator": indicator,
                "status": payload.get("status") or ("error" if payload.get("error") else "ok"),
                "applied_resolution_m": applied,
                "target_resolution_m": resolution,
                "resolution_reason": reason,
                "resolution_coarser_than_target": applied > resolution,
                "resolution_note": note,
                "pixels_analysed": indicators.pixels_for(area, applied),
                "bbox": bbox,
            })
            # Keep the AOI geometry with the artefact so the tile endpoint can
            # clip the raster to the polygon (transparent outside) without the
            # browser re-sending it.
            payload["geometry"] = geometry
            write_artefact(key, indicator, payload)
        record = complete(key, payload, indicator)
        if indicator == "rainfall":
            # A series computed but not uploaded is still ready locally; saying
            # so keeps a store outage from failing work that succeeded.
            record["published"] = (
                rainfall.publish(key) if published_upload(publish) else False
            )
            write_job(record)
        counts.update(processed=1, ready=1)
    except Exception as exc:  # noqa: BLE001 - one bad area must not stop the run
        if os.getenv("RAINFALL_JOB_DEBUG"):
            # Surface the real traceback instead of a one-line reason: a bare
            # "NameError" with no name is not a diagnosis.
            raise
        fail(key, f"{type(exc).__name__}: {exc}", indicator)
        counts.update(processed=1, failed=1)
        if progress:
            progress(f"failed {key[:12]}: {type(exc).__name__}: {exc}")
    except BaseException as exc:  # noqa: BLE001
        # asyncio.CancelledError descends from BaseException, not Exception, so
        # the branch above does not see it. A cancelled sweep therefore used to
        # leave the job it was holding marked running with nothing written, and
        # nothing would ever reclaim it. Record the outcome, then let the
        # cancellation propagate.
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            fail(key, f"interrupted: {type(exc).__name__}", indicator)
            raise
        fail(key, f"cancelled: {type(exc).__name__}: {exc}", indicator)
        counts.update(processed=1, failed=1)
        raise
    return counts


async def run_pending(
    *,
    limit: int = 10,
    source: Optional[Callable] = None,
    publish: bool = True,
    progress: Optional[Callable[[str], None]] = None,
    on_complete: Optional[Callable[[str, str, dict], None]] = None,
) -> dict:
    """Compute queued areas. The runner.

    Skips anything that already has a result, so it is safe to run repeatedly and
    safe to interrupt: completed work is never redone. A job left in running by a
    dead worker is reclaimed once its lease expires.

    Jobs are computed a few at a time rather than strictly one at a time. The
    claim is still serial -- one read-modify-write per job, under the worker lock
    -- because that is what stops two workers from doing the same work; but the
    reads that follow are network waits, and a reading of four modules for one
    boundary was paying for them end to end, so a county that should take a
    couple of minutes took eight. JOBS_CONCURRENCY sets the width, defaulting to
    three: enough to overlap the waits, small enough that a Planetary Computer
    rate limit is a retry rather than an outage.
    """
    concurrency = max(1, int(os.getenv("JOBS_CONCURRENCY", "3")))
    processed = ready = failed = skipped = 0
    while limit > 0:
        batch = []
        while len(batch) < min(concurrency, limit):
            claimed = claim_next()
            if claimed is None:
                break
            batch.append(claimed)
        if not batch:
            break
        limit -= len(batch)
        # return_exceptions so one failure does not cancel its neighbours: each
        # job already records its own outcome, and a cancelled sweep is re-raised
        # after the batch has been accounted for.
        results = await asyncio.gather(*[
            _compute_job(job, source=source, publish=publish, progress=progress)
            for job in batch
        ], return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result
            processed += result["processed"]
            ready += result["ready"]
            failed += result["failed"]
            skipped += result["skipped"]
    return {"processed": processed, "ready": ready, "failed": failed, "skipped": skipped}
