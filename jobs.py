"""Durable jobs for self-service rainfall submission.

A submission is a request to compute a series, and computing one takes about 60
seconds of ERA5 reads. That is longer than a request should take, so the submit
path only records a job and returns; the work happens in a runner.

The job record is what makes this durable rather than a promise: it lives on disk,
so a submission survives a restart, and a runner picks up whatever is still
pending. The runner is idempotent, because an area that already has a series is
skipped rather than recomputed.

There is no supervision here. ``tools/run_rainfall_jobs.py`` is the runner, and
running it from cron or a sidecar is the difference between "usually works" and
"reliably works". Nothing in the design depends on where it runs.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path
from typing import Any, Callable, Optional

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

JOB_VERSION = 1


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def jobs_dir() -> Path:
    return rainfall.cache_dir() / "jobs"


def job_path(key: str) -> Path:
    return jobs_dir() / f"{key}.json"


def read_job(key: str) -> Optional[dict]:
    path = job_path(key)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def write_job(job: dict) -> None:
    path = job_path(job["cache_key"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(job, indent=1, sort_keys=True))
    temporary.replace(path)


def list_jobs(state: Optional[str] = None) -> list[dict]:
    directory = jobs_dir()
    if not directory.exists():
        return []
    jobs = []
    for path in sorted(directory.glob("*.json")):
        try:
            job = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if state is None or job.get("state") == state:
            jobs.append(job)
    jobs.sort(key=lambda j: j.get("submitted_at") or "")
    return jobs


def pending_count() -> int:
    return len(list_jobs(PENDING))


def status_for(key: str) -> dict:
    """What a caller should be told about one study area.

    The cache is authoritative: if a series exists, the area is ready whatever the
    job record says, because a runner may have completed without updating the
    record.
    """
    if rainfall.read_cache(key) is not None:
        return {"state": READY, "cache_key": key, "source": "cache"}
    job = read_job(key)
    if job is None:
        return {"state": "not_submitted", "cache_key": key}
    if job.get("state") == READY and rainfall.read_cache(key) is None:
        # The record says ready but the series is gone, for instance after a
        # deletion. Report what is true rather than what was recorded.
        return {"state": "not_submitted", "cache_key": key}
    return {
        "state": job.get("state", PENDING),
        "cache_key": key,
        "submitted_at": job.get("submitted_at"),
        "started_at": job.get("started_at"),
        "finished_at": job.get("finished_at"),
        "attempts": job.get("attempts", 0),
        "reason": job.get("reason"),
    }


def submit(
    geojson_geom: dict,
    *,
    start: str = "2010-01-01",
    label: Optional[str] = None,
    submitted_by: Optional[str] = None,
) -> dict:
    """Record a request to compute one area. Idempotent, and never blocks.

    An area that already has a series is reported ready without queueing, and
    re-submitting a pending or running area returns the existing job rather than
    starting a second computation of the same thing.
    """
    key = rainfall.geometry_hash(geojson_geom)
    current = status_for(key)
    if current["state"] in {READY, RUNNING, PENDING}:
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
        "state": PENDING,
        "submitted_at": _now(),
        "start": start,
        "label": label,
        "submitted_by_prefix": submitted_by,
        "attempts": 0,
    }
    # The geometry is kept so a runner can compute without the caller. It is the
    # caller's own polygon, in the caller's own cache directory, and is removed
    # once the job reaches a terminal state.
    job["geometry"] = geojson_geom
    write_job(job)
    return status_for(key)


def claim_next(*, retry_failed: bool = False) -> Optional[dict]:
    """Take the oldest runnable job, moving it to running.

    Written as read-then-write rather than a lock, which is safe here because a
    single runner is the intended deployment. Two runners could claim the same
    job; the work is idempotent, so the worst case is duplicated reads.
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
    return None


def complete(key: str, payload: dict) -> dict:
    job = read_job(key) or {"v": JOB_VERSION, "cache_key": key}
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


def fail(key: str, reason: str) -> dict:
    job = read_job(key) or {"v": JOB_VERSION, "cache_key": key}
    job.update({"state": FAILED, "finished_at": _now(), "reason": reason[:300]})
    write_job(job)
    return job


def drop(key: str) -> bool:
    """Forget a job record, for instance after its area was deleted."""
    path = job_path(key)
    if path.exists():
        path.unlink()
        return True
    return False


def run_pending(
    *,
    limit: int = 10,
    source: Optional[Callable] = None,
    publish: bool = True,
    progress: Optional[Callable[[str], None]] = None,
) -> dict:
    """Compute queued areas. The runner.

    Skips anything that already has a series, so it is safe to run repeatedly and
    safe to interrupt: completed work is never redone, and a job left in running is
    recoverable by passing ``retry_failed``.
    """
    processed = ready = failed = skipped = 0
    for _ in range(limit):
        job = claim_next()
        if job is None:
            break
        key = job["cache_key"]
        geometry = job.get("geometry")
        if not geometry:
            # Nothing to compute from; a deletion or a pruned record.
            fail(key, "job record has no geometry; resubmit the area")
            failed += 1
            continue
        if rainfall.read_cache(key) is not None:
            complete(key, rainfall.read_cache(key))
            skipped += 1
            continue
        if progress:
            progress(f"computing {job.get('label') or key[:12]}")
        try:
            union_source = source or rainfall.UnionReader(rainfall.union_grid([geometry]))
            payload = rainfall.build_and_cache(
                geometry,
                start=job.get("start", "2010-01-01"),
                source=union_source,
                upload=publish,
                label=job.get("label"),
            )
            uploaded = publish and rainfall.remote_prefix() is not None
            record = complete(key, payload)
            # A series computed but not uploaded is still ready locally; saying so
            # keeps a store outage from failing work that succeeded.
            if uploaded:
                record["published"] = rainfall.publish(key)
                write_job(record)
            else:
                record["published"] = False
                write_job(record)
            ready += 1
        except Exception as exc:  # noqa: BLE001 - one bad area must not stop the run
            fail(key, f"{type(exc).__name__}: {exc}")
            failed += 1
            if progress:
                progress(f"failed {key[:12]}: {type(exc).__name__}")
        processed += 1
    return {"processed": processed, "ready": ready, "failed": failed, "skipped": skipped}
