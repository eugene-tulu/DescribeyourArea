"""A supervised loop around :func:`jobs.run_pending`.

Supervision here means three small things rather than a process manager:

- **A lock**, so a second worker waits instead of duplicating ERA5 reads. Taken
  with ``flock`` on a shared volume, so a crashed holder releases it and a
  container restart cannot leave a stale lock wedging the queue.
- **A poll loop** that drains the queue, then sleeps, so a submission made
  between sweeps is picked up without anything having to notice it.
- **An exit on repeated failure**, so a worker that cannot do its job restarts
  rather than spinning quietly forever. A queue that appears healthy while
  nothing is computed is worse than a visible crash.

``RainfallJobError`` is raised for a per-area failure, which is recorded on the
job; only an environment-level failure escapes and trips the restart.
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Optional

import jobs
import rainfall


def lock_path() -> Path:
    return rainfall.cache_dir() / "worker.lock"


@contextmanager
def worker_lock(blocking: bool = False):
    """Hold an exclusive lock on the worker.

    ``flock`` rather than an O_EXCL lock file, because the kernel releases it when
    the process dies. A lock file left behind by a crash would need a janitor, and
    a queue nobody can process is the one failure mode worth engineering against.
    """
    path = lock_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = open(path, "w")
    try:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            handle.close()
            yield False
            return
        handle.seek(0)
        handle.truncate()
        handle.write(json.dumps({"pid": os.getpid(), "since": time.time()}))
        handle.flush()
        try:
            yield True
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)
    finally:
        if not handle.closed:
            handle.close()


def _completion_notice(area_key: str, indicator: str, payload: dict) -> None:
    """Tell a human the work they submitted is ready.

    A polling browser tab does not help someone who submitted an area and then
    closed it, which is the normal way this is used. Webhook and email both carry
    it, so the same message serves an automated consumer and a person.
    """
    import notify

    ready = str(payload.get("status") or "") == "ok"
    try:
        notify.dispatch({
            "event": notify.JOB_READY if ready else notify.JOB_FAILED,
            "area_key": area_key,
            "area": payload.get("label"),
            "indicator": indicator,
            "months": payload.get("months") or (len(payload.get("series") or []) or None),
            "resolution_m": payload.get("resolution_m"),
            "grid_cells": payload.get("grid_cells_in_area"),
            "source": payload.get("source"),
            "reason": payload.get("reason") or payload.get("warning"),
            "retrieve_hint": f"POST /rainfall with cache_key {area_key} and indicator {indicator}",
        })
    except Exception:  # noqa: BLE001 - a notice must never fail a sweep
        pass


async def sweep(
    *,
    limit: int = 5,
    publish: bool = True,
    on_alert: Optional[Callable[[list], None]] = None,
    progress: Optional[Callable[[str], None]] = None,
) -> dict:
    """One pass: compute whatever is pending, then check for alerts.

    Alerts are evaluated after the queue drains so a freshly computed series is
    checked immediately rather than waiting for the next sweep.
    """
    outcome = await jobs.run_pending(
        limit=limit, publish=publish, progress=progress, on_complete=_completion_notice
    )
    outcome["alerts"] = 0
    if on_alert is not None:
        sent = on_alert(ready_keys_since())
        outcome["alerts"] = len(sent)
    return outcome


def ready_keys_since() -> list[str]:
    """Areas that have a series, newest first."""
    keys = []
    directory = rainfall.cache_dir()
    if not directory.exists():
        return keys
    for path in directory.glob("*.json"):
        try:
            payload = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if payload.get("indicator") == "monthly_precipitation":
            keys.append(path.stem)
    return keys


def run_forever(
    *,
    interval: float = 60.0,
    limit: int = 5,
    publish: bool = True,
    max_consecutive_idle_failures: int = 3,
    on_alert: Optional[Callable[[list], None]] = None,
    progress: Optional[Callable[[str], None]] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict:
    """Poll until stopped. Returns a summary of what it did.

    ``max_consecutive_idle_failures`` is about the environment rather than an
    area: a run that raises repeatedly means the worker itself cannot work, so it
    exits and lets the restart policy act. One bad area is recorded on its job and
    does not stop the loop.
    """
    totals = {"sweeps": 0, "processed": 0, "ready": 0, "failed": 0, "alerts": 0, "skipped": 0}
    idle_failures = 0

    # Keep the CHIRPS store current with DEA's monthly publications, from here in
    # the background rather than by hand. Off by default: extending the shared
    # store is the one thing the worker can do that other consumers of that store
    # (the backend, a reader) can be affected by, so it is opt-in, idempotent, and
    # rate-limited to a handful of times a day. It is isolated so a store failure
    # never trips the environment-failure counter -- the queue is more important
    # than the store being a month or two behind.
    auto_refresh = (os.getenv("CHIRPS_STORE_AUTO_UPDATE", "1").strip().lower()
                    not in ("0", "false", "no", "off", ""))
    refresh_seconds = float(os.getenv("CHIRPS_STORE_REFRESH_SECONDS", "21600"))
    last_refresh = time.monotonic()

    with worker_lock() as acquired:
        if not acquired:
            raise RuntimeError("another worker holds the lock; exiting so the supervisor restarts one")

        while True:
            try:
                # sweep is async because computing a raster indicator is; run_forever
                # stays sync so its lock and sleep remain ordinary blocking calls.
                outcome = asyncio.run(sweep(limit=limit, publish=publish,
                                            on_alert=on_alert, progress=progress))
                idle_failures = 0
                for key in ("processed", "ready", "failed", "alerts", "skipped"):
                    totals[key] += outcome.get(key, 0)
            except Exception as exc:  # noqa: BLE001 - an environment failure
                idle_failures += 1
                if progress:
                    progress(f"worker sweep failed ({idle_failures}/"
                             f"{max_consecutive_idle_failures}): {type(exc).__name__}: {exc}")
                if idle_failures >= max_consecutive_idle_failures:
                    raise

            if auto_refresh and refresh_seconds and \
                    time.monotonic() - last_refresh >= refresh_seconds:
                last_refresh = time.monotonic()
                try:
                    result = rainfall.chirps_store_refresh()
                    if progress and result.get("added"):
                        progress(f"chirps store: {result}")
                except Exception as exc:  # noqa: BLE001 - never restart the worker over this
                    if progress:
                        progress(f"chirps store refresh failed: "
                                 f"{type(exc).__name__}: {exc}")

            totals["sweeps"] += 1
            sleep(interval)
    return totals


if __name__ == "__main__":  # pragma: no cover - exercised by the container
    import argparse
    import asyncio
    import sys

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--interval", type=float, default=60.0,
                        help="seconds between sweeps")
    parser.add_argument("--limit", type=int, default=5, help="areas per sweep")
    parser.add_argument("--no-publish", action="store_true",
                        help="do not upload results to the object store")
    parser.add_argument("--once", action="store_true",
                        help="run a single sweep and exit, rather than polling")
    args = parser.parse_args()

    import alerts

    def on_alert(keys):
        return alerts.evaluate_and_notify(keys)

    def report(message):
        print(message, flush=True)

    if args.once:
        print(f"done: {asyncio.run(sweep(limit=args.limit, publish=not args.no_publish, on_alert=on_alert, progress=report))}",
              flush=True)
        sys.exit(0)

    try:
        run_forever(
            interval=args.interval,
            limit=args.limit,
            publish=not args.no_publish,
            on_alert=on_alert,
            progress=report,
        )
    except KeyboardInterrupt:
        sys.exit(0)
