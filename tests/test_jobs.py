"""Tests for the rainfall submission queue.

The queue is what makes submission durable rather than a promise: a submission
survives a restart and a runner completes whatever is pending. These tests use a
stubbed compute path, so the ERA5 read is the only thing not covered here.
"""

import json
import os
import tempfile
import unittest

import jobs
import rainfall


def _no_read():
    """A read source that never touches the network.

    ``source`` is the seam for the ERA5 read, so it must be callable: the
    rainfall path hands it straight to ``compute_series``, which calls it.
    """
    import numpy as np

    def read(grid, start, end):
        return (["2000-01"], np.zeros((1, 1, 1), dtype="float32"), {
            "expected_hours": 0, "valid_hours": 0,
            "months_kept": 1, "months_dropped": [], "min_coverage": 1.0,
        })

    return read


GEOM = {"type": "Polygon",
        "coordinates": [[[35.10, -1.55], [35.17, -1.55], [35.17, -1.48],
                         [35.10, -1.48], [35.10, -1.55]]]}


class _FakeCompute:
    """Stands in for the ERA5 read.

    Patches ``rainfall.compute_series`` rather than passing a ``source``: the
    ``source`` argument is a read source (a UnionReader over shared cells), not the
    computation, so stubbing it would not bypass the network.
    """

    def __init__(self, fail_keys=()):
        self.fail_keys = set(fail_keys)
        self.calls = []

    def __enter__(self):
        import unittest.mock

        def fake(geometry, **kwargs):
            key = rainfall.geometry_hash(geometry)
            self.calls.append((key, kwargs))
            if key in self.fail_keys:
                raise RuntimeError("ERA5 read failed")
            return {
                "processing_version": rainfall.RAINFALL_PROCESSING_VERSION,
                "indicator": "monthly_precipitation",
                "series": [{"month": "2020-01", "precip_mm": 2.0}],
                "grid_cells": 2,
            }

        self._patcher = unittest.mock.patch.object(rainfall, "compute_series", fake)
        self._patcher.start()
        return self

    def __exit__(self, *exc):
        self._patcher.stop()
        return False


class JobQueueTests(unittest.TestCase):
    def setUp(self):
        self.previous = os.environ.get("RAINFALL_CACHE_DIR")
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["RAINFALL_CACHE_DIR"] = self.tmp.name
        self.addCleanup(self._restore)
        self.addCleanup(self.tmp.cleanup)

    def _run(self, **kwargs):
        """Drive the now-async runner, with the ERA5 read stubbed out."""
        import asyncio

        return asyncio.run(jobs.run_pending(limit=5, source=_no_read(), **kwargs))

    def _restore(self):
        if self.previous is None:
            os.environ.pop("RAINFALL_CACHE_DIR", None)
        else:
            os.environ["RAINFALL_CACHE_DIR"] = self.previous

    def test_an_unseen_area_is_not_submitted(self):
        self.assertEqual(jobs.status_for(rainfall.geometry_hash(GEOM))["state"],
                         "not_submitted")

    def test_submitting_records_a_pending_job(self):
        state = jobs.submit(GEOM, label="test")
        self.assertEqual(state["state"], jobs.PENDING)
        self.assertEqual(len(state["cache_key"]), 32)
        self.assertEqual(jobs.pending_count(), 1)

    def test_submission_keeps_the_geometry_so_a_runner_needs_no_caller(self):
        jobs.submit(GEOM)
        job = jobs.read_job(rainfall.geometry_hash(GEOM))
        self.assertIn("geometry", job, "a runner must be able to compute without the caller")

    def test_resubmitting_does_not_queue_a_second_job(self):
        first = jobs.submit(GEOM)
        second = jobs.submit(GEOM)
        self.assertEqual(first["cache_key"], second["cache_key"])
        self.assertEqual(jobs.pending_count(), 1)

    def test_an_area_that_already_has_a_series_is_never_queued(self):
        key = rainfall.geometry_hash(GEOM)
        rainfall.write_cache(key, {
            "processing_version": rainfall.RAINFALL_PROCESSING_VERSION,
            "series": [{"month": "2020-01", "precip_mm": 2.0}],
        })
        state = jobs.submit(GEOM)
        self.assertEqual(state["state"], jobs.READY)
        self.assertEqual(jobs.pending_count(), 0)

    def test_the_cache_is_authoritative_over_a_stale_ready_record(self):
        key = rainfall.geometry_hash(GEOM)
        jobs.write_job({"cache_key": key, "state": jobs.READY})
        # The record says ready but the series is gone, for instance after a
        # deletion. Report what is true.
        self.assertEqual(jobs.status_for(key)["state"], "not_submitted")

    def test_the_queue_is_bounded(self):
        original = jobs.MAX_PENDING_JOBS
        jobs.MAX_PENDING_JOBS = 2
        try:
            for index in range(2):
                geom = {"type": "Polygon", "coordinates": [[[
                    35.10 + index, -1.55], [35.11 + index, -1.55],
                    [35.11 + index, -1.54], [35.10 + index, -1.54], [35.10 + index, -1.55]]]}
                jobs.submit(geom)
            rejected = jobs.submit(GEOM)
            self.assertEqual(rejected["state"], "rejected")
            self.assertIn(str(2), rejected["reason"])
        finally:
            jobs.MAX_PENDING_JOBS = original

    def test_a_runner_completes_a_queued_area_and_drops_the_polygon(self):
        jobs.submit(GEOM, label="test")
        with _FakeCompute() as runner:
            outcome = self._run()
        self.assertEqual(outcome["ready"], 1)
        self.assertEqual(outcome["failed"], 0)
        self.assertEqual(outcome["processed"], 1)
        self.assertEqual(len(runner.calls), 1)
        key = rainfall.geometry_hash(GEOM)
        self.assertEqual(jobs.status_for(key)["state"], jobs.READY)
        self.assertNotIn("geometry", jobs.read_job(key),
                         "the polygon is not needed once the series exists")

    def test_a_failing_area_is_recorded_and_does_not_stop_the_run(self):
        jobs.submit(GEOM)
        bad = {"type": "Polygon", "coordinates": [[[
            36.10, -1.55], [36.11, -1.55], [36.11, -1.54], [36.10, -1.54], [36.10, -1.55]]]}
        jobs.submit(bad)
        with _FakeCompute(fail_keys={rainfall.geometry_hash(bad)}):
            outcome = self._run()
        self.assertEqual(outcome["ready"], 1)
        self.assertEqual(outcome["failed"], 1)
        failed = jobs.read_job(rainfall.geometry_hash(bad))
        self.assertEqual(failed["state"], jobs.FAILED)
        self.assertIn("ERA5 read failed", failed["reason"])

    def test_the_runner_is_idempotent(self):
        jobs.submit(GEOM)
        self._run()
        again = self._run()
        self.assertEqual(again["processed"], 0, "completed work must not be redone")

    def test_the_runner_passes_the_requested_window_through(self):
        jobs.submit(GEOM, start="1997-01-01")
        with _FakeCompute() as runner:
            self._run()
        _key, kwargs = runner.calls[0]
        self.assertEqual(kwargs["start"], "1997-01-01")

    def test_a_running_job_is_not_claimed_again(self):
        jobs.submit(GEOM)
        first = jobs.claim_next()
        self.assertEqual(first["state"], jobs.RUNNING)
        self.assertIsNone(jobs.claim_next(), "a running job must not be double-claimed")

    def test_a_failed_job_can_be_retried_up_to_a_limit(self):
        key = rainfall.geometry_hash(GEOM)
        jobs.write_job({"cache_key": key, "state": jobs.FAILED,
                        "attempts": 1, "geometry": GEOM})
        self.assertIsNotNone(jobs.claim_next(retry_failed=True))
        jobs.write_job({"cache_key": key, "state": jobs.FAILED,
                        "attempts": 3, "geometry": GEOM})
        self.assertIsNone(jobs.claim_next(retry_failed=True), "a job that keeps failing is not retried forever")

    def test_a_job_record_without_a_geometry_fails_clearly(self):
        key = rainfall.geometry_hash(GEOM)
        jobs.write_job({"cache_key": key, "state": jobs.PENDING, "indicator": "rainfall"})
        outcome = self._run()
        self.assertEqual(outcome["failed"], 1)
        self.assertIn("no geometry", jobs.read_job(key)["reason"])

    def test_dropping_a_job_removes_the_record(self):
        jobs.submit(GEOM)
        key = rainfall.geometry_hash(GEOM)
        self.assertTrue(jobs.drop(key))
        self.assertIsNone(jobs.read_job(key))
        self.assertFalse(jobs.drop(key), "dropping twice is not an error")

    def test_job_records_are_valid_json_on_disk(self):
        jobs.submit(GEOM, label="with spaces and / slashes")
        key = rainfall.geometry_hash(GEOM)
        payload = json.loads(jobs.job_path(key).read_text())
        self.assertEqual(payload["label"], "with spaces and / slashes")
        self.assertEqual(payload["v"], jobs.JOB_VERSION)


if __name__ == "__main__":
    unittest.main()


class PublishResilienceTests(unittest.TestCase):
    """A store outage must not fail work that already succeeded."""

    def setUp(self):
        self.previous = os.environ.get("RAINFALL_CACHE_DIR")
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["RAINFALL_CACHE_DIR"] = self.tmp.name
        self.addCleanup(self._restore)
        self.addCleanup(self.tmp.cleanup)

    def _restore(self):
        for key in ("RAINFALL_CACHE_DIR", rainfall.REMOTE_URI_ENV):
            if os.environ.get(key) is None:
                os.environ.pop(key, None)

    def _run(self, **kwargs):
        """Drive the async runner with the read and the store stubbed out."""
        import asyncio

        return asyncio.run(jobs.run_pending(limit=5, source=_no_read(), **kwargs))

    def test_publish_reports_failure_instead_of_raising(self):
        import unittest.mock

        os.environ[rainfall.REMOTE_URI_ENV] = "s3://bucket/prefix"

        class Broken:
            def upload_file(self, *_a, **_kw):
                raise OSError("object store unreachable")

        patcher = unittest.mock.patch.object(rainfall, "_s3_client", lambda: Broken())
        patcher.start()
        self.addCleanup(patcher.stop)
        rainfall.write_cache("a" * 32, {"processing_version": rainfall.RAINFALL_PROCESSING_VERSION})
        self.assertFalse(rainfall.publish("a" * 32), "an upload failure must be reported, not raised")

    def test_publish_is_false_when_no_remote_is_configured(self):
        os.environ.pop(rainfall.REMOTE_URI_ENV, None)
        rainfall.write_cache("b" * 32, {"processing_version": rainfall.RAINFALL_PROCESSING_VERSION})
        self.assertFalse(rainfall.publish("b" * 32))

    def test_a_job_stays_ready_when_only_the_upload_fails(self):
        import unittest.mock

        import rainfall as _r
        from tests.test_rainfall import _StubClient  # noqa: F401  (import check)

        os.environ[rainfall.REMOTE_URI_ENV] = "s3://bucket/prefix"
        geom = {"type": "Polygon", "coordinates": [[[
            35.10, -1.55], [35.11, -1.55], [35.11, -1.54], [35.10, -1.54], [35.10, -1.55]]]}

        class Broken:
            def upload_file(self, *_a, **_kw):
                raise OSError("object store unreachable")

        def fake(geometry, **kwargs):
            return {
                "processing_version": rainfall.RAINFALL_PROCESSING_VERSION,
                "indicator": "monthly_precipitation",
                "series": [{"month": "2020-01", "precip_mm": 2.0}],
            }

        compute = unittest.mock.patch.object(rainfall, "compute_series", fake)
        client = unittest.mock.patch.object(rainfall, "_s3_client", lambda: Broken())
        compute.start(); client.start()
        self.addCleanup(compute.stop); self.addCleanup(client.stop)

        jobs.submit(geom)
        outcome = self._run(publish=True)
        self.assertEqual(outcome["ready"], 1, "a store outage must not fail a completed job")
        self.assertEqual(outcome["failed"], 0)
        key = rainfall.geometry_hash(geom)
        self.assertIsNotNone(rainfall.read_cache(key), "the series is still usable locally")
        self.assertFalse(jobs.read_job(key)["published"], "and the job records that it was not uploaded")
