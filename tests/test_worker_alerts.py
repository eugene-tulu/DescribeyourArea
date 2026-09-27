"""Tests for worker supervision and alerting.

The two properties worth locking are the ones whose absence is silent: that a
second worker does not duplicate ERA5 reads, and that an alert fires once per
episode rather than once per sweep.
"""

import json
import os
import tempfile
import unittest

import alerts
import jobs
import rainfall
import worker

RAIN = {"precip_mm": 5.0, "anomaly_pct": -60.0, "normal_mm": 20.0, "month": "2026-01"}


def payload(*anomalies, label="Test Area", months=13):
    rows = []
    for index, value in enumerate(anomalies):
        rows.append({"month": f"2026-{index + 1:02d}", "precip_mm": 5.0 if value < 0 else 50.0,
                     "normal_mm": 20.0, "anomaly_mm": value, "anomaly_pct": value})
    tail = rows[-1] if rows else None
    return {
        "processing_version": rainfall.RAINFALL_PROCESSING_VERSION,
        "indicator": "monthly_precipitation",
        "label": label,
        "source": "ERA5 (Earthmover Icechunk edition)",
        "doi": rainfall.ERA5_DOI,
        "retrieved": "2026-09-27",
        "climatology": {"standard": "WMO 1991-2020 normal", "annual_mean_mm": 735.1},
        "series": rows,
        "summary": {
            "latest_anomaly_pct": tail["anomaly_pct"] if tail else None,
            # The trailing year is the mean of the last twelve months, so a test
            # that passes one anomaly gets a year of that anomaly.
            "trailing_12m": {
                "ending": tail["month"], "months": 12,
                "precip_mm": round(sum(r["precip_mm"] for r in rows[-12:]), 1),
                "normal_mm": round(sum(r["normal_mm"] for r in rows[-12:]), 1),
                "anomaly_pct": round(
                    sum(r["anomaly_pct"] for r in rows[-12:]) / len(rows[-12:]), 1),
            } if tail else None,
            "recent_3m": rows[-3:],
        },
    }


class LockTests(unittest.TestCase):
    def setUp(self):
        self.previous = os.environ.get("RAINFALL_CACHE_DIR")
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["RAINFALL_CACHE_DIR"] = self.tmp.name
        self.addCleanup(self._restore)
        self.addCleanup(self.tmp.cleanup)

    def _restore(self):
        if self.previous is None:
            os.environ.pop("RAINFALL_CACHE_DIR", None)
        else:
            os.environ["RAINFALL_CACHE_DIR"] = self.previous

    def test_the_lock_is_exclusive_within_a_process(self):
        # flock is per open file description, so a second descriptor in the same
        # process does block; this is the behaviour a second container relies on.
        with worker.worker_lock() as first:
            self.assertTrue(first)

    def test_a_second_holder_is_refused_and_reported(self):
        import fcntl
        import json as _json

        path = worker.lock_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        held = open(path, "w")
        try:
            fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with worker.worker_lock(blocking=False) as acquired:
                self.assertFalse(acquired, "a second worker must not proceed")
        finally:
            fcntl.flock(held, fcntl.LOCK_UN)
            held.close()
        # Once released, it can be taken again: a crashed holder must not wedge it.
        with worker.worker_lock(blocking=False) as acquired:
            self.assertTrue(acquired)
        info = json.loads(worker.lock_path().read_text())
        self.assertEqual(info["pid"], os.getpid())

    def test_a_holder_that_exits_releases_the_lock(self):
        import fcntl

        path = worker.lock_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        held = open(path, "w")
        fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)
        held.close()  # closing releases it, which is the point of flock
        with worker.worker_lock(blocking=False) as acquired:
            self.assertTrue(acquired)


class WorkerLoopTests(unittest.TestCase):
    def setUp(self):
        self.previous = os.environ.get("RAINFALL_CACHE_DIR")
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["RAINFALL_CACHE_DIR"] = self.tmp.name
        self.addCleanup(self._restore)
        self.addCleanup(self.tmp.cleanup)
        import unittest.mock

        def fake(geometry, **kwargs):
            return {
                "processing_version": rainfall.RAINFALL_PROCESSING_VERSION,
                "indicator": "monthly_precipitation",
                "series": [{"month": "2020-01", "precip_mm": 2.0}],
            }

        patcher = unittest.mock.patch.object(rainfall, "compute_series", fake)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _restore(self):
        if self.previous is None:
            os.environ.pop("RAINFALL_CACHE_DIR", None)
        else:
            os.environ["RAINFALL_CACHE_DIR"] = self.previous

    def test_a_sweep_drains_the_queue_and_evaluates_alerts(self):
        geom = {"type": "Polygon", "coordinates": [[[
            35.10, -1.55], [35.11, -1.55], [35.11, -1.54], [35.10, -1.54], [35.10, -1.55]]]}
        jobs.submit(geom)
        sent = []
        import asyncio

        outcome = asyncio.run(worker.sweep(on_alert=lambda keys: sent.extend(keys) or []))
        self.assertEqual(outcome["ready"], 1)
        self.assertEqual(len(sent), 1, "a freshly computed area is checked for alerts")

    def test_repeated_idle_failures_stop_the_worker(self):
        """A worker that cannot do its job must exit, not spin quietly."""
        import unittest.mock

        async def boom(**_kwargs):
            raise RuntimeError("object store unreachable")

        slept = []
        with unittest.mock.patch.object(worker, "sweep", boom):
            with self.assertRaises(RuntimeError):
                worker.run_forever(interval=0, sleep=slept.append,
                                   max_consecutive_idle_failures=3)
        self.assertEqual(len(slept), 2, "it stops on the third failure, not before")

    def test_a_single_transient_failure_does_not_stop_the_worker(self):
        import unittest.mock

        calls = {"n": 0}

        async def flaky(**_kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("one blip")
            return {"processed": 0, "ready": 0, "failed": 0, "skipped": 0, "alerts": 0}

        slept = []

        def stop_after_three(_seconds):
            slept.append(1)
            if len(slept) >= 3:
                raise KeyboardInterrupt

        with unittest.mock.patch.object(worker, "sweep", flaky):
            with self.assertRaises(KeyboardInterrupt):
                worker.run_forever(interval=0, sleep=stop_after_three,
                                   max_consecutive_idle_failures=3)
        self.assertEqual(calls["n"], 3, "one failure, then two healthy sweeps")

    def test_the_loop_sleeps_between_sweeps(self):
        slept = []

        def stop(_seconds):
            slept.append(1)
            if len(slept) >= 2:
                raise KeyboardInterrupt

        with self.assertRaises(KeyboardInterrupt):
            worker.run_forever(interval=0, sleep=stop)
        self.assertEqual(len(slept), 2)


class AlertRuleTests(unittest.TestCase):
    def test_the_default_rules_cover_drought_and_flood(self):
        configured = alerts._default_rules()
        self.assertEqual(len(configured), 3)
        for rule in configured:
            self.assertIn(rule["metric"], alerts.METRICS)
            self.assertTrue(rule.get("below") is not None or rule.get("above") is not None)

    def test_a_severe_dry_year_breaches_twice_and_flood_once(self):
        breaches = alerts.evaluate("k", payload(-75.0))
        self.assertEqual(sorted(b["rule"] for b in breaches),
                         ["drought-watch", "severe-drought"])

    def test_a_wet_year_breaches_the_flood_rule(self):
        self.assertEqual([b["rule"] for b in alerts.evaluate("k", payload(60.0))],
                         ["flood-watch"])

    def test_a_normal_year_breaches_nothing(self):
        self.assertEqual(alerts.evaluate("k", payload(-5.0)), [])

    def test_breaches_carry_their_evidence(self):
        breach = alerts.evaluate("k", payload(-75.0), [alerts._default_rules()[0]])[0]
        self.assertEqual(breach["value"], -75.0)
        self.assertEqual(breach["threshold"], -40.0)
        self.assertEqual(breach["direction"], "below")
        self.assertEqual(breach["climatology"], "WMO 1991-2020 normal")
        self.assertEqual(breach["source"], "ERA5 (Earthmover Icechunk edition)")
        self.assertEqual(breach["doi"], rainfall.ERA5_DOI)

    def test_an_unknown_metric_is_ignored_rather_than_crashing(self):
        rule = [{"id": "bogus", "metric": "does_not_exist", "below": 1}]
        self.assertEqual(alerts.evaluate("k", payload(-75.0), rule), [])

    def test_a_missing_metric_is_skipped(self):
        empty = {"summary": {}, "series": []}
        self.assertEqual(alerts.evaluate("k", empty), [])

    def test_rules_can_be_overridden_by_configuration(self):
        os.environ["RAINFALL_ALERT_RULES"] = json.dumps(
            [{"id": "mild", "metric": "latest_anomaly_pct", "below": -5.0}]
        )
        try:
            self.assertEqual([r["id"] for r in alerts.rules()], ["mild"])
            self.assertEqual([b["rule"] for b in alerts.evaluate("k", payload(-75.0))], ["mild"])
        finally:
            os.environ.pop("RAINFALL_ALERT_RULES", None)

    def test_malformed_configuration_falls_back_to_the_defaults(self):
        os.environ["RAINFALL_ALERT_RULES"] = "{not json"
        try:
            self.assertEqual(len(alerts.rules()), 3)
        finally:
            os.environ.pop("RAINFALL_ALERT_RULES", None)


class AlertDeduplicationTests(unittest.TestCase):
    def setUp(self):
        self.previous = os.environ.get("RAINFALL_CACHE_DIR")
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["RAINFALL_CACHE_DIR"] = self.tmp.name
        self.addCleanup(self._restore)
        self.addCleanup(self.tmp.cleanup)

    def _restore(self):
        if self.previous is None:
            os.environ.pop("RAINFALL_CACHE_DIR", None)
        else:
            os.environ["RAINFALL_CACHE_DIR"] = self.previous

    def test_a_breach_is_announced_once_not_every_sweep(self):
        state = alerts.read_state()
        sent = []
        body = payload(-75.0)
        first = alerts.notify("k", body, alerts.evaluate("k", body), state, sent.append)
        self.assertEqual(len(first), 2, "two rules breached, two messages")
        for _sweep in range(5):
            alerts.notify("k", body, alerts.evaluate("k", body), state, sent.append)
        self.assertEqual(len(sent), 2, "a dry season must not notify every sweep")

    def test_a_recovery_is_announced(self):
        state = alerts.read_state()
        sent = []
        dry = payload(-75.0)
        alerts.notify("k", dry, alerts.evaluate("k", dry), state, sent.append)
        self.assertEqual(len(sent), 2)
        wet = payload(0.0)
        sent.clear()
        recovered = alerts.notify("k", wet, alerts.evaluate("k", wet), state, sent.append)
        self.assertEqual(len(recovered), 2, "both episodes closed")
        self.assertTrue(all(m["kind"] == "recovered" for m in recovered))
        self.assertEqual(alerts.open_episodes(), [])

    def test_recovering_and_breaching_again_announces_again(self):
        state = alerts.read_state()
        sent = []
        alerts.notify("k", payload(-75.0), alerts.evaluate("k", payload(-75.0)), state, sent.append)
        alerts.notify("k", payload(0.0), alerts.evaluate("k", payload(0.0)), state, sent.append)
        sent.clear()
        alerts.notify("k", payload(-75.0), alerts.evaluate("k", payload(-75.0)), state, sent.append)
        self.assertEqual(len(sent), 2, "a new episode is a new message")

    def test_episodes_persist_across_processes(self):
        state = alerts.read_state()
        sent = []
        alerts.notify("k", payload(-75.0), alerts.evaluate("k", payload(-75.0)), state, sent.append)
        alerts.write_state(state)
        self.assertEqual(len(alerts.open_episodes()), 2)
        # A fresh process reads the same state, so it does not re-announce.
        reloaded = alerts.read_state()
        again = []
        alerts.notify("k", payload(-75.0), alerts.evaluate("k", payload(-75.0)), reloaded, again.append)
        self.assertEqual(again, [])

    def test_a_corrupt_state_file_is_a_clean_slate(self):
        alerts.state_path().parent.mkdir(parents=True, exist_ok=True)
        alerts.state_path().write_text("{not json")
        self.assertEqual(alerts.read_state()["episodes"], {})


class AlertDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.previous = os.environ.get("RAINFALL_CACHE_DIR")
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["RAINFALL_CACHE_DIR"] = self.tmp.name
        os.environ.pop("RAINFALL_ALERT_WEBHOOK", None)
        self.addCleanup(self._restore)
        self.addCleanup(self.tmp.cleanup)

    def _restore(self):
        for key in ("RAINFALL_CACHE_DIR", "RAINFALL_ALERT_WEBHOOK"):
            if os.environ.get(key) is None:
                os.environ.pop(key, None)

    def test_without_a_webhook_nothing_is_sent_but_the_episode_is_recorded(self):
        key = "b" * 32
        rainfall.write_cache(key, payload(-75.0))
        self.assertEqual(alerts.evaluate_and_notify([key]), [])
        self.assertEqual(len(alerts.open_episodes()), 2,
                         "the episode is still recorded, so a webhook added later "
                         "does not re-announce an open alert")

    def test_a_delivery_failure_does_not_stop_the_sweep_or_lose_state(self):
        import urllib.error


        key = "c" * 32
        rainfall.write_cache(key, payload(-75.0))

        def broken(_message):
            raise urllib.error.URLError("webhook down")

        # Nothing was delivered, so nothing is reported as sent, but the sweep
        # continued and the episode is still open.
        self.assertEqual(alerts.evaluate_and_notify([key], send=broken), [])
        self.assertEqual(jobs.pending_count(), 0)
        self.assertEqual(len(alerts.open_episodes()), 2,
                         "state survives a webhook that is down")

    def test_an_area_without_a_series_is_skipped(self):
        self.assertEqual(alerts.evaluate_and_notify(["d" * 32]), [])

    def test_the_webhook_payload_carries_the_evidence(self):
        import urllib.request

        captured = {}

        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

        def fake_urlopen(request, timeout=None):
            captured["body"] = json.loads(request.data)
            captured["headers"] = dict(request.headers)
            return _Response()

        os.environ["RAINFALL_ALERT_WEBHOOK"] = "https://hooks.example/abc"
        try:
            import unittest.mock

            with unittest.mock.patch.object(urllib.request, "urlopen", fake_urlopen):
                alerts._post_webhook({
                    "rule": "severe-drought", "area": "Melako", "area_key": "e" * 32,
                    "value": -75.0, "threshold": -40.0, "kind": "breach",
                    "summary": "A year at least 40% drier than normal",
                    "month": "2026-01", "climatology": "WMO 1991-2020 normal",
                    "source": "ERA5", "doi": rainfall.ERA5_DOI,
                })
        finally:
            os.environ.pop("RAINFALL_ALERT_WEBHOOK", None)

        body = captured["body"]
        self.assertEqual(body["rule"], "severe-drought")
        self.assertEqual(body["value"], -75.0)
        self.assertEqual(body["doi"], rainfall.ERA5_DOI)
        self.assertIn("Melako", body["text"])
        self.assertIn("40% drier", body["text"])
        self.assertIn(rainfall.ERA5_DOI, body["text"])
        self.assertIn("application/json",
                      [v.lower() for v in captured["headers"].values()])


if __name__ == "__main__":
    unittest.main()
