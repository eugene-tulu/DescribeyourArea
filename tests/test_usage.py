"""Tests for the usage-event schema and the rainfall forget endpoint.

The privacy properties are the tests worth having, because they are the reason the
module exists: no geometry can reach the log, the area is banded, and the client
address is truncated. They are asserted against the record shape, not against
convention, and ``assert_no_geometry`` walks the record so a future field cannot
reintroduce a polygon unnoticed.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import usage


class AreaBandTests(unittest.TestCase):
    def test_bands_align_with_the_shipped_caps(self):
        # The edges are the product's caps, so a band's population reads as
        # something about the product rather than only about its users.
        self.assertEqual(usage.AREA_BANDS[0][1], "0-10")
        self.assertEqual(usage.NDVI_CAP_KM2, 10.0)
        self.assertEqual(usage.SYNC_CAP_KM2, 100.0)
        self.assertEqual(usage.LANDCOVER_CAP_KM2, 1000.0)

    def test_area_is_bucketed_not_carried(self):
        self.assertEqual(usage.area_band(0.5), "0-10")
        self.assertEqual(usage.area_band(9.99), "0-10")
        self.assertEqual(usage.area_band(10.0), "10-100")
        self.assertEqual(usage.area_band(99.9), "10-100")
        self.assertEqual(usage.area_band(100.0), "100-1000")
        self.assertEqual(usage.area_band(999.9), "100-1000")
        self.assertEqual(usage.area_band(1000.0), "1000+")
        self.assertEqual(usage.area_band(5505.0), "1000+")

    def test_absent_area_is_unknown(self):
        self.assertEqual(usage.area_band(None), "unknown")


class ClientPrefixTests(unittest.TestCase):
    def test_ipv4_is_truncated_to_the_network_prefix(self):
        self.assertEqual(usage.client_prefix("197.14.22.33"), "197.14.22.0/24")
        self.assertEqual(usage.client_prefix("8.8.8.8"), "8.8.8.0/24")

    def test_ipv6_is_truncated_to_a_slash_64(self):
        self.assertTrue(usage.client_prefix("2001:db8:85a3:0:0:8a2e:370:7334").endswith("::/64"))

    def test_unusable_addresses_become_unknown(self):
        for value in (None, "", "   ", "not-an-ip", "1.2.3"):
            with self.subTest(value=value):
                self.assertEqual(usage.client_prefix(value), "unknown")


class OutcomeBandingTests(unittest.TestCase):
    """The verdict is kept; the measurement behind it is not."""

    def test_status_values_pass_through(self):
        self.assertEqual(usage.band_outcome({"status": "ok"}), "ok")
        self.assertEqual(usage.band_outcome({"status": "skipped"}), "skipped")
        self.assertEqual(usage.band_outcome({"status": "not_computed"}), "not_computed")

    def test_an_error_dict_reduces_to_its_verdict(self):
        # The detail names a data condition, which is a fact about a place.
        result = {"error": "no_valid_elevation_pixels"}
        self.assertEqual(usage.band_outcome(result), "error")
        self.assertNotIn("no_valid", str(usage.band_outcome(result)))

    def test_absent_and_plain_results(self):
        self.assertEqual(usage.band_outcome(None), "not_requested")
        self.assertEqual(usage.band_outcome({"mean": 42.0, "max": 100.0}), "ok")

    def test_a_demanding_result_never_carries_its_numbers(self):
        result = {"mean": 1650.25, "min": 1400.0, "max": 1900.0, "std": 90.1}
        self.assertEqual(usage.band_outcome(result), "ok")
        self.assertNotIn("1650", json.dumps(usage.band_outcome(result)))


class BuildEventTests(unittest.TestCase):
    def test_the_record_carries_no_geometry(self):
        event = usage.build_event(
            datasets_requested=["ndvi", "dem"],
            outcomes={"ndvi": {"status": "ok", "mean": 0.42}, "dem": None},
            duration_ms={"ndvi": 11400},
            total_ms=31200,
            area_km2=60.0,
            client_address="197.14.22.33",
            sensor="landsat",
        )
        usage.assert_no_geometry(event)
        self.assertEqual(event["aoi_area_km2_band"], "10-100")
        self.assertEqual(event["client_prefix"], "197.14.22.0/24")
        self.assertEqual(event["datasets_requested"], ["dem", "ndvi"])
        self.assertEqual(event["outcomes"], {"dem": "not_requested", "ndvi": "ok"})
        self.assertEqual(event["duration_ms"], {"ndvi": 11400})
        self.assertEqual(event["total_ms"], 31200)
        self.assertEqual(event["sensor"], "landsat")
        serialised = json.dumps(event)
        self.assertNotIn("60.0", serialised, "the raw area must not be present")
        self.assertNotIn("197.14.22.33", serialised)
        self.assertNotIn("0.42", serialised)

    def test_there_is_no_parameter_that_could_carry_a_geometry(self):
        import inspect

        parameters = set(inspect.signature(usage.build_event).parameters)
        for name in parameters:
            with self.subTest(parameter=name):
                self.assertNotIn(name, {"geometry", "geojson", "geom", "coordinates", "bbox"})

    def test_the_guard_catches_a_geometry_that_sneaks_in(self):
        with self.assertRaises(AssertionError):
            usage.assert_no_geometry({"outcomes": {}, "geometry": {"type": "Point"}})
        with self.assertRaises(AssertionError):
            usage.assert_no_geometry({"outcomes": {}, "bbox": [1, 2, 3, 4]})
        usage.assert_no_geometry({"aoi_area_km2_band": "0-10", "outcomes": {}})

    def test_optional_fields_may_be_absent(self):
        event = usage.build_event(datasets_requested=[], outcomes={})
        usage.assert_no_geometry(event)
        self.assertIsNone(event["total_ms"])
        self.assertIsNone(event["sensor"])
        self.assertEqual(event["client_prefix"], "unknown")
        self.assertEqual(event["duration_ms"], {})


class EmitTests(unittest.TestCase):
    def setUp(self):
        self.previous = os.environ.get("USAGE_EVENTS_PATH")
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "nested" / "events.jsonl"
        os.environ["USAGE_EVENTS_PATH"] = str(self.path)

    def tearDown(self):
        if self.previous is None:
            os.environ.pop("USAGE_EVENTS_PATH", None)
        else:
            os.environ["USAGE_EVENTS_PATH"] = self.previous
        self.tmp.cleanup()

    def test_events_append_as_json_lines(self):
        for index in range(3):
            usage.emit(usage.build_event(
                datasets_requested=["dem"], outcomes={"dem": {"status": "ok"}},
                total_ms=1000 + index, area_km2=5.0,
            ))
        lines = self.path.read_text().strip().splitlines()
        self.assertEqual(len(lines), 3)
        for line in lines:
            self.assertIsInstance(json.loads(line), dict)

    def test_a_write_failure_never_raises(self):
        os.environ["USAGE_EVENTS_PATH"] = "/proc/definitely/not/writable/events.jsonl"
        usage.emit(usage.build_event(datasets_requested=["dem"], outcomes={}))

    def test_summary_aggregates_without_any_single_area(self):
        events = [
            usage.build_event(datasets_requested=["ndvi"], outcomes={"ndvi": {"status": "ok"}},
                              total_ms=1000, area_km2=5.0, sensor="landsat"),
            usage.build_event(datasets_requested=["ndvi"], outcomes={"ndvi": {"status": "skipped"}},
                              total_ms=3000, area_km2=500.0, sensor="landsat"),
            usage.build_event(datasets_requested=["dem"], outcomes={"dem": None},
                              total_ms=5000, area_km2=5.0),
        ]
        summary = usage.summarise(events)
        self.assertEqual(summary["events"], 3)
        self.assertEqual(summary["outcomes"], {"not_requested": 1, "ok": 1, "skipped": 1})
        self.assertEqual(summary["area_bands"], {"0-10": 2, "100-1000": 1})
        self.assertEqual(summary["sensors"], {"landsat": 2})
        self.assertEqual(summary["total_ms"]["p50"], 3000)
        self.assertNotIn("5.0", json.dumps(summary))
        self.assertNotIn("500.0", json.dumps(summary))

    def test_summarise_handles_an_empty_batch(self):
        summary = usage.summarise([])
        self.assertEqual(summary["events"], 0)
        self.assertIsNone(summary["total_ms"]["p50"])

    def test_read_events_skips_corrupt_lines(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text('{"ok":1}\nnot json\n\n{"ok":2}\n')
        self.assertEqual(len(usage.read_events(self.path)), 2)

    def test_read_events_on_a_missing_file_is_empty(self):
        self.assertEqual(usage.read_events(Path(self.tmp.name) / "absent.jsonl"), [])


class TimerTests(unittest.TestCase):
    def test_timer_records_milliseconds(self):
        with usage.Timer() as timer:
            pass
        self.assertIsInstance(timer.elapsed_ms, int)
        self.assertGreaterEqual(timer.elapsed_ms, 0)

    def test_timer_records_even_when_the_body_raises(self):
        timer = usage.Timer()
        with self.assertRaises(ValueError):
            with timer:
                raise ValueError("boom")
        self.assertIsNotNone(timer.elapsed_ms)


class AreaHelperTests(unittest.TestCase):
    def test_area_is_derived_from_a_geometry_and_only_a_number_returns(self):
        geom = {"type": "Polygon", "coordinates": [[
            [35.10, -1.55], [35.17, -1.55], [35.17, -1.48], [35.10, -1.48], [35.10, -1.55]]]}
        area = usage.area_km2_of(geom)
        self.assertIsInstance(area, float)
        self.assertTrue(40 < area < 90, area)
        self.assertIsNone(usage.area_km2_of(None))


if __name__ == "__main__":
    unittest.main()


class ForgetEndpointTests(unittest.TestCase):
    """A removal request has to be honourable, or the privacy claim is hollow."""

    AOI = {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[
        [35.10, -1.55], [35.17, -1.55], [35.17, -1.48], [35.10, -1.48], [35.10, -1.55]]]}}

    def setUp(self):
        self.previous = os.environ.get("RAINFALL_CACHE_DIR")
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["RAINFALL_CACHE_DIR"] = self.tmp.name
        import main

        self.main = main
        from fastapi.testclient import TestClient

        self.client = TestClient(main.app)
        import rainfall

        self.rainfall = rainfall
        self.key = rainfall.geometry_hash(self.AOI["geometry"])
        rainfall.write_cache(self.key, {
            "processing_version": rainfall.RAINFALL_PROCESSING_VERSION,
            "indicator": "monthly_precipitation",
            "series": [{"month": "2020-01", "precip_mm": 1.0}],
        })

    def tearDown(self):
        if self.previous is None:
            os.environ.pop("RAINFALL_CACHE_DIR", None)
        else:
            os.environ["RAINFALL_CACHE_DIR"] = self.previous
        self.tmp.cleanup()

    def test_a_stored_series_is_removed_by_geometry(self):
        self.assertIsNotNone(self.rainfall.read_cache(self.key))
        response = self.client.post("/admin/rainfall/forget", json={"geojson": self.AOI})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(body["removed"])
        self.assertEqual(body["cache_key"], self.key)
        self.assertIsNone(self.rainfall.read_cache(self.key))
        self.assertFalse(self.rainfall.cache_path(self.key).exists())

    def test_a_stored_series_is_removed_by_key(self):
        response = self.client.post("/admin/rainfall/forget", json={"cache_key": self.key})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["removed"])

    def test_removal_is_idempotent(self):
        self.client.post("/admin/rainfall/forget", json={"cache_key": self.key})
        response = self.client.post("/admin/rainfall/forget", json={"cache_key": self.key})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["removed"])

    def test_only_the_named_area_is_affected(self):
        other = self.key[:-1] + ("0" if self.key[-1] != "0" else "1")
        self.rainfall.write_cache(other, {
            "processing_version": self.rainfall.RAINFALL_PROCESSING_VERSION,
            "indicator": "monthly_precipitation",
            "series": [{"month": "2020-01", "precip_mm": 2.0}],
        })
        self.client.post("/admin/rainfall/forget", json={"cache_key": self.key})
        self.assertIsNone(self.rainfall.read_cache(self.key))
        self.assertIsNotNone(self.rainfall.read_cache(other),
                             "a sibling area must survive the removal")

    def test_a_valid_but_absent_key_is_a_no_op(self):
        absent = "0" * 32
        response = self.client.post("/admin/rainfall/forget", json={"cache_key": absent})
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["removed"])

    def test_a_traversal_key_is_refused(self):
        for key in ("../../etc/passwd", "..", "z" * 32, "ABCDEF" + "0" * 26, "0" * 31, "",
                    "0" * 33, "0" * 32 + "/../x"):
            with self.subTest(key=key):
                response = self.client.post("/admin/rainfall/forget", json={"cache_key": key})
                self.assertEqual(response.status_code, 422, f"{key!r} was not refused")
        self.assertIsNotNone(self.rainfall.read_cache(self.key), "nothing should have been removed")

    def test_a_non_polygon_geometry_is_refused(self):
        for geometry in ({"type": "Point", "coordinates": [0, 0]},
                         {"type": "LineString", "coordinates": [[0, 0], [1, 1]]},
                         "not-a-geometry"):
            with self.subTest(geometry=geometry):
                response = self.client.post(
                    "/admin/rainfall/forget", json={"geojson": {"type": "Feature", "properties": {},
                                                                "geometry": geometry}})
                self.assertEqual(response.status_code, 422)

    def test_an_empty_request_is_refused(self):
        self.assertEqual(self.client.post("/admin/rainfall/forget", json={}).status_code, 422)

    def test_a_bare_geometry_is_accepted_as_well_as_a_feature(self):
        response = self.client.post("/admin/rainfall/forget", json={"geojson": self.AOI["geometry"]})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["removed"])


class EmitSiteTests(unittest.TestCase):
    """The endpoint must actually record an event, and record nothing sensitive."""

    def setUp(self):
        self.previous = os.environ.get("USAGE_EVENTS_PATH")
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "events.jsonl"
        os.environ["USAGE_EVENTS_PATH"] = str(self.path)
        os.environ["RAINFALL_CACHE_DIR"] = self.tmp.name

    def tearDown(self):
        for key, value in (("USAGE_EVENTS_PATH", self.previous),):
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        os.environ.pop("RAINFALL_CACHE_DIR", None)
        self.tmp.cleanup()

    def test_a_rejected_request_records_nothing(self):
        # Validation failures happen before the emit site, so nothing is written.
        from fastapi.testclient import TestClient

        import main

        client = TestClient(main.app)
        response = client.post("/generate-context?datasets=nonsense",
                               json={"geojson": {"type": "Polygon", "coordinates": []}})
        self.assertEqual(response.status_code, 422)
        self.assertFalse(self.path.exists(), "a rejected request must not be recorded")

    def test_the_event_helper_bands_and_truncates(self):
        import main

        main._emit_usage_event(
            requested={"ndvi"}, outcomes={"ndvi": {"status": "ok", "mean": 0.42}},
            module_ms={"ndvi": 1234}, request_timer=usage.Timer(),
            http_request=None, area_km2=60.0,
            ndvi_stats={"sensor": {"id": "landsat"}},
        )
        self.assertTrue(self.path.exists())
        event = json.loads(self.path.read_text().strip())
        usage.assert_no_geometry(event)
        self.assertEqual(event["aoi_area_km2_band"], "10-100")
        self.assertEqual(event["client_prefix"], "unknown")
        self.assertEqual(event["outcomes"], {"ndvi": "ok"})
        self.assertEqual(event["sensor"], "landsat")
        self.assertNotIn("0.42", self.path.read_text())

    def test_the_helper_never_raises(self):
        import main

        # A broken event must not be able to fail a request that already succeeded.
        main._emit_usage_event(
            requested={"ndvi"}, outcomes={"ndvi": None}, module_ms=None,
            request_timer=object(), http_request=object(), area_km2=None, ndvi_stats=None,
        )


class ClientAddressTests(unittest.TestCase):
    """Behind a loopback-bound proxy the peer is the proxy, not the caller."""

    class _Headers(dict):
        def get(self, key, default=None):
            return dict.get(self, key.lower(), dict.get(self, key, default))

    class _Stub:
        def __init__(self, host, forwarded=None):
            self.client = type("C", (), {"host": host})() if host else None
            if forwarded is not None:
                self.headers = type("H", (dict,), {
                    "get": lambda self, k, d=None: forwarded if k.lower() == "x-forwarded-for" else d
                })()
            else:
                self.headers = {}

    def test_a_direct_peer_is_used_as_is(self):
        import main

        self.assertEqual(main._client_host(self._Stub("203.0.113.9")), "203.0.113.9")

    def test_a_loopback_peer_defers_to_the_forwarded_chain(self):
        import main

        stub = self._Stub("172.18.0.5", forwarded="203.0.113.9, 10.0.0.5")
        self.assertEqual(main._client_host(stub), "203.0.113.9")

    def test_private_entries_in_the_chain_are_skipped(self):
        import main

        stub = self._Stub("127.0.0.1", forwarded="10.0.0.5, 192.168.1.9, 203.0.113.9")
        self.assertEqual(main._client_host(stub), "203.0.113.9")

    def test_a_client_cannot_forge_the_header_from_a_public_peer(self):
        # Only trusted when the peer is private; a public peer is reported itself.
        import main

        stub = self._Stub("198.51.100.4", forwarded="10.0.0.5")
        self.assertEqual(main._client_host(stub), "198.51.100.4")

    def test_no_peer_and_no_header_is_unknown(self):
        import main

        self.assertIsNone(main._client_host(self._Stub(None)))
        self.assertIsNone(main._client_host(None))

    def test_the_emitted_prefix_is_still_truncated(self):
        import main
        import usage

        stub = self._Stub("127.0.0.1", forwarded="203.0.113.9")
        self.assertEqual(usage.client_prefix(main._client_host(stub)), "203.0.113.0/24")


class TimedHelperTests(unittest.TestCase):
    def test_a_timed_coroutine_records_a_plausible_duration(self):
        import asyncio

        import main
        import usage

        async def scenario():
            timer = usage.Timer()
            result = await main._timed(timer, _sleep(0.05))
            return result, timer.elapsed_ms

        result, elapsed = asyncio.run(scenario())
        self.assertEqual(result, "done")
        # An unentered timer would report the interval since the epoch.
        self.assertLess(elapsed, 5_000)
        self.assertGreaterEqual(elapsed, 40)

    def test_the_timer_is_closed_even_when_the_coroutine_raises(self):
        import asyncio

        import main
        import usage

        async def scenario():
            timer = usage.Timer()
            with self.assertRaises(RuntimeError):
                await main._timed(timer, _boom())
            return timer.elapsed_ms

        self.assertIsNotNone(asyncio.run(scenario()))


async def _sleep(seconds):
    import asyncio

    await asyncio.sleep(seconds)
    return "done"


async def _boom():
    raise RuntimeError("boom")
