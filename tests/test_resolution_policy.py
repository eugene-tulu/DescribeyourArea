"""Tests for the resolution policy and multi-indicator asynchronous work.

The area caps in main.py are a guard on the *request* budget, not a claim about
what the data can do. These lock the policy that replaces a refusal with a
coarser answer, and the honesty of the resolution it reports.
"""

import unittest

import indicators
import sensors
from sensors import COARSEST_RESOLUTION_M, RESOLUTION_STEPS, resolution_for_area


class ResolutionPolicyTests(unittest.TestCase):
    def test_the_policy_is_monotonic(self):
        # A larger area is never read at a finer resolution.
        areas = [1, 50, 99, 101, 250, 999, 1_001, 5_000, 9_999, 10_001, 100_000]
        chosen = [resolution_for_area(a)[0] for a in areas]
        self.assertEqual(chosen, sorted(chosen), chosen)

    def test_every_step_is_reachable(self):
        limits = [limit for limit, _m, _r in RESOLUTION_STEPS]
        self.assertEqual(limits, sorted(limits))
        self.assertEqual(limits[-1], 10_000.0)

    def test_a_small_area_keeps_the_composite_resolution(self):
        resolution, reason = resolution_for_area(60)
        self.assertEqual(resolution, 20)
        self.assertTrue(reason)

    def test_a_landscape_is_answered_rather_than_refused(self):
        # 2,462 km2 is the unit the conservancy audience asked for: Laikipia as a
        # landscape rather than a named conservancy.
        resolution, _reason = resolution_for_area(2_462)
        self.assertEqual(resolution, 100)
        self.assertLess(indicators.pixels_for(2_462, resolution), 1_000_000)

    def test_a_country_falls_back_to_the_coarsest_grid(self):
        resolution, reason = resolution_for_area(580_000)
        self.assertEqual(resolution, COARSEST_RESOLUTION_M)
        self.assertIn("beyond the synchronous budget", reason)

    def test_a_coarse_source_is_never_upsampled(self):
        # Asking a 250 m product for 20 m would invent detail.
        for area in (10, 2_462, 580_000):
            with self.subTest(area=area):
                self.assertGreaterEqual(resolution_for_area(area, native_m=250)[0], 250)

    def test_a_fine_source_uses_the_policy_resolution(self):
        for area in (60, 400, 5_000):
            with self.subTest(area=area):
                self.assertEqual(resolution_for_area(area, native_m=10)[0],
                                 resolution_for_area(area)[0])

    def test_landcover_gets_a_larger_budget_than_the_rest(self):
        # The only module whose memory genuinely grows with area.
        self.assertGreater(indicators._synchronous_budget_km2("landcover"),
                           indicators._synchronous_budget_km2("ndvi"))

    def test_the_async_path_does_not_enforce_the_request_budget(self):
        # The point of a worker is to answer what the request path refused, so
        # enforce_budget is the switch that makes a large area legal.
        import asyncio

        tiny = indicators._synchronous_budget_km2("ndvi")
        result = asyncio.run(indicators.compute_indicator(
            "ndvi", [36.4, -0.6, 36.9, -0.2],
            {"type": "Polygon", "coordinates": [[[36.4, -0.6], [36.9, -0.6], [36.9, -0.2], [36.4, -0.2], [36.4, -0.6]]]},
            area_km2=2_462, enforce_budget=False,
        ))
        # Either it computed, or it refused for a reason that is not the area cap.
        if result.get("status") != "ok":
            self.assertNotIn("km²", str(result.get("warning", "")))
        self.assertTrue(result.get("status") in {"ok", "unavailable", "skipped", "error"})


class MultiIndicatorJobTests(unittest.TestCase):
    """Two indicators for one area are two jobs, not one that replaces the other."""

    def setUp(self):
        import os
        import tempfile

        self.previous = os.environ.get("RAINFALL_CACHE_DIR")
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["RAINFALL_CACHE_DIR"] = self.tmp.name
        self.addCleanup(self._restore)
        self.addCleanup(self.tmp.cleanup)

    def _restore(self):
        import os

        if self.previous is None:
            os.environ.pop("RAINFALL_CACHE_DIR", None)
        else:
            os.environ["RAINFALL_CACHE_DIR"] = self.previous

    def test_the_supported_indicators_are_declared(self):
        import jobs

        self.assertEqual(
            set(jobs.INDICATORS),
            {"rainfall", "dem", "landcover", "ndvi", "vegetation_series"},
        )

    def test_artefacts_do_not_collide_across_indicators(self):
        import jobs

        key = "a" * 32
        jobs.write_artefact(key, "dem", {"status": "ok", "mean": 1.0})
        jobs.write_artefact(key, "ndvi", {"status": "ok", "mean": 2.0})
        self.assertEqual(jobs.read_artefact(key, "dem")["mean"], 1.0)
        self.assertEqual(jobs.read_artefact(key, "ndvi")["mean"], 2.0)

    def test_jobs_do_not_collide_across_indicators(self):
        import jobs

        geom = {"type": "Polygon", "coordinates": [[[35.1, -1.5], [35.11, -1.5], [35.11, -1.4], [35.1, -1.4], [35.1, -1.5]]]}
        for indicator in ("dem", "ndvi"):
            jobs.submit(geom, indicator=indicator)
        self.assertEqual(jobs.pending_count(), 2, "one area with two indicators is two jobs")

    def test_an_unknown_indicator_is_refused(self):
        import jobs

        with self.assertRaises(ValueError):
            jobs.submit({"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]},
                        indicator="nope")


if __name__ == "__main__":
    unittest.main()
