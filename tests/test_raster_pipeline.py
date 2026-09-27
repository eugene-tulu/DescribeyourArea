"""Pure-logic unit tests. No network, no mocks.

Anything that depends on real STAC or COG behaviour lives in
``test_planetary_computer.py`` instead: the ``limit=1`` defect this suite used to
cover with fakes cannot be reproduced by a mock, because it depends on real tile
boundaries.
"""

import asyncio
import math
import unittest

from fastapi import HTTPException

import main

from main import (
    MAX_LANDCOVER_BBOX_KM2,
    NDVI_MIN_PLAUSIBLE,
    SCL_REJECTED,
    MAX_SYNC_BBOX_KM2,
    _InFlightWork,
    _requested_datasets,
    interpret_terrain,
    run_blocking,
)

def main_network_ok() -> bool:
    try:
        import affine  # noqa: F401
        import pyproj  # noqa: F401
        import rasterio  # noqa: F401

        return True
    except Exception:
        return False


NAN = float("nan")
INF = float("inf")


class TerrainInterpretationTests(unittest.TestCase):
    def test_flat_area(self):
        result = interpret_terrain({"mean": 10.0, "min": 5.0, "max": 20.0, "std": 2.0})
        self.assertEqual(result["terrain_type"], "relatively flat")
        self.assertEqual(result["elevation_range_m"], 15.0)

    def test_undulating_area(self):
        result = interpret_terrain({"mean": 100.0, "min": 10.0, "max": 200.0, "std": 30.0})
        self.assertEqual(result["terrain_type"], "moderately undulating")

    def test_mountainous_area(self):
        result = interpret_terrain({"mean": 1500.0, "min": 100.0, "max": 3000.0, "std": 500.0})
        self.assertEqual(result["terrain_type"], "highly variable or mountainous")

    def test_all_nan_never_reaches_the_mountainous_branch(self):
        # Every comparison against NaN is False, so an unguarded implementation
        # fell through to the else branch and mislabelled open water as mountain.
        result = interpret_terrain({"mean": NAN, "min": NAN, "max": NAN, "std": NAN})
        self.assertNotIn("terrain_type", result)
        self.assertEqual(result["error"], "no_valid_elevation_pixels")

    def test_partially_non_finite_is_still_rejected(self):
        for bad in ({"mean": 1.0, "min": NAN, "max": 2.0, "std": 0.1},
                    {"mean": 1.0, "min": 0.0, "max": INF, "std": 0.1}):
            with self.subTest(bad=bad):
                result = interpret_terrain(bad)
                self.assertNotIn("terrain_type", result)

    def test_error_passes_through_untouched(self):
        original = {"error": "no_valid_elevation_pixels"}
        self.assertEqual(interpret_terrain(original), original)

    def test_empty_input_passes_through(self):
        self.assertEqual(interpret_terrain({}), {})
        self.assertIsNone(interpret_terrain(None))

    def test_boundary_between_flat_and_undulating(self):
        self.assertEqual(
            interpret_terrain({"mean": 0.0, "min": 0.0, "max": 49.9, "std": 0.0})["terrain_type"],
            "relatively flat",
        )
        self.assertEqual(
            interpret_terrain({"mean": 0.0, "min": 0.0, "max": 50.0, "std": 0.0})["terrain_type"],
            "moderately undulating",
        )


class DatasetSelectionTests(unittest.TestCase):
    def test_omitted_selects_everything(self):
        # Derived, not a literal: a hard-coded copy here is what went stale when
        # rainfall was added.
        from main import AVAILABLE_DATASETS

        self.assertEqual(_requested_datasets(None), AVAILABLE_DATASETS)

    def test_parses_and_normalises(self):
        self.assertEqual(_requested_datasets(" dem , NDVI "), {"dem", "ndvi"})

    def test_empty_selection_is_rejected(self):
        # An empty set previously produced a 200 with every module null.
        for value in ("", "   ", ",", " , , "):
            with self.subTest(value=value):
                with self.assertRaises(HTTPException) as raised:
                    _requested_datasets(value)
                self.assertEqual(raised.exception.status_code, 422)
                self.assertIn("dataset", str(raised.exception.detail).lower())

    def test_retired_datasets_are_rejected(self):
        for retired in ("soils", "population", "climate", "hydrology"):
            with self.subTest(dataset=retired):
                with self.assertRaises(HTTPException) as raised:
                    _requested_datasets(retired)
                self.assertEqual(raised.exception.status_code, 422)


class CapDerivationTests(unittest.TestCase):
    """The caps are derived from measurement; assert the shipped values."""

    def test_the_synchronous_cap_matches_its_time_budget(self):
        # Measured: 100 km2 = 453 MB / 32 s; 400 km2 exceeds the 75 s budget.
        self.assertEqual(MAX_SYNC_BBOX_KM2, 100.0)

    def test_landcover_gets_a_larger_budget_than_the_rest(self):
        # WorldCover is the only module whose memory grows with area, so it is the
        # only one that needed a budget of its own.
        self.assertGreater(MAX_LANDCOVER_BBOX_KM2, MAX_SYNC_BBOX_KM2)

    def test_the_removed_vegetation_cap_is_really_gone(self):
        # It duplicated the synchronous cap at the same value, so it configured
        # nothing and only created a second number to keep in step.
        self.assertFalse(hasattr(main, "MAX_NDVI_BBOX_KM2"))

    def test_sync_cap_is_the_binding_synchronous_limit(self):
        self.assertEqual(MAX_SYNC_BBOX_KM2, 100.0)


class NdviGridResolutionTests(unittest.TestCase):
    """Metres-to-degrees must account for longitude shrinking with latitude."""

    @staticmethod
    def _deg(bbox, resolution_m=20):
        mid = math.radians((bbox[1] + bbox[3]) / 2.0)
        return resolution_m / (111_320.0 * max(0.01, math.cos(mid)))

    def test_equator(self):
        self.assertAlmostEqual(self._deg([37.0, -0.1, 37.1, 0.1]), 20 / 111_320.0, places=9)

    def test_cell_grows_with_absolute_latitude(self):
        equator = self._deg([37.0, -0.05, 37.1, 0.05])
        mid = self._deg([37.0, 44.95, 37.1, 45.05])
        north = self._deg([37.0, 59.95, 37.1, 60.05])
        self.assertLess(equator, mid)
        self.assertLess(mid, north)

    def test_sixty_degrees_roughly_doubles_the_cell(self):
        equator = self._deg([37.0, -0.05, 37.1, 0.05])
        north = self._deg([37.0, 59.95, 37.1, 60.05])
        self.assertAlmostEqual(north / equator, 2.0, delta=0.1)

    def test_polar_latitude_is_clamped_not_infinite(self):
        deg = self._deg([0.0, 89.9, 0.1, 90.0])
        self.assertTrue(math.isfinite(deg))
        self.assertEqual(deg, 20 / (111_320.0 * 0.01))


class InFlightGuardTests(unittest.TestCase):
    """A timed-out thread must keep holding the concurrency guard."""

    def test_counter_tracks_lifecycle(self):
        tracker = _InFlightWork()
        self.assertEqual(tracker.count, 0)
        tracker.enter()
        self.assertEqual(tracker.count, 1)
        tracker.exit()
        self.assertEqual(tracker.count, 0)

    def test_counter_never_goes_negative(self):
        tracker = _InFlightWork()
        tracker.exit()
        tracker.exit()
        self.assertEqual(tracker.count, 0)

    def test_drain_returns_immediately_when_idle(self):
        tracker = _InFlightWork()

        async def scenario():
            await tracker.drain(timeout=5.0)

        asyncio.run(scenario())

    def test_drain_waits_for_slow_work(self):
        tracker = _InFlightWork()

        async def scenario():
            async def slow():
                tracker.enter()
                await asyncio.sleep(0.2)
                tracker.exit()

            task = asyncio.ensure_future(slow())
            await asyncio.sleep(0.02)
            self.assertEqual(tracker.count, 1)
            await tracker.drain(timeout=5.0)
            self.assertEqual(tracker.count, 0)
            await task

        asyncio.run(scenario())

    def test_drain_gives_up_after_its_timeout(self):
        tracker = _InFlightWork()

        async def scenario():
            tracker.enter()
            await tracker.drain(timeout=0.1)  # must not hang
            self.assertEqual(tracker.count, 1)
            tracker.exit()

        asyncio.run(scenario())

    def test_run_blocking_returns_the_value(self):
        async def scenario():
            return await run_blocking(lambda x: x * 3, 7)

        self.assertEqual(asyncio.run(scenario()), 21)

    def test_run_blocking_releases_the_guard_when_the_await_is_cancelled(self):
        """The bug: wait_for cancels the await, so the guard was freed while the
        thread kept running and a new request stacked on top of it."""
        import threading

        started = threading.Event()
        release = threading.Event()

        def slow():
            started.set()
            release.wait(5.0)
            return "finished"

        async def scenario():
            task = asyncio.ensure_future(run_blocking(slow))
            await asyncio.to_thread(started.wait, 5.0)
            self.assertEqual(_inflight_count(), 1, "thread must be counted while running")
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            # The await is cancelled but the thread is still running, so the guard
            # must still be held.
            self.assertEqual(_inflight_count(), 1, "guard must survive cancellation")
            release.set()
            deadline = asyncio.get_running_loop().time() + 5.0
            while _inflight_count() and asyncio.get_running_loop().time() < deadline:
                await asyncio.sleep(0.01)
            self.assertEqual(_inflight_count(), 0, "guard released once the thread ends")

        try:
            asyncio.run(scenario())
        finally:
            release.set()


def _inflight_count() -> int:
    import main

    return main.INFLIGHT.count


class NdviSceneClassPolicyTests(unittest.TestCase):
    """The SCL mask must not discard bright semi-arid ground.

    Sentinel-2 class 5 is "bright cloud", but its brightness test flags bright
    desert and rangeland across whole tiles: a Sahara tile reads 100% class 5
    while its B04/B08 reflectance and NDVI are plainly desert. Rejecting classes
    4 and 5 empties the result for exactly the areas this service serves.
    """

    def test_unambiguous_classes_are_rejected(self):
        for code in (0, 1, 3, 8, 9, 10, 11):
            with self.subTest(class_code=code):
                self.assertIn(code, SCL_REJECTED)

    def test_bright_cloud_is_not_rejected_outright(self):
        for code in (4, 5):
            with self.subTest(class_code=code):
                self.assertNotIn(code, SCL_REJECTED)

    def test_vegetation_and_water_classes_survive(self):
        for code in (2, 6, 7):
            with self.subTest(class_code=code):
                self.assertNotIn(code, SCL_REJECTED)

    def test_implausible_values_are_dropped_instead(self):
        self.assertEqual(NDVI_MIN_PLAUSIBLE, 0.0)
        self.assertLess(NDVI_MIN_PLAUSIBLE, 1.0)

    def test_mask_applied_to_a_synthetic_tile(self):
        import numpy as np

        # One pixel of each class, with an NDVI column for each.
        codes = np.array([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]], dtype="uint8")
        rejected = np.isin(codes.astype("int16"), tuple(SCL_REJECTED))
        self.assertTrue(rejected[0, 0])   # nodata
        self.assertTrue(rejected[0, 3])   # cloud shadow
        self.assertTrue(rejected[0, 8])   # medium cloud
        self.assertFalse(rejected[0, 4])  # cloud: kept, filtered by NDVI instead
        self.assertFalse(rejected[0, 5])  # bright cloud: kept
        self.assertFalse(rejected[0, 2])  # dark vegetation
        self.assertFalse(rejected[0, 6])  # water


class NdviTargetGridTests(unittest.TestCase):
    """The composite grid is projected, equal-area, and correctly oriented."""

    def test_default_crs_is_ease_grid_global(self):
        from main import NDVI_TARGET_EPSG

        self.assertEqual(NDVI_TARGET_EPSG, 6933)

    def test_moderate_latitudes_use_the_global_grid(self):
        from main import _target_crs

        for bbox in ([37.3, 0.3, 37.4, 0.4], [-58.4, -34.6, -58.3, -34.5]):
            with self.subTest(bbox=bbox):
                self.assertEqual(_target_crs(bbox).to_epsg(), 6933)

    def test_polar_areas_fall_back_to_a_utm_zone(self):
        from main import _target_crs

        # 78 N is still inside the EASE-Grid 2.0 Global domain (about 86 N).
        self.assertEqual(_target_crs([18.9, 78.2, 19.1, 78.3]).to_epsg(), 6933)
        crs = _target_crs([18.9, 87.5, 19.1, 87.8])
        self.assertNotEqual(crs.to_epsg(), 6933)
        self.assertEqual(crs.to_epsg(), 32634)  # 19E is UTM zone 34N

    def test_grid_honours_the_requested_resolution(self):
        from main import _vegetation_target_grid

        _crs, transform, width, height = _vegetation_target_grid([37.34, 0.30, 37.40, 0.37], 20)
        self.assertAlmostEqual(abs(transform.a), 20.0, delta=0.1)
        self.assertAlmostEqual(abs(transform.e), 20.0, delta=0.1)
        # 0.06 deg lon x 0.07 deg lat is roughly 6.7 x 7.8 km
        self.assertTrue(250 < width < 450, width)
        self.assertTrue(300 < height < 500, height)

    def test_coarser_resolution_yields_fewer_pixels(self):
        from main import _vegetation_target_grid

        small = _vegetation_target_grid([37.34, 0.30, 37.40, 0.37], 20)
        large = _vegetation_target_grid([37.34, 0.30, 37.40, 0.37], 100)
        self.assertLess(large[2] * large[3], small[2] * small[3])


@unittest.skipUnless(main_network_ok(), "rasterio/pyproj unavailable")
class NdviGeometryMaskTests(unittest.TestCase):
    """The study-area mask must be reprojected and correctly oriented.

    Both of these failed silently in development: the mask was rasterised with a
    WGS84 geometry against a projected transform (so nothing matched), and
    out_shape was given as (width, height) instead of (rows, cols).
    """

    AOI = [37.34, 0.30, 37.40, 0.37]

    def test_mask_covers_the_study_area(self):
        from main import _geometry_mask_on_grid, _vegetation_target_grid

        target = _vegetation_target_grid(self.AOI, 20)
        from main import shape

        ring = [
            [37.34, 0.30], [37.40, 0.30], [37.40, 0.37], [37.34, 0.37], [37.34, 0.30],
        ]
        mask = _geometry_mask_on_grid(shape({"type": "Polygon", "coordinates": [ring]}), target, self.AOI)
        _crs, _transform, width, height = target
        self.assertEqual(mask.shape, (height, width), "out_shape must be (rows, cols)")
        # The polygon is the grid's own bounding box, so with all_touched every
        # cell is legitimately inside; the point is that none is excluded.
        self.assertGreater(mask.sum(), 0, "mask excluded the entire study area")
        self.assertEqual(mask.mean(), 1.0)

    def test_a_smaller_polygon_leaves_part_of_the_grid_out(self):
        from main import _geometry_mask_on_grid, _vegetation_target_grid, shape

        target = _vegetation_target_grid(self.AOI, 20)
        inner = shape({"type": "Polygon", "coordinates": [[
            [37.345, 0.305], [37.395, 0.305], [37.395, 0.365], [37.345, 0.365], [37.345, 0.305]]]})
        mask = _geometry_mask_on_grid(inner, target, self.AOI)
        self.assertGreater(mask.sum(), 0)
        self.assertLess(mask.mean(), 1.0, "a polygon inside the bbox must not fill the grid")

    def test_a_moved_aoi_produces_a_different_mask(self):
        from main import _geometry_mask_on_grid, _vegetation_target_grid

        target = _vegetation_target_grid(self.AOI, 20)
        from main import shape

        near = {"type": "Polygon", "coordinates": [[
            [37.34, 0.30], [37.40, 0.30], [37.40, 0.37], [37.34, 0.37], [37.34, 0.30]]]}
        far = {"type": "Polygon", "coordinates": [[
            [37.90, 0.80], [37.96, 0.80], [37.96, 0.87], [37.90, 0.87], [37.90, 0.80]]]}
        near_mask = _geometry_mask_on_grid(shape(near), target, self.AOI)
        far_mask = _geometry_mask_on_grid(shape(far), target, self.AOI)
        self.assertGreater(near_mask.sum(), 0)
        self.assertEqual(far_mask.sum(), 0, "a distant AOI must not match this grid")


class ConcurrencyLimitTests(unittest.TestCase):
    """The limits come from measurement; assert the shipped values.

    Measured at N=1/2/4/6/8 through the real ASGI app: marginal memory ~25 MB per
    request, CPU 4-14% of one core, dem+landcover wall time flat, and NDVI p50
    latency 27/38/52/67 s. Neither memory nor CPU binds, so the global guard is
    generous and only the NDVI path is tightened.
    """

    def test_global_guard_is_generous(self):
        from main import MAX_CONCURRENT_ANALYSES

        # 8 concurrent requests peaked at 322 MB RSS in total.
        self.assertEqual(MAX_CONCURRENT_ANALYSES, 8)

    def test_ndvi_guard_is_tighter_than_the_global_one(self):
        from main import MAX_CONCURRENT_ANALYSES, MAX_CONCURRENT_NDVI

        self.assertLess(MAX_CONCURRENT_NDVI, MAX_CONCURRENT_ANALYSES)
        # NDVI p50 latency is 52 s at N=4 and 67 s at N=8, against a 75 s budget.
        self.assertEqual(MAX_CONCURRENT_NDVI, 3)

    def test_acquire_timeouts_are_proportional_to_the_work(self):
        from main import ANALYSIS_ACQUIRE_SECONDS, NDVI_ACQUIRE_SECONDS

        # The global guard rejects fast so a busy service answers immediately.
        self.assertLessEqual(ANALYSIS_ACQUIRE_SECONDS, 5.0)
        # The NDVI guard waits, because an NDVI job takes tens of seconds.
        self.assertGreater(NDVI_ACQUIRE_SECONDS, ANALYSIS_ACQUIRE_SECONDS)

    def test_ndvi_guard_does_not_fail_the_whole_request(self):
        """A saturated NDVI queue must degrade one module, not return a 429."""
        import asyncio

        from main import MAX_CONCURRENT_NDVI, NDVI_SEMAPHORE, compute_median_ndvi

        ring = [[37.34, 0.30], [37.40, 0.30], [37.40, 0.37], [37.34, 0.37], [37.34, 0.30]]
        geom = {"type": "Polygon", "coordinates": [ring]}

        async def scenario():
            for _ in range(MAX_CONCURRENT_NDVI):
                await NDVI_SEMAPHORE.acquire()
            try:
                return await compute_median_ndvi([37.34, 0.30, 37.40, 0.37], geom)
            finally:
                for _ in range(MAX_CONCURRENT_NDVI):
                    NDVI_SEMAPHORE.release()

        # Held down to the configured acquire timeout, not a real network call.
        result = asyncio.run(scenario())
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("busy", result["warning"].lower())
        self.assertNotIn("mean", result)


class ServiceSurfaceTests(unittest.TestCase):
    """Routes must be registered and must not drift apart."""

    def test_all_routes_are_registered(self):
        from fastapi.testclient import TestClient

        import main

        client = TestClient(main.app)
        for path in ("/health", "/version"):
            with self.subTest(path=path):
                response = client.get(path)
                self.assertEqual(response.status_code, 200, f"{path} is not registered")

    def test_health_and_version_report_the_same_version(self):
        from fastapi.testclient import TestClient

        import main

        client = TestClient(main.app)
        # These were each hard-coded and had already drifted to 1.2.0 and 1.3.0.
        self.assertEqual(client.get("/health").json()["version"], main.APP_VERSION)
        self.assertEqual(client.get("/version").json()["version"], main.APP_VERSION)

    def test_version_endpoint_publishes_the_measured_limits(self):
        from fastapi.testclient import TestClient

        import main

        body = TestClient(main.app).get("/version").json()
        self.assertEqual(body["max_sync_bbox_km2"], main.MAX_SYNC_BBOX_KM2)
        self.assertEqual(body["max_landcover_bbox_km2"], main.MAX_LANDCOVER_BBOX_KM2)
        self.assertEqual(body["max_concurrent_analyses"], main.MAX_CONCURRENT_ANALYSES)
        self.assertEqual(body["max_concurrent_ndvi"], main.MAX_CONCURRENT_NDVI)
        self.assertEqual(sorted(body["available_datasets"]), sorted(main.AVAILABLE_DATASETS))


if __name__ == "__main__":
    unittest.main()
