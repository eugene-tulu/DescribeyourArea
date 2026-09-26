"""Tests for the sensor registry, the selection policy and the cloud masks.

The Landsat scale/offset test is the important one: the offset does not cancel in
the NDVI ratio, so a port that forgets it produces a plausible number that is
wrong by tens of percent.
"""

import datetime
import unittest

import numpy as np

import main
from sensors import (
    LANDSAT,
    MODIS,
    MODIS_MIN_AREA_KM2,
    NDVI_MIN_PLAUSIBLE,
    SCL_REJECTED,
    SENSORS,
    SENTINEL2,
    cloud_mask,
    get_sensor,
    plausible,
    select_sensor,
)


def days_ago(n: int) -> str:
    return (datetime.date.today() - datetime.timedelta(days=n)).isoformat()


class RegistryTests(unittest.TestCase):
    def test_three_sensors_are_registered(self):
        self.assertEqual(sorted(SENSORS), ["landsat", "modis", "sentinel-2"])

    def test_every_sensor_declares_usable_provenance(self):
        for sensor in SENSORS.values():
            with self.subTest(sensor=sensor.id):
                provenance = sensor.provenance()
                for key in ("label", "collection", "native_resolution_m", "archive_start", "ndvi"):
                    self.assertTrue(provenance[key], f"{sensor.id} missing {key}")
                self.assertRegex(sensor.archive_start, r"^\d{4}-\d{2}-\d{2}$")

    def test_only_modis_reads_a_product_rather_than_computing(self):
        self.assertTrue(MODIS.ndvi_asset)
        self.assertFalse(MODIS.computes_ndvi)
        for sensor in (SENTINEL2, LANDSAT):
            with self.subTest(sensor=sensor.id):
                self.assertTrue(sensor.computes_ndvi)
                self.assertTrue(sensor.red and sensor.nir)

    def test_landsat_offset_is_present_and_does_not_cancel(self):
        # The whole reason the offset is in the registry rather than read from the
        # raster: the file's own band tags are empty, and the STAC value is in
        # reflectance units, so it is a real constant that must be applied.
        self.assertNotEqual(LANDSAT.offset, 0.0)
        self.assertNotEqual(LANDSAT.scale, 1.0)
        red, nir = 2000.0, 3000.0
        with_offset = ((nir * LANDSAT.scale + LANDSAT.offset)
                       - (red * LANDSAT.scale + LANDSAT.offset)) / (
                          (nir * LANDSAT.scale + LANDSAT.offset)
                          + (red * LANDSAT.scale + LANDSAT.offset))
        without = (nir - red) / (nir + red)
        self.assertNotAlmostEqual(with_offset, without, places=3)

    def test_sentinel2_needs_no_offset(self):
        self.assertEqual(SENTINEL2.offset, 0.0)
        self.assertEqual(SENTINEL2.scale, 1.0)

    def test_only_sentinel2_has_a_relaxed_mask(self):
        # Measured cross-sensor over one 60 km2 study area and a 90-day window:
        # MODIS 0.418 and Landsat 0.357 agree to within 0.06, while Sentinel-2 read
        # 0.161 because its SCL classes 4 and 5 are kept.
        self.assertFalse(SENTINEL2.mask_authoritative)
        self.assertTrue(LANDSAT.mask_authoritative)
        self.assertTrue(MODIS.mask_authoritative)

    def test_get_sensor_accepts_auto_and_names(self):
        self.assertIs(get_sensor(None), SENTINEL2)
        self.assertIs(get_sensor("auto"), SENTINEL2)
        self.assertIs(get_sensor("landsat"), LANDSAT)
        self.assertIs(get_sensor(" MODIS "), MODIS)

    def test_unknown_sensor_is_rejected_with_the_options(self):
        with self.assertRaises(ValueError) as raised:
            get_sensor("landsat9")
        self.assertIn("landsat", str(raised.exception))


class PolicyTests(unittest.TestCase):
    def test_recent_window_prefers_an_authoritative_mask(self):
        # Landsat over Sentinel-2 despite being coarser: at 30 m a 100 km2 area
        # gains nothing from 10 m, and the QA mask can be trusted.
        sensor, reason = select_sensor(None, window_start=days_ago(45), bbox_area_km2=60)
        self.assertIs(sensor, LANDSAT)
        self.assertIn("authoritative", reason)

    def test_the_last_few_days_fall_to_sentinel2(self):
        # Landsat has no scene yet for the last few days; Sentinel-2 does.
        for days in (1, 3, 6):
            with self.subTest(days=days):
                sensor, reason = select_sensor(None, window_start=days_ago(days), bbox_area_km2=60)
                self.assertIs(sensor, SENTINEL2)
                self.assertIn("relaxed", reason)

    def test_pre_sentinel2_window_uses_landsat(self):
        sensor, reason = select_sensor(None, window_start="1997-03-01", bbox_area_km2=60)
        self.assertIs(sensor, LANDSAT)
        self.assertIn("1997-03-01", reason)

    def test_large_areas_use_modis(self):
        sensor, reason = select_sensor(None, window_start=days_ago(30),
                                      bbox_area_km2=MODIS_MIN_AREA_KM2)
        self.assertIs(sensor, MODIS)
        self.assertIn("budget", reason)

    def test_explicit_request_always_wins(self):
        for name, expected in (("sentinel-2", SENTINEL2), ("landsat", LANDSAT), ("modis", MODIS)):
            with self.subTest(sensor=name):
                sensor, reason = select_sensor(name, window_start=days_ago(2), bbox_area_km2=60)
                self.assertIs(sensor, expected)
                self.assertEqual(reason, "explicitly requested")

    def test_every_policy_choice_carries_a_reason(self):
        for start, area in (("1997-03-01", 60), (days_ago(3), 60), (days_ago(45), 60), (days_ago(5), 9_000)):
            with self.subTest(start=start, area=area):
                _sensor, reason = select_sensor(None, window_start=start, bbox_area_km2=area)
                self.assertTrue(reason.strip())


class CloudMaskTests(unittest.TestCase):
    def test_scl_rejects_the_unambiguous_classes(self):
        for code in (0, 1, 3, 8, 9, 10, 11):
            self.assertIn(code, SCL_REJECTED)

    def test_scl_keeps_bright_cloud_so_desert_survives(self):
        # Class 5 flags bright desert as cloud across whole tiles; rejecting it
        # empties the result for exactly the semi-arid ground this service serves.
        for code in (4, 5):
            self.assertNotIn(code, SCL_REJECTED)
        values = np.array([[0.0, 0.1]], dtype="float32")
        classes = np.array([[5, 2]], dtype="float32")
        self.assertEqual(cloud_mask(SENTINEL2, values, classes).ravel().tolist(), [True, True])

    def test_scl_rejects_actual_cloud_shadow_and_cirrus(self):
        values = np.array([[0.1, 0.1, 0.1]], dtype="float32")
        classes = np.array([[3, 9, 10]], dtype="float32")
        self.assertEqual(cloud_mask(SENTINEL2, values, classes).ravel().tolist(), [False, False, False])

    def test_qa_bits_reject_cloud_shadow_snow_and_fill(self):
        from sensors import QA_CLOUD_MASK, QA_FILL

        clear = np.array([[0]], dtype="int64")
        cloud = np.array([[1 << 3]], dtype="int64")
        cirrus = np.array([[1 << 2]], dtype="int64")
        snow = np.array([[1 << 5]], dtype="int64")
        fill = np.array([[QA_FILL]], dtype="int64")
        values = np.array([[0.2]], dtype="float32")
        for label, qa in (("clear", clear), ("cloud", cloud), ("cirrus", cirrus),
                          ("snow", snow), ("fill", fill)):
            with self.subTest(flag=label):
                self.assertTrue(bool(cloud_mask(LANDSAT, values, qa)[0]) is (label == "clear"))

    def test_qa_mask_ignores_bits_the_producer_leaves_clear(self):
        from sensors import QA_CLOUD_MASK

        # Bits 6-15 are unused in QA_PIXEL; the mask must ignore them.
        qa = np.array([[0b110000000000000]], dtype="int64")
        self.assertTrue(bool(cloud_mask(LANDSAT, np.array([[0.2]]), qa)[0]))
        self.assertTrue(QA_CLOUD_MASK > 0)

    def test_product_sensor_trusts_the_producer(self):
        values = np.array([[0.4, np.nan]], dtype="float32")
        keep = cloud_mask(MODIS, values, None)
        self.assertTrue(bool(keep[0, 0]))
        self.assertFalse(bool(keep[0, 1]))


class PlausibilityTests(unittest.TestCase):
    def test_implausible_values_are_dropped(self):
        values = np.array([0.2, -0.1, 1.4, np.nan, 0.0], dtype="float32")
        keep = plausible(values).tolist()
        self.assertEqual(keep, [True, False, False, False, True])

    def test_the_floor_is_zero(self):
        self.assertEqual(NDVI_MIN_PLAUSIBLE, 0.0)


class WindowResolutionTests(unittest.TestCase):
    def test_default_lookback_from_today(self):
        start, end = main._resolve_window(None, None, 30)
        self.assertEqual(end, datetime.date.today().isoformat())
        self.assertEqual(
            (datetime.date.fromisoformat(end) - datetime.date.fromisoformat(start)).days, 30
        )

    def test_explicit_range_is_kept(self):
        self.assertEqual(main._resolve_window("1997-01-01", "1999-12-31", 90),
                         ("1997-01-01", "1999-12-31"))

    def test_explicit_start_wins_over_the_lookback(self):
        # An explicit start is honoured and the window runs to today; window_days
        # is only the default when no range is given.
        start, end = main._resolve_window("2024-01-01", None, 90)
        self.assertEqual(start, "2024-01-01")
        self.assertEqual(end, datetime.date.today().isoformat())

    def test_inverted_range_is_rejected(self):
        with self.assertRaises(ValueError) as raised:
            main._resolve_window("2024-06-01", "2024-01-01", 90)
        self.assertIn("after the end", str(raised.exception))

    def test_bad_format_is_rejected_with_the_expected_shape(self):
        with self.assertRaises(ValueError) as raised:
            main._resolve_window("06/01/2024", None, 90)
        self.assertIn("YYYY-MM-DD", str(raised.exception))

    def test_a_decade_is_allowed_and_more_is_not(self):
        # A request takes max_scenes however long the window is, so span is not a
        # cost; the ceiling only catches a mistyped year.
        self.assertTrue(main._resolve_window("2015-01-01", "2024-12-31", 90))
        self.assertTrue(main._resolve_window("1997-01-01", "1999-12-31", 90))
        with self.assertRaises(ValueError) as raised:
            main._resolve_window("1900-01-01", "2024-01-01", 90)
        self.assertIn("synchronous", str(raised.exception))


class VegetationProvenanceTests(unittest.TestCase):
    """The payload must always say which sensor produced the number."""

    AOI = {"type": "Polygon", "coordinates": [[[35.10, -1.55], [35.17, -1.55],
                                               [35.17, -1.48], [35.10, -1.48],
                                               [35.10, -1.55]]]}

    def test_bad_window_is_reported_not_raised(self):
        import asyncio

        result = asyncio.run(main.compute_vegetation_index(
            [35.10, -1.55, 35.17, -1.48], self.AOI, sensor_id="auto",
            start="1900-01-01", end="2024-01-01",
        ))
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("synchronous", result["warning"])

    def test_unknown_sensor_is_reported_not_raised(self):
        import asyncio

        result = asyncio.run(main.compute_vegetation_index(
            [35.10, -1.55, 35.17, -1.48], self.AOI, sensor_id="landsat9",
        ))
        self.assertEqual(result["status"], "unavailable")
        self.assertIn("landsat", result["warning"])

    def test_window_before_a_sensor_archive_is_skipped_explicitly(self):
        import asyncio

        result = asyncio.run(main.compute_vegetation_index(
            [35.10, -1.55, 35.17, -1.48], self.AOI,
            max_area_km2=1e9, sensor_id="sentinel-2",
            start="1997-01-01", end="1999-12-31",
        ))
        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["sensor"]["id"], "sentinel-2")
        self.assertIn(SENTINEL2.archive_start, result["warning"])

    def test_skipped_never_reports_a_value(self):
        import asyncio

        result = asyncio.run(main.compute_vegetation_index(
            [35.10, -1.55, 35.17, -1.48], self.AOI,
            max_area_km2=1.0, sensor_id="sentinel-2",
        ))
        self.assertEqual(result["status"], "skipped")
        self.assertNotIn("mean", result)
        self.assertNotIn("modis", str(result.get("warning", "")).lower())

    def test_version_publishes_the_ladder(self):
        from fastapi.testclient import TestClient

        body = TestClient(main.app).get("/version").json()
        self.assertEqual(sorted(body["available_sensors"]), ["landsat", "modis", "sentinel-2"])
        self.assertEqual(body["default_sensor"], "auto")
        self.assertEqual(body["available_sensors"]["modis"]["ndvi"], "product")
        self.assertEqual(body["available_sensors"]["landsat"]["cloud_mask"], "qa_bits")


if __name__ == "__main__":
    unittest.main()
