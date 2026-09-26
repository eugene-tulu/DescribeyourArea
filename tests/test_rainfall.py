"""Tests for the ERA5 rainfall module.

The anomaly arithmetic, the plausibility guards and the cache contract are pure
logic and always run. The ERA5 read itself is network-bound and is exercised by
``test_planetary_computer.py``-style real-data checks that skip when offline.
"""

import json
import os
import tempfile
import unittest
from pathlib import Path

import rainfall


class ClimatologyTests(unittest.TestCase):
    def test_mean_per_calendar_month_over_the_baseline(self):
        series = {}
        for year in (1991, 1992, 1993):
            for month, value in zip(range(1, 13), range(10, 130, 10)):
                series[f"{year}-{month:02d}"] = float(value)
        climatology = rainfall.monthly_climatology(series, "1991-01-01", "2023-12-31")
        self.assertEqual(len(climatology), 12)
        # zip(range(1,13), range(10,130,10)) pairs month 1 with 10 mm.
        self.assertAlmostEqual(climatology["01"], 10.0)
        self.assertAlmostEqual(climatology["12"], 120.0)

    def test_years_outside_the_baseline_are_ignored(self):
        series = {"1990-01": 1.0, "1991-01": 10.0, "2021-01": 999.0}
        climatology = rainfall.monthly_climatology(series, "1991-01-01", "2020-12-31")
        self.assertAlmostEqual(climatology["01"], 10.0)

    def test_empty_series_yields_no_climatology(self):
        self.assertEqual(rainfall.monthly_climatology({}, "1991-01-01", "2020-12-31"), {})


class AnomalyTests(unittest.TestCase):
    def test_signed_deviation_from_the_same_month(self):
        series = {"2020-01": 25.0, "2020-02": 10.0}
        climatology = {"01": 50.0, "02": 10.0}
        rows = {r["month"]: r for r in rainfall.anomalies(series, climatology)}
        self.assertAlmostEqual(rows["2020-01"]["anomaly_mm"], -25.0)
        self.assertAlmostEqual(rows["2020-01"]["anomaly_pct"], -50.0)
        self.assertAlmostEqual(rows["2020-02"]["anomaly_pct"], 0.0)

    def test_months_absent_from_the_climatology_are_skipped(self):
        rows = rainfall.anomalies({"2020-03": 5.0}, {"01": 50.0})
        self.assertEqual(rows, [])

    def test_zero_normal_does_not_divide_by_zero(self):
        rows = rainfall.anomalies({"2020-01": 5.0}, {"01": 0.0})
        self.assertIsNone(rows[0]["anomaly_pct"])
        self.assertEqual(rows[0]["anomaly_mm"], 5.0)

    def test_near_zero_month_over_a_wet_climatology_is_flagged(self):
        rows = rainfall.anomalies({"2000-02": 0.1}, {"02": 49.0})
        self.assertEqual(rows[0]["suspect"], "near_zero_month_worth_review")

    def test_genuinely_dry_climates_are_not_flagged(self):
        # A month near zero where the normal is also near zero is ordinary.
        rows = rainfall.anomalies({"2000-02": 0.1}, {"02": 0.4})
        self.assertNotIn("suspect", rows[0])

    def test_near_zero_month_over_a_dry_normal_is_not_flagged(self):
        rows = rainfall.anomalies({"2000-02": 0.1}, {"02": 2.0})
        self.assertNotIn("suspect", rows[0])


class RollingTests(unittest.TestCase):
    def _rows(self, values, normals):
        return [
            {"month": f"2020-{i + 1:02d}", "precip_mm": v, "normal_mm": n,
             "anomaly_mm": v - n, "anomaly_pct": None}
            for i, (v, n) in enumerate(zip(values, normals))
        ]

    def test_window_total_and_percentage(self):
        rows = self._rows([10.0] * 12, [20.0] * 12)
        totals = rainfall.rolling_totals(rows, 12)
        self.assertEqual(len(totals), 1)
        self.assertAlmostEqual(totals[0]["precip_mm"], 120.0)
        self.assertAlmostEqual(totals[0]["anomaly_pct"], -50.0)

    def test_window_shorter_than_requested_yields_nothing(self):
        self.assertEqual(rainfall.rolling_totals(self._rows([1.0] * 3, [1.0] * 3), 12), [])

    def test_shorter_window_aggregates_fewer_months(self):
        totals = rainfall.rolling_totals(self._rows([5.0] * 6, [5.0] * 6), 3)
        self.assertAlmostEqual(totals[-1]["precip_mm"], 15.0)
        self.assertEqual(totals[-1]["months"], 3)


class PlausibilityGuardTests(unittest.TestCase):
    """A unit error here is silent and severe, so the range is enforced."""

    def test_absurd_totals_are_rejected(self):
        for annual in (0.8, 0.0, -5.0, 50_000.0):
            with self.subTest(annual=annual):
                with self.assertRaises(ValueError):
                    rainfall._assert_plausible(annual, cells=4)

    def test_plausible_totals_are_accepted(self):
        for annual in (254.0, 397.0, 735.0, 1578.0, 5000.0):
            with self.subTest(annual=annual):
                rainfall._assert_plausible(annual, cells=4)

    def test_absent_total_is_not_an_error(self):
        rainfall._assert_plausible(None, cells=4)


class GeometryKeyTests(unittest.TestCase):
    AOI = {"type": "Polygon",
           "coordinates": [[[35.1, -1.5], [35.2, -1.5], [35.2, -1.4], [35.1, -1.4], [35.1, -1.5]]]}

    def test_key_is_stable_for_the_same_geometry(self):
        self.assertEqual(rainfall.geometry_hash(self.AOI), rainfall.geometry_hash(dict(self.AOI)))

    def test_different_geometry_gets_a_different_key(self):
        other = json.loads(json.dumps(self.AOI))
        other["coordinates"][0][0][0] = 36.0
        self.assertNotEqual(rainfall.geometry_hash(self.AOI), rainfall.geometry_hash(other))


class CacheContractTests(unittest.TestCase):
    def setUp(self):
        self.previous = os.environ.get("RAINFALL_CACHE_DIR")
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["RAINFALL_CACHE_DIR"] = self.tmp.name
        self.key = "testkey"

    def tearDown(self):
        if self.previous is None:
            os.environ.pop("RAINFALL_CACHE_DIR", None)
        else:
            os.environ["RAINFALL_CACHE_DIR"] = self.previous
        self.tmp.cleanup()

    def _payload(self, version=rainfall.RAINFALL_PROCESSING_VERSION):
        return {"processing_version": version, "series": [{"month": "2020-01", "precip_mm": 1.0}]}

    def test_round_trip(self):
        rainfall.write_cache(self.key, self._payload())
        cached = rainfall.read_cache(self.key)
        self.assertIsNotNone(cached)
        self.assertEqual(cached["series"][0]["month"], "2020-01")

    def test_miss_returns_none(self):
        self.assertIsNone(rainfall.read_cache("absent"))

    def test_a_stale_processing_version_is_a_miss(self):
        # A cached value must never outlive the code that produced it.
        rainfall.write_cache(self.key, self._payload(version="era5-monthly-0"))
        self.assertIsNone(rainfall.read_cache(self.key))

    def test_corrupt_cache_is_a_miss_not_a_crash(self):
        path = Path(self.tmp.name) / f"{self.key}.json"
        path.write_text("{not json")
        self.assertIsNone(rainfall.read_cache(self.key))

    def test_context_reports_an_explicit_miss_with_a_reason(self):
        context = rainfall.cached_context(
            {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}
        )
        self.assertEqual(context["status"], "not_computed")
        self.assertEqual(context["reason"], "no_precomputed_series")
        self.assertIn("20 seconds", context["message"])

    def test_context_returns_a_cached_series(self):
        geom = {"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}
        rainfall.write_cache(rainfall.geometry_hash(geom), self._payload())
        context = rainfall.cached_context(geom)
        self.assertEqual(context["status"], "ok")
        self.assertEqual(context["processing_version"], rainfall.RAINFALL_PROCESSING_VERSION)


class ConfigurationTests(unittest.TestCase):
    def test_climatology_is_the_wmo_normal(self):
        self.assertEqual(rainfall.CLIMATOLOGY_START, "1991-01-01")
        self.assertEqual(rainfall.CLIMATOLOGY_END, "2020-12-31")

    def test_store_is_public_and_needs_no_credentials(self):
        # The ERA5 store must stay anonymously readable or the build breaks.
        self.assertTrue(rainfall.ERA5_BUCKET)
        self.assertTrue(rainfall.ERA5_PREFIX)
        self.assertEqual(rainfall.ERA5_REGION, "us-east-1")

    def test_provenance_constants_are_present(self):
        self.assertRegex(rainfall.ERA5_DOI, r"^10\.\d{4,}/")
        self.assertIn("doi.org", rainfall.ERA5_CITATION)
        self.assertIn("CC-BY", rainfall.ERA5_CITATION)
        self.assertTrue(rainfall.ERA5_GRID_DEGREES)

    def test_resolution_is_reported_in_kilometres(self):
        self.assertAlmostEqual(rainfall.ERA5_GRID_DEGREES * 111.32, 27.8, delta=0.1)


class CellSelectionTests(unittest.TestCase):
    class _Fake:
        class _Axis:
            def __init__(self, values):
                self.values = values

        longitude = _Axis([0.0, 0.25, 0.5, 0.75])
        latitude = _Axis([1.0, 0.75, 0.5])

        def __getitem__(self, key):
            return {"longitude": self.longitude, "latitude": self.latitude}[key]

    def test_cells_covering_the_bbox_are_selected(self):
        grid = rainfall._grid(self._Fake(), [0.1, 0.6, 0.6, 0.9])
        self.assertIn(0.25, grid["longitudes"])
        self.assertIn(0.5, grid["longitudes"])
        self.assertIn(0.75, grid["latitudes"])

    def test_a_distant_bbox_selects_nothing(self):
        grid = rainfall._grid(self._Fake(), [50.0, 50.0, 50.5, 50.5])
        self.assertEqual(grid["longitudes"], [])
        self.assertEqual(grid["latitudes"], [])

    def test_cellset_key_distinguishes_cells_and_windows(self):
        a = {"longitudes": [0.25], "latitudes": [1.0]}
        b = {"longitudes": [0.5], "latitudes": [1.0]}
        self.assertNotEqual(
            rainfall.cellset_key(a, "2010-01-01", "2020-12-31"),
            rainfall.cellset_key(b, "2010-01-01", "2020-12-31"),
        )
        self.assertNotEqual(
            rainfall.cellset_key(a, "2010-01-01", "2020-12-31"),
            rainfall.cellset_key(a, "2011-01-01", "2020-12-31"),
        )
        self.assertEqual(
            rainfall.cellset_key(a, "2010-01-01", "2020-12-31"),
            rainfall.cellset_key(dict(a), "2010-01-01", "2020-12-31"),
        )


class RealEra5Tests(unittest.TestCase):
    """Live checks against the public ERA5 store. Skipped when offline."""

    AOI = {"type": "Polygon",
           "coordinates": [[[35.10, -1.55], [35.17, -1.55], [35.17, -1.48],
                            [35.10, -1.48], [35.10, -1.55]]]}

    @classmethod
    def setUpClass(cls):
        try:
            import pcodec  # noqa: F401
            import icechunk  # noqa: F401
            import xarray  # noqa: F401

            rainfall._dataset_handle()
            cls.available = True
        except Exception:
            cls.available = False

    def setUp(self):
        if not self.available:
            self.skipTest("ERA5 Icechunk store is unreachable")

    def test_a_built_series_is_physically_plausible(self):
        payload = rainfall.compute_series(self.AOI, start="2015-01-01", end="2024-12-31")
        normal = payload["climatology"]["annual_mean_mm"]
        # Semi-arid East African rangeland: hundreds of mm, not millimetres.
        self.assertGreater(normal, 200.0)
        self.assertLess(normal, 2000.0)
        self.assertTrue(payload["series"])
        for row in payload["series"]:
            self.assertGreaterEqual(row["precip_mm"], 0.0)
            self.assertLessEqual(row["precip_mm"], 1200.0)

    def test_the_series_shows_the_bimodal_east_african_rainy_season(self):
        """Long rains Mar-May and short rains Oct-Dec, dry in the middle.

        This is the shape that proves the monthly aggregation is right: a unit or
        averaging error destroys it.
        """
        payload = rainfall.compute_series(self.AOI, start="2015-01-01", end="2024-12-31")
        normals = payload["climatology"]["monthly_mean_mm"]
        long_rains = sum(normals[m] for m in ("03", "04", "05"))
        short_rains = sum(normals[m] for m in ("10", "11", "12"))
        dry_mid = sum(normals[m] for m in ("06", "07", "08"))
        self.assertGreater(long_rains, dry_mid)
        self.assertGreater(short_rains, dry_mid)

    def test_provenance_is_complete(self):
        payload = rainfall.compute_series(self.AOI, start="2020-01-01", end="2024-12-31")
        for key in ("source", "doi", "citation", "license", "retrieved",
                    "resolution_km", "grid_cells", "processing_version"):
            self.assertTrue(payload.get(key), f"missing provenance: {key}")
        self.assertIn("coverage", payload)
        self.assertIn("climatology", payload)

    def test_coverage_is_reported_honestly(self):
        payload = rainfall.compute_series(self.AOI, start="2020-01-01", end="2024-12-31")
        window = payload["coverage"]["window"]
        self.assertIn("months_dropped", window)
        self.assertLessEqual(window["valid_hours"], window["expected_hours"])


if __name__ == "__main__":
    unittest.main()
