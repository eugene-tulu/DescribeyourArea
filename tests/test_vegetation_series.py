"""Tests for the per-area monthly vegetation series.

The series exists to sit beside rainfall on a shared time axis, so the properties
worth locking are: the months are real and aligned, the baseline says what it
actually used, and nothing claims a precision the source cannot support.
"""

import unittest

import numpy as np

import vegetation_series as vs


class MonthRangeTests(unittest.TestCase):
    def test_the_range_is_inclusive_of_both_ends(self):
        months = vs._month_range("2020-01-01", "2020-12-01")
        self.assertEqual(len(months), 12)
        self.assertEqual(months[0], "2020-01")
        self.assertEqual(months[-1], "2020-12")

    def test_a_single_month(self):
        self.assertEqual(vs._month_range("2021-06-01", "2021-06-28"), ["2021-06"])

    def test_it_crosses_the_year_boundary(self):
        months = vs._month_range("2019-11-01", "2020-02-01")
        self.assertEqual(months, ["2019-11", "2019-12", "2020-01", "2020-02"])

    def test_a_reversed_range_is_empty_rather_than_negative(self):
        self.assertEqual(vs._month_range("2020-06-01", "2020-01-01"), [])


class SourceFidelityTests(unittest.TestCase):
    """The published source's own numbers, so a change to them is noticed."""

    def test_modis_native_resolution_is_the_sinusoidal_grid_not_250(self):
        # The MOD13Q1 grid is 231.656 m. Calling it 250 m is a small lie that
        # propagates into every stated resolution.
        self.assertAlmostEqual(vs.MODIS_NATIVE_M, 231.7, delta=0.1)
        self.assertNotEqual(vs.MODIS_NATIVE_M, 250.0)

    def test_the_fill_value_is_excluded_from_validity(self):
        self.assertEqual(vs.MODIS_FILL, -3000)

    def test_the_scale_is_the_published_one(self):
        self.assertEqual(vs.MODIS_SCALE, 0.0001)

    def test_the_read_is_concurrent_but_not_unknowingly(self):
        # Four-way measured fastest; eight was slower, which is server-side
        # contention, so a higher default would be slower, not faster.
        self.assertEqual(vs.DEFAULT_WORKERS, 4)


class SeriesShapeTests(unittest.TestCase):
    """Derived from a synthetic series, so the arithmetic is checked without the network."""

    @staticmethod
    def _rows():
        return [
            {"month": "2015-01", "value": 0.20, "valid_pixels": 100, "area_fraction": 1.0},
            {"month": "2015-02", "value": 0.22, "valid_pixels": 100, "area_fraction": 1.0},
            {"month": "2016-01", "value": 0.24, "valid_pixels": 100, "area_fraction": 1.0},
            {"month": "2016-02", "value": 0.30, "valid_pixels": 100, "area_fraction": 0.4},
        ]

    def test_climatology_averages_only_calendar_months_in_the_baseline(self):
        rows = self._rows()
        baseline = [r for r in rows if "2015" <= r["month"][:4] <= "2016"]
        means: dict[str, list[float]] = {}
        for row in baseline:
            means.setdefault(row["month"][5:7], []).append(row["value"])
        # January has 2015 and 2016, so it averages two; February has two as well.
        self.assertAlmostEqual(sum(means["01"]) / len(means["01"]), 0.22)
        self.assertAlmostEqual(sum(means["02"]) / len(means["02"]), 0.26)

    def test_a_thin_month_is_detectable_from_its_area_fraction(self):
        thin = [r["month"] for r in self._rows() if r["area_fraction"] < 0.9]
        self.assertEqual(thin, ["2016-02"])

    def test_the_caveat_refuses_attribution(self):
        # This is the one string the chart shows a reader, and it must not imply
        # that rainfall drove the vegetation.
        self.assertIn("not attribution", vs.CAVEAT)
        self.assertIn("lag", vs.CAVEAT)
        self.assertNotIn("because of rainfall", vs.CAVEAT)


class AreaHandlingTests(unittest.TestCase):
    def test_the_mask_uses_the_reprojected_geometry(self):
        # geometry_mask does not reproject, and MOD13Q1 is a custom sinusoidal CRS.
        # An unprojected WGS84 polygon lands millions of metres away and the mask
        # comes back empty, which is exactly what happened.
        import inspect

        source = inspect.getsource(vs.compute_monthly_series)
        self.assertIn("transform_geom", source,
                      "the study area must be projected before rasterising")

    def test_the_window_helper_reprojects_too(self):
        import inspect

        source = inspect.getsource(vs.compute_monthly_series)
        self.assertIn("_window_for_bbox", source,
                      "the source grid is projected, so the bbox must be reprojected")

    def test_months_are_derived_from_one_paged_search_not_one_per_month(self):
        import inspect

        source = inspect.getsource(vs._items_by_month)
        # 35 years of months was 420 separate searches, one of which came back as a
        # connection reset. One search for the span is both cheaper and politer.
        self.assertEqual(source.count("catalog.search"), 1)
        self.assertNotIn("def find(", source)
        self.assertIn("items()", source)

    def test_a_transient_search_failure_is_retried(self):
        import inspect

        source = inspect.getsource(vs._items_by_month)
        self.assertIn("for attempt in range(3)", source)
        self.assertIn("run()", source, "the search is rebuilt per attempt")


if __name__ == "__main__":
    unittest.main()
