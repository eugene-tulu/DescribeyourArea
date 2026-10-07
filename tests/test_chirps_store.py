"""The CHIRPS tensor store, and the cache that was never written.

Three failures here are silent, which is why each has a test.

**A cache that is checked but never filled.** ``read_cell_monthly`` keyed its
cache entry on the product from 31c3ca8, but returned early whenever a reader was
supplied -- which is every CHIRPS call. So the entry was looked up, never found,
never written, and every job re-read every month while the cache reported a hit
rate that was never true. A reader of that cache directory would conclude the
work was being reused. It was not.

**Two windows read as two passes.** The series window and the 1991-2020
climatology were separate reads of the same cells, so every shared month was
fetched twice. For a typical request the shared part is the entire climatology.

**A hole absorbed as a shorter axis.** CHIRPS does not publish every month. The
store keeps an absent month as NaN so the time axis stays complete; a build that
concatenated only the months that exist would shift every later date.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import rainfall
from rainfall import (
    CHIRPS_STORE_BUILD_VERSION,
    CHIRPS_STORE_CHUNKS,
    CHIRPS_STORE_GROUP,
    CLIMATOLOGY_END,
    CLIMATOLOGY_START,
)


def stub_reader(grid, start, end):
    """A CHIRPS-shaped reader that reads nothing.

    What these tests assert is plumbing: which cache slot a product lands in,
    whether it is written at all, and what a window slice returns. None of that
    needs a raster, and the tests that used the live archive took seventeen
    minutes between them and flaked under load.
    """
    months = rainfall._month_range(start, end)
    shape = (len(months), len(grid["latitudes"]), len(grid["longitudes"]))
    values = np.full(shape, 100.0)
    values[0, 0, 0] = np.nan  # one absent month cell, for coverage reporting
    return months, values, {
        "source": "stub", "months": len(months), "unreadable_months": [],
    }


GRID = {"latitudes": [0.0], "longitudes": [39.0]}


class CellCacheIsWrittenTests(unittest.TestCase):
    """The regression guard: a supplied reader must still populate the cache."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = patch.dict(os.environ, {"RAINFALL_CACHE_DIR": self.tmp.name})
        patcher.start()
        self.addCleanup(patcher.stop)
        rainfall._invalidate_process_caches()

    def test_a_supplied_reader_now_leaves_a_cache_entry(self):
        labels, matrix, coverage = rainfall.read_cell_monthly(
            GRID, "2020-01-01", "2020-06-01", stub_reader, product="chirps"
        )
        self.assertEqual(len(labels), 6)

        entries = list((rainfall.cache_dir() / "cells").glob("*.npz"))
        self.assertEqual(
            len(entries), 1,
            "a CHIRPS read left no cache entry, so every job repeats every month",
        )

        # And the entry has to be usable, not merely present.
        again = rainfall.read_cell_monthly(
            GRID, "2020-01-01", "2020-06-01", stub_reader, product="chirps"
        )
        self.assertEqual(list(again[0]), list(labels))
        np.testing.assert_allclose(again[1], matrix, equal_nan=True)
        self.assertEqual(again[2]["source"], coverage["source"])

    def test_a_second_read_does_not_call_the_reader_again(self):
        rainfall.read_cell_monthly(GRID, "2020-01-01", "2020-06-01",
                                   stub_reader, product="chirps")
        with patch.object(rainfall, "chirps_cell_monthly",
                          side_effect=AssertionError("cache was missed")):
            labels, _matrix, _coverage = rainfall.read_cell_monthly(
                GRID, "2020-01-01", "2020-06-01",
                rainfall.chirps_cell_monthly, product="chirps",
            )
        self.assertEqual(len(labels), 6)

    def test_era5_and_chirps_still_land_in_separate_entries(self):
        rainfall.read_cell_monthly(GRID, "2020-01-01", "2020-06-01",
                                   stub_reader, product="chirps")
        with patch.object(rainfall, "_dataset_handle") as handle:
            handle.return_value = None
            with self.assertRaises(Exception):
                rainfall.read_cell_monthly(
                    GRID, "2020-01-01", "2020-06-01", None, product="rainfall"
                )
        names = {path.name for path in (rainfall.cache_dir() / "cells").glob("*.npz")}
        self.assertEqual(len(names), 1)
        key = names.pop()
        self.assertNotIn("chirps", key, "the cache key must name the product")


class UnionWindowTests(unittest.TestCase):
    """One read over the union of both windows, then sliced."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        patcher = patch.dict(os.environ, {"RAINFALL_CACHE_DIR": self.tmp.name})
        patcher.start()
        self.addCleanup(patcher.stop)
        rainfall._invalidate_process_caches()

    def test_the_union_covers_both_windows(self):
        start, end = "2021-01-01", "2021-12-01"
        union_start = min(start, CLIMATOLOGY_START)
        union_end = max(end, CLIMATOLOGY_END)
        labels, _matrix, _coverage = rainfall.read_cell_monthly(
            GRID, union_start, union_end, stub_reader, product="chirps"
        )
        months = [label[:7] for label in labels]
        self.assertEqual(months[0], CLIMATOLOGY_START[:7])
        self.assertEqual(months[-1], end[:7])
        # The union is a superset of each window.
        self.assertIn("2021-06", months)
        self.assertIn(CLIMATOLOGY_END[:7], months)

    def test_one_union_entry_serves_both_windows(self):
        union_start = min("2021-01-01", CLIMATOLOGY_START)
        union_end = max("2021-12-01", CLIMATOLOGY_END)
        rainfall.read_cell_monthly(GRID, union_start, union_end,
                                   stub_reader, product="chirps")
        entries = list((rainfall.cache_dir() / "cells").glob("*.npz"))
        self.assertEqual(
            len(entries), 1,
            "the union should be one cache entry; more means a window was read "
            "separately and a shared month was fetched twice",
        )


class StoreConfigurationTests(unittest.TestCase):
    """Parsing and endpoint handling, with no network."""

    def test_no_uri_means_no_store(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(rainfall.chirps_store_target())
            self.assertIsNone(rainfall.chirps_store_dataset())

    def test_a_bucket_qualified_endpoint_is_reduced_to_its_region(self):
        env = {
            "CHIRPS_STORE_URI": "s3://primero/geocontextualize/chirps",
            "RAINFALL_S3_ENDPOINT": "https://primero.fra1.digitaloceanspaces.com",
            "RAINFALL_S3_REGION": "fra1",
        }
        with patch.dict(os.environ, env, clear=True):
            target = rainfall.chirps_store_target()
        # Icechunk adds the bucket to the host itself, so a bucket-qualified
        # endpoint yields <bucket>.<bucket>.fra1... and fails TLS validation.
        self.assertEqual(target["endpoint"], "https://fra1.digitaloceanspaces.com")
        self.assertEqual(target["bucket"], "primero")
        self.assertEqual(target["prefix"], "geocontextualize/chirps")

    def test_a_regional_endpoint_is_left_alone(self):
        env = {
            "CHIRPS_STORE_URI": "s3://primero/geocontextualize/chirps",
            "RAINFALL_S3_ENDPOINT": "https://fra1.digitaloceanspaces.com",
            "RAINFALL_S3_REGION": "fra1",
        }
        with patch.dict(os.environ, env, clear=True):
            target = rainfall.chirps_store_target()
        self.assertEqual(target["endpoint"], "https://fra1.digitaloceanspaces.com")

    def test_a_uri_without_the_scheme_is_refused(self):
        with patch.dict(os.environ, {"CHIRPS_STORE_URI": "primero/chirps"},
                        clear=True):
            with self.assertRaises(ValueError):
                rainfall.chirps_store_target()

    def test_a_public_bucket_needs_no_credentials(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(rainfall.credentials_for("anything"), {"anonymous": True})

    def test_configured_credentials_are_passed_through(self):
        env = {"AWS_ACCESS_KEY_ID": "key", "AWS_SECRET_ACCESS_KEY": "secret"}
        with patch.dict(os.environ, env, clear=True):
            creds = rainfall.credentials_for("anything")
        self.assertEqual(creds, {"access_key_id": "key",
                                 "secret_access_key": "secret"})

    def test_the_chunk_shape_is_a_year_of_months(self):
        # Twelve months, so a calendar year is exactly one chunk and committing a
        # year never leaves a partial chunk behind.
        self.assertEqual(CHIRPS_STORE_CHUNKS[0], 12)

    def test_the_store_names_its_build_version(self):
        self.assertTrue(CHIRPS_STORE_BUILD_VERSION)
        self.assertEqual(CHIRPS_STORE_GROUP, "chirps/monthly")


class AbsentMonthTests(unittest.TestCase):
    """A month with no published file is a month of NaN, not a missing label."""

    def test_the_reader_reports_an_all_nan_month_as_unreadable(self):
        months = rainfall._month_range("2023-10-01", "2024-01-01")
        matrix = np.full((len(months), 1, 1), 50.0)
        hole = months.index("2023-12-01")
        matrix[hole] = np.nan

        coverage = rainfall._chirps_coverage([months[hole]], ["2023-12"])
        self.assertEqual(coverage["unreadable_months"], ["2023-12"])
        self.assertEqual(coverage["source"], "chirps-v2.0")

    def test_the_hole_is_recorded_as_data_not_as_a_shorter_axis(self):
        payload = {
            "missing_months": json.dumps(["2022-07", "2023-12", "2024-07",
                                          "2024-08"]),
        }
        self.assertEqual(
            json.loads(payload["missing_months"]),
            ["2022-07", "2023-12", "2024-07", "2024-08"],
            "2022-07 is absent from the source too, and a build that missed it "
            "would shorten the record by a month without saying so",
        )


class EqualAreaWeightingTests(unittest.TestCase):
    """The store reader weights by cos(latitude) so a cell mean is an area mean."""

    def test_the_weights_fall_away_from_the_equator_and_are_symmetric(self):
        latitudes = np.array([-30.0, -10.0, 0.0, 10.0, 30.0])
        weights = np.cos(np.radians(latitudes))
        self.assertAlmostEqual(float(weights[2]), 1.0)
        # A pixel at 30 degrees covers less ground than one at the equator, so it
        # must carry less weight or the mean is not an area mean.
        self.assertLess(float(weights[4]), float(weights[2]))
        self.assertLess(float(weights[1]), float(weights[2]))
        # And it is symmetric about the equator.
        np.testing.assert_allclose(weights[0], weights[4], rtol=1e-12)
        np.testing.assert_allclose(weights[1], weights[3], rtol=1e-12)

    def test_a_constant_field_is_unweighted_wherever_it_is_read(self):
        # The weighting must not change the answer for a field with no gradient,
        # which is the property that makes it safe to apply silently.
        latitudes = np.linspace(-30.0, 30.0, 25)
        block = np.full((25, 4), 12.5)
        valid = np.isfinite(block)
        weight = np.broadcast_to(np.cos(np.radians(latitudes))[:, None], block.shape)
        mean = (np.where(valid, block, 0.0) * weight).sum() / np.where(
            valid, weight, 0.0).sum()
        self.assertAlmostEqual(float(mean), 12.5, places=6)


class FallbackTests(unittest.TestCase):
    """No store, or a window the store cannot cover, means the GeoTIFF path."""

    def test_the_store_reader_declines_when_unconfigured(self):
        with patch.dict(os.environ, {}, clear=True):
            rainfall._invalidate_process_caches()
            result = rainfall._chirps_cell_monthly_from_store(
                GRID, "2020-01-01", "2020-06-01"
            )
        self.assertIsNone(result,
                          "an unconfigured store must decline, not report empty")

    def test_the_public_entry_point_falls_back_to_the_geotiff_path(self):
        labels, matrix, coverage = rainfall.chirps_cell_monthly(
            GRID, "2020-01-01", "2020-03-01"
        )
        # Live read of three published months.
        self.assertEqual(len(labels), 3)
        self.assertEqual(matrix.shape, (3, 1, 1))
        self.assertEqual(coverage["source"], "chirps-v2.0")

    def test_a_cell_outside_the_archive_is_nan_not_zero(self):
        _labels, matrix, _coverage = rainfall.chirps_cell_monthly(
            GRID, "2020-01-01", "2020-01-01"
        )
        self.assertTrue(np.isnan(matrix[0, 0, 0]) or matrix[0, 0, 0] > 0,
                        "an ocean cell must be absent, never zero rain")


if __name__ == "__main__":
    unittest.main()