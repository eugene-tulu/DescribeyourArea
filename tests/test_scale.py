"""Affordable is not meaningful, and the two must never be confused.

Four places used to decide how big an area could be and how finely to read it,
and they disagreed with each other. The registry now knows each product's native
resolution and the area its measure is about, so this module derives the rest --
and reports the two answers separately, because a cap set for memory is not a
statement about what the data can say.
"""

import unittest

import rainfall
import registry
import scale


class PixelTests(unittest.TestCase):
    def test_a_pixel_count_scales_with_area_and_inversely_with_resolution(self):
        coarse = scale.pixels_for(1_000, 100)
        fine = scale.pixels_for(1_000, 10)
        self.assertAlmostEqual(fine / coarse, 100, delta=1,
                               msg="ten times finer is a hundred times the pixels")

    def test_it_is_never_zero_for_a_real_area(self):
        self.assertGreaterEqual(scale.pixels_for(0.01, 250), 1)


class CoarseningTests(unittest.TestCase):
    def test_it_never_reads_finer_than_the_source_publishes(self):
        # Reading a 250 m product at 20 m does not add information, it
        # manufactures pixels.
        self.assertEqual(scale.coarsen_to_meaningful("vegetation_series", 100, 20), 231.7)
        self.assertEqual(scale.coarsen_to_meaningful("dem", 100, 20), 30.0)
        # Asking for 20 m from a 10 m product is fine: 20 is coarser, so it is
        # aggregation rather than manufacture. Only finer would invent pixels.
        self.assertEqual(scale.coarsen_to_meaningful("landcover", 100, 20), 20.0)
        self.assertEqual(scale.coarsen_to_meaningful("landcover", 100, 5), 10.0)

    def test_it_moves_coarser_when_asked_to(self):
        self.assertEqual(scale.coarsen_to_meaningful("ndvi", 5_000, 100), 100.0)


class MeaningTests(unittest.TestCase):
    """The distinction this module exists to keep.

    ERA5 is affordable over any area and meaningful from about one cell upward.
    The synchronous cap is 100 km2 and the cell is 774 km2, so *every* area a user
    can analyse live is below the resolution of the measure -- which is allowed,
    and has to be said.
    """

    def test_a_small_area_is_allowed_but_does_not_mean_what_it_looks_like(self):
        verdict = scale.admit("rainfall", 4.0, max_area_km2=100.0)
        self.assertTrue(verdict.allowed)
        self.assertFalse(verdict.meaning)
        self.assertIn("774 km²", verdict.meaning_note)

    def test_a_large_enough_area_is_allowed_and_meaningful(self):
        verdict = scale.admit("rainfall", 1_500.0, max_area_km2=5_000.0)
        self.assertTrue(verdict.allowed)
        self.assertTrue(verdict.meaning)

    def test_over_the_request_cap_is_refused_outright(self):
        verdict = scale.admit("dem", 600.0, max_area_km2=100.0)
        self.assertFalse(verdict.allowed)
        self.assertIn("600 km²", verdict.reason)

    def test_a_fine_product_has_no_meaning_floor_below_its_native_cell(self):
        # Land cover at 10 m means something about a hectare, which is the whole
        # point of it, so it is not flagged the way rainfall is.
        verdict = scale.admit("landcover", 4.0, max_area_km2=1_000.0)
        self.assertTrue(verdict.allowed)
        self.assertTrue(verdict.meaning)

    def test_an_unregistered_product_is_refused_rather_than_guessed(self):
        verdict = scale.admit("smoke-signals", 4.0, max_area_km2=100.0)
        self.assertFalse(verdict.allowed)
        self.assertIn("not a registered product", verdict.reason)

    def test_the_rainfall_floor_is_one_era5_cell(self):
        floor = registry.RAINFALL.meaningful_min_km2
        cell = (rainfall.ERA5_GRID_DEGREES * 111.32) ** 2
        self.assertAlmostEqual(floor / cell, 1.0, delta=0.05)


class LadderTests(unittest.TestCase):
    def test_it_agrees_with_the_table_it_wraps(self):
        import sensors

        for area in (10.0, 500.0, 5_000.0, 50_000.0):
            with self.subTest(area_km2=area):
                resolution, why = scale.resolution_ladder(area, native_m=10.0)
                self.assertTrue(why)
                self.assertGreaterEqual(resolution, 10.0)

    def test_a_native_250_m_product_floors_the_whole_ladder(self):
        for area in (10.0, 5_000.0, 50_000.0):
            with self.subTest(area_km2=area):
                resolution, why = scale.resolution_ladder(area, native_m=250.0)
                self.assertGreaterEqual(resolution, 250.0,
                                        "a 250 m product must never be read finer")
                self.assertTrue(why, "every branch of the ladder explains itself")


class PublishedTests(unittest.TestCase):
    def test_an_admission_describes_itself(self):
        import json

        json.dumps(scale.admit("rainfall", 4.0, max_area_km2=100.0).describe())

    def test_both_answers_are_present_in_the_published_form(self):
        described = scale.admit("rainfall", 4.0, max_area_km2=100.0).describe()
        self.assertIn("allowed", described)
        self.assertIn("meaningful", described)
        self.assertIn("meaningful_note", described)
