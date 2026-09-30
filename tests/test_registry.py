"""The registry must agree with the code that computes the numbers.

A registry that drifts from reality is worse than no registry: it is a
confident, published, machine-readable source of wrong answers, and a partner
would reasonably cite it. So these tests are not about the registry being
well-formed -- they are about each declared value equalling the value the
product's own code uses, computed rather than restated where possible.

The ERA5 cell area is genuinely computed from the grid spacing and the latitude
it is served at, because "one cell covers about 700 km2" is the single most
load-bearing honest claim in the catalogue and it should be arithmetic, not a
number someone remembered.
"""

import math
import unittest

import rainfall
import registry
import sensors
import vegetation_series as vs

# The published portfolio sits here, and cell area varies with latitude, so the
# meaningful-area floor is computed at the latitude we actually serve rather than
# at the equator, where it would be about 11% larger and quietly wrong.
PORTFOLIO_LATITUDE = 0.9


def era5_cell_area_km2(latitude: float = PORTFOLIO_LATITUDE) -> float:
    """Ground area of one ERA5 cell at a given latitude."""
    side_km = rainfall.ERA5_GRID_DEGREES * 111.32
    return side_km ** 2 * math.cos(math.radians(latitude))


class AgreementWithRealCodeTests(unittest.TestCase):
    """Each declared value must equal what the product's code actually uses."""

    def test_era5_grid_matches_the_rainfall_module(self):
        self.assertEqual(
            registry.RAINFALL.native_grid_degrees, rainfall.ERA5_GRID_DEGREES,
            "the registry's ERA5 grid no longer matches rainfall.ERA5_GRID_DEGREES; "
            "one of them has been edited without the other",
        )

    def test_era5_resolution_km_matches_what_we_already_publish(self):
        # rainfall.py publishes this string to every caller, so a divergence here
        # means the same product is described two ways in two payloads.
        published = round(rainfall.ERA5_GRID_DEGREES * 111.32, 1)
        self.assertEqual(registry.RAINFALL.resolution_km(), published)

    def test_the_vegetation_series_resolution_is_the_grid_not_the_label(self):
        # The single most confusing pair of numbers in the catalogue: MODIS is
        # called 250 m everywhere in the product literature and its actual
        # sinusoidal grid is 231.7 m. Both are true and the difference is 8%.
        self.assertEqual(
            registry.VEGETATION_SERIES.native_resolution_m, vs.MODIS_NATIVE_M
        )
        self.assertNotEqual(
            registry.VEGETATION_SERIES.native_resolution_m,
            sensors.get_sensor("modis").native_res_m,
            "the registry has collapsed the sinusoidal grid and the product "
            "label into one number, which is the confusion this exists to stop",
        )

    def test_evidence_classes_match_what_the_request_path_assigns(self):
        # main.py assigns these at the point of assembly. If a class changes
        # there and not here, /version and the payload disagree.
        import main

        for key, expected in (("dem", registry.OBSERVED),
                              ("landcover", registry.OBSERVED),
                              ("ndvi", registry.DERIVED),
                              ("rainfall", registry.MODELLED)):
            self.assertEqual(registry.get(key).evidence, expected)
        self.assertIn(registry.DERIVED, (registry.OBSERVED, registry.DERIVED,
                                         registry.MODELLED))
        self.assertTrue(hasattr(main, "AVAILABLE_DATASETS"))

    def test_the_cost_classes_match_what_the_worker_can_actually_queue(self):
        # A product the registry calls live-only but the worker can queue will
        # make a user wonder why a button produced a job. Found the hard way:
        # this test failed because ndvi *is* in RUNTIME_INDICATORS, so the
        # original single-class model was wrong -- the area decides, not the
        # product.
        import jobs

        for indicator in jobs.RUNTIME_INDICATORS:
            with self.subTest(indicator=indicator):
                product = registry.get(indicator)
                self.assertIsNotNone(product, f"{indicator} is queueable and "
                                              f"therefore belongs in the registry")
                self.assertIn(registry.COMPUTED, product.costs())
        for key, product in registry.PRODUCTS.items():
            with self.subTest(product=key):
                for cost in product.costs():
                    self.assertIn(cost, (registry.LIVE, registry.COMPUTED))


class MeaningfulAreaTests(unittest.TestCase):
    def test_the_era5_floor_is_one_cell_at_our_latitude(self):
        floor = registry.RAINFALL.meaningful_min_km2
        cell = era5_cell_area_km2()
        self.assertIsNotNone(floor)
        # Tight, because this figure was 9.6% wrong when first written from
        # memory and only became correct by computing it. A loose bound would
        # have let that stand.
        self.assertLess(
            abs(floor - cell) / cell, 0.05,
            f"the declared floor {floor} km2 is not one ERA5 cell at "
            f"{PORTFOLIO_LATITUDE} N (computed {cell:.0f} km2)",
        )

    def test_the_floor_is_larger_than_the_largest_area_we_serve_synchronously(self):
        import main

        # The point of the floor: it is above the 100 km2 synchronous cap, so
        # any area a user can analyse live is below the resolution of the
        # rainfall measure. If that ever stops being true the warning copy that
        # depends on it is wrong.
        self.assertGreater(registry.RAINFALL.meaningful_min_km2,
                           main.MAX_SYNC_BBOX_KM2)


class LatencyTests(unittest.TestCase):
    def test_every_dynamic_product_declares_a_latency(self):
        for product in registry.PRODUCTS.values():
            if product.evidence == registry.MODELLED or product.cost == registry.COMPUTED:
                self.assertIsNotNone(
                    product.latency_days,
                    f"{product.key} is dynamic and must say how old it is; a "
                    f"product with no latency is indistinguishable from a static one",
                )

    def test_latency_is_a_range_where_the_cadence_varies(self):
        # ERA5 monthly is a range, not a number: the newest available month
        # depends on when you ask. Publishing a single integer would be a claim
        # about a cadence that does not exist. A static product declares no
        # latency at all rather than a range of one value.
        self.assertIn("-", registry.RAINFALL.latency_days)
        self.assertIsNone(registry.ELEVATION.latency_days)
        self.assertIsNone(registry.LANDCOVER.latency_days)

    def test_static_products_declare_no_latency(self):
        for key in ("dem", "landcover"):
            self.assertIsNone(registry.get(key).latency_days,
                              f"{key} is a static surface; a latency would imply "
                              f"it changes")


class SubstitutabilityTests(unittest.TestCase):
    def test_products_sharing_a_measure_are_interchangeable_consumers(self):
        # This is what makes a second rainfall product a registry entry rather
        # than a rewrite, so it is the property the whole design rests on.
        by_measure = registry.measures()
        self.assertIn("vegetation_index", by_measure)
        self.assertEqual(
            sorted(by_measure["vegetation_index"]),
            sorted(["ndvi", "vegetation_series"]),
            "two products that measure the same thing must be listed under one "
            "measure, or swapping one for the other is not a registry change",
        )
        self.assertEqual(by_measure["precipitation_total"], ["rainfall"])

    def test_every_product_has_a_measure_and_evidence(self):
        for product in registry.PRODUCTS.values():
            self.assertTrue(product.measure, f"{product.key} has no measure")
            self.assertIn(product.evidence,
                          (registry.OBSERVED, registry.DERIVED, registry.MODELLED))


class PublishedFormTests(unittest.TestCase):
    def test_the_published_form_is_json_serialisable(self):
        import json

        json.dumps(registry.describe_all())

    def test_a_published_product_is_self_contained(self):
        # A partner reads /version and should not have to guess what a missing
        # field means. Everything a reader needs is present.
        for payload in registry.describe_all():
            for field in ("key", "label", "measure", "units", "evidence",
                          "cost", "source", "caveats"):
                self.assertIn(field, payload)
            self.assertTrue(payload["caveats"],
                            f"{payload['key']} publishes with no caveat")

    def test_describe_all_is_sorted_so_output_diffs_cleanly(self):
        keys = [p["key"] for p in registry.describe_all()]
        self.assertEqual(keys, sorted(keys))


class PublishedLimitTests(unittest.TestCase):
    """/version has to publish the numbers the service actually enforces.

    Fifteen things were enforced and unpublished, including the raster caps and
    the vertex limit, so a client could not know what would be accepted. And one
    published string was false: large_area_mode claimed no durable worker was
    deployed, which it was. These assertions exist so the next addition to the
    enforced set arrives already published.
    """

    def _version(self) -> dict:
        import asyncio

        import main

        return asyncio.run(main.get_version())

    def test_it_publishes_every_registered_product(self):
        published = {p["key"] for p in self._version()["products"]}
        self.assertEqual(published, set(registry.PRODUCTS),
                         "the registry holds a product /version does not publish, "
                         "or publishes one the registry does not hold")

    def test_it_publishes_the_enforced_input_limits(self):
        import main

        version = self._version()
        self.assertEqual(version["max_geojson_bytes"], main.MAX_GEOJSON_BYTES)
        self.assertEqual(version["max_aoi_vertices"], main.MAX_AOI_VERTICES)

    def test_it_publishes_the_timeouts_the_request_path_enforces(self):
        # These were inline literals at three call sites, so the only honest
        # version of this line did not exist. Naming them made it possible.
        import main

        version = self._version()
        self.assertEqual(version["analysis_raster_timeout_seconds"],
                         main.RASTER_TIMEOUT_SECONDS)
        self.assertEqual(version["ndvi_timeout_seconds"], main.NDVI_TIMEOUT_SECONDS)

    def test_large_area_mode_is_not_claiming_a_capability_we_lack(self):
        import jobs

        version = self._version()
        self.assertTrue(version["large_area_mode"]["available"])
        self.assertEqual(version["large_area_mode"]["queue_max_pending"],
                         jobs.MAX_PENDING_JOBS)
        self.assertEqual(version["large_area_mode"]["job_lease_seconds"],
                         jobs.JOB_LEASE_SECONDS)

    def test_the_resolution_steps_are_the_ladder_that_actually_runs(self):
        import sensors

        version = self._version()
        published = [(s["max_bbox_km2"], s["resolution_m"])
                     for s in version["ndvi_resolution_steps"]]
        from_ladder = [(b, m) for b, m, _ in sensors.RESOLUTION_STEPS]
        self.assertEqual(published, from_ladder)
        self.assertEqual(version["coarsest_resolution_m"],
                         sensors.COARSEST_RESOLUTION_M)
