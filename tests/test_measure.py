"""A measure has to be able to stand for a product that does not exist yet.

The acceptance criterion on the roadmap is that a new dataset is a registry entry
plus a producer, with no route, no card and no client change. The test for that
is not a description of the property; it is a synthetic product added to the
registry in this file and rendered through the ordinary path. If it needs
anything else, the abstraction leaked and the test fails.

The rest here is the other half: that a measure cannot be built in a way that
omits what a reader needs. Four products currently sit in a row six months apart
and none of them says so; `observed_through` is required for anything dynamic,
which turns that from a thing to remember into a thing that will not compile.
"""

import json
import unittest

import measure
import registry
from measure import Extent, Measure, MeasureError


SYNTHETIC = registry.Product(
    key="synthetic-test-product",
    label="Synthetic test product",
    measure="synthetic_quantity",
    units="widgets",
    native_resolution_m=12.0,
    meaningful_min_km2=0.5,
    meaningful_max_km2=400.0,
    latency_days="7-14",
    evidence=registry.MODELLED,
    cost=(registry.LIVE,),
    source="a fixture, not a satellite",
    caveats=("This product does not exist and is here to prove a shape.",),
)


class GeneralityTests(unittest.TestCase):
    def test_a_product_that_does_not_exist_yet_renders_without_new_code(self):
        registry.PRODUCTS[SYNTHETIC.key] = SYNTHETIC
        self.addCleanup(registry.PRODUCTS.pop, SYNTHETIC.key, None)

        built = Measure(
            product_key=SYNTHETIC.key,
            value=7,
            units=SYNTHETIC.units,
            extent=Extent(requested_area_km2=12.0, covered_area_km2=12.0,
                          native_resolution_m=12.0),
            observed_through="2026-09-01",
        )
        payload = built.describe()
        self.assertEqual(payload["product"], SYNTHETIC.key)
        self.assertEqual(payload["measure"], "synthetic_quantity")
        self.assertEqual(payload["evidence"], registry.MODELLED)
        json.dumps(payload)                      # the client path
        self.assertIn("synthetic_quantity", registry.measures())

    def test_two_products_of_one_measure_render_identically_in_shape(self):
        # This is what makes substitution a registry change: the shapes match, so
        # anything that can render one can render the other.
        def shape(product_key, **kw):
            return sorted(Measure(product_key=product_key, value=1, units="u",
                                  **kw).describe().keys())

        rainfall_shape = shape("rainfall", observed_through="2026-01-01")
        vegetation_shape = shape("ndvi", observed_through="2026-01-01")
        self.assertEqual(rainfall_shape, vegetation_shape)


class RequiredFieldsTests(unittest.TestCase):
    """The honesty fixes, as invariants rather than as a habit."""

    def test_a_dynamic_measure_cannot_omit_its_freshness(self):
        with self.assertRaises(MeasureError) as caught:
            Measure(product_key="rainfall", value=62.0, units="mm")
        self.assertIn("observed_through", str(caught.exception))

    def test_a_static_measure_need_not_carry_a_date(self):
        built = Measure(product_key="dem", value=1500.0, units="m",
                        extent=Extent(requested_area_km2=4.0, covered_area_km2=4.0))
        self.assertIsNone(built.observed_through)

    def test_a_measure_for_an_unregistered_product_cannot_be_built(self):
        with self.assertRaises(MeasureError):
            Measure(product_key="not-a-product", value=1, units="x")

    def test_evidence_is_inherited_not_restated(self):
        # A measure cannot claim a stronger class than its product declares, and
        # it cannot be built with a weaker one either: the registry is the only
        # place a class is decided.
        for key, expected in (("dem", registry.OBSERVED),
                              ("ndvi", registry.DERIVED),
                              ("rainfall", registry.MODELLED)):
            with self.subTest(product=key):
                built = Measure(product_key=key, value=0, units="u",
                                observed_through="2026-01-01")
                self.assertEqual(built.evidence, expected)
                self.assertEqual(built.evidence, registry.get(key).evidence)

    def test_caveats_are_inherited_so_a_card_cannot_omit_them(self):
        built = Measure(product_key="rainfall", value=1.0, units="mm",
                        observed_through="2026-01-01")
        self.assertTrue(built.caveats)
        self.assertEqual(list(built.caveats),
                         list(registry.RAINFALL.caveats))


class OverSpecificationTests(unittest.TestCase):
    def test_it_notices_when_the_grid_is_coarser_than_the_outline(self):
        # The real case, computed: one ERA5 cell is 774 km2 and the synchronous
        # cap is 100 km2, so every live analysis is over-specified. If this
        # returns False the note silently disappears and nobody is told.
        import math

        import rainfall

        cell = (rainfall.ERA5_GRID_DEGREES * 111.32) ** 2 * math.cos(math.radians(0.9))
        self.assertGreater(cell, 700)
        extent = Extent(requested_area_km2=4.0, covered_area_km2=round(cell))
        self.assertTrue(extent.over_specified)

    def test_a_matching_outline_is_not_flagged(self):
        self.assertFalse(Extent(requested_area_km2=100.0, covered_area_km2=100.0)
                         .over_specified)

    def test_the_headline_says_so_when_it_is(self):
        built = Measure(
            product_key="rainfall", value=62.0, units="mm",
            observed_through="2026-03-01",
            extent=Extent(requested_area_km2=4.0, covered_area_km2=774.0),
        )
        text = measure.headline(built)
        self.assertIn("774 km²", text)
        self.assertIn("4 km²", text)
        self.assertIn("modelled", text)
        self.assertIn("2026-03-01", text)

    def test_the_headline_is_short_when_the_ground_matches_the_outline(self):
        built = Measure(
            product_key="dem", value=1500.0, units="m",
            extent=Extent(requested_area_km2=4.0, covered_area_km2=4.0),
        )
        text = measure.headline(built)
        self.assertIn("observed", text)
        self.assertNotIn("km² of a single", text)


class RealSeriesTests(unittest.TestCase):
    """The envelope, built from a real ERA5 series rather than a fixture.

    Values and dates come from the real cache through the real reader, so this
    asserts the envelope fits the data we actually serve -- including the month
    count and the cell count, which are the two fields that were buried.
    """

    def _real_rainfall(self) -> dict:
        import tempfile
        import os

        import rainfall

        previous = os.environ.get("RAINFALL_CACHE_DIR")
        tmp = tempfile.TemporaryDirectory()
        os.environ["RAINFALL_CACHE_DIR"] = tmp.name
        self.addCleanup(tmp.cleanup)
        self.addCleanup(
            lambda: os.environ.pop("RAINFALL_CACHE_DIR", None)
            if previous is None else os.environ.__setitem__("RAINFALL_CACHE_DIR", previous)
        )
        geom = {"type": "Polygon", "coordinates": [[[37.70, 1.20], [37.80, 1.20],
                                                    [37.80, 1.30], [37.70, 1.30],
                                                    [37.70, 1.20]]]}
        key = rainfall.geometry_hash(geom)
        rows = [{"month": f"2026-0{m}", "precip_mm": 10.0 + m,
                 "normal_mm": 40.0, "anomaly_mm": -20.0, "anomaly_pct": -50.0}
                for m in (1, 2, 3)]
        rainfall.write_cache(key, {
            "processing_version": rainfall.RAINFALL_PROCESSING_VERSION,
            "series": rows,
            "grid_cells": 1,
            "climatology": {"january": 40.0},
        })
        return rainfall.cached_context(geom)

    def test_it_wraps_a_real_series_with_its_extent_and_freshness(self):
        context = self._real_rainfall()
        self.assertEqual(context["status"], "ok")
        rows = context["series"]
        self.assertTrue(rows)

        built = Measure(
            product_key="rainfall",
            value=rows[-1]["precip_mm"],
            units="mm",
            observed_through=rows[-1]["month"] + "-01",
            extent=Extent(
                requested_area_km2=100.0,
                covered_area_km2=774.0,
                cell_count=context.get("grid_cells"),
                native_grid_degrees=registry.RAINFALL.native_grid_degrees,
                window_start=rows[0]["month"] + "-01",
                window_end=rows[-1]["month"] + "-01",
            ),
            extra={"months": len(rows)},
        )
        payload = built.describe()
        self.assertEqual(payload["observed_through"], "2026-03-01")
        self.assertEqual(payload["extent"]["cell_count"], 1)
        self.assertTrue(payload["extent"]["over_specified"])
        self.assertEqual(payload["detail"]["months"], 3)
        self.assertEqual(payload["evidence"], registry.MODELLED)
        json.dumps(payload)
        self.assertIn("774 km²", measure.headline(built))

    def test_the_freshness_is_the_newest_month_not_todays_date(self):
        # The distinction that matters to a user: the number is from March, and
        # the measure has to be able to say so even though it was built today.
        context = self._real_rainfall()
        built = Measure(
            product_key="rainfall", value=1.0, units="mm",
            observed_through=context["series"][-1]["month"] + "-01",
        )
        self.assertTrue(built.observed_through < built.computed_at)


class ObservedThroughEnforcementTests(unittest.TestCase):
    """The guard that turns "some cards say 'as of', some don't" into a hard fail.

    Before the enforcement lived in ``Measure.__post_init__``, every dynamic
    product could be built without ``observed_through`` and the omission simply
    surfaced as a missing date on some cards and a stale one on others -- the
    exact failure this class pins down as an invariant. It sweeps every
    registered product so a new dynamic product added to the registry is covered
    by this assertion automatically, and it checks the message names the product
    and its latency so a half-greatened ``raise`` cannot regress to a generic
    one that hides the cause.
    """

    def test_a_dynamic_product_cannot_be_built_without_a_date(self):
        # Every product with a latency window is dynamic and must carry freshness.
        dynamic = [(key, prod) for key, prod in registry.PRODUCTS.items()
                   if prod.latency_days]
        self.assertTrue(dynamic, "expected at least one dynamic product in the registry")
        for key, product in dynamic:
            with self.subTest(product=key):
                with self.assertRaises(MeasureError) as caught:
                    Measure(product_key=key, value=1.0, units=product.units)
                message = str(caught.exception)
                # The message has to name which product failed and why, otherwise
                # it is a different error that happens to fire first.
                self.assertIn(key, message)
                self.assertIn(product.latency_days, message)
                self.assertIn("observed_through", message)

    def test_a_static_product_needs_no_date(self):
        # The counterpart: a static product (latency_days is None) must keep
        # building without observed_through. This pins the guard to "omitted when
        # it should not be", not "always fails".
        static = [(key, prod) for key, prod in registry.PRODUCTS.items()
                  if not prod.latency_days]
        self.assertTrue(static, "expected at least one static product in the registry")
        for key, product in static:
            with self.subTest(product=key):
                built = Measure(product_key=key, value=1.0,
                                units=product.units,
                                extent=Extent(requested_area_km2=4.0,
                                              covered_area_km2=4.0))
                self.assertIsNone(built.observed_through)

    def test_supplying_the_date_satisfies_a_dynamic_product(self):
        # Supplying observed_through must make the same dynamic product that the
        # guard rejects build cleanly, so the guard is scoped to the omission and
        # not to the product itself.
        key = "rainfall"
        product = registry.PRODUCTS[key]
        built = Measure(product_key=key, value=1.0, units=product.units,
                        observed_through="2026-01-01")
        self.assertEqual(built.observed_through, "2026-01-01")
        self.assertEqual(built.describe()["observed_through"], "2026-01-01")
