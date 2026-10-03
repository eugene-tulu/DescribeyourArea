"""CHIRPS, as a second product for the same measure.

Two things this file exists to prevent, both of which fail silently:

**Two products, one cache key.** A cache keyed on the area hash alone means ERA5
and CHIRPS for the same area collide, and a reader who asked for CHIRPS is served
whichever was written first. That is the worst shape of wrong answer: the number
looks right, the evidence label says "observed", and it is a 0.25 degree reanalysis.

**A gap read as a drought.** The published DEA archive is missing 2023-12,
2024-07 and 2024-08 -- verified by direct probe, each returning 404. A reader
that absorbs a hole produces a series with a zero in it, and a zero in a rainfall
series is a drought that never happened.
"""

import json
import unittest

import rainfall
import registry


class CacheIdentityTests(unittest.TestCase):
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

    KEY = "a" * 32
    V = rainfall.RAINFALL_PROCESSING_VERSION

    def test_the_two_products_do_not_collide(self):
        rainfall.write_cache(self.KEY, {"product": "rainfall",
                                         "processing_version": self.V, "series": []})
        rainfall.write_cache(self.KEY, {"product": "chirps",
                                         "processing_version": self.V,
                                         "series": [{"month": "2026-01"}]},
                             product="chirps")
        self.assertEqual(rainfall.read_cache(self.KEY)["product"], "rainfall")
        self.assertEqual(rainfall.read_cache(self.KEY, "chirps")["product"], "chirps")

    def test_the_default_product_keeps_its_original_filename(self):
        # An existing portfolio must not be orphaned by a layout change nobody
        # asked for.
        self.assertEqual(rainfall.cache_path(self.KEY).name, f"{self.KEY}.json")
        self.assertEqual(rainfall.cache_path(self.KEY, "chirps").name,
                         f"{self.KEY}-chirps.json")

    def test_an_older_file_without_a_product_tag_still_reads_as_era5(self):
        rainfall.cache_path(self.KEY).write_text(
            json.dumps({"processing_version": self.V, "series": []}))
        self.assertIsNotNone(rainfall.read_cache(self.KEY))
        self.assertIsNone(rainfall.read_cache(self.KEY, "chirps"),
                          "an ERA5 file must not be served to someone who asked "
                          "for CHIRPS")


class SourceTests(unittest.TestCase):
    def test_a_source_is_named_rather_than_passed_as_a_callable(self):
        # Call sites -- a request, a job, the precompute -- would each decide
        # which product to read, and that is how a request ends up answering a
        # different question from the one a job was queued for.
        self.assertIsNone(rainfall.reader_for("rainfall"))
        self.assertEqual(rainfall.reader_for("chirps").__name__,
                         "chirps_cell_monthly")

    def test_an_unknown_source_names_the_ones_that_exist(self):
        with self.assertRaises(ValueError) as caught:
            rainfall.reader_for("smoke-signals")
        self.assertIn("rainfall, chirps", str(caught.exception))


class RegistryTests(unittest.TestCase):
    def test_it_is_a_second_producer_of_the_same_measure(self):
        # This is the substitutability the registry was built for: two products,
        # one measure, so choosing between them is a selection rather than a
        # rewrite.
        self.assertEqual(registry.measures()["precipitation_total"],
                         ["chirps", "rainfall"])

    def test_it_earns_observed_where_era5_is_modelled(self):
        # It is a satellite-and-gauge blend, not a model output. Calling it
        # modelled would understate it; calling ERA5 observed would overstate it.
        self.assertEqual(registry.RAINFALL_CHIRPS.evidence, registry.OBSERVED)
        self.assertEqual(registry.RAINFALL.evidence, registry.MODELLED)

    def test_it_is_meaningful_where_era5_is_not(self):
        # 0.05 deg against 0.25. ERA5's meaningful floor is one 774 km2 cell;
        # CHIRPS at 31 km2 is about a small area, which is the case most study
        # areas are.
        self.assertLess(registry.RAINFALL_CHIRPS.meaningful_min_km2,
                        registry.RAINFALL.meaningful_min_km2)

    def test_the_archive_holes_are_stated_on_the_product(self):
        # A caveat a reader never sees is not a caveat.
        caveats = " ".join(registry.RAINFALL_CHIRPS.caveats)
        self.assertIn("2023-12", caveats)
        self.assertIn("missing month is not a dry month", caveats.lower()
                      .replace("a missing", "missing"))

    def test_the_measured_difference_from_era5_is_stated(self):
        caveats = " ".join(registry.RAINFALL_CHIRPS.caveats)
        self.assertIn("92%", caveats)
        self.assertIn("24%", caveats)


class GapReportingTests(unittest.TestCase):
    def test_unreadable_months_are_returned_not_absorbed(self):
        # The reader is exercised against the published gaps by asking for a span
        # that includes one. It is slow, so it is skipped without network access.
        import os

        if os.getenv("CHIRPS_LIVE") != "1":
            self.skipTest("set CHIRPS_LIVE=1 to read the published archive")
        grid = {"longitudes": [37.75], "latitudes": [1.25],
                "lon_step": 0.25, "lat_step": 0.25}
        _, _, coverage = rainfall.chirps_cell_monthly(grid, "2024-06-01", "2024-09-01")
        self.assertEqual(coverage["source"], "chirps-v2.0")
        self.assertIn("2024-07-01", coverage["unreadable_months"])
        self.assertIn("2024-08-01", coverage["unreadable_months"])


class ApiPlumbingTests(unittest.TestCase):
    """The product has to survive the whole path, or the choice is decorative.

    Request to job record to worker to cache to reader. A break anywhere leaves a
    reader who asked for CHIRPS being served ERA5, which is the one outcome this
    work exists to prevent.
    """

    def setUp(self):
        import os
        import tempfile

        self.previous = os.environ.get("RAINFALL_CACHE_DIR")
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["RAINFALL_CACHE_DIR"] = self.tmp.name
        self.client = _client()
        self.addCleanup(self._restore)
        self.addCleanup(self.tmp.cleanup)

    def _restore(self):
        import os

        if self.previous is None:
            os.environ.pop("RAINFALL_CACHE_DIR", None)
        else:
            os.environ["RAINFALL_CACHE_DIR"] = self.previous

    KEY = "a" * 32
    AOI = {"type": "Feature", "properties": {},
           "geometry": {"type": "Polygon",
                        "coordinates": [[[37.70, 1.20], [37.80, 1.20], [37.80, 1.30],
                                         [37.70, 1.30], [37.70, 1.20]]]}}

    def test_each_product_gets_its_own_evidence_class(self):
        import main

        rainfall.write_cache(self.KEY, {
            "product": "chirps",
            "processing_version": rainfall.RAINFALL_PROCESSING_VERSION,
            "series": [{"month": "2026-01", "precip_mm": 10.0}]}, product="chirps")
        rainfall.write_cache(self.KEY, {
            "product": "rainfall",
            "processing_version": rainfall.RAINFALL_PROCESSING_VERSION,
            "series": [{"month": "2026-01", "precip_mm": 1.0}]})
        self.assertEqual(main._rainfall_by_key(self.KEY, "chirps")["evidence"]["status"],
                         "observed")
        self.assertEqual(main._rainfall_by_key(self.KEY, "rainfall")["evidence"]["status"],
                         "modelled")

    def test_a_submission_records_the_product_it_was_given(self):
        import os

        os.environ["RAINFALL_AUTORUN"] = "0"
        response = self.client.post(
            "/rainfall/submit", json={"geojson": self.AOI, "product": "chirps"})
        self.assertEqual(response.status_code, 200, response.text[:200])
        self.assertEqual(response.json()["product"], "chirps")
        import jobs

        self.assertEqual(jobs._job_product(response.json()["cache_key"]), "chirps")

    def test_an_unknown_product_is_refused_with_the_ones_that_exist(self):
        response = self.client.post(
            "/rainfall/submit", json={"geojson": self.AOI, "product": "smoke-signals"})
        self.assertEqual(response.status_code, 422)
        self.assertIn("rainfall, chirps", response.json()["detail"])

    def test_the_status_route_reports_the_product_asked_about(self):
        import jobs

        jobs.write_job({"cache_key": self.KEY, "indicator": "rainfall",
                        "product": "chirps", "state": "pending",
                        "submitted_at": "2026-09-28T00:00:00+00:00"})
        body = self.client.get(
            f"/rainfall/status?cache_key={self.KEY}&product=chirps").json()
        self.assertEqual(body["product"], "chirps")
        # An ERA5 series must not make a CHIRPS job look finished.
        rainfall.write_cache(self.KEY, {
            "product": "rainfall",
            "processing_version": rainfall.RAINFALL_PROCESSING_VERSION,
            "series": []})
        self.assertFalse(
            self.client.get(f"/rainfall/status?cache_key={self.KEY}&product=chirps")
            .json()["computed"])


def _client():
    from fastapi.testclient import TestClient

    import main

    return TestClient(main.app)


class CostTests(unittest.TestCase):
    """The estimate a user waits against has to belong to the product they chose.

    CHIRPS is a separate object per month; ERA5 reads its whole series from one
    asset. Measured on this host: 1.04 s a month against ERA5's 0.12. A CHIRPS job
    planned with ERA5's number promised 24 seconds and took 17 minutes -- so the
    cost now travels with the product declaration rather than in a table kept
    somewhere else, which is the same duplication the registry was built to end.
    """

    def test_each_queued_product_declares_its_own_cost(self):
        for key in ("rainfall", "chirps", "vegetation_series"):
            with self.subTest(product=key):
                self.assertIsNotNone(registry.get(key).seconds_per_month,
                                     f"{key} is queued and needs a cost")

    def test_the_two_products_are_not_reported_as_the_same_cost(self):
        import indicators

        plan = indicators.plan_indicator("rainfall", 5_000,
                                          start="2010-01-01", end="2026-03-01",
                                          product="chirps")
        era5 = indicators.plan_indicator("rainfall", 5_000,
                                         start="2010-01-01", end="2026-03-01")
        self.assertGreater(plan["estimated_seconds"], era5["estimated_seconds"] * 4)
        self.assertEqual(plan["product"], "chirps")

    def test_the_estimate_follows_the_window(self):
        import indicators

        short = indicators.plan_indicator("rainfall", 5_000,
                                          start="2024-01-01", end="2024-12-01")
        long = indicators.plan_indicator("rainfall", 5_000,
                                         start="2010-01-01", end="2026-03-01")
        self.assertLess(short["months"], long["months"])
        self.assertLess(short["estimated_seconds"], long["estimated_seconds"])

    def test_the_indicator_names_its_own_product(self):
        # A caller asking for "chirps" plainly means CHIRPS; making it also pass a
        # product argument invites two things naming it and disagreeing.
        import indicators

        plan = indicators.plan_indicator("chirps", 5_000,
                                         start="2010-01-01", end="2026-03-01")
        self.assertEqual(plan["product"], "chirps")
        self.assertLess(plan["resolution_km"], 10.0)

    def test_an_unknown_product_falls_back_rather_than_crashing(self):
        import indicators

        plan = indicators.plan_indicator("rainfall", 5_000,
                                         start="2010-01-01", end="2026-03-01",
                                         product="smoke-signals")
        self.assertEqual(plan["product"], "rainfall")

    def test_the_resolution_reported_is_the_products_own(self):
        import indicators

        chirps = indicators.plan_indicator("chirps", 5_000,
                                           start="2010-01-01", end="2026-03-01")
        era5 = indicators.plan_indicator("rainfall", 5_000,
                                         start="2010-01-01", end="2026-03-01")
        self.assertLess(chirps["resolution_km"], era5["resolution_km"] / 4)

    def test_the_vegetation_cost_is_read_from_its_own_module(self):
        # Restated, it would drift from the constant the estimate actually uses.
        import vegetation_series

        self.assertEqual(registry.get("vegetation_series").seconds_per_month,
                         vegetation_series.SECONDS_PER_READ)


class PlanPricingTests(unittest.TestCase):
    """The preview has to be priced for the product too.

    A preview that prices a CHIRPS job with ERA5's cost is the number the reader
    decides on, so getting it wrong there is worse than getting it wrong on the
    job itself. Found by reading the route's own output: it returned 27.8 km and
    24 seconds for a CHIRPS request because the plan route never passed the
    product through.
    """

    def setUp(self):
        from fastapi.testclient import TestClient

        import main

        self.client = TestClient(main.app)
        self.aoi = {"type": "Feature", "properties": {}, "geometry": {
            "type": "Polygon", "coordinates": [[[37.60, -0.40], [37.70, -0.40],
                                                [37.70, -0.32], [37.60, -0.32],
                                                [37.60, -0.40]]]}}

    def _plan(self, product):
        response = self.client.post("/rainfall/plan", json={
            "geojson": self.aoi, "indicator": "rainfall", "product": product})
        self.assertEqual(response.status_code, 200, response.text[:200])
        return next(p for p in response.json()["plans"] if p["indicator"] == "rainfall")

    def test_the_preview_prices_the_product_asked_for(self):
        chirps, era5 = self._plan("chirps"), self._plan("rainfall")
        self.assertEqual(chirps["product"], "chirps")
        self.assertEqual(era5["product"], "rainfall")

    def test_the_preview_states_the_resolution_and_the_wait(self):
        chirps = self._plan("chirps")
        self.assertLess(chirps["resolution_km"], 10.0,
                        "a CHIRPS preview priced at ERA5's resolution is the "
                        "wrong answer twice over")
        self.assertGreater(chirps["estimated_seconds"], 120)

    def test_the_basis_names_the_product_it_was_measured_on(self):
        basis = self._plan("chirps")["estimate_basis"]
        self.assertIn("1.04", basis)
        self.assertIn("CHIRPS", basis)


class CacheSlotTests(unittest.TestCase):
    """A CHIRPS job that reports ready and stores nothing is the worst outcome.

    The worker resolved the product name to a reader callable before handing it
    over, and the cache works out which slot to write from what it is given. A
    callable carries no name, so it arrived as "rainfall" and the CHIRPS series
    was written into the ERA5 slot. The job completed, the state said ready, and
    a reader found no series at all -- a success message over no data.
    """

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

    def test_a_named_product_reaches_the_cache_it_belongs_in(self):
        import inspect

        # No network: this asserts the identity survives to the cache decision.
        source = inspect.getsource(rainfall.build_and_cache)
        self.assertIn("write_cache(key, payload, product)", source,
                      "a named product must select its own cache slot")

    def test_a_reader_with_no_name_is_not_guessed_at(self):
        import inspect

        source = inspect.getsource(rainfall.build_and_cache)
        self.assertIn('product = named or "rainfall"', source)

    def test_the_worker_passes_the_name_rather_than_resolving_it(self):
        import inspect

        import jobs

        source = inspect.getsource(jobs.run_pending)
        self.assertIn("product=product,", source,
                      "the worker must pass the name through; resolving it to a "
                      "callable loses the identity the cache slot is chosen by")


class ReportedGridTests(unittest.TestCase):
    """A series must report the grid it was actually read at.

    Hard-coding ERA5's had a CHIRPS series printing 27.8 km and 0.25 degrees
    beside a value computed from a completely different raster. The same class of
    error as calling one modelled: the provenance contradicts the number.
    """

    def test_a_chirps_series_reports_its_own_grid(self):
        import os
        import tempfile

        previous = os.environ.get("RAINFALL_CACHE_DIR")
        tmp = tempfile.TemporaryDirectory()
        os.environ["RAINFALL_CACHE_DIR"] = tmp.name
        try:
            geom = {"type": "Polygon", "coordinates": [
                [[37.70, 1.20], [37.80, 1.20], [37.80, 1.30], [37.70, 1.30], [37.70, 1.20]]]}
            chirps = rainfall.build_and_cache(
                geom, start="2024-01-01", end="2024-03-01",
                source=rainfall.reader_for("chirps"), product="chirps")
            era5 = rainfall.build_and_cache(geom, start="2024-01-01", end="2024-03-01")
            self.assertLess(chirps["resolution_km"], 10.0)
            self.assertLess(chirps["resolution_km"], era5["resolution_km"])
            self.assertEqual(chirps["resolution_degrees"],
                             rainfall.CHIRPS_NATIVE_GRID_DEGREES)
            self.assertEqual(chirps["product"], "chirps")
            self.assertEqual(era5["product"], "rainfall")
        finally:
            if previous is None:
                os.environ.pop("RAINFALL_CACHE_DIR", None)
            else:
                os.environ["RAINFALL_CACHE_DIR"] = previous
            tmp.cleanup()
