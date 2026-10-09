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

import numpy as np

import rainfall
import registry


def stub_reader(grid, start, end):
    """A CHIRPS-shaped reader that reads nothing.

    The tests that used the real archive took seventeen minutes between them and
    failed intermittently under load -- the same live-network flake this file's
    other gaps test avoids. What they assert is plumbing: which cache slot a
    product lands in, and which grid it reports. Neither needs a raster.

    The value has to look like weather: an early version returned 12 mm a month,
    which the plausibility guard rightly refused as no product would read that.
    """
    months = rainfall._month_range(start, end)
    shape = (len(months), len(grid["latitudes"]), len(grid["longitudes"]))
    return months, np.full(shape, 100.0), {
        "source": "stub", "months": len(months), "unreadable_months": [],
    }


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

    def test_the_live_preview_honours_the_product_choice(self):
        # The reader's choice reached /generate-context in the body and never
        # left it: cached_context was called without the product, so a request
        # that chose CHIRPS was answered from the ERA5 cache (or an ERA5-shaped
        # miss) with no evidence label saying so. Product-vs-product collision is
        # the failure this suite exists to prevent, and the live preview is where
        # it hid.
        from unittest import mock

        seen = []
        shipped = {
            "status": "ok",
            "cache_key": self.KEY,
            "processing_version": rainfall.RAINFALL_PROCESSING_VERSION,
            "window": {"start": "2026-01-01", "end": "2026-01-31"},
            "series": [{"month": "2026-01", "precip_mm": 10.0, "normal_mm": 30.0}],
        }

        def cache(geojson_geom, product="rainfall", **_window):
            seen.append(product)
            return dict(shipped, product=product)

        # A sub-cap area: /generate-context's admission policy refuses anything
        # past the synchronous limit before the rainfall read is even reached.
        small = {"type": "Feature", "properties": {},
                 "geometry": {"type": "Polygon",
                              "coordinates": [[[37.70, 1.20], [37.75, 1.20],
                                               [37.75, 1.25], [37.70, 1.25],
                                               [37.70, 1.20]]]}}
        with mock.patch.object(rainfall, "cached_context", cache):
            chirps = self.client.post(
                "/generate-context?datasets=rainfall",
                json={"geojson": small, "product": "chirps"})
        self.assertEqual(chirps.status_code, 200, chirps.text[:200])
        rain = chirps.json()["summary"]["rainfall"]
        self.assertEqual(seen, ["chirps"],
                         "generate_context must forward the caller's product to "
                         "cached_context instead of defaulting to ERA5")
        # The CHIRPS series arrives, labelled observed rather than modelled --
        # the one thing that tells a reader which product spoke.
        self.assertEqual(rain["status"], "ok", rain)
        self.assertEqual(rain["evidence"]["status"], "observed",
                         "a CHIRPS selection was labelled with ERA5's modelled class")

    def test_a_completed_chirps_read_is_displayed_not_a_miss(self):
        # /context assembles a finished reading from stored artefacts. It read
        # the rainfall module from the ERA5 cache by the area alone, so a job
        # that computed CHIRPS -- series cached under "chirps" -- came back with
        # no rainfall at all; and with dem/landcover/ndvi never queued for a
        # precipitation job, `modules` was empty and the read 404'd "nothing has
        # been computed yet" over a series that had just finished.
        import jobs

        jobs.write_job({"cache_key": self.KEY, "indicator": "rainfall",
                        "product": "chirps", "state": "ready",
                        "submitted_at": "2026-09-28T00:00:00+00:00"})
        rainfall.write_cache(self.KEY, {
            "product": "chirps",
            "processing_version": rainfall.RAINFALL_PROCESSING_VERSION,
            "window": {"start": "2026-01-01", "end": "2026-01-31"},
            "series": [{"month": "2026-01", "precip_mm": 10.0, "normal_mm": 30.0}],
        }, product="chirps")

        response = self.client.get(f"/context?cache_key={self.KEY}")
        self.assertEqual(response.status_code, 200, response.text[:300])
        rain = response.json()["summary"]["rainfall"]
        self.assertEqual(rain["status"], "ok", rain)
        self.assertEqual(rain["evidence"]["status"], "observed",
                         "the assembled reading labelled a CHIRPS series as "
                         "something other than observed")

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
        self.assertIn(str(registry.get("chirps").seconds_per_month)[:3], basis)
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
                source=stub_reader, product="chirps")
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


class HoleHandlingTests(unittest.TestCase):
    """A month the archive does not have is absent, not a number that is not one.

    CHIRPS's published archive has holes, so this is the normal case rather than
    an edge one. A NaN reached the payload and the response became a 500 -- a
    whole reading lost over one missing month, on a series that was otherwise
    fine -- and it also poisoned the annual normal, so the plausibility guard
    refused the series for what looked like a units error.
    """

    # The hole is named rather than positional. ``compute_series`` reads the union
    # of the series window and the 1991-2020 climatology in one pass and slices it,
    # so a stub that blanked a fixed row would put its hole in the climatology
    # rather than in the series being asserted about. Naming the month tests the
    # behaviour under either windowing.
    HOLE = "2024-02-01"

    def _gappy(self, grid, start, end):
        months = rainfall._month_range(start, end)
        block = np.full((len(months), len(grid["latitudes"]), len(grid["longitudes"])),
                        100.0)
        if self.HOLE in months:
            block[months.index(self.HOLE)] = np.nan  # a hole, as the archive has
            unreadable = [self.HOLE]
        else:
            unreadable = []
        return months, block, {"source": "stub", "months": len(months),
                               "unreadable_months": unreadable}

    def _build(self):
        import os
        import tempfile

        previous = os.environ.get("RAINFALL_CACHE_DIR")
        tmp = tempfile.TemporaryDirectory()
        os.environ["RAINFALL_CACHE_DIR"] = tmp.name
        self.addCleanup(
            lambda: os.environ.__setitem__("RAINFALL_CACHE_DIR", previous)
            if previous else os.environ.pop("RAINFALL_CACHE_DIR", None))
        self.addCleanup(tmp.cleanup)
        geom = {"type": "Polygon", "coordinates": [
            [[37.70, 1.20], [37.80, 1.20], [37.80, 1.30],
             [37.70, 1.30], [37.70, 1.20]]]}
        return rainfall.build_and_cache(
            geom, start="2024-01-01", end="2024-12-01",
            source=self._gappy, product="chirps")

    def test_the_payload_is_serialisable_with_a_hole_in_it(self):
        import json

        payload = self._build()
        json.dumps(payload)          # raised ValueError: nan, taking the response with it

    def test_the_missing_month_is_absent_rather_than_filled(self):
        months = [row["month"] for row in self._build()["series"]]
        self.assertNotIn("2024-02-01", months)
        self.assertNotIn("nan", [str(m) for m in months])

    def test_the_series_states_which_months_it_is_missing(self):
        payload = self._build()
        self.assertEqual(payload["unreadable_months"], ["2024-02-01"],
                         "a series shorter than the window asked for is "
                         "otherwise indistinguishable from a series for a "
                         "shorter period, which is a different claim")

    def test_the_plausibility_guard_is_not_fooled_by_the_gap(self):
        # The guard is right to refuse an implausible total; it must not be
        # refusing one because a single month was NaN.
        payload = self._build()
        annual = sum(payload["climatology"]["monthly_mean_mm"].values())
        self.assertGreater(annual, 200.0)


class ReaderResolutionTests(unittest.TestCase):
    """A named product with no reader beside it must get that product's reader.

    The worker passed a product name and no reader, expecting the callee to
    resolve it, and nothing did -- so a CHIRPS job computed ERA5 and stored it
    under a CHIRPS key. Right product name, wrong raster, and a reader shown a
    0.25 degree value described as 0.05. The job reported ready and every number
    it printed was ERA5's.
    """

    def test_a_named_product_resolves_its_own_reader(self):
        import os
        import tempfile

        previous = os.environ.get("RAINFALL_CACHE_DIR")
        tmp = tempfile.TemporaryDirectory()
        os.environ["RAINFALL_CACHE_DIR"] = tmp.name
        try:
            geom = {"type": "Polygon", "coordinates": [
                [[37.70, 1.20], [37.80, 1.20], [37.80, 1.30],
                 [37.70, 1.30], [37.70, 1.20]]]}
            # The lookup is what is under test, so the reader it returns is a
            # stub: resolving to the real one would make this a live-archive test,
            # which is slow and the flake this file otherwise avoids.
            real_reader_for = rainfall.reader_for
            rainfall.reader_for = lambda key: (
                stub_reader if key == "chirps" else real_reader_for(key))
            try:
                chirps = rainfall.build_and_cache(
                    geom, start="2024-01-01", end="2024-03-01", product="chirps")
            finally:
                rainfall.reader_for = real_reader_for
            era5 = rainfall.build_and_cache(geom, start="2024-01-01", end="2024-03-01")
            self.assertEqual(chirps["product"], "chirps")
            self.assertLess(chirps["resolution_km"], 10.0,
                            "a named product fell back to the default reader")
            self.assertGreater(era5["resolution_km"], 20.0)
        finally:
            if previous is None:
                os.environ.pop("RAINFALL_CACHE_DIR", None)
            else:
                os.environ["RAINFALL_CACHE_DIR"] = previous
            tmp.cleanup()

    def test_the_worker_still_passes_the_name_and_not_a_resolved_reader(self):
        import inspect

        import jobs

        source = inspect.getsource(jobs.run_pending)
        self.assertIn("product=product,", source)
        self.assertNotIn("union_source = rainfall.reader_for(product)", source,
                         "resolving in the worker loses the name the cache slot "
                         "is chosen by")


class CellCacheIdentityTests(unittest.TestCase):
    """The cell cache was the last layer where the two products could confuse
    themselves.

    The series cache was made product-aware, and then a CHIRPS read was served
    ERA5's cells -- so the series was built from the right product name over the
    wrong raster, and reported 27.8 km having been read at 0.05. The cells are the
    data; two products over the same cells are two different datasets.
    """

    def test_the_cell_key_carries_the_product(self):
        grid = {"longitudes": [37.75], "latitudes": [1.25]}
        era5 = rainfall.cellset_key(grid, "2024-01-01", "2024-12-01")
        chirps = rainfall.cellset_key(grid, "2024-01-01", "2024-12-01", "chirps")
        self.assertNotEqual(era5, chirps,
                            "one cell-cache key for both products is how a "
                            "CHIRPS read gets ERA5's cells")

    def test_it_is_stable_for_the_same_product(self):
        grid = {"longitudes": [37.75], "latitudes": [1.25]}
        self.assertEqual(
            rainfall.cellset_key(grid, "2024-01-01", "2024-12-01", "chirps"),
            rainfall.cellset_key(grid, "2024-01-01", "2024-12-01", "chirps"))

    def test_the_reader_infers_the_product_from_the_callable(self):
        # A callable carries no name. Without this, a CHIRPS read through the
        # default key would collide with ERA5's.
        import inspect

        source = inspect.getsource(rainfall.read_cell_monthly)
        self.assertIn("chirps_cell_monthly", source,
                      "a bare reader with no product name must still select its "
                      "own cache slot")

    def test_the_two_products_read_from_different_rasters(self):
        import os
        import tempfile

        previous = os.environ.get("RAINFALL_CACHE_DIR")
        tmp = tempfile.TemporaryDirectory()
        os.environ["RAINFALL_CACHE_DIR"] = tmp.name
        try:
            geom = {"type": "Polygon", "coordinates": [
                [[37.70, 1.20], [37.80, 1.20], [37.80, 1.30],
                 [37.70, 1.30], [37.70, 1.20]]]}
            era5 = rainfall.build_and_cache(geom, start="2024-01-01", end="2024-03-01")
            chirps = rainfall.build_and_cache(
                geom, start="2024-01-01", end="2024-03-01",
                source=stub_reader, product="chirps")
            self.assertGreater(era5["resolution_km"], chirps["resolution_km"] * 4,
                               "one of them was read from the other's raster")
        finally:
            if previous is None:
                os.environ.pop("RAINFALL_CACHE_DIR", None)
            else:
                os.environ["RAINFALL_CACHE_DIR"] = previous
            tmp.cleanup()

    def test_a_chirps_series_carries_chirps_provenance_not_era5(self):
        # compute_series stamped ERA5's source/DOI/citation on every payload, so
        # a CHIRPS series carried the reanalysis's provenance under its own grid
        # and "chirps" product -- a correct number wearing the wrong statement of
        # where it came from.
        import os
        import tempfile

        previous = os.environ.get("RAINFALL_CACHE_DIR")
        tmp = tempfile.TemporaryDirectory()
        os.environ["RAINFALL_CACHE_DIR"] = tmp.name
        try:
            geom = {"type": "Polygon", "coordinates": [
                [[37.70, 1.20], [37.80, 1.20], [37.80, 1.30],
                 [37.70, 1.30], [37.70, 1.20]]]}
            chirps = rainfall.build_and_cache(
                geom, start="2024-01-01", end="2024-03-01",
                source=stub_reader, product="chirps")
            self.assertEqual(chirps["source"], rainfall.CHIRPS_SOURCE)
            self.assertEqual(chirps["doi"], rainfall.CHIRPS_DOI)
            self.assertEqual(chirps["product"], "chirps")
            era5 = rainfall.build_and_cache(geom, start="2024-01-01", end="2024-03-01")
            self.assertEqual(era5["source"], rainfall.ERA5_SOURCE)
            self.assertEqual(era5["doi"], rainfall.ERA5_DOI)
        finally:
            if previous is None:
                os.environ.pop("RAINFALL_CACHE_DIR", None)
            else:
                os.environ["RAINFALL_CACHE_DIR"] = previous
            tmp.cleanup()


class WindowServingTests(unittest.TestCase):
    """The window guides what is served, not only what is displayed.

    The stored series is wider than any one window (it is the union of the
    request and the fixed 1991-2020 climatology). These pin that narrowing the
    window narrows the series actually served -- the monthly rows, the summary,
    the coverage it reports and the window it names -- while the climatology
    reference is left whole.
    """

    def _payload(self):
        def row(month, precip, normal, anomaly, pct):
            return {"month": month, "precip_mm": precip, "normal_mm": normal,
                    "anomaly_mm": anomaly, "anomaly_pct": pct}

        return {
            "status": "ok", "product": "rainfall",
            "series": [
                row("2023-11-01", 10.0, 20.0, -10.0, -50.0),
                row("2023-12-01", 12.0, 20.0, -8.0, -40.0),
                row("2024-01-01", 20.0, 20.0, 0.0, 0.0),
                row("2024-06-01", 30.0, 20.0, 10.0, 50.0),
                row("2024-12-01", 40.0, 20.0, 20.0, 100.0),
            ],
            "coverage": {
                "window": {"months": 5, "unreadable_months": ["2023-12-01"]},
                "climatology": {"months": 360, "unreadable_months": []},
            },
            "unreadable_months": ["2023-12-01"],
            "window": {"start": "2023-11-01", "end": "2024-12-01"},
            "climatology": {"start": "1991-01-01", "end": "2020-12-31"},
        }

    def test_no_window_serves_the_whole_series_unchanged(self):
        self.assertEqual(
            rainfall._window_precomputed_payload(self._payload(), None, None),
            self._payload())

    def test_the_window_serves_only_its_own_months(self):
        out = rainfall._window_precomputed_payload(
            self._payload(), "2024-01-01", "2024-06-30")
        self.assertEqual([r["month"] for r in out["series"]],
                         ["2024-01-01", "2024-06-01"])
        self.assertEqual(out["window"]["start"], "2024-01-01")
        self.assertEqual(out["window"]["end"], "2024-06-30")
        # Coverage reports the served window, not the stored one.
        self.assertEqual(out["coverage"]["window"]["months"], 2)
        self.assertEqual(out["coverage"]["window"]["unreadable_months"], [])
        # An unreadable month outside the window is not claimed for it.
        self.assertEqual(out["unreadable_months"], [])
        # The climatology is a fixed reference and is left whole.
        self.assertEqual(out["coverage"]["climatology"]["months"], 360)
        self.assertEqual(out["climatology"],
                         {"start": "1991-01-01", "end": "2020-12-31"})

    def test_the_chirps_store_serves_the_months_it_holds_and_fills_the_tail(self):
        # The store used to answer a window only whole: any month outside its
        # range returned None and the entire series fell back to one GeoTIFF per
        # month. A window that reaches the present therefore rebuilt most of the
        # record from scratch. Now the store serves the months it holds and only
        # the publication-lag tail is read from GeoTIFFs.
        import numpy as np

        from unittest import mock

        grid = {"latitudes": [0.0], "longitudes": [0.0],
                "lat_step": 0.25, "lon_step": 0.25}

        store_calls = []
        geotiff_calls = []

        def from_store(g, start, end):
            store_calls.append((start, end))
            labels = rainfall._month_range(start, end)  # 2026-01 .. 2026-08
            matrix = np.full((len(labels), 1, 1), 50.0)
            return labels, matrix, rainfall._chirps_coverage(labels, [])

        def from_geotiffs(g, start, end):
            geotiff_calls.append((start, end))
            labels = rainfall._month_range(start, end)  # 2026-09 .. 2026-10
            matrix = np.full((len(labels), 1, 1), 60.0)
            return labels, matrix, rainfall._chirps_coverage(labels, [])

        with mock.patch.object(rainfall, "chirps_store_month_bounds",
                               return_value=("1981-01", "2026-08")), \
             mock.patch.object(rainfall, "_chirps_cell_monthly_from_store",
                               from_store), \
             mock.patch.object(rainfall, "_chirps_cell_monthly_from_geotiffs",
                               from_geotiffs):
            labels, matrix, coverage = rainfall.chirps_cell_monthly(
                grid, "2026-01-01", "2026-10-01")

        self.assertEqual(labels, rainfall._month_range("2026-01-01", "2026-10-01"))
        self.assertEqual(list(np.ravel(matrix)),
                         [50.0] * 8 + [60.0] * 2)
        # The store is read for exactly the months it holds, not the whole window
        # (which reaches past it), and the tail is the only GeoTIFF read.
        self.assertEqual(store_calls, [("2026-01-01", "2026-08-01")])
        self.assertEqual(geotiff_calls, [("2026-09-01", "2026-10-01")])
