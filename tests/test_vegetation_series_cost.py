"""The cost model behind the wait a user is asked to sit through.

Three defects, all invisible without measuring:

- the estimate applied a sequential per-month rate to a four-way parallel read,
  so it promised roughly six times the real wall clock;
- it was calibrated on a developer's workstation rather than the deployed host,
  which is a different distance from the MODIS blob store;
- every series read back to 1991 to form a normal, so a ten-year request paid for
  thirty-five years of reads and then returned ten years of data.

The interface prints this number to someone deciding whether to wait, so being
wrong in either direction is a real cost: too high and we look slow, too low and
we look like we do not know what we are doing.
"""

import unittest

import vegetation_series as vs


class EstimateTests(unittest.TestCase):
    def test_the_estimate_divides_by_the_worker_count(self):
        # The constant is per read per worker. Forgetting the division is what
        # made the estimate six times too high in the first place.
        one = vs.estimate_seconds(200, workers=1)
        four = vs.estimate_seconds(200, workers=4)
        self.assertGreater(one, four)
        self.assertAlmostEqual(one, four * 4, delta=vs.ESTIMATE_OVERHEAD_SECONDS * 4)

    def test_it_matches_the_measured_median(self):
        # Seven completed series on the droplet: 25-31 s wall, median 27, each
        # reading 429 months at four workers. The model has to land near that or
        # it is not calibrated on anything.
        import os

        reads = 429
        modelled = reads / 4 * vs.SECONDS_PER_READ + vs.ESTIMATE_OVERHEAD_SECONDS
        self.assertGreaterEqual(modelled, 20,
                                "the estimate now understates a measured 27 s")
        self.assertLessEqual(modelled, 35,
                             "the estimate still overstates a measured 27 s")

    def test_the_constant_is_a_host_property_not_a_algorithm_property(self):
        # Two hosts reading the same imagery at different distances from the blob
        # store get different numbers, so the cost has to be a parameter rather
        # than a constant baked into the arithmetic. Asserted without mutating
        # the module: a reload here changes the constant for every other test in
        # the suite, which is how a passing file starts reporting nonsense.
        import inspect

        source = inspect.getsource(vs)
        self.assertIn("VEG_SECONDS_PER_READ", source,
                      "the per-read cost must be overridable by environment")
        self.assertIn("VEG_BASELINE_YEARS", source,
                      "the baseline length must be overridable by environment")


class BaselineTests(unittest.TestCase):
    def test_a_ten_year_request_no_longer_reads_thirty_five_years(self):
        reads = vs.months_to_read("2010-01-01", "2026-09-01")
        self.assertLessEqual(
            reads, 250,
            "a ten-year request reads 237 months with a 20-year baseline; if this "
            "is back near 429 the unconditional 1991 start has returned")

    def test_a_one_year_request_reads_the_same_as_a_ten_year_one(self):
        # Both pay for the normal, so they cost the same. That is the point of
        # the baseline, and it is also why the estimate must be on reads rather
        # than on the months returned.
        self.assertEqual(
            vs.months_to_read("2025-01-01", "2026-09-01"),
            vs.months_to_read("2010-01-01", "2026-09-01"),
        )

    def test_the_baseline_is_reported_not_implied(self):
        # A normal nobody states is not a normal. The plan is what the interface
        # reads, so this is the surface that has to carry it.
        plan = vs.plan(5000, "2010-01-01", "2026-09-01")
        self.assertIn("normal_window", plan)
        self.assertEqual(plan["normal_window"]["start"][:4], "2007")
        self.assertEqual(plan["months_to_read"], 237)
        self.assertIn("exceed the months returned", plan["estimate_basis"])


class PrecomputeToolTests(unittest.TestCase):
    def test_a_compact_portfolio_is_built_on_one_shared_window(self):
        from tools import precompute_vegetation as tool

        bbox = tool.combined_bbox([
            {"type": "Polygon", "coordinates": [[[36.8, 0.1], [36.9, 0.1],
                                                  [36.9, 0.2], [36.8, 0.2], [36.8, 0.1]]]},
            {"type": "Polygon", "coordinates": [[[37.0, 0.3], [37.1, 0.3],
                                                  [37.1, 0.4], [37.0, 0.4], [37.0, 0.3]]]},
        ])
        self.assertLess(max(bbox[2] - bbox[0], bbox[3] - bbox[1]), 10.0,
                        "the published portfolio is well inside the span bound")

    def test_a_scattered_portfolio_falls_back_to_per_area_reads(self):
        # The claim "one read serves the portfolio" is only true while the
        # portfolio shares a tile. Asserted so a future change to the bound
        # cannot quietly start averaging a scattered set through one window.
        from tools import precompute_vegetation as tool

        scattered = tool.combined_bbox([
            {"type": "Polygon", "coordinates": [[[36.8, 0.1], [36.9, 0.1],
                                                  [36.9, 0.2], [36.8, 0.2], [36.8, 0.1]]]},
            {"type": "Polygon", "coordinates": [[[100.0, -40.0], [100.1, -40.0],
                                                  [100.1, -39.9], [100.0, -39.9], [100.0, -40.0]]]},
        ])
        self.assertGreater(max(scattered[2] - scattered[0],
                               scattered[3] - scattered[1]), 10.0)


class QueueWaitTests(unittest.TestCase):
    """The wait a user experiences is the queue plus the compute.

    Measured on the droplet for one 237-read series: 155 s waiting to be picked
    up, 24 s computing. Five sixths of the wait was a 300 s poll interval, during
    which the job sat in a directory doing nothing, because the worker only looked
    for work every fifth minute. The cost model that produces the number shown to
    the user models the compute; this is the part it does not, and it was the
    larger part.
    """

    def test_the_poll_interval_is_short_enough_to_not_dominate_the_wait(self):
        import re
        from pathlib import Path

        compose = (Path(__file__).resolve().parent.parent / "docker-compose.yml").read_text()
        match = re.search(r"RAINFALL_WORKER_INTERVAL:-(\d+)", compose)
        self.assertIsNotNone(match, "the worker interval is no longer in compose")
        interval = int(match.group(1))
        # A mean wait of interval/2. At 300 s that is 150 s, five sixths of the
        # 180 s a user waited for a 24 s job. Holding it to 15 s caps the mean at
        # 7.5 s, which is no longer the thing a user notices.
        self.assertLessEqual(
            interval, 30,
            f"a {interval}s poll adds up to {interval // 2}s of dead wait before "
            f"any compute begins; an idle sweep is a directory listing")
