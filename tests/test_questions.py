"""The dates are the question; everything else is a consequence of them.

The properties here are all things the old preset control could not express. If
one of them stops being true, the control has crept back in somewhere and the
historical questions have gone with it.

Sensor coverage is checked against the real archive dates, and the routing
decisions are checked against the real registry, so a product added later changes
the expected answers here rather than silently not being accounted for.
"""

import datetime
import unittest

import questions
import registry
import sensors
from questions import Window, parse_window, plan_comparison, plan_question

# A fixed "today" so a test never fails because the clock moved. The windows below
# are absolute dates precisely so they do not need one.
END = "2026-09-01"


class DatesArePrimaryTests(unittest.TestCase):
    def test_two_dates_make_a_window(self):
        window = parse_window(start="1997-01-01", end="1998-12-01")
        self.assertEqual(window.start, datetime.date(1997, 1, 1))
        self.assertEqual(window.end, datetime.date(1998, 12, 1))
        self.assertEqual(window.months, 24)

    def test_a_preset_is_resolved_to_dates_immediately(self):
        # Presets stay available because the control offers them, but they are
        # converted at the edge so nothing downstream can depend on them.
        self.assertEqual(parse_window(years=10, end=END).months, 121)

    def test_no_dates_at_all_still_yields_a_window(self):
        self.assertIsNotNone(parse_window(end=END).months)

    def test_a_reversed_range_is_refused(self):
        with self.assertRaises(ValueError):
            parse_window(start="2020-01-01", end="2019-01-01")

    def test_a_nonsense_date_says_what_it_wants(self):
        with self.assertRaises(ValueError) as caught:
            parse_window(start="last tuesday", end=END)
        self.assertIn("2019-01-01", str(caught.exception))

    def test_a_zero_year_window_is_refused(self):
        with self.assertRaises(ValueError):
            parse_window(years=0, end=END)


class LabelTests(unittest.TestCase):
    """Labels were wrong for historical windows and this is how that stays fixed.

    An earlier version returned "the past year" for any window between one and two
    years, which is false for a 1997 window. The history questions are the reason
    dates are primary, so a label that is only correct for windows ending now
    undoes the whole change for exactly the case it was made for.
    """

    def test_a_label_is_relative_to_its_own_end_not_to_today(self):
        historical = parse_window(start="1997-01-01", end="1998-12-01")
        self.assertNotIn("past", historical.label)
        self.assertIn("1998-12", historical.label)

    def test_a_single_day_is_named_as_a_month_not_zero_days(self):
        window = parse_window(start="2025-04-01", end="2025-04-01")
        self.assertEqual(window.days, 0)
        self.assertEqual(window.label, "the month of 2025-04")

    def test_a_short_window_is_in_days_and_a_long_one_in_years(self):
        self.assertIn("days", parse_window(start="2026-08-01", end=END).label)
        self.assertIn("years", parse_window(years=10, end=END).label)


class DerivedTests(unittest.TestCase):
    """Each derivation is a stated consequence of the dates, not a choice."""

    def test_bin_width_follows_the_span(self):
        # 390 monthly bars in a readable chart is not the same claim as 32
        # annual ones, and which you get should follow from the dates.
        short = parse_window(years=8, end=END)          # 97 months
        long = parse_window(years=30, end=END)          # 361 months
        self.assertLess(short.months, questions.MONTHLY_BIN_LIMIT)
        self.assertGreater(long.months, questions.MONTHLY_BIN_LIMIT)
        self.assertEqual(short.bin_months, 1)
        self.assertEqual(long.bin_months, 12)

    def test_the_bin_boundary_is_where_it_is_said_to_be(self):
        # An earlier version of this test used a 141-month window and called it
        # short. The threshold was right and the test was wrong, which is the
        # failure this pins: check the boundary, not a comfortable example.
        # Exact dates, because the boundary is not reachable from a whole-year
        # preset: the smallest window over the limit is 10 years, which is 121
        # months, so 120 months sits between the 9 and 10 year presets.
        at_limit = parse_window(start="2016-10-01", end=END)
        self.assertEqual(at_limit.months, questions.MONTHLY_BIN_LIMIT)
        self.assertEqual(at_limit.bin_months, 1)
        over = parse_window(start="2016-09-01", end=END)
        self.assertEqual(over.months, questions.MONTHLY_BIN_LIMIT + 1)
        self.assertEqual(over.bin_months, 12)

    def test_the_client_default_window_already_bins_yearly(self):
        # Found while testing the boundary above: the interface's default is 10
        # years, which is 121 months, which is over the limit. So the default
        # chart has always been yearly. Pinned so that if the limit moves, someone
        # learns what else moves with it.
        default_window = parse_window(years=10, end=END)
        self.assertGreater(default_window.months, questions.MONTHLY_BIN_LIMIT)
        self.assertEqual(default_window.bin_months, 12)

    def test_a_short_window_is_composited_and_a_long_one_binned(self):
        # One month is a single composite; three years is a record. The threshold
        # is a rendering decision and lives in one named place.
        self.assertTrue(parse_window(start="2026-08-01", end=END).is_recent_enough_to_composite)
        self.assertFalse(parse_window(start="2019-01-01", end=END).is_recent_enough_to_composite)

    def test_the_normal_period_is_published_for_rainfall(self):
        import rainfall

        window = parse_window(start="2010-01-01", end=END)
        normal = questions.normal_window_for("rainfall", window)
        self.assertEqual(normal["start"], rainfall.CLIMATOLOGY_START)
        self.assertEqual(normal["end"], rainfall.CLIMATOLOGY_END)

    def test_the_vegetation_normal_starts_where_modis_starts(self):
        # A 1991-2020 normal beside a MOD13Q1 series is a claim about data that
        # does not exist, so the vegetation normal is derived instead.
        window = parse_window(start="2010-01-01", end=END)
        normal = questions.normal_window_for("vegetation_series", window)
        self.assertGreaterEqual(normal["start"], "2000-01-01")
        self.assertIn("2000-02", normal["basis"])

    def test_sensor_coverage_comes_from_the_real_archive_dates(self):
        # Checked against the registry rather than a copy of it, so a sensor added
        # or re-dated changes this rather than going unnoticed.
        window = parse_window(start="1994-01-01", end="1995-12-01")
        covering = questions.sensors_covering(window)
        self.assertIn("landsat", covering)
        self.assertNotIn("sentinel-2", covering,
                         "Sentinel-2 begins in 2015 and cannot answer a 1994 window")
        for sid in covering:
            self.assertTrue(window.start.isoformat() >= sensors.SENSORS[sid].archive_start)


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.window = parse_window(start="2010-01-01", end=END)

    def test_history_is_computed_because_no_history_is_ever_a_live_read(self):
        plan = plan_question("history", self.window)
        self.assertEqual(plan.routing, "computed")
        for key in plan.products:
            self.assertEqual(registry.get(key).costs(), (registry.COMPUTED,))

    def test_watch_is_computed_because_it_reads_the_record(self):
        self.assertEqual(plan_question("watch", self.window).routing, "computed")

    def test_describe_is_mixed_because_vegetation_depends_on_the_area(self):
        # dem and land cover answer live; ndvi is also queueable for a large area.
        # The plan says so rather than claiming a single route it cannot keep.
        plan = plan_question("describe", self.window)
        self.assertEqual(plan.routing, "mixed")
        self.assertTrue(any(n for n in plan.notes if "need the worker" in n))

    def test_routing_is_derived_from_the_registry_not_a_parallel_table(self):
        for key in ("rainfall", "vegetation_series"):
            self.assertNotIn(registry.LIVE, registry.get(key).costs())
        for key in ("dem", "landcover"):
            self.assertIn(registry.LIVE, registry.get(key).costs())

    def test_an_unanswerable_window_says_so_rather_than_returning_nothing(self):
        window = parse_window(start="1975-01-01", end="1975-12-01")
        plan = plan_question("history", window, products=("ndvi",))
        self.assertEqual(plan.sensors, ())
        self.assertTrue(any("No sensor archive reaches back to 1975-01" in n
                            for n in plan.notes),
                        "a 1975 vegetation request must explain itself rather "
                        "than return an empty series")

    def test_a_yearly_bin_is_explained(self):
        plan = plan_question("history", parse_window(years=30, end=END),
                             products=("rainfall",))
        self.assertTrue(any("binned yearly" in n for n in plan.notes))

    def test_an_unknown_question_is_refused(self):
        with self.assertRaises(ValueError):
            plan_question("forecast", self.window)

    def test_an_unregistered_product_is_refused(self):
        with self.assertRaises(ValueError):
            plan_question("describe", self.window, products=("smoke-signals",))


class FourQuestionsTests(unittest.TestCase):
    """Acceptance test two: four questions, and a persona per question.

    The planner needs `compare`, the researcher `history`, the conservancy manager
    `watch`, and the ranger `describe`. None of those needs a route of its own --
    they differ by which products and which dates, which is what makes this a
    layer rather than four endpoints that happen to share a name.
    """

    def test_the_question_set_is_the_one_the_roadmap_committed_to(self):
        self.assertEqual(set(questions.QUESTIONS),
                         {"describe", "compare", "history", "watch"})

    def test_every_question_declares_what_it_needs(self):
        for question in questions.QUESTIONS:
            with self.subTest(question=question):
                self.assertTrue(questions.QUESTION_NEEDS[question])
                for key in questions.QUESTION_NEEDS[question]:
                    self.assertIsNotNone(registry.get(key))

    def test_a_plan_describes_itself_completely(self):
        import json

        described = plan_question("history", parse_window(years=30, end=END)).describe()
        json.dumps(described)
        self.assertEqual(described["routing"], "computed")
        self.assertIn("rainfall", described["normals"])
        self.assertEqual(described["window"]["bin_months"], 12)


class ComparisonTests(unittest.TestCase):
    """The planner's question, which the layer previously only claimed to serve.

    "Which of my wards is worst" is not N independent numbers. For a product
    coarser than the outlines, two nearby areas can resolve to the same grid
    cell and produce the identical figure, and a ranking between them is noise
    dressed as a finding. So the plan carries the cell count, which is what lets a
    caller notice.
    """

    def test_it_carries_every_area_it_was_given(self):
        window = parse_window(years=1, end=END)
        plan = plan_comparison(
            [{"id": "137:1:1", "name": "Narok", "area_km2": 35_694},
             {"id": "137:1:2", "name": "Ndia", "area_km2": 1_540}], window)
        self.assertEqual([a["name"] for a in plan["areas"]], ["Narok", "Ndia"])

    def test_it_says_two_areas_in_one_cell_are_one_measurement(self):
        plan = plan_comparison([{"name": "a"}, {"name": "b"}],
                               parse_window(years=1, end=END))
        joined = " ".join(plan["notes"])
        self.assertIn("774 km2 cell", joined)
        self.assertIn("identical number", joined)

    def test_it_separates_products_that_rank_places_from_products_that_rank_times(self):
        plan = plan_comparison([{"name": "a"}], parse_window(years=1, end=END))
        joined = " ".join(plan["notes"])
        self.assertIn("single-date", joined)

    def test_it_refuses_more_areas_than_one_comparison_carries(self):
        # Twenty synchronous raster reads is a different service from one, and
        # the limit belongs to the question rather than to the route.
        with self.assertRaises(ValueError) as caught:
            plan_comparison([{"name": f"a{i}"} for i in range(
                questions.MAX_COMPARE_AREAS + 1)],
                parse_window(years=1, end=END))
        self.assertIn("batches", str(caught.exception))

    def test_it_refuses_an_empty_comparison(self):
        with self.assertRaises(ValueError):
            plan_comparison([], parse_window(years=1, end=END))

    def test_an_unnamed_area_is_still_identifiable(self):
        plan = plan_comparison([{"id": "137:1:9"}], parse_window(years=1, end=END))
        self.assertTrue(plan["areas"][0]["name"])
