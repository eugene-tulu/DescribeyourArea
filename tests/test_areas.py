"""Resolving an area, against responses the real service actually returned.

The fixtures in ``tests/fixtures`` are captured output from the live boundary
service, not shapes written by hand for this suite, and the fake resolver
subclasses the real client so its parameter construction is exercised rather than
bypassed. Parsing something the service will never send is how a resolver passes
its tests and fails in production.

The last class in this file talks to the real service and is skipped unless
``GAUL_LIVE=1``, because a unit suite that depends on another project's uptime
is the same mistake as the live-data flake in ``test_contract``.
"""

import json
import os
import unittest
from pathlib import Path

import areas
from areas import GaulResolver, ResolverError, ResolverUnavailable, resolve_area

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class RecordedResolver(GaulResolver):
    """The real client, answering from recorded responses.

    Subclassing rather than standing in for it is the point: ``by_id``,
    ``by_name`` and ``containing`` are exercised, including their error branches,
    which a hand-written stub would not have.
    """

    def __init__(self, fixtures: Path = FIXTURES):
        self.fixtures = fixtures
        self.requests: list[tuple[str, dict]] = []

    def _fixture_for(self, path: str, params: dict) -> str:
        if path == "boundary-index":
            return "index"
        if path == "containing":
            return "point"
        if params.get("ids"):
            return "admin1" if int(params.get("level", 0)) == 1 else "admin0"
        return "admin1"

    def get(self, path, params):
        self.requests.append((path, dict(params)))
        return json.loads((self.fixtures / f"gaul_{self._fixture_for(path, params)}.json").read_text())


class RealResponsesTests(unittest.TestCase):
    """Each mode, against a response the live service produced."""

    def setUp(self):
        self.resolver = RecordedResolver()

    def test_an_area_named_by_country_and_level(self):
        area = resolve_area(self.resolver, country="137", level=1, admin1="Narok")
        self.assertEqual(area.name, "Narok")
        self.assertEqual(area.level, 1)
        self.assertEqual(area.id, "137:1:1385")
        self.assertEqual(area.source, "name")
        self.assertGreater(area.area_km2, 30_000,
                           "Narok county is about 35,000 km2; a much smaller "
                           "figure means the geometry was truncated")

    def test_an_area_named_by_administrative_id(self):
        # The handoff from globe arrives this way: a canonical id and nothing else.
        area = resolve_area(self.resolver, id="137:1:1385")
        self.assertEqual(area.name, "Narok")
        self.assertEqual(area.id, "137:1:1385")
        self.assertEqual(area.source, "admin_id")

    def test_the_id_carries_its_own_level_so_the_caller_need_not_repeat_it(self):
        # Requesting an id at the wrong level is a 422 from the service whose
        # message names the right level. Reading it off the id avoids asking the
        # caller to know a convention they have no reason to know.
        resolve_area(self.resolver, id="137:1:1385")
        self.assertEqual(self.resolver.requests[0][1]["level"], 1)

    def test_a_point_resolves_to_the_finest_unit_containing_it(self):
        # The ranger's case: no file, no name, just where they are.
        area = resolve_area(self.resolver, lat=-0.5, lon=37.2)
        self.assertEqual(area.level, 2, "a point should land on a district, "
                                         "not a county, unless nothing finer exists")
        self.assertEqual(area.source, "point")
        self.assertTrue(area.notes, "a point-resolved area must say the outline is "
                                    "the whole unit and not something they drew")

    def test_a_bounding_box_becomes_a_rectangle_that_says_so(self):
        area = resolve_area(self.resolver, bbox="36.7,0.2,36.9,0.4")
        self.assertAlmostEqual(area.area_km2, 440, delta=60)
        self.assertEqual(area.source, "bbox")
        self.assertTrue(any("rectangle" in n for n in area.notes),
                        "a bbox is a rectangle, not the area the caller had in "
                        "mind, and its figures are the rectangle's")

    def test_every_mode_produces_the_same_shape(self):
        # The resolver is a capability, so its output is one shape. A client that
        # renders one renders all four, which is what makes the area picker a
        # single code path rather than four.
        shapes = set()
        for kwargs in (dict(country="137", level=1, admin1="Narok"),
                       dict(id="137:1:1385"),
                       dict(lat=-0.5, lon=37.2),
                       dict(bbox="36.7,0.2,36.9,0.4")):
            shapes.add(tuple(sorted(resolve_area(self.resolver, **kwargs).describe())))
        self.assertEqual(len(shapes), 1, "the four modes return different shapes")


class AdmissionTests(unittest.TestCase):
    def setUp(self):
        self.resolver = RecordedResolver()

    def test_two_ways_of_naming_an_area_is_refused_rather_than_guessed(self):
        # Silently preferring one of two is how someone ends up analysing a place
        # other than where they think, and never finds out.
        with self.assertRaises(ResolverError) as caught:
            resolve_area(self.resolver, id="137:1:1385", lat=-0.5, lon=37.2)
        self.assertIn("one way", str(caught.exception))

    def test_naming_nothing_is_refused(self):
        with self.assertRaises(ResolverError):
            resolve_area(self.resolver)

    def test_a_malformed_id_says_what_one_looks_like(self):
        with self.assertRaises(ResolverError) as caught:
            resolve_area(self.resolver, id="nonsense")
        self.assertIn("137:1:1385", str(caught.exception))

    def test_a_malformed_bbox_is_refused_with_the_format(self):
        with self.assertRaises(ResolverError) as caught:
            resolve_area(self.resolver, bbox="west,south,east,north")
        self.assertIn("minLon", str(caught.exception))


class DegradationTests(unittest.TestCase):
    """A service that is down is not a service that said no.

    The distinction decides what a user does next: retry, or fix their input.
    Collapsing both into a 500 is why "the map is empty" and "your county name is
    misspelled" produce the same shrug.
    """

    class Down(GaulResolver):
        def get(self, path, params):
            raise ResolverUnavailable("the boundary service is not answering (refused)")

    class NoMatch(GaulResolver):
        def get(self, path, params):
            return {"features": [], "items": [], "boundaries": {}}

    def test_an_unreachable_service_is_distinguishable_from_a_rejection(self):
        with self.assertRaises(ResolverUnavailable):
            resolve_area(self.Down(), id="137:1:1385")

    def test_a_name_that_matches_nothing_is_a_rejection_with_a_reason(self):
        with self.assertRaises(ResolverError) as caught:
            resolve_area(self.NoMatch(), country="137", level=1, admin1="Atlantis")
        self.assertEqual(caught.exception.reason, "not_found")
        self.assertIn("Atlantis", str(caught.exception))

    def test_a_point_outside_every_boundary_says_so(self):
        with self.assertRaises(ResolverError) as caught:
            resolve_area(self.NoMatch(), lat=-89.0, lon=0.0)
        self.assertIn("contains", str(caught.exception))

    def test_a_refusal_from_the_service_carries_the_services_own_message(self):
        # The service explains that ids must be requested at their own level. That
        # is better than anything we could invent, so it is passed through.
        class Refusing(GaulResolver):
            def get(self, path, params):
                raise ResolverError(
                    "Boundary id '137:1:1385' is level 1; this request requires "
                    "level 2 ids.", reason="rejected")

        with self.assertRaises(ResolverError) as caught:
            resolve_area(Refusing(), id="137:1:1385", level=2)
        self.assertIn("requires level 2", str(caught.exception))


class RouteTests(unittest.TestCase):
    def test_the_route_exists_and_is_published(self):
        import main

        paths = {route.path for route in main.app.routes}
        self.assertIn("/areas/resolve", paths)

    def test_it_is_a_get_because_it_reads(self):
        import main

        methods = {method
                   for route in main.app.routes if route.path == "/areas/resolve"
                   for method in getattr(route, "methods", set())}
        self.assertIn("GET", methods)


@unittest.skipUnless(os.getenv("GAUL_LIVE") == "1",
                     "set GAUL_LIVE=1 to exercise the real boundary service")
class LiveServiceTests(unittest.TestCase):
    """The recorded fixtures could be stale. This is the check that they are not."""

    def test_the_live_service_still_answers_the_recorded_query(self):
        resolver = GaulResolver(os.getenv("GAUL_API_URL", "http://127.0.0.1:8002"))
        area = resolve_area(resolver, country="137", level=1, admin1="Narok")
        self.assertEqual(area.name, "Narok")
        self.assertEqual(area.id, "137:1:1385")

    def test_the_live_id_round_trip_agrees_with_the_fixture(self):
        # The id path is what the globe handoff depends on, and it is the one
        # that looked broken during an earlier investigation.
        resolver = GaulResolver(os.getenv("GAUL_API_URL", "http://127.0.0.1:8002"))
        area = resolve_area(resolver, id="137:1:1385")
        self.assertEqual(area.name, "Narok")
