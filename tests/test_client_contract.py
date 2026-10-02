"""The seam between the client and the backend, tested.

Four separate defects shipped in this seam before anything checked it: a GET
against a POST-only route, an `indicator` sent as a query parameter the server
read from the body, a query parameter the route never declared, and a
write-only share link. Each side was internally consistent and fully tested.
Nothing compared them.

The client is TypeScript and the server is Python, so there is no shared type to
drift. The only thing both sides agree on is a string in a ``fetch`` call, and
that is exactly what this reads. It is a text scan on purpose: it is the only
check that can run in the backend suite without a Node toolchain, and it fails
loudly the first time someone calls a route that does not exist.

A new backend route is not a failure here. A new *client* call to a missing
route is, and that is the direction that has broken.
"""

import re
import unittest
from pathlib import Path

import main

CLIENT_ROOT = Path(__file__).resolve().parent.parent / "client"

# The backend image does not carry the client, so these cannot run inside the
# container. Skipping says so plainly; erroring 11 times on a server that
# correctly has no frontend would read like a real regression.
CLIENT_PRESENT = CLIENT_ROOT.is_dir()
requires_client = unittest.skipUnless(
    CLIENT_PRESENT,
    "client sources are not present (the backend image ships no frontend)",
)

# Calls are written as `${backendUrl}/path` or a literal. Both are template
# literals in practice, so one pattern covers them; the optional method and the
# query string are stripped because the server's route table has neither.
CALL = re.compile(
    r"fetch\(\s*`?\$\{backendUrl\}(?P<path>/[A-Za-z0-9_/-]*)(?:\?[^`'\"]*)?[`']?"
    r"(?:\s*,\s*\{\s*method:\s*'(?P<method>[A-Z]+)')?",
    re.S,
)

# The client is allowed to construct URLs that are not routes, and these are the
# known ones. Anything added here is a decision, not a convenience.
ALLOWED = {
    # The Vercel analytics beacon is a third-party endpoint.
}


def client_calls() -> list[tuple[str, str, str]]:
    """Every backend route the browser asks for, as (method, path, file)."""
    calls = []
    for path in sorted(CLIENT_ROOT.rglob("*.ts*")):
        if "node_modules" in path.parts or ".next" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        for match in CALL.finditer(text):
            route = match.group("path").rstrip("/") or "/"
            method = (match.group("method") or "GET").upper()
            calls.append((method, route, path.relative_to(CLIENT_ROOT.parent).as_posix()))
    return calls


def server_routes() -> set[tuple[str, str]]:
    return {
        (method, route.path.rstrip("/") or "/")
        for route in main.app.routes
        for method in getattr(route, "methods", set()) or set()
        if method not in ("HEAD", "OPTIONS")
    }


@requires_client
class ClientServerContractTests(unittest.TestCase):
    def test_the_client_calls_something_the_server_serves(self):
        calls = client_calls()
        self.assertTrue(
            calls,
            "the scan found no fetch calls, so it has stopped matching the "
            "client's style and is passing vacuously -- fix the pattern first",
        )
        self.assertGreaterEqual(len(calls), 4,
                                "expected several calls; a drop means the scan broke")

    def test_every_client_call_resolves_to_a_server_route(self):
        routes = server_routes()
        missing = sorted({
            f"{method} {route}  ({where})"
            for method, route, where in client_calls()
            if (method, route) not in routes and route not in ALLOWED
        })
        self.assertEqual(
            missing, [],
            "the client calls routes the server does not serve:\n  "
            + "\n  ".join(missing)
            + "\n\nThis has shipped three times. Either add the route or change "
              "the call; do not silence this by editing ALLOWED.",
        )

    def test_the_indicator_reaches_the_plan_in_the_body(self):
        """The regression that hid three of the four modules.

        ``indicator`` is part of the request's meaning, so it travels in the
        body. Sent as a query parameter the server silently fell back to
        planning rainfall alone and the user was never offered the other three.
        """
        source = (CLIENT_ROOT / "app" / "page.tsx").read_text(encoding="utf-8")
        plan_call = re.search(
            r"rainfall/plan\?\$\{query[^`]*`[^)]*?body:\s*JSON\.stringify\(\{(?P<body>[^}]*)\}",
            source,
            re.S,
        )
        self.assertIsNotNone(plan_call, "the plan call changed shape; update this test")
        self.assertIn("indicator", plan_call.group("body"),
                      "the plan call must carry the indicator in its body")

    def test_the_status_call_names_its_indicator(self):
        """Otherwise a queued module reports another module's state."""
        source = (CLIENT_ROOT / "app" / "page.tsx").read_text(encoding="utf-8")
        status = re.search(r"rainfall/status\?[^`]*`", source, re.S)
        self.assertIsNotNone(status, "the status call changed shape; update this test")
        self.assertIn("indicator", status.group(0),
                      "the status call must say which module it is asking about")

    def test_the_share_link_is_actually_read_back(self):
        """A link that encodes state nothing decodes is worse than no link.

        It copies cleanly, it looks shareable, and it opens a blank page. The
        cost of shipping one is a colleague who quietly stops trusting the tool.

        Asserted on substance rather than idiom: whatever keys the link writes,
        each one has to be read back. A test that demanded ``useSearchParams``
        would fail a correct ``useEffect`` implementation, and passing for the
        wrong reason is worse than failing.
        """
        source = (CLIENT_ROOT / "app" / "page.tsx").read_text(encoding="utf-8")
        written = set(re.findall(r"URLSearchParams\(\{([^}]*)\}", source, re.S))
        keys = set()
        for block in written:
            keys.update(re.findall(r"(\w+):", block))
        keys.discard("area")  # written across the two blocks below
        keys.update(re.findall(r"(\w+):", "".join(re.findall(
            r"URLSearchParams\(\{(.*?)\}\)", source, re.S))))
        # Dates, not a preset count: the link carries the same window the reader
        # was looking at, which is what makes it reproduce the result rather than
        # approximately the result. `admin` is the handoff from another system.
        for expected in ("area", "datasets", "sensor", "window_start", "window_end"):
            self.assertIn(expected, keys,
                          f"the share link no longer encodes {expected!r}: {keys}")
        for key in ("area", "datasets", "sensor", "window_start", "window_end"):
            self.assertIn(
                f"get('{key}')", source,
                f"the share link encodes {key!r} and nothing reads it back, so "
                f"the link restores nothing for that field",
            )
        self.assertTrue(
            "setUploadedGeojson" in source.split("useEffect")[-1],
            "the restored area has to reach the state that the map and the "
            "analysis both read",
        )


class RouteBehaviourTests(unittest.TestCase):
    """The three shipped breaks, asserted at the transport the client uses."""

    def setUp(self):
        import os
        import tempfile

        from fastapi.testclient import TestClient

        self.previous = {k: os.environ.get(k) for k in ("RAINFALL_CACHE_DIR",)}
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["RAINFALL_CACHE_DIR"] = self.tmp.name
        self.client = TestClient(main.app)
        self.addCleanup(self._restore)
        self.addCleanup(self.tmp.cleanup)

    def _restore(self):
        import os

        for key, value in self.previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    KEY = "a" * 32

    def test_get_rainfall_answers_a_cache_key(self):
        # The client has always asked with GET. The route was POST-only, so every
        # vegetation fetch returned 405 and the chart could never render.
        response = self.client.get(f"/rainfall?cache_key={self.KEY}")
        self.assertEqual(response.status_code, 200, response.text[:200])
        body = response.json()
        self.assertEqual(body["cache_key"], self.KEY)
        self.assertEqual(body["rainfall"]["status"], "not_computed")

    def test_get_rainfall_still_refuses_a_malformed_key(self):
        # Registering the route must not have widened it: this is the one input
        # that reaches a filesystem path.
        for bad in ("nope", "A" * 32, "a" * 31, "../../etc/passwd"):
            response = self.client.get(f"/rainfall?cache_key={bad}")
            self.assertEqual(response.status_code, 422, f"{bad!r} was accepted")

    def test_get_and_post_agree_for_the_same_key(self):
        # Two routes for one lookup is a drift risk, so they share a helper; this
        # is the check that keeps them sharing it.
        get = self.client.get(f"/rainfall?cache_key={self.KEY}").json()
        post = self.client.post("/rainfall", json={"cache_key": self.KEY}).json()
        self.assertEqual(get["rainfall"], post["rainfall"])

    def test_status_reports_the_module_asked_about(self):
        import jobs

        key = "b" * 32
        jobs.write_job({
            "cache_key": key, "indicator": "vegetation_series",
            "state": "running", "submitted_at": "2026-09-28T00:00:00+00:00",
        })
        try:
            response = self.client.get(
                f"/rainfall/status?cache_key={key}&indicator=vegetation_series")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(
                response.json()["submission"]["indicator"], "vegetation_series")
            # And the rainfall job for the same area is a different question.
            other = self.client.get(
                f"/rainfall/status?cache_key={key}&indicator=rainfall").json()
            self.assertNotEqual(other["submission"].get("state"), "running")
        finally:
            jobs.drop(key, "vegetation_series")

    def test_status_still_defaults_to_rainfall(self):
        response = self.client.get(f"/rainfall/status?cache_key={self.KEY}")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["submission"].get("indicator", "rainfall"), "rainfall")


@requires_client
class TemporalScopeTests(unittest.TestCase):
    """Which datasets the analysis window actually governs.

    The window sat above all four cards under the bare word "Window", so
    selecting "1y" read as a claim about all of them. It is a claim about one.
    Three of the four datasets are structurally different in time, and the
    interface said nothing about any of it:

    - vegetation: the window selects the composite period, in full;
    - precipitation: the window sets how many monthly reads are needed, which
      sets the sampling resolution, but the series returned is always the whole
      record and the fitted trend spans the whole record;
    - elevation and land cover: no time dimension at all, so no window can
      change them and pretending otherwise would be a lie about the data.

    These are statements about the interface's honesty, which is checkable, and
    about the backend, which is the load-bearing half.
    """

    def test_the_window_control_says_which_datasets_it_moves(self):
        # The label became "Window" when the date fields moved above the presets,
        # because naming one dataset on a control the dates now drive would
        # misdescribe it. The scope moved to the paragraph beneath instead, which
        # is where a reader goes to learn what the control does.
        flat = " ".join(
            (CLIENT_ROOT / "app" / "page.tsx").read_text(encoding="utf-8").split())
        self.assertIn("This window selects the vegetation period", flat,
                      "the control must still say which datasets it moves")
        self.assertIn("Elevation and land cover have no time dimension", flat)

    def test_the_window_control_says_what_it_does_not_govern(self):
        # JSX wraps prose across lines, so the copy is read with whitespace
        # collapsed or every sentence fails to match itself.
        source = " ".join(
            (CLIENT_ROOT / "app" / "page.tsx").read_text(encoding="utf-8").split())
        # The claim, not the wording. A re-skin reworded "record" to "series" and
        # this failed on the word, which is a test guarding copy edits rather
        # than the thing that matters: that the control still tells the reader
        # which of the other three datasets the window does and does not move.
        for phrase in ("no time dimension", "monthly series", "vegetation period"):
            self.assertIn(phrase, source,
                          f"the window control must still say {phrase!r} rather "
                          f"than letting a 1y selection imply all four cards changed")

    def test_the_fitted_trend_names_its_own_span(self):
        source = (CLIENT_ROOT / "app" / "page.tsx").read_text(encoding="utf-8")
        self.assertIn("trend per year (", source,
                      "the slope is fitted over the whole record, so the label "
                      "has to say so next to a window control that implies otherwise")

    def test_vegetation_is_the_only_dataset_the_window_fully_governs(self):
        """The backend half: the window reaches the vegetation search."""
        source = (Path(__file__).resolve().parent.parent / "main.py").read_text(
            encoding="utf-8")
        start = source.index("async def compute_vegetation_index(")
        vegetation = source[start:source.index("\n@app", start)]
        self.assertIn("window_start", vegetation)
        self.assertIn("_search_items(bbox, sensor, bounded_scenes, window_start, window_end)",
                      vegetation,
                      "vegetation must search inside the window, or the control "
                      "governs nothing at all")

    def test_rainfall_ignores_the_window_when_choosing_its_series(self):
        """A series is a record, not a window, and the client now says so.

        The window does reach the rainfall routes, but it chooses a resolution
        rather than a date range. If someone later makes it slice the series,
        this test is the place to change the claim deliberately.
        """
        import rainfall

        self.assertNotIn("window", rainfall.describe.__code__.co_varnames,
                         "the rainfall summary is computed over the whole "
                         "record; if it has become window-aware, update the "
                         "client copy that says so")


class UserFacingSurfaceTests(unittest.TestCase):
    """The things a first-time user needs, asserted so they cannot vanish.

    Each of these was a gap found by auditing what a stranger can do, and each is
    a single line of code that would fail silently if deleted -- a button that
    renders nothing, a message dispatched to a store nobody watches.
    """

    @property
    def source(self) -> str:
        return (CLIENT_ROOT / "app" / "page.tsx").read_text(encoding="utf-8")

    def test_the_toaster_is_mounted(self):
        # useToast dispatched into a store that nothing rendered, so a rejected
        # upload gave no indication at all.
        layout = (CLIENT_ROOT / "app" / "layout.tsx").read_text(encoding="utf-8")
        self.assertIn("<Toaster />", layout,
                      "toasts are dispatched but never rendered")
        self.assertTrue((CLIENT_ROOT / "components" / "Toaster.tsx").exists())

    def test_a_user_can_remove_their_own_area(self):
        # We store a series derived from their geometry and had no way to let
        # them take it back, which contradicts the privacy claim.
        self.assertIn("rainfall/forget", self.source,
                      "no user-facing route to withdraw an area's data")
        self.assertIn("forgetThisArea", self.source,
                      "the route exists with no way to reach it")
        self.assertNotIn(
            "require_admin_key", _handler_source("forget_my_area"),
            "the withdrawal route now requires the admin secret, which a browser "
            "cannot hold, so the feature is dead in the only place it is used")

    def test_the_numbers_can_be_exported(self):
        # The capability, not the button wording. A re-skin shortened
        # "Download CSV" to "CSV" and this failed on the word.
        for needle, why in (
            ("rainfall-series.csv", "a table of the monthly figures is the likely real need"),
            ("analysis.json", "so a result can be re-read without this server"),
            ("window.print()", "printing a result should produce the result"),
        ):
            self.assertIn(needle, self.source, why)
        # And the buttons are actually wired, not just the helpers defined.
        self.assertIn("exportCsv()", self.source,
                      "the CSV helper exists but nothing calls it, so the user "
                      "has a way out in the code and not on the page")

    def test_changing_the_area_clears_the_previous_result(self):
        # A result for the old area sat under a newly drawn boundary with nothing
        # marking it stale, which read as "these are the numbers for this area".
        self.assertIn("clearAnalysis", self.source)
        body = self.source.split("const clearAnalysis")[1][:800]
        for piece in ("setAnalysisSummary(null)", "setOffline(null)", "setResponse('')"):
            self.assertIn(piece, body, f"clearAnalysis does not reset {piece}")

    def test_a_finished_job_fetches_its_result(self):
        # The progress line said "Done" while the card still said no series had
        # been processed, and only pressing Analyze again showed the work.
        self.assertIn("analyzeRef.current()", self.source,
                      "a job that reaches ready does not load its result")

    def test_errors_render_as_errors(self):
        # A failure used to render in the same paragraph style as a result, in
        # the same panel, so it read as a finding.
        self.assertIn("responseIsError", self.source)
        self.assertIn("role=\"alert\"", self.source)
        self.assertIn("Try again", self.source, "no way to retry from the error")

    def test_no_machine_token_is_rendered_verbatim(self):
        # A user read `landcover_area_exceeded` where a sentence belonged.
        self.assertIn("readableNote", self.source)
        body = self.source.split("function readableNote")[1][:700]
        self.assertIn("ERROR_COPY", body,
                      "an unrecognised code must fall through to a sentence")
        # The component that renders the note must route it through the
        # translator, not print it. Checked on the component body, because
        # asserting the call exists somewhere in the file would pass even if the
        # note were still rendered raw right beside it.
        line = self.source[self.source.index("function EvidenceLine("):][:900]
        self.assertIn("readableNote(", line,
                      "EvidenceLine renders the raw note, so a code can still "
                      "reach the reader")
        self.assertNotIn("{evidence?.note}", line,
                         "EvidenceLine still prints the untranslated note")

    def test_the_limit_shown_is_the_one_the_server_reports(self):
        # The banner hard-coded 100 km2, so a deployment that changed the cap
        # published a number its own API disagreed with.
        self.assertNotIn("within 100 km²", self.source,
                         "the analysis limit is a server setting, not a constant")
        self.assertIn("syncLimitKm2", self.source)

    def test_only_an_oversized_area_offers_the_offline_route(self):
        # A 413 is three things; sending all of them to the offline panel told
        # the user to wait for a job that could not help.
        self.assertIn("isAreaTooLarge", self.source)
        backend = (Path(__file__).resolve().parent.parent / "main.py").read_text(
            encoding="utf-8")
        self.assertIn("synchronous limit", backend,
                      "the phrase the client matches on must exist server-side")


def _handler_source(name: str) -> str:
    source = (Path(__file__).resolve().parent.parent / "main.py").read_text(
        encoding="utf-8")
    start = source.index(f"async def {name}(")
    return source[start:source.index("\n@app", start)]


class CardHonestyTests(unittest.TestCase):
    @property
    def source(self) -> str:
        return (CLIENT_ROOT / "app" / "page.tsx").read_text(encoding="utf-8")

    """The two fields that were computed and then filed away.

    Both were already in every payload and both were inside a collapsed
    "show your working", which is where a number goes to stop being read. Four
    products sit in one row and can be six months apart, and an ERA5 area mean
    over a small paddock is one 28 km cell. Asserted on the card, not in the
    disclosure.
    """

    def test_the_card_states_the_series_end(self):
        self.assertIn("series ends", self.source,
                      "freshness is computed and then hidden; the rainfall data is "
                      "months older than the vegetation beside it")

    def test_the_series_end_is_where_the_data_stops_not_where_the_window_ends(self):
        # The first version read window.end, so a window ending this month claimed
        # the series ran to this month when the newest ERA5 month is six months
        # behind. That is the same overstatement the line was added to remove,
        # introduced by the fix for it. Found by reading the browser output.
        body = self.source.split("series ends")[0][-400:]
        self.assertNotIn("rain.window?.end.slice", body,
                         "the card must not report the requested window as the "
                         "end of the data")

    def test_the_card_states_the_grid_the_number_came_from(self):
        self.assertIn("grid cell", self.source)
        self.assertIn("resolution_km", self.source)

    def test_it_warns_when_one_cell_stands_for_the_whole_outline(self):
        self.assertIn("overSpecified", self.source,
                      "a single ERA5 cell says the same thing about 10 km2 and "
                      "600 km2, and the reader is not told")
        # The first version required two or more cells, so the strongest case for
        # the warning -- one cell, a small outline -- suppressed it. The condition
        # is now about the ratio of the outline to the cell, not the cell count.
        body = self.source.split("function overSpecified")[1][:900]
        self.assertIn("bbox_area_km2", body)
        self.assertNotIn("rain.grid_cells < 2", body,
                         "a single cell is the case that most needs saying")

    def test_the_two_reference_periods_are_distinguished(self):
        # Rainfall uses the published WMO 1991-2020 normal; the MODIS series forms
        # its own from the years it has, which begin in 2000. Both are correct and
        # they sit in the same section, so a reader comparing the panels needs to
        # know they are not against the same baseline.
        chart = (CLIENT_ROOT / "components" / "TimeSection.tsx").read_text(encoding="utf-8")
        self.assertIn("1991–2020 normal", chart)
        self.assertIn("from 2000", chart,
                      "the vegetation normal is not 1991-2020 and the chart says so")

    def test_the_disclosure_is_still_available_for_the_rest(self):
        # The point is not to delete the detail, only to stop burying the two
        # fields a reader needs in order to trust the number above them.
        self.assertIn("Working", self.source)
        self.assertIn("['series ends'", self.source)


class HandoffTests(unittest.TestCase):
    @property
    def source(self) -> str:
        return (CLIENT_ROOT / "app" / "page.tsx").read_text(encoding="utf-8")

    """An administrative id, in and out.

    The handoff is the point of the resolver. globe holds canonical ids from the
    boundary service, and a link carrying one is small, stable, and resolvable
    without shipping geometry in a URL -- which a 10,000-vertex boundary cannot do
    inside the 8,000 characters a pasted link survives.
    """

    def test_the_resolver_is_called_by_the_client(self):
        self.assertIn("/areas/resolve", self.source,
                      "the resolver exists as an API with no path from the page")

    def test_a_share_link_carries_the_administrative_id(self):
        self.assertIn("admin:", self.source,
                      "a link should carry the id another system already holds, "
                      "not a megabyte of coordinates")
        self.assertIn("adminArea?.id", self.source)

    def test_a_link_carrying_an_id_resolves_rather_than_needing_geometry(self):
        self.assertIn("params.get('admin')", self.source)
        self.assertIn("setAdminArea", self.source)

    def test_the_resolved_area_says_it_is_the_whole_unit(self):
        # A named administrative boundary is not something the user drew, and the
        # figures describe the unit. Saying so is the difference between a named
        # area and a precise-looking one.
        self.assertIn("not something you drew", self.source)

    def test_a_failed_resolution_offers_the_ways_that_still_work(self):
        self.assertIn("Could not resolve that area", self.source)
        self.assertIn("draw it on the map or upload a file", self.source)


class DatesPrimaryTests(unittest.TestCase):
    @property
    def source(self) -> str:
        return (CLIENT_ROOT / "app" / "page.tsx").read_text(encoding="utf-8")

    """The window control must lead with the dates.

    The question layer has answered "the 2019/20 drought" and "since the last
    rains" since it was written, and the interface still offered 1/3/10/30 years
    with the date fields second -- so the ranger's and the NRT manager's actual
    questions needed a hunt for a secondary field.
    """

    def test_the_date_fields_come_before_the_presets(self):
        # Scoped to the control itself: the same attribute appears in the link
        # restore, so searching the whole file compared two unrelated places.
        source = self.source
        control = source[source.index('aria-label="Window start"') - 1200:
                         source.index('aria-label="Window start"') + 1200]
        self.assertLess(control.index('type="date"'), control.index("windowYears === years"),
                        "dates are the question; presets are a convenience")

    def test_choosing_a_preset_clears_the_dates(self):
        self.assertIn("setCustomStart('')", self.source,
                      "a preset and a custom range cannot both be active, and "
                      "silently preferring one is how a shared link stops "
                      "reproducing what the sender saw")

    def test_the_window_says_what_it_means(self):
        # Bar width and sensor reach are consequences of the dates, and the only
        # reader who can act on them is the one who chose them.
        self.assertIn("windowExplanation", self.source)
        body = self.source.split("const windowExplanation")[1][:900]
        self.assertIn("no sensor archive reaches back", body,
                      "a window before Landsat has no vegetation answer and the "
                      "control should say so before the request is made")
        self.assertIn("bin it yearly", body,
                      "361 monthly bars is a different claim from 32 annual ones")


class ContainingUnitTests(unittest.TestCase):
    @property
    def source(self) -> str:
        return (CLIENT_ROOT / "app" / "page.tsx").read_text(encoding="utf-8")

    """The other direction: having drawn something, which unit is it in?"""

    def test_the_containing_unit_can_be_found_from_a_drawn_outline(self):
        self.assertIn("findContainingArea", self.source)
        self.assertIn("/areas/resolve?lat=", self.source,
                      "the boundary service answers a point, and the outline's "
                      "centroid is the honest point to ask with")

    def test_adopting_a_unit_is_offered_rather_than_done(self):
        # Snapping replaces the outline the person drew. Doing that unasked would
        # be the resolver overwriting a decision the reader made.
        self.assertIn("useContainingArea", self.source)
        self.assertIn("Which unit is this in?", self.source)

    def test_it_states_both_sizes_when_offering(self):
        self.assertIn("outline sits", self.source)
        self.assertIn("which covers", self.source,
                      "a reader deciding whether to swap a 60 km2 outline for a "
                      "774 km2 unit needs to see both numbers")

    def test_the_server_publishes_the_box_so_the_browser_need_not_measure_it(self):
        import main

        source = (Path(__file__).resolve().parent.parent / "main.py").read_text()
        self.assertIn('"bbox": [round(v, 6) for v in aoi["bbox"]]', source,
                      "the area was published without the box, so the client had "
                      "to measure the outline again to ask the question")


class LowVegetationTests(unittest.TestCase):
    @property
    def source(self) -> str:
        return (CLIENT_ROOT / "app" / "page.tsx").read_text(encoding="utf-8")

    def test_a_low_index_says_it_may_be_cloud(self):
        # Documented in the README and reachable through the registry, and shown
        # to nobody: a persistently cloudy area reads low, and that figure is the
        # one most likely to be quoted as bare ground.
        # Whitespace collapsed: the formatter wraps this sentence across lines, and
        # a literal search for it fails on the wrap rather than on the copy.
        self.assertIn("reading as uncertain rather than as bare",
                      " ".join(self.source.split()))
