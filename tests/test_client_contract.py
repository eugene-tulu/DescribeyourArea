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
        self.assertTrue({"area", "datasets", "sensor", "years"} <= keys,
                        f"the share link no longer encodes the expected keys: {keys}")
        for key in ("area", "datasets", "sensor", "years"):
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

    def test_the_window_control_names_what_it_governs(self):
        source = (CLIENT_ROOT / "app" / "page.tsx").read_text(encoding="utf-8")
        self.assertIn("Vegetation window", source,
                      "the window control must name the one dataset it governs")

    def test_the_window_control_says_what_it_does_not_govern(self):
        # JSX wraps prose across lines, so the copy is read with whitespace
        # collapsed or every sentence fails to match itself.
        source = " ".join(
            (CLIENT_ROOT / "app" / "page.tsx").read_text(encoding="utf-8").split())
        for phrase in ("no time dimension", "full monthly record"):
            self.assertIn(phrase, source,
                          f"the window control must say {phrase!r} rather than "
                          f"letting a 1y selection imply all four cards changed")

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
