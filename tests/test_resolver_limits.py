"""The resolver is the one public route that costs somebody else a round trip.

Free, unauthenticated, and it forwards to the boundary service on every call, so a
single unknown person could drive it all day. The guard counts by the caller's own
prefix -- not the proxy's -- so a forged X-Forwarded-For cannot buy extra calls.

It is in-process and fixed-window: enough to stop a script, useless against a
distributed one. The tests say so rather than implying more protection than exists.
"""

import unittest

import main
from fastapi.testclient import TestClient


class RateLimitTests(unittest.TestCase):
    def setUp(self):
        self.previous = main.RESOLVE_RATE_LIMIT
        main.RESOLVE_RATE_LIMIT = 4
        main._resolve_hits.clear()
        self.client = TestClient(main.app)
        self.addCleanup(self._restore)

    def _restore(self):
        main.RESOLVE_RATE_LIMIT = self.previous
        main._resolve_hits.clear()

    def test_it_refuses_the_caller_who_asks_too_often(self):
        codes = [self.client.get("/areas/resolve?name=Atlantis").status_code
                 for _ in range(6)]
        self.assertEqual(codes[:4], [404] * 4,
                         "a name that matches nothing is still a lookup")
        self.assertEqual(codes[4:], [429, 429])

    def test_it_says_when_the_caller_may_retry(self):
        for _ in range(4):
            self.client.get("/areas/resolve?name=Atlantis")
        response = self.client.get("/areas/resolve?name=Atlantis")
        self.assertEqual(response.status_code, 429)
        self.assertIn("try again in", response.json()["detail"])
        self.assertTrue(response.headers.get("Retry-After"),
                        "a refusal with no Retry-After is a shrug")

    def test_the_shipped_limit_is_not_much(self):
        # Thirty a minute is enough for a person naming a few areas and not enough
        # for a script. Asserted so raising it is a deliberate act.
        # Read from a fresh module state: setUp lowers the constant to 4 so the
        # refusal path is reachable without making 30 calls.
        import importlib

        reloaded = importlib.reload(main)
        self.addCleanup(importlib.reload, main)
        self.assertGreaterEqual(reloaded.RESOLVE_RATE_LIMIT, 20)
        self.assertLessEqual(reloaded.RESOLVE_RATE_LIMIT, 60)

    def test_the_window_does_not_grow_without_bound(self):
        # Every address that ever called leaves a key; a long-running process
        # would otherwise accumulate one per caller forever.
        main._resolve_hits.clear()
        for i in range(5000):
            main._resolve_hits[f"1.2.3.{i}"] = [0.0]
        main._RESOLVE_RATE_LIMIT = 1
        try:
            main._rate_limit_resolve("9.9.9.9")
        except Exception:
            pass
        finally:
            main._RESOLVE_RATE_LIMIT = self.previous
        self.assertLessEqual(len(main._resolve_hits), 4096,
                             "the rate table must be swept, not just grown")


class StoredContextTests(unittest.TestCase):
    """A queued area must be able to be shown.

    `/generate-context` refuses anything past the synchronous cap and always
    will, so re-running it to display a queued area's results returned the same
    refusal and the page showed an error over four completed modules. Queueing a
    large area produced work and no output.
    """

    def setUp(self):
        import os
        import tempfile

        self.previous = os.environ.get("RAINFALL_CACHE_DIR")
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["RAINFALL_CACHE_DIR"] = self.tmp.name
        self.client = TestClient(main.app)
        self.addCleanup(self._restore)
        self.addCleanup(self.tmp.cleanup)

    def _restore(self):
        import os

        if self.previous is None:
            os.environ.pop("RAINFALL_CACHE_DIR", None)
        else:
            os.environ["RAINFALL_CACHE_DIR"] = self.previous

    def test_it_refuses_an_unknown_key(self):
        self.assertEqual(self.client.get("/context?cache_key=nope").status_code, 422)

    def test_it_reports_an_area_with_nothing_stored_rather_than_inventing_one(self):
        response = self.client.get("/context?cache_key=" + "c" * 32)
        self.assertEqual(response.status_code, 404)
        self.assertIn("nothing has been computed", response.json()["detail"])

    def test_it_assembles_the_reading_from_stored_artefacts(self):
        import jobs

        key = "d" * 32
        jobs.write_artefact(key, "dem", {"status": "ok", "mean": 1500.0, "unit": "m"})
        jobs.write_artefact(key, "rainfall", {"status": "ok", "series": []})
        response = self.client.get(f"/context?cache_key={key}")
        self.assertEqual(response.status_code, 200, response.text[:200])
        summary = response.json()["summary"]
        self.assertEqual(summary["analysis"]["mode"], "from computed artefacts")
        self.assertEqual(summary["dem"]["mean"], 1500.0)
        # The same shape the synchronous path returns, so the interface does not
        # need to know how the area was read.
        self.assertIn("dem", summary)
        self.assertIn("caveats", summary)

    def test_it_says_which_module_is_still_missing_and_shows_the_rest(self):
        import jobs

        key = "e" * 32
        jobs.write_artefact(key, "dem", {"status": "ok", "mean": 1500.0})
        response = self.client.get(f"/context?cache_key={key}&indicators=dem,ndvi")
        self.assertEqual(response.status_code, 200)
        caveats = " ".join(response.json()["summary"]["caveats"])
        self.assertIn("ndvi", caveats)
        self.assertIn("still being computed", caveats)


class RainfallIsNotAnArtefactTests(unittest.TestCase):
    """Rainfall does not live in the artefact store, and pretending it does
    silently drops the headline module from an assembled reading.

    The worker writes rainfall to the ERA5 cache; the artefact store holds the
    raster products. A first version of `/context` read artefacts only, so a
    queued area came back with elevation, land cover and vegetation and no
    rainfall at all -- and named it as "still being computed" rather than
    admitting it had looked in the wrong place.
    """

    def setUp(self):
        import os
        import tempfile

        self.previous = os.environ.get("RAINFALL_CACHE_DIR")
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["RAINFALL_CACHE_DIR"] = self.tmp.name
        self.client = TestClient(main.app)
        self.addCleanup(self._restore)
        self.addCleanup(self.tmp.cleanup)

    def _restore(self):
        import os

        if self.previous is None:
            os.environ.pop("RAINFALL_CACHE_DIR", None)
        else:
            os.environ["RAINFALL_CACHE_DIR"] = self.previous

    def test_rainfall_is_read_from_its_own_cache(self):
        import rainfall

        geom = {"type": "Polygon", "coordinates": [[[37.70, 1.20], [37.80, 1.20],
                                                    [37.80, 1.30], [37.70, 1.30],
                                                    [37.70, 1.20]]]}
        key = rainfall.geometry_hash(geom)
        rainfall.write_cache(key, {
            "processing_version": rainfall.RAINFALL_PROCESSING_VERSION,
            "series": [{"month": "2026-01", "precip_mm": 10.0}],
            "grid_cells": 1,
        })
        summary = self.client.get(f"/context?cache_key={key}").json()["summary"]
        self.assertIn("rainfall", summary,
                      "the headline module vanished from the assembled reading")
        self.assertEqual(len(summary["rainfall"]["series"]), 1)
        self.assertNotIn("Not yet available: rainfall", " ".join(summary["caveats"]))
