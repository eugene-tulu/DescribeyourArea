"""The served response must match the published contract.

The endpoint used to declare ``summary: Dict[str, Any]``, so the OpenAPI document
said nothing about the payload and the frontend carried twelve hand-written
TypeScript interfaces with nothing keeping them in step. That is how a duplicated
literal drifted and shipped a stale limit to production.

These tests fail when a producer changes shape without the contract changing,
which is the whole point.
"""

import json
import unittest

from fastapi.testclient import TestClient

import main
from contract import ContextResponse, ContextSummary

SMALL = {"type": "Feature", "properties": {}, "geometry": {"type": "Polygon", "coordinates": [[
    [35.10, -1.55], [35.17, -1.55], [35.17, -1.48], [35.10, -1.48], [35.10, -1.55]
]]}}


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)

    def test_the_response_model_is_declared_not_a_bare_dict(self):
        fields = set(ContextResponse.model_fields)
        self.assertEqual(fields, {"summary"})
        self.assertTrue(issubclass(ContextResponse.model_fields["summary"].annotation, ContextSummary))

    def test_every_summary_key_is_declared(self):
        declared = set(ContextSummary.model_fields)
        # The keys the endpoint actually produces.
        produced = {
            "dem", "ndvi", "landcover", "rainfall", "country",
            "scene_dates", "scene_ids", "analysis", "caveats",
        }
        self.assertEqual(produced - declared, set(), "produced a key the contract omits")

    def test_openapi_documents_the_summary_rather_than_a_free_dict(self):
        schema = self.client.get("/openapi.json").json()
        summary = schema["components"]["schemas"]["ContextSummary"]
        # A Dict[str, Any] documents nothing; a model documents its properties.
        self.assertIn("properties", summary)
        self.assertIn("dem", summary["properties"])
        self.assertIn("caveats", summary["properties"])

    def test_a_served_response_validates_against_the_contract(self):
        response = self.client.post("/generate-context?datasets=dem,landcover", json={"geojson": SMALL})
        self.assertEqual(response.status_code, 200, response.text[:300])
        # Raises if the producers and the contract have drifted.
        ContextResponse.model_validate(response.json())

    def test_a_rejected_request_never_reaches_the_contract(self):
        response = self.client.post("/generate-context?datasets=nonsense", json={"geojson": SMALL})
        self.assertEqual(response.status_code, 422)


class EvidenceTests(unittest.TestCase):
    """A number without a stated kind is an assertion pretending to be a measurement."""

    def setUp(self):
        self.client = TestClient(main.app)

    def test_every_module_carries_a_status(self):
        response = self.client.post(
            "/generate-context?datasets=dem,landcover,ndvi,rainfall", json={"geojson": SMALL}
        )
        self.assertEqual(response.status_code, 200, response.text[:300])
        summary = response.json()["summary"]
        for name in ("dem", "landcover", "ndvi", "rainfall"):
            with self.subTest(module=name):
                self.assertIn("status", summary[name], f"{name} has no status")
                self.assertIn("evidence", summary[name], f"{name} has no evidence")

    def test_observed_derived_and_modelled_are_distinguished(self):
        response = self.client.post(
            "/generate-context?datasets=dem,landcover,ndvi", json={"geojson": SMALL}
        )
        self.assertEqual(response.status_code, 200, response.text[:300])
        summary = response.json()["summary"]
        self.assertEqual(summary["dem"]["evidence"]["status"], "observed")
        self.assertEqual(summary["landcover"]["evidence"]["status"], "observed")
        # An index computed from a reflectance product is derived, not observed.
        self.assertEqual(summary["ndvi"]["evidence"]["status"], "derived")

    def test_rainfall_is_declared_modelled_not_observed(self):
        # ERA5 is a reanalysis: a model that assimilates observations, not a gauge
        # reading. This is the distinction most worth stating and easiest to lose.
        result = main._with_rainfall_evidence({
            "status": "ok", "source": "ERA5", "doi": "10.x", "license": "CC-BY 4.0",
        })
        self.assertEqual(result["evidence"]["status"], "modelled")
        self.assertIn("not a gauge reading", result["evidence"]["note"])
        self.assertEqual(result["evidence"]["doi"], "10.x")

    def test_an_absent_module_is_unconfirmed_not_silently_missing(self):
        result = main._with_rainfall_evidence({"status": "not_computed", "reason": "no_precomputed_series"})
        self.assertEqual(result["evidence"]["status"], "unconfirmed")
        self.assertEqual(result["evidence"]["note"], "no_precomputed_series")

    def test_an_unavailable_module_is_unconfirmed(self):
        result = main._with_dem_evidence({"error": "no_valid_elevation_pixels"})
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["evidence"]["status"], "unconfirmed")


class CaveatTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)

    def test_a_missing_module_is_explained_rather_than_left_as_a_gap(self):
        response = self.client.post(
            "/generate-context?datasets=dem,rainfall", json={"geojson": SMALL}
        )
        self.assertEqual(response.status_code, 200, response.text[:300])
        caveats = response.json()["summary"]["caveats"]
        self.assertTrue(any("Rainfall" in c for c in caveats), caveats)

    def test_a_repaired_boundary_is_disclosed(self):
        bowtie = {"type": "Polygon", "coordinates": [[
            [35.10, -1.55], [35.17, -1.45], [35.17, -1.55], [35.10, -1.45], [35.10, -1.55]
        ]]}
        from fastapi.testclient import TestClient as TC

        response = TC(main.app).post("/generate-context?datasets=dem", json={"geojson": {
            "type": "Feature", "properties": {}, "geometry": bowtie}})
        self.assertEqual(response.status_code, 200, response.text[:300])
        caveats = response.json()["summary"]["caveats"]
        self.assertTrue(any("boundary was modified" in c for c in caveats), caveats)


if __name__ == "__main__":
    unittest.main()


class WindowParameterTests(unittest.TestCase):
    """A malformed window is a client error, not a missing result.

    Left to the vegetation module it degraded to "unavailable", which tells an API
    caller the area has no data when the truth is that the request was wrong.
    """

    def setUp(self):
        self.client = TestClient(main.app)

    def _post(self, **params):
        query = "&".join(f"{k}={v}" for k, v in params.items())
        return self.client.post(f"/generate-context?datasets=ndvi&{query}", json={"geojson": SMALL})

    def test_a_user_defined_range_is_honoured(self):
        response = self._post(window_start="1997-01-01", window_end="1999-12-31")
        self.assertEqual(response.status_code, 200, response.text[:300])
        result = response.json()["summary"]["ndvi"]
        self.assertEqual(result["window"], {"start": "1997-01-01", "end": "1999-12-31"})
        # The 1997/98 El Nino is the question the range exists to answer.
        if result.get("status") == "ok":
            self.assertGreater(result["mean"], 0.0)

    def test_an_inverted_range_is_rejected(self):
        response = self._post(window_start="2026-06-01", window_end="2020-01-01")
        self.assertEqual(response.status_code, 422)
        self.assertIn("after the end", str(response.json()["detail"]))

    def test_a_malformed_date_is_rejected(self):
        response = self._post(window_start="06/01/2024", window_end="2024-06-01")
        self.assertEqual(response.status_code, 422)
        self.assertIn("ISO", str(response.json()["detail"]))

    def test_a_window_before_the_archive_reports_why_rather_than_emptily(self):
        response = self._post(
            window_start="1900-01-01", window_end="2026-01-01", sensor="landsat"
        )
        self.assertEqual(response.status_code, 200, response.text[:300])
        result = response.json()["summary"]["ndvi"]
        self.assertEqual(result["status"], "skipped")
        self.assertIn("begins on", result["warning"])
        self.assertNotIn("mean", result)
