import unittest
from unittest.mock import MagicMock, patch

from connectors import (
    arcgis_query_url,
    features_to_observations,
    fetch_arcgis,
    fetch_kobo,
)

FEATURES = [
    {
        "geometry": {"type": "Point", "coordinates": [35.4, -1.5]},
        "properties": {
            "OBJECTID": 1,
            "date": 1750000000000,
            "observer": "A. Kima",
            "land_use": "grazing",
        },
    },
    {
        "geometry": {
            "type": "Polygon",
            "coordinates": [[[35.0, -1.0], [35.2, -1.0], [35.2, -1.2], [35.0, -1.2], [35.0, -1.0]]],
        },
        "properties": {"plot_id": "P-2"},
    },
]


class ArcGISUrlTests(unittest.TestCase):
    def test_service_gets_the_layer_query_path(self):
        url = arcgis_query_url(
            "https://services.arcgis.com/x/arcgis/rest/services/Foo/FeatureServer",
            "0",
            [35.0, -2.0, 36.0, -1.0],
        )
        self.assertTrue(url.startswith(
            "https://services.arcgis.com/x/arcgis/rest/services/Foo/FeatureServer/0/query?"
        ))
        self.assertIn("geometry=35.0%2C-2.0%2C36.0%2C-1.0", url)
        self.assertIn("f=geojson", url)
        self.assertIn("inSR=4326", url)

    def test_layer_url_gets_query(self):
        url = arcgis_query_url(
            "https://services.arcgis.com/x/FeatureServer/0", "0", [35, -2, 36, -1]
        )
        self.assertTrue(url.startswith("https://services.arcgis.com/x/FeatureServer/0/query?"))

    def test_query_url_is_reused(self):
        url = arcgis_query_url("https://x/y/0/query", "0", [35, -2, 36, -1])
        self.assertTrue(url.startswith("https://x/y/0/query?"))

    def test_token_is_forwarded(self):
        url = arcgis_query_url(
            "https://x/FeatureServer", "0", [35, -2, 36, -1], token="secret"
        )
        self.assertIn("token=secret", url)

    def test_missing_bbox_is_rejected(self):
        with self.assertRaises(Exception):
            arcgis_query_url("https://x/FeatureServer", "0", [])


class FeatureMappingTests(unittest.TestCase):
    def test_point_polygon_epoch_and_fields(self):
        observations = features_to_observations(
            FEATURES, id_field="plot_id", date_field="date"
        )
        self.assertEqual(len(observations), 2)
        first = observations[0]
        self.assertEqual(first.lon, 35.4)
        self.assertEqual(first.lat, -1.5)
        self.assertEqual(first.observer, "A. Kima")
        self.assertEqual(first.type, "grazing")
        self.assertTrue(first.timestamp.startswith("2025-06-15"))
        second = observations[1]
        self.assertEqual(second.id, "P-2")
        self.assertTrue(35.0 <= second.lon <= 35.2)
        self.assertTrue(-1.2 <= second.lat <= -1.0)

    def test_missing_geometry_is_skipped(self):
        observations = features_to_observations([
            {"geometry": None, "properties": {"OBJECTID": 1}},
        ])
        self.assertEqual(observations, [])


class FetchArcGISTests(unittest.TestCase):
    @patch("connectors.requests.get")
    def test_parses_a_geojson_response(self, get):
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"features": FEATURES}
        get.return_value = response
        observations, errors = fetch_arcgis({
            "url": "https://x/FeatureServer",
            "layer_id": "0",
            "bbox": [35, -2, 36, -1],
        })
        self.assertEqual(len(observations), 2)
        self.assertEqual(errors, [])

    @patch("connectors.requests.get")
    def test_surfaces_an_arcgis_error(self, get):
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"error": {"message": "Invalid token"}}
        get.return_value = response
        observations, errors = fetch_arcgis({
            "url": "https://x/FeatureServer",
            "layer_id": "0",
            "bbox": [35, -2, 36, -1],
        })
        self.assertEqual(observations, [])
        self.assertIn("Invalid token", errors[0])


if __name__ == "__main__":
    unittest.main()

class KoboTests(unittest.TestCase):
    @patch("connectors.requests.get")
    def test_maps_geolocation_and_filters_bbox_and_date(self, get):
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"results": [
            {"_id": 1, "_submission_time": "2026-06-15T10:00:00", "_geolocation": [-1.5, 35.4],
             "_submitted_by": "A. Kima", "land_use": "grazing"},
            {"_id": 2, "_submission_time": "2026-07-01T10:00:00", "_geolocation": [-1.5, 40.0],
             "_submitted_by": "B. Siron", "land_use": "crops"},
        ]}
        get.return_value = response
        observations, errors = fetch_kobo({
            "server_url": "https://kf.kobotoolbox.org",
            "asset_uid": "abc",
            "bbox": [35.0, -2.0, 36.0, -1.0],
            "start": "2026-06-01",
            "end": "2026-06-30",
            "type_field": "land_use",
        })
        self.assertEqual(errors, [])
        self.assertEqual(len(observations), 1)
        self.assertEqual(observations[0].lon, 35.4)
        self.assertEqual(observations[0].lat, -1.5)
        self.assertEqual(observations[0].observer, "A. Kima")
        self.assertEqual(observations[0].type, "grazing")

    @patch("connectors.requests.get")
    def test_missing_geolocation_is_skipped(self, get):
        response = MagicMock()
        response.raise_for_status.return_value = None
        response.json.return_value = {"results": [{"_id": 1, "_submission_time": "2026-06-15"}]}
        get.return_value = response
        observations, errors = fetch_kobo({"server_url": "https://kf.kobotoolbox.org", "asset_uid": "abc"})
        self.assertEqual(observations, [])
        self.assertEqual(errors, [])
