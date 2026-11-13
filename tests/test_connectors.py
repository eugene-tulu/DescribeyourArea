import unittest
from unittest.mock import MagicMock, patch

from connectors import (
    arcgis_query_url,
    features_to_observations,
    fetch_arcgis,
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