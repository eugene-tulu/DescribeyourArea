"""Offline checks for the request-area admission policy.

These tests deliberately exercise only deterministic geometry helpers.  They
must not contact a STAC API, download a raster, or invoke a background task.
Run with ``python -m unittest discover -s tests -v``.
"""

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

from main import (
    AVAILABLE_DATASETS,
    _requested_datasets,
    aoi_bbox,
    canonicalize_geojson,
    compute_median_ndvi,
    validate_aoi,
)


SMALL_POLYGON = {
    "type": "Polygon",
    "coordinates": [
        [
            [36.8000, -1.3000],
            [36.8100, -1.3000],
            [36.8100, -1.2900],
            [36.8000, -1.2900],
            [36.8000, -1.3000],
        ]
    ],
}


class CanonicalizeGeoJSONTests(unittest.TestCase):
    def test_wraps_a_raw_polygon_in_a_feature(self) -> None:
        feature = canonicalize_geojson(SMALL_POLYGON)

        self.assertEqual(feature["type"], "Feature")
        self.assertEqual(feature["geometry"]["type"], "Polygon")
        self.assertEqual(aoi_bbox(feature), [36.8, -1.3, 36.81, -1.29])
        self.assertEqual(feature["properties"], {})

    def test_unions_polygon_features_in_a_feature_collection(self) -> None:
        collection = {
            "type": "FeatureCollection",
            "features": [
                {"type": "Feature", "properties": {"name": "west"}, "geometry": SMALL_POLYGON},
                {
                    "type": "Feature",
                    "properties": {"name": "east"},
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [
                            [
                                [36.8200, -1.3000],
                                [36.8300, -1.3000],
                                [36.8300, -1.2900],
                                [36.8200, -1.2900],
                                [36.8200, -1.3000],
                            ]
                        ],
                    },
                },
            ],
        }

        feature = canonicalize_geojson(collection)

        self.assertEqual(feature["type"], "Feature")
        self.assertEqual(feature["geometry"]["type"], "MultiPolygon")
        self.assertEqual(aoi_bbox(feature), [36.8, -1.3, 36.83, -1.29])

    def test_rejects_non_polygon_geojson(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            canonicalize_geojson({"type": "Point", "coordinates": [36.8, -1.3]})

        self.assertEqual(raised.exception.status_code, 400)

    def test_rejects_an_empty_feature_collection(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            canonicalize_geojson({"type": "FeatureCollection", "features": []})

        self.assertEqual(raised.exception.status_code, 400)


class ValidateAOITests(unittest.TestCase):
    def test_returns_canonical_feature_bbox_and_geodesic_bbox_area(self) -> None:
        result = validate_aoi(
            SMALL_POLYGON,
            max_bytes=10_000,
            max_vertices=100,
            max_bbox_km2=10.0,
        )

        self.assertEqual(set(result), {"feature", "bbox", "bbox_area_km2"})
        self.assertEqual(result["feature"]["type"], "Feature")
        self.assertEqual(result["bbox"], [36.8, -1.3, 36.81, -1.29])
        # A 0.01° by 0.01° bounding box at this latitude is approximately
        # 1.2 km².  A broad range keeps this independent of small geodesic
        # implementation details while still detecting degree-squared output.
        self.assertGreater(result["bbox_area_km2"], 1.0)
        self.assertLess(result["bbox_area_km2"], 1.5)

    def test_rejects_payloads_over_the_byte_limit_before_processing(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            validate_aoi(
                SMALL_POLYGON,
                max_bytes=20,
                max_vertices=100,
                max_bbox_km2=10.0,
            )

        self.assertEqual(raised.exception.status_code, 413)


class NDVIGuardrailTests(unittest.TestCase):
    def test_large_ndvi_bbox_is_skipped_before_a_scene_search(self) -> None:
        large_bbox = [36.0, -1.5, 36.2, -1.3]
        with patch("main._search_sentinel_items", new_callable=AsyncMock) as search:
            result = asyncio.run(
                compute_median_ndvi(
                    large_bbox,
                    SMALL_POLYGON,
                    max_area_km2=10.0,
                    max_scenes=4,
                )
            )

        self.assertEqual(result["status"], "skipped")
        self.assertIn("10", result["warning"])
        search.assert_not_awaited()

    def test_rejects_too_many_vertices(self) -> None:
        with self.assertRaises(HTTPException) as raised:
            validate_aoi(
                SMALL_POLYGON,
                max_bytes=10_000,
                max_vertices=3,
                max_bbox_km2=10.0,
            )

        self.assertEqual(raised.exception.status_code, 413)

    def test_rejects_an_aoi_whose_bbox_exceeds_the_synchronous_cap(self) -> None:
        large_polygon = {
            "type": "Polygon",
            "coordinates": [
                [
                    [36.0, -1.5],
                    [36.2, -1.5],
                    [36.2, -1.3],
                    [36.0, -1.3],
                    [36.0, -1.5],
                ]
            ],
        }

        with self.assertRaises(HTTPException) as raised:
            validate_aoi(
                large_polygon,
                max_bytes=10_000,
                max_vertices=100,
                max_bbox_km2=10.0,
            )

        self.assertEqual(raised.exception.status_code, 413)


class RequestedDatasetsTests(unittest.TestCase):
    def test_defaults_to_every_supported_module(self) -> None:
        self.assertEqual(_requested_datasets(None), AVAILABLE_DATASETS)

    def test_parses_a_comma_separated_selection(self) -> None:
        self.assertEqual(_requested_datasets(" dem , NDVI "), {"dem", "ndvi"})

    def test_rainfall_is_selectable(self) -> None:
        self.assertEqual(_requested_datasets("rainfall,dem"), {"rainfall", "dem"})
        self.assertIn("rainfall", AVAILABLE_DATASETS)

    def test_rejects_datasets_that_are_no_longer_supported(self) -> None:
        for retired in ("soils", "population", "climate", "hydrology"):
            with self.subTest(dataset=retired):
                with self.assertRaises(HTTPException) as raised:
                    _requested_datasets(retired)
                self.assertEqual(raised.exception.status_code, 422)


if __name__ == "__main__":
    unittest.main()
