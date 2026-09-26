"""Integration tests against live Planetary Computer data.

These use the real Northern Rangelands Trust conservancy polygons because the
defect under test is specifically about real source-tile boundaries: Melako spans
four NASADEM tiles, and a single-tile read silently analysed 3% of its area.
No mock can reproduce that.

Run explicitly, since it needs network access:

    python -m unittest tests.test_planetary_computer -v

Set GEOCONTEXT_SKIP_INTEGRATION=1 to skip in a restricted environment.
"""

import asyncio
import math
import os
import unittest

import numpy as np

import main
from main import (
    MAX_SOURCE_TILES,
    _find_core_assets,
    aoi_bbox,
    canonicalize_geojson,
    compute_landcover_percentages,
    compute_raster_stats,
    compute_median_ndvi,
    interpret_terrain,
    validate_aoi,
)


def _bypass_sync_cap(feature: dict) -> dict:
    """Canonicalise a polygon and get its bbox without the synchronous area cap.

    The raster layer is independent of the admission policy, and Melako is
    exactly the case the cap is meant to defer to the future worker.
    """
    canonical = canonicalize_geojson(feature)
    bbox = aoi_bbox(canonical)
    return {"feature": canonical, "bbox": bbox, "bbox_area_km2": main._bbox_area_km2(bbox)}

# Simplified (~1 km) outlines from the published NRT conservancies GeoJSON.
# II Ngwesi is 89 km2 and sits inside one NASADEM tile; Melako is 5,505 km2 and
# straddles four, which is the case that used to break.
II_NGWESI = {
    "type": "Feature",
    "properties": {"NAME": "II Ngwesi"},
    "geometry": {
        "type": "Polygon",
        "coordinates": [[
            [37.3865, 0.2736], [37.3932, 0.3524], [37.3835, 0.3549], [37.3801, 0.3774],
            [37.3635, 0.4075], [37.3545, 0.4107], [37.3219, 0.3654], [37.3233, 0.2895],
            [37.3865, 0.2736],
        ]],
    },
}

MELAKO = {
    "type": "Feature",
    "properties": {"NAME": "Melako"},
    "geometry": {
        "type": "Polygon",
        "coordinates": [[
            [37.4462, 1.5905], [37.5803, 1.3648], [37.5983, 1.3926], [37.6116, 1.3754],
            [37.6548, 1.4024], [37.8432, 1.4259], [37.8518, 1.4563], [37.9530, 1.3835],
            [37.9478, 1.2608], [38.3429, 1.5733], [38.3873, 1.7574], [38.0818, 2.0585],
            [37.9196, 2.0193], [37.7359, 2.0260], [37.6874, 1.8813], [37.6638, 1.8548],
            [37.6024, 1.8297], [37.6085, 1.7475], [37.5723, 1.7318], [37.5026, 1.6193],
            [37.4462, 1.5905],
        ]],
    },
}


def _network_available() -> bool:
    if os.getenv("GEOCONTEXT_SKIP_INTEGRATION"):
        return False
    try:
        import pystac_client

        client = pystac_client.Client.open(main.STAC_URL)
        client.get_collection("nasadem")
        return True
    except Exception:
        return False


AVAILABLE = _network_available()


@unittest.skipUnless(AVAILABLE, "Planetary Computer is unreachable")
class RealElevationTests(unittest.TestCase):
    def test_single_tile_area_returns_real_statistics(self):
        aoi = _bypass_sync_cap(II_NGWESI)
        assets = asyncio.run(_find_core_assets(aoi["bbox"], need_dem=True, need_landcover=False))
        stats = compute_raster_stats(assets["dem"], aoi["feature"])
        self.assertNotIn("error", stats, f"unexpected error: {stats}")
        # Northern Kenya conservancies sit around 500-1,500 m.
        self.assertTrue(400.0 < stats["mean"] < 2000.0, stats)
        self.assertLessEqual(stats["min"], stats["mean"])
        self.assertLessEqual(stats["mean"], stats["max"])
        self.assertGreaterEqual(stats["std"], 0.0)
        self.assertLessEqual(stats["valid_pixel_fraction"], 1.0)
        self.assertGreater(stats["valid_pixel_fraction"], 0.5)

    def test_multi_tile_area_collects_every_intersecting_tile(self):
        aoi = _bypass_sync_cap(MELAKO)
        assets = asyncio.run(_find_core_assets(aoi["bbox"], need_dem=True, need_landcover=False))
        self.assertGreater(
            len(assets["dem"]),
            1,
            "Melako straddles multiple NASADEM tiles; the search must return them all",
        )
        self.assertLessEqual(len(assets["dem"]), MAX_SOURCE_TILES)

    def test_multi_tile_area_covers_far_more_than_one_tile(self):
        """The core regression.

        Before the fix, ``limit=1`` returned a single tile overlapping ~3% of
        Melako's bounding box, so the reported statistics described a fraction of
        the requested study area while looking entirely normal.
        """
        aoi = _bypass_sync_cap(MELAKO)
        assets = asyncio.run(_find_core_assets(aoi["bbox"], need_dem=True, need_landcover=False))
        full = compute_raster_stats(assets["dem"], aoi["feature"])
        partial = compute_raster_stats(assets["dem"][:1], aoi["feature"])

        self.assertNotIn("error", full)
        self.assertNotIn("error", partial)
        # The single-tile read samples a small slice of the AOI; the accumulated
        # read samples all of it, so the absolute valid-pixel count must grow.
        self.assertGreater(
            full["valid_pixel_count"],
            partial["valid_pixel_count"],
            "accumulating every tile must cover more of the AOI than the first tile alone",
        )
        # And the accumulated result must be a genuine, finite statistic.
        for key in ("mean", "min", "max", "std"):
            self.assertTrue(math.isfinite(full[key]), f"{key} is not finite: {full}")

    def test_terrain_classification_is_consistent_with_the_real_relief(self):
        aoi = _bypass_sync_cap(MELAKO)
        assets = asyncio.run(_find_core_assets(aoi["bbox"], need_dem=True, need_landcover=False))
        stats = interpret_terrain(compute_raster_stats(assets["dem"], aoi["feature"]))
        self.assertNotIn("error", stats)
        self.assertTrue(400.0 < stats["mean"] < 2500.0, stats)
        # Melako climbs the Mau Escarpment, so the classification must follow the
        # measured range rather than a hard-coded expectation.
        span = stats["max"] - stats["min"]
        expected = (
            "relatively flat" if span < 50
            else "moderately undulating" if span < 300
            else "highly variable or mountainous"
        )
        self.assertEqual(stats["terrain_type"], expected, stats)
        self.assertAlmostEqual(stats["elevation_range_m"], span, delta=0.05)


@unittest.skipUnless(AVAILABLE, "Planetary Computer is unreachable")
class RealLandcoverTests(unittest.TestCase):
    def test_percentages_are_a_distribution(self):
        aoi = _bypass_sync_cap(II_NGWESI)
        assets = asyncio.run(_find_core_assets(aoi["bbox"], need_dem=False, need_landcover=True))
        result = compute_landcover_percentages(assets["landcover"], aoi["feature"])
        self.assertNotIn("error", result, f"unexpected error: {result}")
        self.assertAlmostEqual(sum(result["classes"].values()), 100.0, delta=1.0)
        self.assertIn(result["dominant_class"], result["classes"])
        self.assertEqual(
            result["classes"][result["dominant_class"]], result["dominant_percentage"]
        )
        self.assertLessEqual(result["valid_pixel_fraction"], 1.0)

    def test_multi_tile_area_aggregates_every_tile(self):
        aoi = _bypass_sync_cap(MELAKO)
        assets = asyncio.run(_find_core_assets(aoi["bbox"], need_dem=False, need_landcover=True))
        full = compute_landcover_percentages(assets["landcover"], aoi["feature"])
        partial = compute_landcover_percentages(assets["landcover"][:1], aoi["feature"])
        self.assertNotIn("error", full)
        self.assertNotIn("error", partial)
        self.assertAlmostEqual(sum(full["classes"].values()), 100.0, delta=1.0)
        self.assertGreater(
            full["valid_pixel_count"],
            partial["valid_pixel_count"],
            "accumulating every tile must cover more of the AOI than the first tile alone",
        )


@unittest.skipUnless(AVAILABLE, "Planetary Computer is unreachable")
class RealNdviTests(unittest.TestCase):
    def ndvi_for(self, aoi, **kwargs):
        """Compute NDVI, retrying once on a transient upstream timeout.

        The test asserts a real result rather than tolerating "unavailable",
        because four separate code defects surfaced as a polite "unavailable"
        during development. But a slow CDN is not a code defect, so one timeout is
        retried before failing, and any other failure mode still fails outright.
        """
        result = asyncio.run(
            compute_median_ndvi(aoi["bbox"], aoi["feature"]["geometry"], **kwargs)
        )
        if result.get("status") == "unavailable" and "timed out" in result.get("warning", ""):
            result = asyncio.run(
                compute_median_ndvi(aoi["bbox"], aoi["feature"]["geometry"], **kwargs)
            )
        return result

    def test_returns_a_real_ndvi_result(self):
        # Not "unacceptable": a code bug in the read, reprojection or mask stages
        # used to surface as a polite "unavailable", which the suite accepted.
        aoi = _bypass_sync_cap(II_NGWESI)
        result = self.ndvi_for(aoi, max_area_km2=1e9)
        self.assertEqual(result.get("status"), "ok", result)
        for key in ("mean", "min", "max", "std", "p25", "p75"):
            self.assertTrue(math.isfinite(result[key]), f"{key} not finite: {result}")
        self.assertTrue(-1.0 <= result["min"] <= 1.0)
        self.assertTrue(-1.0 <= result["max"] <= 1.0)
        self.assertLessEqual(result["min"], result["mean"])
        self.assertLessEqual(result["mean"], result["max"])
        self.assertLessEqual(result["p25"], result["p75"])
        self.assertGreaterEqual(result["scene_count"], 1)
        self.assertEqual(len(result["scene_ids"]), result["scene_count"])
        self.assertEqual(len(result["scene_dates"]), result["scene_count"])
        self.assertEqual(result["resolution_m"], 20)
        self.assertEqual(result["method"], "sentinel_2_median_composite")

    def test_reports_provenance_coverage(self):
        aoi = _bypass_sync_cap(II_NGWESI)
        result = self.ndvi_for(aoi, max_area_km2=1e9)
        self.assertEqual(result.get("status"), "ok", result)
        self.assertGreater(result["valid_pixel_count"], 0)
        self.assertGreater(result["valid_pixel_fraction"], 0.0)
        self.assertLessEqual(result["valid_pixel_fraction"], 1.0)
        self.assertGreaterEqual(result["scenes_examined"], result["scene_count"])

    def test_bright_terrain_is_not_discarded_as_cloud(self):
        """Sentinel-2 class 5 over bright ground is a known false positive.

        A Sahara tile reads 100% "bright cloud" while its reflectance and NDVI are
        plainly desert. If class 4/5 were rejected outright the result would be
        empty for exactly the semi-arid range this service targets.
        """
        desert = {
            "type": "Feature",
            "properties": {},
            "geometry": {"type": "Polygon", "coordinates": [[
                [30.00, 25.00], [30.08, 25.00], [30.08, 25.08], [30.00, 25.08], [30.00, 25.00]]]},
        }
        aoi = _bypass_sync_cap(desert)
        result = self.ndvi_for(aoi, max_area_km2=1e9)
        self.assertEqual(result.get("status"), "ok", result)
        self.assertGreater(result["valid_pixel_fraction"], 0.5)
        # Desert must read as sparse vegetation, not lush growth.
        self.assertLess(result["mean"], 0.4, result)

    def test_large_area_is_skipped_explicitly_not_downgraded(self):
        aoi = _bypass_sync_cap(MELAKO)
        result = asyncio.run(
            compute_median_ndvi(aoi["bbox"], aoi["feature"]["geometry"], max_area_km2=100.0)
        )
        self.assertEqual(result["status"], "skipped")
        self.assertIn("warning", result)
        # No unbounded fallback product may appear in a skipped result.
        self.assertNotIn("mean", result)
        self.assertNotIn("modis", str(result).lower())


@unittest.skipUnless(AVAILABLE, "Planetary Computer is unreachable")
class RealAdmissionTests(unittest.TestCase):
    def test_synchronous_cap_rejects_the_large_conservancy(self):
        with self.assertRaises(Exception) as raised:
            validate_aoi(MELAKO)
        self.assertEqual(raised.exception.status_code, 413)

    def test_small_conservancy_bbox_illustrates_the_bbox_penalty(self):
        # The cap applies to the bounding box, not the polygon. II Ngwesi is a
        # real 89 km2 conservancy whose irregular outline has a 120 km2 bounding
        # box, so the synchronous path rejects it. An area cap on the polygon
        # would admit it.
        with self.assertRaises(Exception) as raised:
            validate_aoi(II_NGWESI)
        self.assertEqual(raised.exception.status_code, 413)
        self.assertIn("120", str(raised.exception.detail))

    def test_a_compact_area_is_admitted(self):
        compact = {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [37.34, 0.30], [37.40, 0.30], [37.40, 0.37], [37.34, 0.37], [37.34, 0.30],
                ]],
            },
        }
        aoi = validate_aoi(compact)
        self.assertLess(aoi["bbox_area_km2"], 100.0)
        self.assertEqual(len(aoi["bbox"]), 4)

    def test_asset_search_is_bounded(self):
        aoi = _bypass_sync_cap(MELAKO)
        assets = asyncio.run(
            _find_core_assets(aoi["bbox"], need_dem=True, need_landcover=True)
        )
        self.assertIn("dem", assets)
        self.assertIn("landcover", assets)
        for name, hrefs in assets.items():
            self.assertIsInstance(hrefs, list, f"{name} must be a list of tiles")
            self.assertTrue(all(isinstance(h, str) and h.startswith("http") for h in hrefs))
            self.assertLessEqual(len(hrefs), MAX_SOURCE_TILES)


@unittest.skipUnless(AVAILABLE, "Planetary Computer is unreachable")
class RealEndToEndTests(unittest.TestCase):
    def test_request_returns_a_well_formed_summary(self):
        from fastapi.testclient import TestClient

        # The synchronous path only admits small areas, so use a sub-polygon.
        small = {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [37.34, 0.30], [37.40, 0.30], [37.40, 0.37], [37.34, 0.37], [37.34, 0.30],
                ]],
            },
        }
        with TestClient(main.app) as client:
            response = client.post(
                "/generate-context?datasets=dem,landcover", json={"geojson": small}
            )
        self.assertEqual(response.status_code, 200, response.text[:400])
        summary = response.json()["summary"]
        self.assertIn("dem", summary)
        self.assertIn("landcover", summary)
        self.assertNotIn("error", summary["dem"], summary["dem"])
        self.assertIn("terrain_type", summary["dem"])
        self.assertAlmostEqual(
            sum(summary["landcover"]["classes"].values()), 100.0, delta=1.0
        )
        self.assertEqual(summary["analysis"]["datasets"], ["dem", "landcover"])
        self.assertEqual(summary["analysis"]["mode"], "synchronous")

    def test_no_signed_asset_url_reaches_the_client(self):
        from fastapi.testclient import TestClient

        small = {
            "type": "Feature",
            "properties": {},
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [37.34, 0.30], [37.40, 0.30], [37.40, 0.37], [37.34, 0.37], [37.34, 0.30],
                ]],
            },
        }
        with TestClient(main.app) as client:
            response = client.post(
                "/generate-context?datasets=dem,landcover", json={"geojson": small}
            )
        body = response.text
        for marker in ("sig=", "skoid=", "blob.core.windows.net", "se=20"):
            self.assertNotIn(
                marker, body, f"response leaked an asset credential ({marker})"
            )


if __name__ == "__main__":
    unittest.main()
