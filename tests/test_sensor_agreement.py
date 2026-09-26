"""Real-data cross-sensor validation.

The acid test for a multi-sensor index. If the sensors do not land in the same
range over the same ground and season, at least one is being read wrongly — which
a per-sensor "returns a plausible number" test cannot catch, and which is exactly
what happened while building this: a missing scale or offset still yields an NDVI
inside [-1, 1].
"""

import asyncio
import unittest

import main
from sensors import LANDSAT, MODIS, SENTINEL2

# A 60 km2 study area in the semi-arid northern Kenya rangeland, where all three
# collections have coverage.
AOI = {
    "type": "Feature", "properties": {},
    "geometry": {"type": "Polygon", "coordinates": [[
        [35.10, -1.55], [35.17, -1.55], [35.17, -1.48], [35.10, -1.48], [35.10, -1.55]]]},
}


def _network() -> bool:
    try:
        import pystac_client

        client = pystac_client.Client.open(main.STAC_URL)
        for collection in ("sentinel-2-l2a", "landsat-c2-l2", "modis-13Q1-061"):
            client.get_collection(collection)
        return True
    except Exception:
        return False


AVAILABLE = _network()


@unittest.skipUnless(AVAILABLE, "Planetary Computer is unreachable")
class CrossSensorTests(unittest.TestCase):
    """Each sensor independently, then compared against the others."""

    @classmethod
    def setUpClass(cls):
        feature = main.validate_aoi(AOI)["feature"]
        cls.results = {}
        for sensor in (SENTINEL2, LANDSAT, MODIS):
            cls.results[sensor.id] = asyncio.run(main.compute_vegetation_index(
                [35.10, -1.55, 35.17, -1.48], feature["geometry"],
                max_area_km2=1e9,
                max_scenes=sensor.default_max_scenes,
                resolution_m=20,
                sensor_id=sensor.id,
            ))

    def _ok(self, sensor_id):
        result = self.results.get(sensor_id, {})
        self.assertEqual(result.get("status"), "ok",
                         f"{sensor_id} did not return a result: {result}")
        return result

    def test_every_sensor_returns_a_vegetation_index(self):
        for sensor_id in ("sentinel-2", "landsat", "modis"):
            with self.subTest(sensor=sensor_id):
                result = self._ok(sensor_id)
                for key in ("mean", "min", "max", "std", "p25", "p75"):
                    self.assertTrue(-1.0 <= result[key] <= 1.0, f"{sensor_id} {key}={result[key]}")
                self.assertLessEqual(result["p25"], result["p75"])
                self.assertGreaterEqual(result["scene_count"], 1)
                self.assertGreater(result["valid_pixel_count"], 0)

    def test_each_result_names_the_sensor_that_produced_it(self):
        for sensor_id, result in self.results.items():
            if result.get("status") != "ok":
                continue
            with self.subTest(sensor=sensor_id):
                self.assertEqual(result["sensor"]["id"], sensor_id)
                self.assertTrue(result["sensor"]["collection"])

    def test_modis_is_reported_as_a_product_not_a_computation(self):
        result = self._ok("modis")
        self.assertFalse(result["sensor"].get("ndvi", "").startswith("computed"))
        self.assertIn("product", result["method"])

    def test_the_authoritative_masks_agree_with_each_other(self):
        """MODIS is masked by NASA and Landsat by QA_PIXEL; they must agree.

        This is the check that catches a mis-scaled sensor: skipping Landsat's
        offset moved one real scene's NDVI by 73%, and it would still have landed
        inside [-1, 1].
        """
        modis = self._ok("modis")["mean"]
        landsat = self._ok("landsat")["mean"]
        difference = abs(modis - landsat)
        self.assertLess(
            difference, 0.20,
            f"MODIS {modis:+.3f} and Landsat {landsat:+.3f} disagree by {difference:.3f}; "
            "one of them is probably mis-scaled or mis-masked",
        )

    def test_the_relaxed_sentinel2_mask_reads_low_in_cloudy_conditions(self):
        """Sentinel-2 keeps SCL classes 4 and 5, so it includes cloud.

        Measured over this study area and window: MODIS 0.418, Landsat 0.357,
        Sentinel-2 0.161. That is the cost of the relaxation, and it is why Landsat
        is the automatic choice. The test records the direction rather than a bare
        number so a future change to the mask has to be deliberate.
        """
        sentinel = self._ok("sentinel-2")["mean"]
        landsat = self._ok("landsat")["mean"]
        self.assertLess(
            sentinel, landsat,
            "Sentinel-2 now reads at or above the QA-masked Landsat; the SCL "
            "relaxation may have been tightened, which is fine but should be "
            "reconsidered deliberately",
        )

    def test_bright_desert_is_not_discarded_as_cloud(self):
        """The reason classes 4 and 5 are kept at all.

        A Sahara tile reads 100% class 5 while its B04/B08 reflectance and NDVI are
        plainly desert. Rejecting those classes empties the result for exactly the
        semi-arid ground this service is built for.
        """
        desert = {
            "type": "Feature", "properties": {},
            "geometry": {"type": "Polygon", "coordinates": [[
                [30.00, 25.00], [30.08, 25.00], [30.08, 25.08], [30.00, 25.08], [30.00, 25.00]]]},
        }
        canonical = main.canonicalize_geojson(desert)
        bbox = main.aoi_bbox(canonical)
        result = asyncio.run(main.compute_vegetation_index(
            bbox, canonical["geometry"], max_area_km2=1e9, resolution_m=20,
            sensor_id="sentinel-2",
        ))
        self.assertEqual(result.get("status"), "ok", result)
        self.assertGreater(result["valid_pixel_fraction"], 0.5)
        # Desert is sparse, not lush.
        self.assertLess(result["mean"], 0.4, result)

    def test_a_pre_sentinel2_window_is_served_by_landsat(self):
        """The historical question the conservancy audience actually asked."""
        feature = main.validate_aoi(AOI)["feature"]
        result = asyncio.run(main.compute_vegetation_index(
            [35.10, -1.55, 35.17, -1.48], feature["geometry"],
            max_area_km2=1e9, resolution_m=20, sensor_id="auto",
            start="1997-01-01", end="1999-12-31",
        ))
        self.assertEqual(result.get("status"), "ok", result)
        self.assertEqual(result["sensor"]["id"], "landsat")
        self.assertTrue(-1.0 <= result["mean"] <= 1.0)
        self.assertEqual(result["window"], {"start": "1997-01-01", "end": "1999-12-31"})


if __name__ == "__main__":
    unittest.main()
