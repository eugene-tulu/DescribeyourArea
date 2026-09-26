"""Tests for the rainfall lookup path.

A cache lookup is not a raster job, and treating it as one made the published
portfolio unreachable for the areas it was built for: the synchronous admission
policy rejected 8 of the 21 NRT conservancies on payload size and the rest on
vertex count, before a single cached value was consulted.

Geometries are built with :func:`polygon` rather than nested literals, so there
are no brackets to miscount.
"""

import json
import math
import unittest

import main
import rainfall


def polygon(points, close=True):
    ring = [list(map(float, p)) for p in points]
    if close and ring[0] != ring[-1]:
        ring.append(list(ring[0]))
    return {"type": "Polygon", "coordinates": [ring]}


def feature(geometry, **properties):
    return {"type": "Feature", "properties": properties, "geometry": geometry}


def box_ring(minx, miny, maxx, maxy, steps=1):
    """A rectangle, optionally subdivided into `steps` vertices per edge."""
    ring = []
    for index in range(steps):
        ring.append((minx + (maxx - minx) * index / steps, miny))
    for index in range(steps):
        ring.append((maxx, miny + (maxy - miny) * index / steps))
    for index in range(steps):
        ring.append((maxx - (maxx - minx) * index / steps, maxy))
    for index in range(steps):
        ring.append((minx, maxy - (maxy - miny) * index / steps))
    return ring


# A bow tie: the shape two of the 21 published NRT conservancies actually have.
BOWTIE = polygon([
    (35.10, -1.55), (35.20, -1.45), (35.20, -1.55), (35.10, -1.45), (35.10, -1.55),
])
CLEAN = polygon(box_ring(35.10, -1.55, 35.17, -1.48))
HUGE = polygon(box_ring(30.0, 20.0, 40.0, 30.0))


class LookupAdmissionTests(unittest.TestCase):
    def test_a_lookup_skips_the_raster_area_cap(self):
        # A 12,000 km2 bounding box is far past the synchronous limit, and fine
        # for a cache read that costs a millisecond and about 1 KB.
        area = main.validate_for_lookup(HUGE)
        self.assertGreater(area["bbox_area_km2"], main.MAX_SYNC_BBOX_KM2)
        with self.assertRaises(Exception):
            main.validate_aoi(HUGE)

    def test_a_lookup_tolerates_a_vertex_count_the_raster_path_rejects(self):
        # 60,000 vertices: measured worst case among the NRT conservancies is
        # 61,035, and the raster cap is 10,000.
        dense = polygon(box_ring(35.10, -1.55, 35.17, -1.48, steps=15_000))
        self.assertGreater(15_000 * 4, main.MAX_AOI_VERTICES)
        self.assertIsNotNone(main.validate_for_lookup(dense))
        with self.assertRaises(Exception):
            main.validate_aoi(dense)

    def test_the_lookup_byte_cap_is_well_above_the_worst_real_case(self):
        # Measured on the 21 NRT conservancies: the largest is 1,358 KB, and
        # 8 of 21 exceed the 500 KB raster cap.
        self.assertGreater(main.MAX_LOOKUP_BYTES, 1_358 * 1024)
        self.assertGreater(main.MAX_LOOKUP_BYTES, main.MAX_GEOJSON_BYTES)

    def test_the_payload_cap_still_applies(self):
        blob = json.dumps(feature(HUGE, pad="x" * 4096)).encode()
        oversize = {"type": "FeatureCollection", "features": [
            feature(CLEAN, pad="x" * 4096) for _ in range(2_000)
        ]}
        self.assertGreater(len(json.dumps(oversize).encode()), main.MAX_LOOKUP_BYTES)
        with self.assertRaises(Exception) as raised:
            main.validate_for_lookup(oversize)
        self.assertEqual(raised.exception.status_code, 413)
        self.assertTrue(blob)


class GeometryRepairTests(unittest.TestCase):
    """A ring self-intersection is a defect in someone else's file, not a reason
    to refuse a whole conservancy. Two of the 21 published areas have them."""

    def test_a_bowtie_is_repaired_and_reported(self):
        from shapely.geometry import shape

        canonical = main.canonicalize_geojson(BOWTIE)
        self.assertIn("geometry_repaired", canonical["properties"])
        self.assertTrue(shape(canonical["geometry"]).is_valid)

    def test_a_clean_geometry_is_not_reported_as_repaired(self):
        canonical = main.canonicalize_geojson(CLEAN)
        self.assertNotIn("geometry_repaired", canonical["properties"])

    def test_the_repair_is_identical_on_both_paths(self):
        """Otherwise the build hashes raw geometry and the lookup hashes the
        repair, and the published series is unreachable."""
        from_geometry = main.canonicalize_geojson(BOWTIE)
        from_feature = main.canonicalize_geojson(feature(BOWTIE))
        self.assertEqual(rainfall.geometry_hash(from_geometry["geometry"]),
                         rainfall.geometry_hash(from_feature["geometry"]))

    def test_the_repair_yields_a_usable_polygon(self):
        # A self-intersecting ring has no meaningful signed area, so there is
        # nothing to compare against; the assertion is that the result is a valid
        # polygon with positive area rather than a degenerate remnant.
        from shapely.geometry import shape

        self.assertEqual(shape(BOWTIE).area, 0.0, "a bow tie has no signed area")
        repaired = shape(main.canonicalize_geojson(BOWTIE)["geometry"])
        self.assertTrue(repaired.is_valid)
        self.assertFalse(repaired.is_empty)
        self.assertGreater(repaired.area, 0.0)

    def test_a_genuinely_degenerate_geometry_is_still_refused(self):
        for geometry in ({"type": "Polygon", "coordinates": []},
                         {"type": "Polygon", "coordinates": [[[35.1, -1.5]]]}):
            with self.subTest(geometry=geometry):
                with self.assertRaises(Exception):
                    main.canonicalize_geojson(geometry)


class LookupEndpointTests(unittest.TestCase):
    def setUp(self):
        import os
        import tempfile

        self.previous = os.environ.get("RAINFALL_CACHE_DIR")
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["RAINFALL_CACHE_DIR"] = self.tmp.name
        from fastapi.testclient import TestClient

        self.client = TestClient(main.app)
        self.addCleanup(self._restore)
        self.addCleanup(self.tmp.cleanup)

    def _restore(self):
        import os

        if self.previous is None:
            os.environ.pop("RAINFALL_CACHE_DIR", None)
        else:
            os.environ["RAINFALL_CACHE_DIR"] = self.previous

    def test_a_miss_is_reported_with_the_key_and_a_reason(self):
        response = self.client.post("/rainfall", json={"geojson": CLEAN})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["rainfall"]["status"], "not_computed")
        self.assertEqual(body["rainfall"]["reason"], "no_precomputed_series")
        self.assertEqual(len(body["cache_key"]), 32)

    def test_a_cached_area_is_served(self):
        key = rainfall.geometry_hash(CLEAN)
        rainfall.write_cache(key, {
            "processing_version": rainfall.RAINFALL_PROCESSING_VERSION,
            "indicator": "monthly_precipitation",
            "series": [{"month": "2020-01", "precip_mm": 3.0}],
        })
        body = self.client.post("/rainfall", json={"geojson": CLEAN}).json()
        self.assertEqual(body["rainfall"]["status"], "ok")
        self.assertEqual(body["cache_key"], key)

    def test_a_caller_with_the_key_never_resends_a_polygon(self):
        key = rainfall.geometry_hash(CLEAN)
        rainfall.write_cache(key, {
            "processing_version": rainfall.RAINFALL_PROCESSING_VERSION,
            "series": [{"month": "2020-01", "precip_mm": 3.0}],
        })
        body = self.client.post("/rainfall", json={"cache_key": key}).json()
        self.assertEqual(body["rainfall"]["status"], "ok")
        # No geometry was supplied, so no area can be reported.
        self.assertIsNone(body["analysis"]["bbox_area_km2"])

    def test_a_malformed_key_is_refused(self):
        for key in ("nope", "A" * 32, "0" * 31, "", "../../etc"):
            with self.subTest(key=key):
                self.assertEqual(
                    self.client.post("/rainfall", json={"cache_key": key}).status_code, 422
                )

    def test_an_empty_request_is_refused(self):
        self.assertEqual(self.client.post("/rainfall", json={}).status_code, 422)

    def test_a_non_polygon_is_refused(self):
        response = self.client.post(
            "/rainfall", json={"geojson": {"type": "Point", "coordinates": [0, 0]}}
        )
        self.assertEqual(response.status_code, 400)

    def test_the_raster_endpoint_still_applies_its_own_caps(self):
        # The separation is the point: a lookup is cheap, a raster request is not.
        response = self.client.post("/generate-context?datasets=dem", json={"geojson": HUGE})
        self.assertEqual(response.status_code, 413)


class FetchReliabilityTests(unittest.TestCase):
    """A momentary network blip must not look like "this area has no series"."""

    def _with_remote(self, stub, call):
        import os
        import tempfile

        previous_remote = os.environ.get(rainfall.REMOTE_URI_ENV)
        previous_cache = os.environ.get("RAINFALL_CACHE_DIR")
        original = rainfall._s3_client
        with tempfile.TemporaryDirectory() as tmp:
            os.environ[rainfall.REMOTE_URI_ENV] = "s3://b/p"
            os.environ["RAINFALL_CACHE_DIR"] = tmp
            rainfall._s3_client = lambda: stub
            try:
                return call()
            finally:
                rainfall._s3_client = original
                for key, value in ((rainfall.REMOTE_URI_ENV, previous_remote),
                                   ("RAINFALL_CACHE_DIR", previous_cache)):
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value

    def test_a_transient_failure_is_retried(self):
        attempts = {"n": 0}
        body = json.dumps({"processing_version": rainfall.RAINFALL_PROCESSING_VERSION,
                           "series": []}).encode()

        class Flaky:
            def get_object(self, **_kw):
                attempts["n"] += 1
                if attempts["n"] < 2:
                    raise ConnectionError("connection reset by peer")
                return {"Body": type("B", (), {"read": staticmethod(lambda: body)})()}

        result = self._with_remote(Flaky(), lambda: rainfall.fetch("c" * 32))
        self.assertIsNotNone(result, "a retryable failure was believed to be a miss")
        self.assertGreaterEqual(attempts["n"], 2)

    def test_a_genuine_miss_is_not_retried(self):
        calls = {"n": 0}

        class Missing:
            def get_object(self, **_kw):
                calls["n"] += 1
                raise KeyError("NoSuchKey")

        result = self._with_remote(Missing(), lambda: rainfall.fetch("d" * 32))
        self.assertIsNone(result)
        self.assertEqual(calls["n"], 1, "a genuine miss should not be retried")

    def test_an_s3_style_no_such_key_is_treated_as_a_miss(self):
        class S3Missing(Exception):
            def __init__(self):
                self.response = {"Error": {"Code": "NoSuchKey"}}
                super().__init__("NoSuchKey")

        calls = {"n": 0}

        class Missing:
            def get_object(self, **_kw):
                calls["n"] += 1
                raise S3Missing()

        self.assertIsNone(self._with_remote(Missing(), lambda: rainfall.fetch("e" * 32)))
        self.assertEqual(calls["n"], 1)


if __name__ == "__main__":
    unittest.main()
