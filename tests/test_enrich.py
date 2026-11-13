import os
import tempfile
import unittest

import numpy as np
import rasterio
from rasterio.transform import from_bounds

from contract import EnrichPoint, EnrichRequest
from enrich import enrich_records, sample_point


def _write_raster(directory: str, name: str, value: float, nodata: float | None) -> str:
    path = os.path.join(directory, name)
    with rasterio.open(
        path, "w", driver="GTiff", width=10, height=10, count=1,
        dtype="float32", crs="EPSG:4326",
        transform=from_bounds(35.0, -2.0, 36.0, -1.0, 10, 10),
        nodata=nodata,
    ) as dst:
        dst.write(np.full((10, 10), value, dtype="float32"), 1)
        if nodata is not None:
            # Top-left pixel is the nodata corner.
            dst.write(np.full((1, 1), nodata, dtype="float32"), 1, window=((0, 1), (0, 1)))
    return path


class SamplePointTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dem = _write_raster(self.tmp.name, "dem.tif", 1800.0, -9999.0)
        self.landcover = _write_raster(self.tmp.name, "lc.tif", 11.0, None)

    def tearDown(self):
        self.tmp.cleanup()

    def test_reads_a_value_at_the_point(self):
        self.assertEqual(sample_point([self.dem], 35.5, -1.5), 1800.0)
        self.assertEqual(sample_point([self.landcover], 35.5, -1.5), 11.0)

    def test_nodata_is_no_sample(self):
        self.assertIsNone(sample_point([self.dem], 35.05, -1.05))

    def test_a_missing_tile_is_no_sample(self):
        self.assertIsNone(sample_point(["/does/not/exist.tif"], 35.5, -1.5))


class EnrichRecordsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dem = _write_raster(self.tmp.name, "dem.tif", 1800.0, -9999.0)
        self.landcover = _write_raster(self.tmp.name, "lc.tif", 11.0, None)

    def tearDown(self):
        self.tmp.cleanup()

    def _request(self, observations, rainfall=None):
        return EnrichRequest(
            observations=observations,
            dem_tiles=[self.dem],
            landcover_tiles=[self.landcover],
            ndvi_mean=0.4,
            landcover_dominant="Rangeland",
            rainfall_series=rainfall if rainfall is not None else [
                {"month": "2026-06", "precip_mm": 120}
            ],
            bbox=[35.0, -2.0, 36.0, -1.0],
        )

    def test_joins_evidence_and_leaves_no_flags(self):
        records = enrich_records(self._request([
            EnrichPoint(id="P-1", lon=35.5, lat=-1.5, timestamp="2026-06-15"),
        ]))
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record.elevation_m, 1800.0)
        self.assertEqual(record.landcover, "Rangeland")
        self.assertEqual(record.landcover_code, 11)
        self.assertEqual(record.rainfall_mm, 120.0)
        self.assertEqual(record.ndvi, 0.4)
        self.assertEqual(record.flags, [])

    def test_flags_a_month_without_rainfall(self):
        records = enrich_records(self._request([
            EnrichPoint(id="P-2", lon=35.5, lat=-1.5, timestamp="2026-04-01"),
        ]))
        self.assertIsNone(records[0].rainfall_mm)
        self.assertIn("no rainfall for month", records[0].flags)

    def test_flags_a_point_outside_the_bbox(self):
        records = enrich_records(self._request([
            EnrichPoint(id="P-3", lon=40.0, lat=-1.5, timestamp="2026-06-15"),
        ]))
        self.assertIn("outside AOI bbox", records[0].flags)

    def test_flags_a_missing_timestamp(self):
        records = enrich_records(self._request([
            EnrichPoint(id="P-4", lon=35.5, lat=-1.5, timestamp=None),
        ]))
        self.assertIn("no timestamp", records[0].flags)


if __name__ == "__main__":
    unittest.main()