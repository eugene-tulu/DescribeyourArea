"""Unit tests for pc_geoparquet's scene/tile asset lookups.

The index path was written against MODIS monthly parts (month_assets). These
pin the two column-filtered siblings -- scene_assets for temporal scene
collections (Sentinel-2, Landsat) and tile_assets for static tile collections
(Copernicus DEM, Esri land cover) -- through a hand-built Parquet file so the
bbox/cloud-cover/date filtering is checked without the network.
"""

import datetime
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import pandas as pd
import pc_geoparquet as pg


def _write_part(tmp: Path, rows: list[dict]) -> Path:
    path = tmp / "part.parquet"
    df = pd.DataFrame(rows)
    df.to_parquet(path, index=False, engine="pyarrow")
    return path


def _parts(rows: list[dict]):
    tmp = tempfile.mkdtemp()
    path = _write_part(Path(tmp), rows)
    return [path]


class SceneAssetsTests(unittest.TestCase):
    BBOX = [36.0, -1.0, 38.0, 1.0]
    ROWS = [
        {"id": "s1", "start_datetime": datetime.datetime(2025, 1, 15, 12, 0, tzinfo=datetime.timezone.utc),
         "datetime": datetime.datetime(2025, 1, 15, 12, 0, tzinfo=datetime.timezone.utc),
         "bbox.xmin": 35.0, "bbox.ymin": -2.0, "bbox.xmax": 39.0, "bbox.ymax": 2.0,
         "properties.eo:cloud_cover": 5.0,
         "assets.B04.href": "https://example.invalid/s1-B04.tif",
         "assets.B08.href": "https://example.invalid/s1-B08.tif",
         "assets.SCL.href": "https://example.invalid/s1-SCL.tif"},
        {"id": "s2", "start_datetime": datetime.datetime(2025, 2, 15, 12, 0, tzinfo=datetime.timezone.utc),
         "datetime": datetime.datetime(2025, 2, 15, 12, 0, tzinfo=datetime.timezone.utc),
         "bbox.xmin": 35.0, "bbox.ymin": -2.0, "bbox.xmax": 39.0, "bbox.ymax": 2.0,
         "properties.eo:cloud_cover": 95.0,
         "assets.B04.href": "https://example.invalid/s2-B04.tif",
         "assets.B08.href": "https://example.invalid/s2-B08.tif",
         "assets.SCL.href": "https://example.invalid/s2-SCL.tif"},
        # outside bbox
        {"id": "s3", "start_datetime": datetime.datetime(2025, 1, 15, 12, 0, tzinfo=datetime.timezone.utc),
         "datetime": datetime.datetime(2025, 1, 15, 12, 0, tzinfo=datetime.timezone.utc),
         "bbox.xmin": 40.0, "bbox.ymin": 5.0, "bbox.xmax": 41.0, "bbox.ymax": 6.0,
         "properties.eo:cloud_cover": 1.0,
         "assets.B04.href": "https://example.invalid/s3-B04.tif"},
        # no assets at all
        {"id": "s4", "start_datetime": datetime.datetime(2025, 1, 15, 12, 0, tzinfo=datetime.timezone.utc),
         "datetime": datetime.datetime(2025, 1, 15, 12, 0, tzinfo=datetime.timezone.utc),
         "bbox.xmin": 35.0, "bbox.ymin": -2.0, "bbox.xmax": 39.0, "bbox.ymax": 2.0,
         "properties.eo:cloud_cover": 2.0,
         "assets.B04.href": None, "assets.B08.href": None, "assets.SCL.href": None},
    ]

    def _scene(self, **kw):
        with mock.patch.object(pg, "ensure_index", return_value=_parts(self.ROWS)):
            return pg.scene_assets(
                "sentinel-2-l2a", self.BBOX, "2025-01-01", "2025-12-31",
                assets=["B04", "B08", "SCL"], **kw)

    def test_keeps_only_intersecting_scenes_with_assets(self):
        rows = self._scene()
        ids = sorted(r["id"] for r in rows)
        # s3 is disjoint, s4 has no hrefs; s1 and s2 remain before cloud filter.
        self.assertIn("s1", ids)
        self.assertIn("s2", ids)
        self.assertNotIn("s3", ids)
        self.assertNotIn("s4", ids)

    def test_cloud_filter_drops_cloudy_scene(self):
        rows = self._scene(cloud_cover_max=30.0)
        ids = [r["id"] for r in rows]
        self.assertIn("s1", ids)
        self.assertNotIn("s2", ids)

    def test_assets_are_carried_through(self):
        rows = self._scene()
        s1 = next(r for r in rows if r["id"] == "s1")
        self.assertIn("B04", s1["assets"])
        self.assertIn("B08", s1["assets"])
        self.assertIn("SCL", s1["assets"])

    def test_no_index_returns_empty_for_fallback(self):
        with mock.patch.object(pg, "ensure_index", return_value=[]):
            rows = pg.scene_assets("sentinel-2-l2a", self.BBOX, "2025-01-01", "2025-12-31",
                                   assets=["B04"])
        self.assertEqual(rows, [])


class SnapshotCacheTests(unittest.TestCase):
    """Planetary Computer republishes a partition with a new cumulative name.

    The active partition (part-044) arrives as 25 files whose end dates grow
    through the year. Treating each as a separate part made one lookup read
    6.66 GB. These pin the dedupe and the cleanup that prevents it.
    """

    SPECS = [
        ("landsat-c2-l2.parquet/part-044_2026-01-01T00_00_00_2026-06-02T00_00_00.parquet",
         "2026-01", "2026-06"),
        ("landsat-c2-l2.parquet/part-044_2026-01-01T00_00_00_2026-11-03T00_00_00.parquet",
         "2026-01", "2026-11"),
        ("landsat-c2-l2.parquet/part-043_2025-01-01T00_00_00_2025-12-31T00_00_00.parquet",
         "2025-01", "2025-12"),
    ]

    def test_keeps_only_the_latest_snapshot_per_logical_part(self):
        deduped = pg._dedupe_specs(self.SPECS)
        names = {name for name, _first, _last in deduped}
        self.assertEqual(len(deduped), 2)
        self.assertIn(self.SPECS[1][0], names)
        self.assertNotIn(self.SPECS[0][0], names)
        self.assertIn(self.SPECS[2][0], names)

    def test_ensure_index_returns_one_part_and_deletes_the_superseded_snapshot(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "landsat-c2-l2-part-044_2026-06-02.parquet"
            new = root / "landsat-c2-l2-part-044_2026-11-03.parquet"
            old.write_bytes(b"old")
            new.write_bytes(b"new")
            specs = [
                ("landsat-c2-l2.parquet/part-044_2026-06-02.parquet", "2026-01", "2026-06"),
                ("landsat-c2-l2.parquet/part-044_2026-11-03.parquet", "2026-01", "2026-11"),
            ]
            with mock.patch.object(pg, "index_dir", return_value=root), \
                    mock.patch.object(pg, "_cached_specs",
                                      return_value=pg._dedupe_specs(specs)), \
                    mock.patch.object(pg, "_collection_blob",
                                      return_value=("items", "landsat-c2-l2.parquet")), \
                    mock.patch.object(pg, "_part_path",
                                      side_effect=lambda c, n: root / f"{c}-{n}"):
                paths = pg.ensure_index("landsat-c2-l2", "2025-11-11", "2026-11-11")
            self.assertEqual([p.name for p in paths],
                             ["landsat-c2-l2-part-044_2026-11-03.parquet"])
            self.assertFalse(old.exists())
            self.assertTrue(new.exists())

    def test_prune_scene_indexes_removes_only_scene_collections(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            landsat = root / "landsat-c2-l2-part-044.parquet"
            sentinel = root / "sentinel-2-l2a-part-001.parquet"
            modis = root / "modis-13Q1-061-part-026.parquet"
            parts = root / "landsat-c2-l2-parts.json"
            for path in (landsat, sentinel, modis, parts):
                path.write_bytes(b"x")
            with mock.patch.object(pg, "index_dir", return_value=root):
                result = pg.prune_scene_indexes()
            self.assertFalse(landsat.exists())
            self.assertFalse(sentinel.exists())
            self.assertFalse(parts.exists())
            self.assertTrue(modis.exists())
            self.assertEqual(result["files"], 3)

    def test_cache_cap_evicts_the_oldest_part_but_keeps_the_newest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = root / "modis-old.parquet"
            new = root / "modis-new.parquet"
            old.write_bytes(b"1234")
            new.write_bytes(b"5678")
            os.utime(old, (1, 1))
            os.utime(new, (2, 2))
            with mock.patch.object(pg, "index_dir", return_value=root), \
                    mock.patch.dict(os.environ, {"PC_INDEX_MAX_BYTES": "4"}):
                pg._enforce_cache_cap()
            self.assertFalse(old.exists())
            self.assertTrue(new.exists())


class TileAssetsTests(unittest.TestCase):
    BBOX = [36.0, -1.0, 38.0, 1.0]
    ROWS = [
        {"bbox.xmin": 35.0, "bbox.ymin": -2.0, "bbox.xmax": 37.0, "bbox.ymax": 0.0,
         "assets.elevation.href": "https://example.invalid/tile-1.tif"},
        {"bbox.xmin": 37.0, "bbox.ymin": 0.0, "bbox.xmax": 39.0, "bbox.ymax": 2.0,
         "assets.elevation.href": "https://example.invalid/tile-2.tif"},
        {"bbox.xmin": 40.0, "bbox.ymin": 5.0, "bbox.xmax": 41.0, "bbox.ymax": 6.0,
         "assets.elevation.href": "https://example.invalid/tile-3.tif"},
    ]

    def _tile(self, **kw):
        with mock.patch.object(pg, "ensure_index", return_value=_parts(self.ROWS)):
            return pg.tile_assets("cop-dem-glo-30", self.BBOX, "elevation", **kw)

    def test_returns_intersecting_tiles_only(self):
        hrefs = self._tile()
        self.assertIn("https://example.invalid/tile-1.tif", hrefs)
        self.assertIn("https://example.invalid/tile-2.tif", hrefs)
        self.assertNotIn("https://example.invalid/tile-3.tif", hrefs)

    def test_limit_bounds_the_result(self):
        hrefs = self._tile(limit=1)
        self.assertEqual(len(hrefs), 1)

    def test_no_index_returns_empty_for_fallback(self):
        with mock.patch.object(pg, "ensure_index", return_value=[]):
            hrefs = pg.tile_assets("cop-dem-glo-30", self.BBOX, "elevation")
        self.assertEqual(hrefs, [])


if __name__ == "__main__":
    unittest.main()
