"""The interactive scene search is STAC-only and runs off the event loop.

The outage came from reading the Landsat GeoParquet index synchronously on the
event loop. These tests pin both halves of the fix: the GeoParquet scene lookup
is not called at all, and the blocking STAC search is handed to a thread and
bounded by ASSET_SEARCH_SECONDS.
"""

import asyncio
import datetime
import unittest
from types import SimpleNamespace
from unittest import mock

import main
import pc_geoparquet as pg
import sensors


def _item() -> SimpleNamespace:
    return SimpleNamespace(
        id="scene-1",
        datetime=datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc),
        properties={"eo:cloud_cover": 5.0},
    )


class SceneSearchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.sensor, _ = sensors.select_sensor(
            "sentinel-2", window_start="2025-11-11", bbox_area_km2=10)

    async def test_scene_search_uses_stac_and_never_the_geoparquet_scene_index(self):
        item = _item()
        catalog = mock.Mock()
        catalog.search.return_value.items.return_value = [item]
        calls = []

        async def fake_to_thread(fn, *args, **kwargs):
            calls.append(fn)
            return fn(*args, **kwargs)

        with mock.patch.object(main.pystac_client.Client, "open",
                               return_value=catalog), \
                mock.patch.object(pg, "scene_assets") as scene_assets, \
                mock.patch.object(main.asyncio, "to_thread", fake_to_thread):
            items = await main._search_items(
                [36.0, -1.0, 38.0, 1.0], self.sensor, 2,
                "2025-11-11", "2026-11-11")

        self.assertEqual(items, [item])
        scene_assets.assert_not_called()
        self.assertEqual(len(calls), 1,
                         "the blocking STAC search must run in a thread")

    async def test_scene_search_is_bounded_by_the_search_timeout(self):
        async def slow_to_thread(fn, *args, **kwargs):
            await asyncio.sleep(1)
            return []

        with mock.patch.object(main, "ASSET_SEARCH_SECONDS", 0.05), \
                mock.patch.object(main.asyncio, "to_thread", slow_to_thread):
            started = asyncio.get_running_loop().time()
            items = await main._search_items(
                [36.0, -1.0, 38.0, 1.0], self.sensor, 2,
                "2025-11-11", "2026-11-11")
            elapsed = asyncio.get_running_loop().time() - started

        self.assertEqual(items, [])
        self.assertLess(elapsed, 1.0, "a hung search must not exceed its budget")


if __name__ == "__main__":
    unittest.main()
