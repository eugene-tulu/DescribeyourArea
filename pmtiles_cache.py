"""Low-zoom PMTiles cache for large AOIs.

At low zoom a live COG read fans out across many source tiles; the map only
needs a small image. This module pre-bakes the AOI's low-zoom tiles (using the
same rio-tiler mask/render as the live endpoint) into a single PMTiles archive.
The tile endpoint serves from the archive at low zoom and falls back to the COG
for high zoom, where the reader wants real pixels.

The PMTiles file lives under the writable usage volume, one per area/indicator
(and per land-cover-history year), and is built lazily on the first request.
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Callable, Iterable, Optional

from morecantile import tms

_LOCKS: dict[str, threading.Lock] = {}
_GUARD = threading.Lock()


def _lock(key: str) -> threading.Lock:
    with _GUARD:
        return _LOCKS.setdefault(key, threading.Lock())


def path_for(key: str, indicator: str, year: Optional[int] = None) -> Path:
    suffix = f"{indicator}:{year}" if year else indicator
    return Path(os.environ.get("PROJECTS_DIR", "/app/.usage/projects")) / "tiles" / key / f"{suffix}.pmtiles"


def _tiles_for_bbox(bbox: list[float], z: int) -> Iterable[tuple[int, int]]:
    tms_ = tms.get("WebMercatorQuad")
    north_west = tms_.tile(bbox[0], bbox[3], z)
    south_east = tms_.tile(bbox[2], bbox[1], z)
    for x in range(north_west.x, south_east.x + 1):
        for y in range(north_west.y, south_east.y + 1):
            yield x, y


def get_or_build(
    key: str,
    indicator: str,
    *,
    bbox: list[float],
    geometry: dict,
    hrefs: list[str],
    render: Callable[..., bytes],
    year: Optional[int] = None,
    low_zoom_max: int = 9,
) -> Path:
    path = path_for(key, indicator, year)
    if path.exists():
        return path
    with _lock(f"{key}:{indicator}:{year}"):
        if path.exists():
            return path
        import shutil
        import tempfile

        from pmtiles.convert import disk_to_pmtiles

        path.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix="pmtiles-", dir=path.parent))
        try:
            (staging / "metadata.json").write_text(json.dumps({
                "name": f"{indicator} {key}",
                "format": "png",
                "bounds": [bbox[0], bbox[1], bbox[2], bbox[3]],
                "minzoom": 0,
                "maxzoom": low_zoom_max,
            }))
            for z in range(low_zoom_max + 1):
                for x, y in _tiles_for_bbox(bbox, z):
                    data = render(hrefs, indicator, z, x, y, geometry)
                    if not data:
                        continue
                    tile_dir = staging / str(z) / str(x)
                    tile_dir.mkdir(parents=True, exist_ok=True)
                    (tile_dir / f"{y}.png").write_bytes(data)
            disk_to_pmtiles(str(staging), str(path), low_zoom_max)
        finally:
            shutil.rmtree(staging, ignore_errors=True)
    return path


def get_tile(path: Path, z: int, x: int, y: int) -> Optional[bytes]:
    from pmtiles.reader import Reader

    with open(path, "rb") as stream:
        def get_bytes(offset: int, size: int) -> bytes:
            stream.seek(offset)
            return stream.read(size)

        reader = Reader(get_bytes)
        return reader.get(z, x, y)