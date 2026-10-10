"""STAC-GeoParquet as the index, with only the index cached.

Planetary Computer publishes one GeoParquet part per collection-year, each an
unsigned snapshot of that year's STAC items. Reading it replaces a search per
month against the STAC API -- which for a twelve-year baseline is a hundred and
twelve searches for one reading, against a shared public service that has reset
a connection on this app before.

Only the INDEX is cached, never the imagery. A year's part is about 6 MB of item
metadata, so once a year has been indexed it is read from disk and the second
reading of any area covering those months does no searching at all. That is the
whole of the trade: pay a few megabytes once per collection-year, then query
metadata locally. The cache holds what the parts contain -- item metadata. The
pixels are still read, once a month, from the COGs the parts point at.

`fastparquet` is the reader rather than `pyarrow` for two reasons: it flattens
the nested structs into dotted columns, so `bbox.xmin` and
`assets.<asset>.href` are plain columns and nothing has to be parsed out of a
JSON blob; and its wheel is 2 MB where pyarrow's is 54 MB, in an image that
already carries GDAL.
"""

from __future__ import annotations

import os
import re
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

import rainfall

# The account that hosts the published snapshots, named by each collection's
# `geoparquet-items` asset in its `table:storage_options` block.
ITEMS_ACCOUNT = os.getenv("PC_ITEMS_ACCOUNT", "pcstacitems")
INDEX_CACHE_DIRNAME = "stac-geoparquet"
_UA = "geocontextualize"

_ROOT_CACHE: dict[str, str] = {}


def index_dir() -> Path:
    """Where the cached index parts live. Under the cache dir, so it persists."""
    path = rainfall.cache_dir() / INDEX_CACHE_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def _part_path(collection: str, name: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in name)
    return index_dir() / f"{collection}-{safe}"


def _signed_root(collection: str) -> str:
    """The collection's published snapshot root, signed, resolved once per process."""
    cached = _ROOT_CACHE.get(collection)
    if cached:
        return cached
    import json

    url = (f"https://planetarycomputer.microsoft.com/api/stac/v1/collections/{collection}")
    with urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": _UA}), timeout=60) as response:
        document = json.loads(response.read())
    href = ((document.get("assets") or {}).get("geoparquet-items") or {}).get("href") or ""
    if not href:
        raise RuntimeError(f"{collection} publishes no geoparquet-items asset")
    # ``abfs://items/<collection>.parquet`` names the dataset root: the path after
    # ``://`` is container, then blob. An abfs path is not a URL, and the signer
    # refuses it, so it is rebuilt as one against the items account.
    if href.startswith("abfs://"):
        _, _, rest = href.partition("://")
        container, _, blob = rest.partition("/")
        href = f"https://{ITEMS_ACCOUNT}.blob.core.windows.net/{container}/{blob}"

    import planetary_computer

    signed = planetary_computer.sign(href)
    _ROOT_CACHE[collection] = signed
    return signed


def _blob_list(prefix: str, sas_query: str) -> list[str]:
    """The blob names under a dataset root, using Azure's own listing verb."""
    url = (f"https://{ITEMS_ACCOUNT}.blob.core.windows.net/items"
           f"?restype=container&comp=list&prefix={urllib.parse.quote(prefix)}")
    if sas_query:
        url += f"&{sas_query}"
    with urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": _UA}), timeout=60) as response:
        body = response.read().decode("utf-8", "replace")
    return re.findall(r"<Name>([^<]+)</Name>", body)


def _part_specs(names: list[str]) -> list[tuple[str, int]]:
    """`(name, first_year)` for each part, read from its filename.

    Planetary Computer names each part with the date range it covers, so the parts
    a range needs can be chosen from the listing alone: no request per candidate
    part, and no reading a part to find out what is in it.
    """
    specs: list[tuple[str, int]] = []
    for name in names:
        if not name.endswith(".parquet"):
            continue
        tail = name.rsplit("/", 1)[-1].replace(".parquet", "")
        year = None
        for token in tail.split("_"):
            if len(token) >= 4 and token[:4].isdigit():
                year = int(token[:4])
                break
        if year is not None:
            specs.append((name, year))
    return specs


def _years_between(start: str, end: str) -> set[int]:
    return set(range(int(start[:4]), int(end[:4]) + 1))


def ensure_index(collection: str, start: str, end: str) -> list[Path]:
    """The locally cached parts covering `start`..`end`, downloading any missing.

    Returns whatever could be obtained. An empty list is not an error: the caller
    falls back to searching, which is the honest degradation for a snapshot that
    cannot be read. A part is written to a temp name and moved into place, so a
    download cut short leaves no truncated file behind for the next read to trust.
    """
    try:
        root = _signed_root(collection)
        sas_query = root.split("?", 1)[1] if "?" in root else ""
        specs = _part_specs(_blob_list(f"{collection}.parquet/", sas_query))
        if not specs:
            return []
        wanted = _years_between(start, end)
        local: list[Path] = []
        for name, year in specs:
            if year not in wanted:
                continue
            path = _part_path(collection, name.rsplit("/", 1)[-1])
            if not path.exists() or path.stat().st_size == 0:
                url = _signed_one(name)
                with urllib.request.urlopen(
                        urllib.request.Request(url, headers={"User-Agent": _UA}),
                        timeout=300) as response:
                    tmp = path.with_suffix(".part")
                    tmp.write_bytes(response.read())
                tmp.replace(path)
            local.append(path)
        return local
    except Exception as exc:  # noqa: BLE001 - degrade to searching, never fail a read
        print(f"stac-geoparquet: index unavailable ({type(exc).__name__}: {exc}); "
              "searching the STAC API instead", flush=True)
        return []


def _signed_one(name: str) -> str:
    import planetary_computer

    return planetary_computer.sign(
        f"https://{ITEMS_ACCOUNT}.blob.core.windows.net/{name}")


def month_assets(collection: str, bbox: list[float], start: str, end: str,
                 asset: str) -> dict[str, str]:
    """`{"YYYY-MM": asset_href}` for one collection, bbox and range, from the index.

    Filtering is on the item's own bbox, which fastparquet exposes as the four
    plain columns ``bbox.xmin``..``bbox.ymax``, and the href is the single column
    ``assets.<asset>.href`` -- so a 6 MB part is read as the handful of columns
    that answer the question and nothing else, with no JSON to parse and no
    geometry library needed.
    """
    import datetime as _dt

    from fastparquet import ParquetFile

    paths = ensure_index(collection, start, end)
    if not paths:
        return {}

    minx, miny, maxx, maxy = bbox
    href_column = f"assets.{asset}.href"
    lo = _dt.datetime.fromisoformat(f"{start[:7]}-01T00:00:00+00:00")
    hi = _dt.datetime.fromisoformat(f"{end[:7]}-28T23:59:59+00:00")

    found: dict[str, str] = {}
    for path in paths:
        handle = ParquetFile(str(path))
        available = set(handle.columns)
        columns = [c for c in ("start_datetime", "datetime",
                               "bbox.xmin", "bbox.ymin", "bbox.xmax", "bbox.ymax",
                               href_column) if c in available]
        if href_column not in columns or "start_datetime" not in columns:
            # A snapshot without this asset or without a start date cannot answer.
            continue
        frame = handle.to_pandas(columns=columns)
        for x0, y0, x1, y1, stamp, href in zip(
                frame["bbox.xmin"], frame["bbox.ymin"],
                frame["bbox.xmax"], frame["bbox.ymax"],
                frame["start_datetime"], frame[href_column]):
            # Disjoint means the item cannot cover any part of the area.
            if x1 < minx or x0 > maxx or y1 < miny or y0 > maxy:
                continue
            if not isinstance(stamp, _dt.datetime):
                continue
            if stamp < lo or stamp > hi:
                continue
            key = f"{stamp.year:04d}-{stamp.month:02d}"
            if key in found or not href:
                continue
            found[key] = href
    return found


def signed_hrefs(found: dict[str, str]) -> dict[str, str]:
    """Sign each href the index handed back.

    The snapshot stores unsigned blob URLs -- the STAC API signs its own on the way
    out -- so signing here is what makes the COGs readable, and it is the same step
    the search path already takes per item.
    """
    import planetary_computer

    return {month: planetary_computer.sign(href) for month, href in found.items()}
