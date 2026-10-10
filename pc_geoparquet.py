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
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Optional

import rainfall

# The account that hosts the published snapshots, named by each collection's
# `geoparquet-items` asset in its `table:storage_options` block.
ITEMS_ACCOUNT = os.getenv("PC_ITEMS_ACCOUNT", "pcstacitems")
# The container holding the snapshots, from the ``abfs://<container>/<blob>``
# href each collection publishes.
ITEMS_CONTAINER = os.getenv("PC_ITEMS_CONTAINER", "items")
INDEX_CACHE_DIRNAME = "stac-geoparquet"
SAS_URL = "https://planetarycomputer.microsoft.com/api/sas/v1/token"
_UA = "geocontextualize"

_ROOT_CACHE: dict[str, str] = {}
_TOKEN: dict[tuple[str, str], tuple[str, float]] = {}


def index_dir() -> Path:
    """Where the cached index parts live. Under the cache dir, so it persists."""
    path = rainfall.cache_dir() / INDEX_CACHE_DIRNAME
    path.mkdir(parents=True, exist_ok=True)
    return path


def _part_path(collection: str, name: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in name)
    return index_dir() / f"{collection}-{safe}"


def _token_file(account: str, container: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in f"{account}-{container}")
    return index_dir() / f"sas-{safe}.json"


def _sas(account: str, container: str) -> str:
    """A short-lived read token for one container, from the public SAS API.

    Free and anonymous -- no Planetary Computer account, no login. The form is
    ``/sas/v1/token/{account}/{container}``; getting the two the other way round
    returns 404, which is a confusing way to learn the order.

    Cached on disk until close to its expiry. The API rate-limits, so fetching a
    token per read is both wasteful and, at volume, refused: a reading asks for a
    token once an hour rather than once a request.
    """
    key = (account, container)
    cached = _TOKEN.get(key)
    if cached and cached[1] - time.time() > 60:
        return cached[0]

    path = _token_file(account, container)
    try:
        import json

        stored = json.loads(path.read_text())
        if stored.get("expiry", 0) - time.time() > 60:
            _TOKEN[key] = (stored["token"], stored["expiry"])
            return stored["token"]
    except (OSError, ValueError, KeyError):
        pass

    token, expiry = _fetch_token(account, container)
    _TOKEN[key] = (token, expiry)
    try:
        path.write_text(f'{{"token": {token!r}, "expiry": {expiry!r}}}')
    except OSError:
        pass  # a cache that cannot be written is not worth failing a read over
    return token


def _fetch_token(account: str, container: str) -> tuple[str, float]:
    """One token from the SAS API, backing off when it says too many requests."""
    import json

    url = f"{SAS_URL}/{account}/{container}"
    last: Exception | None = None
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(
                    url, headers={"User-Agent": _UA}), timeout=60) as response:
                document = json.loads(response.read())
            token = document.get("token") or ""
            if not token:
                raise RuntimeError(f"the SAS API returned no token for {account}/{container}")
            expiry = document.get("msft:expiry") or ""
            try:
                epoch = time.mktime(time.strptime(expiry[:19], "%Y-%m-%dT%H:%M:%S"))
            except ValueError:
                epoch = time.time() + 3600
            return token, epoch
        except Exception as exc:  # noqa: BLE001 - 429 and 503 both clear on a retry
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise last if last else RuntimeError("could not obtain a token")


def _blob_url(path: str) -> str:
    """A signed URL for one blob in the items account.

    The container is named in the path, and the blob part is percent-encoded with
    ``/`` kept as a separator: the part filenames carry timestamps, so they contain
    ``+`` and ``:``. Those are legal in an Azure blob name and in a URL path, but a
    ``+`` decodes to a space on the way in, so the resource is sought under a name
    it does not have and Azure answers 400 "invalid characters" -- which reads as a
    server problem rather than the encoding slip it is.
    """
    encoded = urllib.parse.quote(path, safe="/")
    return (f"https://{ITEMS_ACCOUNT}.blob.core.windows.net/{ITEMS_CONTAINER}/{encoded}"
            f"?{_sas(ITEMS_ACCOUNT, ITEMS_CONTAINER)}")


def _collection_blob(collection: str) -> tuple[str, str]:
    """The ``(container, blob)`` a collection's ``geoparquet-items`` asset names.

    ``abfs://items/<collection>.parquet`` -- the path after ``://`` is container,
    then blob. An abfs path is not a URL a blob client will read, so it is
    separated here rather than being rebuilt into a URL and re-parsed later.
    """
    import json

    url = (f"https://planetarycomputer.microsoft.com/api/stac/v1/collections/{collection}")
    with urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": _UA}), timeout=60) as response:
        document = json.loads(response.read())
    href = ((document.get("assets") or {}).get("geoparquet-items") or {}).get("href") or ""
    if not href:
        raise RuntimeError(f"{collection} publishes no geoparquet-items asset")
    if href.startswith("abfs://"):
        _, _, rest = href.partition("://")
        container, _, blob = rest.partition("/")
        return container, blob
    parsed = urllib.parse.urlparse(href)
    parts = parsed.path.lstrip("/").split("/", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return ITEMS_CONTAINER, parts[0]


def _blob_list(prefix: str) -> list[str]:
    """The blob names under a dataset root, using Azure's own listing verb.

    The prefix is relative to the container -- ``<collection>.parquet/``, not
    ``items/<collection>.parquet/`` -- because the listing already names the
    container in its path. Including it matches nothing, and every part then looks
    absent, which reads as "this collection publishes nothing".

    The slash in the prefix is percent-encoded. Azure accepts either form, but a
    literal one puts the separator inside the query string, where it reads as the
    end of the resource name and the request is refused.
    """
    url = (f"https://{ITEMS_ACCOUNT}.blob.core.windows.net/{ITEMS_CONTAINER}"
           f"?restype=container&comp=list&prefix={urllib.parse.quote(prefix)}"
           f"&maxresults=5000&{_sas(ITEMS_ACCOUNT, ITEMS_CONTAINER)}")
    with urllib.request.urlopen(
            urllib.request.Request(url, headers={"User-Agent": _UA}), timeout=60) as response:
        body = response.read().decode("utf-8", "replace")
    return re.findall(r"<Name>([^<]+)</Name>", body)


def _part_specs(names: list[str]) -> list[tuple[str, str, str]]:
    """`(name, first_month, last_month)` for each part, read from its filename.

    Planetary Computer names each part with the date range it covers, and the
    granularity is finer than a year -- the collection's own partition metadata
    says weekly, so a recent year is many parts. Reading the range rather than
    only the year means a window inside 2026 downloads the parts covering it
    instead of all of 2026's, which is most of the difference between a cold read
    costing 3 MB and costing 20.
    """
    specs: list[tuple[str, str, str]] = []
    for name in names:
        if not name.endswith(".parquet"):
            continue
        tail = name.rsplit("/", 1)[-1].replace(".parquet", "")
        months = [token[:7] for token in tail.split("_")
                  if len(token) >= 7 and token[:4].isdigit() and token[4] == "-"]
        if len(months) >= 2:
            specs.append((name, months[0], months[1]))
        elif months:
            specs.append((name, months[0], months[0]))
    return specs


def _years_between(start: str, end: str) -> set[int]:
    return set(range(int(start[:4]), int(end[:4]) + 1))


def _cached_specs(collection: str) -> list[tuple[str, str, str]]:
    """The collection's parts, listed once and remembered.

    The listing changes about once a week, when a new part appears. Asking Azure
    for it on every read is a request that answers the same thing, and Azure
    rate-limits it -- which is how a cached read comes back 429. It is re-listed
    when a wanted span has no part yet, so a new month still turns up on its own.
    """
    import json

    path = index_dir() / f"{collection}-parts.json"
    try:
        return [(entry[0], entry[1], entry[2]) for entry in json.loads(path.read_text())]
    except (OSError, ValueError, IndexError):
        return []


def _remember_specs(collection: str, specs: list[tuple[str, str, str]]) -> None:
    import json

    try:
        (index_dir() / f"{collection}-parts.json").write_text(json.dumps(specs))
    except OSError:
        pass


def ensure_index(collection: str, start: str, end: str) -> list[Path]:
    """The locally cached parts covering `start`..`end`, downloading any missing.

    A part is chosen by its own date range, not by its year, so a window inside a
    year pulls only the parts covering it. Returns whatever could be obtained; an
    empty list is not an error, because the caller falls back to searching. A part
    is written to a temp name and moved into place, so a download cut short leaves
    no truncated file behind for the next read to trust.
    """
    try:
        _container, blob = _collection_blob(collection)
        span = (start[:7], end[:7])
        specs = _cached_specs(collection)
        if not any(first <= span[1] and last >= span[0] for _n, first, last in specs):
            # Nothing remembered covers this span -- a cold cache, or a part that
            # has never been listed.
            specs = _part_specs(_blob_list(f"{blob}/"))
            _remember_specs(collection, specs)
        if not specs:
            return []
        local: list[Path] = []
        for name, first, last in specs:
            if last < span[0] or first > span[1]:
                continue
            path = _part_path(collection, name.rsplit("/", 1)[-1])
            if not path.exists() or path.stat().st_size == 0:
                with urllib.request.urlopen(
                        urllib.request.Request(_blob_url(name),
                                               headers={"User-Agent": _UA}),
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
            # The snapshot stores UTC timestamps without their offset, and the
            # bounds carry one. Comparing them as they arrive raises, so the naive
            # stamp is read as the UTC it is rather than being coerced by luck.
            if isinstance(stamp, _dt.datetime) and stamp.tzinfo is None:
                stamp = stamp.replace(tzinfo=_dt.timezone.utc)
            if not isinstance(stamp, _dt.datetime):
                continue
            if stamp < lo or stamp > hi:
                continue
            key = f"{stamp.year:04d}-{stamp.month:02d}"
            if key in found or not href:
                continue
            found[key] = href
    return found


def sign_blob(href: str) -> str:
    """Sign one asset href, deriving its account and container from the URL.

    The snapshot's hrefs point at whatever account holds that collection's pixels,
    which is a different account from the one holding the snapshot -- MODIS's COGs
    live in ``modiseuwest``. Both are readable with a token from the same public
    API, so the account and container come from the href rather than being assumed.
    An href already carrying a query is returned as it is: it is signed already.
    """
    parsed = urllib.parse.urlparse(href)
    if parsed.query or not parsed.netloc.endswith("blob.core.windows.net"):
        return href
    account, _, _rest = parsed.netloc.partition(".")
    container, _, _blob = parsed.path.lstrip("/").partition("/")
    if not container:
        return href
    return f"{href}?{_sas(account, container)}"
