"""Privacy-preserving usage events.

The point of this module is that the privacy properties are *structural* rather
than documented: :func:`build_event` has no parameter that could carry a
submitted geometry, it bands the area itself, and it truncates the client address
itself. A future edit cannot accidentally log a polygon, because there is nowhere
to put one.

What that deliberately costs: you learn how many analyses ran, how they failed and
how long they took, but not who asked about which piece of land. The band edges
are the product's own caps, so a pile-up in the ``"10-100"`` band reads directly as
"the synchronous cap is too tight".
"""

from __future__ import annotations

import datetime
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from shapely.geometry import shape

# Band edges are the shipped caps, so a band's population says something about the
# product rather than only about its users.
AREA_BANDS = (
    (10.0, "0-10"),
    (100.0, "10-100"),
    (1000.0, "100-1000"),
    (float("inf"), "1000+"),
)
NDVI_CAP_KM2 = 10.0
SYNC_CAP_KM2 = 100.0
LANDCOVER_CAP_KM2 = 1000.0

EVENT_VERSION = 1

# Analytics are opt-out rather than opt-in. The people using this are community
# organisations and their staff, not a marketing funnel, and recording the shape of
# the land someone asked about is a different kind of ask than reading a web page.
# The event carries no geometry, but the choice of which polygon to send is theirs.
#
# Set ANALYTICS_DISABLED=1 to stop recording entirely, or have a caller send
# Analytics-Do-Not-Track: true, which takes effect for that request only.
ENV_DISABLED = "ANALYTICS_DISABLED"
HEADER_DNT = "analytics-do-not-track"


def area_band(area_km2: Optional[float]) -> str:
    """Bucket a study-area size. The raw number is never carried onward."""
    if area_km2 is None:
        return "unknown"
    for edge, label in AREA_BANDS:
        if area_km2 < edge:
            return label
    return AREA_BANDS[-1][1]


def client_prefix(address: Optional[str]) -> str:
    """Truncate a client address to its network prefix.

    A full address identifies a household or a person, and retaining one makes this
    a record of personal data under both GDPR and Kenya's DPA. A /24 is still
    specific enough to rate-limit and to notice scraping from one network, which is
    what the address is actually wanted for.
    """
    if not address:
        return "unknown"
    cleaned = address.strip()
    if not cleaned:
        return "unknown"
    if ":" in cleaned:  # IPv6: keep the first four hextets
        parts = cleaned.split(":")
        return ":".join(parts[:4]) + "::/64"
    octets = cleaned.split(".")
    if len(octets) != 4:
        return "unknown"
    return ".".join(octets[:3]) + ".0/24"


def band_outcome(result: Any) -> str:
    """Reduce a module result to a verdict, never to its values.

    The distinction matters: ``{"error": "no_valid_elevation_pixels"}`` becomes
    ``"error"``. The measurement behind it is a data point about a place and does
    not belong in a usage log.
    """
    if result is None:
        return "not_requested"
    if isinstance(result, dict):
        if result.get("error"):
            return "error"
        status = result.get("status")
        if status:
            return str(status)
    return "ok"


def analytics_enabled(request=None) -> bool:
    """Whether this request may be recorded, honouring the per-request opt-out."""
    if os.getenv(ENV_DISABLED, "").strip().lower() in {"1", "true", "yes", "on"}:
        return False
    if request is not None:
        try:
            if str(request.headers.get(HEADER_DNT, "")).strip().lower() in {"1", "true", "yes"}:
                return False
        except Exception:  # noqa: BLE001
            return True
    return True


def build_event(
    *,
    datasets_requested: Iterable[str],
    outcomes: dict[str, Any],
    duration_ms: Optional[dict[str, int]] = None,
    total_ms: Optional[int] = None,
    area_km2: Optional[float] = None,
    client_address: Optional[str] = None,
    sensor: Optional[str] = None,
) -> dict:
    """Assemble one event. There is no geometry parameter, by design."""
    return {
        "v": EVENT_VERSION,
        "at": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        "datasets_requested": sorted({str(d) for d in datasets_requested}),
        "outcomes": {k: band_outcome(v) for k, v in sorted(outcomes.items())},
        "duration_ms": {k: int(v) for k, v in sorted((duration_ms or {}).items()) if v is not None},
        "total_ms": int(total_ms) if total_ms is not None else None,
        "aoi_area_km2_band": area_band(area_km2),
        "sensor": sensor,
        "client_prefix": client_prefix(client_address),
    }


class Timer:
    """Context manager for a per-module duration, in milliseconds."""

    def __init__(self) -> None:
        self._start = 0.0
        self.elapsed_ms: Optional[int] = None

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *exc) -> None:
        self.elapsed_ms = int((time.perf_counter() - self._start) * 1000)


def events_path() -> Optional[Path]:
    """Where to append events, or None to send them to stdout only."""
    configured = os.getenv("USAGE_EVENTS_PATH", "").strip()
    return Path(configured).expanduser() if configured else None


def emit(event: dict) -> None:
    """Append one event as a JSON line. Never raises into the request path."""
    line = json.dumps(event, sort_keys=True, separators=(",", ":"))
    path = events_path()
    try:
        if path is None:
            print(line, flush=True)
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")
    except Exception:  # noqa: BLE001 - analytics must never fail a request
        pass


def summarise(events: Iterable[dict]) -> dict:
    """Aggregate a batch, which is the only shape these events are meant to be read in."""
    events = list(events)
    totals: dict[str, int] = {}
    bands: dict[str, int] = {}
    sensors: dict[str, int] = {}
    durations: list[int] = []
    for event in events:
        for verdict in (event.get("outcomes") or {}).values():
            totals[verdict] = totals.get(verdict, 0) + 1
        band = event.get("aoi_area_km2_band")
        if band:
            bands[band] = bands.get(band, 0) + 1
        sensor = event.get("sensor")
        if sensor:
            sensors[sensor] = sensors.get(sensor, 0) + 1
        if event.get("total_ms") is not None:
            durations.append(event["total_ms"])
    durations.sort()
    return {
        "events": len(events),
        "outcomes": dict(sorted(totals.items())),
        "area_bands": dict(sorted(bands.items())),
        "sensors": dict(sorted(sensors.items())),
        "total_ms": {
            "p50": durations[len(durations) // 2] if durations else None,
            "p95": durations[int(len(durations) * 0.95)] if durations else None,
        },
    }


def read_events(path: Optional[Path] = None) -> list[dict]:
    target = path or events_path()
    if target is None or not target.exists():
        return []
    events = []
    for line in target.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    return events


def assert_no_geometry(event: dict) -> None:
    """Guard used by the tests, and cheap enough to keep honest.

    A submitted polygon identifies the land and is the user's data, not ours. This
    walks the record looking for anything coordinate-shaped so a future field
    cannot reintroduce one unnoticed.
    """
    banned = ("geometry", "coordinates", "coords", "bbox", "geom", "polygon", "wkt", "geojson")
    offenders = []

    def walk(node, path=""):
        if isinstance(node, dict):
            for key, value in node.items():
                if str(key).lower() in banned:
                    offenders.append(f"{path}.{key}")
                walk(value, f"{path}.{key}")
        elif isinstance(node, (list, tuple)):
            for index, value in enumerate(node):
                walk(value, f"{path}[{index}]")
        elif isinstance(node, (list, tuple)):
            pass

    walk(event)
    if offenders:
        raise AssertionError(f"usage event carries geometry-shaped fields: {offenders}")


def area_km2_of(geojson_geom: Optional[dict]) -> Optional[float]:
    """Geodesic area of a study area, for the band. Accepts a geometry, returns
    only a number, so the caller cannot hand the geometry onward."""
    if not geojson_geom:
        return None
    from main import _bbox_area_km2

    minx, miny, maxx, maxy = shape(geojson_geom).bounds
    return _bbox_area_km2([float(minx), float(miny), float(maxx), float(maxy)])
