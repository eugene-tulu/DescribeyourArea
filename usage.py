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
import sys
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


def _percentile(values: list[int], fraction: float) -> Optional[int]:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def breakdown(events: list[dict]) -> dict:
    """The detail :func:`summarise` deliberately leaves out.

    ``summarise`` answers "is it working"; this answers "working on what". Two
    questions need per-dataset detail and neither is visible in the totals: which
    datasets people actually ask for (so a dataset nobody requests is a candidate
    for removal, not for investment), and which dataset is slow (the total is an
    average over datasets the user may not have wanted, so a slow total can hide
    a fast request and a fast total can hide a slow one).
    """
    requested: dict[str, int] = {}
    outcomes: dict[str, dict[str, int]] = {}
    durations: dict[str, list[int]] = {}
    by_day: dict[str, int] = {}
    clients: dict[str, int] = {}
    totals: list[int] = []

    for event in events:
        for name in event.get("datasets_requested") or []:
            requested[name] = requested.get(name, 0) + 1
        for name, verdict in (event.get("outcomes") or {}).items():
            outcomes.setdefault(name, {})
            outcomes[name][verdict] = outcomes[name].get(verdict, 0) + 1
        for name, ms in (event.get("duration_ms") or {}).items():
            durations.setdefault(name, []).append(ms)
        at = (event.get("at") or "")[:10]
        if at:
            by_day[at] = by_day.get(at, 0) + 1
        client = event.get("client_prefix")
        if client:
            clients[client] = clients.get(client, 0) + 1
        if event.get("total_ms") is not None:
            totals.append(event["total_ms"])

    return {
        "requested": dict(sorted(requested.items(), key=lambda kv: -kv[1])),
        "outcomes": {k: dict(sorted(v.items(), key=lambda kv: -kv[1]))
                     for k, v in sorted(outcomes.items())},
        "duration_ms": {k: {"n": len(v), "p50": _percentile(v, 0.5),
                            "p95": _percentile(v, 0.95)}
                        for k, v in sorted(durations.items())},
        "by_day": dict(sorted(by_day.items())),
        "clients": dict(sorted(clients.items(), key=lambda kv: -kv[1])),
        "total_ms_p95": _percentile(totals, 0.95),
    }


def _table(title: str, rows: list[tuple[str, str]]) -> str:
    if not rows:
        return f"\n{title}\n  (none)"
    width = max(len(label) for label, _ in rows)
    body = "\n".join(f"  {label.ljust(width)}  {value}" for label, value in rows)
    return f"\n{title}\n{body}"


def render(events: list[dict]) -> str:
    """A summary a human reads, rather than one a browser fetches.

    The shape is a report, not JSON: this is for answering a question at a
    terminal, and the questions are always comparative -- which dataset, which
    day, which client. JSON forces the reader to do the summing.
    """
    if not events:
        return (
            "no usage events found.\n\n"
            "Events go to stdout unless USAGE_EVENTS_PATH names a file. To read the\n"
            "container log without copying it off the host:\n"
            "  docker logs <container> 2>&1 | python -m usage -"
        )

    head = summarise(events)
    detail = breakdown(events)
    lines = [f"{head['events']} event(s)"]

    if head["total_ms"]["p50"] is not None:
        lines.append(
            f"  request total_ms   p50 {head['total_ms']['p50']}   "
            f"p95 {head['total_ms']['p95']}"
        )
    lines.append(_table(
        "outcomes",
        [(k, str(v)) for k, v in head["outcomes"].items()],
    ))
    lines.append(_table(
        "datasets requested",
        [(k, str(v)) for k, v in detail["requested"].items()],
    ))

    per_dataset = []
    for name, verdicts in detail["outcomes"].items():
        timing = detail["duration_ms"].get(name, {})
        rendered = ", ".join(f"{k}={v}" for k, v in verdicts.items())
        if timing.get("p95") is not None:
            rendered += f"  (p50 {timing['p50']}ms, p95 {timing['p95']}ms, n={timing['n']})"
        per_dataset.append((name, rendered))
    lines.append(_table("per dataset", per_dataset))

    if head["sensors"]:
        lines.append(_table("sensors",
                            [(k, str(v)) for k, v in head["sensors"].items()]))
    if head["area_bands"]:
        lines.append(_table("area bands (km2)",
                            [(k, str(v)) for k, v in head["area_bands"].items()]))
    lines.append(_table("by day",
                        [(k, str(v)) for k, v in detail["by_day"].items()]))
    lines.append(_table("client prefixes (truncated in the event itself)",
                        [(k, str(v)) for k, v in detail["clients"].items()]))
    return "\n".join(lines)


def _events_from_stdin() -> list[dict]:
    """Read a log stream, ignoring everything that is not a usage event.

    Reading ``docker logs`` directly is the point: the container's log is a
    rotating buffer, so copying it off the host and analysing it later is
    analysing whatever survived rotation. Parsing lines that turn out not to be
    events is cheaper than a round trip, and a log full of tracebacks is the
    normal case, not an error.
    """
    events = []
    for line in sys.stdin:
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            candidate = json.loads(line)
        except ValueError:
            continue
        if isinstance(candidate, dict) and candidate.get("v") == EVENT_VERSION:
            events.append(candidate)
    return events


def main(argv: Optional[list[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m usage",
        description="Summarise recorded usage events.",
    )
    parser.add_argument(
        "path", nargs="?", default=None,
        help="events file, or - for stdin (e.g. 'docker logs <c> 2>&1 | python -m usage -')",
    )
    parser.add_argument("--json", action="store_true",
                        help="emit the aggregates as JSON instead of a report")
    args = parser.parse_args(argv)

    # No path and a pipe on stdin means a log is being fed in. Without this,
    # `docker logs <c> | python -m usage --json` prints a confident zero-event
    # report, which reads as "no traffic" rather than "you forgot to say -".
    piped = args.path is None and not sys.stdin.isatty()

    if args.path == "-" or piped:
        events = _events_from_stdin()
    else:
        target = Path(args.path).expanduser() if args.path else events_path()
        events = read_events(target)
        if not events and args.path:
            print(f"no events at {target}", file=sys.stderr)
            return 1

    if args.json:
        print(json.dumps({"summary": summarise(events), "detail": breakdown(events)},
                         indent=2, sort_keys=True))
    else:
        print(render(events))
    return 0


if __name__ == "__main__":  # pragma: no cover - a terminal entry point
    raise SystemExit(main())
