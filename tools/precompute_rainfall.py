"""Precompute rainfall series for named study areas.

The request path never computes rainfall: reading a single ERA5 grid cell takes
about 20 seconds, which is longer than a whole request. This script fills the
cache that the endpoint reads.

    # one area from an inline GeoJSON file
    python -m tools.precompute_rainfall --areas areas.json

    # the Northern Rangelands Trust conservancies
    python -m tools.precompute_rainfall --conservancies path/to/NRT_Conservancies.geojson

    # check what is cached without touching the network
    python -m tools.precompute_rainfall --areas areas.json --dry-run

Each area is written as a small JSON document keyed by a hash of its geometry, so
re-running after a geometry change produces a new entry rather than silently
serving a stale one. ``processing_version`` in the payload invalidates the cache
when the computation changes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import rainfall


def load_areas(path: Path) -> list[tuple[str, dict]]:
    """Accept a FeatureCollection, a bare Feature, or a list of either."""
    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and payload.get("type") == "FeatureCollection":
        features = payload["features"]
    elif isinstance(payload, dict) and payload.get("type") == "Feature":
        features = [payload]
    elif isinstance(payload, list):
        features = payload
    else:
        raise SystemExit(f"{path}: expected a FeatureCollection, Feature, or list")

    out: list[tuple[str, dict]] = []
    for index, feature in enumerate(features):
        name = (feature.get("properties") or {}).get("NAME") or (
            feature.get("properties") or {}
        ).get("name") or f"area-{index + 1}"
        out.append((str(name), feature))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--areas", type=Path, help="GeoJSON file of study areas")
    source.add_argument("--conservancies", type=Path, help="alias for --areas")
    parser.add_argument("--start", default="2010-01-01", help="first month of the series")
    parser.add_argument("--end", default=None, help="last month (default: this month)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report cache status without reading ERA5")
    parser.add_argument("--force", action="store_true",
                        help="recompute even when a valid cached entry exists")
    parser.add_argument("--only", default=None, help="process only areas whose name matches")
    args = parser.parse_args(argv)

    areas = load_areas(args.areas or args.conservancies)
    if args.only:
        areas = [(n, a) for n, a in areas if args.only.lower() in n.lower()]

    print(f"cache: {rainfall.cache_dir()}")
    print(f"areas: {len(areas)}")

    usable = [(n, f.get("geometry")) for n, f in areas
              if (f.get("geometry") or {}).get("type") in {"Polygon", "MultiPolygon"}]
    source = None
    if usable and not args.dry_run:
        # One read for the union of every area's cells instead of one per area.
        union = rainfall.union_grid([g for _n, g in usable])
        print(f"union grid: {len(union['longitudes'])} x {len(union['latitudes'])} cells")
        source = rainfall.UnionReader(union)
    print()

    built = cached = missed = failed = 0
    for name, feature in areas:
        geom = feature.get("geometry")
        if not geom or geom.get("type") not in {"Polygon", "MultiPolygon"}:
            print(f"  {name:24s} SKIP  not a polygon")
            failed += 1
            continue
        key = rainfall.geometry_hash(geom)
        existing = rainfall.read_cache(key)
        if existing and not args.force:
            print(f"  {name:24s} cached   {len(existing.get('series', []))} months"
                  f"  cells={existing.get('grid_cells')}")
            cached += 1
            continue
        if args.dry_run:
            print(f"  {name:24s} MISS    key={key}")
            missed += 1
            continue
        start = time.perf_counter()
        try:
            payload = rainfall.build_and_cache(
                geom, start=args.start, end=args.end, source=source)
        except Exception as exc:  # noqa: BLE001 - one bad area must not stop the run
            print(f"  {name:24s} FAIL    {type(exc).__name__}: {str(exc)[:90]}")
            failed += 1
            continue
        elapsed = time.perf_counter() - start
        dropped = len(payload["coverage"]["window"]["months_dropped"])
        suspect = len(payload["summary"].get("suspect_months", []))
        print(f"  {name:24s} built    {len(payload['series']):4d} months  "
              f"cells={payload['grid_cells']:3d}  {elapsed:5.1f}s  "
              f"dropped={dropped} suspect={suspect}")
        built += 1

    print(f"\nbuilt {built}, cached {cached}, missing {missed}, failed {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
