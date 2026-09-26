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
import os
import sys
import time
from pathlib import Path

import rainfall


def load_areas(path: Path) -> list[tuple[str, dict, dict]]:
    """Accept a FeatureCollection, a bare Feature, or a list of either.

    Every geometry goes through the same canonicaliser the request path uses, and
    the canonical form is what gets hashed and stored. Skipping that step published
    series under a key the lookup could never derive, because the lookup repairs
    self-intersecting rings and hashing the raw geometry does not.
    """
    import main

    payload = json.loads(path.read_text())
    if isinstance(payload, dict) and payload.get("type") == "FeatureCollection":
        features = payload["features"]
    elif isinstance(payload, dict) and payload.get("type") == "Feature":
        features = [payload]
    elif isinstance(payload, list):
        features = payload
    else:
        raise SystemExit(f"{path}: expected a FeatureCollection, Feature, or list")

    out: list[tuple[str, dict, dict]] = []
    for index, feature in enumerate(features):
        properties = feature.get("properties") or {}
        name = str(properties.get("NAME") or properties.get("name") or f"area-{index + 1}")
        try:
            canonical = main.canonicalize_geojson(
                feature,
                max_bytes=main.MAX_LOOKUP_BYTES,
                max_vertices=main.MAX_LOOKUP_VERTICES,
            )
        except Exception as exc:  # noqa: BLE001 - a bad area must not stop the run
            print(f"  {name:24s} INVALID  {getattr(exc, 'detail', exc)}")
            continue
        out.append((name, canonical, canonical.get("properties") or {}))
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
    parser.add_argument("--publish", action="store_true",
                        help="upload each freshly built series to RAINFALL_CACHE_S3_URI")
    args = parser.parse_args(argv)

    if args.publish and rainfall.remote_prefix() is None:
        raise SystemExit(
            "--publish needs RAINFALL_CACHE_S3_URI (and optionally "
            "RAINFALL_S3_ENDPOINT / RAINFALL_S3_REGION) to be set"
        )

    areas = load_areas(args.areas or args.conservancies)
    if args.only:
        areas = [(n, a) for n, a in areas if args.only.lower() in n.lower()]

    print(f"cache: {rainfall.cache_dir()}")
    if rainfall.remote_prefix():
        print(f"remote: s3://{rainfall.remote_prefix()}"
              + (f" via {os.environ.get('RAINFALL_S3_ENDPOINT')}" if os.environ.get("RAINFALL_S3_ENDPOINT") else ""))
    print(f"areas: {len(areas)}")

    usable = [(n, f.get("geometry")) for n, f, _props in areas
              if (f.get("geometry") or {}).get("type") in {"Polygon", "MultiPolygon"}]
    source = None
    if usable and not args.dry_run:
        # One read for the union of every area's cells instead of one per area.
        union = rainfall.union_grid([g for _n, g in usable])
        print(f"union grid: {len(union['longitudes'])} x {len(union['latitudes'])} cells")
        source = rainfall.UnionReader(union)
    print()

    built = cached = missed = failed = 0
    for name, feature, properties in areas:
        geom = feature.get("geometry")
        if not geom or geom.get("type") not in {"Polygon", "MultiPolygon"}:
            print(f"  {name:24s} SKIP  not a polygon")
            failed += 1
            continue
        key = rainfall.geometry_hash(geom)
        if properties.get("geometry_repaired"):
            print(f"  {name:24s} note   {properties['geometry_repaired']}")
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
                geom, start=args.start, end=args.end, source=source, upload=args.publish)
        except Exception as exc:  # noqa: BLE001 - one bad area must not stop the run
            print(f"  {name:24s} FAIL    {type(exc).__name__}: {str(exc)[:90]}")
            failed += 1
            continue
        elapsed = time.perf_counter() - start
        dropped = len(payload["coverage"]["window"]["months_dropped"])
        suspect = len(payload["summary"].get("suspect_months", []))
        uploaded = "  published" if args.publish else ""
        print(f"  {name:24s} built    {len(payload['series']):4d} months  "
              f"cells={payload['grid_cells']:3d}  {elapsed:5.1f}s  "
              f"dropped={dropped} suspect={suspect}{uploaded}")
        built += 1

    print(f"\nbuilt {built}, cached {cached}, missing {missed}, failed {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
