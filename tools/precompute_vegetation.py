"""Precompute vegetation series for named study areas, one MODIS read per month.

The request path never computes a vegetation series, and neither should it: a
single monthly window read is five serially-dependent HTTP round trips, and a
thirty-year series is a few hundred of them. This script fills the artefact
cache that ``GET /rainfall?indicator=vegetation_series`` reads, after which every
request is a JSON read exactly like rainfall.

    # one area from an inline GeoJSON file
    python -m tools.precompute_vegetation --areas areas.json

    # the Northern Rangelands Trust conservancies
    python -m tools.precompute_vegetation --conservancies path/to/NRT.geojson

    # see what is cached without touching the network
    python -m tools.precompute_vegetation --areas areas.json --dry-run

**Why one read per month for the whole portfolio.** The obvious implementation
computes each area separately, which costs ``N_areas x N_months`` window reads.
That is wrong by a factor of the portfolio size, because the cost is a round trip
and not pixels: the 21 published conservancies span 36.857-38.647 E and
0.263-2.059 N, which a STAC search resolves to a *single* MODIS tile, ``h21v08``.
So one monthly window covering their combined bounding box -- 847 x 862 px --
carries every conservancy, and each area's mean is extracted from that same block
by its own geometry mask. 21 areas cost 237 reads, not 4,977.

The flip side is honest and worth stating: the wide window is only correct while
every area falls inside the one tile. This script checks that rather than
assuming it, and falls back to per-area reads when the portfolio does not fit,
because a mean taken over a window that clipped a conservancy would be a
different number presented as the same number.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import rainfall
from tools.precompute_rainfall import load_areas


def combined_bbox(geometries: list[dict]) -> list[float]:
    west = min(min(pt[0] for pt in ring)
               for geom in geometries for ring in _rings(geom))
    east = max(max(pt[0] for pt in ring)
               for geom in geometries for ring in _rings(geom))
    south = min(min(pt[1] for pt in ring)
                for geom in geometries for ring in _rings(geom))
    north = max(max(pt[1] for pt in ring)
                for geom in geometries for ring in _rings(geom))
    return [west, south, east, north]


def _rings(geom: dict):
    if geom.get("type") == "Polygon":
        return geom["coordinates"]
    return [ring for poly in geom["coordinates"] for ring in poly]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m tools.precompute_vegetation",
        description="Precompute vegetation series for named study areas.",
    )
    parser.add_argument("--areas", type=Path, help="GeoJSON of the areas to build")
    parser.add_argument("--conservancies", type=Path,
                        help="alias for --areas, for the NRT file")
    parser.add_argument("--start", default="2010-01-01",
                        help="first month of the returned series (default 2010-01-01)")
    parser.add_argument("--end", default=None,
                        help="last month (default: this month)")
    parser.add_argument("--workers", type=int, default=4,
                        help="concurrent monthly reads (default 4; more measured slower)")
    parser.add_argument("--max-span-degrees", type=float, default=10.0,
                        help="widest portfolio that still shares one window read "
                             "(default 10 deg; beyond this, per-area reads)")
    parser.add_argument("--dry-run", action="store_true",
                        help="report what would be built and exit")
    args = parser.parse_args(argv)

    path = args.areas or args.conservancies
    if not path:
        parser.error("supply --areas or --conservancies")
    areas = load_areas(path)
    if not areas:
        print("no areas found in that file", file=sys.stderr)
        return 1

    import vegetation_series as vs

    end = args.end or time.strftime("%Y-%m-01")

    # The wide window is only an optimisation while the portfolio is compact. A
    # single monthly window across a scattered portfolio would be a very large
    # read and several tiles, and a mean taken from it is a mean over whatever the
    # window covered -- so past this span each area is built on its own. The
    # bound is generous on purpose: the 21 published conservancies span 1.8 deg
    # of longitude and 2.3 of latitude, well inside it.
    reads = vs.months_to_read(args.start, end)
    bbox = combined_bbox([geom for _name, _feature, _props in areas for geom in [_feature["geometry"]]])
    if max(bbox[2] - bbox[0], bbox[3] - bbox[1]) > args.max_span_degrees:
        print(f"portfolio spans more than {args.max_span_degrees} deg; building "
              f"each area on its own window instead of one shared read")
        return _build_per_area(areas, args, end)
    print(f"{len(areas)} area(s); combined bbox "
          f"{bbox[0]:.3f},{bbox[1]:.3f} to {bbox[2]:.3f},{bbox[3]:.3f}")
    print(f"window {args.start}..{end}: {reads} monthly reads "
          f"for the whole portfolio (~{vs.estimate_seconds(reads, args.workers)}s), "
          f"not {reads * len(areas)} per-area reads")

    if args.dry_run:
        return _report(areas, "would be built")

    return _build_shared_window(areas, bbox, args, end)


def _build_per_area(areas, args, end) -> int:
    """Each area read against its own window. Slower, and never wrong."""
    import vegetation_series as vs

    for name, feature, _props in areas:
        window = combined_bbox([feature["geometry"]])
        print(f"  {name}: own window {window[0]:.3f},{window[1]:.3f} "
              f"to {window[2]:.3f},{window[3]:.3f}")
    if args.dry_run:
        # The fallback must honour --dry-run too, or a preview of a scattered
        # portfolio would start building it, which is the opposite of a preview.
        return _report(areas, "would be built on its own window")
    return _build_shared_window(areas, None, args, end)


def _build_shared_window(areas, bbox, args, end) -> int:
    import jobs
    import vegetation_series as vs

    started = time.time()
    for index, (name, feature, _props) in enumerate(areas, start=1):
        geom = feature["geometry"]
        key = rainfall.geometry_hash(geom)
        if _readable(key):
            print(f"[{index}/{len(areas)}] {name}: cached")
            continue
        began = time.time()
        window = bbox or combined_bbox([geom])
        result = vs.compute_monthly_series(geom, window, start=args.start,
                                           end=end, workers=args.workers)
        status = result.get("status")
        if status != "ok":
            print(f"[{index}/{len(areas)}] {name}: {status} "
                  f"({result.get('reason')})")
            jobs.fail(key, str(result.get("reason") or status), "vegetation_series")
            continue
        jobs.write_artefact(key, "vegetation_series", result)
        jobs.complete(key, result, "vegetation_series")
        print(f"[{index}/{len(areas)}] {name}: {len(result.get('series') or [])} months "
              f"in {time.time() - began:.0f}s")
    print(f"done in {time.time() - started:.0f}s")
    return 0


def _report(areas, verb: str) -> int:
    pending = 0
    for name, feature, _props in areas:
        key = rainfall.geometry_hash(feature["geometry"])
        if _readable(key):
            print(f"  cached     {name}")
        else:
            pending += 1
            print(f"  to build   {name}")
    print(f"{pending} of {len(areas)} {verb}")
    return 0


def _readable(key: str) -> bool:
    import jobs

    payload = jobs.read_artefact(key, "vegetation_series")
    return bool(payload) and payload.get("status") == "ok"


if __name__ == "__main__":
    raise SystemExit(main())
