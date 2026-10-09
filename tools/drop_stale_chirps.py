"""Drop cached CHIRPS series computed before the product-aware fixes.

A CHIRPS series computed before the provenance and opening-month fixes carries
the wrong provenance -- it is stamped with ERA5's source/DOI even though its
grid, resolution and values are CHIRPS's -- and it is missing the first month of
its window. A series computed after carries the real CHIRPS source, which is how
the two are told apart here: a payload whose source is not the CHIRPS source is
the stale kind and is dropped.

The matching job record is dropped with it, so the area reads as "not computed"
and a resubmit recomputes it cleanly against the current code -- which is now
fast, because the store clip-and-fill reads the months the store holds instead
of rebuilding thousands of them from one GeoTIFF per month.

Dry by default; pass ``--apply`` to delete.

    python -m tools.drop_stale_chirps            # report, change nothing
    python -m tools.drop_stale_chirps --apply    # drop stale series + records
"""

from __future__ import annotations

import argparse
import json

import jobs
import rainfall

_SUFFIX = "-chirps.json"


def stale(path) -> tuple:
    """(key, source) for a stale CHIRPS payload, or None for a current one."""
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return (path.name[: -len(_SUFFIX)], "<unreadable>")
    if payload.get("source") != rainfall.CHIRPS_SOURCE:
        return (path.name[: -len(_SUFFIX)], payload.get("source"))
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true",
                        help="delete the stale series and their job records; "
                             "without it this only reports")
    args = parser.parse_args(argv)

    cache = rainfall.cache_dir()
    dropped = 0
    for path in sorted(cache.glob("*" + _SUFFIX)):
        found = stale(path)
        if found is None:
            continue
        key, source = found
        record = jobs.read_job(key, "rainfall")
        drop_record = bool(record and (record.get("product") or "rainfall") == "chirps")
        action = "dropped" if args.apply else "would drop"
        print(f"  {action} {key[:12]}  source={source!r}  job={'yes' if drop_record else 'no'}")
        if args.apply:
            path.unlink(missing_ok=True)
            if drop_record:
                jobs.drop(key, "rainfall")
        dropped += 1

    print(f"{dropped} stale CHIRPS series "
          f"{'dropped' if args.apply else 'found'}"
          f"{'' if args.apply or not dropped else ' (re-run with --apply to delete)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
