"""Pull or push the rainfall series cache to an S3-compatible store.

The portfolio is a published artefact, not state tied to one machine: the build
pushes it, and a deployment pulls it. Neither the request path nor this tool needs
the remote to be configured, in which case they are no-ops.

    # deployment: fetch the portfolio before serving
    python -m tools.sync_rainfall_cache --pull

    # after adding areas, publish what is local
    python -m tools.sync_rainfall_cache --push

    # inspect
    python -m tools.sync_rainfall_cache --list

Configuration:

    RAINFALL_CACHE_S3_URI=s3://my-bucket/geocontextualize/rainfall
    RAINFALL_S3_ENDPOINT=https://nyc3.digitaloceanspaces.com
    RAINFALL_S3_REGION=nyc3

The store is assumed to be S3-compatible, which DigitalOcean Spaces is. The
per-read cell cache under ``cells/`` is a build accelerator and is deliberately
not transferred.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import rainfall

LOCAL_PREFIX = "series/"


def _bucket() -> str:
    return rainfall.remote_prefix().split("/", 1)[0]


def _iter_local() -> list[Path]:
    directory = rainfall.cache_dir() / LOCAL_PREFIX.strip("/")
    if not directory.exists():
        # Series are written as <key>.json directly in the cache directory, so
        # accept both layouts rather than assuming one.
        return [
            p for p in rainfall.cache_dir().glob("*.json")
            if p.is_file()
        ]
    return sorted(directory.glob("*.json"))


def pull() -> int:
    client = rainfall._s3_client()
    bucket, prefix = _bucket(), rainfall.remote_prefix()
    directory = rainfall.cache_dir()
    directory.mkdir(parents=True, exist_ok=True)
    paginator = client.get_paginator("list_objects_v2")
    copied = skipped = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/series/"):
        for entry in page.get("Contents", []):
            name = entry["Key"].rsplit("/", 1)[-1]
            if not name.endswith(".json"):
                continue
            target = directory / name
            response = client.get_object(Bucket=bucket, Key=entry["Key"])
            body = response["Body"].read()
            if target.exists() and target.read_bytes() == body:
                skipped += 1
                continue
            temporary = target.with_suffix(".json.tmp")
            temporary.write_bytes(body)
            temporary.replace(target)
            copied += 1
    print(f"pulled {copied} series, {skipped} already current -> {directory}")
    return copied


def push() -> int:
    client = rainfall._s3_client()
    bucket, prefix = _bucket(), rainfall.remote_prefix()
    pushed = 0
    for path in _iter_local():
        key = path.stem
        rainfall.write_cache(key, rainfall.json.loads(path.read_text()))
        client.upload_file(str(path), bucket, f"{prefix}/{rainfall.series_object(key)}")
        print(f"  published {key[:12]}  {path.stat().st_size / 1024:.0f} KiB")
        pushed += 1
    print(f"published {pushed} series to s3://{prefix}/series/")
    return pushed


def listing() -> None:
    client = rainfall._s3_client()
    bucket, prefix = _bucket(), rainfall.remote_prefix()
    paginator = client.get_paginator("list_objects_v2")
    total = count = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{prefix}/series/"):
        for entry in page.get("Contents", []):
            if not entry["Key"].endswith(".json"):
                continue
            count += 1
            total += entry["Size"]
    print(f"s3://{prefix}/series/  {count} objects, {total / 1024:.0f} KiB")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--pull", action="store_true", help="fetch the portfolio locally")
    action.add_argument("--push", action="store_true", help="upload local series")
    action.add_argument("--list", action="store_true", help="summarise the remote portfolio")
    args = parser.parse_args(argv)

    if rainfall.remote_prefix() is None:
        print(
            "no remote configured; set RAINFALL_CACHE_S3_URI (and optionally "
            "RAINFALL_S3_ENDPOINT / RAINFALL_S3_REGION)",
            file=sys.stderr,
        )
        return 2
    print(f"local:  {rainfall.cache_dir()}")
    print(f"remote: s3://{rainfall.remote_prefix()}\n")

    try:
        if args.pull:
            pull()
        elif args.push:
            push()
        else:
            listing()
    except Exception as exc:  # noqa: BLE001 - a CLI should report, not traceback
        # Credentials, endpoint and DNS problems are the common cases here, and a
        # raw botocore traceback tells the operator nothing they can act on.
        name = type(exc).__name__
        detail = str(exc).split("(")[0].strip()
        print(
            f"remote {rainfall.remote_prefix()!r} is not usable: {name}: {detail}\n"
            f"check RAINFALL_CACHE_S3_URI, RAINFALL_S3_ENDPOINT, RAINFALL_S3_REGION "
            f"and the AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY pair",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
