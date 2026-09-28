#!/bin/sh
# Container entrypoint: fetch the rainfall portfolio, then serve.
#
# The rainfall series are a build artefact, not application state, so they are
# pulled from object storage before the server starts. A local hit is served from
# the filesystem and never touches the network, so this runs once rather than per
# request.
#
# A missing or unreachable remote is not fatal: the service starts and rainfall
# simply reports "not computed" for areas it does not have. Losing the portfolio
# should degrade a card, not take the API down.
set -e

# Fail legibly rather than as a PermissionError later. Docker only applies the
# image's ownership when it creates a named volume, so a volume made by an older
# image stays root-owned while this container runs as uid 999. The symptom is a
# 500 on /rainfall/submit and a worker crash-looping on worker.lock, which does
# not obviously point at the directory. Say so, and name the fix.
for DIR in "${RAINFALL_CACHE_DIR:-/app/.rainfall-cache}" "${USAGE_EVENTS_PATH:-}"; do
    [ -n "$DIR" ] || continue
    [ -d "$DIR" ] || continue
    if [ ! -w "$DIR" ]; then
        echo "not writable by uid $(id -u): $DIR" >&2
        echo "run 'sh scripts/fix-cache-ownership.sh' on the host, then restart." >&2
        exit 1
    fi
done

if [ -n "${RAINFALL_CACHE_S3_URI:-}" ]; then
    echo "rainfall: pulling portfolio from ${RAINFALL_CACHE_S3_URI}"
    if ! python -m tools.sync_rainfall_cache --pull; then
        echo "rainfall: pull failed, continuing without a portfolio" >&2
    fi
else
    echo "rainfall: no RAINFALL_CACHE_S3_URI set, using only what is local"
fi

exec "$@"
