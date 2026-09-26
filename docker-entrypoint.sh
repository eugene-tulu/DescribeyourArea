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

if [ -n "${RAINFALL_CACHE_S3_URI:-}" ]; then
    echo "rainfall: pulling portfolio from ${RAINFALL_CACHE_S3_URI}"
    if ! python -m tools.sync_rainfall_cache --pull; then
        echo "rainfall: pull failed, continuing without a portfolio" >&2
    fi
else
    echo "rainfall: no RAINFALL_CACHE_S3_URI set, using only what is local"
fi

exec "$@"
