#!/bin/sh
# Repair ownership of the named volumes the app writes to.
#
# Why this exists: the backend and the worker run as uid 999 (app), but Docker
# only applies the image's directory ownership when it *creates* a named volume.
# A volume created by an older image keeps its old ownership forever, and both
# services then fail to write. The backend surfaces a 500 on /rainfall/submit and
# the worker crash-loops on worker.lock with a PermissionError, which is a
# confusing pair of symptoms for one cause.
#
# This is safe to run at any time: it changes only the owner of these
# directories to the uid the containers already run as.
#
# Usage:  sh scripts/fix-cache-ownership.sh [project-name]
set -eu

PROJECT="${1:-geocontextualize}"
APP_UID=999
APP_GID=999
REPAIRED=0

for NAME in rainfall-cache usage-events; do
    VOLUME="${PROJECT}_${NAME}"
    if ! docker volume inspect "$VOLUME" >/dev/null 2>&1; then
        echo "skip $VOLUME: does not exist yet"
        continue
    fi

    MOUNTPOINT=$(docker volume inspect "$VOLUME" --format '{{.Mountpoint}}')
    CURRENT=$(ls -ldn "$MOUNTPOINT" | awk '{print $3":"$4}')
    echo "volume $VOLUME at $MOUNTPOINT, currently $CURRENT"

    if [ "$CURRENT" = "${APP_UID}:${APP_GID}" ]; then
        echo "  already owned by ${APP_UID}:${APP_GID}; nothing to do"
        continue
    fi

    chown -R "${APP_UID}:${APP_GID}" "$MOUNTPOINT"
    echo "  repaired -> $(ls -ldn "$MOUNTPOINT" | awk '{print $3":"$4}')"
    REPAIRED=1
done

[ "$REPAIRED" = "0" ] && exit 0
echo
echo "Start the stack and confirm the worker is no longer restarting:"
echo "  docker compose up -d && docker compose logs --tail 20 worker"
