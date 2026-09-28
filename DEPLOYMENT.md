# Deployment guide

The production stack is intentionally small: a FastAPI backend and a Next.js
frontend run on loopback-only Docker ports, while Nginx is the only public
entry point. Browser calls use the same-origin `/api` path.

## Prerequisites

- Ubuntu with Docker Compose v2, Git, and Nginx
- A real `.env` file; copy `.env.example`, do not commit the resulting file
- A public HTTPS hostname is preferred. The checked-in Nginx examples also
  support the current server IP as a short-lived certificate fallback.

## Deploy the `main` source

```bash
git clone --branch main https://github.com/eugene-tulu/DescribeyourArea.git /srv/geocontextualize
cd /srv/geocontextualize
cp .env.example .env
# Set the explicit CORS_ORIGINS if the defaults do not match your host.
docker compose up -d --build
docker compose ps
curl --fail http://127.0.0.1:8001/health
```

The Compose file deliberately binds only `127.0.0.1:3000` and
`127.0.0.1:8001`. It also caps application resources, retains only small local
logs, and has no disk-growing STAC cache.

## Rainfall portfolio

Precipitation is computed offline, because a single ERA5 grid cell takes about 20
seconds to read and a whole request budget is 20-30 seconds. The request path only
reads a cache, so the portfolio must be present before the service starts.

It lives in an S3-compatible store, which DigitalOcean Spaces is, so it is a
published artefact rather than state tied to one image. It is small: the 21
Northern Rangelands Trust conservancies are about 590 KB, roughly 25 KB each.

```bash
# build, from a machine with the dependencies installed
RAINFALL_CACHE_DIR=.rainfall-cache \
RAINFALL_CACHE_S3_URI=s3://my-bucket/geocontextualize/rainfall \
RAINFALL_S3_ENDPOINT=https://nyc3.digitaloceanspaces.com \
RAINFALL_S3_REGION=nyc3 \
python -m tools.precompute_rainfall --conservancies areas.geojson \
    --start 2010-01-01 --publish

# deploy: fetch before serving
python -m tools.sync_rainfall_cache --pull
```

Set `RAINFALL_CACHE_DIR` to a writable path inside the container, for example
`/app/.rainfall-cache`, and run the pull in the same entrypoint as the server
start. Both variables are optional: unset, everything stays local and the remote
calls are no-ops.

Two properties worth relying on:

- **A cache hit never touches the network.** `cached_context()` reads the local
  file first and consults the remote only on a miss, so the hot path gains neither
  latency nor a new failure mode.
- **A miss stays a miss.** An unreachable remote returns `not_computed` rather
  than an error, so a Spaces outage degrades to "not processed yet" instead of a
  failed request.

Spaces credentials come from the standard AWS environment variables
(`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`). The per-read cell cache under
`cells/` is a build accelerator and is deliberately not transferred.

`RAINFALL_S3_ENDPOINT` accepts either documented form —
`https://<bucket>.<region>.digitaloceanspaces.com` or
`https://<region>.digitaloceanspaces.com`. A leading bucket is stripped before
the request, because boto3 puts the bucket in the hostname itself and the
bucket-qualified form otherwise requests it twice and fails with `NoSuchKey`.

### Serving the portfolio

`POST /rainfall` returns a cached series and is **not** subject to the synchronous
area or vertex caps, because it reads a cache rather than pixels. Give it a
polygon or a `cache_key`; the key form needs no geometry at all.

```bash
curl -s -X POST localhost:8001/rainfall -H 'content-type: application/json' \
     -d '{"cache_key":"675e97f48f735dccb2fd8707fe65bbc4"}'
```

A miss returns `not_computed` with the reason. The build path and the lookup path
share one canonicaliser, so a published series is always derivable; a
self-intersecting boundary is repaired on both sides and reported as
`geometry_repaired`.

`POST /admin/rainfall/forget` clears **both** stores: the local file and the
published object. It reports `local` and `remote` separately so a partial failure
is visible rather than assumed.

## Usage events

Set `USAGE_EVENTS_PATH` to append one JSON line per analysis. Unset, events go to
stdout only.

```bash
USAGE_EVENTS_PATH=/app/usage-events.jsonl
```

Each record carries the datasets requested, a per-module verdict, per-module and
total durations, a **coarse area band**, the sensor used, and a truncated client
prefix. It deliberately carries no submitted geometry, no raw area, no
measurement values and no full IP address, so it cannot be used to work out who
asked about which piece of land. `python -c "import usage, json; print(json.dumps(usage.summarise(usage.read_events()), indent=1))"`
aggregates a batch, which is the only shape these are meant to be read in.

`POST /admin/rainfall/forget` deletes a cached rainfall series for one area,
given a geometry or a 32-character cache key. The cache is keyed by a hash of the
submitted geometry, so it is a record about a specific place; this is how a
removal request is honoured. The endpoint is unauthenticated because it deletes
only recomputable derived data, and the API is loopback-bound.

## Worker and alerts

`docker compose up -d` starts a `worker` alongside the backend. It drains the
submission queue, checks every published series against the alert rules, and sleeps
`RAINFALL_WORKER_INTERVAL` seconds (default 300). It shares the
`rainfall-cache` volume with the backend, so a series it writes is one the API
serves.

It holds an `flock` for the whole run, so a second worker exits instead of
duplicating ERA5 reads, and it exits after three consecutive sweep failures so the
restart policy can act. A single failure, or one bad area, does not trip it.

To notify rather than only record, set a webhook. Anything that accepts a JSON
POST works, including Slack and Discord:

```bash
RAINFALL_ALERT_WEBHOOK=https://hooks.slack.com/services/...
RAINFALL_ALERT_RULES=[{"id":"severe-drought","metric":"trailing_12m_anomaly_pct","below":-40}]
```

The payload is Slack-shaped and also sends `content`, so Mattermost, n8n, Zapier
and Discord all work unchanged.

**Email** carries the same messages to people, including a notice when a queued
job finishes, which a browser tab that has been closed cannot deliver:

```bash
AGENTMAIL_API_KEY=am_...
AGENTMAIL_INBOX_ID=inbox_...
ALERT_EMAIL_TO=you@example.org
```

Neither channel is required. Unconfigured, both are no-ops and the sweep is
unaffected.

Rules are evaluated against each calendar month's 1991-2020 normal, so a
percentage means the same thing in a wet and a dry month. An alert fires once per
episode, and a recovery is announced too. Episodes are recorded whether or not a
webhook is configured, so adding one later does not re-announce alerts that are
already open.

Reconcile the bucket against the areas you expect with:

```bash
python -m tools.sync_rainfall_cache --prune areas.geojson      # lists
python -m tools.sync_rainfall_cache --prune areas.geojson --yes
```

## Nginx

Install `deploy/nginx/geocontextualize-rate-limit.conf` under
`/etc/nginx/conf.d/`. Use `geocontextualize-http.conf` while issuing a
certificate, then replace it with `geocontextualize-ip.conf` (or an equivalent
named-host configuration). Test every change before reloading:

```bash
nginx -t && systemctl reload nginx
```

The proxy strips the `/api/` prefix before forwarding to FastAPI, applies
per-IP request and connection limits, and keeps Docker service ports private.

## Verify and maintain

```bash
curl --fail https://209.38.197.161/api/health
docker compose logs --tail=100 backend
docker compose up -d --build
```

The API admits only polygonal GeoJSON. Defaults are a 500 KB payload, 10,000
vertices, and measured area budgets: 100 km² for a synchronous study area and
for NDVI, and 1,000 km² for land cover. The application returns an explicit
`skipped` NDVI result or a `landcover_area_exceeded` error rather than degrading
silently; large asynchronous analyses require a separately deployed durable
queue and worker.

Every budget is overridable by environment variable. They fall into three groups,
and only the first are limits on input:

- **Guards**, each with a measured failure behind it: `MAX_GEOJSON_BYTES`,
  `MAX_AOI_VERTICES`, `MAX_SYNC_BBOX_KM2`, `MAX_LANDCOVER_BBOX_KM2`,
  `MAX_SOURCE_TILES`, `MAX_CONCURRENT_ANALYSES`, `MAX_CONCURRENT_NDVI`.
- **Timeouts**, which bound waiting rather than input: `ANALYSIS_ACQUIRE_SECONDS`,
  `NDVI_ACQUIRE_SECONDS`, `ANALYSIS_DRAIN_SECONDS`.
- **Product defaults**, which are choices rather than limits: `MAX_PC_SCENES`,
  `NDVI_TARGET_EPSG`.

There is no separate vegetation cap: it duplicated the synchronous cap at the same
value. See [Measured limits](README.md#measured-limits) for the measurements.

`client_max_body_size` must be **at least** `MAX_GEOJSON_BYTES` in both Nginx
configs. A lower proxy limit does not protect anything, it only rejects a body
the application would have accepted, with a less explicable reason.

`MAX_CONCURRENT_ANALYSES` defaults to 8 and `MAX_CONCURRENT_NDVI` to 3. The
service is bound by read latency against Planetary Computer rather than by local
resources: eight concurrent analyses peak at 301 MB RSS and 14% of one core. The
Nginx 2-second `limit_req` is the outermost throttle, so raise it alongside
`ANALYSIS_ACQUIRE_SECONDS` or clients will see 503s before the application
rejects anything. An 1,800 MB `mem_limit` is sufficient headroom for these limits;
if you raise concurrency substantially, re-measure before assuming it still is.

Open only SSH, HTTP, and HTTPS in the firewall after confirming the Nginx
route. Do not expose ports 3000 or 8001 publicly.
