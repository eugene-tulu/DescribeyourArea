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

Every budget is overridable by environment variable — `MAX_SYNC_BBOX_KM2`,
`MAX_NDVI_BBOX_KM2`, `MAX_LANDCOVER_BBOX_KM2`, `MAX_SOURCE_TILES`,
`MAX_CONCURRENT_ANALYSES`, `MAX_CONCURRENT_NDVI`, `ANALYSIS_ACQUIRE_SECONDS`,
`NDVI_ACQUIRE_SECONDS`, and `ANALYSIS_DRAIN_SECONDS`. See
[Measured limits](README.md#measured-limits) for the measurements behind them.

`MAX_CONCURRENT_ANALYSES` defaults to 8 and `MAX_CONCURRENT_NDVI` to 3. The
service is bound by read latency against Planetary Computer rather than by local
resources: eight concurrent analyses peak at 301 MB RSS and 14% of one core. The
Nginx 2-second `limit_req` is the outermost throttle, so raise it alongside
`ANALYSIS_ACQUIRE_SECONDS` or clients will see 503s before the application
rejects anything. An 1,800 MB `mem_limit` is sufficient headroom for these limits;
if you raise concurrency substantially, re-measure before assuming it still is.

Open only SSH, HTTP, and HTTPS in the firewall after confirming the Nginx
route. Do not expose ports 3000 or 8001 publicly.
