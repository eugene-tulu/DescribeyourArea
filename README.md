# GeoContextualize

GeoContextualize turns a polygonal study area into a concise geospatial
context: terrain from NASADEM, land cover from ESA WorldCover, and a bounded
Sentinel-2 NDVI composite from Microsoft Planetary Computer, plus the country
containing the area.

See [CHANGELOG.md](CHANGELOG.md) for what changed, why, and the measurements
behind each decision, including options that were evaluated and rejected.

## Reliable-by-default analysis

- Accepts GeoJSON `Polygon`, `MultiPolygon`, `Feature`, and `FeatureCollection`.
  FeatureCollection polygons are combined rather than silently discarding all
  but the first feature.
- Rejects malformed, oversized, or overly complex inputs before calling an
  external service.
- Caps the **bounding box** of a synchronous study area at 100 km², NDVI at
  100 km², and land cover at 1,000 km². These are measured, not guessed: see
  the derivation table in `main.py` and [Measured limits](#measured-limits).
- Accumulates statistics across every source tile that intersects the study
  area, so an area spanning several tiles is not silently analysed from one.
- Returns a `valid_pixel_count` and `valid_pixel_fraction` with every raster
  module, so a caller can tell a real result from a thinly covered one.
- Rejects an empty or unsupported `datasets` selection with `422` rather than
  returning an empty summary.
- Searches at most four recent Sentinel-2 scenes, takes the least-cloudy
  candidates first, and clips to the submitted geometry before calculating the
  median.
- Uses Planetary Computer only for NDVI. There is no EOPF or unsafe full-raster
  MODIS fallback, and no datacube layer: bands are read as COG windows.
- Limits concurrent analyses so one request cannot exhaust the server or shared
  public data services.

Large study areas are not silently downgraded: NDVI returns a clear `skipped`
status. Supporting large asynchronous analysis needs a durable job queue and
worker, which is deliberately outside this synchronous service.

## Measured limits

The area caps are derived from measurements against Planetary Computer taken on
2026-09-26, reporting peak RSS delta per request from a fresh process:

| Bounding box | nasadem 30 m | worldcover 10 m | sentinel-2 20 m |
| --- | --- | --- | --- |
| 10 km² | 2.4 s / 17 MB | 3.6 s / 21 MB | 16 s / 33 MB |
| 60 km² | 2.4 s / 18 MB | 3.0 s / 42 MB | 16–23 s / 3–53 MB |
| 100 km² | 2.4 s / 18 MB | 3.0 s / 42 MB | 32 s / 423 MB (odc-stac) |
| 1,000 km² | 3.7 s / 31 MB | 6.2 s / 235 MB | exceeds the 75 s budget |
| 5,500 km² | 5.2 s / 69 MB | 22.3 s / 1,205 MB | — |

Only land cover has a genuinely area-driven memory curve, which is why it has
its own 1,000 km² budget.

The NDVI path reads Sentinel-2 bands directly from COG windows rather than
building a datacube. It previously used `odc-stac`/dask, whose ~372 MB floor was
framework overhead rather than pixels and dominated concurrency. The rewrite also
removes `dask`, `xarray`, `rioxarray` and `odc-stac` from the runtime entirely.
Two details carry most of the remaining cost:

- **COG overviews are read explicitly.** B04/B08 are published at 10 m while the
  composite is built at 20 m. Requesting an overview band, with the *window*
  expressed on the overview's own grid, cut a single band read from 11.0 s to
  2.5 s. Passing only `out_shape` does not achieve this.
- **The composite is built in EPSG:6933** (WGS 84 / NSIDC EASE-Grid 2.0 Global),
  an equal-area projection in metres, so a pixel is the same area everywhere and
  a reported percentage means the same thing in Kenya as in Canada. A degree grid
  cannot honour a metres-per-pixel request, because a degree of longitude
  shrinks with latitude.

Two consequences worth knowing:

- **The cap applies to the bounding box, not the polygon.** An irregular outline
  is penalised by the area of its own bounding rectangle. II Ngwesi is a real
  89 km² conservancy whose bounding box is 120 km², so the synchronous path
  rejects it. Raise `MAX_SYNC_BBOX_KM2` rather than shrinking the polygon.
**The binding constraint is remote read latency, not memory or CPU.** Measured
through the real ASGI app at N=1/2/4/6/8 on 1.25 vCPU:

| | N=1 | N=2 | N=4 | N=8 |
| --- | --- | --- | --- | --- |
| Peak RSS, 8 requests in flight | | | | 301 MB |
| CPU, 8 requests in flight | | | | 14% of one core |
| dem+landcover wall time | 7.0 s | 6.9 s | 6.8 s | **4.4 s** |
| dem+landcover+ndvi p50 latency | 33.3 s | 24.3 s | 38.7 s | **47.8 s** |

Marginal memory is about **25 MB per concurrent request** — 301 MB total at
N=8 — and CPU never passes 14% of one core. Neither is what limits the service.
Reading from Planetary Computer is.

So there are two guards rather than one:

- `MAX_CONCURRENT_ANALYSES` (default 8) bounds total load. Requests that cannot
  enter within `ANALYSIS_ACQUIRE_SECONDS` (2 s) get a 429 rather than a queue.
- `MAX_CONCURRENT_NDVI` (default 3) bounds the NDVI path specifically, because
  it is the only one whose latency degrades. `dem+landcover` scales flat to
  N=8; NDVI p50 latency was 52 s at N=4 and 67 s at N=8 against a 75 s budget.

A caller that cannot enter the NDVI guard within `NDVI_ACQUIRE_SECONDS` (20 s)
still receives terrain and land cover, with NDVI marked unavailable and a "busy"
reason. Degrading one module is better than failing the whole request.

The limit trades user-facing latency for throughput rather than being free: at
N=8 a full analysis takes ~48 s instead of ~30 s, but throughput rises from
roughly 0.035 to 0.145 requests per second.

## How NDVI is computed

A median composite over up to four Sentinel-2 scenes, in this order:

1. Scenes are searched over the last 90 days, filtered on scene-level cloud
   cover, then **sorted least-cloudy first**. Scene cloud cover describes the
   whole 110 km tile, so a scene can score near zero and still be fully overcast
   over the study area.
2. B04, B08 and SCL are read as COG windows decimated to 20 m.
3. NDVI is computed per scene, then reprojected onto one EPSG:6933 grid.
4. The study-area polygon is reprojected and rasterised onto that grid.
5. The per-pixel median across scenes is summarised as mean/min/max/std/p25/p75.

**Scene-class masking.** Classes 0, 1, 3, 8, 9, 10 and 11 are rejected
outright. Classes 4 (cloud) and 5 (bright cloud) are **not**: Sentinel-2's
brightness test for class 5 flags bright semi-arid ground and desert as cloud
across entire tiles — a Sahara tile reads 100% class 5 while its B04/B08
reflectance (0.41/0.49) and NDVI (~0.09) are plainly desert. Rejecting them
empties the result for exactly the rangeland this service exists to serve.
Residual cloud is removed by discarding implausible NDVI instead, because both
cloud and open water give a near-zero or negative index.

**Known limitation.** Because classes 4/5 are kept, NDVI over a persistently
cloudy area is depressed by residual thin cloud. The reported
`valid_pixel_fraction` describes AOI coverage, not cloud contamination. Treat a
figure below roughly 0.2 as uncertain rather than as bare ground, and say so in
anything EIA-facing.

## Local setup

```bash
cp .env.example .env
python -m pip install -r requirements.txt
uvicorn main:app --reload
```

In a second terminal:

```bash
cd client
npm ci
npm run dev
```

The frontend uses `/api` by default. Next.js rewrites that path to the local
backend in development. Set `NEXT_PUBLIC_BACKEND_URL` only when intentionally
using a different API origin.

## API

`POST /generate-context` accepts a JSON body with `geojson` and optional query
parameters:

- `datasets=dem,landcover,ndvi` selects which data modules run. Available
  values are `dem`, `landcover`, and `ndvi`; omitted defaults to all three. An
  empty selection is rejected with `422`.
- `include_ndvi=false` skips NDVI even if it is selected.

`GET /health` reports service readiness and `GET /version` describes active
limits.

## Production

See [DEPLOYMENT.md](DEPLOYMENT.md). The Compose configuration binds app ports
only to loopback and expects Nginx to proxy the frontend and `/api/` route.
