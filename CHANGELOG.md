# Changelog

All notable changes to GeoContextualize, newest first. This file is the running
record of what changed, why, and what evidence supported it. Update it on every
major change or milestone.

## Conventions

Each entry records:

- **What** changed, at file:line granularity where it matters.
- **Why** — the defect, constraint, or measurement that prompted it.
- **Evidence** — a measurement, a reproduction, or a real-data test. Changes
  without evidence belong in the code, not here.
- **Supersedes** where a decision was later reversed. Reversals are kept, not
  deleted: knowing what was tried and why it failed is worth more than a clean
  history.

Verification commands used throughout:

```bash
# backend, 39 offline tests
~/miniconda3/envs/ndvi/bin/python -m unittest tests.test_raster_pipeline tests.test_aoi_policy

# backend, full suite including live Planetary Computer tests (needs network)
~/miniconda3/envs/ndvi/bin/python -m unittest discover -s tests

# frontend
cd client && npx tsc --noEmit && npm run lint && npm run build
```

---

## 1.7.0 — Deployment configuration aligned with the code; Spaces wired end to end

### A real deployment bug, found while prefilling `.env`

`docker-compose.yml` pins defaults in its `environment:` block, and that block
**overrides** `env_file` and the compiled defaults. It had been left at the
pre-1.4.0 values while the code moved on:

```yaml
MAX_NDVI_BBOX_KM2: ${MAX_NDVI_BBOX_KM2:-10}        # code default is 100
MAX_CONCURRENT_ANALYSES: ${MAX_CONCURRENT_ANALYSES:-1}  # code default is 8
```

So the container would have run with a 10 km² NDVI cap and a single concurrency
slot, and **the entire 1.4.0 concurrency work would have been dead in
production** while `/version` reported the tuned values from the code. Nothing
would have failed; it would simply have served 429s at the old rate.

All twelve injected values are now declared and asserted equal to the code
defaults, so the two cannot drift apart again. Also aligned: the payload cap
against Nginx's 512k, and the cache directory against a writable non-root path.

### Environment files

- **`.env.example` rewritten.** It was documenting the pre-1.4.0 values and was
  missing six variables. It now lists every knob, the shipped default, and what
  it controls, with a note that `MAX_GEOJSON_BYTES` must not exceed Nginx's
  `client_max_body_size`.
- **`.env` prefilled** with the Spaces variables and the web settings, using
  explicit `REPLACE_WITH_...` placeholders for the bucket and keys. The bucket
  name and credentials are not guessable and were not invented.
- **The dead `GEMINI_API_KEY` removed** from `.env`. It had no effect since 1.3.0
  removed the narrative module.

### Container entrypoint

`docker-entrypoint.sh` pulls the portfolio before serving, then `exec`s the server.
A missing or unreachable remote is **not fatal** — the service starts and rainfall
reports "not computed" for areas it lacks. Two tests execute the script with a
failing remote and assert the command after it still runs, rather than
string-matching the source.

### Test flake distinguished from a defect

The real-data NDVI test failed once with a 75 s timeout — a slow CDN, not a code
fault. But four separate defects surfaced during development as a polite
"unavailable", so the test cannot simply tolerate it. It now retries **once**, and
only for a timeout; any other failure mode still fails outright.

---

## 1.6.0 — Rainfall portfolio published to an S3-compatible store

The rainfall series is a **build artefact**, so it has to exist somewhere durable
and outside the image. Spaces is the natural home: it is S3-compatible, already
running, and the portfolio is tiny — **21 conservancies, 590 KB, ~25 KB each**.

### Design: the hot path never touches the network

`cached_context()` reads the local file first and consults the remote **only on a
miss**, caching the result locally so the next request is a plain file read again.
Two consequences worth relying on:

- A cache hit gains no latency and no new failure mode.
- A miss stays a miss: an unreachable remote returns `not_computed`, so a Spaces
  outage degrades to "not processed yet" rather than a failed request. Verified
  against a stub client that raises.

### Interfaces

```bash
# build and publish
python -m tools.precompute_rainfall --conservancies areas.geojson --start 2010-01-01 --publish

# deploy: fetch before serving
python -m tools.sync_rainfall_cache --pull
```

Configured with `RAINFALL_CACHE_S3_URI`, plus optional `RAINFALL_S3_ENDPOINT` and
`RAINFALL_S3_REGION`; credentials come from the standard `AWS_ACCESS_KEY_ID` /
`AWS_SECRET_ACCESS_KEY` pair. **All optional** — unset, everything stays local and
every remote call is a no-op, so nothing about local development changes.

The per-read cell cache under `cells/` is a build accelerator, not an artefact, and
is deliberately not transferred.

This is also the mechanism self-service submission needs later: a submitted polygon
gets computed and published, and the next deployment or cache miss picks it up with
no rebuild.

### Failure modes and their exits

- No remote configured: exit 2 with the variables to set.
- Unreachable or misconfigured remote: exit 1 with the client error and the four
  knobs to check, rather than a botocore traceback.
- `--publish` with no remote configured: refuses rather than silently skipping.

### One bug found and fixed

`_remote_key` built the object key from the whole configured prefix, which
duplicated the bucket name into the key — every upload and fetch addressed a
non-existent object. The bucket is not part of an S3 key, so it is now stripped;
the test asserts the exact key.

Ten new offline tests cover the remote layer against a stub client: no-op when
unconfigured, URI parsing, rejection of a malformed URI, upload, fetch-then-cache,
stale-version rejection, survival of an unreachable remote, miss-then-remote
fallback, and that **a local hit makes no remote call at all**.

---

## 1.5.0 — Rainfall and drought anomaly from ERA5

The indicator asked about twice in the conservancy webinar ("can we see how the
97/98 situation was, along the Ewaso Ng'iro" and "what does the dashboard have
for drought preparedness"), and effectively absent from the incumbent dashboard.

### Data source: why ERA5, and from where

- **ERA5, read from the Earthmover Icechunk store on AWS Open Data**
  (`s3://earthmover-icechunk-era5/icechunkV2`). It is **public and anonymous — no
  account, no AWS credentials, no login** — CC-BY 4.0, CF-compliant, 1940 to
  present, 35 surface fields, and citable (`doi.org/10.24381/cds.adbb2d47`).
- **Not the Planetary Computer copy.** `era5-pds` there is the deprecated
  1979–2020 subset: a search for 2024 returns zero items. Earthmover's own
  comparison table lists it as "same stale subset as ERA5-PDS, no longer
  maintained".
- Note the Earthmover *docs sample* (`earthmover-public/era5-surface-aws`) is a
  different, smaller repo with 18 variables and no precipitation. The registry
  repo used here has 39 variables including `tp`.
- Requires `xarray`, `zarr`, `icechunk` and `pcodec` — **offline build only**. The
  request path reads a cache and needs none of them.

### Why it is precomputed

Reading a single ERA5 grid cell over the 30-year climatology takes about
**20 seconds**, against a whole request budget of 20–30 seconds. So the request
path never computes: `rainfall.cached_context()` returns a cached series, or an
explicit `not_computed` with the reason and the cache key. Computing is
`rainfall.build_and_cache()`, driven by `tools/precompute_rainfall.py`.

### The union read

A portfolio of adjacent areas would pay that 20 s once per area. The 21 Northern
Rangelands Trust conservancies need 11 distinct cell sets, whose **union is only
9 longitudes × 2 latitude bands**, so `UnionReader` reads the union once and
slices each area's cells out of it. Measured: **47.2 s for the first area, then
0.0–0.1 s for the other 20** — 48 s for the whole portfolio against roughly 16
minutes naive. The ERA5 `temporal` chunks are `time=8736, lat=12, lon=12`, so a
small read still decodes a year of a 12×12 tile, which is why a wide read is
nearly free.

### Results, and what they show

Built for all 21 conservancies, 195 months each, 0 failed, 0 months dropped,
142,416/142,416 hours valid per area.

Annual normals run **894 mm (Naibunga Upper, on the Mau Escarpment) down to
289 mm (Biliqo Bulesa, arid Baringo)** — the correct gradient. The monthly
climatology is textbook bimodal East African: long rains March–May (268 mm),
short rains October–December (253 mm), dry June–August (63 mm). A test asserts
that shape, because a units or aggregation error destroys it.

Requested to cover 1995–2000 over Baringo, the trailing-12-month anomaly peaks at
**+74.3% ending September 1998** — the 1997/98 El Niño, which Sentinel-2 cannot
reach. The 1999–2000 dip to about −49% is a real signal, not an artefact: those
years rank among the driest in the full 86-year record alongside documented
regional droughts in 1942–45, 1984, 2008/09 and 2022.

### Guards, each added after being bitten

- **Plausibility range.** An off-by-1000 bug produced an annual "normal" of
  **0.8 mm** for semi-arid rangeland, which reads as catastrophic drought and is
  absurd to anyone who knows the region. `_assert_plausible` now refuses anything
  outside 20–12,000 mm/yr.
- **Hourly coverage per month.** A month is dropped unless ≥80% of its hours are
  present and finite, so a mostly-missing month is never reported as a dry one.
  `valid_hours`/`expected_hours` and the dropped list are published in the payload.
- **Suspect-month flagging.** A month under 0.5 mm over a cell whose normal
  exceeds 5 mm is flagged `near_zero_month_worth_review` and surfaced in the
  summary, rather than silently dropped — real extremes should be visible.
- **`processing_version` on every cache entry.** Bumping it invalidates everything,
  so a cached value can never outlive the code that produced it.

### Bugs found while building it

Recorded because each produced a confident, plausible, wrong number:

1. **Hourly timestamps labelled as months** — 0.8 mm/yr instead of 735.
2. **A per-month counter used as an index into the global time axis**, so every
   month after the first read the wrong hours. Symptom: the seasonal cycle
   flattened to three repeating values and the annual total roughly doubled.
3. **`np.savez_compressed` appends `.npz` to a path that lacks it**, which broke
   the atomic rename and failed every write.
4. **Cell slicing on the wrong axis** — `matrix[rows][:, cols]` selects the time
   axis, taking the first N *months* instead of the area's cells.
5. **Iterating a `(months, lat, lon)` array yields single 2-D months**, not the
   whole array.
6. **A stale cache in the wrong directory, which cost the most time.** The code
   was correct and the artefact I was deleting was not the one being read: several
   commands cleared `/tmp/kilo/raincache` while the module defaulted to
   `./.rainfall-cache`. The 1611 mm result survived a cache wipe, a version bump
   and a full rewrite because the wiped directory was never consulted. Fixed by
   gitignoring the cache, always setting `RAINFALL_CACHE_DIR` explicitly, and
   treating a "fix" that changes nothing as evidence the fix is not the problem.
7. **Two hard-coded copies of the dataset set** (in `main.py` and a test)
   drifted when rainfall was added. Both are now derived from
   `AVAILABLE_DATASETS`.

A hypothesis that measurement refuted: suspected months were assumed to be NaN
gaps, but the coverage report shows 0 missing hours. The near-zero values are
real, and 447,057 of 756,048 hours are exactly 0.0 because most hours have no
rain.

### Surface

- `rainfall` is the fourth selectable dataset, default-on in the UI.
- New card: last 12 months, percentage against the 1991–2020 normal, the annual
  normal, the driest month, and a provenance line carrying source, grid resolution
  in km, cell count and retrieval date. Suspect months surface in amber.
- The copy/summarise text now states the trailing-12-month anomaly in words.
- `/version` lists the dataset; the rainfall card appears in the text summary.

---

## 1.4.0 — Concurrency limits re-derived from measurement

### What changed

- `MAX_CONCURRENT_ANALYSES` raised from **1 to 8**.
- New `MAX_CONCURRENT_NDVI`, default **3**, guarding the NDVI path separately.
- New `ANALYSIS_ACQUIRE_SECONDS` (2 s) and `NDVI_ACQUIRE_SECONDS` (20 s) replace
  the hard-coded 2-second wait, so the two guards can be tuned independently.
- A caller that cannot enter the NDVI guard still receives terrain and land cover,
  with NDVI marked unavailable and a "busy" reason. Previously the whole request
  would have been rejected.
- Both limits exposed on `/version`.

### Why — and a wrong turn of my own

The previous entry stated that a 100 km² request costs ~480 MB and that an
1,800 MB container therefore sustains ~3 concurrent analyses. **That was wrong.**
It came from per-process `ru_maxrss` deltas, which are high-water marks for a
one-shot process and include first-touch allocation, not from the marginal cost
of a request. Measuring N=1/2/4/6/8 concurrent requests through the real ASGI
app gives:

| | N=1 | N=2 | N=4 | N=6 | N=8 |
| --- | --- | --- | --- | --- | --- |
| Peak RSS (all datasets) | 265 MB | 265 MB | 288 MB | 298 MB | **301 MB** |
| CPU, share of one core | 3% | 9% | 15% | 11% | **14%** |
| dem+landcover wall time | 7.0 s | 6.9 s | 6.8 s | 5.0 s | **4.4 s** |
| dem+landcover+ndvi p50 latency | 33.3 s | 24.3 s | 38.7 s | 50.3 s | **47.8 s** |

Marginal memory is **~25 MB per concurrent request**, and CPU never passes 14% of
one core. **Neither memory nor CPU is the binding constraint — remote read
latency is.** The single limit sized on memory was the wrong instrument.

The two paths also behave completely differently, which is why there are two
guards rather than one:

- `dem+landcover` is I/O-bound and scales *better* with concurrency: wall time
  falls from 7.0 s to 4.4 s as N goes 1→8, because requests overlap their waits.
- `dem+landcover+ndvi` degrades: p50 latency 52 s at N=4 and 67 s at N=8 against
  a 75 s budget. Left unbounded, a burst of eight NDVI requests would have timed
  most of them out.

### Result

Eight concurrent full analyses, all returning 200, in 55.2 s wall with p50 latency
47.8 s. Throughput rises from roughly **0.035 to 0.145 requests per second, about
4×**, and no request is refused. The limit is not free — a full analysis now takes
~48 s under load instead of ~30 s — so the numbers are a latency-for-throughput
trade, documented as such.

### Operational note

The Nginx `limit_req` zone is 5 requests per second with a 10-second burst, which
is looser than anything the application now does. If concurrency is raised
further, that outer throttle will return 503s before the application's own guards
engage, so raise both together.

---

## 1.3.0 — NDVI path rewritten; five features removed; correctness pass

The largest change so far. Two separate efforts: a code review that found seven
silent wrong-answer defects, and a rewrite of the NDVI path that cut its memory
by an order of magnitude.

### Removed

- **Dropped four external datasets and the Gemini narrative**: soil organic
  carbon, population, climate normals, hydrology, and AI-generated narrative
  text. `main.py` went from 1,044 to 727 lines as a result.
- Deleted `prompts/study_area_v1.txt` and `prompts/study_area_v2.txt` with the
  narrative code, and `.github/ISSUE_TEMPLATE/sprint-2-extra-datasets.md`, which
  specified the reverted sprint.
- Removed `google-generativeai` and `overpy` from `requirements.txt`.

This also disposed of four defects that the review found in the removed code:
`get_aoi_centroid` raised `KeyError` on every call (climate was permanently
`None`); `fetch_hydrology` queried Overpass with `out body`, which does not
populate element geometry, so it always reported zero water; the `audience`
parameter was accepted and then never interpolated into the prompt; and the NDVI
citation block read singular `scene_date`/`scene_id` keys that no code ever
wrote.

### Correctness fixes

- **Per-tile statistic accumulation.** `_find_core_assets` searched with
  `limit=1` and returned a single tile regardless of how many intersected the
  study area. Measured against Melako (5,505 km², 4 NASADEM tiles): the single
  returned tile covered **3.0% of the bounding box**, so the reported elevation
  statistics described a fraction of the requested area with no error raised.
  Now every intersecting tile is collected (bounded by `MAX_SOURCE_TILES=64`) and
  statistics accumulate per tile via sum/count/sum-of-squares. Mosaicking is
  deliberately avoided: a zonal mean does not need it and it would multiply peak
  memory by the tile count.
- **All-nodata guard.** `compute_raster_stats` returned `float(np.nanmean(...))`.
  Pydantic serialises NaN as `null`, and `interpret_terrain` compared NaN against
  thresholds — every comparison is False, so an all-nodata raster fell through
  to the `else` branch and reported `terrain_type: "highly variable or
  mountainous"`. Now an empty tile returns a stable error code, and terrain
  interpretation rejects any non-finite statistic.
- **No credential leakage.** `compute_raster_stats` and
  `compute_landcover_percentages` returned `{"error": str(e)}` to the client.
  rasterio and GDAL error text routinely embeds the URL being read, and these are
  `planetary_computer.sign()` outputs **with SAS tokens**. Errors now return
  stable codes; the detail goes to the server log. The generic 500 no longer
  echoes exception text. A real-data test asserts `sig=`, `skoid=`,
  `blob.core.windows.net` and `se=20` never appear in a response body.
- **Empty dataset selection rejected.** `datasets=` parsed to an empty set and
  returned 200 with every module null. Now a 422.
- **Latitude-aware NDVI grid.** The metres-to-degrees conversion used a fixed
  `111320.0` divisor, so a "20 m" cell was ~40 m on the ground at 60°N. Now
  divided by `cos(latitude)`, with a polar clamp because `cos(90°)` would blow
  the cell size up.
- **Concurrency guard covers real work lifetime.** `asyncio.wait_for` cancels the
  *await*, not the thread, so the semaphore was released while a timed-out
  analysis kept occupying a thread, memory, and bandwidth to Planetary Computer.
  `run_blocking` now uses a shielded executor future with a done-callback, and
  the middleware drains outstanding work before releasing the guard. Tested with
  real threads and events: the count stays at 1 after cancellation.
- **Frontend graticule leak.** The graticule was added inside the draw-control
  effect, whose handlers changed identity on every parent render, so a new
  graticule was added and never removed. Moved to its own `[map]`-only effect
  with a `removeLayer` cleanup, and the two callbacks are now `useCallback`-stable.
- **Reversed a wrong turn of my own.** I claimed the raster functions called
  `src.read(1)` on the whole tile and that this cost 1.30 GB per request. The
  measurement was real (an ESA WorldCover tile genuinely is 36,000 × 36,000 and a
  full read genuinely costs 1.296 GB, 1,818 MB RSS, 133.6 s) but the production
  code uses `mask(..., crop=True)`, which is already windowed. I had asserted it
  without reading the function, and drew a deployment conclusion on top of it.
  Superseded; the 1.3 GB figure is a hazard to avoid, not a bug that existed.

### Measured caps, replacing unvalidated defaults

`MAX_SYNC_BBOX_KM2 = 100.0` and `MAX_NDVI_BBOX_KM2 = 10.0` had no derivation
anywhere — only a comment stating a governance intent. `git log -S` found no
introducing commit, and the `benchmark_*.json` files measured a loading
micro-benchmark of an EOPF/Zarr path that is not in the code. They were
unvalidated defaults.

Re-derived from measurement (peak RSS delta per request, fresh process):

| Bounding box | nasadem 30 m | worldcover 10 m | sentinel-2 20 m |
| --- | --- | --- | --- |
| 10 km² | 2.4 s / 17 MB | 3.6 s / 21 MB | 22.9 s / 394 MB *(odc-stac)* |
| 100 km² | 2.4 s / 18 MB | 3.0 s / 42 MB | 31.8 s / 453 MB *(odc-stac)* |
| 1,000 km² | 3.7 s / 31 MB | 6.2 s / 235 MB | exceeds the 75 s budget |
| 5,500 km² | 5.2 s / 69 MB | 22.3 s / 1,205 MB | — |

Only land cover has a genuinely area-driven memory curve, so it received its own
`MAX_LANDCOVER_BBOX_KM2 = 1000.0`. NDVI is bounded by its time budget, not its
area — its ~372 MB floor was framework overhead — so `MAX_NDVI_BBOX_KM2` moved
from 10 to **100**, where it measures 32 s against a 75 s budget. All caps are
now env-overridable and the derivation is in `main.py` next to the constants.

**Documented side effect:** the cap applies to the *bounding box*, not the
polygon, so an irregular outline is penalised by the area of its own rectangle.
II Ngwesi is a real 89 km² conservancy whose bounding box is 120 km², so the
synchronous path rejects it; Melako's 5,505 km² polygon has a 9,237 km² bounding
box. A test records this.

### NDVI path rewritten

Replaced the `odc-stac`/dask cube with direct COG window reads. Measured at
60 km²: **423 MB → 3–53 MB**, **20–32 s → 16–23 s**. `dask`, `xarray`,
`rioxarray` and `odc-stac` are gone from `requirements.txt`, verified via
`sys.modules`.

- **COG overviews are read explicitly.** B04/B08 are published at 10 m while the
  composite is 20 m. GDAL serves a request from an overview only when the
  *window* is expressed on the overview's own grid; passing `out_shape` alone
  did nothing. A single band read went **11.0 s → 2.5 s**.
- **Composite grid is EPSG:6933** (WGS 84 / NSIDC EASE-Grid 2.0 Global) —
  equal-area, in metres, so a pixel is the same area everywhere and a reported
  percentage means the same thing in Kenya as in Canada, with no UTM zone-edge
  stitching. Valid only to ~86° latitude, so polar areas fall back to a local UTM
  zone.
- **Scenes are sorted least-cloudy first.** Scene-level `eo:cloud_cover` describes
  the whole 110 km tile, so a scene scoring 0.00006% was still fully overcast
  over the study area.
- Every raster module now reports `valid_pixel_count` and `valid_pixel_fraction`,
  making AOI coverage observable rather than implied.

### Scene-class masking corrected

The old mask was `isin([4, 5, 6, 7, 11])`, which **keeps** class 4 (cloud) and
5 (bright cloud) and rejects 2 (dark) and 3 (shadow). The previous NDVI figures
over cloudy areas were therefore cloud-contaminated, and legitimate
dark-vegetation pixels were discarded.

Correcting it naively emptied every result. Control-testing against the Sahara
explained why: **Sentinel-2 class 5 flags bright semi-arid ground and desert as
"bright cloud" across entire tiles.** A Sahara tile reads 100% class 5 while its
B04/B08 reflectance (0.41/0.49) and NDVI (~0.09) are plainly desert. Rejecting
class 5 wholesale destroys the result for exactly the rangeland this service
targets.

Current policy: reject only the unambiguous classes (`0, 1, 3, 8, 9, 10, 11`) and
remove residual cloud by discarding implausible NDVI, since both cloud and open
water give a near-zero or negative index. A real-data test asserts the Sahara
returns `ok` with mean NDVI < 0.4 — sparse vegetation, not empty.

### Bugs found while building the rewrite

Four of my own, all of which failed silently as a polite `"unavailable"` result:

1. **Inverted decimation condition** — triggered only when the source was
   *coarser* than the target, so B04/B08 were read at full 10 m resolution.
2. **Height/width transpose in `geometry_mask`** — `out_shape` needs
   `(rows, cols)`, so the mask was transposed.
3. **Geometry mask not reprojected** — `geometry_mask` does not reproject, and a
   WGS84 geometry was passed with a projected transform, so the mask matched
   nothing.
4. **`dst_nodata=np.nan` in `reproject`** — GDAL then treats every output pixel
   as no-data and writes nothing. Verified in isolation: 0 valid pixels with
   `dst_nodata=nan` versus 14,300 with `dst_nodata=None`.

The integration test was therefore strengthened to require `status == "ok"` for a
known-good AOI rather than merely accepting `"unavailable"`. The lenient
assertion would have shipped all four.

### Tests

From 9 to **67**, in two suites:

- `tests/test_raster_pipeline.py` — 39 offline tests: terrain classification and
  the NaN guard, dataset selection, the cap derivation, latitude-aware
  resolution, the in-flight guard, the SCL policy, target-grid construction, and
  geometry-mask orientation. No network, no mocks.
- `tests/test_planetary_computer.py` — 28 tests against live MPC using the real
  Northern Rangelands Trust conservancy polygons. Melako spans four NASADEM
  tiles and is the case that used to break; a test asserts the accumulated read
  covers materially more of the AOI than a single-tile read. Skips itself when
  MPC is unreachable.

Mocked tests for the tile-coverage defect were tried and **discarded**: the defect
is about real source-tile boundaries and no mock can reproduce it.

### Performance notes

Concurrency is memory-bound, not CPU-bound — requests spend most of their time
waiting on the network. At 100 km² a request costs ~480 MB, so an 1,800 MB
container sustains ~3 concurrent analyses regardless of vCPU count. This should be
re-measured now that NDVI is ~50 MB; see *Next*.

---

## 1.2.0 and earlier — bounded synchronous analysis

- Synchronous `POST /generate-context` over DEM (NASADEM), land cover (ESA
  WorldCover), and a bounded Sentinel-2 NDVI composite from Planetary Computer.
- AOI admission policy: union FeatureCollections rather than taking the first
  feature, `is_valid`/`is_empty` checks, 500 KB payload cap, 10,000-vertex cap,
  geodesic bounding-box cap, concurrency semaphore, and explicit `skipped` NDVI
  with no MODIS fallback.
- Docker Compose on a single droplet behind Nginx, with ports bound to loopback.

---

## Evaluated and rejected

Kept here so the reasoning is not repeated.

| Option | Verdict | Reason |
| --- | --- | --- |
| `spyndex` (Awesome Spectral Indices) | not now | v0.12.0, actively maintained, MIT, but self-classifies as pre-alpha and pre-1.0. The one index in use is 3 lines and correct. Trigger: needing EVI + NBR + NDVI together. |
| Zarr / Kerchunk for the series store | no | Portfolio outputs are a few thousand scalar numbers — a table, not a cube. Revisit only if publishing per-pixel raster cubes. |
| `lazycogs` | no, structurally | See below. |
| `deck.gl-raster`, `@developmentseed/deck.gl-geotiff` | later | Real and client-side COG rendering with no server. A genuine differentiator for this audience, but it competes with backend work that is worth more. |
| Earthmover Arraylake Community tier | adopt, $0 | Permanent free tier: 10 GB, one connected bucket, 50 compute credits/month, unlimited free Marketplace datasets. The free **ERA5 Icechunk cube** retires the blocking rainfall-data question at no cost. |
| Admin-gated portfolio | no | That is the incumbent's exact failure mode, and the most-asked question in the conservancy webinar was "how do I add my polygon" — asked five times by four people, answered zero times. |
| Charging individual users | no | That audience is the acquisition channel. The buyers are NRT, DE Africa, county governments, and EIA consultancies. |

### lazycogs spike — findings

Tested against live MPC. `rustac` works well: it harvested a 20-item Sentinel-2
catalogue to GeoParquet in 1.5 s. The blocker is the read path, and it is
structural rather than a missing line of code:

- MPC requires a per-request SAS token, and `obstore.HTTPStore` double-encodes
  the query string (`?` → `%3F`, `%` → `%25`), so the token arrives mangled and
  Azure returns `409 PublicAccessNotPermitted`.
- `lazycogs.open` accepts a single `store`, but MPC's Sentinel-2 scene set spans
  multiple storage accounts (`sentinel2l2a01`, `sentinel2l2a02`, …) for adjacent
  MGRS tiles.
- The installed version is 0.5.0, which predates the `path_from_href` and
  custom-store API documented for 0.7.0.

Its `rustac` component is the part worth keeping for a future precompute worker.

---

## Known limitations

- **Residual thin cloud depresses NDVI.** Because scene classes 4/5 are retained,
  NDVI over a persistently cloudy area is biased low. `valid_pixel_fraction`
  describes AOI coverage, not cloud contamination, so it does not warn about this.
  Treat a figure below ~0.2 as uncertain rather than as bare ground. A
  `scl_bright_cloud_fraction` field would be the proper fix.
- **The area cap applies to the bounding box**, penalising irregular polygons by
  up to ~2× their true area.
- **No request-size guard in the application.** The 500 KB check happens after
  FastAPI has parsed the body. Mitigated by `client_max_body_size 512k` in
  Nginx and the loopback-only port bind, but the app itself has no cap — and
  512 KiB in Nginx admits bodies the app then rejects at 488 KiB.
- **No rain, historical, or country-scale indicator.** The three questions most
  asked in the conservancy webinar remain unanswered.

---

## Next

Ordered by value per unit of effort.

1. **Sentinel ladder: Landsat 30 m and MODIS 250 m.** Landsat is the only source
   that reaches before 2015 for the vegetation index, matching what ERA5 now does
   for rainfall. Two traps: the band is `nir08`, not `nir`, and PC's Landsat C2 L2
   surface reflectance carries `scale=2.75e-05, offset=-0.2` — the offset does
   **not** cancel in NDVI, so a naive port yields plausible wrong numbers. MODIS
   `modis-13Q1-061` ships NDVI as a finished 16-day product, a stronger
   provenance claim than anything computed.
2. **Usage instrumentation.** The schema is already written in
   `NEXT_FEATURE.md`; ship it with the current app. Aggregate outcomes, duration,
   selected datasets, coarse AOI-size band, no raw geometry, no full IP. Nothing
   else can be prioritised without it.
3. **Self-service polygon submission.** Submit any polygon, queue it for
   precompute, return a permanent shareable page. The rainfall miss state is
   already the hook: it names the cache key and explains why. This is the
   loudest unanswered question in the webinar — "how do we share our polygons",
   asked five times by four people — and `NEXT_FEATURE.md` as written answers it
   with the wrong answer, an administrator-gated portfolio.
4. **Alerting.** A dashboard is visited once; a subscription is visited monthly.
   Diff the precomputed series on refresh and notify. No new infrastructure, and
   it cannot be retrofitted cheaply.

Only with evidence from 2–4: a durable queue and a monitoring view. The store
should be Postgres or DuckDB — a table, not a cube.
