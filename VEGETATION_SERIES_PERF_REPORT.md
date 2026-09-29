# Why the MODIS vegetation series is slow, and how to make it as fast as rainfall

**Scope:** research only. No product code was changed. All measurements below come from
throwaway scripts run outside the repository (`/tmp/kilo/vegbench/`) against the live
public Planetary Computer endpoints.

**Test host note (important for reading every number below):** measurements were taken
from this workstation, whose round-trip time to the Planetary Computer's MODIS blob
storage (`modiseuwest.blob.core.windows.net`, Azure **West Europe**) is
**610–833 ms, median 671 ms**. That is very high — see §1.3 — and it is the single
largest multiplier in every number in this report. Your deployed host appears to have a
far lower RTT (your own figures — 0.2 s STAC root, 0.1 s signing — imply ~100–200 ms).
**Scale all read costs by (your RTT / 671 ms)** before acting on them.

Versions measured against: rasterio 1.5.0, GDAL 3.12.1, pystac-client 0.9.0, Python 3.12.2.

---

## 1. Why it is slow, concretely

### 1.1 The cost model

A single monthly COG window read is not one network operation. Tracing GDAL's HTTP
traffic (`CPL_CURL_VERBOSE=YES`) for one read of a 67×98 px window on a MOD13Q1 NDVI
COG produces this exact sequence:

```
> HEAD /modis-061-cogs/.../MOD13Q1.A2024177.h21v08.061..._250m_16_days_NDVI.tif   <- file size probe
> GET  /modis-061-cogs/.../MOD13Q1.A2024177.h21v08.061..._250m_16_days_NDVI.tif   <- header / IFD
> GET  /modis-061-cogs/MOD13Q1/21/08/2024177                                       <- DIRECTORY LISTING -> HTTP 409
> GET  /modis-061-cogs/.../MOD13Q1.A2024177.h21v08.061..._250m_16_days_NDVI.tif   <- tile data
> GET  /modis-061-cogs/.../MOD13Q1.A2024177.h21v08.061..._250m_16_days_NDVI.tif   <- tile data
```

Measured totals: **1 HEAD + 4 GET, 3 of them byte-ranged, 0.61 MB transferred, 1 HTTP 4xx.**

These five requests are **serially dependent** — you cannot know where the IFD is until
you have the header, and you cannot know which tiles to fetch until you have the IFD.
So the floor for one month is `5 × RTT` of pure latency, and:

```
cost_per_series  ≈  N_months  ×  n_round_trips  ×  RTT  /  effective_parallelism
```

**The bytes are irrelevant.** 0.61 MB for a whole month. A 320-month series is ~195 MB —
trivial. The 195th month costs exactly as much as the 1st.

This is the same structural fact the Pangeo community states: *"the minimal unit for GDAL
reads is a 'tile' or a block… Even if you want the value of just 1 pixel you need at least
two HTTP GET requests"* — https://discourse.pangeo.io/t/optimizing-read-from-cog/3291

### 1.2 Your window is small, and that is the problem

For Naibunga Upper (the bbox I took from `.rainfall-cache/`), the reprojected read window
on the MODIS sinusoidal grid is **67 × 98 px = 6,566 pixels**, which fits inside a single
512×512 COG block. So you are already reading the minimum possible. There is no
"read less data" lever left at full resolution. The latency is fixed overhead, not payload.

### 1.3 The multiplier: you are far from Azure West Europe

| Operation | Time |
|---|---|
| Raw HTTP HEAD to `modiseuwest.blob.core.windows.net` (min/median/max of 8) | **610 / 671 / 833 ms** |
| Planetary Computer STAC root | 0.2 s (yours) |

The Planetary Computer maintainer's own guidance is unambiguous: *"Making requests from the
Azure West Europe region will also result in the lowest latencies"* and *"access data from
within Azure West Europe… to get the best results"*
— https://github.com/microsoft/PlanetaryComputer/discussions/246

The MODIS collection lives in `modiseuwest.blob.core.windows.net`; the account name is
literally `...euwest...`. If your server is not in Azure West Europe, **you are paying a
cross-region penalty on every one of the 5 round trips per month, and you are also in the
most heavily rate-limited tier** (see §1.7). This is the biggest single lever in the whole
system and it is a deployment decision, not a code change.

### 1.4 Threads help far less than they should — and it is GDAL, not the network

Same endpoint, same 8 requests, two libraries:

| Concurrency | GDAL/rasterio (`/vsicurl`) | Plain `requests` HEAD | Plain `requests` GET 16 KB |
|---|---|---|---|
| 1 worker  | 1.59 s/item | 0.73 s/req | 0.66 s/req |
| 4 workers | **0.98 s/item** (1.62× speedup) | 2.51 s total (2.3×) | 4.30 s total (1.2×) |
| 8 workers | **0.77 s/item** (2.06× speedup) | 0.69 s total (**8.5×**) | 0.91 s total (**5.8×**) |

**Plain HTTP concurrency scales near-linearly to 8×. GDAL at 4 workers scales 1.6×, and at
8 workers it is *slower per item* than at 4.** Per-read wall latency inflates from 1.59 s
to ~3.9 s under 4× concurrency. That is contention inside GDAL, not a network limit and
not a Python problem (proved below).

Corroborating evidence that this is a known GDAL characteristic:

- GDAL's own threading page: *"Unless otherwise stated, no GDAL public C functions and C++
  methods should be assumed to be thread-safe… Block cache related structures for a given
  GDALDataset are not thread-safe"*, and *"performance issues may arise when writing several
  datasets from several threads, due to lock contention in the global structures of the
  block cache mechanism."* — https://gdal.org/en/stable/user/multithreading.html
- Community report specifically that concurrent `rasterio.open()` calls do not proceed
  concurrently: https://gis.stackexchange.com/questions/482692/opening-rasterio-datasets-in-threads
- GDAL 3.10 added RFC 101 read-only thread-safe datasets
  (`GDAL_OF_THREAD_SAFE`) precisely for this: https://gdal.org/en/stable/development/rfc/rfc101_raster_dataset_threadsafety.html

**Honest caveat:** I can demonstrate that GDAL serialises, but I could not isolate the
exact mutex responsible (block cache vs. the `/vsicurl` LRU vs. curl handle pool). The
*fixes* below are measured, so the diagnosis does not depend on naming the lock.

### 1.5 What is NOT the cause (each tested and ruled out)

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Range-request size / payload | **No** | 0.61 MB per month read. Cost is flat vs. bytes. |
| Deadlock | **No** | Every read completes; no hangs observed. |
| GIL (Python-level) | **No** | `ProcessPoolExecutor` at 4 workers = 0.67 s/item, statistically identical to `ThreadPoolExecutor` at 0.67 s/item. If the GIL were the constraint, processes would win. |
| Full-resolution vs. overview selection | **Not applicable** | You read full-resolution and a 67×98 window fits in one 512 block; there is no larger window to downsample. |
| Switching to a coarser product (MOD13A1 500 m) | **No** | Carefully re-measured in fresh processes with alternating order: MOD13Q1 = 1.21/1.52/1.26/1.75 s/item; MOD13A1 = 1.09/1.07 s/item. **~1.2×, not 50×.** (My first measurement showed 0.012 s for MOD13A1 — that was a warm-cache artifact of reading both in one process. Discarded.) |
| Raising `CPL_VSIL_CURL_CHUNK_SIZE` | **Actively harmful** | With `CPL_VSIL_CURL_CHUNK_SIZE=4194304` + `GDAL_HTTP_MULTIRANGE=YES`: 1.08 → **4.10 s/item**. Do not do this. |
| More workers | **Harmful past 4** | 4w = 0.63–0.72, 8w = 0.72–0.90, 12w = 0.90, 16w = 1.15 s/item. Your choice of 4 is already right. |
| Window area (for small polygons) | **Largely flat, with a caveat** | 67×98 px = 1.08 s/mo; 847×862 px = 3.40 s/mo. 12.8× the pixels → 3.1× the time. So the code's comment *"cost is request latency, not pixels"* is **half right**: cost tracks **the number of 512×512 COG blocks touched**, not pixels. The 847×862 window crosses ~4 blocks; the small one crosses 1. |

### 1.6 The estimate constant is wrong for a structural reason, not a calibration reason

`plan()` estimates `months` from `start` (default 2010-01-01) → **201 months** as of
2026-09. But `compute_monthly_series()` ignores `start` for the read loop and calls:

```python
months = _month_range(CLIMATOLOGY_START, end)   # CLIMATOLOGY_START = "1991-01-01"
```

→ **429 months iterated**, of which **315 have imagery** (MOD13Q1 begins 2000-02; I
confirmed 315 distinct months from the live catalogue). The baseline climatology is
deliberately computed in the same pass, which is a sound design choice — but it means the
worker does **~1.6× the work the estimate predicts, every time**. `SECONDS_PER_MONTH =
0.68` is being applied to the wrong denominator.

That accounts for part of your discrepancy. The measured per-month read cost from this
host (0.71 s at 4 workers, small window) happens to be near 0.68, but the *realised*
per-month cost in production is whatever your host's RTT makes it, and the series length
is fixed at ~315 months regardless of what the user asked for.

### 1.7 A second, independent problem: the SAS signing rate limit

`_read_month` calls `planetary_computer.sign(href)` **inside the thread pool, once per
month** — ~315 signing requests per series, from 4 threads.

The signing API is rate limited, and the failure is explicit and hostile:

```
HTTP 429 {"statusCode": 429, "message": "Rate limit is exceeded. Try again in 35 seconds."}
```

I hit this repeatedly while researching (twice), with a ~35–46 s backoff demanded. The
limiting tier depends on **whether you supply a subscription key and whether you are in
West Europe**:

> *"Rate limiting and token expiry are dependent on two aspects of each request: Whether
> or not the request is originating from within the same data center as the Planetary
> Computer service (West Europe); Whether or not a valid API subscription key has been
> supplied."*
> — https://planetarycomputer.microsoft.com/docs/concepts/sas/#rate-limits-and-access-restrictions

And the maintainer names the fix directly:

> *"You can also request tokens for a **storage container** and not just a single file.
> This means that you can reuse a single token (up to its expiry time) for many file
> requests and can cut out a significant number of requests to that API."*
> — https://github.com/microsoft/PlanetaryComputer/discussions/246

**Verified:** one collection-scoped signing call
(`GET /api/sas/v1/sign?href=<any file>&collection=modis-13Q1-061`) returns a SAS that I
successfully used to `HEAD` *and* `rasterio.open()`+read a **completely different month's
file**. So 315 signing calls can become **1**.

*Caveat, stated honestly:* the documented `GET /api/sas/v1/token/{collection_id}` endpoint
also returns HTTP 200 for `modis-13Q1-061`, but the token it returned was rejected with
**HTTP 403 on every one of 7 attempts over 84 seconds**. I could not get that endpoint
working anonymously from outside West Europe. The `/sign?...&collection=` form did work.

### 1.8 Two things I initially got wrong, corrected

For rigour, since both were plausible and both were wrong:

1. **I thought MPC's STAC paging was returning wrong-planet tiles.** Following the `next`
   link with a `GET` and only the URL query string drops the POST body (which carries
   `bbox`), and indeed you get one-item-per-tile from h00v10 to h35v10. But `pystac-client`
   POSTs the body correctly: replicating `_items_by_month` exactly returned **1,138 items,
   1 distinct tile (h21v08), 315 months, zero wrong tiles, in 11.4 s**. There is no paging
   bug. If you ever hand-roll the paging, you must POST the body.
2. **I thought coarser MODIS products were dramatically faster.** They are not (§1.5).

### 1.9 Bottom line on the cause

> The vegetation series is slow because it performs **~315 sequential, multi-round-trip
> remote reads** from a host that is far from Azure West Europe, each costing 5 dependent
> HTTP round trips, with only ~1.6× effective parallelism available from GDAL.
> It is **not** slow because of pixel volume, window size, GIL, overview selection, or
> a deadlock. And it reads **315 months even when the user asked for 196**.

Measured wall-clock for the current on-demand path from this host, Naibunga Upper:
**~11 s STAC search + ~315 × 0.71 s / month ≈ 3.7 minutes.**

---

## 2. Ranked comparison of candidate approaches

Per-month figures are measured from this host (671 ms RTT). "196 mo" is your stated
span; "315 mo" is what the code actually reads today. Multiply by `your_RTT / 0.671` for
your deployment.

| # | Approach | Resolution | Cadence | Access | Measured s/month (1→4 workers) | 196 mo | 315 mo | Verdict |
|---|---|---|---|---|---|---|---|---|
| 1 | **MOD13Q1 via MPC, precomputed once for the portfolio** (era5 pattern) | 231.66 m actual | 16-day → monthly | STAC + COG range | 3.40 → 2.22 (whole-portfolio window) | ~7–15 min **once** | ~10–16 min **once** | **Recommended.** Turns every subsequent request into a cache hit. |
| 2 | MOD13Q1 via MPC, per-request, as today | 231.66 m | 16-day | STAC + COG range | 1.21 → 0.71 | **2.3 min** | **3.7 min** | Works, but never "instant". |
| 2b | …same, after §4 fixes (1 signing call, `DISABLE_READDIR`, `thread_safe`, 4 workers, correct month count) | 231.66 m | 16-day | STAC + COG range | ~1.08 → **0.63** | **2.1 min** | **3.3 min** | ~12% better. Necessary, not sufficient. |
| 3 | MOD13A1 500 m, per-request | 463.31 m | 16-day | STAC + COG range | 1.07 → — | 2.4 min | 3.8 min | **No faster.** Halves resolution for ~1.2× latency. Not worth it. |
| 4 | MOD15A2H LAI/FPAR 500 m | 463.31 m | 8-day | STAC + COG range | ~1.1 (same class) | ~2.4 min | ~3.8 min | Different variable; a good *complement*, not a substitute. |
| 5 | HLS (`hls2-l30` / `hls2-s30`) on MPC | 30 m | 2–3 days | STAC + COG range | **not measured** — expect ≥1 COG tile per scene + scene-selection cost, and ~10× more items than a 16-day composite | much worse | much worse | **Reject for a monthly series.** |
| 6 | `sentinel-2-l2a` | 10 m | 5 days + clouds | STAC + COG range | not measured | much worse | much worse | **Reject.** Needs cloud masking decisions the product explicitly avoids. |
| 7 | MPC Data API `GET /api/data/v1/item/statistics?…&bbox=` | — | per item | 1 HTTP request | **0.94–1.30 s** | 3.4–4.2 min | 5.3–6.8 min | **Reject — and see the warning.** |
| 8 | MPC TiTiler XYZ tiles via the `tilejson` asset | selectable | per item | 1 request/tile | 0.56–0.88 s **per tile** | worse | worse | Reject: a polygon needs many tiles. |
| 9 | Bulk download from LP DAAC / AWS Open Data, then read locally | 231.66 m | 16-day | S3 | ~1.1 GB/yr/tile for NDVI-only COGs | 15.3 GB for the full record | 15.3 GB | **Reject — see below.** |
| 10 | "Download a small global VI dataset wholesale" | — | — | — | — | — | — | **Does not exist.** There is no small global NDVI raster. |

#### Notes and corrections to the specific products you asked about

- **MOD13A2/MYD13A2 (1 km): not on Planetary Computer.** I enumerated all 138 MPC
  collections; the only MODIS vegetation-index collections are `modis-13Q1-061` (250 m) and
  `modis-13A1-061` (500 m). There is no `modis-13A2-061`. MPC's MODIS group page states
  *"Each product type is combined into a single collection"*, and `modis-13Q1-061`
  contains **both Terra (`MOD13Q1`) and Aqua (`MYD13Q1`)** items — I confirmed both prefixes
  in the catalogue results. (https://planetarycomputer.microsoft.com/dataset/group/modis)
- **MOD13Q3/MYD13Q3: not on Planetary Computer either**, and it is not smaller in a way that
  helps — it still has 250 m NDVI/EVI bands. Even if it were, §1.5 shows bytes are not the
  cost.
- **MOD16A2 / "modis-16QPI-061" / "modis-16A-061": neither exists on MPC.** The full MODIS
  list I retrieved contains `modis-16A3GF-061` (Net ET, **yearly** gap-filled) and no
  16-day ET collection. Yearly cadence makes it useless for a monthly series.
- **Warning on row 7.** `GET /api/data/v1/item/statistics` *does* work (HTTP 200) and
  returns mean/min/max/std/median over a bbox in one request, which looks like a perfect
  replacement. **I do not believe its `bbox` is honoured correctly for this collection.**
  For the Naibunga bbox it reported `count: 1,046,388` valid pixels. The correct count is
  6,566 — a factor of 159×. It appears to be computing a whole-tile (or worse) statistic.
  Do not use it for polygon means without validating the count against a locally-computed
  value. I did not determine the cause.
- **Row 9 — the bulk-download answer to your question 3.** The AWS Open Data registry entry
  for MOD13Q1 points at `arn:aws:s3:::lp-prod-protected/MOD13Q1.061` in `us-west-2`, and
  classifies it **"Controlled Access"** requiring Earthdata Cloud S3 credentials
  (https://registry.opendata.aws/nasa-mod13q1/). That violates your "keyless-or-free"
  constraint. You would still have to authenticate.
  Sizes I measured directly by `HEAD` on the MPC mirror of the same granule:
  - `MOD13Q1..._250m_16_days_NDVI.tif` (COG): **47.9 MB**
  - `MOD13Q1..._250m_16_days_EVI.tif` (COG): 47.6 MB
  - `MOD13Q1` HDF-EOS (the LP DAAC native format): **232.1 MB**

  At 23 composites/year: **1.10 GB/yr for NDVI COGs** (2.20 GB for Terra+Aqua), or
  **5.3 GB/yr for the full HDF**. For the whole 2000-02 → 2026-09 record on one tile that
  is **~15.3 GB of COGs**. Downloading that is *slower* than reading 315 windows, and you
  only need the pixels that fall inside the 21 polygons. **Bulk download loses.**

---

## 3. Recommendation (a): the precompute-portfolio path — this is the real fix

**The ERA5 pattern transfers, and it is dramatically cheaper than 21 separate series
because all 21 conservancies fall inside a single MODIS tile.**

I verified this: the 21 precomputed areas span 36.857–38.647 °E, 0.263–2.059 °N. A STAC
search over that combined bbox returns items from **exactly one MODIS 250 m tile, `h21v08`**
(bbox 29.8879, −0.0034, 40.626, 10.003). Every conservancy is in it.

So the precompute is **not 21 × 315 reads**. It is **315 reads, once** — one window per
month covering the whole portfolio (847 × 862 px), from which all 21 polygon means are
extracted in memory.

**Measured cost of the entire 21-area portfolio build** (12-month sample, extrapolated):

| Concurrency | s/month | × 315 months |
|---|---|---|
| 1 worker | 3.40 | **~15.5 min** |
| 4 workers | 2.22 | **~9.5 min** |
| 8 workers | 1.82 | ~7.6 min (but less stable; see §1.4) |

Plus ~11 s for the STAC search and one signing call. **One-off cost, from this
poorly-placed host, is under 16 minutes.** From Azure West Europe (RTT ~20 ms rather than
671 ms) the same build should be **well under a minute** — the maintainer's own guidance is
that West Europe gets the lowest latency *and* the best rate-limit tier.

After that, every request is a JSON cache hit served by the existing area-matching service,
**exactly like rainfall**. That is the only path that makes vegetation as fast as rainfall.

Storage: 21 areas × 315 months × ~6 numbers ≈ trivial; the ERA5 cache format already in
`.rainfall-cache/` can be reused verbatim.

**Caveat:** precompute only serves the 21 precomputed areas. Users drawing an arbitrary
polygon elsewhere still need path (b) below. That is fine — it is the same product
tension ERA5 already has, and the CHANGELOG shows the "refuse a live read, offer the
offline route" behaviour is already built.

---

## 4. Recommendation (b): on-demand optimisation

None of these make it instant. They take it from ~3.7 min to ~3.3 min here and will take
it from ~1 min to ~50 s on your host. **They are worth doing because they are cheap and
they also speed up the precompute build**, but they are not a substitute for §3.

Ranked by measured benefit:

1. **Sign once, not 315 times.** Use one collection-scoped SAS call
   (`/api/sas/v1/sign?href=<any asset>&collection=modis-13Q1-061` — verified to work across
   files) and reuse the query string for every month. Removes 314 signing requests, removes
   the 429 failure mode entirely, and removes a per-thread cache race. *Highest value per
   line of change.*
2. **Set `GDAL_DISABLE_READDIR_ON_OPEN=TRUE`.** Removes the directory-listing `GET` that
   returns HTTP 409 on every single open. Measured: 1.59 → 1.08 s/month sequential,
   0.98 → 0.71 s/month at 4 workers (**−28%**). This is one environment variable.
3. **`rasterio.open(..., thread_safe=True)`.** RFC 101, needs GDAL ≥ 3.10 — you have 3.12.1.
   Measured at 4 workers: 0.718 → 0.628 s/month (**−13%**). Small but real and free.
4. **Keep 4 workers.** Confirmed optimal; 8/12/16 are all worse.
5. **Fix the month count.** The estimate should be computed from the months actually read
   (~315), not from `start` (~201). Either report 315 months of work honestly, or split the
   baseline into its own cached artefact so the user's 196-month request only reads 196.
   This is a **1.6× work reduction on the baseline**, and it is the only lever of that size.
6. **Re-measure `SECONDS_PER_MONTH` from your production host** and state it as a function
   of RTT, e.g. "≈ 5 × RTT ÷ 2 effective parallelism", so the estimate survives a host move.

Things I considered and recommend **against**:

- **Process pool instead of thread pool** — measured identical (0.67 vs 0.67 s/month). Not
  worth the complexity, and rasterio warns about forking after GDAL drivers are registered:
  *"Deadlocks are easy to produce if we fork after GDAL drivers have been registered"*
  (https://rasterio.readthedocs.io/en/latest/topics/concurrency.html).
- **Raising `CPL_VSIL_CURL_CHUNK_SIZE` / `GDAL_HTTP_MULTIRANGE`** — measured 3.8× *worse*.
- **Reading at reduced resolution / using overviews** — the COG has overviews `[2,4,8,16]`
  and `_window_for_bbox` already produces whole-pixel windows, but at 67×98 px you are
  already inside a single 512 block, so there is nothing to gain. This lever only becomes
  relevant if you read the whole-portfolio window, and even then only for large polygons.
- **Caching by tile rather than polygon** — irrelevant for MODIS. Every month is a
  *different file* (one per 16-day composite per tile), so there is no tile reuse to exploit
  within a series. (It *would* matter for Sentinel-2/HLS, where many dates share a tile.)
- **`obstore`** (Development Seed's Rust-backed S3/GCS/Azure client,
  https://developmentseed.org/obstore/latest/api/auth/planetary-computer/) is the most
  promising unexplored option — it advertises "the simplest, highest-throughput Python
  interface to S3, GCS & Azure Storage" and has first-class MPC auth. I did **not** benchmark
  it; treat as a promising unknown, not a recommendation.

---

## 5. Existing open-source implementations to compare against

| Implementation | What it documents | Relevance |
|---|---|---|
| **rasterio official concurrency example** | Local-file COG benchmark: 4.277 s at `-j 1` → 1.251 s at `-j 4` ("over 3x speed up"). https://rasterio.readthedocs.io/en/latest/topics/concurrency.html | **This is the comparison that matters.** rasterio itself demonstrates ~3.4× on 4 workers — but on a *local* file, where there are no round trips. Against `/vsicurl` you get 1.6×. The gap *is* your problem. |
| **Pangeo "Optimizing read from COG"** | "the minimal unit for GDAL reads is a tile or a block… Even for just one pixel you need at least two HTTP GET requests." https://discourse.pangeo.io/t/optimizing-read-from-cog/3291 | Independent confirmation of the cost model. |
| **Terrafloww / Rasteret** | Same 3-request pattern described from the other side ("1. Initial GET request to read the file header; 2. Additional requests if needed; 3. Final requests to read the actual data tiles"), and a fix: pre-bake COG tile-offset metadata into STAC GeoParquet so you compute byte ranges yourself and issue **1 request per required tile** with no GDAL in the loop. https://blog.terrafloww.com/efficient-cloud-native-raster-data-access-an-alternative-to-rasterio-gdal/ | The most directly comparable work. Their benchmark chart is an image and their stated scenario is 20 Sentinel-2 scenes / 1 year / 1 polygon on 2 vCPU — **so their absolute numbers are not comparable to yours and I could not extract them.** Their structural claim (eliminate the header round-trips) is the one worth stealing if §4 isn't enough. |
| **GeoServer ImageMosaic over MODIS VI COGs** | Proves the multi-temporal-COG-tile pattern is a solved, production-grade shape — but via a server that caches tiles. https://github.com/geoserver/geoserver/blob/master/doc/en/user/community/cog/mosaic.md | Validates the architecture, not the latency. |

I did **not** find a public service that publishes a per-request latency figure for
"monthly NDVI time series for an arbitrary polygon" that you could diff against. That is
a genuine gap in the evidence, not an oversight.

---

## 6. What I could not verify / where I am uncertain

1. **The absolute per-month numbers are host-specific.** They scale linearly with RTT. Your
   deployed numbers will differ. The *structure* (5 dependent round trips, ~1.6× GDAL
   parallelism at 4 workers) is host-independent.
2. **I could not name the exact GDAL mutex** causing the 1.6× ceiling. I proved it is GDAL
   (plain HTTP gets 8.5×; processes get 0× benefit) but not which lock.
3. **MPC's `/api/sas/v1/token/{collection_id}` returned a 403-unusable token** for
   `modis-13Q1-061` across 7 attempts / 84 s. The `/sign?...&collection=` form worked. I
   do not know why they differ, and I did not test whether an API key or a West Europe
   source IP fixes the former.
4. **I did not measure HLS or Sentinel-2 read costs.** My rejection of them is based on
   item counts and block geometry, not measurement. The reasoning is solid but it is
   inference, not evidence.
5. **I did not benchmark `obstore`.**
6. **`/item/statistics` bbox handling** — I showed the count is 159× too large; I did not
   determine whether the parameter is in the wrong CRS, silently ignored, or something else.
7. **MOD13A1's ~1.2× advantage** is within the run-to-run noise I saw (1.07–1.75 s for the
   same 13Q1 read). It is probably closer to zero than 1.2×. Treat it as "no meaningful
   difference."
8. I did not verify MOD13Q1's official start date against an LP DAAC product page; I
   inferred 2000-02 from the live catalogue's earliest item. The AWS registry page does say
   *"Update Frequency: From 2000-02-18 to Ongoing"*, which corroborates it.

---

## 7. Sources

**Planetary Computer (authoritative, primary)**
- MODIS Version 6.1 group: https://planetarycomputer.microsoft.com/dataset/group/modis
- MOD13Q1 dataset page: https://planetarycomputer.microsoft.com/dataset/modis-13Q1-061
- MOD13A1 dataset page: https://planetarycomputer.microsoft.com/dataset/modis-13A1-061
- SAS tokens, rate limits, and access restrictions: https://planetarycomputer.microsoft.com/docs/concepts/sas/
- **Maintainer answer on STAC/search/signing/data rate limits and West Europe latency:** https://github.com/microsoft/PlanetaryComputer/discussions/246
- Data API OpenAPI spec (source of the `/item/statistics`, `/item/tiles/…` endpoint list): https://planetarycomputer.microsoft.com/api/data/v1/openapi.json
- MODIS 6.1 product description: https://www.earthdata.nasa.gov/data/catalog/lpcloud-mod13q1-061
- MOD13A2 1 km (confirms the product exists at NASA but not on MPC): https://www.earthdata.nasa.gov/data/catalog/lpcloud-mod13a2-061
- AWS Open Data MOD13Q1 (bulk bucket is **controlled access**): https://registry.opendata.aws/nasa-mod13q1/

**GDAL / rasterio**
- GDAL multi-threading (thread-safety, block-cache lock contention, fork guidance): https://gdal.org/en/stable/user/multithreading.html
- RFC 101: raster dataset read-only thread-safety (`GDAL_OF_THREAD_SAFE`, GDAL ≥ 3.10): https://gdal.org/en/stable/development/rfc/rfc101_raster_dataset_threadsafety.html
- rasterio concurrent processing (GIL release, `thread_safe=True`, fork/deadlock warning): https://rasterio.readthedocs.io/en/latest/topics/concurrency.html
- rasterio windowed read / block-granularity semantics: https://rasterio.readthedocs.io/en/latest/topics/windowed-rw.html
- Community report — concurrent `rasterio.open()` does not proceed concurrently: https://gis.stackexchange.com/questions/482692/opening-rasterio-datasets-in-threads

**Comparators**
- Pangeo — "Optimizing read from COG" (minimum 2 HTTP GETs per COG access): https://discourse.pangeo.io/t/optimizing-read-from-cog/3291
- Terrafloww — bypassing GDAL's header round-trips via pre-computed COG byte ranges in STAC GeoParquet: https://blog.terrafloww.com/efficient-cloud-native-raster-data-access-an-alternative-to-rasterio-gdal/
- GeoServer ImageMosaic over MODIS VI COGs: https://github.com/geoserver/geoserver/blob/master/doc/en/user/community/cog/mosaic.md
- obstore Planetary Computer auth (promising, unbenchmarked): https://developmentseed.org/obstore/latest/api/auth/planetary-computer/
