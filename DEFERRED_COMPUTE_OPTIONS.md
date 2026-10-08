# Deferred: query/compute engines, and a spatial layer over Zarr

**Status:** held deliberately. Not scheduled, not blocked, not recommended-against.
Reconsider when the trigger named under each entry is met.

Written after the CHIRPS data-path review, because both options were proposed as
answers to the same question ("how do we stop issuing hundreds of HTTP requests
per job?") and neither was adopted. The reasoning for *not* adopting them now is
recorded here so the next person does not have to re-derive it, and so the
revisit starts from evidence rather than from the pitch.

The CHIRPS work itself did **not** stop and did **not** depend on either of these:
the Zarr store in DigitalOcean Spaces was built directly, on Icechunk, using the
same reader path ERA5 already uses. See `ROADMAP.md`.

---

## 1. Zax / Zax-SQL (Earthmover)

**What it is.** A managed Arraylake compute service exposing Icechunk repos as
SQL tables over the Postgres wire protocol and Arrow Flight SQL, planned by
Apache DataFusion. Rust core. Announced 2026-09-15.

Sources:
- https://docs.earthmover.io/compute/sql
- https://www.earthmover.io/blog/compute-roadmap

**Why it was not adopted now.** Three blockers, all verified against the docs
rather than inferred:

1. **Spatial operators are still listed as in-progress.** The roadmap post puts
   "spatial extensions: PostGIS geometry types and operators" under *in
   progress*. Area-weighted zonal statistics over a polygon is this project's
   core operation, and there is no documented operator for it.
2. **The documented query shape is point-at-coordinate**, e.g.
   `WHERE latitude = 40.5 AND longitude = 290`. `rainfall.py` rejects exactly
   this as a product decision — see the `_chirps_block` docstring, "a point
   sample of a 5 km grid would be a different claim from the ERA5 cell it is
   being compared with."
3. **`AVG()` would silently break the coverage guard.** `MIN_MONTH_COVERAGE`
   (0.8) exists precisely because averaging the surviving hours of a
   mostly-missing month yields a confident, citable, wrong total. SQL `AVG()`
   discards nulls with no denominator, so every aggregate would need a
   hand-written `COUNT()` alongside it, forever.

Secondary costs: it is **managed only** (no self-host path), it draws credits
**for as long as the service is up** rather than per query, and it runs on a
single node. Data must be reachable from Arraylake, which our self-hosted
`earthmover-icechunk-era5` read is not without brokering.

**What is genuinely good about it, and worth remembering.** Version pinning:
`AT(SNAPSHOT => 'K0RSJ5F1XCRJP8CVXY0G')` addresses a committed snapshot
permanently, and the docs are explicit that "nothing here is relative — there is
no 'yesterday'" — a query carrying a ref is a stable citation. That is our
"a cached value must never outlive the code that produced it" expressed as a
query feature. Icechunk's snapshot model is the part worth keeping; Zax is one
interface onto it.

**Revisit trigger.** Any one of: (a) PostGIS spatial operators ship; (b) a
coverage-aware aggregate is available; (c) Icechunk gains regridding/rolling
window operations, which our anomaly-vs-climatology computation needs.

**Free evaluation, whenever wanted.** ERA5 is in the Earthmover Marketplace and
the free Community Tier is enough to query it. Our ERA5 store is *already*
Icechunk, so nothing needs to move to find out whether any of this is useful.
That is a day of work, not a migration.

**Footgun if anyone tries it.** `ATTACH`ing the DuckDB catalog and filtering
client-side reads the **whole variable** — projections push down, predicates do
not. Filters must live inside `READ_ADBC`.

---

## 2. PostGIS FDW over Zarr (experimental, pangeo)

**What it is.** A foreign data wrapper that lets PostgreSQL/PostGIS query remote
Zarr v2/v3 stores in place, with no ETL. Supports aggregate pushdown
(`count`, `sum`, `avg`, `min`, `max`), PostGIS point sampling and zonal
statistics, coordinate/chunk pruning, and `EXPLAIN ANALYZE` I/O metrics.

Source: https://discourse.pangeo.io/t/experiment-querying-large-zarr-datasets-directly-from-postgresql-postgis-without-etl/5809

**Why it was not adopted now.** It is experimental and **the upstream PR is not
merged**. Its own stated limitations matter to us: spatial operations target
rectilinear rank-2 grids; **polygon statistics use cell-center inclusion rather
than fractional cell coverage**; coordinate CF packing is not decoded; and the
time model supports one temporal dimension with a limited calendar. The
cell-center-vs-fractional-coverage gap is a direct conflict with our
area-weighted semantics.

**Why keep an eye on it.** It is doing the thing Zax is not: real spatial
aggregation over a Zarr cube, joined against polygons held in PostGIS. Its
author names our exact use case — "EO statistics over farm polygons" — and
"rainfall over watersheds."

**Revisit trigger.** The PR merges, *and* fractional/fraction-weighted polygon
coverage lands. Until both, its numbers would not mean what ours mean.

---

## 3. Adjacent options screened, not adopted

Recorded so the same survey is not repeated.

| Option | Verdict |
| --- | --- |
| `odc-geo` / `odc-stac` (0.5.0 / 0.5.2 installed in `ndvi`) | Maintained and the modern successor to stackstac. Lazy-Dask by design, which is the wrong shape for the ≤100 km² synchronous path. Not needed: Icechunk plus `xr.open_zarr` already gives laziness without a task graph. |
| `stackstac` (0.5.1 installed) | Dormant — last release 2024-08-10, no commits since. |
| `rustac` | **Correction to an earlier premise:** this is `stac-utils/rustac`, not a Development Seed repo, and it is STAC *metadata* only — no COG, no pixel reads, no range requests. Irrelevant to read throughput. |
| `lazycogs` / `async-geotiff` (0.5.0 / 0.5.1 installed) | Real and current; a GDAL-free lazy COG→xarray path with built-in `MeanMethod`/`MedianMethod`. Optimises per-pixel decode, which was never our bottleneck (our cost is per-object opens). Worth revisiting only if the GDAL dependency itself becomes the problem. |
| `duckdb-cog` (`st-layer`) | GDAL-free COG reader with `RS_ZonalStats`; published head-to-head 95 ms vs PostGIS 273 ms. Their operational finding is the transferable part: **group zone calls by scene, not by zone** — 74 min → 17 min on a season-scale workload. |
| `VirtualiZarr` zero-copy TIFF→Zarr | The right answer at 115 TB. Wrong answer at 3.1 GB: Earthmover themselves "recommend native Zarr when feasible", and virtual stores inherit the source chunk shape and cannot be re-chunked. Native re-encode also costs ~$0.07/month here. |

---

## 4. Method note

Two measurement errors were made during this review and corrected before any
conclusion depended on them. Both are recorded because both would have pointed
at the wrong fix:

1. A **2000× speedup from GDAL environment tuning** was an artifact of GDAL's
   block cache persisting across measurement phases within one process. Ruled
   out by running each configuration in a cold process. In a cold process the
   tuning had **no measurable effect** on this endpoint.
2. **"DigitalOcean Spaces is bandwidth-capped"** was an artifact of the test
   host, not the endpoint. A control download from a Fastly-hosted CDN ran at
   0.03 MB/s from the same machine — *slower* than the endpoint under test. All
   absolute latency figures from that host are unreliable; the structural
   findings (cost is per object open, not per area) are not.

A third conclusion was also revised: `async-geotiff` appeared unable to read the
source files at all, which was a bug in the benchmark harness — it was handed a
COG URL where a STAC parquet catalog was expected.