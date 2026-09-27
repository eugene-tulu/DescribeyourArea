# Next feature: Portfolio Monitoring Packs

> **Status: delivered.** Everything in *First release* below shipped between 1.9.0
> and 1.12.0: precomputed versioned series, a lookup path that serves them, a
> supervised worker, self-service submission, and alerting. One deliberate change
> from the plan below: the portfolio is **not** administrator-gated. Areas are
> submitted by their own users, because the loudest question in the conservancy
> webinar was "how do we share our polygons", asked five times by four people.
> The live `generate-context` endpoint remains the path for arbitrary small areas.
>
> The *Guardrails* and *Not in the first release* sections are still the reason
> several things were deliberately left unbuilt, so they are kept. The current
> forward plan lives in [CHANGELOG.md](CHANGELOG.md#next).

Tracked in GitHub as [DescribeyourArea #3](https://github.com/eugene-tulu/DescribeyourArea/issues/3)
and [geocontextualize #1](https://github.com/kiprutoYG/geocontextualize/issues/1).

## Why

GeoContextualize already answers **what is this user-supplied area like now?**
The next product capability should answer **how are the named areas an
organisation cares about changing over time?**

This deliberately complements the live study-area workflow. It does not
replace it with a dashboard limited to preloaded areas.

## First release

- An administrator defines a small portfolio of named Polygon or MultiPolygon
  areas (for example, conservancies, projects, or counties).
- An offline, resumable worker creates versioned historical indicator series
  for each area: vegetation condition, rainfall/anomaly, fractional or land
  cover, and optional true-colour imagery where data licensing permits.
- The public app exposes a monitoring view with time-series charts, an
  explicit data-date label, provenance, and comparison across portfolio areas.
- The existing live `generate-context` endpoint remains the path for arbitrary
  uploaded or drawn areas.

## Guardrails

- Precompute and cache portfolio data; do not run multi-year raster analysis in
  an interactive browser request.
- Store the named-area definition, processing version, source version, date
  range, and refresh status with every output.
- Make refreshes idempotent and retryable, with clear partial-failure status.
- Establish a small privacy-preserving usage-event schema before the monitoring
  page launches. Keep aggregate outcomes, durations, selected modules, and
  coarse AOI-size bands; do not retain raw submitted geometries or full IP
  addresses for product analytics.

## Not in the first release

- Arbitrary large-AOI or bulk asynchronous analysis for every visitor.
- A wholesale switch from the current Planetary Computer workflow to a single
  DE Africa data source.
- Public user accounts, billing, or a general-purpose analytics product.

## Definition of done

1. A small test portfolio can be processed from source to published results
   without manual chart editing.
2. A viewer can select an area, see dated historical indicators, and compare it
   with another portfolio area.
3. Every indicator identifies its source, period, processing version, and
   latest refresh outcome.
4. Live arbitrary-AOI analysis remains bounded and unaffected by background
   monitoring work.
