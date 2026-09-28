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

## 1.17.0 — Two notification channels, and a completion message

### Webhook for machines, email for people

`notify.py` separates the two rather than letting one carry both. The alerting
pipeline stays on the webhook, because a webhook delivers in seconds and returns
something an automated system can act on. **Email is for humans** -- a breach a
conservancy manager should hear about, and the case a polling browser tab cannot
solve: someone submits an area, closes the tab, and three minutes later the work is
done.

A completion notice is a new trigger, not an alert. It carries the indicator, the
month count, the resolution, the grid cells, and **the cache key, which is the
retrieval handle** -- a test caught that a message without it was not actionable
even when the area had a name.

The webhook payload now sends both `text` and `content`, so a Slack or Mattermost
endpoint shows the message and a Discord one is not silently empty. That was the
one common exception worth handling.

### Email over AgentMail

Plumbed over plain HTTP with no SDK, matching how the webhook already works and
keeping the dependency list shorter.

```bash
AGENTMAIL_API_KEY=am_...
AGENTMAIL_INBOX_ID=inbox_...
ALERT_EMAIL_TO=you@example.org,someone@example.org
```

Neither channel is required. Unconfigured, both are no-ops and the sweep is
unaffected, and a channel that fails never prevents another from being tried nor
loses the record of which episodes are open.

An email carries the area *label* and key, the value, the threshold and the source.
Never a submitted geometry, never a raw area, never a client address -- an alert
that names a place is a disclosure of interest in that place, which is the same
restraint the usage log follows.

### The Spaces key

Confirmed the secret is rejected, not malformed: no carriage return, no quotes, no
stray whitespace, correct lengths. The value itself is wrong, so it needs re-copying
from the panel -- and the access key ID alongside it, because DigitalOcean issues
both together and a new secret with an old ID is a pair that never existed.

Tests 362 -> 377.

---

## 1.16.0 — A large area is a route, not a refusal

The last gap between "refused" and "answered". A bounding box past the
synchronous cap used to produce a string explaining the refusal and nothing else,
even though the worker will happily compute the same area at a coarser
resolution. The caller was told no and offered no alternative.

`POST /rainfall/plan` is the honest way to state the alternative: **read-only, no
queueing**, returning per module the resolution it would use and how long it
should take, with `already_computed` set for any that exist. A test asserts the
plan leaves the queue empty, because a preview that starts jobs is a different
thing than a preview.

The interface then offers it: on a 413 it shows the area against the cap, one row
per module with its resolution and estimate, and a single button to queue them all
and watch the progress line. **The resolution is listed rather than buried**,
because a 100 m land-cover composition is a different kind of claim from a 10 m one
and the user should see it before waiting three minutes for it.

Measured for a 2,462 km² landscape:

| module | resolution | estimate |
| --- | --- | --- |
| elevation | 100 m | 81 s |
| land cover | 100 m | 81 s |
| vegetation | 100 m | 81 s |
| rainfall | 27.8 km grid | 60 s |

### Blocked on a credential

The Spaces key pair in `.env` is rejected with `InvalidAccessKeyId`. Both values
are non-placeholder and correctly formatted, and the bucket worked earlier in the
session with the original pair, so the new secret is very likely paired with the
old access key ID. **Both halves must come from the same key.** Nothing is lost:
the 21 series already in the bucket are the ones published earlier, and pulling
still works once the pair is right.

Tests 356 -> 362.

---

## 1.15.0 — A wait you can read, and instrumentation that gates everything else

### "Ready in about 3 minutes"

A vegetation series is minutes of monthly reads, not seconds, and a spinner over
an unquantified wait is exactly what this product should not ship. So the
estimate is computed from the measured rate and **named as an estimate** rather
than presented as a promise:

> Ready in about 3 minutes. 201 monthly reads at 232 m.

It comes from `0.68 s per month measured at 4-way concurrency, plus overhead`, and
the reason is stated too: *cost is request latency, not pixels, so it barely moves
with area*. A 60 km² area and a 5,505 km² one both read in about 1.3 s, which is
why the plan for a small area and a large one is the same number. A test asserts
exactly that, so a future "optimisation" that makes the estimate area-dependent has
to be deliberate.

The estimate also corrected a blanket lie. The submit endpoint returned
`compute_seconds_typical: 60` for every indicator, which was roughly right for
rainfall and off by three times for a vegetation series. Planning is now per
indicator, because the cost drivers genuinely differ: the raster modules are
bounded by pixels at a policy resolution, a rainfall series by one annual ERA5 read,
and a vegetation series by the number of months it has to read.

The rainfall card now offers **"Build a vegetation series"** when there is none,
states why the wait is minutes rather than seconds, and shows the estimate rather
than a queue with no end.

### Instrumentation, complete and wired

One event per analysis, verified end to end: datasets requested, a per-module
verdict, per-module and total duration, a coarse area band, the sensor chosen,
and a truncated client prefix. No submitted geometry, no raw area, no full address
— asserted by tests that walk the record for anything coordinate-shaped.

The developer side is `GET /analytics/summary`: outcome counts by module, dataset
request counts, area-band distribution, sensor distribution, and p50/p95 latency.
**Aggregates only.** The per-event rows are deliberately not returned — a row
carries a timestamp, a duration and an area band, and a long enough tail of those
starts to describe a person even with no directly identifying field. A test
asserts the response contains no raw area value and only the band.

This is the piece that gates prioritising everything else: which modules are
actually used, which silently degrade, and where the latency sits.

Tests 346 -> 356.

---

## 1.14.0 — Rainfall and vegetation on one time axis

The comparison the DEA dashboard appears to offer, built on data that is actually
there: 189 months of ERA5 rainfall and 189 months of MODIS NDVI over the same
months, each with its own climatological normal, on one shared time axis.

### Two panels, not a dual axis

Millimetres and NDVI cannot share a scale without implying they are comparable,
and implying that is the one thing this product exists not to do. So: rainfall as
bars with its normal as a stepped rule on top, vegetation as a line with its
normal dashed beneath, sharing the time axis. The reader compares the *shape* of the
two seasons vertically, which is the honest reading.

The single-series chart remains for a series with no vegetation counterpart, and the
figure degrades to it rather than showing an empty panel.

### Why MOD13Q1 and not Landsat

A 16-day NDVI **product**, already cloud-masked by NASA, so no cloud decisions are
made here and its cadence maps onto ERA5's monthly buckets. Landsat monthly is
possible but thin at 16-day revisit, and thin composites produce swings that look
like change and are not — the failure mode already diagnosed once in this project
and again in the Baringo work.

Two facts taken from that work rather than re-derived: the QA_PIXEL bit mask it
uses is exactly ours, and so are the radiometric constants. Its more useful
contribution is the principle behind its water mask — a single index cannot
separate open water from moist soil, so each pixel must satisfy several
physically independent signals. That is structurally our SCL class-5 problem, and
it is the pattern for any classifier in a surface where one signal is unreliable.

### Measured cost, and the concurrency that made it viable

Over a 5,505 km² area: 1.42 s per month sequentially, **0.41 s at 4-way
concurrency**, so 1991 to present is about 3.5 minutes. Eight-way was *worse* than
four, which is server-side contention rather than local, so the default is four.

Four bugs found by building it, all of the same species as the rest of this
project — a right-looking number that is not what it claims:

- One STAC search per month meant 420 requests for a 35-year span, and one came
  back as a connection reset. Now one paged search for the whole span, bucketed
  locally, with the search rebuilt per retry so a reset cannot yield silence.
- `geometry_mask` does not reproject, and the MODIS grid is a custom sinusoidal
  CRS. The unprojected WGS84 polygon landed millions of metres away and the mask
  came back empty, which surfaced as "this area covers no whole MODIS cell".
- The baseline was labelled **1991-2020**. MOD13Q1 begins in 2000-02, so it was
  really 2000-2020. It now reports the period it actually used, and names
  `nominal_start` separately.
- The native resolution is **231.656 m**, not 250. The grid is sinusoidal. Calling
  it 250 m would be a small lie propagated into every stated resolution.

### What the data shows

Over Melako, the vegetation climatology is textbook East African bimodal: low
through June to September (0.206 down to 0.179), peaking at **0.300 in April** for
the long rains and **0.279 in November** for the short rains. Driest month
2022-03 at 0.107, wettest 2018-04 at 0.667, and **zero thin months** across 189.
The 32 KiB artefact is smaller than a screenshot.

### Honesty, structurally

- A rainfall overlay beside vegetation is exactly where an implied causal claim
  creeps in, so the caveat is a module constant, asserted by a test, and printed
  under the figure: *co-variation is not attribution; vegetation responds with a
  lag that varies by season, and water is not always the limiting factor.*
- Every month carries its valid-pixel fraction, and any month below 90% is listed
  as thin and surfaced beneath the chart.
- The two series carry different epistemic statuses — rainfall `modelled`,
  vegetation `derived` — and the figure says so.

Tests 331 -> 346.

---

## 1.17.0 — Snapshot and series are different questions, and the window could not tell them apart

Asked how to tell a user who wants a quick look at a place from one who wants a
trend. Answering the question turned up a defect in how the window worked at all.

### A 30-year window was returning six weeks

`_search_items` capped the STAC query at `limit=max_items=max_scenes` and *then*
sorted the results by cloud cover. The search therefore returned only the most
recent items, the sort had nothing to choose from, and the "clearest scenes" logic
was decorative. Measured over the same study area:

| window asked for | scenes returned | what they actually were |
| --- | --- | --- |
| 90 days | 4 | 2026-08-06 to 2026-08-30, 0.2–17.5% cloud |
| **30 years** | 4 | **2025-11-23 to 2025-12-09 — six weeks** |

Both were then medianed into a single number and returned as though the window
meant something. Raising the candidate pool to `max(4 × scenes, 50)` and
truncating *after* the sort fixes it: the 5-year window now selects from 50
candidates spanning 1.9 years, with the widest pool for selection and only the best
four composited, so the read cost is unchanged. This is the same failure the
project keeps finding — a number that looks like what was asked for and is not —
and it was there precisely because one control was being asked to serve two
incompatible intents.

### Why one control cannot do both

A snapshot is a **state** and a series is a **trajectory**. They are not the same
computation over a different span:

- A snapshot is a median composite over a handful of recent, clear scenes. That is
  a legitimate estimate of "what this place looks like now", and `max_scenes=4`
  is the right shape for it.
- A series is **monthly values**. It cannot be produced by medianing four scenes
  chosen for clarity, at any window length. Widening the window does not make a
  median into a trend; it only makes the sample more arbitrary.

So the fix is not a wider window, it is a different reduction, and the mode has to
be explicit because one is cheap and synchronous and the other is worker work.

### The uncomfortable third thing

**Elevation and land cover have no time dimension at all.** NASADEM is a static
DEM and ESA WorldCover a single-date classification, which the evidence line
already says: *"a single-date classification, so it reflects the scene, not a
year."* The data has been telling us this and the window control was offering a
capability two of the four modules do not have.

### What is proposed, not yet built

A question rather than a mode toggle, because the audience is mixed and "snapshot
vs time series" means nothing to a ward planner:

- **"What is it like now?"** — the numbers, no chart, no window control, cache
  hit, synchronous. The default, and what the webinar audience actually asked for.
- **"Is it changing?"** — monthly values, a per-decade slope, the series shown so
  it can be checked. Worker territory, and visibly a different cost.

The mode is also the unit of cost, which matters against the droplet credits and
eventually for pricing: a snapshot is a cache read, a series is a job.

Tests 328 -> 331.

---

## 1.16.0 — The watermark removed from the satellite basemap, and user-defined date ranges

### The watermark was CARTO, not Esri

Reported as a subtitle reading `carto.com/basemaps/apikey`. The map had **two**
tile layers: Esri World Imagery underneath, and CARTO's `light_only_labels`
overlaid on top. The watermark belonged entirely to the second.

The first attempt at this got two things wrong. It read only the first tile
layer, so it blamed Esri, and it then replaced that layer — removing the satellite
imagery the product had — while leaving the CARTO overlay in place, watermark and
all. Replaced again, correctly this time.

**Shipped:** Esri World Imagery retained as the base, with CARTO's label overlay
swapped for Esri's own `Reference/World_Boundaries_and_Places`, which is the same
idea — light labels and boundaries drawn over dark imagery — and is keyless. All
four candidate services were fetched and confirmed to return real tiles before
being wired in:

| service | key | bytes at z6 |
| --- | --- | --- |
| `World_Imagery` (base) | no | 6,008 JPEG |
| `Reference/World_Boundaries_and_Places` (labels) | no | 885 PNG, mostly transparent |
| `Canvas/World_Light_Gray_Base` | no | 2,419 JPEG |

CARTO is gone from the client entirely.

### On the alternatives considered

**CARTO Positron is not an option any more.** As of 2026 the basemaps require a
free API key even for non-commercial use, so it would have replaced one watermark
with another.

**OpenFreeMap is the right answer for a vector basemap** — keyless, no limits, no
cookies, MIT, commercial use explicitly allowed, self-hostable — but it is a
light cartographic base, not imagery, so it does not serve this requirement. It
remains the answer if the design later moves to a printed-map aesthetic.

For imagery, the honest position is that **Esri's keyless tile service is what
works today and its terms have tightened**, so it belongs in the same category as
the MapTiler key it replaces: functional, unkeyed, and worth re-checking before
any growth. The durable keyless alternative is **Copernicus Sentinel-2 cloudless**,
which is thematically ideal for conservation and is the imagery the product
already analyses — but it is published as WMS rather than XYZ tiles, so Leaflet
would want it as a WMS layer. Recorded as the fallback rather than wired in blind.

### User-defined date ranges

The backend has accepted an explicit `window_start`/`window_end` since 1.14.0 — only
the control was missing. There are now two date inputs beside the 1/3/10/30 presets,
a stated effective window so it is never ambiguous which is in force, and an invalid
range blocks the request rather than producing a 422.

Wiring it up exposed a real gap: **an inverted or malformed window was absorbed by
the vegetation module and returned `unavailable`**, which tells an API caller the
area has no data when the truth is that the request was wrong. Both are now `422`
with a specific message. A window that predates the sensor's archive is still
`skipped`, but it now says so and names the archive start.

Verified against live Planetary Computer: 1997-01 to 1999-12 returns NDVI
0.2396, which is the 1997/98 signal the range exists to ask for.

Tests 324 -> 328.

---

## 1.15.1 — The light theme was unreadable; contrast is now tested

Reported as "I am struggling to see the copy". Measured, the cause was a hard
failure, not a matter of taste:

| class | on the new paper ground | |
| --- | --- | --- |
| `text-white` (17 occurrences, left from the dark theme) | **1.08:1** | invisible |
| `text-slate-200` | 1.14:1 | fails badly |
| `text-slate-300` (10) | 1.37:1 | fails badly |
| `text-slate-400` (9) | 2.37:1 | fails badly |
| `--ink-3` (my own label grey) | 2.63:1 | fails at 11px uppercase |

I changed the ground from a dark gradient to paper and did not change the ink.
Tailwind made that possible, which is the actual lesson: a utility like
`text-slate-300` is an **absolute** colour, so moving `--paper` moved the
background and not one piece of text. The same commit also shipped a label grey
that failed at the size it is used.

### Fixed by measurement, not by eye

- `--ink-2` and `--ink-3` darkened until every text token clears its floor on both
  the paper and a white card. `--ink` 16.8:1, `--ink-2` 7.2:1, `--ink-3` 5.5:1,
  `--accent` 7.4:1, `--caution` 6.6:1, the modelled-data chip 6.4:1.
- All 65 dark-theme class occurrences replaced with the token utilities. No
  `text-white`, `text-slate-*`, `bg-slate-9*` or `bg-white/10` remains.

### The durable answer to "what about Tailwind"

We already use it — v4, CSS-first, with `@theme inline` surfacing `:root` tokens
as `--color-*`. It is the right tool and it stays.

What changes is the rule: **tokens are the only way colour is expressed.** That is
what makes `text-ink-2` a semantic utility rather than a hex, and it is why a
future theme change moves ink and ground together.

### Now tested rather than trusted

`tests/test_theme_contrast.py` resolves each token through the `@theme` mapping and
asserts it clears its contrast floor on both surfaces, that the ink hierarchy is
ordered, that `--paper` is actually light, and that the dark theme's ink classes
appear nowhere in the markup. Sixteen assertions that fail the moment this
regresses, which is the response to having shipped an unreadable page and called
it verified.

Tests 316 -> 324. Two failures on the first run were transient live-data reads
against Planetary Computer; they pass on rerun.

---

## 1.15.0 — The interface rebuilt as a document rather than a dashboard

A design change with a rule behind it: the page should read like a survey
document, because that is what it is.

### The basis

**International Typographic Style, as a system.** Müller-Brockmann and Ruder built
the visual language of Swiss cartography and scientific information out of a grid
and a type hierarchy with no ornament. That is not decoration for this product; it
is the honest form for measured values with stated uncertainty, arranged so a
reader can check them.

**Crouwel is the constraint behind it.** Two greys, one accent, one prose face, one
figure face, one grid, written down once and applied without discussion. The
invention appears in what you do with the constraint, and the page becomes
coherent without anyone needing taste.

**Ruder: hierarchy through position and rule weight, never through containers.**

**Vignelli, against originality.** "A system, not a style." Do not redesign; make
one system and apply it. Unfashionable, and correct for a small product.

**Sutherland, taken literally.** Everyone claims to be excellent, so the claim is
worthless. What we can claim, and can be checked, is narrower: *we tell you what
kind of number this is, and we tell you when we do not know.* That is now the most
distinctive thing on the page rather than a grey footnote.

### What changed

- **`globals.css` rewritten** around the system: paper, ink, two greys, one accent,
  a caution colour reserved for modelled or unconfirmed figures, a 68-character
  measure, and rules instead of boxes.
- **Every figure is monospaced and tabular, everywhere, without exception.** The
  single highest-value rule in the set: aligned digits cannot be misread
  column-wise, and it reads as instrumentation rather than marketing.
- **The card grid is gone.** One column divided by rules. A card grid reads as a
  dashboard; this reads as a document.
- **The evidence panel.** Each module states what kind of number it is, its source,
  and its method, in the same monospace as the figure, so provenance reads as part
  of the measurement instead of a disclaimer underneath it.
- **"Working", disclosed.** Grid cells, pixels, scenes examined, valid-pixel
  fraction, the resolution actually read against the one requested, the window,
  the DOI. All of it was already computed and thrown away.
- **The wait is described rather than decorated.** A submitted area shows its real
  stages, timestamps and planned resolution and pixel count. Every one of those
  fields existed; none was rendered. This is the same move as the evidence panel
  applied to time.
- **A 1/3/10/30-year window and a source picker** that says why `auto` chose what
  it chose, plus a least-squares trend per year beside the mean.
- The `ResultCard` and `Metric` components are deleted, along with the dark
  gradient surface.

### A stale limit, found by looking

The help text still advertised a 10 km² vegetation limit, raised to 100 km² in
1.11.0. It survived a release and a review because nothing rendered the page.
It is now correct.

### Honest limits of this verification

`tsc`, `eslint` and a production build are clean, and the new components are
confirmed present in the client bundle. **The results layout has not been seen in a
browser** — there is no headless browser available in this environment, and the
results branch only mounts after an analysis. The shell, the controls and the
caveats were fetched and inspected; the module sections, the evidence line and the
Working panel were verified by type-check, build and bundle inspection only. They
should be looked at before anyone else is.

Tests unchanged at 316; this was a frontend change with no backend behaviour
touched.

---

## 1.14.0 — A declared contract, a resolution policy, and analytics you can refuse

The consolidated list. Four capabilities, all verified against live data.

### The response contract is declared, and every figure says what kind it is

The endpoint declared `summary: Dict[str, Any]`, so `/openapi.json` documented
nothing and the frontend carried twelve hand-written TypeScript interfaces with
nothing asserting they matched. That is how a duplicated literal drifted and
shipped a stale limit to production — found by writing a test after the fact,
which is the wrong order.

`contract.py` now declares the summary, and the response validates against it
before it leaves, so a producer that changes shape fails the request rather than
shipping. It caught a design slip on its first live request: the status enum
omitted `"ok"`, which is the only value a successful response ever has.

Each figure carries an **epistemic status**, borrowed from a Japanese data map
that marks every value 実測 or 推計:

| module | status | why |
| --- | --- | --- |
| elevation, land cover | `observed` | a measurement of the surface |
| vegetation index | `derived` | computed from a reflectance product |
| rainfall | **`modelled`** | ERA5 is a reanalysis, not a gauge reading |

That last one is the distinction most worth stating and easiest to lose. An
absent module is `unconfirmed` rather than silently missing, and a `caveats` list
names every skipped module, a repaired boundary, suspicious months, and where the
rainfall series stops.

### A resolution policy, so a large area is answered rather than refused

The 100 km² cap is a guard on the *request* budget, not a claim about what the
data can do. It was standing in for a resolution policy recommended in review and
never implemented.

| bounding box | resolution | pixels |
| --- | --- | --- |
| ≤ 100 km² | 20 m | 0.25 M |
| ≤ 1,000 km² | 60 m | 0.28 M |
| ≤ 10,000 km² | 100 m | 1.0 M |
| beyond | 250 m | — |

A **2,462 km² landscape** — the "Laikipia, not a conservancy" unit from the
webinar — is now computed end to end: DEM, land cover and vegetation, all three
at 100 m over 246,176 pixels, through the worker. A coarser source is never
upsampled, and when the chosen source publishes a coarser product than requested
the artefact records both the target and what was actually read, with the reason.

### Jobs handle one indicator each, and the queue grew an indicator dimension

`jobs` was wired to rainfall alone, keyed by geometry — so submitting dem and
ndvi for one area collapsed into a single record and the second silently replaced
the first. Now `(area, indicator)` is the job identity, artefacts live in separate
directories, and a runtime indicator skips the request budget entirely, because
answering what the request path refused is the worker's whole purpose.

### Analytics you can refuse

We recorded usage events with no opt-out, for an audience of community
organisations. `GET /analytics` now states plainly what is recorded; a caller can
send `Analytics-Do-Not-Track: true` for one request, or an operator can set
`ANALYTICS_DISABLED=1`.

### Frontend

A 1/3/10/30-year window selector and a source picker that defaults to `auto`, a
least-squares **trend per year** beside the mean, an evidence line under every
card, the caveats rendered where they will be read, and a share link that
reproduces the analysis.

### Cleaned payload

`response_model_exclude_none` removed the nulls FastAPI was re-adding to every
declared field: a 2,282-byte response instead of one padded with empties, and a
module that was not requested is absent rather than a field of nulls.

### Bugs found while building this

- The autorun handed a coroutine to `asyncio.to_thread`, which never awaited it,
  so submissions sat pending forever. It now spawns the same worker process the
  container runs, which also avoids sharing the server's event loop with a
  vegetation path that takes a loop-bound semaphore.
- `complete()` inferred the indicator from the payload instead of being told, and
  wrote the job record to a different directory than the reader looked in.
- The resolution policy's comparison was inverted, so it returned native
  resolution at every size.
- A bare `NameError` on the DEM path was invisible because a failing job records
  a one-line reason. `RAINFALL_JOB_DEBUG=1` now re-raises with the traceback.

Tests 291 -> 316.

---

## 1.13.0 — A visitor can submit an area; launch-readiness audit

### The headline feature was unreachable from the UI

A pre-launch walkthrough of exactly what a first-time visitor does — draw a box,
press Analyse — found that `POST /rainfall/submit` had **zero references in the
frontend**. The backend could queue an area and the API could report its state,
and a user could do neither. The question the conservancy webinar asked five times
had an endpoint and no button.

The rainfall card now offers **"Process precipitation for this area"** on a
missing series, shows a queued or queue-full state, and the submission is keyed by
the cache key the backend already returned, so switching areas does not show
another area's progress. The active geometry is now one `currentGeojson()` helper
shared by analysis and submission, replacing a block of duplicated logic.

Verified end to end against live ERA5 and the live bucket: analysis returns
`not_computed` with a key, the button queues it, the worker has it `ready` in
under a second, and a re-analysis returns `ok` with 195 months and a 628.8 mm
annual normal.

The not-computed card also stopped explaining a dead end. It said why no series
existed and offered nothing to do about it; it now says no series has been
processed for this boundary yet, and offers the action.

### What else the audit found, and what is still open

- **A modest draw can be refused.** A 0.1° × 0.1° box is a 123 km² bounding box
  against a 100 km² synchronous cap, and gets `413` with a clear reason. Correct,
  but the UI does not route the visitor to the uncapped rainfall path, which would
  have served it. Left as a follow-up; the cap is measured, and raising it is a
  deployment decision rather than a UI fix.
- **Alerting is inert until a webhook is set.** Rules evaluate and episodes are
  recorded, but nothing is delivered. One environment variable.
- **`@radix-ui/*` packages are unused** after the component trim, still in
  `package.json`.

Not blockers, but worth knowing before a post drives traffic: an area over the
cap is refused rather than partially served, and NDVI defaults to Landsat rather
than Sentinel-2 because its cloud mask is the one that can be trusted.

---

## 1.12.1 — Credential rotated, and the repository trimmed

### Credential rotated

The Spaces secret was exposed in plaintext by `docker compose config` during
1.12.0 and has been rotated. Recorded here because the exposure is a fact about
this repository's history, and because the lesson generalises: `docker compose
config` resolves `env_file` values, so it is not a safe way to inspect a rendered
configuration on a machine holding live credentials. Verify mounts from the
compose file, or from output with the credential lines filtered.

### A store outage could fail work that had already succeeded

Found by the full suite, not by reading the code: `publish()` had no error
handling, so when the object store was unreachable the upload raised inside
`run_pending`, which caught it and marked the job **failed** — even though the
series had been computed and written to the local cache, and was fully usable.
A transient Spaces fault was discarding a successful ERA5 computation.

`publish()` now returns a boolean and never raises. The job is recorded `ready`
with `published: false`, which is the truth: the series exists locally, the upload
did not happen, and the next deploy's pull reconciles it. The same principle the
alert webhook already followed — a side effect that can fail must not destroy the
result of the work that succeeded.

It also turned out the queue tests had been reaching ERA5, because `run_pending`
builds a real `UnionReader` before the compute is stubbed. They now pass a source,
so the queue is testable without a network.

### Files removed

Everything below was tracked but not part of the product, and the first three were
actively misleading.

- **`__pycache__/main.cpython-312.pyc`** — a compiled bytecode artefact, tracked,
  and for Python 3.12 while the environment runs 3.13. It is in `.gitignore` now
  but had been committed before that, so ignoring it did nothing.
- **`benchmark_final.json`, `benchmark_results.json`** — a loading micro-benchmark
  of an EOPF/Zarr path removed in 1.5.0. They measured code that no longer exists
  and their numbers were being cited as if current.
- **`Procfile`** — a Heroku `web:` declaration, from a deployment model this project
  does not use. It deploys by Docker Compose to a droplet, and its presence
  implied otherwise.
- **`.kilocode/`, `.vscode/`** — editor and agent configuration, already listed in
  `.gitignore` but committed before it.
- **`client/README.md`** — the Next.js scaffold readme, pointing at
  `describeyourarea-production.up.railway.app`, a Railway deployment that is not
  the one in use. Stale infrastructure information in a tracked file is worse than
  no file. The root README documents the client configuration.
- **35 unused `client/components/ui/` components** — the shadcn/ui library was
  scaffolded in full and six were ever used (`alert`, `button`, `card`, `input`,
  `label`, `toast`). The rest are dead weight in the image.
- **Five scaffold SVGs** in `client/public/` — `file`, `globe`, `next`, `vercel`,
  `window`. Nothing referenced them; `icon8.png` is used by `layout.tsx` and stays.

`NEXT_FEATURE.md` is **kept** and given a status header. Its *Guardrails* and *Not
in the first release* sections are the record of what was deliberately left
unbuilt, which is worth more than a clean tree, and it links two GitHub issues.

One follow-up not taken: the dropped components leave their `@radix-ui/*` packages
in `package.json`. Pruning them is worth doing, but it is a dependency change with
a real blast radius and no user-visible benefit, so it is noted rather than rushed.

Tests 288 -> 291; `tsc`, `eslint` and `next build` clean after the removals.

---

## 1.12.0 — Worker supervision and alerting

### Supervision

A `worker` service runs `python -m worker`, a supervised loop around the existing
queue: it drains pending jobs, sleeps, and sweeps alerts. Three things make it
more than a `while True`:

- **An `flock` on the shared volume**, so a second worker exits rather than
  duplicating ERA5 reads. `flock` rather than a lock file, because the kernel
  releases it when the process dies and a container restart cannot leave a stale
  lock wedging the queue.
- **Exit after three consecutive sweep failures**, so a worker that cannot do its
  job is restarted rather than spinning quietly. A queue that looks healthy while
  nothing is computed is worse than a visible crash. A single failure, or a bad
  area, does not trip it.
- **A shared volume.** The cache lived in the image filesystem, so a separate
  worker would have written series the backend never sees. `rainfall-cache` is now
  a named volume mounted into both. The frontend does not mount it, and the worker
  does not run the portfolio pull, which the backend owns.

### Alerting

`alerts.py` re-checks the precomputed series on every sweep and notifies on a
threshold crossing. Three shipped rules against each calendar month's 1991-2020
normal, so a percentage means the same thing in a wet and a dry month: a
12-month anomaly at or beyond −40% (severe drought), −20% (drought watch), and
+40% (flood watch).

Two properties matter more than the thresholds:

- **Once per episode.** State is keyed by area and rule, so a dry season produces
  one message rather than one an hour for months. **Recoveries are announced too**,
  because a notification that only ever says "bad" teaches people to ignore it.
- **Self-describing.** Every message carries the value, the threshold, the window,
  the source and its DOI, because the recipient has to defend the number to
  somebody.

`evaluate_and_notify` returns what was **delivered**, not what was attempted, so
"what was sent" is a fact rather than a claim. Episodes are persisted even when
nothing was delivered, so configuring a webhook later does not re-announce every
open alert. A webhook that is down prints a line and does not stop the sweep.
Delivery is one generic POST, which covers Slack, Discord and anything else.

Series now carry a `label` and a `recent_3m` window, so a notification can name
its area and the three-month metric works.

Verified against the live portfolio: **4 of 21 conservancies currently in drought
watch** — Melako −37.7%, Biliqo Bulesa −34.7%, Sera −24.8%, Kalepo −24.0% — with
4 episodes recorded and nothing delivered, because no webhook is configured.

### Two things I got wrong, both from testing against reality

**I published a test polygon to the live bucket.** Verifying the submit loop
earlier, the autorun computed a throwaway area and uploaded it. I dropped the
local job record and never deleted the object, so a test artefact sat in
production storage until the alert sweep counted 22 series where 21 were
expected. Removed.

Because that had already happened once with orphaned pre-repair objects, the sync
tool gained `--prune`, which reconciles the bucket against the areas a GeoJSON
accounts for. It lists by default and only deletes with `--yes`. The bucket now
matches the expected 21 exactly.

**A broad `except` hid a code fault as bad data.** `load_areas` still referenced
the two constants removed in 1.11.0, so every area raised `NameError` and printed
as `INVALID` — twenty-one identical lines that read like a data problem. It now
catches `HTTPException` separately, reports an unexpected exception with its type
as `ERROR`, and uses the current single payload cap with the vertex check waived.

### A credential was exposed in a terminal transcript, and rotated

While verifying the volume I ran `docker compose config`, which **prints resolved
secrets in plaintext** — the Spaces access key and secret went into that shell's
scrollback. The secret has been rotated. Worth knowing for next time: render
compose config with the credentials stripped, or verify the volume mount from the
file rather than the resolved output.

### Tests: 263 -> 288

25 new: lock exclusivity and release, a worker that stops on repeated failure and
survives one blip, sweeps draining the queue, rule evaluation with evidence
attached, a breach announced once, recovery announced, a second episode announced
again, state surviving a process boundary and a corrupt file, an unconfigured
webhook recording the episode without announcing, a delivery failure losing
neither the sweep nor the state, and the webhook payload carrying its evidence.

---

## 1.11.0 — Self-service submission, and the cap consolidation

### Self-service submission

`POST /rainfall/submit` queues a study area and returns immediately; computing a
series takes about 60 seconds of ERA5 reads, so the submit path never waits. The
job record is what makes this durable rather than a promise: it lives on disk, so a
submission survives a restart, and a runner completes whatever is still pending.
The polygon is kept with the job so the runner needs no caller, and is dropped the
moment the series exists.

`GET /rainfall/status?cache_key=` reports `not_submitted`, `pending`, `running`,
`ready` or `failed` without resending the polygon. The cache is authoritative
over the record: if a series exists the area is ready whatever the record says,
because a runner may have completed without updating it.

Verified end to end against live ERA5 and the live bucket: `not_submitted` →
`pending` → `ready` → lookup `ok`, with the polygon gone from the job record
afterwards. An area that already has a series is reported ready and never queued;
resubmitting a pending one returns the existing job; a queue with no worker behind
it is bounded at 25 so it cannot grow without limit.

There is no supervision. `jobs.run_pending` is the runner, and running it from
cron or a sidecar is the difference between "usually works" and "reliably works".

### The cap consolidation

There were 19 operational knobs, and having four numbers where one belongs is how
the wrong one ships — which is exactly what happened with the vegetation cap in
`docker-compose.yml`, left at its pre-1.4.0 value while the code moved on.

**Removed:**

- `MAX_NDVI_BBOX_KM2` — duplicated the synchronous cap at the same value, so it
  configured nothing and only created a second number to keep in step.
- `MAX_LOOKUP_BYTES` and `MAX_LOOKUP_VERTICES` — added in 1.10.0 because the lookup
  path reuses the raster canonicaliser. The correct fix was one cap with the
  lookup exempt, not a second pair.
- `MAX_WINDOW_DAYS` — set at 12,000, above the 29 years between 1997 and today,
  i.e. above any request a person would make. A cap that provably cannot fire is
  documentation pretending to be a control.

**Now 14 knobs in three labelled groups:** 8 guards, each with a measured failure
behind it; 3 timeouts, which bound waiting rather than input; 2 product defaults,
which are choices rather than limits (`MAX_PC_SCENES`, `NDVI_TARGET_EPSG`). Data
quality thresholds in `rainfall.py` and `sensors.py` are grouped separately so
they stop reading as limits.

### The duplication was concealing a broken deployment

`client_max_body_size` was 512k in both Nginx configs while the application
advertised 4 MB, so the 1,358 KB NRT conservancies — the largest in the published
portfolio — **could not be submitted through the deployed service at all**. They
only ever worked in a local test. Both configs are now 4m, matching a single
`MAX_GEOJSON_BYTES` of 4,000,000, and a test asserts the proxy is never *below*
the application, because a lower proxy limit only moves the rejection somewhere
less explicable. This closes the mismatch DEPLOYMENT has flagged since 1.3.0.

### A test-hygiene bug worth recording

`load_dotenv()` ran at import, so a developer's own bucket and credentials entered
**every test process**. An assertion that an area had no cached series could be
satisfied by the live bucket — which is exactly how `test_a_miss_is_reported` came
to fail intermittently, and why a run was mutating real storage. Tests now set
`GEOCONTEXT_NO_DOTENV=1` before importing the app, tests that assert a miss clear
the remote explicitly, and a stubbed object client is installed with
`mock.patch` so it is restored rather than outliving the test.

### Tests: 242 -> 263

16 for the job queue, including idempotence, a failure not stopping the run, a
bounded queue, a stale `ready` record, and the polygon being dropped. Plus tests
that the removed caps are really gone, that the lookup is exempt from the vertex
cap, and that the proxy and application agree on body size.

---

## 1.10.0 — The portfolio is live, and reachable

Built and published all 21 Northern Rangelands Trust conservancies to Spaces
(21 objects, 603 KiB) and served **21/21 from a cold local cache in 22 s** with no
local build. Getting there exposed four defects, three of them structural.

### Rainfall was gated behind the raster admission policy

Putting rainfall in `/generate-context` was my mistake. The admission policy there
is about bounding *raster* work, but reading a cached series costs about a
millisecond and 1 KB. Applied to rainfall it rejected the portfolio it had just
been built for: **8 of 21 on payload size** (the largest is 1,358 KB against a
500 KB cap) and the rest on vertex count, before a cached value was consulted.

`POST /rainfall` is now separate, with its own measured limits —
`MAX_LOOKUP_BYTES` 4,000,000 and `MAX_LOOKUP_VERTICES` 250,000 — and no area cap
at all. It also accepts a bare `cache_key`, so a caller with the key never
resends a polygon: a lookup for Melako answers in **7 ms with a 2-byte request**,
against 800 KB for the polygon.

The separation is tested in both directions: a 12,000 km² area is refused by
`/generate-context` and served by `/rainfall`.

### Two published conservancies could never be looked up

The build path hashed the **raw** geometry while the read path canonicalised it.
Two of the 21 — Kalama and Leparua — carry ring self-intersections in the
published source file, so the build published series under a key the lookup could
never derive. Two entries in the portfolio were permanently unreachable.

Both now go through one shared canonicaliser, and a self-intersecting ring is
**repaired rather than refused**: a defect in someone else's file is not a reason
to lose a conservancy. The repair is reported as `geometry_repaired` in the
canonical properties rather than applied silently, and it is refused when the
area would change by more than half.

A bow tie has no meaningful signed area, so the area-similarity guard is skipped
for exactly the case it was written for. That is now deliberate and commented: the
repair is the only way such a geometry becomes usable, and a genuine problem
fails the `is_valid` check on the result instead.

This also left **3 orphaned objects** in the bucket from the earlier build, which
were unreachable by construction. Removed; the bucket now matches the expected set
exactly at 21.

### A network blip looked like "this area has no data"

`fetch` swallowed every exception, so a transient Spaces failure was
indistinguishable from a genuine miss — a wrong answer to a user rather than an
absent one. It now tells them apart and retries only the former: a `NoSuchKey` is
believed immediately, a connection reset is retried once. Verified against both a
flaky and a missing-object stub.

### Tests: 223 -> 242

19 new, covering the admission split, the repair (including that both paths hash
identically, which is the invariant that was broken), the lookup endpoint and its
guards, and the fetch reliability split. Plus 5 fix passes on my own test bugs —
including a bow-tie area assertion that divided by zero because a
self-intersecting ring has no signed area.

---

## 1.9.1 — Spaces round trip verified against the live bucket

With credentials configured, the first end-to-end run against the real DigitalOcean
Spaces bucket found three defects that the stub client could not, because both
sides of every assertion shared the same wrong convention.

### 1. A bucket-qualified endpoint is unusable as given

DigitalOcean documents both `https://<bucket>.fra1.digitaloceanspaces.com` and
`https://fra1.digitaloceanspaces.com`. boto3 puts the bucket in the hostname
itself, so the first form requests
`<bucket>.fra1.digitaloceanspaces.com/<bucket>/...` and fails with `NoSuchKey`.
Measured against the live bucket:

| Endpoint form | Result |
| --- | --- |
| `https://primero.fra1.digitaloceanspaces.com` | `NoSuchKey` |
| `https://primero.digitaloceanspaces.com` | `NoSuchKey` |
| `https://fra1.digitaloceanspaces.com` | works |

Both spellings are now accepted: a leading bucket is stripped so boto3 adds it
back, and a legacy form takes the region from the configuration.

### 2. Publishing and listing disagreed about the key

Publishing wrote to `geocontextualize/rainfall/series/<key>.json`; the sync tool
listed under `primero/geocontextualize/rainfall/series`, so a **successful
publish reported zero objects**. An object key does not repeat the bucket. The
write side and the read side now share one helper. A stub client could not catch
this, because the stub agreed with both.

### 3. Deletion only cleared one of the two stores

`forget` removed the local file and left the published object in the bucket —
which is the copy that survives a redeploy, and therefore the copy a deletion
request most needs to reach. It now clears both and reports `local` and `remote`
separately, so a partial failure is visible rather than assumed.

### Also fixed: the upload flag shadowed the upload function

`build_and_cache(..., publish=False)` shadowed the module-level `publish()`, so
`--publish` called a **bool** and raised `TypeError: 'bool' object is not
callable`. That path only executes when uploading, which is why no test had
reached it. The flag is now `upload`, with a test that asserts the name and
another that asserts an upload actually happens.

### Verified against the live bucket

Build from ERA5 → publish → list → pull into a clean cache on a second "host" →
serve a request from that cold cache → forget → bucket empty. The served series
carried its full provenance: ERA5, 2 cells, 135 months, a 735.1 mm annual normal
and the source DOI. Test objects were removed; the prefix is empty.

### One test flake of my own

Two window tests asserted `date.today()` across the call. A run that crossed
midnight UTC saw `2026-09-26` on one side and `2026-09-27` on the other. They now
assert the span, with a day of tolerance on the end date.

Tests 211 -> 223.

---

## 1.9.0 — Privacy-preserving usage events, and a way to honour a deletion request

### The schema, with the privacy properties made structural

`usage.py` exists so the guarantees cannot be forgotten. `build_event` has **no
parameter that could carry a submitted polygon** — a test asserts the absence of
every geometry-shaped name in its signature — the area is reduced to a band
before it leaves the builder, and the client address is truncated to a /24 (or a
/64) inside it. `assert_no_geometry` walks the finished record for anything
coordinate-shaped, so a future field cannot reintroduce a polygon unnoticed.

Bands align with the shipped caps (`0-10`, `10-100`, `100-1000`, `1000+`), so a
pile-up in the `10-100` band reads directly as "the synchronous cap is too tight".
A recorded outcome is a verdict — `ok`, `skipped`, `unavailable`, `not_computed`,
`error` — never a measurement, because `{"mean": 1650.25}` is a fact about a
place. A test asserts the numbers are absent from the serialised record.

```json
{"v":1,"at":"2026-09-26T20:52:11+00:00","datasets_requested":["dem","landcover","ndvi"],
 "outcomes":{"dem":"ok","landcover":"ok","ndvi":"ok","rainfall":"not_requested"},
 "duration_ms":{"dem":3591,"landcover":3591,"ndvi":24010},"total_ms":33180,
 "aoi_area_km2_band":"10-100","sensor":"landsat","client_prefix":"203.0.113.0/24"}
```

### The client address needed a trust decision

`request.client.host` is the *proxy*, because Compose binds the API to loopback
and only Nginx can reach it — so the obvious implementation logs Nginx's address
and is useless for rate limiting or spotting scraping. The forwarded chain is read,
but **only when the peer can only be our own Nginx**, so a client cannot forge it.
Proxy ranges are enumerated explicitly rather than taken from
`ipaddress.is_private`, which also reports the documentation ranges as private and
would make `203.0.113.9` look like a proxy. This is safe only while the port stays
loopback-bound, and the docstring says so.

### The cache is a data store about specific land

The rainfall cache is keyed by a hash of the submitted geometry and holds that
area's monthly series, so it is a record about a particular place even though the
polygon is never stored. A request to remove an area's data could not be honoured.
`POST /admin/rainfall/forget` takes a geometry or a 32-character key, deletes only
that series, and is idempotent. It refuses anything that is not an exact
lowercase-hex key, refuses a path outside the cache, and leaves the shared
per-read cell cache alone.

### Tests: 168 -> 211

43 on the schema and the endpoint: band edges against the shipped caps, IPv4 and
IPv6 truncation, verdicts that drop their measurements, the absent-geometry
parameter, a record that carries no geometry, append-as-JSON-lines, a write
failure that cannot raise, aggregation with no area values recoverable, deletion
by geometry and by key, idempotency, a sibling area surviving, seven malformed
keys refused, non-polygon geometry refused, forwarded-header trust including the
forgery case, and the timer.

### Four bugs found by writing the tests

1. **The endpoint called `_emit_usage_event` but the function did not exist** —
   every successful request would have returned 500. My own test missed it because
   it only exercised the rejection path, which returns before the emit site. A
   string-anchored replacement had silently not applied.
2. **The NDVI duration read 14,738,217 ms** — four hours. The timer was
   constructed but never entered, so it measured the interval since the epoch.
3. **The raster timer recorded nothing** for the same reason, so `duration_ms` was
   empty.
4. **`_PRIVATE_PEERS` was a fixed string set** that did not recognise the Docker
   bridge range, so the forwarded branch never ran and every prefix came out
   `unknown`.

Two of these were string-anchored replacements that silently failed to apply. I
now assert the replacement landed.

---

## 1.8.0 — Sensor ladder: Landsat 30 m and MODIS 250 m, chosen by measurement

One code path now serves three sensors, selected automatically, and the selection
is driven by a cross-sensor measurement rather than by resolution alone.

### The cross-sensor measurement that changed the default

The same 60 km2 study area, the same 90-day window, three sources:

| Sensor | Mask | NDVI mean |
| --- | --- | --- |
| MODIS MOD13Q1 | NASA, producer-masked | **+0.418** |
| Landsat C2 L2 | QA_PIXEL bits | **+0.357** |
| Sentinel-2 L2A | SCL, relaxed | **+0.161** |

MODIS and Landsat agree to within 0.06. Sentinel-2 read **0.20 low**, because
its SCL classes 4 and 5 are deliberately kept — that relaxation is what stops
bright desert being flagged as cloud, but in cloudy conditions it lets cloud into
the composite and cloud drags the median down.

So `auto` now prefers **Landsat** despite being coarser. At 30 m a 100 km2 area
gains nothing from 10 m, and the QA_PIXEL mask can be trusted. Sentinel-2 still
wins for the last few days, where Landsat has no scene yet, and its result is
labelled "relaxed cloud mask" so a low reading is explainable. Both facts live in
the registry as `mask_authoritative`, and the tests assert the direction of the
disagreement rather than a bare number.

### The Landsat offset trap, confirmed rather than assumed

`landsat-c2-l2` publishes `scale=2.75e-05, offset=-0.2` **in the STAC item only** —
the file's own band tags are empty. The offset is in reflectance units and does
**not** cancel in the NDVI ratio: on a real scene it moved the mean from 0.085 to
0.147, a 73% change, while staying comfortably inside [-1, 1]. It is therefore a
constant in the registry, applied before any band arithmetic, with a test that
fails if the ratio and the scaled ratio ever agree.

### MODIS is a product, not a computation

`modis-13Q1-061` ships `250m_16_days_NDVI`: int16, `scale=0.0001`, fill -3000, in
a sinusoidal projection. No band arithmetic and no cloud mask to get wrong, which
makes it both the cheapest and the most defensible source at country scale. Two
details handled: its items carry `datetime: null` and only `start_datetime`, so
ordering falls back to that, and the product is read at its own resolution rather
than resampled from bands.

### A historical window is now expressible

`window_days` is a lookback, so a user could not ask for 1997 — which was the
whole point of adding Landsat. `window_start` and `window_end` accept an explicit
ISO range, and the 1997/98 El Niño question is answered end to end: Landsat, NDVI
0.240 over 1997-01 to 1999-12.

`MAX_WINDOW_DAYS` is 12,000 and explicitly **not** a cost limit: a request takes
`max_scenes` however long the window is, so a decade costs the same as a month. It
catches a mistyped year.

### Surface

- `?sensor=auto|sentinel-2|landsat|modis` and `?window_start=&window_end=`.
- Every vegetation result carries its sensor provenance — collection, native
  resolution, scale, offset, cloud mask, whether NDVI is computed or a product —
  plus the reason the sensor was chosen and the window read.
- `/version` publishes the whole ladder.
- The vegetation card names the sensor and flags a relaxed mask; the text summary
  says "Vegetation (Landsat Collection 2 Level 2)".

### Tests: 128 -> 168

- 33 offline tests: registry completeness, the offset trap, the SCL and QA bit
  masks, the plausibility filter, the full selection-policy table, and window
  resolution including inverted, malformed and over-long ranges.
- 7 real-data cross-sensor tests, including the one that matters: MODIS and
  Landsat must agree within 0.20, which is what catches a mis-scaled sensor. A
  per-sensor "plausible value" test cannot, because a mis-scaled sensor still
  returns something inside [-1, 1].

### Mistakes made

Index-based slicing of `main.py` to generalise a function deleted three helpers
(`_read_window`, `_reproject_to_grid` and the scene reader) that only showed up
as a `NameError` at runtime, then twice more under new names. Restored by
rewriting the block once and verifying by name, rather than by inspecting splices.

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
2. **Self-service polygon submission.** Submit any polygon, queue it for
   precompute, return a permanent shareable page. The rainfall miss state is
   already the hook: it names the cache key and explains why. This is the
   loudest unanswered question in the webinar — "how do we share our polygons",
   asked five times by four people — and `NEXT_FEATURE.md` as written answers it
   with the wrong answer, an administrator-gated portfolio.
2. **Alerting.** A dashboard is visited once; a subscription is visited monthly.
   Diff the precomputed series on refresh and notify. No new infrastructure, and
   it cannot be retrofitted cheaply.

Only with evidence from 1–2: a durable queue and a monitoring view. The store
should be Postgres or DuckDB — a table, not a cube.
