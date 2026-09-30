# Roadmap

Where this is going, in the order it needs to happen, and what is deliberately
not happening yet.

## The shape

The catalogue is four products of radically different valid scale, and the app
presents them as four tiles of one thing. That is the defect underneath most of
the others. So the foundation is four structures, in dependency order:

1. **A product registry.** One declaration per capability — what it measures,
   its native ground-sample distance, the AOI range where the measure is
   *meaningful* (not merely affordable), its latency, evidence class, cost class,
   and the conditions under which it fails with a reason. This retires the
   constants currently scattered across five modules.
2. **`Measure` as the universal result.** Value, units, extent (cell count, ground
   footprint, window), evidence class, observed-through date, uncertainty, source.
   Every product produces a `Measure`; nothing produces a "card". The UI becomes
   dataset-agnostic and a new product gets a card for free.
3. **A question layer above the products.** The geometry is constant; the question
   is the variable. `describe`, `compare`, `history`, `watch`. Each routes to live
   or computed *itself*, from the cost class of the measures it needs.
4. **Generic scale mechanics.** Resolution ladder, pixel budget, admission order,
   freshness-aware routing — one module reading registry metadata, not per-dataset
   constants.

Alongside those: an **area resolver**, because every area currently enters as raw
GeoJSON and the planner persona has no way to say "that ward" without a file.

## Acceptance criteria

A foundation that cannot be shown to generalise is not a foundation. Two tests:

- **Can a new dataset be added without touching the UI?** CHIRPS is the test case:
  one registry entry plus one `Measure` producer. If it needs a card, a route, or
  a client change, the abstraction leaked.
- **Can a new persona be served without a new endpoint?** The four questions are
  the natural set. If the planner needs a fifth route, the question layer is not
  a layer.

## Phase 0 — unblock

Nothing here is foundation, and all of it is small.

- [ ] **Rotate the Spaces credentials — at the end, deliberately.** Held open
      while the build and testing run, then rotated once everything is complete.
      Note: `.env` is already gitignored and untracked, so there is no exposure
      through git; an earlier audit claimed otherwise and was wrong.
- [x] **`/boundaries?ids=` — investigated, not a bug.** It requires `level` to
      match the id and returns a 422 saying so when they disagree. An earlier
      probe parsed a `features` key out of a 422 body, which has `detail`, and
      reported "zero features". The id → geometry round-trip works, so P4 is not
      gated.

## Phase 1 — close what we already know is wrong

Each is a real defect found in review, and none needs the foundation. Doing them
first means the foundation is not built on a layer with holes in it.

- [ ] **Evidence label on every path.** `contract.py` declares it required;
      `/generate-context` is the only place it is emitted. `POST /rainfall`,
      `GET /rainfall` and every worker artefact return payloads with no `evidence`
      key.
- [ ] **`/version` tells the truth.** It publishes
      `"large_area_mode": "not available until a durable asynchronous worker is
      deployed"` — the worker is deployed. It publishes `ndvi_resolution_m: 20`,
      true only on the synchronous path. `DEPLOYMENT.md` says the worker sleeps
      300 s; it deploys at 15 s. A partner reading this file would cite a
      falsehood.
- [ ] **Extent and freshness on the card.** `grid_cells` and the series end date
      exist and are both buried in a collapsed "Show the working". Four products
      currently sit side by side six months apart with nothing saying so.
- [ ] **One reference period.** The vegetation normal is a rolling 20 years; the
      rainfall card says 1991–2020. Both correct, both on screen together, and a
      reader comparing the two panels is comparing against different periods.
- [ ] **Pin the live-network flake.** `test_contract.CaveatTests` hits live
      Planetary Computer search and has failed in about half of full-suite runs
      for several sessions. Six isolated runs pass. A flake is worse than a
      failure because it teaches you to ignore red.

## Phase 2 — the foundation

- [x] **Product registry** — `registry.py`, 20 tests. Five products declared
      once, published through `/version`, and asserted against the code that
      computes the numbers. Three inline timeouts named so they could be
      published. `/version`'s false `large_area_mode` string replaced with the
      real queue depth and lease. Still to retire: the landcover budget defined
      twice, `PIXEL_BUDGET` unreferenced, and the 27.8 literal in
      `indicators.py`.
- [x] **`Measure` envelope** — `measure.py`, 13 tests. A dynamic measure
      cannot be built without `observed_through`, so Phase 1's freshness item is
      now a constructor error rather than a habit. Evidence and caveats are
      inherited from the registry, so a result cannot disagree with its product.
      Proven generically: a synthetic product is registered and rendered in a
      test, which is acceptance test one executed rather than described.
- [x] **Area resolver** — `areas.py`, 18 tests. Admin id, country+level name,
      point and bbox behind one route, all returning the same `Area` shape. Keeps
      `validate_for_lookup` rather than the synchronous admission path, since
      Kenya's admin0 is about 580,000 km² and would be refused outright. Tests run
      on responses recorded from the live service, and the two live tests
      confirmed the recordings are not stale. An unreachable service is reported
      as 503 and a bad name as 404, because those demand different responses.
- [x] **Question layer** — `questions.py`, 27 tests, two routes. Dates are the
      primary input and the label, bin width, comparison normal, sensor coverage
      and routing are all derived from them. Routing is read from the registry,
      not a parallel table. `/questions/plan` returns the plan without doing the
      work, so a client can choose between showing a result and offering the
      offline route cheaply. Not yet: executing a plan still means calling
      `/generate-context`, and `compare` is not implemented.
- [ ] **Generic scale mechanics.** Ladder, pixel budget, admission, freshness
      routing, as one module over registry metadata.

## Phase 3 — the temporal model

- [ ] **Dates primary, windows derived.** The user names two dates; the display
      label, bin width (3 months → monthly bars, 40 years → annual), comparison
      normal, alert horizon and sensor validity are all derived from them.
      Presets `1y/3y/10y/30y` cannot express "the 2019/20 drought" or "since the
      last rains", which are the ranger's and NRT's questions.

## Phase 4 — connect

- [ ] **globe → geocontextualize handoff.** A selected ADM1/ADM2 becomes a study
      area by id, with no file transfer. Not gated — see Phase 0.
- [ ] **Reverse.** Offer to snap a drawn area to its containing admin unit, via
      `/containing`.

## Deferred by decision

Named, researched, and deliberately not built until the registry makes them a
one-entry change. Recorded so the work is not rediscovered.

**Data.** CHIRPS (5 months fresher than our ERA5, 0.05° against 0.25°). IMERG
(~4 h latency, the freshness answer for three personas). ECMWF forecast (the
only thing that answers "when do I act"). TerraClimate (published SPEI, soil
moisture and PET — a leading signal, and a real drought index to replace our
hand-rolled anomaly). Sentinel-1 SAR (sees through cloud, which is the weakness
our own README admits in the NDVI path). DEA `protected_areas` + MPC `gbif` (the
EIA *scoping* job — not a biodiversity baseline, which is fieldwork and which we
should keep refusing to imply). DEA WorldCereal. DEA JRC Global Surface Water.
NOAA climate normals, so we stop hand-rolling our own. `nested-eagle-global`
(annual 10 m land cover back to 1972, the researcher's ask).

**Product.** Comparison across areas. Saved areas and a landscape view — the
planner, NRT and ranger personas all think in landscapes, and we are
one-polygon-at-a-time with no list. A 3-month alert horizon alongside the 12-month
one; the code already computes `trailing_3m_anomaly_pct` and nothing uses it. A
3–6 month window picker, deferred behind dates-primary.

## Known and not yet scheduled

- The `MODIS_MIN_AREA_KM2` ladder step is unreachable on the synchronous path
  because `select_sensor` runs after the area cap rejects everything past 100 km².
  The ladder is only reachable via the async path or by naming the sensor.
- The landcover error path offers the user "a coarser resolution" offline, and the
  async path still reads native 10 m. The offered remedy is not what the code
  does.
- The code documents DEM as flat to 5,500 km² and then refuses it at 100 km².
- `area_exceeded` is a declared status that nothing emits; the real path returns
  `status: "error"` with a different code.
