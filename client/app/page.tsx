"use client";

import { useState, useRef, useCallback, useEffect, type ReactNode } from 'react';
import dynamic from 'next/dynamic';
import {
  Search, MapPin, Loader2, Globe, Satellite, ArrowDown, Download,
  Link2, Printer, Info, Upload,
} from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Label } from '@/components/ui/label';
import { FeatureCollection, Geometry, GeoJsonObject } from "geojson";
import CopySummary from '@/components/Copy';
import TimeSection, { type RainPoint, type VegPoint } from '@/components/TimeSection';
import { Wordmark } from '@/components/Mark';
import { useToast } from '@/hooks/use-toast';


// Dynamic imports to avoid SSR issues with Leaflet
const MapComponent = dynamic(() => import('@/components/MapComponent'), {
  ssr: false,
  loading: () => (
    <div className="flex h-[600px] items-center justify-center bg-void">
      <div className="flex items-center gap-2.5">
        <Globe className="h-5 w-5 animate-spin text-signal" />
        <span className="label">Loading satellite basemap</span>
      </div>
    </div>
  )
});

/* The four public datasets this reads, named in the open.

   This is the credibility move, and it is placed above the headline on purpose:
   a land manager or an EIA reviewer can look up every one of these and find what
   they claim to be. No adjective a marketing department could invent would make
   the same argument, and putting the evidence before the claim is also the more
   confident order — it says the product does not need the claim to be believed. */
const SOURCES = [
  { id: 'nasadem', label: 'NASADEM', role: 'Elevation · 30 m' },
  { id: 'worldcover', label: 'ESA WorldCover', role: 'Land cover · 10 m' },
  { id: 'sentinel-2', label: 'Sentinel-2', role: 'Vegetation · 20 m' },
  { id: 'era5', label: 'ERA5', role: 'Precipitation · monthly' },
] as const;

interface SearchResult {
  place_id: number;
  display_name: string;
  lat: string;
  lon: string;
  boundingbox: [string, string, string, string];
}

interface BoundingBox {
  north: number;
  south: number;
  east: number;
  west: number;
}

interface AnalysisWarning {
  message: string;
  status?: string;
}

interface SubmissionState {
  state: string;
  reason?: string | null;
  cacheKey?: string;
}

interface WorkPlan {
  indicator?: string;
  resolution_m?: number;
  resolution_km?: number;
  reason?: string;
  months?: number;
  pixels_analysed?: number;
  estimated_seconds?: number;
  estimate_basis?: string;
}

interface OfflineOffer {
  areaKm2: number;
  limitKm2: number;
  plans: Array<WorkPlan & { already_computed?: boolean }>;
  queued: string[];
}

interface ClimateVegSeries {
  status: string;
  source?: string | null;
  resolution_km?: number | null;
  months?: number | null;
  thin_months?: string[];
  caveat?: string | null;
  climatology?: { standard?: string | null; start?: string | null; end?: string | null } | null;
  series?: Array<{
    month: string; value: number; min?: number; max?: number;
    normal?: number; anomaly?: number; anomaly_pct?: number | null;
  }>;
}

interface JobProgress {
  state: string;
  indicator: string;
  submittedAt?: string | null;
  startedAt?: string | null;
  finishedAt?: string | null;
  attempts?: number;
  reason?: string | null;
  months?: number | null;
}

type DatasetId = 'dem' | 'landcover' | 'ndvi' | 'rainfall';

// Pasted above roughly this length and chat clients, email servers and proxies
// start truncating, so the link stops being a link.
const SHARE_URL_LIMIT = 8000;

const DATASET_IDS: readonly DatasetId[] = ['dem', 'landcover', 'ndvi', 'rainfall'];

const DATASET_OPTIONS: Array<{
  id: DatasetId;
  label: string;
  description: string;
}> = [
  { id: 'dem', label: 'Elevation & terrain', description: 'Elevation range and terrain variation' },
  { id: 'landcover', label: 'Land cover', description: 'ESA WorldCover composition' },
  { id: 'ndvi', label: 'Vegetation (NDVI)', description: 'Recent vegetation condition' },
  { id: 'rainfall', label: 'Rainfall & drought', description: 'ERA5 monthly totals and anomaly vs normal' },
];

const LANDCOVER_LABELS: Record<string, string> = {
  "10": "Tree cover",
  "20": "Shrubland",
  "30": "Grassland",
  "40": "Cropland",
  "50": "Built-up areas",
  "60": "Bare or sparse vegetation",
  "70": "Snow & ice",
  "80": "Permanent water bodies",
  "90": "Herbaceous wetlands",
  "95": "Mangroves",
  "100": "Moss & lichen",
};

// These mirror the declared contract in contract.py. Generate them from
// /openapi.json rather than keeping a hand-written copy: a duplicated literal
// that drifted once already and shipped a stale limit to production.
type EvidenceStatus = 'observed' | 'derived' | 'modelled' | 'unconfirmed';
type ModuleStatus =
  | 'ok' | 'not_requested' | 'not_computed' | 'area_exceeded'
  | 'no_valid_pixels' | 'unavailable' | 'skipped' | 'error' | 'busy';

interface ModuleEvidence {
  status: EvidenceStatus;
  source?: string | null;
  method?: string | null;
  note?: string | null;
  doi?: string | null;
  license?: string | null;
  retrieved?: string | null;
}

interface DemStats {
  status: ModuleStatus;
  evidence?: ModuleEvidence;
  valid_pixel_count?: number;
  valid_pixel_fraction?: number;
  mean?: number;
  min?: number;
  max?: number;
  std?: number;
  elevation_range_m?: number;
  terrain_type?: string;
  error?: string;
}

interface SensorProvenance {
  id: string;
  label: string;
  collection: string;
  native_resolution_m: number;
  archive_start: string;
  ndvi: string;
  cloud_mask: string;
  scale?: number;
  offset?: number;
  mask_authoritative?: boolean;
}

interface NdviStats {
  status: ModuleStatus;
  evidence?: ModuleEvidence;
  mean?: number;
  min?: number;
  max?: number;
  std?: number;
  p25?: number;
  p75?: number;
  scene_count?: number;
  resolution_m?: number;
  method?: string;
  warning?: string;
  source?: string;
  valid_pixel_count?: number;
  valid_pixel_fraction?: number;
  scenes_examined?: number;
  sensor?: SensorProvenance;
  sensor_reason?: string;
  window?: { start?: string; end?: string };
  scene_ids?: string[];
  scene_dates?: string[];
}

interface LandcoverStats {
  status: ModuleStatus;
  evidence?: ModuleEvidence;
  classes?: Record<string, number>;
  dominant_class?: string;
  dominant_percentage?: number;
  valid_pixel_count?: number;
  // Declared explicitly: the index signature below would otherwise make this
  // `unknown`, and the arithmetic in the working panel would not type-check.
  valid_pixel_fraction?: number;
  error?: string;
  [key: string]: unknown;
}

interface RainfallSummary {
  latest_month?: string;
  latest_precip_mm?: number;
  latest_anomaly_pct?: number | null;
  driest_month?: { month: string; precip_mm: number };
  wettest_month?: { month: string; precip_mm: number };
  trailing_12m?: {
    ending: string; months: number; precip_mm: number;
    normal_mm: number; anomaly_mm: number; anomaly_pct: number | null;
  } | null;
  suspect_months?: string[];
}

interface RainfallClimatology {
  standard?: string | null;
  start?: string | null;
  end?: string | null;
  annual_mean_mm?: number | null;
  monthly_mean_mm?: Record<string, number>;
}

interface RainfallResult {
  status: ModuleStatus;
  evidence?: ModuleEvidence;
  label?: string | null;
  indicator?: string | null;
  source?: string | null;
  doi?: string | null;
  license?: string | null;
  retrieved?: string | null;
  resolution_km?: number | null;
  grid_cells?: number | null;
  processing_version?: string | null;
  window?: { start?: string; end?: string } | null;
  climatology?: RainfallClimatology | null;
  suspect_months?: string[];
  series?: Array<{ month: string; precip_mm: number; normal_mm: number; anomaly_pct: number | null }>;
  summary?: RainfallSummary;
  message?: string;
  reason?: string;
}

interface AnalysisMetadata {
  bbox_area_km2?: number;
  /** The box, so "which unit is this in?" needs no second measurement. */
  bbox?: number[];
  datasets?: string[];
  mode?: string;
  applied_resolution_m?: Record<string, number>;
}

/** One entry from the backend's `summary.measures` list (Measure.describe).
 *  Mirrors contract.py:ContextSummary.measures / measure.py:Measure.describe. */
interface MeasureExtent {
  requested_area_km2?: number | null;
  covered_area_km2?: number | null;
  cell_count?: number | null;
  native_resolution_m?: number | null;
  native_grid_degrees?: number | null;
  over_specified?: boolean;
  window?: { start?: string | null; end?: string | null };
}

interface MeasureSummary {
  product: string;
  measure: string;
  label: string;
  value: unknown;
  units: string;
  evidence: string;
  extent?: MeasureExtent;
  observed_through?: string | null;
  computed_at: string;
  source?: string | null;
  doi?: string | null;
  caveats?: string[];
  uncertainty?: string | null;
  detail?: Record<string, unknown>;
}

interface Summary {
  dem?: DemStats | null;
  ndvi?: NdviStats | null;
  landcover?: LandcoverStats | null;
  rainfall?: RainfallResult | null;
  country?: string | null;
  analysis?: AnalysisMetadata;
  /** Uniform measure envelope shipped as of 1.21.0 (contract.py). */
  measures?: MeasureSummary[];
  caveats?: string[];
}

function isDatasetId(value: string): value is DatasetId {
  return DATASET_OPTIONS.some((dataset) => dataset.id === value);
}

function formatNumber(value: number | null | undefined, maximumFractionDigits = 1): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—';
  return new Intl.NumberFormat(undefined, { maximumFractionDigits }).format(value);
}

function landcoverEntries(landcover: LandcoverStats): Array<[string, number]> {
  const classes = landcover.classes && typeof landcover.classes === 'object'
    ? landcover.classes
    : Object.fromEntries(
        Object.entries(landcover).filter(
          ([code, value]) => /^\d+$/.test(code) && typeof value === 'number',
        ),
      ) as Record<string, number>;

  return Object.entries(classes)
    .filter(([, value]) => typeof value === 'number' && Number.isFinite(value))
    .sort(([, first], [, second]) => second - first);
}

// Least-squares slope of the series per year. A mean tells you where a place is;
// a trend tells you which way it is going, and that is the question a manager
// actually asks.
function trendPerYear(series: Array<{ month: string; precip_mm: number }>): number | null {
  if (series.length < 24) return null;
  const first = Date.parse(series[0].month + '-01T00:00:00Z');
  if (Number.isNaN(first)) return null;
  const xs = series.map((r) => (Date.parse(r.month + '-01T00:00:00Z') - first) / (365.25 * 86400000));
  const ys = series.map((r) => r.precip_mm);
  const n = xs.length;
  const meanX = xs.reduce((a, b) => a + b, 0) / n;
  const meanY = ys.reduce((a, b) => a + b, 0) / n;
  let num = 0; let den = 0;
  for (let i = 0; i < n; i += 1) {
    num += (xs[i] - meanX) * (ys[i] - meanY);
    den += (xs[i] - meanX) ** 2;
  }
  return den === 0 ? null : num / den;
}

/**
 * Backend error codes, in words.
 *
 * These used to arrive as the evidence note verbatim, so a user read
 * `landcover_area_exceeded` where a sentence belonged. Translating at the
 * presentation boundary rather than in the API means a code added later cannot
 * leak: anything unrecognised falls through to a sentence that at least says
 * something went wrong and what to try.
 */
const ERROR_COPY: Record<string, string> = {
  elevation_unavailable:
    'No elevation data covers this area. Try a location nearer a land mass, or check the coordinates.',
  no_valid_elevation_pixels:
    'Elevation data exists for this region but not inside the boundary. The outline may be in the sea or on the wrong side of the antimeridian.',
  landcover_unavailable:
    'No land-cover data covers this area.',
  no_valid_landcover_pixels:
    'Land-cover data exists for this region but not inside the boundary.',
  landcover_area_exceeded:
    // Promised a coarser offline read that the async path does not implement:
    // both the DEM and the land-cover readers take their tiles at native
    // resolution, so a large area queued for them still reads 10 m. Saying so is
    // better than offering a remedy that does not exist.
    'Land cover is read at 10 m and a large area still is, so drawing a smaller boundary is the only way to get it. Elevation and vegetation answer large areas at a coarser resolution.',
};

/** An untranslated code is a defect, so never render one. */
function readableNote(note?: string | null): string | null {
  if (!note) return null;
  const known = ERROR_COPY[note];
  if (known) return known;
  if (/^[a-z0-9]+(_[a-z0-9]+)+$/.test(note)) {
    return 'This source could not be read for the area. Try a smaller or differently placed boundary.';
  }
  return note;
}

function EvidenceLine({
  evidence,
  status,
}: {
  evidence?: ModuleEvidence;
  status: ModuleStatus;
}) {
  if (!evidence && status === 'ok') return null;
  const kind = evidence?.status;
  return (
    <p className="fig mt-2 flex flex-wrap items-baseline gap-x-2 gap-y-1 text-xs">
      {kind && <span className={`status status-${kind}`}>{kind}</span>}
      {evidence?.source && <span className="text-ink-2">{evidence.source}</span>}
      {readableNote(evidence?.note) && (
        <span className="text-ink-2">{readableNote(evidence?.note)}</span>
      )}
      {status !== 'ok' && <span className="text-ink-3">{STATUS_COPY[status]}</span>}
    </p>
  );
}

const STATUS_COPY: Record<ModuleStatus, string> = {
  ok: 'reported',
  not_requested: 'not requested',
  not_computed: 'not computed',
  area_exceeded: 'area too large',
  no_valid_pixels: 'no valid pixels',
  unavailable: 'unavailable',
  skipped: 'skipped',
  error: 'error',
  busy: 'busy',
};

/** The sentence a rainfall figure needs when the grid is coarser than the outline.
 *
 * An ERA5 "area mean" over a small paddock is one 28 km cell, which says the same
 * thing about 10 km2 and about 600 km2. Saying so is cheap; leaving it in a
 * collapsed disclosure is how a reader ends up trusting four square kilometres of
 * precision that is not there.
 */
function overSpecified(rain: {
  grid_cells?: number | null;
  resolution_km?: number | null;
  bbox_area_km2?: number | null;
  label?: string | null;
}): string | null {
  const cell = rain.resolution_km ?? 27.8;
  const cellKm2 = cell * cell;
  // The strongest case for this warning is exactly one cell standing for a small
  // outline, and the first version of this function suppressed it by requiring
  // two or more cells. Found by driving the page, not by reading it.
  if (rain.bbox_area_km2 == null) {
    return rain.grid_cells === 1
      ? `One grid cell of about ${Math.round(cellKm2 / 100) * 100} km² covers this outline, so the figure is the cell's and not the outline's.`
      : null;
  }
  if (rain.bbox_area_km2 >= cellKm2 * 0.7) return null;
  return `This outline is about ${Math.round(rain.bbox_area_km2).toLocaleString()} km² and one grid cell is about ${Math.round(cellKm2 / 100) * 100} km², so the figure is the cell's rather than the outline's.`;
}

function stamp(value?: string | null): string | null {
  return value ? value.replace('T', ' ').slice(0, 19) : null;
}

/* Show your working, one section at a time. Everything here was already computed
   and discarded: grid cells, pixels, scenes examined, valid-pixel fraction, and
   the resolution actually read against the one requested. */
function Working({
  rows,
}: {
  rows?: Array<[string, string | number | null | undefined]>;
}) {
  const present = (rows || []).filter(
    ([, value]) => value !== null && value !== undefined && value !== '',
  );
  if (!present.length) return null;
  return (
    <details className="working">
      <summary>Show the working</summary>
      <dl className="working-grid">
        {present.map(([term, value]) => (
          <div key={term} style={{ display: 'contents' }}>
            <dt>{term}</dt>
            <dd>{value}</dd>
          </div>
        ))}
      </dl>
    </details>
  );
}

/* A wait, described. The stages the worker actually goes through, and the
   conditions it will run under, instead of a spinner over a pending. */
function minutes(seconds?: number | null): string {
  if (seconds == null) return 'under a minute';
  if (seconds < 90) return `about ${Math.round(seconds / 15) * 15} seconds`;
  return `about ${Math.round(seconds / 60)} minutes`;
}

function JobProgressLine({
  progress,
  planned,
  onDismiss,
}: {
  progress: JobProgress;
  planned?: WorkPlan | null;
  onDismiss: () => void;
}) {
  const stage: Record<string, string> = {
    pending: 'Queued, waiting for a worker.',
    running: 'Being computed now.',
    ready: 'Done.',
    failed: 'Could not be computed.',
  };
  const active = progress.state === 'pending' || progress.state === 'running';
  return (
    <div className="rule-t mt-4 pt-3">
      <div className="flex items-baseline justify-between gap-3">
        <p className="fig text-sm">
          <span className={`status ${active ? 'status-modelled' : 'status-observed'}`}>
            {progress.state}
          </span>{' '}
          <span className="text-ink">{stage[progress.state] || progress.state}</span>{' '}
          <span className="text-ink-2">{progress.indicator}</span>
        </p>
        {!active && (
          <button type="button" onClick={onDismiss} className="label hover:text-ink">
            dismiss
          </button>
        )}
      </div>
      {planned?.estimated_seconds != null && (
        <p className="fig mt-1 text-sm text-ink">
          Ready in {minutes(planned.estimated_seconds)}.
          {planned.months ? ` ${planned.months} monthly reads` : null}
          {planned.resolution_m ? ` at ${planned.resolution_m} m` : null}
          {planned.resolution_km ? ` on a ${planned.resolution_km} km grid` : null}.
        </p>
      )}
      {planned?.reason && <p className="fig mt-1 text-xs text-ink-2">{planned.reason}</p>}
      <Working
        rows={[
          ['resolution', planned?.resolution_m ? `${planned.resolution_m} m` : null],
          ['pixels', planned?.pixels_analysed?.toLocaleString() ?? null],
          ['submitted', stamp(progress.submittedAt)],
          ['started', stamp(progress.startedAt)],
          ['finished', stamp(progress.finishedAt)],
          ['attempts', progress.attempts ?? null],
          ['months', progress.months ?? null],
          ['reason', progress.reason],
        ]}
      />
    </div>
  );
}



/* One module, one section. Divided by a rule rather than enclosed in a card,
   because a card grid reads as a dashboard and this is a survey document.

   Every module takes a top rule, including the first. Suppressing it on the
   first child is right for a single column and wrong for the two-column grid
   this becomes on a wide screen, where it leaves a rule hanging above the
   right-hand module and none above its neighbour. */
function ModuleBlock({
  title,
  description,
  children,
}: {
  title: string;
  description: string;
  children?: ReactNode;
}) {
  return (
    <section className="module rule-t py-7">
      <div className="flex flex-wrap items-baseline justify-between gap-x-4 gap-y-1">
        <h3 className="headline text-[1.125rem]">{title}</h3>
        <p className="label">{description}</p>
      </div>
      <div className="mt-4">{children}</div>
    </section>
  );
}

/* The number the reader came for.

   The old version set this at 24px next to a 15px caption, which is a
   comfortable size for a document and a completely forgettable one for a tool.
   A land manager opening a result is scanning for a magnitude, and magnitude is
   a size relationship, not a value. So the figure is now the largest thing in
   its section, set in the same monospace as every other figure on the page, and
   the caption sits under it rather than beside it. Nothing about the number
   changed; everything about how loudly it arrives did. */
function Headline({ value, caption }: { value: string; caption: string }) {
  return (
    <div className="fig mb-5">
      <p className="text-[clamp(2.25rem,5.5vw,3.5rem)] font-medium leading-[0.9] tracking-[-0.03em] text-signal">
        {value}
      </p>
      <p className="label mt-2.5">{caption}</p>
    </div>
  );
}

function Figure({ term, value }: { term: string; value: string }) {
  return (
    <div className="hoverline flex items-baseline justify-between gap-4 px-2 -mx-2 py-2">
      <dt className="text-sm text-ink-2">{term}</dt>
      <dd className="fig text-sm text-ink">{value}</dd>
    </div>
  );
}

/** Renders one entry of the summary.measures envelope.
 *  Mirrors the backend's honest-read framing: the value, its ground
 *  (extent / over-specified), and its evidence class are inseparable. */
function MeasureCard({ measure }: { measure: MeasureSummary }) {
  const freshness = measure.observed_through
    ? measure.observed_through.slice(0, 10)
    : null;
  const overSpecified = measure.extent?.over_specified && measure.extent?.covered_area_km2;
  return (
    <div className="rounded-xl border border-rule p-3">
      <div className="flex items-baseline justify-between gap-3">
        <dt className="text-sm font-medium text-ink-2">{measure.label}</dt>
        {measure.units ? (
          <dd className="fig text-sm text-ink">
            {measure.value != null ? String(measure.value) : '—'} {measure.units}
          </dd>
        ) : (
          <dd className="fig text-sm text-ink">
            {measure.value != null ? String(measure.value) : '—'}
          </dd>
        )}
      </div>
      <p className="mt-1 text-[0.6875rem] text-ink-3">
        {measure.evidence}
        {freshness ? ` · as of ${freshness}` : ''}
      </p>
      {overSpecified && (
        <p className="mt-1 text-[0.6875rem] text-ink-3">
          read over {formatNumber(measure.extent!.covered_area_km2, 0)} km² for a{' '}
          {measure.extent!.requested_area_km2
            ? `${formatNumber(measure.extent!.requested_area_km2, 0)} km²`
            : '? km²'} outline
        </p>
      )}
      {measure.caveats && measure.caveats.length > 0 && (
        <ul className="mt-1 space-y-0.5 pl-3 text-[0.6875rem] text-ink-3 marker:text-stressed">
          {measure.caveats.map((c, i) => (
            <li key={i} className="list-disc">{c}</li>
          ))}
        </ul>
      )}
    </div>
  );
}

function DatasetResultCard({
  dataset,
  summary,
  submission,
  onSubmit,
  vegetationSeries,
}: {
  dataset: DatasetId;
  summary: Summary;
  submission?: SubmissionState;
  onSubmit?: () => void;
  vegetationSeries?: ClimateVegSeries | null;
}) {
  const option = DATASET_OPTIONS.find((item) => item.id === dataset);
  if (!option) return null;

  if (dataset === 'dem') {
    const dem = summary.dem;
    if (!dem || dem.status !== 'ok') {
      return (
        <ModuleBlock title={option.label} description={option.description}>
          <EvidenceLine evidence={dem?.evidence} status={dem?.status || 'not_requested'} />
        </ModuleBlock>
      );
    }
    const demCoverage: number | null = dem.valid_pixel_fraction ?? null;
    return (
      <ModuleBlock title={option.label} description={option.description}>
        <Headline value={`${formatNumber(dem.elevation_range_m ?? 0, 0)} m`} caption="elevation range" />
        <dl className="rows">
          <Figure term="mean elevation" value={`${formatNumber(dem.mean, 0)} m`} />
          <Figure term="lowest" value={dem.min == null ? '—' : `${formatNumber(dem.min, 0)} m`} />
          <Figure term="highest" value={dem.max == null ? '—' : `${formatNumber(dem.max, 0)} m`} />
          <Figure term="variation" value={dem.std == null ? '—' : `σ ${formatNumber(dem.std, 0)} m`} />
          <Figure term="terrain" value={dem.terrain_type || '—'} />
        </dl>
        <EvidenceLine evidence={dem.evidence} status={dem.status} />
        <Working
          rows={[
            ['valid pixels', dem.valid_pixel_count?.toLocaleString()],
            ['coverage', demCoverage == null ? null : `${formatNumber(demCoverage * 100, 1)}%`],
            ['source product', 'NASADEM 30 m'],
          ]}
        />
      </ModuleBlock>
    );
  }

  if (dataset === 'landcover') {
    const landcover = summary.landcover;
    const entries: Array<[string, number]> = landcover
      ? (Object.entries(landcover.classes || {}) as Array<[string, number]>)
      : [];
    if (!landcover || landcover.status !== 'ok' || entries.length === 0) {
      return (
        <ModuleBlock title={option.label} description={option.description}>
          <EvidenceLine evidence={landcover?.evidence} status={landcover?.status || 'not_requested'} />
        </ModuleBlock>
      );
    }
    const coverage: number | null = landcover.valid_pixel_fraction ?? null;
    const dominant = entries.reduce((a, b) => (a[1] >= b[1] ? a : b)) as [string, number];
    return (
      <ModuleBlock title={option.label} description={option.description}>
        <Headline
          value={`${formatNumber(dominant[1], 1)}%`}
          caption={LANDCOVER_LABELS[dominant[0]] || dominant[0]}
        />
        <dl className="rows">
          {entries.map(([code, percentage]) => (
            <Figure key={code} term={LANDCOVER_LABELS[code] || code} value={`${formatNumber(percentage, 1)}%`} />
          ))}
        </dl>
        <EvidenceLine evidence={landcover.evidence} status={landcover.status} />
        <Working
          rows={[
            ['classes', entries.length],
            ['valid pixels', landcover.valid_pixel_count?.toLocaleString()],
            ['coverage', coverage == null ? null : `${formatNumber(coverage * 100, 1)}%`],
            ['source product', 'ESA WorldCover 10 m'],
          ]}
        />
      </ModuleBlock>
    );
  }

  if (dataset === 'rainfall') {
    const rain = summary.rainfall;
    if (!rain || rain.status !== 'ok') {
      const state = submission?.state;
      const pending = state === 'pending' || state === 'running';
      const rejected = state === 'rejected';
      return (
        <ModuleBlock title={option.label} description={option.description}>
          <p className="fig text-sm text-ink-2">
            {pending
              ? 'Queued. Precipitation is processed offline, usually within a few minutes.'
              : rejected
                ? submission?.reason || 'The submission queue is full. Try again shortly.'
                : 'No series has been processed for this exact boundary yet.'}
          </p>
          {!pending && onSubmit && (
            <button
              type="button"
              onClick={onSubmit}
                        className="btn btn-primary mt-4 h-10 px-4 text-[0.8125rem]"
            >
              {rejected ? 'Try again' : 'Process precipitation for this area'}
            </button>
          )}
          <EvidenceLine evidence={rain?.evidence} status={rain?.status || 'not_computed'} />
        </ModuleBlock>
      );
    }
    const s = rain.summary || {};
    const t = s.trailing_12m;
    const trend = trendPerYear(rain.series || []);
    const suspect = rain.suspect_months || [];
    return (
      <ModuleBlock title={option.label} description={option.description}>
        <Headline
          value={
            t?.anomaly_pct == null
              ? '—'
              : `${t.anomaly_pct > 0 ? '+' : ''}${formatNumber(t.anomaly_pct, 0)}%`
          }
          caption="last 12 months vs 1991–2020 normal"
        />
        <dl className="rows">
          <Figure term="last 12 months" value={t ? `${formatNumber(t.precip_mm, 0)} mm` : '—'} />
          <Figure
            term="annual normal"
            value={
              rain.climatology?.annual_mean_mm == null
                ? '—'
                : `${formatNumber(rain.climatology.annual_mean_mm, 0)} mm`
            }
          />
          <Figure
            // The slope is fitted over the whole record, not the chosen window.
            // Labelling it "trend per year" next to a 1y window button invites
            // the reader to attribute sixteen years of slope to their year.
            term={`trend per year (${(rain.series || []).length} months)`}
            value={trend == null ? '—' : `${trend > 0 ? '+' : ''}${formatNumber(trend, 1)} mm`}
          />
          <Figure
            term="driest month"
            value={s.driest_month ? `${s.driest_month.month} · ${formatNumber(s.driest_month.precip_mm, 0)} mm` : '—'}
          />
          <Figure
            term="wettest month"
            value={s.wettest_month ? `${s.wettest_month.month} · ${formatNumber(s.wettest_month.precip_mm, 0)} mm` : '—'}
          />
        </dl>
        {suspect.length > 0 && (
          <p className="fig mt-2 text-xs text-caution">
            {suspect.length} month{suspect.length === 1 ? '' : 's'} reported near-zero totals
            and are worth review: {suspect.slice(0, 3).join(', ')}
            {suspect.length > 3 ? ' …' : ''}
          </p>
        )}
        {/* Extent and freshness, on the card rather than inside a disclosure.
            Both were already computed and then filed under "show your working",
            which is where a number goes to stop being read. Four products sit in
            one row and can be six months apart; a reader deserves to see that
            without opening anything. */}
        <p className="fig mt-2 text-xs text-ink-2">
          {rain.grid_cells != null
            ? `${rain.grid_cells} ERA5 grid cell${rain.grid_cells === 1 ? '' : 's'} of ${rain.resolution_km ?? 27.8} km`
            : `ERA5 grid of ${rain.resolution_km ?? 27.8} km`}
          {(() => {
            // The month the data actually stops, not the end of the window that
            // was asked for. Reading window.end claimed a series ran to today
            // when the newest ERA5 month is six months behind -- which is the
            // exact confusion this line exists to remove.
            const last = (rain.series || []).at(-1)?.month;
            return last ? ` · series ends ${last.slice(0, 7)}` : '';
          })()}
          {rain.label ? ` · ${rain.label}` : ''}
        </p>
        {overSpecified(rain) && (
          <p className="fig mt-1 text-xs text-caution">
            {overSpecified(rain)}
          </p>
        )}
        <EvidenceLine evidence={rain.evidence} status={rain.status} />
        {/* The charts used to live here, inside a module, inside a two-column
            grid — which left each plot about 480x150 rendered pixels with 9px
            type. They now live in their own full-width section under the
            dossier, where each one is a chart rather than a caption. */}
        {vegetationSeries && vegetationSeries.thin_months?.length ? (
          <p className="fig mt-1 text-xs text-caution">
            {vegetationSeries.thin_months.length} vegetation month(s) had thin
            coverage and may not reflect a real change.
          </p>
        ) : null}
        <Working
          rows={[
            ['area', rain.label],
            ['grid', rain.resolution_km == null ? null : `${rain.resolution_km} km`],
            ['grid cells', rain.grid_cells],
            ['months', (rain.series || []).length],
            ['climatology', rain.climatology?.standard],
            ['doi', rain.evidence?.doi],
            ['licence', rain.evidence?.license],
            ['retrieved', rain.evidence?.retrieved],
            ['series ends', rain.window?.end],
          ]}
        />
      </ModuleBlock>
    );
  }

  const ndvi = summary.ndvi;
  if (!ndvi || ndvi.status !== 'ok') {
    return (
      <ModuleBlock title={option.label} description={option.description}>
        <EvidenceLine evidence={ndvi?.evidence} status={ndvi?.status || 'not_requested'} />
        {ndvi?.warning && <p className="fig mt-2 text-xs text-ink-2">{ndvi.warning}</p>}
      </ModuleBlock>
    );
  }
  const ndviCoverage: number | null = ndvi.valid_pixel_fraction ?? null;
  return (
    <ModuleBlock title={option.label} description={option.description}>
      <Headline value={formatNumber(ndvi.mean ?? 0, 3)} caption="median composite NDVI" />
      <dl className="rows">
        <Figure term="middle 50%" value={`${formatNumber(ndvi.p25 ?? 0, 3)} – ${formatNumber(ndvi.p75 ?? 0, 3)}`} />
        <Figure term="range" value={`${formatNumber(ndvi.min ?? 0, 3)} – ${formatNumber(ndvi.max ?? 0, 3)}`} />
        <Figure term="scenes" value={ndvi.scene_count == null ? '—' : `${ndvi.scene_count}`} />
        <Figure term="grid" value={ndvi.resolution_m == null ? '—' : `${ndvi.resolution_m} m`} />
      </dl>
      {/* A persistently cloudy area reads low. That is documented in the README
          and reachable through the product registry, and a reader of the number
          never sees it -- because the coverage figure describes how much of the
          outline was measured, not how much cloud was in it. A low index over a
          thin composite is the case most likely to be quoted wrongly, so it is
          stated here rather than filed under show-your-working. */}
      {(ndvi.mean ?? 1) < 0.2 && (
        <p className="fig mt-2 text-[0.75rem] leading-relaxed text-caution">
          A figure below about 0.2 is worth reading as uncertain rather than as bare
          ground. Cloud and thin cover depress the index, and the coverage figure
          below describes the outline rather than the cloud in it.
        </p>
      )}
      <EvidenceLine evidence={ndvi.evidence} status={ndvi.status} />
      <Working
        rows={[
          ['source', ndvi.sensor?.label],
          ['collection', ndvi.sensor?.collection],
          ['cloud mask', ndvi.sensor?.cloud_mask],
          ['why this source', ndvi.sensor_reason],
          ['valid pixels', ndvi.valid_pixel_count?.toLocaleString()],
          ['coverage', ndviCoverage == null ? null : `${formatNumber(ndviCoverage * 100, 1)}%`],
          ['scenes examined', ndvi.scenes_examined],
          ['window', ndvi.window?.start && ndvi.window?.end ? `${ndvi.window.start} to ${ndvi.window.end}` : null],
          ['scene dates', (ndvi.scene_dates || []).slice(0, 2).join(', ')],
        ]}
      />
    </ModuleBlock>
  );
}

export default function Home() {
   const { toast } = useToast();
   const [searchQuery, setSearchQuery] = useState('');
   const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
   const [summaryText, setSummaryText] = useState<string>('');
   const [showResults, setShowResults] = useState(false);
   const [selectedLocation, setSelectedLocation] = useState<{lat: number, lng: number} | null>(null);
   const [boundingBox, setBoundingBox] = useState<BoundingBox | null>(null);
   const [isLoading, setIsLoading] = useState(false);
   const [response, setResponse] = useState<string>('');
   const [analysisWarnings, setAnalysisWarnings] = useState<AnalysisWarning[]>([]);
   const [isSearching, setIsSearching] = useState(false);
   const [selectedDatasets, setSelectedDatasets] = useState<DatasetId[]>(['dem', 'landcover', 'ndvi', 'rainfall']);
   const [analysisSummary, setAnalysisSummary] = useState<Summary | null>(null);
  // How far back to ask for. The API already accepts an explicit range; this is
  // the affordance that makes it reachable.
  const [windowYears, setWindowYears] = useState<1 | 3 | 10 | 30>(10);
  // A user-defined range. Empty means "use the preset above".
  const [customStart, setCustomStart] = useState('');
  const [customEnd, setCustomEnd] = useState('');
  const [selectedSensor, setSelectedSensor] = useState('auto');
  // Which precipitation product. Two providers for one measure, differing in
  // resolution, freshness and evidence class -- so it is a choice with a reason,
  // not a preference, and the reader is told what the choice costs.
  const [rainProduct, setRainProduct] = useState<'rainfall' | 'chirps'>('rainfall');
  // The cache key the backend reported for this analysis, so a submission state
  // can be matched to the area that produced it.
  const [activeCacheKey, setActiveCacheKey] = useState<string | null>(null);
   const [drawnFeatures, setDrawnFeatures] = useState<FeatureCollection<Geometry> | null>(null);
  // Read the share link back on arrival. Without this the link was write-only:
  // it encoded the area, the datasets, the sensor and the window, and nothing
  // ever decoded any of it, so a colleague opened a blank page.
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const adminId = params.get('admin');
    if (adminId) {
      // Resolve the id rather than carrying the geometry, so a link from another
      // system stays small and stays correct if the boundary service updates.
      void (async () => {
        try {
          const response = await fetch(
            `/api/areas/resolve?id=${encodeURIComponent(adminId)}&level=${params.get('level') ?? '1'}`);
          if (!response.ok) return;
          const area = await response.json();
          setUploadedGeojson(area.geometry as GeoJsonObject);
          setAdminArea({ id: area.id ?? adminId, name: area.name ?? null, level: area.level ?? null });
        } catch {
          // A stale or unreachable boundary service must not leave a blank page;
          // the drawn and uploaded paths still work.
        }
      })();
    }
    const area = params.get('area');
    if (!area) return;
    try {
      const parsed = JSON.parse(decodeURIComponent(area));
      if (!parsed || typeof parsed !== 'object') return;
      setUploadedGeojson(parsed as GeoJsonObject);
    } catch {
      return;
    }
    const datasets = (params.get('datasets') || '')
      .split(',')
      .map((d) => d.trim())
      .filter((d): d is DatasetId => DATASET_IDS.includes(d as DatasetId));
    if (datasets.length) setSelectedDatasets(datasets);
    const sensor = params.get('sensor');
    if (sensor) setSelectedSensor(sensor);
    // Prefer the dates the link was built with. `years` is still honoured for a
    // link made before the switch, so an older shared link keeps working.
    const start = params.get('window_start');
    const end = params.get('window_end');
    if (start && end) {
      setCustomStart(start);
      setCustomEnd(end);
    } else {
      const years = Number(params.get('years'));
      if (years === 1 || years === 3 || years === 10 || years === 30) {
        setWindowYears(years);
      }
    }
  }, []);

  // State of a request to have an area's precipitation processed. Keyed by the
  // area, so switching areas does not show another area's progress.
  const [submission, setSubmission] = useState<Record<string, SubmissionState>>({});
  // Progress for a queued area. A spinner is a decorated wait; this shows the
  // conditions the work happens under, which we already compute and discard.
  const [progress, setProgress] = useState<JobProgress | null>(null);
  // The per-area monthly vegetation series, fetched alongside the summary.
  const [vegetationSeries, setVegetationSeries] = useState<ClimateVegSeries | null>(null);
  const [plan, setPlan] = useState<WorkPlan | null>(null);
  const [pendingIndicator, setPendingIndicator] = useState<string | null>(null);
  // A study area past the synchronous cap is not an error to read but a route to
  // take: it is answered offline at a coarser resolution, and the user should see
  // what they would get before committing.
  const [offline, setOffline] = useState<OfflineOffer | null>(null);
  // The synchronous cap is a server setting, not a constant. The banner used to
  // hard-code 100 km², which meant a deployment that raised or lowered the cap
  // published a number its own API disagreed with. The plan response carries the
  // real value, so prefer that and fall back to the documented default.
  const [syncLimitKm2, setSyncLimitKm2] = useState<number | null>(null);
   const searchTimeout = useRef<NodeJS.Timeout | null>(null);
   const [uploadedGeojson, setUploadedGeojson] = useState<GeoJsonObject | null>(null);
  // An administrative boundary, named rather than drawn. The planner thinks in
  // wards and sub-counties and does not have the file; before this the only ways
  // in were drawing it or uploading it.
  const [adminArea, setAdminArea] = useState<{ id: string | null; name: string | null; level: number | null } | null>(null);
  const [adminQuery, setAdminQuery] = useState('');
  const [adminBusy, setAdminBusy] = useState(false);
  // "Which administrative unit is this in?", asked once the user has drawn
  // something. Answered for a point at the outline's centroid, and offered
  // rather than automatic: snapping replaces the outline the person drew, and
  // that has to be their choice.
  const [drawnInfo, setDrawnInfo] = useState<{ areaKm2: number; lon: number; lat: number } | null>(null);
  const [containingArea, setContainingArea] = useState<{ name: string; level: number; id: string | null; area_km2: number } | null>(null);
  const [containingBusy, setContainingBusy] = useState(false);


  // Search for places using Nominatim
  const searchPlaces = async (query: string) => {
    if (query.length < 3) {
      setSearchResults([]);
      return;
    }

    setIsSearching(true);
    try {
      const response = await fetch(
        `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}&limit=5&addressdetails=1`
      );
      const data = await response.json();
      setSearchResults(data);
      setShowResults(true);
    } catch (error) {
      console.error('Search error:', error);
      setSearchResults([]);
      setShowResults(false);
      toast({
        title: 'Place search is unavailable',
        description:
          'Could not reach the search service. You can still draw the area on the map, or upload a GeoJSON file.',
      });
    } finally {
      setIsSearching(false);
    }
  };

  /** Ask the boundary service what this outline sits inside, and offer to use it. */
  const findContainingArea = async () => {
    if (!drawnInfo) return;
    setContainingBusy(true);
    try {
      const response = await fetch(
        `/api/areas/resolve?lat=${drawnInfo.lat}&lon=${drawnInfo.lon}`);
      if (!response.ok) {
        const failure = await response.json().catch(() => null);
        throw new Error(failure?.detail || 'no administrative unit contains that point');
      }
      const area = await response.json();
      setContainingArea({ name: area.name ?? 'that area', level: area.level ?? 0,
                          id: area.id ?? null, area_km2: area.area_km2 ?? 0 });
    } catch (error) {
      setContainingArea(null);
      toast({
        title: 'Could not find a containing area',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      });
    } finally {
      setContainingBusy(false);
    }
  };

  /** Adopt the containing unit, replacing the drawn outline with it. */
  const adoptContainingArea = async () => {
    if (!containingArea?.id) return;
    setContainingBusy(true);
    try {
      const response = await fetch(
        `/api/areas/resolve?id=${encodeURIComponent(containingArea.id)}&level=${containingArea.level}`);
      if (!response.ok) throw new Error('could not fetch that boundary');
      const area = await response.json();
      setUploadedGeojson(area.geometry as GeoJsonObject);
      setAdminArea({ id: area.id ?? containingArea.id, name: area.name ?? containingArea.name,
                     level: area.level ?? containingArea.level });
      setContainingArea(null);
      toast({
        title: 'Area replaced',
        description: `Now using ${area.name}. Its outline is the whole administrative unit, which covers ${Math.round(area.area_km2).toLocaleString()} km² rather than the ${Math.round(drawnInfo!.areaKm2).toLocaleString()} km² you drew.`,
      });
    } catch (error) {
      toast({
        title: 'Could not use that area',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      });
    } finally {
      setContainingBusy(false);
    }
  };

  /** Resolve a place name to an administrative boundary and use it as the area. */
  const resolveAdminArea = async () => {
    const query = adminQuery.trim();
    if (!query) return;
    setAdminBusy(true);
    try {
      // Free text. The server translates 'Narok' or 'Kenya, Narok' into the
      // country-plus-area form the boundary service needs.
      const response = await fetch(`/api/areas/resolve?name=${encodeURIComponent(query)}`);
      if (!response.ok) {
        const failure = await response.json().catch(() => null);
        throw new Error(failure?.detail || `no administrative area named ${query}`);
      }
      const area = await response.json();
      setUploadedGeojson(area.geometry as GeoJsonObject);
      setAdminArea({ id: area.id ?? null, name: area.name ?? query, level: area.level ?? null });
      setAdminQuery('');
      // Say what kind of thing this is while the reader is still looking at the
      // control that chose it. A county is about 780 km2 and needs the queue;
      // a sub-county may not. Finding that out from a 100 km2 limit and an
      // "offline" panel is a worse way to learn it than being told.
      const km2 = Math.round(area.area_km2 ?? 0);
      const level = { 0: 'country', 1: 'region or county', 2: 'district' }[area.level as 0 | 1 | 2]
        ?? `level ${area.level ?? '?'}`;
      toast({
        title: `${area.name} — ${km2.toLocaleString()} km²`,
        description: km2 > (syncLimitKm2 ?? 100)
          ? `That is the whole ${level}, too large to read live, so it is read offline at a coarser resolution and you will be told which. The map has moved to it.`
          : `A ${level}, small enough to read live. The map has moved to it.`,
        duration: 9000,
      });
    } catch (error) {
      toast({
        title: 'Could not resolve that area',
        description: error instanceof Error
          ? `${error.message} You can still draw it on the map or upload a file.`
          : 'You can still draw it on the map or upload a file.',
        variant: 'destructive',
      });
    } finally {
      setAdminBusy(false);
    }
  };

  // Handle search input changes with debouncing
  /** What the chosen window means, in the words the person chose it with.
   *
   * A window is more than a pair of dates: a bar every month and a bar every year
   * are different claims, and a sensor that does not reach back to the start is a
   * refusal rather than a thin answer. All three are consequences of the dates
   * rather than choices offered beside them, so they are stated here.
   */
  const windowExplanation = (() => {
    const start = customStart || windowStartISO();
    const end = customEnd || windowEndISO();
    const months = Math.max(1, (Number(end.slice(0, 4)) - Number(start.slice(0, 4))) * 12
      + Number(end.slice(5, 7)) - Number(start.slice(5, 7)) + 1);
    const parts: string[] = [];
    parts.push(customStart && customEnd ? ' — dates you set' : ` — ${windowYears}-year preset`);
    // Landsat is the archive that reaches furthest back; if it cannot answer then
    // nothing can.
    const reaches = new Date('1982-08-22');
    if (new Date(start) < reaches) {
      parts.push(' — no sensor archive reaches back this far, so there will be no vegetation value');
    } else {
      parts.push(months > 120
        ? ` — ${months} months, so the charts bin it yearly; one bar a month would not be readable`
        : ` — ${months} month${months === 1 ? '' : 's'}`);
    }
    return parts.join('');
  })();

  const handleSearchChange = (value: string) => {
    setSearchQuery(value);
    
    if (searchTimeout.current) {
      clearTimeout(searchTimeout.current);
    }

    searchTimeout.current = setTimeout(() => {
      searchPlaces(value);
    }, 300);
  };

  // Handle location selection
  const handleLocationSelect = (result: SearchResult) => {
    const lat = parseFloat(result.lat);
    const lng = parseFloat(result.lon);
    
    setSelectedLocation({ lat, lng });
    setSearchQuery(result.display_name);
    setShowResults(false);
    setSearchResults([]);
  };

  // Handle bounding box creation from map
  // Stable identity: the map's draw-control effect depends on these, and a new
  // function on every render tore down and re-registered the control each time.
  // Changing the area invalidates everything downstream of the old one. None of
  // this was cleared: the results pane kept showing the previous area's numbers
  // under a freshly drawn boundary, and the offline panel survived every later
  // analysis until a page reload. Both read as "these are the numbers for this
  // area", which was the one thing they were not.
  const clearAnalysis = useCallback(() => {
    setAnalysisSummary(null);
    setSummaryText('');
    setResponse('');
    setPlan(null);
    setOffline(null);
    setProgress(null);
    setSubmission({});
    setActiveCacheKey(null);
    setVegetationSeries(null);
    setContainingArea(null);
  }, []);

  const handleBoundingBoxCreated = useCallback((bbox: BoundingBox | null) => {
    setBoundingBox(bbox);
    clearAnalysis();
  }, [clearAnalysis]);

  const handleFeaturesChange = useCallback((geojson: FeatureCollection) => {
    setDrawnFeatures(geojson);
    clearAnalysis();
  }, [clearAnalysis]);

  // Handle file upload
  const handleGeojsonUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (file.size > 500_000) {
      toast({
        title: "File is too large",
        description: "GeoJSON uploads are limited to 500 KB. Simplify the geometry and try again.",
        variant: "destructive",
      });
      e.target.value = '';
      return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const parsed = JSON.parse(event.target?.result as string);
        if (!parsed || typeof parsed !== 'object' || !('type' in parsed)) {
          throw new Error('The file is not a GeoJSON object.');
        }
        setUploadedGeojson(parsed as GeoJsonObject);
        toast({
          title: "Success",
          description: "GeoJSON file uploaded successfully.",
        });
      } catch (err) {
        console.error("Invalid GeoJSON file", err);
        toast({
          title: "Error",
          description: "Invalid GeoJSON file. Please check the file format.",
          variant: "destructive",
        });
      }
    };
    reader.readAsText(file);
  };
  // Build a concise text fallback for copying and for the results pane. The
  // structured cards below remain the authoritative presentation of values.
  function summarizeData(summary: Summary): string {
  const lines: string[] = [];

  if (summary.country) {
    lines.push(`Location context: ${summary.country}.`);
  }
  if (summary.analysis?.bbox_area_km2 != null) {
    lines.push(`Analysis bounding box: ${formatNumber(summary.analysis.bbox_area_km2, 2)} km².`);
  }

  const dem = summary.dem;
  if (dem && !dem.error && dem.mean != null) {
    lines.push(
      `Elevation: averages around ${formatNumber(dem.mean, 0)} m (range: ${formatNumber(dem.min, 0)}–${formatNumber(dem.max, 0)} m).`,
    );
    if (dem.terrain_type) {
      lines.push(`Terrain: ${dem.terrain_type}.`);
    }
  }

  const ndvi = summary.ndvi;
  if (ndvi?.mean != null) {
    lines.push(
      `Vegetation: NDVI mean is ${formatNumber(ndvi.mean, 2)} (higher values generally indicate denser vegetation).`,
    );
  }
  if (ndvi?.warning) {
    const prefix = ndvi.status === 'skipped'
      ? 'Vegetation analysis was skipped'
      : 'Vegetation analysis note';
    lines.push(`${prefix}: ${ndvi.warning}.`);
  }

  const landcover = summary.landcover;
  const rain = summary.rainfall;
  if (rain?.status === 'ok' && rain.summary?.trailing_12m) {
    const t = rain.summary.trailing_12m;
    lines.push(
      `Rainfall: ${formatNumber(t.precip_mm, 0)} mm over the last 12 months, ` +
      `${t.anomaly_pct != null ? `${t.anomaly_pct > 0 ? '+' : ''}${formatNumber(t.anomaly_pct, 0)}% ` : ''}` +
      `against the 1991-2020 normal of ${formatNumber(t.normal_mm, 0)} mm.`,
    );
  }
  if (landcover && !landcover.error) {
    const parts = landcoverEntries(landcover)
      .map(([code, percentage]) => `${LANDCOVER_LABELS[code] || code} (${formatNumber(percentage, 1)}%)`);
    if (parts.length > 0) lines.push(`Land cover: ${parts.join(', ')}.`);
  }

  // The measure envelope (summary.measures) carries derived/observed/modelled
  // figures in a uniform shape. Surface each headline so the text summary stays
  // in step with the structured panel below.
  if (summary.measures && summary.measures.length > 0) {
    for (const m of summary.measures) {
      const head = m.observed_through ? `${m.label} ${m.value} ${m.units}, as of ${m.observed_through.slice(0, 10)}` : `${m.label} ${m.value} ${m.units}`;
      lines.push(`${head} — ${m.evidence}.`);
    }
  }

  return lines.join(' ') || 'The selected data sources did not return values for this area.';
}


  // Send request to backend
  function windowEndISO(): string {
    if (customStart && customEnd && customStart <= customEnd) return customEnd;
    const today = new Date();
    return `${today.getFullYear()}-${String(today.getMonth() + 1).padStart(2, '0')}-01`;
  }

  function windowStartISO(): string {
    // A user-defined range wins over the presets. The backend has accepted an
    // explicit range since 1.14.0; only the control was missing.
    if (customStart && customEnd && customStart <= customEnd) return customStart;
    const end = new Date(windowEndISO() + 'T00:00:00');
    const start = new Date(end);
    start.setFullYear(start.getFullYear() - windowYears);
    return `${start.getFullYear()}-${String(start.getMonth() + 1).padStart(2, '0')}-01`;
  }

  // A range the backend would reject, caught before the request rather than as a 422.
  const customRangeValid =
    !customStart || !customEnd || (customStart <= customEnd && customStart <= windowEndISO());

  // Queue any indicator for this area and report when it will be ready.
  //
  // The wait is stated rather than implied: a vegetation series is a few minutes
  // of monthly reads, and a spinner over an unquantified wait is the thing this
  // replaces. The estimate is the backend's measured rate, and it is labelled an
  // estimate everywhere it appears.
  async function submitIndicator(indicator: string) {
    const geojson = currentGeojson();
    if (!geojson) return;
    const backendUrl = (process.env.NEXT_PUBLIC_BACKEND_URL || '/api').replace(/\/$/, '');
    setPendingIndicator(indicator);
    try {
      const query = new URLSearchParams({
        window_start: windowStartISO(),
        window_end: windowEndISO(),
      });
      const response = await fetch(`${backendUrl}/rainfall/submit?${query.toString()}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ geojson, indicator, product: rainProduct }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data?.detail || 'Submission failed');
      }
      setPlan(data.planned || null);
      setProgress({
        state: data.submission?.state || 'pending',
        indicator,
        attempts: 0,
        reason: data.submission?.reason ?? null,
      });
      if (data.submission?.state !== 'ready') {
        void pollSubmission(data.cache_key, indicator);
      } else {
        void loadVegetationSeries(data.cache_key);
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Submission failed';
      setResponse(message);
      setSummaryText(message);
    } finally {
      setPendingIndicator(null);
    }
  }

  // Ask what the worker would do for this area. Read-only, so nothing is queued
  // until the user says so.
  async function loadOfflineOffer(geojson: GeoJsonObject) {
    const backendUrl = (process.env.NEXT_PUBLIC_BACKEND_URL || '/api').replace(/\/$/, '');
    try {
      // The indicator belongs in the body. It used to ride along as a query
      // parameter, where the server ignored it and fell back to planning
      // rainfall alone -- so a user who selected every dataset was offered
      // "process 1 module offline" and only rainfall was ever queued.
      const query = new URLSearchParams({
        window_start: windowStartISO(),
        window_end: windowEndISO(),
      });
      const response = await fetch(`${backendUrl}/rainfall/plan?${query.toString()}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          geojson,
          indicator: selectedDatasets.join(','),
          product: rainProduct,
        }),
      });
      if (!response.ok) return;
      const body = await response.json();
      if (typeof body.analysis.synchronous_limit_km2 === 'number') {
        setSyncLimitKm2(body.analysis.synchronous_limit_km2);
      }
      setOffline({
        areaKm2: body.analysis.bbox_area_km2,
        limitKm2: body.analysis.synchronous_limit_km2,
        plans: body.plans || [],
        queued: [],
      });
    } catch {
      /* the existing error path reports the 413 */
    }
  }

  async function queueOffline() {
    if (!offline) return;
    setOffline({ ...offline, queued: offline.plans.map((p) => p.indicator || '') });
    for (const plan of offline.plans) {
      const indicator = plan.indicator;
      if (!indicator) continue;
      await submitIndicator(indicator);
    }
  }

  // The monthly vegetation series lives in its own artefact, keyed by area, so it
  // is fetched after the summary rather than returned inside it.
  /** Show a finished reading from the artefacts the worker wrote.
   *
   * `/generate-context` refuses anything past the synchronous cap, and it always
   * will -- so re-running it to display a queued area's results returned the same
   * refusal and the page showed an error over four completed modules. This reads
   * what exists, which is the same summary the synchronous path returns, so the
   * interface does not need to know how the area was read.
   */
  async function showComputedResult(cacheKey: string | null) {
    if (!cacheKey) return;
    try {
      const response = await fetch(
        `${(process.env.NEXT_PUBLIC_BACKEND_URL || '/api').replace(/\/$/, '')}/context?cache_key=${cacheKey}`);
      if (!response.ok) {
        const failure = await response.json().catch(() => null);
        throw new Error(failure?.detail || `could not read the stored result (${response.status}).`);
      }
      const data = await response.json() as { summary?: Summary };
      const summary = data.summary || {};
      setAnalysisSummary(summary);
      setAnalysisWarnings([]);
      const result = summarizeData(summary);
      setResponse(result);
      setSummaryText(result);
      setActiveCacheKey(cacheKey);
      void loadVegetationSeries(cacheKey);
    } catch (error) {
      setResponse('Error: ' + (error instanceof Error ? error.message : 'could not read the stored result.'));
      setSummaryText(response);
    }
  }

  async function loadVegetationSeries(cacheKey: string) {
    const backendUrl = (process.env.NEXT_PUBLIC_BACKEND_URL || '/api').replace(/\/$/, '');
    try {
      const response = await fetch(
        `${backendUrl}/rainfall?cache_key=${cacheKey}&indicator=vegetation_series`,
      );
      if (!response.ok) return;
      const body = await response.json();
      const loaded = (body.rainfall as ClimateVegSeries) ?? null;
      setVegetationSeries(loaded);
      if (loaded?.status === 'ok') {
        setProgress(null);
        setPlan(null);
      }
    } catch {
      /* the chart falls back to rainfall alone */
    }
  }

  // `handleAnalyze` is declared below this poller, so reaching for it directly
  // is a temporal-dead-zone error. The ref is assigned on every render, and by
  // the time a poll can run the current function is in it.
  const analyzeRef = useRef<() => Promise<void>>(async () => {});

  // Failures arrive in the same string slot as a successful summary, prefixed
  // "Error: ". Splitting here is what lets the panel style them differently.
  const responseIsError = response.startsWith('Error:');
  const errorHeadline = responseIsError
    ? (response.split('\n')[0].replace(/^Error:\s*/, '') || 'The analysis did not finish')
    : '';
  const errorDetail = responseIsError ? response.split('\n').slice(1).join(' ') : '';

  // Poll a queued area and describe what the worker is actually doing. The job
  // record already holds the timestamps, the indicator and the reason; all that
  // was missing was showing it.
  async function pollSubmission(cacheKey: string, indicator: string) {
    const backendUrl = (process.env.NEXT_PUBLIC_BACKEND_URL || '/api').replace(/\/$/, '');
    for (let attempt = 0; attempt < 600; attempt += 1) {
      let data: { submission?: Record<string, unknown> } = {};
      try {
        const response = await fetch(
          `${backendUrl}/rainfall/status?cache_key=${cacheKey}&indicator=${indicator}&product=${rainProduct}`,
        );
        if (response.ok) data = await response.json();
      } catch {
        /* tolerate a transient failure rather than abandoning the job */
      }
      const submission = (data.submission || {}) as Record<string, unknown>;
      const state = String(submission.state || 'pending');
      setProgress({
        state,
        indicator,
        submittedAt: (submission.submitted_at as string) ?? null,
        startedAt: (submission.started_at as string) ?? null,
        finishedAt: (submission.finished_at as string) ?? null,
        attempts: (submission.attempts as number) ?? 0,
        reason: (submission.reason as string) ?? null,
        months: (submission.months as number) ?? null,
      });
      if (state === 'ready' || state === 'failed' || state === 'not_submitted') {
        if (state === 'ready') {
          setSubmission((previous) => ({ ...previous, [cacheKey]: { state, cacheKey } }));
          // Fetch the finished series. The progress line said "Done" and the
          // card still read "no series has been processed for this exact
          // boundary yet", and the only way to see the result was to notice that
          // and press Analyze again. The work is already done; showing it is the
          // last step, and skipping it is the difference between a queue and a
          // black hole.
          // Show the finished reading. A queued area cannot be re-computed --
          // it was refused synchronously and would be refused again -- so the
          // artefacts the worker just wrote are assembled instead. Before this
          // a large area could be queued, completed, and still shown an error.
          void showComputedResult(activeCacheKey);
        }
        return;
      }
      await new Promise((resolve) => setTimeout(resolve, 3000));
    }
  }

  // A link that reproduces this exact analysis, so a result can be sent to a
  // colleague instead of described. The area has to travel somehow; encoding the
  // drawn geometry is smaller than uploading a file.
  // Withdraw this area. The app holds a series derived from the submitted
  // geometry and a job record naming it, and there are no accounts to ask
  // through, so without this the user has no way to take any of it back. The
  // cache key is the capability: only someone who submitted that geometry has it.
  async function forgetThisArea() {
    if (!activeCacheKey) return;
    const backendUrl = (process.env.NEXT_PUBLIC_BACKEND_URL || '/api').replace(/\/$/, '');
    try {
      const response = await fetch(`${backendUrl}/rainfall/forget`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ cache_key: activeCacheKey }),
      });
      if (!response.ok) {
        const failure = await response.json().catch(() => null);
        throw new Error(failure?.detail || `Request failed (${response.status}).`);
      }
      const body = await response.json();
      toast({
        title: 'Area removed',
        description: body.removed
          ? 'The stored series for this area has been deleted from this server and from object storage.'
          : 'Nothing was stored for this area, so there was nothing to delete.',
      });
      clearAnalysis();
    } catch (error) {
      toast({
        title: 'Could not remove the area',
        description: error instanceof Error ? error.message : 'Unknown error',
        variant: 'destructive',
      });
    }
  }

  /**
   * The numbers as a file.
   *
   * Prose to the clipboard was the only way out, and prose is the wrong shape for
   * the likely actual need: someone who wants the monthly figures for a report,
   * a spreadsheet or another tool. CSV carries the series; JSON carries everything
   * the cards show, so a result can be re-read later without the server.
   */
  function exportCsv(): string {
    const rows = (analysisSummary?.rainfall as { series?: Array<Record<string, number | string | null>> } | undefined)?.series
      ?? [];
    const header = 'month,precip_mm,normal_mm,anomaly_mm,anomaly_pct';
    const body = rows
      .map((r) => [r.month, r.precip_mm, r.normal_mm, r.anomaly_mm, r.anomaly_pct]
        .map((v) => (v == null ? '' : String(v))).join(','))
      .join('\n');
    return [header, body].join('\n');
  }

  function exportJson(): string {
    return JSON.stringify(
      {
        exported_at: new Date().toISOString(),
        note: 'Derived from public satellite and reanalysis products. '
          + 'Every figure carries an evidence label; read them before relying on them.',
        country: analysisSummary?.country ?? null,
        area_km2: (analysisSummary?.rainfall as { label?: string } | undefined)?.label ?? null,
        summary: analysisSummary,
      },
      null,
      2,
    );
  }

  function download(filename: string, contents: string, type: string) {
    // A data URL keeps this a single click with no round trip and no server
    // round trip to hold a file. The byte cap matters: a 195-month series is
    // small, but an unbounded summary is not something to base64 into a URL.
    const blob = new Blob([contents], { type });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = filename;
    document.body.appendChild(anchor);
    anchor.click();
    document.body.removeChild(anchor);
    URL.revokeObjectURL(url);
  }

  // A link that reproduces this exact analysis, so a result can be sent to a
  // colleague instead of described. The area travels in the query string because
  // there is no account, no storage, and no id to point at.
  //
  // This used to be gated on an *uploaded* file, so a drawn area never got a
  // link at all, which is the common case: drawing is what the map invites.
  const shareLink = (() => {
    const geojson = currentGeojson();
    if (!geojson) return '';
    try {
      const q = new URLSearchParams({
        area: encodeURIComponent(JSON.stringify(geojson)),
        // An administrative id is stable, short, and is what another system
        // already holds -- so a link from globe or a colleague's bookmark
        // resolves rather than carrying a megabyte of coordinates.
        ...(adminArea?.id ? { admin: adminArea.id, level: String(adminArea.level ?? 1) } : {}),
        datasets: selectedDatasets.join(','),
        sensor: selectedSensor,
        // The window that was actually analysed, not the preset it came from. A
        // link encoded `years` while the reader could be looking at a custom
        // range, so a shared link reproduced an approximation of what the sender
        // saw rather than the thing itself.
        window_start: windowStartISO(),
        window_end: windowEndISO(),
      });
      const link = `${window.location.origin}${window.location.pathname}?${q.toString()}`;
      // A detailed boundary is a megabyte of coordinates and the link silently
      // dies the moment it is pasted into a chat client or an email. Better to
      // say so than to hand over something that looks shareable and is not.
      return link.length > SHARE_URL_LIMIT ? '' : link;
    } catch {
      return '';
    }
  })();

  // Why there is no link, when there is no link. Silent absence would read as
  // "this app cannot be shared", which is the wrong conclusion.
  const shareLinkUnavailable = (() => {
    if (currentGeojson() && !shareLink) {
      return 'This boundary is too detailed for a link. Download the GeoJSON and share the file instead.';
    }
    return '';
  })();

  // Preserve the source geometry whenever one was supplied. A bounding box is
  // only a fallback for selections created by older map interactions.
  function currentGeojson(): GeoJsonObject | undefined {
    if (uploadedGeojson) return uploadedGeojson;
    if (drawnFeatures?.features.length) return drawnFeatures;
    if (!boundingBox) return undefined;
    return {
      type: "Feature",
      geometry: {
        type: "Polygon",
        coordinates: [[
          [boundingBox.west, boundingBox.south],
          [boundingBox.east, boundingBox.south],
          [boundingBox.east, boundingBox.north],
          [boundingBox.west, boundingBox.north],
          [boundingBox.west, boundingBox.south]
        ]]
      },
      properties: {}
    } as GeoJsonObject;
  }

  // Queue a precomputation for the current area. This is the path that makes
  // "bring your own polygon" real: a drawn or uploaded boundary with no series
  // can be queued without leaving the page.
  async function submitForPreprocessing() {
    const geojson = currentGeojson();
    if (!geojson) return;
    const backendUrl = (process.env.NEXT_PUBLIC_BACKEND_URL || '/api').replace(/\/$/, '');
    setIsLoading(true);
    try {
      const response = await fetch(`${backendUrl}/rainfall/submit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ geojson, product: rainProduct }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data?.detail || 'Submission failed');
      }
      setPlan(data.planned || null);
      setSubmission((previous) => ({
        ...previous,
        [data.cache_key]: {
          state: data.submission?.state || 'pending',
          reason: data.submission?.reason,
          cacheKey: data.cache_key,
        },
      }));
      if (data.submission?.state === 'ready') {
        setProgress(null);
      } else {
        void pollSubmission(data.cache_key, data.indicator || 'rainfall');
      }
    } catch (error) {
      setResponse(error instanceof Error ? error.message : 'Submission failed');
      setShowResults(true);
    } finally {
      setIsLoading(false);
    }
  }

  const handleAnalyze = async () => {
    if (!uploadedGeojson && !drawnFeatures?.features.length && !boundingBox) return;
    if (selectedDatasets.length === 0) {
      toast({
        title: "Choose at least one dataset",
        description: "Select the information you want to include before running the analysis.",
        variant: "destructive",
      });
      return;
    }

    setIsLoading(true);
    setResponse('');
    setAnalysisWarnings([]);
    setAnalysisSummary(null);

    try {
      // Same-origin requests work through the reverse proxy in production and
      // through the Next.js rewrite in local Docker development.
      const backendUrl = (process.env.NEXT_PUBLIC_BACKEND_URL || '/api').replace(/\/$/, '');
      const geojson = currentGeojson();
      if (!geojson) return;

      const params = new URLSearchParams({
        include_ndvi: String(selectedDatasets.includes('ndvi')),
        datasets: selectedDatasets.join(','),
        sensor: selectedSensor,
        window_start: windowStartISO(),
        window_end: windowEndISO(),
      });
      const response = await fetch(`${backendUrl}/generate-context?${params.toString()}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ geojson, product: rainProduct }),
      });

      if (!response.ok) {
        const failure = await response.json().catch(() => null);
        // A 413 is not one thing. Only the area-too-large case has an offline
        // route to offer; "too many vertices" and "payload too large" are
        // answered by simplifying the geometry, and sending those to the
        // offline panel told the user to wait for a job that would never help.
        const detail = String(failure?.detail || '');
        const isAreaTooLarge = detail.includes('synchronous limit');
        if (response.status === 413) {
          if (geojson && isAreaTooLarge) {
            await loadOfflineOffer(geojson);
            return;
          }
          // Other 413 cases (payload too large, too many vertices) are also
          // retryable by simplifying the geometry, but we surface the error
          // so the user knows what happened and can retry with a simpler shape.
          throw new Error(
            `Area too large or payload too large. Try a smaller boundary or ` +
            `fewer vertices. ${failure?.detail ? `: ${failure.detail}` : ``}`
          );
        }
        throw new Error(failure?.detail || `Analysis request failed (${response.status}).`);
      }

      const data = await response.json() as { summary?: Summary };
      // The analysis block lives *inside* summary. Read at the top level it was
      // always undefined, which is why the containing-unit offer never appeared
      // -- found by driving the page rather than by reading the payload shape.
      // Remember where this outline is, so "which administrative unit is this?"
      // can be asked from the server's own figure rather than a second estimate
      // of the same outline made in the browser.
      const analysis = data.summary?.analysis;
      const box = analysis?.bbox;
      if (Array.isArray(box) && box.length === 4 && analysis?.bbox_area_km2 != null) {
        setDrawnInfo({
          areaKm2: analysis.bbox_area_km2,
          lon: (Number(box[0]) + Number(box[2])) / 2,
          lat: (Number(box[1]) + Number(box[3])) / 2,
        });
      }
      setActiveCacheKey(
        (data.summary?.rainfall as { cache_key?: string } | undefined)?.cache_key ?? null
      );
      const summary = data.summary || {};
      const result = summarizeData(summary);
      const ndviWarning = summary.ndvi?.warning;
      setAnalysisWarnings(ndviWarning ? [{ message: ndviWarning, status: summary.ndvi?.status }] : []);
      setAnalysisSummary(summary);
      const areaKey = (summary.rainfall as { cache_key?: string } | undefined)?.cache_key;
      setActiveCacheKey(areaKey ?? null);
      if (areaKey) void loadVegetationSeries(areaKey);
      setResponse(result);
      setSummaryText(result);
    } catch (err) {
      console.error(err);
      const message = "Error: " + (err as Error).message;
      setResponse(message);
      setSummaryText(message);
    } finally {
      setIsLoading(false);
    }
  };

  // Keep the poller's view of the analysis current.
  analyzeRef.current = handleAnalyze;

  return (
    <div className="min-h-screen bg-void text-ink">
      {/* The glow off the instrument. One radial in the signal hue, low enough
          to read as a lit surface rather than as a gradient background. */}
      <div
        aria-hidden
        className="pointer-events-none fixed inset-x-0 top-0 z-0 h-[560px]"
        style={{
          background:
            'radial-gradient(1200px 460px at 16% -12%, rgba(185,232,75,0.11), transparent 62%),' +
            'radial-gradient(900px 400px at 84% -16%, rgba(79,168,60,0.09), transparent 60%)',
        }}
      />

      {/* ------------------------------------------------------------- top bar */}
      <header
        data-print-hide
        className="sticky top-0 z-[900] border-b border-rule/60 bg-void/85 backdrop-blur-xl"
      >
        <div className="mx-auto flex max-w-[1440px] items-center gap-5 px-5 py-3 sm:px-8">
          <a href="#top" className="flex items-center gap-2.5">
            <Wordmark />
          </a>

          <span className="fig ml-auto hidden text-[0.6875rem] tracking-[0.08em] text-ink-3 md:block">
            {SOURCES.map((source) => source.label).join('  ·  ')}
          </span>

          <a
            href="#workspace"
            className="btn btn-ghost h-8 px-3 text-[0.8125rem] sm:ml-auto md:ml-0"
          >
            Open the map
            <ArrowDown className="h-3.5 w-3.5" />
          </a>
        </div>
      </header>

      <main className="relative z-10">
        {/* -------------------------------------------------------------- hero */}
        <section id="top" className="mx-auto max-w-[1440px] px-5 pt-16 sm:px-8 sm:pt-24">
          <p className="fig flex flex-wrap items-center gap-x-2 gap-y-1 text-[0.6875rem] tracking-[0.14em] text-ink-3 uppercase">
            {SOURCES.map((source) => (
              <span key={source.id} className="inline-flex items-center gap-2">
                <span className="h-1 w-1 rounded-full bg-signal-dim" />
                {source.label}
              </span>
            ))}
          </p>

          {/* Scale contrast is the whole Do argument in one element: a claim set
              far larger than anything else on the page, with the italic doing
              the work a second colour would otherwise do. */}
          <h1 className="display display-xl mt-7 max-w-[19ch]">
            Point at a piece of land. <span className="mark">Get the truth about it.</span>
          </h1>

          <div
            data-print-hide
            className="mt-9 grid gap-10 border-t border-rule pt-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,0.85fr)] lg:gap-16"
          >
            <p className="max-w-[54ch] text-[1.0625rem] leading-relaxed text-ink-2">
              Terrain, vegetation, rainfall and drought for any boundary you draw on
              Earth — read from the same public sources a satellite report would
              use, with the provenance attached to every number. No account, no
              upload, no survey licence.
            </p>

            {/* The four modules, named before anyone has to discover them. A
                stranger should know what the product does in under a second,
                which is the only honest place to spend a first impression. */}
            <ul className="grid grid-cols-2 gap-x-6 gap-y-4 self-start">
              {SOURCES.map((source, index) => (
                <li key={source.id} className="border-l border-rule pl-3.5">
                  <span className="fig text-[0.6875rem] text-ink-3">
                    {String(index + 1).padStart(2, '0')}
                  </span>
                  <p className="mt-1 text-[0.8125rem] font-medium leading-tight">{source.label}</p>
                  <p className="fig mt-0.5 text-[0.6875rem] text-ink-3">{source.role}</p>
                </li>
              ))}
            </ul>
          </div>

          <div data-print-hide className="mt-10 flex items-center gap-4">
            <div className="ramp w-40" aria-hidden />
            <span className="label">the false-colour scale this reads</span>
          </div>
        </section>

        {/* --------------------------------------------------------- workspace */}
        <section
          id="workspace"
          className="mx-auto max-w-[1440px] scroll-mt-20 px-5 pt-20 sm:px-8"
        >
          {/* The cap used to be an amber alarm across the top of the page, read
              on arrival and never again. It is a fact about the service, so it
              now sits with the controls it constrains, as a quiet note. */}
          <div className="mb-6 flex items-start gap-2.5 rounded-xl border border-rule bg-surface/60 px-4 py-3">
            <Info className="mt-0.5 h-4 w-4 shrink-0 text-ink-3" />
            <p className="text-[0.8125rem] leading-relaxed text-ink-3">
              A study area is read live up to{' '}
              <span className="fig text-ink-2">{formatNumber(syncLimitKm2 ?? 100, 0)} km²</span>.
              Anything larger is not refused — it is queued and read offline at a
              coarser resolution, and the app tells you which before you commit.
            </p>
          </div>

          <div className="grid items-start gap-6 lg:grid-cols-[minmax(0,380px)_minmax(0,1fr)]">
            {/* ---------------------------------------------------- the rail */}
            <aside
              data-print-hide
              className="rail lg:sticky lg:top-20 lg:max-h-[calc(100vh-6rem)] lg:overflow-y-auto"
            >
              <div className="border-b border-rule px-5 py-4">
                <h2 className="text-[0.9375rem] font-semibold tracking-[-0.01em]">
                  Choose your area
                </h2>
                <p className="mt-1 text-[0.8125rem] leading-relaxed text-ink-3">
                  Name it, find a place, draw on the map, or bring your own
                  boundary.
                </p>
              </div>

              {/* Naming an area leads. It is what a county planner, a
                  conservancy manager and anyone with a boundary already knows
                  their area by name actually types, and it was the third control
                  in the rail behind a place search that only zooms -- so the one
                  input that produces a ready-made boundary was the hardest to
                  find. The four ways in are now ordered by how much work they save
                  the reader. */}
              <div className="px-5 pt-5">
                <Label htmlFor="admin-area" className="label mb-2.5 block">
                  Name your area
                </Label>
                <div className="flex gap-2">
                  <Input
                    id="admin-area"
                    type="text"
                    placeholder="Narok, Isiolo, Meru…"
                    value={adminQuery}
                    onChange={(e) => setAdminQuery(e.target.value)}
                    onKeyDown={(e) => { if (e.key === 'Enter') void resolveAdminArea(); }}
                  />
                  <Button
                    size="sm"
                    variant="outline"
                    disabled={adminBusy || adminQuery.trim().length === 0}
                    onClick={() => void resolveAdminArea()}
                  >
                    {adminBusy ? <Loader2 className="h-3.5 w-3.5 animate-spin" /> : 'Find'}
                  </Button>
                </div>
                <p className="mt-2 text-[0.75rem] leading-relaxed text-ink-3">
                  A county, district or region. You get its exact boundary and
                  the map moves to it.
                </p>
                {adminArea && (
                  <p className="fig mt-2 text-xs text-ink-2">
                    Using {adminArea.name}, administrative level {adminArea.level}.
                    Its outline is the whole unit, not something you drew.
                  </p>
                )}
              </div>

              {/* Search */}
              <div className="rule-t px-5 py-5">
                <Label htmlFor="place-search" className="label mb-2.5 block">
                  Or find a place
                </Label>
                <div className="relative">
                  <Search className="pointer-events-none absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 text-ink-3" />
                  <Input
                    id="place-search"
                    type="text"
                    placeholder="Nairobi, Kajiado, Mara North…"
                    value={searchQuery}
                    onChange={(e) => handleSearchChange(e.target.value)}
                    className="pl-10 pr-10"
                    onFocus={() => searchResults.length > 0 && setShowResults(true)}
                  />
                  {isSearching && (
                    <Loader2 className="absolute right-3.5 top-1/2 h-4 w-4 -translate-y-1/2 animate-spin text-ink-3" />
                  )}
                </div>

                {showResults && searchResults.length > 0 && (
                  <div className="mt-2 overflow-hidden rounded-xl border border-rule bg-raised">
                    {searchResults.map((result) => (
                      <button
                        key={result.place_id}
                        type="button"
                        onClick={() => handleLocationSelect(result)}
                        className="hoverline flex w-full items-start gap-2.5 border-b border-rule/60 px-3.5 py-2.5 text-left last:border-b-0"
                      >
                        <MapPin className="mt-0.5 h-3.5 w-3.5 shrink-0 text-signal-dim" />
                        <span className="text-[0.8125rem] leading-snug text-ink-2">
                          {result.display_name}
                        </span>
                      </button>
                    ))}
                  </div>
                )}
              </div>

              {/* The other direction. Having drawn something, it is worth knowing
                  which administrative unit it sits in -- and worth being able to
                  adopt that unit, because a conservancy's drought is a question
                  about the whole unit rather than about the 60 km2 someone
                  circled. Offered, never automatic: snapping replaces the outline
                  the person drew. */}
              <div className="rule-t px-5 py-5">
                {drawnInfo && !adminArea && (
                  <div>
                    <Button
                      size="sm"
                      variant="outline"
                      disabled={containingBusy}
                      onClick={() => void findContainingArea()}
                    >
                      {containingBusy
                        ? <Loader2 className="h-3.5 w-3.5 animate-spin" />
                        : 'Which unit is this in?'}
                    </Button>
                    {containingArea && (
                      <div className="mt-2">
                        <p className="fig text-xs text-ink-2">
                          Your {formatNumber(drawnInfo.areaKm2, 0)} km² outline sits
                          inside {containingArea.name}, which covers{' '}
                          {formatNumber(containingArea.area_km2, 0)} km².
                        </p>
                        <Button
                          size="sm"
                          variant="ghost"
                          className="mt-1.5"
                          disabled={containingBusy}
                          onClick={() => void adoptContainingArea()}
                        >
                          Use the whole {containingArea.name} instead
                        </Button>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Upload */}
              <div className="rule-t px-5 py-5">
                <Label htmlFor="geojson-upload" className="label mb-2.5 block">
                  Or bring a boundary
                </Label>
                <input
                  id="geojson-upload"
                  type="file"
                  accept=".geojson,application/geo+json,application/json"
                  onChange={handleGeojsonUpload}
                  className="peer sr-only"
                />
                {/* The native file input renders an unstyleable "Choose File /
                    No file chosen" pair. It is kept in the DOM for the label and
                    for keyboard focus, and replaced by a control that can
                    actually be designed. */}
                <label
                  htmlFor="geojson-upload"
                  className="btn btn-ghost h-10 w-full cursor-pointer px-3.5 text-[0.8125rem] peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-signal"
                >
                  <Upload className="h-4 w-4" />
                  Choose a .geojson
                </label>
                <p className="mt-2.5 text-[0.75rem] leading-relaxed text-ink-3">
                  A <span className="fig">.geojson</span> Polygon, MultiPolygon or
                  FeatureCollection, up to 4 MB. The server and the proxy were raised
                  together, so a large conservancy boundary can be posted rather than
                  refused at the door.
                </p>
              </div>

              {/* Options */}
              <div className="rule-t px-5 py-5">
                <p className="label mb-4">Options</p>

                <div className="space-y-5">
                  <div>
                    {/* Dates first and presets second, because the dates are the
                        question. The preset row used to lead, which made "the 2019/20
                        drought" and "since the last rains" -- the ranger's and the NRT
                        manager's actual questions -- inexpressible without finding a
                        date field second. */}
                    <div className="mb-2.5 flex flex-wrap items-center gap-1.5">
                      <Label className="mr-1 text-[0.8125rem] text-ink-2">Window</Label>
                      <input
                        type="date"
                        value={customStart}
                        max={customEnd || windowEndISO()}
                        onChange={(e) => setCustomStart(e.target.value)}
                        aria-label="Window start"
                        className="fig h-8 rounded-lg border border-rule bg-raised px-2 text-ink-2"
                      />
                      <span className="text-[0.75rem] text-ink-3">to</span>
                      <input
                        type="date"
                        value={customEnd}
                        min={customStart || undefined}
                        onChange={(e) => setCustomEnd(e.target.value)}
                        aria-label="Window end"
                        className="fig h-8 rounded-lg border border-rule bg-raised px-2 text-ink-2"
                      />
                    </div>

                    <div className="fig mb-2.5 flex flex-wrap items-center gap-1.5 text-[0.75rem] text-ink-3">
                      <span>or</span>
                      {([1, 3, 10, 30] as const).map((years) => (
                        <button
                          key={years}
                          type="button"
                          aria-pressed={!customStart && windowYears === years}
                          onClick={() => { setWindowYears(years); setCustomStart(''); setCustomEnd(''); }}
                          className={`fig h-7 rounded-lg border px-2.5 transition-colors duration-200 ${
                            !customStart && windowYears === years
                              ? 'border-signal bg-signal/12 text-signal'
                              : 'border-rule text-ink-3 hover:border-line-2 hover:text-ink-2'
                          }`}
                        >
                          {years}y
                        </button>
                      ))}
                    </div>

                    <p className="fig mt-2.5 text-[0.75rem] text-ink-2">
                      {windowStartISO()} → {windowEndISO()}
                      <span className="text-ink-3">{windowExplanation}</span>
                    </p>
                    {!customRangeValid && (
                      <p className="mt-1.5 text-[0.75rem] text-stressed">
                        The start date is after the end date, or in the future.
                      </p>
                    )}
                    <p className="mt-2 text-[0.75rem] leading-relaxed text-ink-3">
                      This window selects the vegetation period, and sets how finely
                      precipitation is sampled. The rainfall record behind it is always
                      the full monthly series — the charts below simply show the part
                      of it you asked to see. Elevation and land cover have no time
                      dimension, so no window changes them.
                    </p>
                  </div>

                  <div>
                    {/* Which precipitation product. Not a preference: they differ
                        in resolution, freshness and evidence class, and the reader
                        is told what choosing costs. */}
                    <div className="mb-2.5 flex items-center gap-2">
                      <Label htmlFor="rain-product" className="text-[0.8125rem] text-ink-2">
                        Rainfall source
                      </Label>
                      <select
                        id="rain-product"
                        value={rainProduct}
                        onChange={(e) => setRainProduct(e.target.value as 'rainfall' | 'chirps')}
                        className="h-8 rounded-lg border border-rule bg-raised px-2 text-[0.75rem] text-ink-2"
                      >
                        <option value="rainfall">ERA5 · 28 km · modelled</option>
                        <option value="chirps">CHIRPS · 5.6 km · observed</option>
                      </select>
                    </div>
                    <p className="text-[0.75rem] leading-relaxed text-ink-3">
                      {rainProduct === 'chirps'
                        ? 'CHIRPS is a satellite-and-gauge blend at 0.05°, so a small area is described by one 5.6 km cell rather than by a cell covering about 780 km². It is about five months fresher than ERA5. Measured against ERA5 here it reads 0–24% higher, the difference growing as the land gets drier.'
                        : 'ERA5 is a reanalysis at 0.25°. It is modelled output, not a gauge reading, and below about 780 km² the figure is one grid cell rather than this outline. CHIRPS is finer, fresher and classified observed — switch above if your area is small.'}
                    </p>
                  </div>

                  <div>
                    <div className="mb-2.5 flex items-center gap-2">
                      <Label htmlFor="vegetation-source" className="text-[0.8125rem] text-ink-2">
                        Vegetation source
                      </Label>
                      <select
                        id="vegetation-source"
                        value={selectedSensor}
                        onChange={(event) => setSelectedSensor(event.target.value)}
                        className="h-8 rounded-lg border border-rule bg-raised px-2 text-[0.75rem] text-ink-2"
                      >
                        <option value="auto">auto</option>
                        <option value="sentinel-2">Sentinel-2</option>
                        <option value="landsat">Landsat</option>
                        <option value="modis">MODIS</option>
                      </select>
                    </div>
                    <p className="text-[0.75rem] leading-relaxed text-ink-3">
                      Auto picks the source whose cloud mask can be trusted, and tells
                      you why it picked it.
                    </p>
                  </div>

                  <div>
                    <p className="label mb-2.5">Read</p>
                    <div className="space-y-0.5">
                      {DATASET_OPTIONS.map((dataset) => (
                        <label
                          key={dataset.id}
                          htmlFor={`dataset-${dataset.id}`}
                          className="hoverline flex cursor-pointer items-start gap-2.5 px-2 py-2"
                        >
                          <input
                            type="checkbox"
                            id={`dataset-${dataset.id}`}
                            checked={selectedDatasets.includes(dataset.id)}
                            onChange={(event) => {
                              setSelectedDatasets((current) => event.target.checked
                                ? [...current, dataset.id]
                                : current.filter((id) => id !== dataset.id),
                              );
                            }}
                            className="mt-0.5 h-4 w-4 shrink-0 cursor-pointer appearance-none rounded-[5px] border border-line-2 bg-raised transition-colors duration-150 checked:border-signal checked:bg-signal focus-visible:outline-none focus-visible:ring-4 focus-visible:ring-signal/20"
                            style={
                              selectedDatasets.includes(dataset.id)
                                ? {
                                    backgroundImage:
                                      "url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 16 16' fill='%230a1405'%3E%3Cpath d='M6.2 11.3 3.4 8.5l1.1-1.1 1.7 1.7 4.6-4.6 1.1 1.1z'/%3E%3C/svg%3E\")",
                                    backgroundSize: '100%',
                                  }
                                : undefined
                            }
                          />
                          <span>
                            <span className="block text-[0.8125rem] text-ink-2">{dataset.label}</span>
                            <span className="block text-[0.75rem] text-ink-3">{dataset.description}</span>
                          </span>
                        </label>
                      ))}
                    </div>
                  </div>
                </div>
              </div>

              <div className="rule-t px-5 py-5">
                <Button
                  onClick={handleAnalyze}
                  disabled={
                    (!boundingBox && !uploadedGeojson && !drawnFeatures?.features.length)
                    || selectedDatasets.length === 0
                    || isLoading
                    || !customRangeValid
                  }
                  className="btn-primary h-12 w-full text-[0.9375rem]"
                >
                  {isLoading ? (
                    <>
                      <span className="h-1.5 w-1.5 rounded-full bg-void beat" />
                      Reading the sources
                    </>
                  ) : (
                    <>
                      <Satellite className="h-4 w-4" />
                      Read this area
                    </>
                  )}
                </Button>
                <p className="mt-2.5 text-center text-[0.75rem] text-ink-3">
                  Nothing is stored against an account. The boundary is hashed into
                  a key, and you can withdraw it whenever you like.
                </p>
              </div>
            </aside>

            {/* ---------------------------------------------------- the stage */}
            <div className="space-y-6">
              {/* Map */}
              <div className="panel overflow-hidden p-0">
                <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-1 border-b border-rule px-5 py-3.5">
                  <div className="flex items-center gap-2.5">
                    <Globe className="h-4 w-4 text-signal" />
                    <span className="text-[0.9375rem] font-medium">Study area</span>
                  </div>
                  <p className="fig text-[0.75rem] text-ink-3">
                    {boundingBox || drawnFeatures?.features.length || uploadedGeojson
                      ? 'Area selected — ready to read'
                      : 'Esri World Imagery · draw with the toolbar top right'}
                  </p>
                </div>

                <div className="relative">
                  <MapComponent
                    selectedLocation={selectedLocation}
                    onBoundingBoxCreated={handleBoundingBoxCreated}
                    uploadedGeoJSON={uploadedGeojson}
                    onSaveFeatures={handleFeaturesChange}
                  />
                  {drawnFeatures && drawnFeatures.features.length > 0 && (
                    <button
                      onClick={() => {
                        const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(drawnFeatures, null, 2));
                        const downloadAnchorNode = document.createElement('a');
                        downloadAnchorNode.setAttribute("href", dataStr);
                        downloadAnchorNode.setAttribute("download", "drawn_features.geojson");
                        document.body.appendChild(downloadAnchorNode); // required for firefox
                        downloadAnchorNode.click();
                        downloadAnchorNode.remove();
                      }}
                      className="btn btn-ghost absolute bottom-10 right-4 z-[1000] h-9 px-3 text-[0.8125rem]"
                    >
                      <Download className="h-3.5 w-3.5" />
                      Download GeoJSON
                    </button>
                  )}
                </div>
              </div>

              {/* Results */}
              <div className="panel overflow-hidden p-0">
                <div className="flex flex-wrap items-center justify-between gap-x-4 gap-y-2 border-b border-rule px-5 py-3.5">
                  <div className="flex items-center gap-2.5">
                    <MapPin className="h-4 w-4 text-signal" />
                    <span className="text-[0.9375rem] font-medium">The reading</span>
                  </div>
                  {summaryText && <CopySummary summaryText={summaryText} />}
                </div>

                <div className="px-5 py-5">
                  {shareLinkUnavailable && (
                    <div data-print-hide className="mb-4 text-xs text-ink-3">
                      {shareLinkUnavailable}
                    </div>
                  )}
                  {shareLink && (
                    <div
                      data-print-hide
                      className="mb-6 flex flex-wrap items-center gap-2 rounded-xl border border-rule bg-raised/60 px-3.5 py-3"
                    >
                      <span className="text-[0.75rem] text-ink-3">Take it with you</span>
                      <code className="fig max-w-xs truncate rounded-md border border-rule bg-void px-2 py-1 text-[0.75rem] text-ink-3">
                        {shareLink}
                      </code>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => navigator.clipboard?.writeText(shareLink)}
                      >
                        <Link2 className="h-3.5 w-3.5" />
                        Copy link
                      </Button>
                      <span className="basis-full" />
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => download('rainfall-series.csv', exportCsv(), 'text/csv')}
                      >
                        CSV
                      </Button>
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => download('analysis.json', exportJson(), 'application/json')}
                      >
                        JSON
                      </Button>
                      <Button size="sm" variant="outline" onClick={() => window.print()}>
                        <Printer className="h-3.5 w-3.5" />
                        Print
                      </Button>
                      {activeCacheKey && (
                        // Withdrawal is a first-class action, not a support email.
                        // The app holds a series derived from the submitted
                        // geometry, and the person who submitted it should be the
                        // one who can say to remove it.
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={() => void forgetThisArea()}
                        >
                          Remove this area
                        </Button>
                      )}
                    </div>
                  )}
                  {offline && (
                    <div className="mb-6 rule-t pt-5">
                      <p className="headline">
                        This area is <span className="fig text-signal">{formatNumber(offline.areaKm2, 0)} km²</span>
                      </p>
                      <p className="fig mt-1.5 text-[0.8125rem] leading-relaxed text-ink-2">
                        A live request handles up to {formatNumber(offline.limitKm2, 0)} km²
                        so it stays inside a few seconds. A larger area is not refused
                        — it is read offline, at a coarser resolution, and you choose
                        whether to wait.
                      </p>
                      <dl className="rows mt-4">
                        {offline.plans.map((plan) => (
                          <Figure
                            key={plan.indicator}
                            term={
                              plan.already_computed
                                ? `${plan.indicator} — already computed`
                                : plan.indicator || ''
                            }
                            value={`${
                              plan.resolution_m ? `${plan.resolution_m} m` : `${plan.resolution_km} km`
                            } · ${plan.estimated_seconds ?? '?'}s est.`}
                          />
                        ))}
                      </dl>
                      {offline.queued.length ? (
                        <p className="fig mt-3 text-sm text-ink">
                          Queueing {offline.queued.join(', ')}. The progress line reports
                          when each is ready.
                        </p>
                      ) : (
                        <button
                          type="button"
                          onClick={queueOffline}
                          className="btn btn-primary mt-4 h-10 px-4 text-[0.8125rem]"
                        >
                          Process {offline.plans.length} module
                          {offline.plans.length === 1 ? '' : 's'} offline
                        </button>
                      )}
                      <p className="fig mt-2.5 text-[0.6875rem] text-ink-3">
                        Times are estimates from the measured per-read cost, not
                        guarantees. A coarser grid is a different kind of claim, so the
                        resolution each module will use is listed rather than buried.
                      </p>
                    </div>
                  )}
                  {analysisWarnings.map((warning) => (
                    <Alert key={warning.message} variant="caution" className="mb-4">
                      <Satellite className="h-4 w-4" />
                      <AlertDescription>
                        <strong className="font-semibold">
                          {warning.status === 'skipped'
                            ? 'Vegetation analysis was skipped:'
                            : 'Vegetation analysis note:'}
                        </strong>{' '}
                        {warning.message}
                      </AlertDescription>
                    </Alert>
                  ))}

                  {isLoading ? (
                    /* A wait, described. The old interface put a spinning globe
                       over a grey box; this states what is being read and draws a
                       sweep across it, so the pause looks like an instrument
                       working rather than a page that has stopped. */
                    <div className="flex flex-col gap-5 py-8">
                      <div className="flex items-center gap-3">
                        <span className="beat h-2 w-2 rounded-full bg-signal" />
                        <span className="label">Reading the sources</span>
                      </div>
                      <div className="h-px w-full overflow-hidden bg-raised">
                        <div className="scan h-px w-full" />
                      </div>
                      <ul className="grid gap-2.5 sm:grid-cols-2">
                        {SOURCES.map((source) => (
                          <li
                            key={source.id}
                            className="flex items-center gap-2.5 text-[0.8125rem] text-ink-3"
                          >
                            <span className="h-1 w-1 rounded-full bg-signal-dim" />
                            {source.label}
                            <span className="text-ink-3/60">— {source.role}</span>
                          </li>
                        ))}
                      </ul>
                      <p className="max-w-[52ch] text-[0.8125rem] leading-relaxed text-ink-3">
                        Public satellite services take a moment on a cold cache. The
                        first read of any area is slower than the ones after it.
                      </p>
                    </div>
                  ) : response ? (
                    <>
                      <div className="max-w-[68ch] text-[1.0625rem] leading-relaxed text-ink-2">
                        {responseIsError ? (
                          // An error used to render in the same paragraph style as
                          // a successful result, in the same panel, with the same
                          // weight. Reading "Error: Raster processing timed out" as
                          // a finding is exactly the failure this avoids.
                          <div
                            role="alert"
                            className="rounded-xl border border-bare/40 bg-bare/10 p-4"
                          >
                            <p className="mb-2 flex items-center gap-2 text-sm font-semibold text-bare">
                              <span aria-hidden>!</span>
                              {errorHeadline}
                            </p>
                            {errorDetail && (
                              <p className="text-sm text-bare/90">{errorDetail}</p>
                            )}
                            <p className="mt-2 text-xs text-bare/75">
                              Nothing was charged for a failed analysis. Try a smaller
                              boundary, or press Read this area again.
                            </p>
                            <Button
                              size="sm"
                              variant="outline"
                              className="mt-3.5"
                              onClick={() => void handleAnalyze()}
                            >
                              Try again
                            </Button>
                          </div>
                        ) : (
                          response.split('\n').map((paragraph, index) => (
                            <p key={index} className="mb-3 last:mb-0">{paragraph}</p>
                          ))
                        )}
                      </div>

                      {analysisSummary && (
                        <div className="mt-8 border-t border-rule pt-7">
                          <div className="mb-6 flex flex-wrap items-baseline gap-x-5 gap-y-1.5">
                            <h3 className="display display-md">Selected data</h3>
                            {analysisSummary.country && (
                              <span className="text-[0.9375rem] text-ink-2">
                                {analysisSummary.country}
                              </span>
                            )}
                            {analysisSummary.analysis?.bbox_area_km2 != null && (
                              <span className="fig text-[0.8125rem] text-ink-3">
                                bounding box{' '}
                                {formatNumber(analysisSummary.analysis.bbox_area_km2, 2)} km²
                              </span>
                            )}
                          </div>
                          {analysisSummary.caveats && analysisSummary.caveats.length > 0 && (
                            <div className="mb-6 rounded-xl border border-stressed/35 bg-stressed/10 p-4">
                              <p className="label text-stressed">
                                Before relying on these numbers
                              </p>
                              <ul className="mt-2 space-y-1.5 pl-4 text-[0.8125rem] leading-relaxed text-ink-2 marker:text-stressed">
                                {analysisSummary.caveats.map((caveat, i) => (
                                  <li key={i} className="list-disc">{caveat}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                          <div className="grid gap-x-10 gap-y-2 xl:grid-cols-2">
                            {progress && (
                              <div className="xl:col-span-2">
                                <JobProgressLine
                                  progress={progress}
                                  planned={plan}
                                  onDismiss={() => setProgress(null)}
                                />
                              </div>
                            )}
                            {(analysisSummary.analysis?.datasets || selectedDatasets)
                              .filter(isDatasetId)
                              .map((dataset) => (
                                <DatasetResultCard
                                  key={dataset}
                                  dataset={dataset}
                                  summary={analysisSummary}
                                  submission={activeCacheKey ? submission[activeCacheKey] : undefined}
                                  onSubmit={dataset === 'rainfall' ? submitForPreprocessing : undefined}
                                  vegetationSeries={vegetationSeries}
                                 />
                               ))}
                           </div>
                           {analysisSummary.measures && analysisSummary.measures.length > 0 && (
                             <div className="mt-8">
                               <h4 className="label mb-3">Measures</h4>
                               <dl className="grid gap-3 sm:grid-cols-2">
                                 {analysisSummary.measures.map((m) => (
                                   <MeasureCard key={m.product} measure={m} />
                                 ))}
                               </dl>
                             </div>
                           )}
                         </div>
                      )}
                    </>
                  ) : (
                    /* The empty state is the first thing a stranger reads after
                       the promise, so it confirms the promise instead of
                       describing a control. */
                    <div className="flex flex-col items-start gap-5 py-10">
                      <div className="ramp w-28" aria-hidden />
                      <p className="display display-md max-w-[24ch] text-ink">
                        Nothing here yet. That is normal.
                      </p>
                      <p className="max-w-[52ch] text-[0.9375rem] leading-relaxed text-ink-3">
                        Draw a boundary on the map, or search for a place, and this
                        fills in within a few seconds. You will get terrain, land
                        cover, vegetation and rainfall — each with the source it came
                        from attached.
                      </p>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </div>

          {/* The time section sits after the grid, not inside it.

              It used to be the third child of a two-column grid, so it
              auto-placed into column one, row two: the 380px rail column,
              directly beneath the sticky rail rather than below the dossier.
              The charts were rendered down there at 380px wide, which is where
              they looked like something hidden behind the controls. A comment
              here claimed it was full width for several deploys; only the
              structure was wrong. */}
          {response && !responseIsError && analysisSummary && (
            <TimeSection
              rain={(analysisSummary.rainfall?.series ?? []) as RainPoint[]}
              vegetation={(vegetationSeries?.series ?? []) as VegPoint[]}
              windowStart={windowStartISO()}
              windowEnd={windowEndISO()}
              vegetationSource={vegetationSeries?.source}
              thinMonths={vegetationSeries?.thin_months}
              onBuildVegetation={
                activeCacheKey
                  ? () => submitIndicator('vegetation_series')
                  : undefined
              }
              buildingVegetation={pendingIndicator === 'vegetation_series'}
              jobState={
                progress?.indicator === 'vegetation_series' ? progress.state : null
              }
              jobMonths={progress?.months ?? null}
              vegetationNotice={vegetationSeries?.caveat ?? null}
            />
          )}
        </section>
      </main>

      {/* Not hidden in print. The four sources are the provenance for every
          number below, which is exactly what belongs on the foot of a printed
          report, and the line under them is the product's own standard. */}
      <footer className="relative z-10 mt-24 border-t border-rule">
        <div className="mx-auto flex max-w-[1440px] flex-wrap items-center justify-between gap-4 px-5 py-8 sm:px-8">
          <p className="fig text-[0.75rem] text-ink-3">
            NASADEM · ESA WorldCover · Sentinel-2 · ERA5
          </p>
          <p className="text-[0.75rem] text-ink-3">
            Read, not estimated. Every figure keeps its provenance.
          </p>
        </div>
      </footer>
    </div>
  );
}
