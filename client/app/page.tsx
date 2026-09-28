"use client";

import { useState, useRef, useCallback, useEffect, type ReactNode } from 'react';
import dynamic from 'next/dynamic';
import { Search, MapPin, Loader2, Globe, Satellite } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Label } from '@/components/ui/label';
import { FeatureCollection, Geometry, GeoJsonObject } from "geojson";
import CopySummary from '@/components/Copy';
import ClimateChart from '@/components/ClimateChart';
import RainChart from '@/components/RainChart';
import { useToast } from '@/hooks/use-toast';


// Dynamic imports to avoid SSR issues with Leaflet
const MapComponent = dynamic(() => import('@/components/MapComponent'), {
  ssr: false,
  loading: () => (
    <div className="h-[600px] bg-slate-100 rounded-lg flex items-center justify-center">
      <div className="flex items-center space-x-2">
        <Globe className="w-6 h-6 animate-spin text-blue-600" />
        <span className="text-ink-3">Loading satellite map…</span>
      </div>
    </div>
  )
});

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
  { id: 'ndvi', label: 'Vegetation (NDVI)', description: 'Recent Sentinel-2 vegetation condition' },
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
  datasets?: string[];
  mode?: string;
  applied_resolution_m?: Record<string, number>;
}

interface Summary {
  dem?: DemStats | null;
  ndvi?: NdviStats | null;
  landcover?: LandcoverStats | null;
  rainfall?: RainfallResult | null;
  country?: string | null;
  analysis?: AnalysisMetadata;
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
    'This area is too large for land cover, which is read at 10 m. Draw a smaller boundary, or process this area offline at a coarser resolution.',
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
      <summary>Working</summary>
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
   because a card grid reads as a dashboard and this is a survey document. */
function ModuleBlock({ title, description, children }: { title: string; description: string; children?: ReactNode }) {
  return (
    <section className="rule-t py-4">
      <div className="flex items-baseline justify-between gap-4">
        <h3 className="headline">{title}</h3>
        <p className="label">{description}</p>
      </div>
      <div className="mt-3">{children}</div>
    </section>
  );
}

function Headline({ value, caption }: { value: string; caption: string }) {
  return (
    <p className="fig mb-3">
      <span className="text-2xl leading-none">{value}</span>{' '}
      <span className="label">{caption}</span>
    </p>
  );
}

function Figure({ term, value }: { term: string; value: string }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-1">
      <dt className="text-sm text-ink-2">{term}</dt>
      <dd className="fig text-sm">{value}</dd>
    </div>
  );
}

function DatasetResultCard({
  dataset,
  summary,
  submission,
  onSubmit,
  vegetationSeries,
  onSubmitSeries,
  pendingIndicator,
  progress,
}: {
  dataset: DatasetId;
  summary: Summary;
  submission?: SubmissionState;
  onSubmit?: () => void;
  vegetationSeries?: ClimateVegSeries | null;
  onSubmitSeries?: () => void;
  pendingIndicator?: string | null;
  progress?: { state: string; indicator: string } | null;
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
              className="mt-3 border border-ink px-3 py-1.5 text-sm hover:bg-ink hover:text-paper"
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
        <EvidenceLine evidence={rain.evidence} status={rain.status} />
        {!(vegetationSeries?.series?.length ?? 0) && (
          <div className="mt-3 rule-t pt-3">
            <p className="text-sm text-ink-2">
              No vegetation series for this boundary yet.
            </p>
            <p className="fig mt-1 text-xs text-ink-3">
              A monthly NDVI series is computed offline from MODIS. It reads once
              per month, so the wait is minutes rather than seconds, and it does not
              grow with the size of the area.
            </p>
            {!progress && (
              <button
                type="button"
                onClick={() => onSubmitSeries && onSubmitSeries()}
                disabled={pendingIndicator === 'vegetation_series'}
                className="mt-3 border border-ink px-3 py-1.5 text-sm hover:bg-ink hover:text-paper disabled:opacity-50"
              >
                {pendingIndicator === 'vegetation_series' ? 'Queueing…' : 'Build a vegetation series'}
              </button>
            )}
          </div>
        )}
        {(vegetationSeries?.series?.length ?? 0) > 0 ? (
          <ClimateChart
            rain={rain.series || []}
            vegetation={vegetationSeries?.series || []}
            rainNormal={rain.climatology?.standard ?? null}
            vegetationNormal={vegetationSeries?.climatology?.standard ?? null}
            vegetationStatus={vegetationSeries?.status}
            vegetationSource={vegetationSeries?.source}
          />
        ) : (
          <RainChart
            series={rain.series || []}
            normalByMonth={rain.climatology?.monthly_mean_mm}
          />
        )}
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
  // The cache key the backend reported for this analysis, so a submission state
  // can be matched to the area that produced it.
  const [activeCacheKey, setActiveCacheKey] = useState<string | null>(null);
   const [drawnFeatures, setDrawnFeatures] = useState<FeatureCollection<Geometry> | null>(null);
  // Read the share link back on arrival. Without this the link was write-only:
  // it encoded the area, the datasets, the sensor and the window, and nothing
  // ever decoded any of it, so a colleague opened a blank page.
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
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
    const years = Number(params.get('years'));
    if (years === 1 || years === 3 || years === 10 || years === 30) {
      setWindowYears(years);
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

  // Handle search input changes with debouncing
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
        body: JSON.stringify({ geojson, indicator }),
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
        body: JSON.stringify({ geojson, indicator: selectedDatasets.join(',') }),
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
          `${backendUrl}/rainfall/status?cache_key=${cacheKey}&indicator=${indicator}`,
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
          void analyzeRef.current();
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
        datasets: selectedDatasets.join(','),
        sensor: selectedSensor,
        years: String(windowYears),
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
        body: JSON.stringify({ geojson }),
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
        body: JSON.stringify({ geojson }),
      });

      if (!response.ok) {
        const failure = await response.json().catch(() => null);
        // A 413 is not one thing. Only the area-too-large case has an offline
        // route to offer; "too many vertices" and "payload too large" are
        // answered by simplifying the geometry, and sending those to the
        // offline panel told the user to wait for a job that would never help.
        const detail = String(failure?.detail || '');
        const isAreaTooLarge = detail.includes('synchronous limit');
        if (response.status === 413 && geojson && isAreaTooLarge) {
          await loadOfflineOffer(geojson);
          return;
        }
        throw new Error(failure?.detail || `Analysis request failed (${response.status}).`);
      }

      const data = await response.json() as { summary?: Summary };
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
    <div className="min-h-screen bg-paper text-ink">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center mb-4">
            <Globe className="w-12 h-12 text-accent mr-3" />
            <h1 className="text-4xl font-bold text-ink tracking-tight">
              Geo<span className="text-accent">Contextualize</span>
            </h1>
          </div>
          <p className="text-ink-2 text-lg max-w-2xl mx-auto">
            Discover geographical context and insights by selecting any area on Earth.
            Search, draw, and analyze with advanced geospatial tools.
          </p>
        </div>

        {/* Analysis limits */}
        <Alert className="mb-6 bg-amber-50 border-amber-200 max-w-4xl mx-auto">
          <Satellite className="h-4 w-4 text-amber-600" />
          <AlertDescription className="text-amber-800">
            <strong>Analysis limits:</strong> Keep the study-area bounding box within{' '}
            {syncLimitKm2 ?? 100} km². Larger areas can be processed offline; the worker reads
            them at a coarser resolution and says which.
          </AlertDescription>
        </Alert>

        <div className="grid lg:grid-cols-3 gap-8 max-w-7xl mx-auto">
          {/* Left Panel - Search and Controls */}
          <div className="lg:col-span-1 space-y-6">
            {/* Search Section */}
            <Card className="bg-white backdrop-blur border-rule">
              <CardHeader>
                <CardTitle className="text-ink flex items-center">
                  <Search className="w-5 h-5 mr-2" />
                  Location Search
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="relative">
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-ink-3 w-4 h-4" />
                    <Input
                      type="text"
                      placeholder="Search for places..."
                      value={searchQuery}
                      onChange={(e) => handleSearchChange(e.target.value)}
                      className="pl-10 bg-white border-rule text-ink placeholder:text-ink-2"
                      onFocus={() => searchResults.length > 0 && setShowResults(true)}
                    />
                    {isSearching && (
                      <Loader2 className="absolute right-3 top-1/2 transform -translate-y-1/2 text-ink-3 w-4 h-4 animate-spin" />
                    )}
                  </div>
                  
                  {/* Search Results Dropdown */}
                  {showResults && searchResults.length > 0 && (
                    <div className="absolute z-50 w-full mt-1 bg-white rounded-md shadow-lg border border-gray-200 max-h-60 overflow-y-auto">
                      {searchResults.map((result) => (
                        <div
                          key={result.place_id}
                          className="px-4 py-3 hover:bg-gray-50 cursor-pointer border-b border-gray-100 last:border-b-0"
                          onClick={() => handleLocationSelect(result)}
                        >
                          <div className="flex items-start">
                            <MapPin className="w-4 h-4 text-blue-600 mt-0.5 mr-2 flex-shrink-0" />
                            <span className="text-sm text-gray-900 leading-tight">
                              {result.display_name}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Instructions */}
            <Card className="bg-white backdrop-blur border-rule">
              <CardHeader>
                <CardTitle className="text-ink">How to Use</CardTitle>
              </CardHeader>
              <CardContent className="text-ink-2 space-y-3">
                <div className="flex items-start">
                  <div className="w-6 h-6 rounded-full bg-blue-500 text-ink text-xs flex items-center justify-center mr-3 mt-0.5 flex-shrink-0">1</div>
                  <p className="text-sm">Search and select a location to zoom to</p>
                </div>
                <div className="flex items-start">
                  <div className="w-6 h-6 rounded-full bg-blue-500 text-ink text-xs flex items-center justify-center mr-3 mt-0.5 flex-shrink-0">2</div>
                  <p className="text-sm">Upload a GeoJSON file, or draw the area yourself</p>
                </div>
                <div className="flex items-start">
                  <div className="w-6 h-6 rounded-full bg-blue-500 text-ink text-xs flex items-center justify-center mr-3 mt-0.5 flex-shrink-0">3</div>
                  <p className="text-sm">Draw a polygon or rectangle to define the area to analyze</p>
                </div>
                <div className="flex items-start">
                  <div className="w-6 h-6 rounded-full bg-blue-500 text-ink text-xs flex items-center justify-center mr-3 mt-0.5 flex-shrink-0">4</div>
                  <p className="text-sm">Click &quot;Analyze Area&quot; to get geographical context</p>
                </div>
              </CardContent>
            </Card>
            {/* GeoJSON Upload */}
            <Card className="bg-white backdrop-blur border-rule">
              <CardHeader>
                <CardTitle className="text-ink">Upload GeoJSON</CardTitle>
              </CardHeader>
              <CardContent>
                <input
                  type="file"
                  accept=".geojson,application/geo+json,application/json"
                  onChange={handleGeojsonUpload}
                  className="block w-full text-sm text-ink-2 file:mr-4 file:py-2 file:px-4
                            file:rounded-md file:border-0 file:text-sm file:font-semibold
                            file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                />
                <p className="text-xs text-ink-3 mt-2">
                  Upload a <code>.geojson</code> file to define your study area.
                </p>
              </CardContent>
            </Card>
            {/* Options */}
            <Card className="bg-white backdrop-blur border-rule">
              <CardHeader>
                <CardTitle className="text-ink">Options</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="space-y-2">
                    <div className="space-y-2">
                      <div className="flex flex-wrap items-center gap-2">
                        <Label className="text-sm font-medium text-ink-2">Vegetation window</Label>
                        {([1, 3, 10, 30] as const).map((years) => (
                          <button
                            key={years}
                            type="button"
                            onClick={() => setWindowYears(years)}
                            className={`rounded px-2 py-1 text-xs ${
                              windowYears === years
                                ? 'bg-sky-600 text-ink'
                                : 'bg-white text-ink-2 hover:bg-paper'
                            }`}
                          >
                            {years}y
                          </button>
                        ))}
                        <div className="fig flex items-center gap-1 text-xs text-ink-3">
                          <input
                            type="date"
                            value={customStart}
                            max={customEnd || windowEndISO()}
                            onChange={(e) => setCustomStart(e.target.value)}
                            aria-label="Window start"
                            className="border border-rule bg-white px-1 py-0.5 text-ink"
                          />
                          <span>to</span>
                          <input
                            type="date"
                            value={customEnd}
                            min={customStart || undefined}
                            onChange={(e) => setCustomEnd(e.target.value)}
                            aria-label="Window end"
                            className="border border-rule bg-white px-1 py-0.5 text-ink"
                          />
                        </div>
                        <p className="w-full text-xs text-ink-3">
                          This window selects the vegetation period. It also sets how
                          finely precipitation is sampled, but the rainfall series below is
                          always the full monthly record. Elevation and land cover have no
                          time dimension: NASADEM is a static surface and WorldCover a
                          single-date classification, so no window changes them.
                        </p>
                        <Label className="ml-2 text-sm font-medium text-ink-2">Source</Label>
                        <select
                          value={selectedSensor}
                          onChange={(event) => setSelectedSensor(event.target.value)}
                          aria-label="Vegetation source"
                          className="rounded border border-rule bg-white px-2 py-1 text-xs text-ink"
                        >
                          <option value="auto">auto</option>
                          <option value="sentinel-2">Sentinel-2</option>
                          <option value="landsat">Landsat</option>
                          <option value="modis">MODIS</option>
                        </select>
                      </div>
                      <p className="fig text-xs text-ink-2">
                        {windowStartISO()} → {windowEndISO()}
                        {customStart && customEnd ? ' (set)' : ` (${windowYears}y preset)`}
                      </p>
                      {!customRangeValid && (
                        <p className="text-xs text-caution">
                          The start date is after the end date, or in the future.
                        </p>
                      )}
                      <p className="text-xs text-ink-3">
                        auto picks the source whose cloud mask can be trusted, and says why.
                      </p>
                    </div>
                    <Label className="text-sm font-medium text-ink-2">Datasets to Analyze</Label>
                    <div className="space-y-2">
                      {DATASET_OPTIONS.map((dataset) => (
                        <label key={dataset.id} htmlFor={`dataset-${dataset.id}`} className="flex cursor-pointer items-start gap-2 rounded-md px-2 py-1.5 hover:bg-paper">
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
                            className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
                          />
                          <span>
                            <span className="block text-sm text-ink-2">{dataset.label}</span>
                            <span className="block text-xs text-ink-3">{dataset.description}</span>
                          </span>
                        </label>
                      ))}
                    </div>
                  </div>
                  <p className="text-xs text-ink-3 mt-2">
                    All datasets are selected by default. Vegetation analysis may be skipped for larger study areas.
                  </p>
                </div>
              </CardContent>
            </Card>
            
            {/* Analyze Button */}
            <Button
              onClick={handleAnalyze}
              disabled={
                (!boundingBox && !uploadedGeojson && !drawnFeatures?.features.length)
                || selectedDatasets.length === 0
                || isLoading
                || !customRangeValid
              }
              className="w-full bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-ink py-6 text-lg font-semibold disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? (
                <div className="flex items-center">
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
                  Analyzing Area...
                </div>
              ) : (
                <div className="flex items-center">
                  <Satellite className="w-5 h-5 mr-2" />
                  Analyze Selected Area
                </div>
              )}
            </Button>
          </div>

          {/* Right Panel - Map and Results */}
          <div className="lg:col-span-2 space-y-6">
            {/* Map */}
            <Card className="bg-white backdrop-blur border-rule">
              <CardHeader>
                <CardTitle className="text-ink flex items-center">
                  <Globe className="w-5 h-5 mr-2" />
                  Satellite Map
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="rounded-lg overflow-hidden">
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
                        className="absolute bottom-4 right-4 bg-blue-600 hover:bg-blue-700 text-ink px-3 py-2 rounded-md text-sm z-[1000]"
                      >
                        Download GeoJSON
                      </button>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Results */}
            <Card className="bg-white backdrop-blur border-rule">
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="text-ink flex items-center">
                  <MapPin className="w-5 h-5 mr-2" />
                  Analysis Results
                </CardTitle>
                {summaryText && <CopySummary summaryText={summaryText} />}
              </CardHeader>
              <CardContent>
                {shareLinkUnavailable && (
                  <div className="mb-4 text-xs text-ink-3">{shareLinkUnavailable}</div>
                )}
                {shareLink && (
                  <div className="mb-4 flex flex-wrap items-center gap-2 text-xs text-ink-3">
                    <span>Share this analysis:</span>
                    <code className="max-w-sm truncate rounded bg-white px-2 py-1 text-ink-2">
                      {shareLink}
                    </code>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => navigator.clipboard?.writeText(shareLink)}
                    >
                      Copy link
                    </Button>
                    <span className="basis-full" />
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => download('rainfall-series.csv', exportCsv(), 'text/csv')}
                    >
                      Download CSV
                    </Button>
                    <Button
                      size="sm"
                      variant="outline"
                      onClick={() => download('analysis.json', exportJson(), 'application/json')}
                    >
                      Download JSON
                    </Button>
                    <Button size="sm" variant="outline" onClick={() => window.print()}>
                      Print
                    </Button>
                    {activeCacheKey && (
                      // Withdrawal is a first-class action, not a support email.
                      // The app holds a series derived from the submitted
                      // geometry, and the person who submitted it should be the
                      // one who can say to remove it.
                      <Button
                        size="sm"
                        variant="outline"
                        onClick={() => void forgetThisArea()}
                      >
                        Remove this area
                      </Button>
                    )}
                  </div>
                )}
                {offline && (
                  <div className="mb-4 rule-t pt-4">
                    <p className="headline">
                      This area is {formatNumber(offline.areaKm2, 0)} km²
                    </p>
                    <p className="fig mt-1 text-sm text-ink-2">
                      A request handles up to {formatNumber(offline.limitKm2, 0)} km² so
                      it stays inside a few seconds. A larger area is not refused — it
                      is read offline, at a coarser resolution, and you choose whether
                      to wait.
                    </p>
                    <dl className="rows mt-3">
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
                        className="mt-3 border border-ink px-3 py-1.5 text-sm hover:bg-ink hover:text-paper"
                      >
                        Process {offline.plans.length} module
                        {offline.plans.length === 1 ? '' : 's'} offline
                      </button>
                    )}
                    <p className="fig mt-2 text-[10px] text-ink-3">
                      Times are estimates from the measured per-read cost, not
                      guarantees. A coarser grid is a different kind of claim, so the
                      resolution each module will use is listed rather than buried.
                    </p>
                  </div>
                )}
                {analysisWarnings.map((warning) => (
                  <Alert key={warning.message} className="mb-4 border-amber-300 bg-amber-50">
                    <Satellite className="h-4 w-4 text-amber-700" />
                    <AlertDescription className="text-amber-900">
                      <strong>{warning.status === 'skipped' ? 'Vegetation analysis was skipped:' : 'Vegetation analysis note:'}</strong> {warning.message}
                    </AlertDescription>
                  </Alert>
                ))}
                <div className="bg-white rounded-lg p-4 min-h-[200px]">
                  {isLoading ? (
                    <div className="flex items-center justify-center h-48">
                      <div className="text-center">
                        <div className="relative">
                          <Globe className="w-16 h-16 text-accent mx-auto animate-pulse" />
                          <div className="absolute inset-0 flex items-center justify-center">
                            <div className="w-6 h-6 border-2 border-blue-400 border-t-transparent rounded-full animate-spin"></div>
                          </div>
                        </div>
                        <p className="text-ink-2 mt-4">Collecting selected geographic context...</p>
                        <p className="text-ink-3 text-xs mt-2">Remote data sources can take a moment to respond.</p>
                      </div>
                    </div>
                  ) : response ? (
                    <>
                      <div className="text-ink-2 whitespace-pre-wrap break-words text-base leading-relaxed">
                        {responseIsError ? (
                          // An error used to render in the same paragraph style as
                          // a successful result, in the same panel, with the same
                          // weight. Reading "Error: Raster processing timed out" as
                          // a finding is exactly the failure this avoids.
                          <div role="alert" className="rounded border border-red-300 bg-red-50 p-4">
                            <p className="mb-2 flex items-center gap-2 text-sm font-medium text-red-800">
                              <span aria-hidden>!</span>
                              {errorHeadline}
                            </p>
                            {errorDetail && (
                              <p className="text-sm text-red-700">{errorDetail}</p>
                            )}
                            <p className="mt-2 text-xs text-red-600">
                              Nothing was charged for a failed analysis. Try a smaller
                              boundary, or press Analyze again.
                            </p>
                            <Button
                              size="sm"
                              variant="outline"
                              className="mt-3"
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
                        <div className="mt-6 border-t border-white/10 pt-5">
                          <div className="mb-4 flex flex-wrap items-center gap-x-4 gap-y-1 text-sm text-ink-2">
                            <h3 className="font-semibold text-ink">Selected data details</h3>
                            {analysisSummary.country && <span>Country: {analysisSummary.country}</span>}
                            {analysisSummary.analysis?.bbox_area_km2 != null && (
                              <span>Bounding box: {formatNumber(analysisSummary.analysis.bbox_area_km2, 2)} km²</span>
                            )}
                          </div>
                          {analysisSummary.caveats && analysisSummary.caveats.length > 0 && (
                            <div className="mb-4 rounded-md border border-amber-500/30 bg-amber-500/10 p-3">
                              <p className="text-xs font-medium text-amber-200">
                                Before relying on these numbers
                              </p>
                              <ul className="mt-1 list-disc pl-4 text-xs text-amber-100/80">
                                {analysisSummary.caveats.map((caveat, i) => (
                                  <li key={i}>{caveat}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                          <div className="measure">
                            {progress && (
                              <JobProgressLine
                                progress={progress}
                                planned={plan}
                                onDismiss={() => setProgress(null)}
                              />
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
                                  onSubmitSeries={dataset === 'rainfall' ? () => submitIndicator('vegetation_series') : undefined}
                                  pendingIndicator={pendingIndicator}
                                  progress={progress}
                                />
                              ))}
                          </div>
                        </div>
                      )}
                    </>
                  ) : (
                    <div className="flex items-center justify-center h-48 text-ink-3">
                      <div className="text-center">
                        <Satellite className="w-12 h-12 mx-auto mb-3 opacity-50" />
                        <p>Select an area on the map and click &quot;Analyze Area&quot; to see results</p>
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
