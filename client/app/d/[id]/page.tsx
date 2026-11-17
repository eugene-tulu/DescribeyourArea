'use client';

import { useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import { useParams } from 'next/navigation';
import dynamic from 'next/dynamic';
import type { FeatureCollection, GeoJsonObject } from 'geojson';
import TimeSection, { type RainPoint, type VegPoint } from '@/components/TimeSection';
import MapLegend, { type LegendClass } from '@/components/MapLegend';
import { observationsToCsv, observationsToGeojson, type EnrichedObservation } from '@/lib/observations';
import { BASEMAPS, type BasemapId } from '@/lib/basemaps';
import { LANDCOVER_COLOURS, LANDCOVER_LABELS, LANDCOVER_NAME_TO_CODE } from '@/lib/landcover';
import { DEM_FALLBACK, NDVI_FALLBACK, numericRange, rangeQuery } from '@/lib/mapScales';

interface ProjectRecord {
  id?: string | null;
  lon: number;
  lat: number;
  timestamp?: string | null;
  ndvi?: number | null;
  landcover?: string | null;
  rainfall_mm?: number | null;
  elevation_m?: number | null;
  flags?: string[];
}

interface ProjectSnapshot {
  dem?: {
    mean?: number | null; min?: number | null; max?: number | null;
    p05?: number | null; p95?: number | null; terrain_type?: string | null;
    evidence?: { source?: string | null } | null;
  } | null;
  landcover?: {
    dominant_class?: string | null; dominant_percentage?: number | null;
    classes?: Record<string, number> | null; year?: number | null;
    evidence?: { source?: string | null } | null;
  } | null;
  ndvi?: {
    mean?: number | null; min?: number | null; max?: number | null;
    p25?: number | null; p75?: number | null;
    sensor?: { label?: string | null } | null;
    window?: { start?: string | null; end?: string | null } | null;
  } | null;
  landcover_history?: {
    series?: Array<{
      year?: number | null;
      dominant_class?: string | null;
      classes?: Record<string, number> | null;
    }> | null;
  } | null;
  rainfall?: { series?: RainPoint[]; product?: string | null } | null;
  analysis?: { bbox_area_km2?: number | null } | null;
}

type ThematicLayer = 'dem' | 'landcover' | 'ndvi' | 'landcover_history';

interface LegendData {
  title: string;
  subtitle?: string | null;
  gradient?: string;
  minLabel?: string;
  maxLabel?: string;
  classes?: LegendClass[];
}

interface ProjectView {
  id: string;
  name: string;
  type: string;
  geojson: GeoJsonObject;
  bbox_area_km2?: number | null;
  window_start?: string | null;
  window_end?: string | null;
  cache_key?: string | null;
  records: ProjectRecord[];
  snapshot?: ProjectSnapshot | null;
}

function formatNumber(value: number | null | undefined, digits = 1): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—';
  return new Intl.NumberFormat(undefined, { maximumFractionDigits: digits }).format(value);
}

function landcoverClassesFromCodes(classes: Record<string, number>): LegendClass[] {
  return Object.entries(classes)
    .filter(([, percentage]) => Number.isFinite(percentage) && percentage > 0.05)
    .sort(([, first], [, second]) => second - first)
    .map(([code, percentage]) => ({
      label: LANDCOVER_LABELS[code] || code,
      color: LANDCOVER_COLOURS[code] || '#6b7280',
      value: `${formatNumber(percentage, 1)}%`,
    }));
}

function landcoverClassesFromNames(classes: Record<string, number>): LegendClass[] {
  return Object.entries(classes)
    .filter(([, percentage]) => Number.isFinite(percentage) && percentage > 0.05)
    .sort(([, first], [, second]) => second - first)
    .map(([name, percentage]) => {
      const code = LANDCOVER_NAME_TO_CODE[name.toLowerCase()];
      return {
        label: name,
        color: code ? LANDCOVER_COLOURS[code] : '#6b7280',
        value: `${formatNumber(percentage, 1)}%`,
      };
    });
}

function landcoverClasses(classes: Record<string, number>): LegendClass[] {
  const numericCodes = Object.keys(classes).some((key) => /^\d+$/.test(key));
  return numericCodes ? landcoverClassesFromCodes(classes) : landcoverClassesFromNames(classes);
}

function boundsOf(input: GeoJsonObject): [number, number, number, number] | undefined {
  let west = Infinity, south = Infinity, east = -Infinity, north = -Infinity;
  const walk = (value: unknown): void => {
    if (!Array.isArray(value)) return;
    if (typeof value[0] === 'number' && typeof value[1] === 'number') {
      west = Math.min(west, value[0]); east = Math.max(east, value[0]);
      south = Math.min(south, value[1]); north = Math.max(north, value[1]);
      return;
    }
    value.forEach(walk);
  };
  const visit = (geojson: GeoJsonObject): void => {
    const value = geojson as unknown as {
      type: string; features?: Array<{ geometry?: GeoJsonObject | null }>;
      geometry?: GeoJsonObject | null; geometries?: GeoJsonObject[]; coordinates?: unknown;
    };
    if (value.type === 'FeatureCollection') value.features?.forEach((feature) => feature.geometry && visit(feature.geometry));
    else if (value.type === 'Feature') { if (value.geometry) visit(value.geometry); }
    else if (value.type === 'GeometryCollection') value.geometries?.forEach(visit);
    else walk(value.coordinates);
  };
  visit(input);
  return Number.isFinite(west) ? [west, south, east, north] : undefined;
}

function download(name: string, content: string, type: string): void {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  URL.revokeObjectURL(url);
}

const MapComponent = dynamic(() => import('@/components/MapComponent'), {
  ssr: false,
  loading: () => <div className="h-[440px] w-full rounded-xl border border-rule bg-raised sm:h-[540px]" />,
});

export default function DashboardPage() {
  const params = useParams();
  const id = typeof params.id === 'string' ? params.id : Array.isArray(params.id) ? params.id[0] : '';
  const [project, setProject] = useState<ProjectView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [vegetation, setVegetation] = useState<VegPoint[]>([]);
  const [basemap, setBasemap] = useState<BasemapId>('hybrid');
  const [terrain3d, setTerrain3d] = useState(false);
  const [layerOpacity, setLayerOpacity] = useState(0.85);
  const [activeLayer, setActiveLayer] = useState<ThematicLayer>('landcover');
  const [historyYear, setHistoryYear] = useState<number | null>(null);
  const historyYears = useMemo(
    () => (project?.snapshot?.landcover_history?.series ?? [])
      .map((row) => row.year)
      .filter((year): year is number => typeof year === 'number'),
    [project?.snapshot?.landcover_history?.series],
  );
  const demStretch = useMemo(
    () => project?.snapshot?.dem ? numericRange(project.snapshot.dem, DEM_FALLBACK) : null,
    [project?.snapshot?.dem],
  );
  const ndviStretch = useMemo(
    () => project?.snapshot?.ndvi
      ? numericRange(project.snapshot.ndvi, NDVI_FALLBACK, [-0.1, 1])
      : null,
    [project?.snapshot?.ndvi],
  );
  useEffect(() => {
    if (historyYear === null && historyYears.length) {
      setHistoryYear(historyYears[historyYears.length - 1]);
    }
  }, [historyYear, historyYears]);

  useEffect(() => {
    if (!id) return;
    fetch(`/api/v1/projects/${id}`)
      .then((response) => (response.ok ? response.json() : Promise.reject(new Error('Project not found'))))
      .then((data: ProjectView) => setProject(data))
      .catch((err: Error) => setError(err.message));
  }, [id]);

  useEffect(() => {
    if (!project?.cache_key || !project.window_start || !project.window_end) return;
    const url = `/api/rainfall?cache_key=${project.cache_key}&indicator=vegetation_series&window_start=${project.window_start}&window_end=${project.window_end}`;
    fetch(url)
      .then((response) => (response.ok ? response.json() : null))
      .then((data: { rainfall?: { series?: VegPoint[] } } | null) => setVegetation(data?.rainfall?.series ?? []))
      .catch(() => setVegetation([]));
  }, [project?.cache_key, project?.window_start, project?.window_end]);

  const rain = useMemo(() => project?.snapshot?.rainfall?.series ?? [], [project?.snapshot?.rainfall?.series]);
  const rainTotal = rain.reduce((sum, month) => sum + (month.precip_mm ?? 0), 0);
  const rainNormal = rain.reduce((sum, month) => sum + (month.normal_mm ?? 0), 0);

  const tileLayers = useMemo(() => {
    if (!project?.cache_key) return undefined;
    const backendUrl = (process.env.NEXT_PUBLIC_BACKEND_URL || '/api').replace(/\/$/, '');
    const bounds = boundsOf(project.geojson);
    const tileUrl = (indicator: string, year?: number,
                     stretch?: [number, number] | null) => {
      const query: string[] = [];
      if (year) query.push(`year=${year}`);
      const range = rangeQuery(stretch);
      if (range) query.push(range);
      return `${backendUrl}/tiles/${project.cache_key}/${indicator}/{z}/{x}/{y}.png${query.length ? `?${query.join('&')}` : ''}`;
    };
    const layers: Array<{
      id: string; url: string; attribution: string;
      opacity?: number; zIndex?: number; bounds?: [number, number, number, number];
    }> = [];
    if (activeLayer === 'dem') {
      layers.push({ id: 'dem', url: tileUrl('dem', undefined, demStretch), attribution: 'Elevation', opacity: layerOpacity, zIndex: 400, bounds });
    }
    if (activeLayer === 'landcover') {
      layers.push({ id: 'landcover', url: tileUrl('landcover'), attribution: 'Land cover', opacity: layerOpacity, zIndex: 410, bounds });
    }
    if (activeLayer === 'ndvi') {
      layers.push({ id: 'ndvi', url: tileUrl('ndvi', undefined, ndviStretch), attribution: 'NDVI', opacity: layerOpacity, zIndex: 420, bounds });
    }
    if (activeLayer === 'landcover_history' && historyYear) {
      layers.push({
        id: 'lch', url: tileUrl('landcover_history', historyYear),
        attribution: `Land cover ${historyYear}`, opacity: layerOpacity, zIndex: 430, bounds,
      });
    }
    return layers;
  }, [project?.cache_key, project?.geojson, layerOpacity, activeLayer, historyYear, demStretch, ndviStretch]);

  const activeLegend = useMemo<LegendData | null>(() => {
    if (activeLayer === 'dem' && demStretch) {
      return {
        title: 'Elevation',
        subtitle: project?.snapshot?.dem?.evidence?.source?.split('(')[0]?.trim() || 'Copernicus DEM GLO-30',
        gradient: 'linear-gradient(90deg,#1a3a5c,#2e6b3a,#78964c,#a8844a,#cdbea0,#f5f5f5)',
        minLabel: `${formatNumber(demStretch[0], 0)} m`,
        maxLabel: `${formatNumber(demStretch[1], 0)} m`,
      };
    }
    if (activeLayer === 'ndvi' && ndviStretch) {
      const sensor = project?.snapshot?.ndvi?.sensor?.label?.split('(')[0]?.trim();
      const start = project?.snapshot?.ndvi?.window?.start?.slice(0, 10);
      const end = project?.snapshot?.ndvi?.window?.end?.slice(0, 10);
      return {
        title: 'NDVI',
        subtitle: [sensor || 'Vegetation index', start && end ? `${start} → ${end}` : null].filter(Boolean).join(' · '),
        gradient: 'linear-gradient(90deg,#78542e,#c49e5a,#e7e0a0,#b4da8c,#67b761,#2b8144,#084d2c)',
        minLabel: formatNumber(ndviStretch[0], 2),
        maxLabel: formatNumber(ndviStretch[1], 2),
      };
    }
    if (activeLayer === 'landcover') {
      const classes = project?.snapshot?.landcover?.classes;
      if (!classes || Object.keys(classes).length === 0) return null;
      return {
        title: `Land cover${project?.snapshot?.landcover?.year ? ` · ${project.snapshot.landcover.year}` : ''}`,
        subtitle: 'Esri 10 m annual classification',
        classes: landcoverClasses(classes),
      };
    }
    if (activeLayer === 'landcover_history' && historyYear) {
      const row = project?.snapshot?.landcover_history?.series?.find((entry) => entry.year === historyYear);
      if (!row) return null;
      return {
        title: `Land cover · ${historyYear}`,
        subtitle: 'Esri 10 m annual classification',
        classes: landcoverClassesFromNames(row.classes ?? {}),
      };
    }
    return null;
  }, [activeLayer, project, historyYear, demStretch, ndviStretch]);

  const rows: EnrichedObservation[] = (project?.records ?? []).map((record) => ({
    id: record.id ?? '',
    timestamp: record.timestamp ?? null,
    lon: record.lon,
    lat: record.lat,
    observer: null,
    type: null,
    attributes: {},
    ndvi: record.ndvi ?? null,
    landcover: record.landcover ?? null,
    rainfall_mm: record.rainfall_mm ?? null,
    elevation_m: record.elevation_m ?? null,
    flags: record.flags ?? [],
  }));

  if (error) {
    return (
      <main className="mx-auto max-w-[1200px] px-5 py-16">
        <h1 className="text-2xl font-semibold text-ink">Dashboard not found</h1>
        <p className="mt-2 text-sm text-ink-3">{error}</p>
      </main>
    );
  }
  if (!project) {
    return (
      <main className="mx-auto max-w-[1200px] px-5 py-16">
        <span className="label">Loading the dashboard</span>
      </main>
    );
  }

  const stats = [
    { label: 'Area', value: `${formatNumber(project.bbox_area_km2, 0)} km²`, caption: project.type },
    { label: 'Elevation', value: `${formatNumber(project.snapshot?.dem?.mean, 0)} m`, caption: 'mean' },
    {
      label: 'Land cover',
      value: project.snapshot?.landcover?.dominant_class ?? '—',
      caption: `${formatNumber(project.snapshot?.landcover?.dominant_percentage, 1)}% dominant`,
    },
    { label: 'NDVI', value: formatNumber(project.snapshot?.ndvi?.mean, 3), caption: 'mean of median composite' },
    {
      label: 'Rainfall',
      value: `${formatNumber(rainTotal, 0)} mm`,
      caption: rain.length > 0 ? `${rain.length} months · ${formatNumber(rainNormal, 0)} mm normal` : 'no months',
    },
  ];

  return (
    <main className="mx-auto max-w-[1440px] px-5 py-8 sm:px-8">
      <header className="flex flex-wrap items-end justify-between gap-4 border-b border-rule pb-5">
        <div>
          <Link href="/" className="label text-ink-3">GeoContextualize</Link>
          <h1 className="mt-2 text-3xl font-semibold tracking-[-0.02em] text-ink">{project.name}</h1>
          <p className="mt-1 text-sm text-ink-3">
            {project.type} · {project.window_start?.slice(0, 10)} → {project.window_end?.slice(0, 10)} · {project.records.length} observations
          </p>
        </div>
        <div className="flex items-center gap-2" data-print-hide>
          <Link href="/" className="btn btn-ghost h-9 px-3 text-sm">Workspace</Link>
          <button type="button" onClick={() => window.print()} className="btn btn-ghost h-9 px-3 text-sm">Print</button>
        </div>
      </header>

      <dl className="mt-5 grid grid-cols-2 gap-px overflow-hidden rounded-xl border border-rule bg-rule lg:grid-cols-5">
        {stats.map((stat) => (
          <div key={stat.label} className="bg-surface px-4 py-3.5">
            <dt className="text-[0.75rem] uppercase tracking-[0.12em] text-ink-3">{stat.label}</dt>
            <dd className="fig mt-1 text-xl text-ink">{stat.value}</dd>
            <dd className="mt-0.5 text-[0.75rem] text-ink-3">{stat.caption}</dd>
          </div>
        ))}
      </dl>

      <section className="mt-6 grid gap-6 lg:grid-cols-[1.2fr_0.8fr]">
        <div>
          <div className="mb-3 flex flex-wrap items-center gap-2" data-print-hide>
            <label className="flex items-center gap-1.5 text-[0.75rem] text-ink-3">
              Basemap
              <select
                value={basemap}
                onChange={(event) => setBasemap(event.target.value as BasemapId)}
                className="h-7 rounded-lg border border-rule bg-raised px-2 text-[0.75rem] text-ink-2"
              >
                {Object.entries(BASEMAPS).map(([id, base]) => (
                  <option key={id} value={id}>{base.label}</option>
                ))}
              </select>
            </label>
            <label className="flex cursor-pointer items-center gap-1.5 rounded-full border border-rule bg-surface px-2.5 py-1 text-[0.75rem] text-ink-2">
              <input
                type="checkbox"
                checked={terrain3d}
                onChange={(event) => setTerrain3d(event.target.checked)}
                className="h-3.5 w-3.5 accent-signal"
              />
              3D terrain
            </label>
            <label className="flex items-center gap-1.5 text-[0.75rem] text-ink-3">
              Opacity
              <input
                type="range"
                min={20}
                max={100}
                step={5}
                value={Math.round(layerOpacity * 100)}
                onChange={(event) => setLayerOpacity(Number(event.target.value) / 100)}
                className="h-1 w-24 accent-signal"
              />
            </label>
            <span className="label text-ink-3">Thematic layer</span>
            <div role="group" aria-label="Thematic layer" className="flex flex-wrap items-center gap-1.5">
              {([
                ['dem', 'Elevation'],
                ['landcover', 'Land cover'],
                ['ndvi', 'NDVI'],
                ['landcover_history', 'History'],
              ] as const).map(([id, label]) => (
                <button
                  key={id}
                  type="button"
                  aria-pressed={activeLayer === id}
                  disabled={id === 'landcover_history' && historyYears.length === 0}
                  onClick={() => setActiveLayer(id)}
                  className={`rounded-full border px-2.5 py-1 text-[0.75rem] transition ${
                    activeLayer === id
                      ? 'border-signal bg-signal/15 text-signal'
                      : 'border-rule bg-surface text-ink-2 hover:border-ink-3'
                  } disabled:cursor-not-allowed disabled:opacity-40`}
                >
                  {label}
                </button>
              ))}
            </div>
            {activeLayer === 'landcover_history' && historyYears.length > 0 && (
              <select
                value={historyYear ?? historyYears[historyYears.length - 1]}
                onChange={(event) => setHistoryYear(Number(event.target.value))}
                aria-label="Land-cover history year"
                className="h-7 rounded-lg border border-rule bg-raised px-2 text-[0.75rem] text-ink-2"
              >
                {historyYears.map((year) => <option key={year} value={year}>{year}</option>)}
              </select>
            )}
          </div>
          <div className="relative">
            <MapComponent
              aoiGeoJSON={project.geojson as FeatureCollection}
              onBoundingBoxCreated={() => {}}
              onSaveFeatures={() => {}}
              points={project.records.map((record) => ({ id: record.id ?? '', lon: record.lon, lat: record.lat }))}
              tileLayers={tileLayers}
              basemap={basemap}
              terrain3d={terrain3d}
            />
            {activeLegend && <MapLegend {...activeLegend} />}
          </div>
          <p className="mt-2 text-[0.75rem] text-ink-3">
            Points are the saved observations; the raster layer is the selected thematic dataset, clipped to the AOI.
          </p>
        </div>
        <div className="rounded-xl border border-rule bg-raised/40 p-4">
          <h2 className="text-base font-semibold text-ink">Rainfall and vegetation</h2>
          <p className="mt-1 text-sm leading-relaxed text-ink-3">
            The monthly series behind the KPI strip, with the same window and normal used in the workspace.
          </p>
          <div className="mt-4">
            <TimeSection
              rain={rain}
              vegetation={vegetation}
              windowStart={project.window_start ?? ''}
              windowEnd={project.window_end ?? ''}
              rainfallProduct={project.snapshot?.rainfall?.product ?? 'rainfall'}
              vegetationSource={project.snapshot?.ndvi?.sensor?.label ?? 'MODIS'}
            />
          </div>
        </div>
      </section>

      <section className="mt-8">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h2 className="text-base font-semibold text-ink">Observations</h2>
            <p className="mt-1 text-sm text-ink-3">Every collected point with its remote-sensing context and flags.</p>
          </div>
          <div className="flex items-center gap-2" data-print-hide>
            <button
              type="button"
              onClick={() => download('observations.csv', observationsToCsv(rows), 'text/csv')}
              className="btn btn-ghost h-9 px-3 text-sm"
            >
              Export CSV
            </button>
            <button
              type="button"
              onClick={() => download('observations.geojson', observationsToGeojson(rows), 'application/geo+json')}
              className="btn btn-ghost h-9 px-3 text-sm"
            >
              Export GeoJSON
            </button>
          </div>
        </div>
        <div className="mt-3 max-h-[32rem] overflow-auto rounded-xl border border-rule">
          <table className="w-full border-collapse text-[0.8125rem]">
            <thead>
              <tr>
                {['ID', 'Date', 'Lon, lat', 'Elevation (m)', 'Land cover', 'NDVI', 'Rain (mm)', 'Flags'].map((column) => (
                  <th key={column} scope="col" className="sticky top-0 bg-surface px-3 py-2 text-left font-semibold text-ink-2">{column}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={row.id} className="border-t border-rule/60">
                  <td className="px-3 py-1.5 text-ink-2">{row.id}</td>
                  <td className="px-3 py-1.5 text-ink-3">{row.timestamp?.slice(0, 10) ?? '—'}</td>
                  <td className="fig px-3 py-1.5 text-ink-3">{row.lon.toFixed(4)}, {row.lat.toFixed(4)}</td>
                  <td className="fig px-3 py-1.5 text-ink-2">{formatNumber(row.elevation_m, 0)}</td>
                  <td className="px-3 py-1.5 text-ink-2">{row.landcover ?? '—'}</td>
                  <td className="fig px-3 py-1.5 text-ink">{formatNumber(row.ndvi, 3)}</td>
                  <td className="fig px-3 py-1.5 text-ink-2">{formatNumber(row.rainfall_mm, 0)}</td>
                  <td className="px-3 py-1.5">{row.flags.length === 0 ? <span className="text-ink-3">ok</span> : <span className="text-caution">{row.flags.join('; ')}</span>}</td>
                </tr>
              ))}
              {rows.length === 0 && (
                <tr><td colSpan={8} className="px-3 py-6 text-center text-ink-3">No observations saved with this project.</td></tr>
              )}
            </tbody>
          </table>
        </div>
      </section>
    </main>
  );
}
