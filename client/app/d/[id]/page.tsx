'use client';

import { useEffect, useMemo, useState } from 'react';
import Link from 'next/link';
import { useParams } from 'next/navigation';
import type { FeatureCollection, GeoJsonObject } from 'geojson';
import MapComponent from '@/components/MapComponent';
import TimeSection, { type RainPoint, type VegPoint } from '@/components/TimeSection';
import { observationsToCsv, observationsToGeojson, type EnrichedObservation } from '@/lib/observations';

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
  dem?: { mean?: number | null } | null;
  landcover?: { dominant_class?: string | null; dominant_percentage?: number | null } | null;
  ndvi?: { mean?: number | null; sensor?: { label?: string | null } | null } | null;
  rainfall?: { series?: RainPoint[]; product?: string | null } | null;
  analysis?: { bbox_area_km2?: number | null } | null;
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

function download(name: string, content: string, type: string): void {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  URL.revokeObjectURL(url);
}

export default function DashboardPage() {
  const params = useParams();
  const id = typeof params.id === 'string' ? params.id : Array.isArray(params.id) ? params.id[0] : '';
  const [project, setProject] = useState<ProjectView | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [vegetation, setVegetation] = useState<VegPoint[]>([]);

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
    return [
      { id: 'dem', url: `${backendUrl}/tiles/${project.cache_key}/dem/{z}/{x}/{y}.png`, attribution: 'Elevation', opacity: 0.5, zIndex: 400 },
      { id: 'landcover', url: `${backendUrl}/tiles/${project.cache_key}/landcover/{z}/{x}/{y}.png`, attribution: 'Land cover', opacity: 0.6, zIndex: 410 },
    ];
  }, [project?.cache_key]);

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
          <MapComponent
            uploadedGeoJSON={project.geojson as FeatureCollection}
            onBoundingBoxCreated={() => {}}
            onSaveFeatures={() => {}}
            points={project.records.map((record) => ({ id: record.id ?? '', lon: record.lon, lat: record.lat }))}
            tileLayers={tileLayers}
          />
          <p className="mt-2 text-[0.75rem] text-ink-3">
            Points are the saved observations; the raster layers are the AOI’s elevation and land cover.
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