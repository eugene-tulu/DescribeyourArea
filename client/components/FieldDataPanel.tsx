'use client';

import { useRef, useState } from 'react';
import type { ChangeEvent } from 'react';
import {
  observationsToCsv,
  observationsToGeojson,
  parseObservationCsv,
  parseObservationGeojson,
  type EnrichedObservation,
  type Observation,
} from '../lib/observations';

function download(name: string, content: string, type: string): void {
  const blob = new Blob([content], { type });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = name;
  anchor.click();
  URL.revokeObjectURL(url);
}

function fmt(value: number | null, digits = 2): string {
  return value == null || !Number.isFinite(value) ? '—' : value.toFixed(digits);
}

interface FieldDataPanelProps {
  rows: EnrichedObservation[];
  fileName: string | null;
  onObservationsChange: (observations: Observation[], fileName: string | null) => void;
}

export function FieldDataPanel({
  rows,
  fileName,
  onObservationsChange,
}: FieldDataPanelProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [errors, setErrors] = useState<string[]>([]);
  const [busy, setBusy] = useState(false);

  const handleFile = async (file: File) => {
    setBusy(true);
    setErrors([]);
    try {
      const text = await file.text();
      const isCsv = /\.csv$/i.test(file.name);
      const parsed = isCsv
        ? parseObservationCsv(text)
        : parseObservationGeojson(JSON.parse(text));
      onObservationsChange(parsed.observations, file.name);
      setErrors(parsed.errors);
    } catch (error) {
      setErrors([error instanceof Error ? error.message : 'Could not read the file.']);
      onObservationsChange([], file.name);
    } finally {
      setBusy(false);
    }
  };

  const onInput = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) void handleFile(file);
    event.target.value = '';
  };

  const enriched = rows;

  return (
    <section data-print-hide className="mt-6 rounded-xl border border-rule bg-raised/40 p-4 sm:p-5">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h4 className="text-base font-semibold text-ink">Field data</h4>
          <p className="mt-1 max-w-[70ch] text-sm leading-relaxed text-ink-2">
            Upload observations from Survey123, Kobo, ODK or a CSV/GeoJSON. They are joined to
            this area’s NDVI, land cover, rainfall and elevation, with the source and caveats.
          </p>
        </div>
        <span className="label whitespace-nowrap">{rows.length} observations</span>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-2">
        <input
          ref={inputRef}
          type="file"
          accept=".csv,.geojson,.json"
          className="hidden"
          onChange={onInput}
        />
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          disabled={busy}
          className="btn btn-ghost h-9 px-3 text-sm font-semibold"
        >
          {busy ? 'Reading…' : 'Upload CSV / GeoJSON'}
        </button>
        {rows.length > 0 && (
          <>
            <button
              type="button"
              onClick={() => download('field-observations.csv', observationsToCsv(enriched), 'text/csv')}
              className="btn btn-ghost h-9 px-3 text-sm"
            >
              Export CSV
            </button>
            <button
              type="button"
              onClick={() => download('field-observations.geojson', observationsToGeojson(enriched), 'application/geo+json')}
              className="btn btn-ghost h-9 px-3 text-sm"
            >
              Export GeoJSON
            </button>
          </>
        )}
        {fileName && <span className="fig text-[0.75rem] text-ink-3">{fileName}</span>}
      </div>

      {errors.length > 0 && (
        <div className="mt-3 rounded-lg border border-caution/30 bg-caution/10 px-3 py-2 text-[0.75rem] leading-relaxed text-caution">
          {errors.slice(0, 5).join(' · ')}
          {errors.length > 5 ? ` · +${errors.length - 5} more` : ''}
        </div>
      )}

      {rows.length === 0 ? (
        <p className="mt-4 text-sm text-ink-3">
          No observations yet. Upload a file; the points appear on the map and the table below.
        </p>
      ) : (
        <div className="mt-4 max-h-80 overflow-auto rounded-lg border border-rule">
          <table className="w-full border-collapse text-[0.8125rem]">
            <caption className="sr-only">Uploaded observations enriched with the area evidence</caption>
            <thead>
              <tr>
                {['ID', 'Date', 'Type', 'Lon, lat', 'NDVI', 'Land cover', 'Rain (mm)', 'Elev (m)', 'Flags'].map((column) => (
                  <th key={column} scope="col" className="sticky top-0 bg-surface px-2 py-1.5 text-left font-semibold text-ink-2">
                    {column}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {enriched.map((row) => (
                <tr key={row.id} className="border-t border-rule/60">
                  <td className="px-2 py-1 text-ink-2">{row.id}</td>
                  <td className="px-2 py-1 text-ink-3">{row.timestamp?.slice(0, 10) || '—'}</td>
                  <td className="px-2 py-1 text-ink-3">{row.type || '—'}</td>
                  <td className="fig px-2 py-1 text-ink-3">{row.lon.toFixed(4)}, {row.lat.toFixed(4)}</td>
                  <td className="fig px-2 py-1 text-ink">{fmt(row.ndvi, 3)}</td>
                  <td className="px-2 py-1 text-ink-2">{row.landcover || '—'}</td>
                  <td className="fig px-2 py-1 text-ink-2">{fmt(row.rainfall_mm, 1)}</td>
                  <td className="fig px-2 py-1 text-ink-2">{fmt(row.elevation_m, 0)}</td>
                  <td className="px-2 py-1">
                    {row.flags.length === 0 ? (
                      <span className="text-ink-3">ok</span>
                    ) : (
                      <span className="text-caution">{row.flags.join('; ')}</span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <p className="mt-3 text-[0.75rem] leading-relaxed text-ink-3">
        First pass: enrichment is area-level, not per-point. Per-point NDVI sampling and live
        Survey123 / EarthRanger / Kobo connectors are next.
      </p>
    </section>
  );
}