/**
 * The monthly series as one tidy CSV: rainfall and vegetation joined on month.
 *
 * Why this is its own module. The export used to be rainfall only, with no
 * source or unit beyond the column names, so a file opened a week later could
 * not say whether it held CHIRPS or ERA5. Someone who takes the CSV into a GIS
 * or a thesis needs the file to be citable without the screen it came from.
 *
 * Shape: one row per month, rainfall and vegetation side by side, the area id
 * and the source repeated on every row. Repetition is deliberate. A '#' comment
 * header would be tidier, but Excel and ArcGIS both read it as data.
 */

export interface RainRow {
  month: string;
  precip_mm: number | null;
  normal_mm?: number | null;
  anomaly_mm?: number | null;
  anomaly_pct?: number | null;
}

export interface VegRow {
  month: string;
  value: number | null;
  normal?: number | null;
  anomaly?: number | null;
  anomaly_pct?: number | null;
}

export interface CsvInput {
  areaId?: string | null;
  rainfall?: {
    product?: string | null;
    source?: string | null;
    resolution_km?: number | null;
    series?: RainRow[] | null;
  } | null;
  vegetation?: {
    source?: string | null;
    resolution_km?: number | null;
    series?: VegRow[] | null;
  } | null;
}

export const CSV_COLUMNS = [
  'area_id',
  'month',
  'rainfall_mm',
  'rainfall_normal_mm',
  'rainfall_anomaly_mm',
  'rainfall_anomaly_pct',
  'rainfall_product',
  'rainfall_source',
  'rainfall_resolution_km',
  'ndvi',
  'ndvi_normal',
  'ndvi_anomaly',
  'ndvi_anomaly_pct',
  'ndvi_source',
  'ndvi_resolution_km',
] as const;

/** RFC 4180 quoting: sources contain commas ("250 m, 16-day"), so they need it. */
export function csvCell(value: unknown): string {
  if (value == null) return '';
  const text = String(value);
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

/** Rainfall months arrive as 2026-03-01 and vegetation as 2026-03; join on YYYY-MM. */
export function monthKey(month: string): string {
  return String(month ?? '').slice(0, 7);
}

export function buildSeriesCsv(input: CsvInput): string {
  const rain = input.rainfall;
  const veg = input.vegetation;
  const rainRows = new Map<string, RainRow>();
  for (const row of rain?.series ?? []) rainRows.set(monthKey(row.month), row);
  const vegRows = new Map<string, VegRow>();
  for (const row of veg?.series ?? []) vegRows.set(monthKey(row.month), row);

  // A month present in only one series still gets a row, with the other side
  // blank. Dropping it would make the two series look the same length.
  const months = Array.from(new Set([...rainRows.keys(), ...vegRows.keys()])).sort();

  const lines = [CSV_COLUMNS.join(',')];
  for (const month of months) {
    const r = rainRows.get(month);
    const v = vegRows.get(month);
    lines.push([
      input.areaId ?? '',
      month,
      r?.precip_mm,
      r?.normal_mm,
      r?.anomaly_mm,
      r?.anomaly_pct,
      r ? rain?.product : '',
      r ? rain?.source : '',
      r ? rain?.resolution_km : '',
      v?.value,
      v?.normal,
      v?.anomaly,
      v?.anomaly_pct,
      v ? veg?.source : '',
      v ? veg?.resolution_km : '',
    ].map(csvCell).join(','));
  }
  return lines.join('\n');
}
