import { describe, it, expect } from 'vitest';
import { buildSeriesCsv, csvCell, monthKey, CSV_COLUMNS } from './exportCsv';

describe('csvCell', () => {
  it('quotes values that contain a comma, a quote or a newline', () => {
    expect(csvCell('MODIS MOD13Q1 (250 m, 16-day)')).toBe('"MODIS MOD13Q1 (250 m, 16-day)"');
    expect(csvCell('say "hi"')).toBe('"say ""hi"""');
    expect(csvCell('a\nb')).toBe('"a\nb"');
  });
  it('leaves null and undefined blank, and zero as zero', () => {
    expect(csvCell(null)).toBe('');
    expect(csvCell(undefined)).toBe('');
    expect(csvCell(0)).toBe('0');
  });
});

describe('monthKey', () => {
  it('joins rainfall-style and vegetation-style months', () => {
    expect(monthKey('2026-03-01')).toBe('2026-03');
    expect(monthKey('2026-03')).toBe('2026-03');
  });
});

describe('buildSeriesCsv', () => {
  const rainfall = {
    product: 'chirps',
    source: 'CHIRPS v2.0 (satellite and gauge, 0.05 degrees)',
    resolution_km: 5.6,
    series: [
      { month: '2026-01-01', precip_mm: 8, normal_mm: 30, anomaly_mm: -22, anomaly_pct: -73.3 },
      { month: '2026-02-01', precip_mm: 40, normal_mm: 35, anomaly_mm: 5, anomaly_pct: 14.3 },
    ],
  };
  const vegetation = {
    source: 'MODIS MOD13Q1 (250 m, 16-day) via Planetary Computer',
    resolution_km: 0.25,
    series: [
      { month: '2026-02', value: 0.26, normal: 0.3, anomaly: -0.04, anomaly_pct: -13.3 },
      { month: '2026-03', value: 0.31 },
    ],
  };

  it('has a fixed header so a file can be parsed without guessing', () => {
    const csv = buildSeriesCsv({ areaId: 'abc', rainfall, vegetation });
    expect(csv.split('\n')[0]).toBe(CSV_COLUMNS.join(','));
  });

  it('joins on month and keeps months present in only one series', () => {
    const rows = buildSeriesCsv({ areaId: 'abc', rainfall, vegetation }).split('\n').slice(1);
    expect(rows.map((r) => r.split(',')[1])).toEqual(['2026-01', '2026-02', '2026-03']);
  });

  it('puts the product and a quoted source on every rainfall row', () => {
    const csv = buildSeriesCsv({ areaId: 'abc', rainfall, vegetation });
    expect(csv).toContain('chirps');
    expect(csv).toContain('"MODIS MOD13Q1 (250 m, 16-day) via Planetary Computer"');
  });

  it('does not claim a source on a month the series does not cover', () => {
    const rows = buildSeriesCsv({ areaId: 'abc', rainfall, vegetation }).split('\n').slice(1);
    // 2026-03 has NDVI but no rainfall: the rainfall source columns stay blank.
    expect(rows[2]).toContain('abc,2026-03,,,,,,,,0.31');
  });

  it('works with rainfall only, as before', () => {
    const csv = buildSeriesCsv({ areaId: 'abc', rainfall });
    expect(csv.split('\n')).toHaveLength(3);
  });

  it('returns just the header when there is nothing to export', () => {
    expect(buildSeriesCsv({})).toBe(CSV_COLUMNS.join(','));
  });
});
