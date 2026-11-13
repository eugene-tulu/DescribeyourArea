import { describe, it, expect } from 'vitest';
import {
  enrichObservations,
  observationsToCsv,
  parseObservationCsv,
  parseObservationGeojson,
  type Observation,
} from './observations';

describe('parseObservationGeojson', () => {
  it('reads points and polygon centroids with attributes', () => {
    const geojson = {
      type: 'FeatureCollection',
      features: [
        {
          type: 'Feature',
          properties: { plot_id: 'P-1', date: '2026-06-15', observer: 'A. Kima', land_use: 'grazing' },
          geometry: { type: 'Point', coordinates: [35.4, -1.5] },
        },
        {
          type: 'Feature',
          properties: { id: 'P-2', timestamp: '2026-07-01' },
          geometry: {
            type: 'Polygon',
            coordinates: [[[35.0, -1.0], [35.2, -1.0], [35.2, -1.2], [35.0, -1.2]]],
          },
        },
      ],
    };
    const result = parseObservationGeojson(geojson);
    expect(result.errors).toEqual([]);
    expect(result.observations).toHaveLength(2);
    expect(result.observations[0]).toMatchObject({
      id: 'P-1',
      timestamp: '2026-06-15',
      lon: 35.4,
      lat: -1.5,
      observer: 'A. Kima',
      type: 'grazing',
    });
    expect(result.observations[1].id).toBe('P-2');
    expect(result.observations[1].lon).toBeCloseTo(35.1, 5);
    expect(result.observations[1].lat).toBeCloseTo(-1.1, 5);
  });

  it('accepts a bare geometry', () => {
    const result = parseObservationGeojson({ type: 'Point', coordinates: [36.0, -2.0] });
    expect(result.observations).toHaveLength(1);
    expect(result.observations[0]).toMatchObject({ lon: 36.0, lat: -2.0 });
  });

  it('reports features without usable coordinates', () => {
    const result = parseObservationGeojson({
      type: 'FeatureCollection',
      features: [{ type: 'Feature', properties: {}, geometry: null }],
    });
    expect(result.observations).toHaveLength(0);
    expect(result.errors[0]).toContain('no geometry');
  });
});

describe('parseObservationCsv', () => {
  it('maps common column names and keeps attributes', () => {
    const csv = [
      'plot_id,longitude,latitude,date,observer,land_use,ndvi',
      'P-1,35.4,-1.5,2026-06-15,A. Kima,grazing,0.42',
      'P-2,35.5,-1.6,2026-07-01,B. Siron,cultivation,0.31',
    ].join('\n');
    const result = parseObservationCsv(csv);
    expect(result.errors).toEqual([]);
    expect(result.observations).toHaveLength(2);
    expect(result.observations[0]).toMatchObject({
      id: 'P-1', lon: 35.4, lat: -1.5, timestamp: '2026-06-15',
      observer: 'A. Kima', type: 'grazing',
    });
    expect(result.observations[0].attributes.ndvi).toBe('0.42');
  });

  it('skips rows without coordinates and names the row', () => {
    const csv = 'lon,lat,date\n35.4,-1.5,2026-06-15\nnot-a-number,-1.6,2026-07-01';
    const result = parseObservationCsv(csv);
    expect(result.observations).toHaveLength(1);
    expect(result.errors[0]).toContain('Row 3');
  });

  it('rejects a CSV without longitude/latitude', () => {
    const result = parseObservationCsv('id,date\nP-1,2026-06-15');
    expect(result.observations).toHaveLength(0);
    expect(result.errors[0]).toContain('longitude/latitude');
  });
});

describe('enrichObservations', () => {
  const summary = {
    dem: { mean: 1800 },
    landcover: { dominant_class: '11', classes: { '11': 62 } },
    ndvi: { mean: 0.42 },
    rainfall: { series: [{ month: '2026-05', precip_mm: 80 }, { month: '2026-06', precip_mm: 120 }] },
    analysis: { bbox: [35.0, -2.0, 36.0, -1.0] as [number, number, number, number] },
  };

  const base: Observation = {
    id: 'P-1', timestamp: '2026-06-15', lon: 35.4, lat: -1.5,
    observer: 'A. Kima', type: 'grazing', attributes: {},
  };

  it('joins the month to the rainfall series and adds the AOI context', () => {
    const [row] = enrichObservations([base], summary);
    expect(row).toMatchObject({
      ndvi: 0.42, landcover: 'Rangeland', rainfall_mm: 120, elevation_m: 1800,
    });
    expect(row.flags).toEqual([]);
  });

  it('flags a month with no rainfall row', () => {
    const [row] = enrichObservations([{ ...base, timestamp: '2026-04-01' }], summary);
    expect(row.rainfall_mm).toBeNull();
    expect(row.flags).toContain('no rainfall for month');
  });

  it('flags observations outside the AOI bbox', () => {
    const [row] = enrichObservations([{ ...base, lon: 40.0, lat: -1.5 }], summary);
    expect(row.flags).toContain('outside AOI bbox');
  });

  it('flags a missing timestamp', () => {
    const [row] = enrichObservations([{ ...base, timestamp: null }], summary);
    expect(row.flags).toContain('no timestamp');
  });

  it('flags when there is no AOI evidence yet', () => {
    const [row] = enrichObservations([base], null);
    expect(row.ndvi).toBeNull();
    expect(row.flags).toContain('no AOI evidence yet');
  });
});

describe('observationsToCsv', () => {
  it('writes the enriched columns, attributes and escaped values', () => {
    const rows = enrichObservations(
      [{ ...{
        id: 'P-1', timestamp: '2026-06-15', lon: 35.4, lat: -1.5,
        observer: 'A. Kima', type: 'grazing', attributes: { note: 'near river, "good"' },
      } }],
      {
        ndvi: { mean: 0.42 }, landcover: { dominant_class: '11' },
        dem: { mean: 1800 },
        rainfall: { series: [{ month: '2026-06', precip_mm: 120 }] },
        analysis: { bbox: [35.0, -2.0, 36.0, -1.0] },
      },
    );
    const csv = observationsToCsv(rows);
    expect(csv.split('\n')[0]).toContain('ndvi,landcover,rainfall_mm,elevation_m,flags,note');
    expect(csv).toContain('"near river, ""good"""');
    expect(csv).toContain('Rangeland');
  });
});