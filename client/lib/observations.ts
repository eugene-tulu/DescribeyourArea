export interface Observation {
  id: string;
  timestamp: string | null;
  lon: number;
  lat: number;
  observer: string | null;
  type: string | null;
  attributes: Record<string, string | number | boolean | null>;
}

export interface EnrichedObservation extends Observation {
  ndvi: number | null;
  landcover: string | null;
  rainfall_mm: number | null;
  elevation_m: number | null;
  flags: string[];
}

export interface EnrichmentSummary {
  dem?: { mean?: number | null } | null;
  landcover?: {
    dominant_class?: string | null;
    classes?: Record<string, number> | null;
  } | null;
  ndvi?: { mean?: number | null } | null;
  rainfall?: {
    series?: Array<{ month?: string | null; precip_mm?: number | null }> | null;
  } | null;
  analysis?: { bbox?: number[] | null } | null;
}

export const LANDCOVER_LABELS: Record<string, string> = {
  "1": "Water",
  "2": "Trees",
  "4": "Flooded Vegetation",
  "5": "Crops",
  "7": "Built Area",
  "8": "Bare Ground",
  "9": "Snow/Ice",
  "10": "Clouds",
  "11": "Rangeland",
};

export interface ParseResult {
  observations: Observation[];
  errors: string[];
}

const TIMESTAMP_KEYS = ["timestamp", "date", "time", "observed_at", "survey_date", "observedat"];
const ID_KEYS = ["id", "plot_id", "uuid", "name"];
const OBSERVER_KEYS = ["observer", "observer_name", "enumerator", "created_by", "collector"];
const TYPE_KEYS = ["type", "land_use", "condition", "class", "landcover"];
const COMBINED_KEYS = ["coordinates", "coordinate", "location", "gps", "point", "latlon", "latlong", "latitudelongitude", "latlng"];

function firstString(source: Record<string, unknown>, keys: string[]): string | null {
  for (const key of keys) {
    const value = source[key];
    if (typeof value === "string" && value.trim()) return value.trim();
    if (typeof value === "number" && Number.isFinite(value)) return String(value);
  }
  return null;
}

function propertiesOf(feature: Record<string, unknown>): Record<string, unknown> {
  const props = feature.properties;
  return props && typeof props === "object" && !Array.isArray(props)
    ? (props as Record<string, unknown>)
    : {};
}

function centroidOf(geometry: Record<string, unknown>): [number, number] | null {
  const type = geometry.type;
  const coordinates = geometry.coordinates;
  if (type === "Point" && Array.isArray(coordinates)
    && typeof coordinates[0] === "number" && typeof coordinates[1] === "number") {
    return [coordinates[0], coordinates[1]];
  }
  // Polygon / MultiPolygon: average the first exterior ring. Good enough for a
  // plot centroid; survey points are the common case.
  let ring: unknown = coordinates;
  if (type === "MultiPolygon" && Array.isArray(coordinates) && Array.isArray(coordinates[0])) {
    ring = coordinates[0];
  }
  if (Array.isArray(ring) && Array.isArray(ring[0])) ring = ring[0];
  if (!Array.isArray(ring)) return null;
  let lon = 0;
  let lat = 0;
  let n = 0;
  for (const position of ring) {
    if (Array.isArray(position) && typeof position[0] === "number" && typeof position[1] === "number") {
      lon += position[0];
      lat += position[1];
      n += 1;
    }
  }
  return n > 0 ? [lon / n, lat / n] : null;
}

function observationFrom(
  feature: Record<string, unknown>,
  fallbackId: string,
): Observation | { error: string } {
  const geometry = feature.geometry;
  if (!geometry || typeof geometry !== "object") return { error: `${fallbackId}: no geometry` };
  const point = centroidOf(geometry as Record<string, unknown>);
  if (!point) return { error: `${fallbackId}: no usable coordinates` };
  const props = propertiesOf(feature);
  const attributes: Record<string, string | number | boolean | null> = {};
  for (const [key, value] of Object.entries(props)) {
    if (typeof value === "string" || typeof value === "number" || typeof value === "boolean" || value === null) {
      attributes[key] = value;
    }
  }
  return {
    id: firstString(props, ID_KEYS) || fallbackId,
    timestamp: firstString(props, TIMESTAMP_KEYS),
    lon: point[0],
    lat: point[1],
    observer: firstString(props, OBSERVER_KEYS),
    type: firstString(props, TYPE_KEYS),
    attributes,
  };
}

export function parseObservationGeojson(input: unknown): ParseResult {
  const observations: Observation[] = [];
  const errors: string[] = [];
  if (!input || typeof input !== "object") {
    return { observations, errors: ["The file is not a GeoJSON object."] };
  }
  const root = input as Record<string, unknown>;
  const type = root.type;
  let features: unknown[] = [];
  if (type === "FeatureCollection" && Array.isArray(root.features)) {
    features = root.features;
  } else if (type === "Feature") {
    features = [root];
  } else if (type && typeof root.coordinates !== "undefined") {
    features = [{ type: "Feature", geometry: root, properties: {} }];
  } else {
    return { observations, errors: ["Expected a FeatureCollection, Feature, or geometry."] };
  }
  features.forEach((item, index) => {
    if (!item || typeof item !== "object") {
      errors.push(`Feature ${index + 1}: not an object.`);
      return;
    }
    const result = observationFrom(item as Record<string, unknown>, `feature-${index + 1}`);
    if ("error" in result) errors.push(result.error);
    else observations.push(result);
  });
  return { observations, errors };
}

function splitCsvLine(line: string): string[] {
  const cells: string[] = [];
  let current = "";
  let quoted = false;
  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (quoted) {
      if (char === '"' && line[i + 1] === '"') {
        current += '"';
        i += 1;
      } else if (char === '"') {
        quoted = false;
      } else {
        current += char;
      }
    } else if (char === '"') {
      quoted = true;
    } else if (char === ",") {
      cells.push(current);
      current = "";
    } else {
      current += char;
    }
  }
  cells.push(current);
  return cells.map((cell) => cell.trim());
}

function normalizedHeader(header: string): string {
  return header.toLowerCase().replace(/[^a-z0-9]/g, "");
}

function isLatHeader(header: string): boolean {
  return header === "lat" || header === "latitude" || header === "y"
    || header.startsWith("lat") || header.endsWith("lat") || header.includes("latitude")
    || header.startsWith("gpslat");
}

function isLonHeader(header: string): boolean {
  return header === "lon" || header === "lng" || header === "long" || header === "x"
    || header.startsWith("lon") || header.endsWith("lon") || header.includes("longitude")
    || header.startsWith("gpslon") || header.startsWith("gpslng");
}

function findColumn(headers: string[], kind: "lat" | "lon"): number {
  const normalized = headers.map(normalizedHeader);
  for (let index = 0; index < normalized.length; index += 1) {
    if (kind === "lat" ? isLatHeader(normalized[index]) : isLonHeader(normalized[index])) return index;
  }
  return -1;
}

function findAny(headers: string[], candidates: string[]): number {
  const lower = headers.map(normalizedHeader);
  for (const candidate of candidates) {
    const index = lower.indexOf(normalizedHeader(candidate));
    if (index >= 0) return index;
  }
  return -1;
}

function parseCoordinateCell(cell: string): [number, number] | null {
  const text = cell.trim();
  if (!text) return null;
  if (/^point\s*\(/i.test(text)) {
    const match = text.match(/point\s*\(\s*([-\d.]+)\s+([-\d.]+)/i);
    if (match) return [Number(match[1]), Number(match[2])];
  }
  const parts = text.split(/[,;\s]+/).filter(Boolean).map(Number);
  if (parts.length < 2 || !Number.isFinite(parts[0]) || !Number.isFinite(parts[1])) return null;
  const [a, b] = parts;
  // Assume "lat, lon"; swap if the first value cannot be a latitude.
  if (a >= -90 && a <= 90 && b >= -180 && b <= 180) return [b, a];
  if (b >= -90 && b <= 90 && a >= -180 && a <= 180) return [a, b];
  return [b, a];
}

/** Parse a table of strings into observations. Shared by CSV and XLSX. */
export function parseObservationRows(headers: string[], rows: string[][]): ParseResult {
  const observations: Observation[] = [];
  const errors: string[] = [];
  const lonIndex = findColumn(headers, "lon");
  const latIndex = findColumn(headers, "lat");
  const combinedIndex = findAny(headers, COMBINED_KEYS);
  if (lonIndex < 0 && latIndex < 0 && combinedIndex < 0) {
    return { observations, errors: ["The file needs longitude/latitude columns (lon/lat, GPS Lat/Long, X/Y, decimal degrees) or a coordinates/location column."] };
  }
  const timeIndex = findAny(headers, TIMESTAMP_KEYS);
  const idIndex = findAny(headers, ID_KEYS);
  const observerIndex = findAny(headers, OBSERVER_KEYS);
  const typeIndex = findAny(headers, TYPE_KEYS);
  rows.forEach((cells, rowOffset) => {
    let lon: number;
    let lat: number;
    if (lonIndex >= 0 && latIndex >= 0) {
      lon = Number(cells[lonIndex]);
      lat = Number(cells[latIndex]);
    } else if (combinedIndex >= 0) {
      const coords = parseCoordinateCell(cells[combinedIndex] ?? "");
      if (!coords) {
        errors.push(`Row ${rowOffset + 2}: invalid coordinates.`);
        return;
      }
      [lon, lat] = coords;
    } else {
      errors.push(`Row ${rowOffset + 2}: missing longitude/latitude.`);
      return;
    }
    if (!Number.isFinite(lon) || !Number.isFinite(lat)) {
      errors.push(`Row ${rowOffset + 2}: invalid longitude/latitude.`);
      return;
    }
    const attributes: Record<string, string | number | boolean | null> = {};
    headers.forEach((header, index) => {
      const value = cells[index];
      if (value !== undefined && value !== "") attributes[header] = value;
    });
    observations.push({
      id: (idIndex >= 0 && cells[idIndex]) || `row-${rowOffset + 2}`,
      timestamp: timeIndex >= 0 && cells[timeIndex] ? cells[timeIndex] : null,
      lon,
      lat,
      observer: observerIndex >= 0 && cells[observerIndex] ? cells[observerIndex] : null,
      type: typeIndex >= 0 && cells[typeIndex] ? cells[typeIndex] : null,
      attributes,
    });
  });
  return { observations, errors };
}

export function parseObservationCsv(text: string): ParseResult {
  const lines = text.split(/\r?\n/).filter((line) => line.trim());
  if (lines.length < 2) {
    return { observations: [], errors: ["The CSV needs a header row and at least one data row."] };
  }
  return parseObservationRows(splitCsvLine(lines[0]), lines.slice(1).map(splitCsvLine));
}

export function enrichObservations(
  observations: Observation[],
  summary: EnrichmentSummary | null,
): EnrichedObservation[] {
  const bbox = summary?.analysis?.bbox ?? null;
  const ndvi = summary?.ndvi?.mean ?? null;
  const elevation = summary?.dem?.mean ?? null;
  const dominant = summary?.landcover?.dominant_class ?? null;
  const landcover = dominant ? LANDCOVER_LABELS[dominant] || dominant : null;
  const series = summary?.rainfall?.series ?? [];
  return observations.map((observation) => {
    const flags: string[] = [];
    if (!observation.timestamp) flags.push("no timestamp");
    const month = observation.timestamp && /^\d{4}-\d{2}/.test(observation.timestamp)
      ? observation.timestamp.slice(0, 7)
      : null;
    if (month && series.length > 0 && !series.some((point) => point.month === month)) {
      flags.push("no rainfall for month");
    }
    if (bbox && bbox.length >= 4) {
      const [minLon, minLat, maxLon, maxLat] = bbox;
      if (observation.lon < minLon || observation.lon > maxLon
        || observation.lat < minLat || observation.lat > maxLat) {
        flags.push("outside AOI bbox");
      }
    }
    if (!summary) flags.push("no AOI evidence yet");
    return {
      ...observation,
      ndvi: ndvi ?? null,
      landcover,
      rainfall_mm: month
        ? series.find((point) => point.month === month)?.precip_mm ?? null
        : null,
      elevation_m: elevation ?? null,
      flags,
    };
  });
}

function csvCell(value: string | number | boolean | null | undefined): string {
  const text = value == null ? "" : String(value);
  return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}

export function observationsToCsv(rows: EnrichedObservation[]): string {
  const attributeKeys = [...new Set(rows.flatMap((row) => Object.keys(row.attributes)))].sort();
  const header = [
    "id", "timestamp", "lon", "lat", "observer", "type",
    "ndvi", "landcover", "rainfall_mm", "elevation_m", "flags",
    ...attributeKeys,
  ];
  const lines = [header.join(",")];
  for (const row of rows) {
    lines.push([
      row.id,
      row.timestamp ?? "",
      row.lon,
      row.lat,
      row.observer ?? "",
      row.type ?? "",
      row.ndvi ?? "",
      row.landcover ?? "",
      row.rainfall_mm ?? "",
      row.elevation_m ?? "",
      row.flags.join("; "),
      ...attributeKeys.map((key) => row.attributes[key] ?? ""),
    ].map(csvCell).join(","));
  }
  return lines.join("\n");
}

export function observationsToGeojson(rows: EnrichedObservation[]): string {
  return JSON.stringify({
    type: "FeatureCollection",
    features: rows.map((row) => ({
      type: "Feature",
      geometry: { type: "Point", coordinates: [row.lon, row.lat] },
      properties: {
        id: row.id,
        timestamp: row.timestamp,
        observer: row.observer,
        type: row.type,
        ndvi: row.ndvi,
        landcover: row.landcover,
        rainfall_mm: row.rainfall_mm,
        elevation_m: row.elevation_m,
        flags: row.flags,
        ...row.attributes,
      },
    })),
  });
}