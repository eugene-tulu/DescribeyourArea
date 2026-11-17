export interface RobustStats {
  min?: number | null;
  max?: number | null;
  p05?: number | null;
  p95?: number | null;
  p25?: number | null;
  p75?: number | null;
}

function finite(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

export function numericRange(
  stats: RobustStats | null | undefined,
  fallback: [number, number],
  edges: [number, number] = [-Infinity, Infinity],
): [number, number] {
  if (!stats) return fallback;
  if (finite(stats.p05) && finite(stats.p95) && stats.p95 > stats.p05) {
    return [Math.max(edges[0], stats.p05), Math.min(edges[1], stats.p95)];
  }
  if (finite(stats.p25) && finite(stats.p75) && stats.p75 > stats.p25) {
    // Tukey fences: a robust stretch that does not let one hot pixel flatten
    // the whole display.
    const iqr = stats.p75 - stats.p25;
    return [
      Math.max(edges[0], stats.p25 - 1.5 * iqr),
      Math.min(edges[1], stats.p75 + 1.5 * iqr),
    ];
  }
  if (finite(stats.min) && finite(stats.max) && stats.max > stats.min) {
    return [Math.max(edges[0], stats.min), Math.min(edges[1], stats.max)];
  }
  return fallback;
}

export const DEM_FALLBACK: [number, number] = [0, 3000];
export const NDVI_FALLBACK: [number, number] = [-1, 1];

export function rangeQuery(range: [number, number] | null | undefined): string {
  if (!range) return '';
  const [min, max] = range;
  if (!Number.isFinite(min) || !Number.isFinite(max) || max <= min) return '';
  return `min=${min.toFixed(4)}&max=${max.toFixed(4)}`;
}
