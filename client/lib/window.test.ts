import { describe, it, expect } from 'vitest';
import { computeStartDate } from './window';

// The end-date + lookback control derives the start rather than reading it. This
// pins the arithmetic that the whole window (and every downstream chart bin and
// vegetation reach) depends on. It exists as a plain module — no clock, no
// browser — precisely so these edge cases can be checked without the page.
describe('computeStartDate', () => {
  it('looks back the given number of whole years', () => {
    expect(computeStartDate('2026-10-09', 10, 'years')).toBe('2016-10-09');
    expect(computeStartDate('2026-10-09', 1, 'years')).toBe('2025-10-09');
    expect(computeStartDate('2026-10-09', 30, 'years')).toBe('1996-10-09');
  });

  it('looks back the given number of months', () => {
    expect(computeStartDate('2023-03-15', 2, 'months')).toBe('2023-01-15');
  });

  it('looks back the given number of days across a month boundary', () => {
    expect(computeStartDate('2026-10-05', 9, 'days')).toBe('2026-09-26');
  });

  it('crosses a year boundary when the lookback exceeds the span', () => {
    // 31 January minus one month is 31 December, not January.
    expect(computeStartDate('2023-01-31', 1, 'months')).toBe('2022-12-31');
    // 5 January minus ten days is late December.
    expect(computeStartDate('2020-01-05', 10, 'days')).toBe('2019-12-26');
  });

  it('zero-pads the month and day so the result stays YYYY-MM-DD', () => {
    expect(computeStartDate('2026-10-09', 1, 'days')).toBe('2026-10-08');
    // A single-digit month/day must not read as 2026-5-26 or 2026-10-6.
    expect(computeStartDate('2026-05-10', 30, 'years')).toBe('1996-05-10');
  });

  it('floors a missing or non-positive amount at 1, matching the control', () => {
    expect(computeStartDate('2026-10-09', 0, 'years')).toBe('2025-10-09');
    expect(computeStartDate('2026-10-09', -5, 'days')).toBe('2026-10-08');
  });
});
