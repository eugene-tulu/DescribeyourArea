import { describe, it, expect } from 'vitest';
import { fractionOfEstimate, secondsSince } from './elapsed';

// The wait a job has actually endured, not the wait the page has been open.
describe('secondsSince', () => {
  const submit = '2026-10-09T12:00:00Z';
  const twoMinutesLater = Date.parse('2026-10-09T12:02:00Z');

  it('measures from the timestamp, not from mount', () => {
    expect(secondsSince(submit, twoMinutesLater)).toBe(120);
  });

  it('never reports a negative wait for a clock skewed backwards', () => {
    expect(secondsSince(submit, Date.parse('2026-10-09T11:00:00Z'))).toBe(0);
  });

  it('is null when there is no timestamp to measure from', () => {
    expect(secondsSince(null, twoMinutesLater)).toBeNull();
    expect(secondsSince(undefined, twoMinutesLater)).toBeNull();
    expect(secondsSince('not a date', twoMinutesLater)).toBeNull();
  });
});

describe('fractionOfEstimate', () => {
  it('is a usable fraction of the estimate', () => {
    expect(fractionOfEstimate(30, 120)).toBeCloseTo(0.25);
  });

  it('clamps to completion rather than overrunning past 100%', () => {
    expect(fractionOfEstimate(500, 120)).toBe(1);
  });

  it('clamps to zero rather than going backwards', () => {
    expect(fractionOfEstimate(-5, 120)).toBe(0);
  });

  it('is null when there is no denominator to divide by', () => {
    expect(fractionOfEstimate(30, null)).toBeNull();
    expect(fractionOfEstimate(30, 0)).toBeNull();
    expect(fractionOfEstimate(null, 120)).toBeNull();
  });
});
