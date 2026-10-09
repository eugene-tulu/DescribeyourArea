// The study-area window, derived rather than entered.
//
// The reader sets an end date and a lookback (an amount and a unit); the start
// is computed. Keeping this as a pure function — no clocks, no React, no time
// zone — is what lets it be unit-tested directly, which is the only place its
// edge cases (month-end rollover, leap days, the 1-of-anything floor) can be
// pinned without standing up the whole page.

export type WindowUnit = 'years' | 'months' | 'days';

/**
 * The inclusive start of a window ending at `endISO`, `amount` `unit`s earlier.
 *
 * The end is parsed as local midnight so the arithmetic runs in the same frame
 * the reader's date is in; the result is formatted from the local fields for the
 * same reason. A missing or non-positive amount floors at 1, matching the
 * control's own floor.
 */
export function computeStartDate(endISO: string, amount: number, unit: WindowUnit): string {
  const d = new Date(`${endISO}T00:00:00`);
  const safeAmount = Math.max(1, Number(amount) || 1);
  if (unit === 'months') d.setMonth(d.getMonth() - safeAmount);
  else if (unit === 'days') d.setDate(d.getDate() - safeAmount);
  else d.setFullYear(d.getFullYear() - safeAmount);
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
}
