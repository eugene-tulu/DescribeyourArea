/** Seconds between `when` and `now`, or null when there is nothing to measure.

   A job's wait is read from the timestamps the job record carries, never from
   when the page mounted. A reader who refreshes mid-job is still shown the wait
   that has actually happened -- restarting the clock at zero on every refresh
   would make a two-minute job look like it had just begun, which is the one
   thing a progress figure must never do.
 */
export function secondsSince(when: string | null | undefined, now: number): number | null {
  if (!when) return null;
  const parsed = Date.parse(when);
  if (Number.isNaN(parsed)) return null;
  return Math.max(0, Math.round((now - parsed) / 1000));
}

/** How much of an estimate has elapsed, clamped to 0..1.

   The estimate is an estimate, so the caller has to keep saying so; what this
   guarantees is only that the fraction is a usable one -- never negative, never
   past completion, and null when there is no denominator to divide by.
 */
export function fractionOfEstimate(elapsed: number | null, estimate: number | null): number | null {
  if (elapsed == null || estimate == null || estimate <= 0) return null;
  return Math.min(1, Math.max(0, elapsed / estimate));
}
