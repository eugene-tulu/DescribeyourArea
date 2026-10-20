'use client';

import posthog from 'posthog-js';

/** Capture an app event only when PostHog is configured in the deployment. */
export function trackEvent(
  event: string,
  properties?: Record<string, string | number | boolean | string[] | null>,
) {
  if (process.env.NEXT_PUBLIC_POSTHOG_PROJECT_TOKEN) {
    posthog.capture(event, properties);
  }
}
