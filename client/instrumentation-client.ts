import posthog from 'posthog-js';

const projectToken = process.env.NEXT_PUBLIC_POSTHOG_PROJECT_TOKEN;

if (projectToken) {
  posthog.init(projectToken, {
    api_host: process.env.NEXT_PUBLIC_POSTHOG_HOST || 'https://us.i.posthog.com',
    defaults: '2026-05-30',
    // Shareable analyses may contain area geometry in query parameters. Send
    // page views with only the pathname below so those values are never sent.
    capture_pageview: false,
    capture_pageleave: false,
    autocapture: false,
    person_profiles: 'identified_only',
  });

  if (typeof window !== 'undefined') {
    posthog.capture('$pageview', {
      $current_url: `${window.location.origin}${window.location.pathname}`,
    });
  }
}

/** Next.js App Router hook for client navigation, with query parameters removed. */
export function onRouterTransitionStart(url: string) {
  if (!projectToken || typeof window === 'undefined') return;
  const pathname = new URL(url, window.location.origin).pathname;
  posthog.capture('$pageview', {
    $current_url: `${window.location.origin}${pathname}`,
  });
}
