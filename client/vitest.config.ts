import { defineConfig } from 'vitest/config';

// The first JS-level test runner for the client. Unit tests target pure modules
// only (`lib/`) — determining, no clock and no DOM — so the environment stays
// `node` and rendering the page (Leaflet, auth, network) is not a prerequisite
// for a regression guard. Add render tests with `jsdom` + @testing-library if a
// component seam ever needs one.
export default defineConfig({
  css: {
    // Vitest otherwise loads the project's postcss.config.mjs and fails on the
    // Tailwind v4 plugin outside Next's runtime. These tests import no CSS, so
    // an empty plugin set is both safe and enough.
    postcss: { plugins: [] },
  },
  test: {
    environment: 'node',
    include: ['lib/**/*.test.ts', 'components/**/*.test.ts'],
  },
});
