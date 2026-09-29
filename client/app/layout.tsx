import type { Metadata, Viewport } from "next";
import { Analytics } from "@vercel/analytics/next";
import { Toaster } from "@/components/Toaster";
import "./globals.css";

/* Fonts are self-hosted rather than pulled from a CDN at runtime. The audience
   for this is often in a vehicle or on a bad rural connection, and a page that
   falls back to Helvetica because a font CDN was slow is a page that looks like
   every other geospatial form. 123 KB for four faces is a reasonable price for
   the page never flickering through the wrong typeface. */

export const metadata: Metadata = {
  title: 'GeoContextualize — point at any land and get the truth about it',
  description:
    'Terrain, vegetation, rainfall and drought for any boundary you draw on Earth, read from NASADEM, ESA WorldCover, Sentinel-2 and ERA5 — with the provenance attached to every number.',
  keywords: 'geography, geospatial, mapping, satellite imagery, geographic context, earth analysis, NDVI, land cover, rainfall, EIA',
  icons: {
    icon: '/icon8.png',
  },
};

export const viewport: Viewport = {
  themeColor: '#05080a',
  colorScheme: 'dark',
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link
          rel="preload"
          href="/fonts/inter-var-latin.woff2"
          as="font"
          type="font/woff2"
          crossOrigin="anonymous"
        />
        <link
          rel="preload"
          href="/fonts/instrument-serif-latin.woff2"
          as="font"
          type="font/woff2"
          crossOrigin="anonymous"
        />
        <link
          rel="preload"
          href="/fonts/jetbrains-mono-var-latin.woff2"
          as="font"
          type="font/woff2"
          crossOrigin="anonymous"
        />
      </head>
      <body className="antialiased">
        {children}
        <Toaster />
        <Analytics />
      </body>
    </html>
  );
}
