import type { Metadata } from "next";
import { Analytics } from "@vercel/analytics/next";
import { Toaster } from "@/components/Toaster";
import "./globals.css";

/* Fonts are self-hosted so the page does not depend on a third-party font CDN. */

export const metadata: Metadata = {
  title: 'GeoContextualize',
  description:
    'Explore terrain, vegetation, rainfall and drought for any area using public satellite and climate data.',
  keywords: 'geography, geospatial, mapping, satellite imagery, geographic context, earth analysis, NDVI, land cover, rainfall, EIA',
  icons: {
    icon: '/icon8.png',
  },
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
