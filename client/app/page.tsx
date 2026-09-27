"use client";

import { useState, useRef, useCallback, type ReactNode } from 'react';
import dynamic from 'next/dynamic';
import { Search, MapPin, Loader2, Globe, Satellite } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Label } from '@/components/ui/label';
import { FeatureCollection, Geometry, GeoJsonObject } from "geojson";
import CopySummary from '@/components/Copy';
import { useToast } from '@/hooks/use-toast';


// Dynamic imports to avoid SSR issues with Leaflet
const MapComponent = dynamic(() => import('@/components/MapComponent'), {
  ssr: false,
  loading: () => (
    <div className="h-[600px] bg-slate-100 rounded-lg flex items-center justify-center">
      <div className="flex items-center space-x-2">
        <Globe className="w-6 h-6 animate-spin text-blue-600" />
        <span className="text-slate-600">Loading satellite map...</span>
      </div>
    </div>
  )
});

interface SearchResult {
  place_id: number;
  display_name: string;
  lat: string;
  lon: string;
  boundingbox: [string, string, string, string];
}

interface BoundingBox {
  north: number;
  south: number;
  east: number;
  west: number;
}

interface AnalysisWarning {
  message: string;
  status?: string;
}

type DatasetId = 'dem' | 'landcover' | 'ndvi' | 'rainfall';

const DATASET_OPTIONS: Array<{
  id: DatasetId;
  label: string;
  description: string;
}> = [
  { id: 'dem', label: 'Elevation & terrain', description: 'Elevation range and terrain variation' },
  { id: 'landcover', label: 'Land cover', description: 'ESA WorldCover composition' },
  { id: 'ndvi', label: 'Vegetation (NDVI)', description: 'Recent Sentinel-2 vegetation condition' },
  { id: 'rainfall', label: 'Rainfall & drought', description: 'ERA5 monthly totals and anomaly vs normal' },
];

const LANDCOVER_LABELS: Record<string, string> = {
  "10": "Tree cover",
  "20": "Shrubland",
  "30": "Grassland",
  "40": "Cropland",
  "50": "Built-up areas",
  "60": "Bare or sparse vegetation",
  "70": "Snow & ice",
  "80": "Permanent water bodies",
  "90": "Herbaceous wetlands",
  "95": "Mangroves",
  "100": "Moss & lichen",
};

interface DemStats {
  mean?: number;
  min?: number;
  max?: number;
  std?: number;
  elevation_range_m?: number;
  terrain_type?: string;
  error?: string;
}

interface SensorProvenance {
  id: string;
  label: string;
  collection: string;
  native_resolution_m: number;
  archive_start: string;
  ndvi: string;
  cloud_mask: string;
  scale?: number;
  offset?: number;
  mask_authoritative?: boolean;
}

interface NdviStats {
  mean?: number;
  min?: number;
  max?: number;
  std?: number;
  p25?: number;
  p75?: number;
  scene_count?: number;
  resolution_m?: number;
  method?: string;
  status?: string;
  warning?: string;
  source?: string;
  valid_pixel_count?: number;
  valid_pixel_fraction?: number;
  scenes_examined?: number;
  sensor?: SensorProvenance;
  sensor_reason?: string;
  window?: { start?: string; end?: string };
  scene_ids?: string[];
  scene_dates?: string[];
}

interface LandcoverStats {
  classes?: Record<string, number>;
  dominant_class?: string;
  dominant_percentage?: number;
  error?: string;
  [key: string]: unknown;
}

interface RainfallSummary {
  latest_month?: string;
  latest_precip_mm?: number;
  latest_anomaly_pct?: number | null;
  driest_month?: { month: string; precip_mm: number };
  wettest_month?: { month: string; precip_mm: number };
  trailing_12m?: {
    ending: string; months: number; precip_mm: number;
    normal_mm: number; anomaly_mm: number; anomaly_pct: number | null;
  } | null;
  suspect_months?: string[];
}

interface RainfallResult {
  status: 'ok' | 'not_computed';
  indicator?: string;
  source?: string;
  doi?: string;
  license?: string;
  retrieved?: string;
  resolution_km?: number;
  grid_cells?: number;
  processing_version?: string;
  climatology?: {
    standard: string;
    annual_mean_mm: number | null;
  };
  series?: Array<{ month: string; precip_mm: number; normal_mm: number; anomaly_pct: number | null }>;
  summary?: RainfallSummary;
  message?: string;
}

interface AnalysisMetadata {
  bbox_area_km2?: number;
  datasets?: string[];
  mode?: string;
}

interface Summary {
  dem?: DemStats | null;
  ndvi?: NdviStats | null;
  landcover?: LandcoverStats | null;
  rainfall?: RainfallResult | null;
  country?: string | null;
  analysis?: AnalysisMetadata;
}

function isDatasetId(value: string): value is DatasetId {
  return DATASET_OPTIONS.some((dataset) => dataset.id === value);
}

function formatNumber(value: number | null | undefined, maximumFractionDigits = 1): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—';
  return new Intl.NumberFormat(undefined, { maximumFractionDigits }).format(value);
}

function landcoverEntries(landcover: LandcoverStats): Array<[string, number]> {
  const classes = landcover.classes && typeof landcover.classes === 'object'
    ? landcover.classes
    : Object.fromEntries(
        Object.entries(landcover).filter(
          ([code, value]) => /^\d+$/.test(code) && typeof value === 'number',
        ),
      ) as Record<string, number>;

  return Object.entries(classes)
    .filter(([, value]) => typeof value === 'number' && Number.isFinite(value))
    .sort(([, first], [, second]) => second - first);
}

function ResultCard({
  title,
  description,
  children,
  unavailable,
}: {
  title: string;
  description: string;
  children?: ReactNode;
  unavailable?: string;
}) {
  return (
    <section className="rounded-lg border border-white/15 bg-slate-950/30 p-4" aria-label={title}>
      <div className="mb-3">
        <h3 className="font-semibold text-white">{title}</h3>
        <p className="text-xs text-slate-400">{description}</p>
      </div>
      {unavailable ? (
        <p className="text-sm text-slate-300">{unavailable}</p>
      ) : children}
    </section>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md bg-white/5 px-3 py-2">
      <dt className="text-xs text-slate-400">{label}</dt>
      <dd className="mt-0.5 text-sm font-medium text-slate-100">{value}</dd>
    </div>
  );
}

interface SubmissionState {
  state: string;
  reason?: string | null;
  cacheKey?: string;
}

function DatasetResultCard({
  dataset,
  summary,
  submission,
  onSubmit,
}: {
  dataset: DatasetId;
  summary: Summary;
  submission?: SubmissionState;
  onSubmit?: () => void;
}) {
  const option = DATASET_OPTIONS.find((item) => item.id === dataset);
  if (!option) return null;

  if (dataset === 'dem') {
    const dem = summary.dem;
    if (!dem || dem.error) {
      return <ResultCard title={option.label} description={option.description} unavailable={dem?.error || 'No elevation data was returned for this area.'} />;
    }
    return (
      <ResultCard title={option.label} description={option.description}>
        <dl className="grid grid-cols-2 gap-2">
          <Metric label="Mean elevation" value={`${formatNumber(dem.mean, 0)} m`} />
          <Metric label="Elevation range" value={`${formatNumber(dem.elevation_range_m, 0)} m`} />
          <Metric label="Lowest point" value={`${formatNumber(dem.min, 0)} m`} />
          <Metric label="Highest point" value={`${formatNumber(dem.max, 0)} m`} />
        </dl>
        {dem.terrain_type && <p className="mt-3 text-sm text-slate-300">Terrain: {dem.terrain_type}</p>}
      </ResultCard>
    );
  }

  if (dataset === 'landcover') {
    const landcover = summary.landcover;
    if (!landcover || landcover.error) {
      return <ResultCard title={option.label} description={option.description} unavailable={landcover?.error || 'No land-cover data was returned for this area.'} />;
    }
    const entries = landcoverEntries(landcover);
    if (entries.length === 0) {
      return <ResultCard title={option.label} description={option.description} unavailable="No land-cover classes were returned for this area." />;
    }
    return (
      <ResultCard title={option.label} description={option.description}>
        {landcover.dominant_class && (
          <p className="mb-3 text-sm text-slate-200">
            Dominant: <span className="font-medium">{landcover.dominant_class}</span>
            {typeof landcover.dominant_percentage === 'number' && ` (${formatNumber(landcover.dominant_percentage, 1)}%)`}
          </p>
        )}
        <dl className="space-y-2">
          {entries.map(([code, percentage]) => (
            <div key={code} className="flex items-center justify-between gap-3 text-sm">
              <dt className="text-slate-300">{LANDCOVER_LABELS[code] || code}</dt>
              <dd className="font-medium text-slate-100">{formatNumber(percentage, 1)}%</dd>
            </div>
          ))}
        </dl>
      </ResultCard>
    );
  }

  if (dataset === 'rainfall') {
    const rain = summary.rainfall;
    if (!rain || rain.status === 'not_computed') {
      // A missing series is the expected state for an area nobody has submitted
      // yet, so it gets an action rather than an explanation alone.
      const state = submission?.state;
      const pending = state === 'pending' || state === 'running';
      const rejected = state === 'rejected';
      return (
        <ResultCard
          title={option.label}
          description={option.description}
          unavailable={
            pending
              ? 'Queued. Precipitation is processed offline, usually within a few minutes.'
              : rejected
                ? submission?.reason || 'The submission queue is full. Try again shortly.'
                : 'No series has been processed for this exact boundary yet.'
          }
        >
          {!pending && onSubmit && (
            <Button size="sm" variant="outline" onClick={onSubmit} className="mt-3">
              {rejected ? 'Try again' : 'Process precipitation for this area'}
            </Button>
          )}
          {pending && (
            <p className="mt-3 text-xs text-slate-400">
              Processing in the background. Reload the analysis to see the result.
            </p>
          )}
        </ResultCard>
      );
    }
    const s = rain.summary || {};
    const t = s.trailing_12m;
    return (
      <ResultCard title={option.label} description={option.description}>
        <dl className="grid grid-cols-2 gap-2">
          <Metric label="Last 12 months" value={t ? `${formatNumber(t.precip_mm, 0)} mm` : '—'} />
          <Metric
            label="vs 1991–2020 normal"
            value={t?.anomaly_pct == null ? '—' : `${t.anomaly_pct > 0 ? '+' : ''}${formatNumber(t.anomaly_pct, 0)}%`}
          />
          <Metric label="Annual normal" value={rain.climatology?.annual_mean_mm == null ? '—' : `${formatNumber(rain.climatology.annual_mean_mm, 0)} mm`} />
          <Metric label="Driest month" value={s.driest_month ? `${s.driest_month.month} · ${formatNumber(s.driest_month.precip_mm, 0)} mm` : '—'} />
        </dl>
        {s.suspect_months && s.suspect_months.length > 0 && (
          <p className="mt-3 text-xs text-amber-300">
            {s.suspect_months.length} month(s) reported near-zero totals and are worth review.
          </p>
        )}
        <p className="mt-3 text-xs text-slate-400">
          {rain.source} · {rain.resolution_km} km grid · {rain.grid_cells} cell{rain.grid_cells === 1 ? '' : 's'} ·{' '}
          retrieved {rain.retrieved}
        </p>
      </ResultCard>
    );
  }

  const ndvi = summary.ndvi;
  if (!ndvi || ndvi.status === 'skipped' || ndvi.status === 'unavailable') {
    return <ResultCard title={option.label} description={option.description} unavailable={ndvi?.warning || 'No recent NDVI result was returned for this area.'} />;
  }
  return (
    <ResultCard title={option.label} description={option.description}>
      <dl className="grid grid-cols-2 gap-2">
        <Metric label="Median composite mean" value={formatNumber(ndvi.mean, 2)} />
        <Metric label="Middle 50%" value={`${formatNumber(ndvi.p25, 2)}–${formatNumber(ndvi.p75, 2)}`} />
        <Metric label="Value range" value={`${formatNumber(ndvi.min, 2)}–${formatNumber(ndvi.max, 2)}`} />
        <Metric label="Scenes used" value={formatNumber(ndvi.scene_count, 0)} />
      </dl>
      <p className="mt-3 text-xs text-slate-400">
        {ndvi.source || 'Sentinel-2'}{ndvi.resolution_m ? ` · ${ndvi.resolution_m} m` : ''}
      </p>
    </ResultCard>
  );
}

export default function Home() {
   const { toast } = useToast();
   const [searchQuery, setSearchQuery] = useState('');
   const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
   const [summaryText, setSummaryText] = useState<string>('');
   const [showResults, setShowResults] = useState(false);
   const [selectedLocation, setSelectedLocation] = useState<{lat: number, lng: number} | null>(null);
   const [boundingBox, setBoundingBox] = useState<BoundingBox | null>(null);
   const [isLoading, setIsLoading] = useState(false);
   const [response, setResponse] = useState<string>('');
   const [analysisWarnings, setAnalysisWarnings] = useState<AnalysisWarning[]>([]);
   const [isSearching, setIsSearching] = useState(false);
   const [selectedDatasets, setSelectedDatasets] = useState<DatasetId[]>(['dem', 'landcover', 'ndvi', 'rainfall']);
   const [analysisSummary, setAnalysisSummary] = useState<Summary | null>(null);
  // The cache key the backend reported for this analysis, so a submission state
  // can be matched to the area that produced it.
  const [activeCacheKey, setActiveCacheKey] = useState<string | null>(null);
   const [drawnFeatures, setDrawnFeatures] = useState<FeatureCollection<Geometry> | null>(null);
  // State of a request to have an area's precipitation processed. Keyed by the
  // area, so switching areas does not show another area's progress.
  const [submission, setSubmission] = useState<Record<string, SubmissionState>>({});
   const searchTimeout = useRef<NodeJS.Timeout | null>(null);
   const [uploadedGeojson, setUploadedGeojson] = useState<GeoJsonObject | null>(null);


  // Search for places using Nominatim
  const searchPlaces = async (query: string) => {
    if (query.length < 3) {
      setSearchResults([]);
      return;
    }

    setIsSearching(true);
    try {
      const response = await fetch(
        `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}&limit=5&addressdetails=1`
      );
      const data = await response.json();
      setSearchResults(data);
      setShowResults(true);
    } catch (error) {
      console.error('Search error:', error);
    } finally {
      setIsSearching(false);
    }
  };

  // Handle search input changes with debouncing
  const handleSearchChange = (value: string) => {
    setSearchQuery(value);
    
    if (searchTimeout.current) {
      clearTimeout(searchTimeout.current);
    }

    searchTimeout.current = setTimeout(() => {
      searchPlaces(value);
    }, 300);
  };

  // Handle location selection
  const handleLocationSelect = (result: SearchResult) => {
    const lat = parseFloat(result.lat);
    const lng = parseFloat(result.lon);
    
    setSelectedLocation({ lat, lng });
    setSearchQuery(result.display_name);
    setShowResults(false);
    setSearchResults([]);
  };

  // Handle bounding box creation from map
  // Stable identity: the map's draw-control effect depends on these, and a new
  // function on every render tore down and re-registered the control each time.
  const handleBoundingBoxCreated = useCallback((bbox: BoundingBox | null) => {
    setBoundingBox(bbox);
  }, []);

  const handleFeaturesChange = useCallback((geojson: FeatureCollection) => {
    setDrawnFeatures(geojson);
  }, []);

  // Handle file upload
  const handleGeojsonUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (file.size > 500_000) {
      toast({
        title: "File is too large",
        description: "GeoJSON uploads are limited to 500 KB. Simplify the geometry and try again.",
        variant: "destructive",
      });
      e.target.value = '';
      return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const parsed = JSON.parse(event.target?.result as string);
        if (!parsed || typeof parsed !== 'object' || !('type' in parsed)) {
          throw new Error('The file is not a GeoJSON object.');
        }
        setUploadedGeojson(parsed as GeoJsonObject);
        toast({
          title: "Success",
          description: "GeoJSON file uploaded successfully.",
        });
      } catch (err) {
        console.error("Invalid GeoJSON file", err);
        toast({
          title: "Error",
          description: "Invalid GeoJSON file. Please check the file format.",
          variant: "destructive",
        });
      }
    };
    reader.readAsText(file);
  };
  // Build a concise text fallback for copying and for the results pane. The
  // structured cards below remain the authoritative presentation of values.
  function summarizeData(summary: Summary): string {
  const lines: string[] = [];

  if (summary.country) {
    lines.push(`Location context: ${summary.country}.`);
  }
  if (summary.analysis?.bbox_area_km2 != null) {
    lines.push(`Analysis bounding box: ${formatNumber(summary.analysis.bbox_area_km2, 2)} km².`);
  }

  const dem = summary.dem;
  if (dem && !dem.error && dem.mean != null) {
    lines.push(
      `Elevation: averages around ${formatNumber(dem.mean, 0)} m (range: ${formatNumber(dem.min, 0)}–${formatNumber(dem.max, 0)} m).`,
    );
    if (dem.terrain_type) {
      lines.push(`Terrain: ${dem.terrain_type}.`);
    }
  }

  const ndvi = summary.ndvi;
  if (ndvi?.mean != null) {
    lines.push(
      `Vegetation: NDVI mean is ${formatNumber(ndvi.mean, 2)} (higher values generally indicate denser vegetation).`,
    );
  }
  if (ndvi?.warning) {
    const prefix = ndvi.status === 'skipped'
      ? 'Vegetation analysis was skipped'
      : 'Vegetation analysis note';
    lines.push(`${prefix}: ${ndvi.warning}.`);
  }

  const landcover = summary.landcover;
  const rain = summary.rainfall;
  if (rain?.status === 'ok' && rain.summary?.trailing_12m) {
    const t = rain.summary.trailing_12m;
    lines.push(
      `Rainfall: ${formatNumber(t.precip_mm, 0)} mm over the last 12 months, ` +
      `${t.anomaly_pct != null ? `${t.anomaly_pct > 0 ? '+' : ''}${formatNumber(t.anomaly_pct, 0)}% ` : ''}` +
      `against the 1991-2020 normal of ${formatNumber(t.normal_mm, 0)} mm.`,
    );
  }
  if (landcover && !landcover.error) {
    const parts = landcoverEntries(landcover)
      .map(([code, percentage]) => `${LANDCOVER_LABELS[code] || code} (${formatNumber(percentage, 1)}%)`);
    if (parts.length > 0) lines.push(`Land cover: ${parts.join(', ')}.`);
  }

  return lines.join(' ') || 'The selected data sources did not return values for this area.';
}


  // Send request to backend
  // Preserve the source geometry whenever one was supplied. A bounding box is
  // only a fallback for selections created by older map interactions.
  function currentGeojson(): GeoJsonObject | undefined {
    if (uploadedGeojson) return uploadedGeojson;
    if (drawnFeatures?.features.length) return drawnFeatures;
    if (!boundingBox) return undefined;
    return {
      type: "Feature",
      geometry: {
        type: "Polygon",
        coordinates: [[
          [boundingBox.west, boundingBox.south],
          [boundingBox.east, boundingBox.south],
          [boundingBox.east, boundingBox.north],
          [boundingBox.west, boundingBox.north],
          [boundingBox.west, boundingBox.south]
        ]]
      },
      properties: {}
    } as GeoJsonObject;
  }

  // Queue a precomputation for the current area. This is the path that makes
  // "bring your own polygon" real: a drawn or uploaded boundary with no series
  // can be queued without leaving the page.
  async function submitForPreprocessing() {
    const geojson = currentGeojson();
    if (!geojson) return;
    const backendUrl = (process.env.NEXT_PUBLIC_BACKEND_URL || '/api').replace(/\/$/, '');
    setIsLoading(true);
    try {
      const response = await fetch(`${backendUrl}/rainfall/submit`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ geojson }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data?.detail || 'Submission failed');
      }
      setSubmission((previous) => ({
        ...previous,
        [data.cache_key]: {
          state: data.submission?.state || 'pending',
          reason: data.submission?.reason,
          cacheKey: data.cache_key,
        },
      }));
    } catch (error) {
      setResponse(error instanceof Error ? error.message : 'Submission failed');
      setShowResults(true);
    } finally {
      setIsLoading(false);
    }
  }

  const handleAnalyze = async () => {
    if (!uploadedGeojson && !drawnFeatures?.features.length && !boundingBox) return;
    if (selectedDatasets.length === 0) {
      toast({
        title: "Choose at least one dataset",
        description: "Select the information you want to include before running the analysis.",
        variant: "destructive",
      });
      return;
    }

    setIsLoading(true);
    setResponse('');
    setAnalysisWarnings([]);
    setAnalysisSummary(null);

    try {
      // Same-origin requests work through the reverse proxy in production and
      // through the Next.js rewrite in local Docker development.
      const backendUrl = (process.env.NEXT_PUBLIC_BACKEND_URL || '/api').replace(/\/$/, '');
      const geojson = currentGeojson();
      if (!geojson) return;

      const params = new URLSearchParams({
        include_ndvi: String(selectedDatasets.includes('ndvi')),
        datasets: selectedDatasets.join(','),
      });
      const response = await fetch(`${backendUrl}/generate-context?${params.toString()}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ geojson }),
      });

      if (!response.ok) {
        const failure = await response.json().catch(() => null);
        throw new Error(failure?.detail || `Analysis request failed (${response.status}).`);
      }

      const data = await response.json() as { summary?: Summary };
      setActiveCacheKey(
        (data.summary?.rainfall as { cache_key?: string } | undefined)?.cache_key ?? null
      );
      const summary = data.summary || {};
      const result = summarizeData(summary);
      const ndviWarning = summary.ndvi?.warning;
      setAnalysisWarnings(ndviWarning ? [{ message: ndviWarning, status: summary.ndvi?.status }] : []);
      setAnalysisSummary(summary);
      setResponse(result);
      setSummaryText(result);
    } catch (err) {
      console.error(err);
      const message = "Error: " + (err as Error).message;
      setResponse(message);
      setSummaryText(message);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-800">
      <div className="container mx-auto px-4 py-8">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center mb-4">
            <Globe className="w-12 h-12 text-blue-400 mr-3" />
            <h1 className="text-4xl font-bold text-white tracking-tight">
              Geo<span className="text-blue-400">Contextualize</span>
            </h1>
          </div>
          <p className="text-slate-300 text-lg max-w-2xl mx-auto">
            Discover geographical context and insights by selecting any area on Earth.
            Search, draw, and analyze with advanced geospatial tools.
          </p>
        </div>

        {/* Analysis limits */}
        <Alert className="mb-6 bg-amber-50 border-amber-200 max-w-4xl mx-auto">
          <Satellite className="h-4 w-4 text-amber-600" />
          <AlertDescription className="text-amber-800">
            <strong>Analysis limits:</strong> Keep the study-area bounding box within 100 km². Recent vegetation analysis is available for bounding boxes up to 10 km².
          </AlertDescription>
        </Alert>

        <div className="grid lg:grid-cols-3 gap-8 max-w-7xl mx-auto">
          {/* Left Panel - Search and Controls */}
          <div className="lg:col-span-1 space-y-6">
            {/* Search Section */}
            <Card className="bg-white/10 backdrop-blur border-white/20">
              <CardHeader>
                <CardTitle className="text-white flex items-center">
                  <Search className="w-5 h-5 mr-2" />
                  Location Search
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="relative">
                  <div className="relative">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-slate-400 w-4 h-4" />
                    <Input
                      type="text"
                      placeholder="Search for places..."
                      value={searchQuery}
                      onChange={(e) => handleSearchChange(e.target.value)}
                      className="pl-10 bg-white/20 border-white/30 text-white placeholder:text-slate-300"
                      onFocus={() => searchResults.length > 0 && setShowResults(true)}
                    />
                    {isSearching && (
                      <Loader2 className="absolute right-3 top-1/2 transform -translate-y-1/2 text-slate-400 w-4 h-4 animate-spin" />
                    )}
                  </div>
                  
                  {/* Search Results Dropdown */}
                  {showResults && searchResults.length > 0 && (
                    <div className="absolute z-50 w-full mt-1 bg-white rounded-md shadow-lg border border-gray-200 max-h-60 overflow-y-auto">
                      {searchResults.map((result) => (
                        <div
                          key={result.place_id}
                          className="px-4 py-3 hover:bg-gray-50 cursor-pointer border-b border-gray-100 last:border-b-0"
                          onClick={() => handleLocationSelect(result)}
                        >
                          <div className="flex items-start">
                            <MapPin className="w-4 h-4 text-blue-600 mt-0.5 mr-2 flex-shrink-0" />
                            <span className="text-sm text-gray-900 leading-tight">
                              {result.display_name}
                            </span>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>

            {/* Instructions */}
            <Card className="bg-white/10 backdrop-blur border-white/20">
              <CardHeader>
                <CardTitle className="text-white">How to Use</CardTitle>
              </CardHeader>
              <CardContent className="text-slate-300 space-y-3">
                <div className="flex items-start">
                  <div className="w-6 h-6 rounded-full bg-blue-500 text-white text-xs flex items-center justify-center mr-3 mt-0.5 flex-shrink-0">1</div>
                  <p className="text-sm">Search and select a location to zoom to</p>
                </div>
                <div className="flex items-start">
                  <div className="w-6 h-6 rounded-full bg-blue-500 text-white text-xs flex items-center justify-center mr-3 mt-0.5 flex-shrink-0">2</div>
                  <p className="text-sm">Upload a GeoJSON file to define your study area or if none available, use the drawing tool</p>
                </div>
                <div className="flex items-start">
                  <div className="w-6 h-6 rounded-full bg-blue-500 text-white text-xs flex items-center justify-center mr-3 mt-0.5 flex-shrink-0">3</div>
                  <p className="text-sm">Draw a polygon or rectangle to define the area to analyze</p>
                </div>
                <div className="flex items-start">
                  <div className="w-6 h-6 rounded-full bg-blue-500 text-white text-xs flex items-center justify-center mr-3 mt-0.5 flex-shrink-0">4</div>
                  <p className="text-sm">Click &quot;Analyze Area&quot; to get geographical context</p>
                </div>
              </CardContent>
            </Card>
            {/* GeoJSON Upload */}
            <Card className="bg-white/10 backdrop-blur border-white/20">
              <CardHeader>
                <CardTitle className="text-white">Upload GeoJSON</CardTitle>
              </CardHeader>
              <CardContent>
                <input
                  type="file"
                  accept=".geojson,application/geo+json,application/json"
                  onChange={handleGeojsonUpload}
                  className="block w-full text-sm text-slate-200 file:mr-4 file:py-2 file:px-4
                            file:rounded-md file:border-0 file:text-sm file:font-semibold
                            file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
                />
                <p className="text-xs text-slate-400 mt-2">
                  Upload a <code>.geojson</code> file to define your study area.
                </p>
              </CardContent>
            </Card>
            {/* Options */}
            <Card className="bg-white/10 backdrop-blur border-white/20">
              <CardHeader>
                <CardTitle className="text-white">Options</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="space-y-2">
                    <Label className="text-sm font-medium text-slate-300">Datasets to Analyze</Label>
                    <div className="space-y-2">
                      {DATASET_OPTIONS.map((dataset) => (
                        <label key={dataset.id} htmlFor={`dataset-${dataset.id}`} className="flex cursor-pointer items-start gap-2 rounded-md px-2 py-1.5 hover:bg-white/5">
                          <input
                            type="checkbox"
                            id={`dataset-${dataset.id}`}
                            checked={selectedDatasets.includes(dataset.id)}
                            onChange={(event) => {
                              setSelectedDatasets((current) => event.target.checked
                                ? [...current, dataset.id]
                                : current.filter((id) => id !== dataset.id),
                              );
                            }}
                            className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
                          />
                          <span>
                            <span className="block text-sm text-slate-200">{dataset.label}</span>
                            <span className="block text-xs text-slate-400">{dataset.description}</span>
                          </span>
                        </label>
                      ))}
                    </div>
                  </div>
                  <p className="text-xs text-slate-400 mt-2">
                    All datasets are selected by default. Vegetation analysis may be skipped for larger study areas.
                  </p>
                </div>
              </CardContent>
            </Card>
            
            {/* Analyze Button */}
            <Button
              onClick={handleAnalyze}
              disabled={(!boundingBox && !uploadedGeojson && !drawnFeatures?.features.length) || selectedDatasets.length === 0 || isLoading}
              className="w-full bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white py-6 text-lg font-semibold disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? (
                <div className="flex items-center">
                  <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin mr-2"></div>
                  Analyzing Area...
                </div>
              ) : (
                <div className="flex items-center">
                  <Satellite className="w-5 h-5 mr-2" />
                  Analyze Selected Area
                </div>
              )}
            </Button>
          </div>

          {/* Right Panel - Map and Results */}
          <div className="lg:col-span-2 space-y-6">
            {/* Map */}
            <Card className="bg-white/10 backdrop-blur border-white/20">
              <CardHeader>
                <CardTitle className="text-white flex items-center">
                  <Globe className="w-5 h-5 mr-2" />
                  Satellite Map
                </CardTitle>
              </CardHeader>
              <CardContent>
                <div className="rounded-lg overflow-hidden">
                  <div className="relative">
                    <MapComponent
                      selectedLocation={selectedLocation}
                      onBoundingBoxCreated={handleBoundingBoxCreated}
                      uploadedGeoJSON={uploadedGeojson}
                      onSaveFeatures={handleFeaturesChange}
                    />
                    {drawnFeatures && drawnFeatures.features.length > 0 && (
                      <button
                        onClick={() => {
                          const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(drawnFeatures, null, 2));
                          const downloadAnchorNode = document.createElement('a');
                          downloadAnchorNode.setAttribute("href", dataStr);
                          downloadAnchorNode.setAttribute("download", "drawn_features.geojson");
                          document.body.appendChild(downloadAnchorNode); // required for firefox
                          downloadAnchorNode.click();
                          downloadAnchorNode.remove();
                        }}
                        className="absolute bottom-4 right-4 bg-blue-600 hover:bg-blue-700 text-white px-3 py-2 rounded-md text-sm z-[1000]"
                      >
                        Download GeoJSON
                      </button>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Results */}
            <Card className="bg-white/10 backdrop-blur border-white/20">
              <CardHeader className="flex flex-row items-center justify-between">
                <CardTitle className="text-white flex items-center">
                  <MapPin className="w-5 h-5 mr-2" />
                  Analysis Results
                </CardTitle>
                {summaryText && <CopySummary summaryText={summaryText} />}
              </CardHeader>
              <CardContent>
                {analysisWarnings.map((warning) => (
                  <Alert key={warning.message} className="mb-4 border-amber-300 bg-amber-50">
                    <Satellite className="h-4 w-4 text-amber-700" />
                    <AlertDescription className="text-amber-900">
                      <strong>{warning.status === 'skipped' ? 'Vegetation analysis was skipped:' : 'Vegetation analysis note:'}</strong> {warning.message}
                    </AlertDescription>
                  </Alert>
                ))}
                <div className="bg-black/20 rounded-lg p-4 min-h-[200px]">
                  {isLoading ? (
                    <div className="flex items-center justify-center h-48">
                      <div className="text-center">
                        <div className="relative">
                          <Globe className="w-16 h-16 text-blue-400 mx-auto animate-pulse" />
                          <div className="absolute inset-0 flex items-center justify-center">
                            <div className="w-6 h-6 border-2 border-blue-400 border-t-transparent rounded-full animate-spin"></div>
                          </div>
                        </div>
                        <p className="text-slate-300 mt-4">Collecting selected geographic context...</p>
                        <p className="text-slate-400 text-xs mt-2">Remote data sources can take a moment to respond.</p>
                      </div>
                    </div>
                  ) : response ? (
                    <>
                      <div className="text-slate-200 whitespace-pre-wrap break-words text-base leading-relaxed">
                        {response.split('\n').map((paragraph, index) => (
                          <p key={index} className="mb-3 last:mb-0">{paragraph}</p>
                        ))}
                      </div>

                      {analysisSummary && (
                        <div className="mt-6 border-t border-white/10 pt-5">
                          <div className="mb-4 flex flex-wrap items-center gap-x-4 gap-y-1 text-sm text-slate-300">
                            <h3 className="font-semibold text-white">Selected data details</h3>
                            {analysisSummary.country && <span>Country: {analysisSummary.country}</span>}
                            {analysisSummary.analysis?.bbox_area_km2 != null && (
                              <span>Bounding box: {formatNumber(analysisSummary.analysis.bbox_area_km2, 2)} km²</span>
                            )}
                          </div>
                          <div className="grid gap-4 md:grid-cols-2">
                            {(analysisSummary.analysis?.datasets || selectedDatasets)
                              .filter(isDatasetId)
                              .map((dataset) => (
                                <DatasetResultCard
                                  key={dataset}
                                  dataset={dataset}
                                  summary={analysisSummary}
                                  submission={activeCacheKey ? submission[activeCacheKey] : undefined}
                                  onSubmit={dataset === 'rainfall' ? submitForPreprocessing : undefined}
                                />
                              ))}
                          </div>
                        </div>
                      )}
                    </>
                  ) : (
                    <div className="flex items-center justify-center h-48 text-slate-400">
                      <div className="text-center">
                        <Satellite className="w-12 h-12 mx-auto mb-3 opacity-50" />
                        <p>Select an area on the map and click &quot;Analyze Area&quot; to see results</p>
                      </div>
                    </div>
                  )}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  );
}
