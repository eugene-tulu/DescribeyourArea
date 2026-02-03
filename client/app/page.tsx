"use client";

import { useState, useRef } from 'react';
import dynamic from 'next/dynamic';
import { Search, MapPin, Loader2, Globe, Satellite } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Feature, FeatureCollection, Geometry } from "geojson";
import CopySummary from '@/components/Copy';


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

export default function Home() {
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState<SearchResult[]>([]);
  const [summaryText, setSummaryText] = useState<string>('');
  const [showResults, setShowResults] = useState(false);
  const [selectedLocation, setSelectedLocation] = useState<{lat: number, lng: number} | null>(null);
  const [boundingBox, setBoundingBox] = useState<BoundingBox | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [response, setResponse] = useState<string>('');
  const [isSearching, setIsSearching] = useState(false);
  const [includeNarrative, setIncludeNarrative] = useState<boolean>(true);
  const searchTimeout = useRef<NodeJS.Timeout | null>(null);
  const [uploadedGeojson, setUploadedGeojson] = useState<Feature<Geometry> | FeatureCollection<Geometry> | null>(null);


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
  const handleBoundingBoxCreated = (bbox: BoundingBox) => {
    setBoundingBox(bbox);
  };

  // Handle file upload
  const handleGeojsonUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const parsed = JSON.parse(event.target?.result as string);
        setUploadedGeojson(parsed);
        // boundingBox auto-updated in MapComponent
      } catch (err) {
        console.error("Invalid GeoJSON file", err);
        alert("Invalid GeoJSON file.");
      }
    };
    reader.readAsText(file);
  };
  //summarize data function
  interface DemStats {
  mean: number;
  min: number;
  max: number;
  std: number;
  }

  // interface TemperatureStats {
  // mean: number;
  // min: number;
  // max: number;
  // std: number;
  // }

  interface NdviStats {
  mean: number;
  min: number;
  max: number;
  std: number;
  }

  interface LandcoverStats {
    [code: string]: number; // e.g. { "10": 23.71, "20": 36.9 }
  }

  interface Summary {
    dem?: DemStats;
    // temperature?: TemperatureStats;
    ndvi?: NdviStats;
    landcover?: LandcoverStats;
  }

  function summarizeData(summary: Summary, narrative?: string): string {
  // If narrative is provided, use it instead of the basic summary
  if (narrative) {
    return narrative;
  }
  
  if (!summary) return "No summary available.";

  const { dem, ndvi, landcover } = summary;
  const lines: string[] = [];

  // 1. Elevation context
  if (dem) {
    lines.push(
      `Elevation: averages around ${dem.mean.toFixed(0)} m (range: ${dem.min}–${dem.max} m, Standard deviation of ${dem.std.toFixed(1)}).`
    );
  }

  // // 2. Temperature context
  // if (temperature?.mean != null) {
  //   lines.push(
  //     `Climate: Averaged a mean annual temperature of ${temperature.mean} °C.`
  //   );
  // }

  // 3. Vegetation NDVI context
  if (ndvi?.mean != null) {
    lines.push(
      `Vegetation health: (NDVI ≈ ${ndvi.mean.toFixed(2)}, on a scale -1 to +1).`
    );
  }

  // 4. Landcover breakdown
  const classMap: Record<string, string> = {
    "10": "Tree cover",
    "20": "Shrubland",
    "30": "Grassland",
    "40": "Cropland",
    "50": "Built-up areas",
    "60": "Bare or sparse vegetation",
    "70": "Snow & Ice",
    "80": "Permanent water bodies",
    "90": "Herbaceous wetlands",
    "95": "Mangroves",
    "100": "Moss & Lichen"
  };

  if (landcover && typeof landcover === "object") {
    const parts: string[] = [];
    for (const [code, pct] of Object.entries(landcover)) {
      const label = classMap[code] || `Class ${code}`;
      parts.push(`${label} (${(+pct).toFixed(2)}%)`);
    }
    if (parts.length > 0) {
      lines.push(`Land cover composition: ${parts.join(", ")}.`);
    }
  }

  return lines.join(" ");
}


  // Send request to backend
  const handleAnalyze = async () => {
    if ( !uploadedGeojson && !boundingBox) return;

    setIsLoading(true);
    setResponse('');

    try {
      const backendUrl = 'https://geocontextualize.onrender.com'; // Replace with your backend URL
      let geojson;
      // Construct GeoJSON polygon from boundingBox
      if (uploadedGeojson) {
      // use the uploaded file directly
      geojson = uploadedGeojson;
      } else if (boundingBox) {
        // construct from bbox
        geojson = {
          type: "Feature",
          geometry: {
            type: "Polygon",
            coordinates: [[
              [boundingBox.west, boundingBox.south],
              [boundingBox.east, boundingBox.south],
              [boundingBox.east, boundingBox.north],
              [boundingBox.west, boundingBox.north],
              [boundingBox.west, boundingBox.south] // close polygon
            ]]
          },
          properties: {}
        };
      }
      console.log("Sending to backend:", JSON.stringify({ geojson }, null, 2));
      // Add include_narrative parameter to the URL based on checkbox state
      const response = await fetch(`${backendUrl}/generate-context?include_narrative=${includeNarrative}&audience=academic`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ geojson }),
      });

      if (!response.ok) throw new Error(`HTTP error! ${response.status}`);

      const reader = response.body?.getReader();
      if (!reader) throw new Error("No response body");

      const decoder = new TextDecoder();
      let fullText = "";
      let jsonString = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        // eslint-disable-next-line @typescript-eslint/no-unused-vars
        fullText += chunk;

        // Each SSE message is separated by newlines
        chunk.split("\n").forEach((line) => {
          if (line.startsWith("data:")) {
            const msg = line.replace("data:", "").trim();

            try {
              // Check if it's valid JSON (the final summary)
              jsonString = msg; // save final JSON string
            } catch {
              // Not JSON (just a progress message), ignore
              console.log("SSE message:", msg);
            }
          }
        });
      }

      // ✅ Now parse only the final JSON payload
      if(!jsonString) throw new Error("No JSON summary received");
      const data = JSON.parse(jsonString);
      setResponse(summarizeData(data.summary, data.narrative));
      setSummaryText(summarizeData(data.summary, data.narrative));
      } catch (err) {
        console.error(err);
        setSummaryText("Error: " + (err as Error).message);
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
            Search, draw, and analyze with advanced geospatial intelligence.
          </p>
        </div>

        {/* Patient Notice */}
        <Alert className="mb-6 bg-amber-50 border-amber-200 max-w-4xl mx-auto">
          <Satellite className="h-4 w-4 text-amber-600" />
          <AlertDescription className="text-amber-800">
            <strong>Please be patient:</strong> Our backend is hosted on free infrastructure and may take 30-60 seconds to wake up for the first request. 
            Subsequent requests will be faster.
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
                  <p className="text-sm">Use the drawing tool to create a bounding box on the map</p>
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
                <div className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    id="includeNarrative"
                    checked={includeNarrative}
                    onChange={(e) => setIncludeNarrative(e.target.checked)}
                    className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
                  />
                  <label htmlFor="includeNarrative" className="text-sm text-slate-300">
                    Include AI-generated narrative description
                  </label>
                </div>
                <p className="text-xs text-slate-400 mt-2">
                  Enhances statistical data with contextual descriptions
                </p>
              </CardContent>
            </Card>
            
            {/* Analyze Button */}
            <Button
              onClick={handleAnalyze}
              disabled={!boundingBox && !uploadedGeojson || isLoading}
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
                  <MapComponent
                    selectedLocation={selectedLocation}
                    onBoundingBoxCreated={handleBoundingBoxCreated}
                    uploadedGeoJSON={uploadedGeojson}
                  />
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
                <div className="bg-black/20 rounded-lg p-4 min-h-[200px] font-mono text-sm">
                  {isLoading ? (
                    <div className="flex items-center justify-center h-48">
                      <div className="text-center">
                        <div className="relative">
                          <Globe className="w-16 h-16 text-blue-400 mx-auto animate-pulse" />
                          <div className="absolute inset-0 flex items-center justify-center">
                            <div className="w-6 h-6 border-2 border-blue-400 border-t-transparent rounded-full animate-spin"></div>
                          </div>
                        </div>
                        <p className="text-slate-300 mt-4">Analyzing geographical context...</p>
                        <p className="text-slate-400 text-xs mt-2">This may take up to 60 seconds on first request</p>
                      </div>
                    </div>
                  ) : response ? (
                    <pre className="text-slate-200 whitespace-pre-wrap break-words">{response}</pre>
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