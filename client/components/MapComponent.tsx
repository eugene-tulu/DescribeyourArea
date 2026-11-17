'use client';

import { useEffect, useRef } from 'react';
import * as maplibregl from 'maplibre-gl';
import type { Map as MlMap, StyleSpecification, GeoJSONSource, IControl, MapEventType } from 'maplibre-gl';
import MapboxDraw from '@mapbox/mapbox-gl-draw';
import 'maplibre-gl/dist/maplibre-gl.css';
import '@mapbox/mapbox-gl-draw/dist/mapbox-gl-draw.css';
import type * as GeoJSON from 'geojson';

import { BASEMAPS, type BasemapId } from '@/lib/basemaps';

const TERRAIN_TILES = 'https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png';
const EMPTY_FC: GeoJSON.FeatureCollection = { type: 'FeatureCollection', features: [] };

interface BoundingBox {
  north: number;
  south: number;
  east: number;
  west: number;
}

interface MapComponentProps {
  onBoundingBoxCreated: (bbox: BoundingBox | null) => void;
  uploadedGeoJSON?: GeoJSON.GeoJsonObject | null;
  secondaryGeoJSON?: GeoJSON.GeoJsonObject | null;
  onSaveFeatures?: (features: GeoJSON.FeatureCollection) => void;
  viewBounds?: [[number, number], [number, number]] | null;
  tileLayers?: Array<{ id: string; url: string; attribution: string; opacity?: number; zIndex?: number }>;
  onClearOverlay?: () => void;
  points?: Array<{ id: string; lon: number; lat: number }>;
  basemap?: BasemapId;
  terrain3d?: boolean;
}

function bboxOf(input: GeoJSON.GeoJsonObject | null | undefined): BoundingBox | null {
  let west = Infinity;
  let south = Infinity;
  let east = -Infinity;
  let north = -Infinity;
  const walk = (coords: unknown): void => {
    if (!Array.isArray(coords)) return;
    if (typeof coords[0] === 'number' && typeof coords[1] === 'number') {
      west = Math.min(west, coords[0]);
      east = Math.max(east, coords[0]);
      south = Math.min(south, coords[1]);
      north = Math.max(north, coords[1]);
      return;
    }
    coords.forEach(walk);
  };
  const collect = (geometry: GeoJSON.GeoJsonObject | null | undefined): void => {
    if (!geometry) return;
    if (geometry.type === 'FeatureCollection') {
      (geometry as GeoJSON.FeatureCollection).features.forEach((f) => collect(f.geometry));
    } else if (geometry.type === 'Feature') {
      collect((geometry as GeoJSON.Feature).geometry);
    } else if (geometry.type === 'GeometryCollection') {
      (geometry as GeoJSON.GeometryCollection).geometries.forEach((g) => collect(g));
    } else {
      walk((geometry as { coordinates?: unknown }).coordinates);
    }
  };
  collect(input);
  return Number.isFinite(west) ? { west, south, east, north } : null;
}

function styleFor(basemap: BasemapId): StyleSpecification {
  const sources: StyleSpecification['sources'] = {
    terrain: {
      type: 'raster-dem',
      tiles: [TERRAIN_TILES],
      encoding: 'terrarium',
      tileSize: 256,
      maxzoom: 14,
      attribution: 'Terrain: Mapzen / AWS',
    },
  };
  const layers: StyleSpecification['layers'] = [];
  BASEMAPS[basemap].sources.forEach((source, index) => {
    const id = `basemap-${index}`;
    sources[id] = {
      type: 'raster',
      tiles: source.tiles,
      tileSize: 256,
      attribution: source.attribution,
      maxzoom: source.maxzoom ?? 19,
    };
    layers.push({ id, type: 'raster', source: id, paint: { 'raster-fade-duration': 0 } });
  });
  layers.push({
    id: 'hillshade',
    type: 'hillshade',
    source: 'terrain',
    layout: { visibility: 'none' },
    paint: {
      'hillshade-shadow-color': '#2b2418',
      'hillshade-highlight-color': '#f3ead9',
      'hillshade-accent-color': '#5a4a33',
    },
  });
  return { version: 8, sources, layers, sky: {} };
}

function addSecondary(map: MlMap, data: GeoJSON.GeoJsonObject | null | undefined): void {
  const value = (data ?? EMPTY_FC) as unknown as GeoJSON.FeatureCollection;
  const source = map.getSource('secondary') as GeoJSONSource | undefined;
  if (source) {
    source.setData(value);
    return;
  }
  map.addSource('secondary', { type: 'geojson', data: value });
  map.addLayer({
    id: 'secondary-line',
    type: 'line',
    source: 'secondary',
    paint: { 'line-color': '#8b98ad', 'line-width': 1, 'line-dasharray': [3, 2], 'line-opacity': 0.8 },
  });
}

function addAoi(map: MlMap, data: GeoJSON.GeoJsonObject | null | undefined): void {
  const value = (data ?? EMPTY_FC) as unknown as GeoJSON.FeatureCollection;
  const source = map.getSource('aoi') as GeoJSONSource | undefined;
  if (source) {
    source.setData(value);
    return;
  }
  map.addSource('aoi', { type: 'geojson', data: value });
  map.addLayer({ id: 'aoi-fill', type: 'fill', source: 'aoi', paint: { 'fill-color': '#b9e84b', 'fill-opacity': 0.06 } });
  map.addLayer({ id: 'aoi-line', type: 'line', source: 'aoi', paint: { 'line-color': '#b9e84b', 'line-width': 1.6, 'line-opacity': 0.95 } });
}

function addRasters(map: MlMap, layers: MapComponentProps['tileLayers']): void {
  const wanted = new Set((layers ?? []).map((layer) => `raster-${layer.id}`));
  for (const layer of [...(map.getStyle().layers ?? [])]) {
    if (layer.id.startsWith('raster-') && !wanted.has(layer.id)) {
      map.removeLayer(layer.id);
      const sourceId = layer.id.replace('raster-', 'tile-');
      if (map.getSource(sourceId)) map.removeSource(sourceId);
    }
  }
  for (const layer of layers ?? []) {
    const sourceId = `tile-${layer.id}`;
    const layerId = `raster-${layer.id}`;
    if (!map.getSource(sourceId)) {
      map.addSource(sourceId, {
        type: 'raster',
        tiles: [layer.url],
        tileSize: 256,
        attribution: layer.attribution,
      });
      map.addLayer({
        id: layerId,
        type: 'raster',
        source: sourceId,
        paint: { 'raster-opacity': layer.opacity ?? 0.7, 'raster-fade-duration': 0 },
      }, map.getLayer('aoi-fill') ? 'aoi-fill' : undefined);
    } else {
      map.setPaintProperty(layerId, 'raster-opacity', layer.opacity ?? 0.7);
    }
  }
}

function addPoints(map: MlMap, points: MapComponentProps['points']): void {
  const handles = points ?? [];
  const data: GeoJSON.FeatureCollection = {
    type: 'FeatureCollection',
    features: handles.map((point) => ({
      type: 'Feature',
      geometry: { type: 'Point', coordinates: [point.lon, point.lat] },
      properties: { id: point.id },
    })),
  };
  const source = map.getSource('points') as GeoJSONSource | undefined;
  if (source) {
    source.setData(data);
    return;
  }
  map.addSource('points', { type: 'geojson', data });
  map.addLayer({
    id: 'point-circles',
    type: 'circle',
    source: 'points',
    paint: {
      'circle-radius': 5,
      'circle-color': '#b9e84b',
      'circle-stroke-color': '#0b1020',
      'circle-stroke-width': 1.5,
    },
  });
  map.addLayer({
    id: 'point-labels',
    type: 'symbol',
    source: 'points',
    layout: {
      'text-field': ['get', 'id'],
      'text-size': 10,
      'text-offset': [0, 1.2],
      'text-anchor': 'top',
    },
    paint: {
      'text-color': '#e8ecf4',
      'text-halo-color': '#0b1020',
      'text-halo-width': 1,
    },
  });
  map.on('click', 'point-circles', (event) => {
    const feature = event.features?.[0];
    if (!feature) return;
    const coordinates = (feature.geometry as GeoJSON.Point).coordinates as [number, number];
    new maplibregl.Popup({ offset: 8 })
      .setLngLat(coordinates)
      .setHTML(`<strong>${String(feature.properties?.id ?? '')}</strong>`)
      .addTo(map);
  });
  map.on('mouseenter', 'point-circles', () => { map.getCanvas().style.cursor = 'pointer'; });
  map.on('mouseleave', 'point-circles', () => { map.getCanvas().style.cursor = ''; });
}

export default function MapComponent({
  onBoundingBoxCreated,
  uploadedGeoJSON,
  secondaryGeoJSON,
  onSaveFeatures,
  viewBounds,
  tileLayers,
  onClearOverlay,
  points,
  basemap = 'satellite',
  terrain3d = false,
}: MapComponentProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const mapRef = useRef<MlMap | null>(null);
  const drawRef = useRef<MapboxDraw | null>(null);
  const loadedRef = useRef(false);
  const saveRef = useRef(onSaveFeatures);
  const bboxRef = useRef(onBoundingBoxCreated);
  saveRef.current = onSaveFeatures;
  bboxRef.current = onBoundingBoxCreated;

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: styleFor(basemap),
      center: [35, -1.5],
      zoom: 6,
      maxZoom: 19,
      maxPitch: 85,
      attributionControl: { compact: true },
    });
    map.addControl(new maplibregl.NavigationControl({ visualizePitch: true, showCompass: true }), 'top-right');
    map.addControl(new maplibregl.ScaleControl({ unit: 'metric' }), 'bottom-right');

    const draw = new MapboxDraw({
      displayControlsDefault: false,
      controls: { polygon: true, rectangle: true, trash: true } as never,
    });
    map.addControl(draw as unknown as IControl, 'top-left');
    const syncDraw = () => {
      const features = draw.getAll() as GeoJSON.FeatureCollection;
      saveRef.current?.(features);
      bboxRef.current?.(bboxOf(features));
    };
    map.on('draw.create' as keyof MapEventType, syncDraw);
    map.on('draw.update' as keyof MapEventType, syncDraw);
    map.on('draw.delete' as keyof MapEventType, syncDraw);

    map.on('load', () => {
      loadedRef.current = true;
      addSecondary(map, secondaryGeoJSON);
      addAoi(map, uploadedGeoJSON);
      addRasters(map, tileLayers);
      addPoints(map, points);
      if (terrain3d) {
        map.setTerrain({ source: 'terrain', exaggeration: 1.3 });
        if (map.getLayer('hillshade')) map.setLayoutProperty('hillshade', 'visibility', 'visible');
        map.easeTo({ pitch: 55, duration: 0 });
      }
      if (uploadedGeoJSON) {
        const box = bboxOf(uploadedGeoJSON);
        if (box) map.fitBounds([[box.west, box.south], [box.east, box.north]], { padding: 60, maxZoom: 14, duration: 0 });
      }
    });

    mapRef.current = map;
    drawRef.current = draw;
    return () => {
      map.remove();
      mapRef.current = null;
      drawRef.current = null;
      loadedRef.current = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !loadedRef.current) return;
    const style = map.getStyle();
    for (const layer of [...(style.layers ?? [])]) {
      if (layer.id.startsWith('basemap-')) map.removeLayer(layer.id);
    }
    for (const id of Object.keys(style.sources ?? {})) {
      if (id.startsWith('basemap-')) map.removeSource(id);
    }
    const before = map.getLayer('hillshade') ? 'hillshade' : undefined;
    BASEMAPS[basemap].sources.forEach((source, index) => {
      const id = `basemap-${index}`;
      map.addSource(id, {
        type: 'raster',
        tiles: source.tiles,
        tileSize: 256,
        attribution: source.attribution,
        maxzoom: source.maxzoom ?? 19,
      });
      map.addLayer({ id, type: 'raster', source: id, paint: { 'raster-fade-duration': 0 } }, before);
    });
  }, [basemap]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !loadedRef.current) return;
    map.setTerrain(terrain3d ? { source: 'terrain', exaggeration: 1.3 } : null);
    if (map.getLayer('hillshade')) {
      map.setLayoutProperty('hillshade', 'visibility', terrain3d ? 'visible' : 'none');
    }
    map.easeTo({ pitch: terrain3d ? 55 : 0, duration: 700 });
  }, [terrain3d]);

  useEffect(() => {
    const map = mapRef.current;
    const draw = drawRef.current;
    if (!map || !loadedRef.current) return;
    if (uploadedGeoJSON && draw) draw.deleteAll();
    addAoi(map, uploadedGeoJSON);
    if (uploadedGeoJSON) {
      const box = bboxOf(uploadedGeoJSON);
      if (box) map.fitBounds([[box.west, box.south], [box.east, box.north]], { padding: 60, maxZoom: 14, duration: 600 });
    }
  }, [uploadedGeoJSON]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !loadedRef.current) return;
    addSecondary(map, secondaryGeoJSON);
  }, [secondaryGeoJSON]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !loadedRef.current) return;
    addRasters(map, tileLayers);
  }, [tileLayers]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !loadedRef.current) return;
    addPoints(map, points);
  }, [points]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !loadedRef.current || !viewBounds) return;
    map.fitBounds(viewBounds, { padding: 60, maxZoom: 14, duration: 700 });
  }, [viewBounds]);

  return (
    <div className="relative h-[440px] w-full sm:h-[540px] xl:h-[620px]">
      <div ref={containerRef} className="h-full w-full" />
      {onClearOverlay && (tileLayers?.length ?? 0) > 0 && (
        <button
          type="button"
          onClick={onClearOverlay}
          className="btn btn-ghost absolute bottom-4 left-4 z-10 h-8 px-3 text-[0.75rem]"
        >
          Clear overlay
        </button>
      )}
      <style>{`
        .maplibregl-ctrl-group{background:#121a2b;border:1px solid #24314a}
        .maplibregl-ctrl-group button+button{border-top:1px solid #24314a}
        .maplibregl-ctrl-group button span{filter:invert(1)}
        .maplibregl-ctrl-attrib{background:rgba(11,16,32,.85);color:#8b98ad}
        .maplibregl-ctrl-attrib a{color:#aab5cb}
        .maplibregl-ctrl-scale{background:rgba(11,16,32,.8);border-color:#24314a;color:#aab5cb}
        .mapbox-gl-draw_ctrl-draw-btn{filter:invert(1)}
      `}</style>
    </div>
  );
}