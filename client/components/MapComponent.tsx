'use client';

import { useEffect, useRef } from 'react';
import * as maplibregl from 'maplibre-gl';
import type { Map as MlMap, StyleSpecification, GeoJSONSource, IControl, MapEventType } from 'maplibre-gl';
import MapboxDraw from '@mapbox/mapbox-gl-draw';
import 'maplibre-gl/dist/maplibre-gl.css';
import '@mapbox/mapbox-gl-draw/dist/mapbox-gl-draw.css';
import type * as GeoJSON from 'geojson';

import { BASEMAPS, type BasemapId } from '@/lib/basemaps';

// Next.js does not copy MapLibre's sibling worker bundle next to the emitted
// chunk, so the default import.meta.url lookup 404s. Serve the worker from
// /public and point MapLibre at it explicitly; without it GeoJSON sources
// (the AOI and drawn features) can fail to render depending on the browser.
if (typeof window !== 'undefined') {
  maplibregl.setWorkerUrl('/maplibre-gl-worker.mjs');
}

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
  /** The one authoritative study area: uploaded, drawn or resolved by name. */
  aoiGeoJSON?: GeoJSON.GeoJsonObject | null;
  onSaveFeatures?: (features: GeoJSON.FeatureCollection) => void;
  viewBounds?: [[number, number], [number, number]] | null;
  tileLayers?: Array<{
    id: string; url: string; attribution: string; opacity?: number; zIndex?: number;
    /** Limits network tile requests to the AOI envelope; the server clips to its exact shape. */
    bounds?: [number, number, number, number];
  }>;
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
    } else if (geometry.type === 'MultiPolygon') {
      (geometry as GeoJSON.MultiPolygon).coordinates.forEach(walk);
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
  for (const [id, base] of Object.entries(BASEMAPS) as Array<[BasemapId, (typeof BASEMAPS)[BasemapId]]>) {
    const visible = id === basemap;
    base.sources.forEach((source, index) => {
      const sourceId = `basemap-${id}-${index}`;
      sources[sourceId] = {
        type: 'raster',
        tiles: source.tiles,
        tileSize: 256,
        attribution: source.attribution,
        maxzoom: source.maxzoom ?? 19,
      };
      layers.push({
        id: sourceId,
        type: 'raster',
        source: sourceId,
        layout: { visibility: visible ? 'visible' : 'none' },
        paint: { 'raster-fade-duration': 120 },
      });
    });
  }
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

function setAoiVisible(map: MlMap, visible: boolean): void {
  for (const id of ['aoi-glow', 'aoi-fill', 'aoi-line']) {
    if (map.getLayer(id)) {
      map.setLayoutProperty(id, 'visibility', visible ? 'visible' : 'none');
    }
  }
}

function asFeature(data: GeoJSON.GeoJsonObject | null | undefined): GeoJSON.Feature | GeoJSON.FeatureCollection {
  if (!data) return EMPTY_FC;
  if (data.type === 'Feature' || data.type === 'FeatureCollection') {
    return data as GeoJSON.Feature | GeoJSON.FeatureCollection;
  }
  return { type: 'Feature', properties: {}, geometry: data as GeoJSON.Geometry };
}

function addAoi(map: MlMap, data: GeoJSON.GeoJsonObject | null | undefined): void {
  const value = asFeature(data);
  const source = map.getSource('aoi') as GeoJSONSource | undefined;
  if (source) {
    source.setData(value);
    return;
  }
  map.addSource('aoi', { type: 'geojson', data: value });
  map.addLayer({ id: 'aoi-fill', type: 'fill', source: 'aoi', paint: { 'fill-color': '#b9e84b', 'fill-opacity': 0.07 } });
  // A white core on a dark halo is legible over satellite imagery, the green
  // land-cover layer and the light basemaps. Lime-on-green disappeared.
  map.addLayer({
    id: 'aoi-glow', type: 'line', source: 'aoi',
    paint: { 'line-color': '#08101f', 'line-width': 7, 'line-opacity': 0.55, 'line-blur': 2 },
  });
  map.addLayer({ id: 'aoi-line', type: 'line', source: 'aoi', paint: { 'line-color': '#ffffff', 'line-width': 2, 'line-opacity': 1 } });
}

function addRasters(map: MlMap, layers: MapComponentProps['tileLayers']): void {
  // Raster source bounds stop the browser asking for a global grid of tiles.
  // The API also alpha-clips each remaining tile to the actual (not rectangular)
  // AOI, so the bounding box is only a request optimisation, never a display mask.
  const wanted = new Set((layers ?? []).map((layer) => `raster-${layer.id}`));
  for (const layer of [...(map.getStyle().layers ?? [])]) {
    if (layer.id.startsWith('raster-') && !wanted.has(layer.id)) {
      map.removeLayer(layer.id);
      const sourceId = layer.id.replace('raster-', 'tile-');
      if (map.getSource(sourceId)) map.removeSource(sourceId);
    }
  }
  const before = map.getLayer('aoi-fill') ? 'aoi-fill' : undefined;
  for (const layer of [...(layers ?? [])].sort((a, b) => (a.zIndex ?? 0) - (b.zIndex ?? 0))) {
    const sourceId = `tile-${layer.id}`;
    const layerId = `raster-${layer.id}`;
    const current = map.getStyle().sources[sourceId] as {
      tiles?: string[]; bounds?: [number, number, number, number];
    } | undefined;
    const sameSource = current?.tiles?.[0] === layer.url
      && JSON.stringify(current?.bounds ?? null) === JSON.stringify(layer.bounds ?? null);
    if (!sameSource) {
      if (map.getLayer(layerId)) map.removeLayer(layerId);
      if (map.getSource(sourceId)) map.removeSource(sourceId);
      map.addSource(sourceId, {
        type: 'raster',
        tiles: [layer.url],
        tileSize: 256,
        attribution: layer.attribution,
        bounds: layer.bounds,
      });
      map.addLayer({
        id: layerId,
        type: 'raster',
        source: sourceId,
        paint: { 'raster-opacity': layer.opacity ?? 0.7, 'raster-fade-duration': 0 },
      }, before);
    } else {
      map.setPaintProperty(layerId, 'raster-opacity', layer.opacity ?? 0.7);
      if (before) map.moveLayer(layerId, before);
    }
  }
}

/** A MapLibre-native rectangle tool. Mapbox Draw has no rectangle mode, and a
 * type cast on the old control advertised a button that did nothing. */
class RectangleControl implements IControl {
  private map?: MlMap;
  private button?: HTMLButtonElement;
  private drawing = false;
  private start?: maplibregl.LngLat;

  constructor(private readonly draw: MapboxDraw, private readonly onCreated: () => void) {}

  onAdd(map: MlMap): HTMLElement {
    this.map = map;
    const container = document.createElement('div');
    container.className = 'maplibregl-ctrl maplibregl-ctrl-group';
    const button = document.createElement('button');
    button.type = 'button';
    button.className = 'maplibregl-ctrl-icon geocontext-rectangle-control';
    button.title = 'Draw rectangle';
    button.setAttribute('aria-label', 'Draw rectangle');
    button.setAttribute('aria-pressed', 'false');
    button.innerHTML = '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" aria-hidden="true"><rect x="4" y="6" width="16" height="12" rx="1"/></svg>';
    button.addEventListener('click', this.begin);
    map.on('draw.modechange' as keyof MapEventType, this.onModeChange as never);
    document.addEventListener('keydown', this.onKeyDown);
    container.appendChild(button);
    this.button = button;
    return container;
  }

  onRemove(): void {
    this.stop();
    this.button?.removeEventListener('click', this.begin);
    this.button?.parentElement?.remove();
    this.map?.off('draw.modechange' as keyof MapEventType, this.onModeChange as never);
    document.removeEventListener('keydown', this.onKeyDown);
    this.map = undefined;
  }

  private begin = (): void => {
    const map = this.map;
    if (!map) return;
    this.stop();
    this.draw.changeMode('simple_select');
    this.drawing = true;
    map.getCanvas().style.cursor = 'crosshair';
    this.button?.classList.add('active');
    this.button?.setAttribute('aria-pressed', 'true');
    map.dragPan.disable();
    map.once('mousedown', this.onDown);
  };

  private onDown = (event: maplibregl.MapMouseEvent): void => {
    const map = this.map;
    if (!map || !this.drawing) return;
    this.start = event.lngLat;
    map.on('mousemove', this.onMove);
    map.on('mouseup', this.onUp);
    this.renderPreview(this.start, this.start);
  };

  private onMove = (event: maplibregl.MapMouseEvent): void => {
    if (this.start) this.renderPreview(this.start, event.lngLat);
  };

  private onUp = (event: maplibregl.MapMouseEvent): void => {
    if (!this.start) return;
    const polygon = this.rectangle(this.start, event.lngLat);
    if (polygon) {
      const ids = this.draw.add({ type: 'Feature', properties: {}, geometry: polygon });
      // Select the new rectangle so Draw's trash control and vertex editing work
      // immediately, as they do for a polygon that finishes in draw mode.
      this.draw.changeMode('simple_select', { featureIds: ids });
      // Draw's API add() stores the feature but does not fire draw.create.
      // Without this callback the rectangle is visible and invisible to the app.
      this.onCreated();
    }
    this.stop();
  };

  private rectangle(a: maplibregl.LngLat, b: maplibregl.LngLat): GeoJSON.Polygon | null {
    const west = Math.min(a.lng, b.lng);
    const east = Math.max(a.lng, b.lng);
    const south = Math.min(a.lat, b.lat);
    const north = Math.max(a.lat, b.lat);
    if (east - west <= 0 || north - south <= 0) return null;
    return {
      type: 'Polygon',
      coordinates: [[[west, south], [east, south], [east, north], [west, north], [west, south]]],
    };
  }

  private renderPreview(a: maplibregl.LngLat, b: maplibregl.LngLat): void {
    const map = this.map;
    if (!map) return;
    if (!map.getSource('rectangle-preview')) {
      map.addSource('rectangle-preview', { type: 'geojson', data: EMPTY_FC });
      map.addLayer({
        id: 'rectangle-preview',
        type: 'line',
        source: 'rectangle-preview',
        paint: { 'line-color': '#4de2ff', 'line-width': 1.5, 'line-dasharray': [2, 1.5] },
      }, map.getLayer('aoi-fill') ? 'aoi-fill' : undefined);
    }
    const polygon = this.rectangle(a, b);
    const source = map.getSource('rectangle-preview') as GeoJSONSource;
    source.setData(polygon
      ? { type: 'Feature', properties: {}, geometry: polygon }
      : EMPTY_FC);
  }

  private onModeChange = (event: { mode: string }): void => {
    if (this.drawing && event.mode.startsWith('draw_')) this.stop();
  };

  private onKeyDown = (event: KeyboardEvent): void => {
    if (event.key === 'Escape') this.stop();
  };

  private stop = (): void => {
    const map = this.map;
    if (!map) return;
    this.drawing = false;
    this.start = undefined;
    map.dragPan.enable();
    map.getCanvas().style.cursor = '';
    this.button?.classList.remove('active');
    this.button?.setAttribute('aria-pressed', 'false');
    map.off('mousedown', this.onDown);
    map.off('mousemove', this.onMove);
    map.off('mouseup', this.onUp);
    const source = map.getSource('rectangle-preview') as GeoJSONSource | undefined;
    if (source) source.setData(EMPTY_FC);
  };
}

const DRAW_STYLES: object[] = [
  {
    id: 'gl-draw-polygon-fill',
    type: 'fill',
    filter: ['all', ['==', '$type', 'Polygon']],
    paint: {
      'fill-color': ['case', ['==', ['get', 'active'], 'true'], '#d6ff7a', '#b9e84b'],
      'fill-opacity': 0.1,
    },
  },
  {
    id: 'gl-draw-lines-halo',
    type: 'line',
    filter: ['any', ['==', '$type', 'LineString'], ['==', '$type', 'Polygon']],
    layout: { 'line-cap': 'round', 'line-join': 'round' },
    paint: { 'line-color': '#08101f', 'line-width': 6, 'line-opacity': 0.55 },
  },
  {
    id: 'gl-draw-lines',
    type: 'line',
    filter: ['any', ['==', '$type', 'LineString'], ['==', '$type', 'Polygon']],
    layout: { 'line-cap': 'round', 'line-join': 'round' },
    paint: {
      'line-color': ['case', ['==', ['get', 'active'], 'true'], '#4de2ff', '#ffffff'],
      'line-width': 2,
    },
  },
  {
    id: 'gl-draw-vertex-outer',
    type: 'circle',
    filter: ['all', ['==', '$type', 'Point'], ['==', 'meta', 'vertex'], ['!=', 'mode', 'simple_select']],
    paint: {
      'circle-radius': ['case', ['==', ['get', 'active'], 'true'], 7, 5],
      'circle-color': '#08101f',
    },
  },
  {
    id: 'gl-draw-vertex-inner',
    type: 'circle',
    filter: ['all', ['==', '$type', 'Point'], ['==', 'meta', 'vertex'], ['!=', 'mode', 'simple_select']],
    paint: {
      'circle-radius': ['case', ['==', ['get', 'active'], 'true'], 5, 3],
      'circle-color': '#4de2ff',
    },
  },
  {
    id: 'gl-draw-midpoint',
    type: 'circle',
    filter: ['all', ['==', 'meta', 'midpoint']],
    paint: { 'circle-radius': 3, 'circle-color': '#4de2ff' },
  },
];

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
  aoiGeoJSON,
  onSaveFeatures,
  viewBounds,
  tileLayers,
  onClearOverlay,
  points,
  basemap = 'hybrid',
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
      controls: { polygon: true, trash: true },
      styles: DRAW_STYLES,
    });
    map.addControl(draw as unknown as IControl, 'top-left');
    const syncDraw = () => {
      const features = draw.getAll() as GeoJSON.FeatureCollection;
      saveRef.current?.(features);
      bboxRef.current?.(bboxOf(features));
      // While a feature is editable, Mapbox Draw renders it. Hide the duplicate
      // read-only AOI layer so the boundary is drawn once, in one style.
      setAoiVisible(map, features.features.length === 0);
    };
    map.addControl(new RectangleControl(draw, syncDraw), 'top-left');
    map.on('draw.create' as keyof MapEventType, syncDraw);
    map.on('draw.update' as keyof MapEventType, syncDraw);
    map.on('draw.delete' as keyof MapEventType, syncDraw);

    map.on('load', () => {
      loadedRef.current = true;
      addAoi(map, aoiGeoJSON);
      setAoiVisible(map, !!aoiGeoJSON);
      addRasters(map, tileLayers);
      addPoints(map, points);
      if (terrain3d) {
        map.setTerrain({ source: 'terrain', exaggeration: 1.3 });
        if (map.getLayer('hillshade')) map.setLayoutProperty('hillshade', 'visibility', 'visible');
        map.easeTo({ pitch: 55, duration: 0 });
      }
      if (aoiGeoJSON) {
        const box = bboxOf(aoiGeoJSON);
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
    for (const [id, base] of Object.entries(BASEMAPS) as Array<[BasemapId, (typeof BASEMAPS)[BasemapId]]>) {
      base.sources.forEach((_source, index) => {
        const layerId = `basemap-${id}-${index}`;
        if (map.getLayer(layerId)) {
          map.setLayoutProperty(layerId, 'visibility', id === basemap ? 'visible' : 'none');
        }
      });
    }
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
    // An uploaded, searched or restored AOI replaces anything that was drawn.
    // When the AOI came from the draw itself, keep the editable feature.
    if (draw && aoiGeoJSON) {
      const drawn = draw.getAll() as GeoJSON.FeatureCollection;
      if (drawn.features.length > 0
          && JSON.stringify(drawn) !== JSON.stringify(aoiGeoJSON)) {
        draw.deleteAll();
      }
    }
    addAoi(map, aoiGeoJSON);
    const drawing = !!(draw && draw.getAll().features.length);
    setAoiVisible(map, !drawing && !!aoiGeoJSON);
    if (aoiGeoJSON) {
      const box = bboxOf(aoiGeoJSON);
      if (box) map.fitBounds([[box.west, box.south], [box.east, box.north]], { padding: 60, maxZoom: 14, duration: 600 });
    }
  }, [aoiGeoJSON]);

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
    map.fitBounds(viewBounds, { padding: 60, maxZoom: 11, duration: 700 });
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
        .geocontext-rectangle-control{display:flex;align-items:center;justify-content:center;color:#e8ecf4;filter:none!important}
        .geocontext-rectangle-control svg{pointer-events:none}
        .geocontext-rectangle-control.active{background:#b9e84b;color:#0b1020}
      `}</style>
    </div>
  );
}
