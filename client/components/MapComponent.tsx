"use client";

import { useEffect, useMemo, useRef, useCallback } from "react";
import { MapContainer, TileLayer, useMap, GeoJSON, CircleMarker, Tooltip } from "react-leaflet";
import "leaflet-draw";
import "leaflet/dist/leaflet.css";
import "leaflet-draw/dist/leaflet.draw.css";
import L from "leaflet";
import "leaflet-graticule";

// Fix default marker icons
// eslint-disable-next-line @typescript-eslint/no-explicit-any
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon-2x.png",
  iconUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-icon.png",
  shadowUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.7.1/images/marker-shadow.png",
});

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
}

/** The [south, west] / [north, east] pair Leaflet wants, from any GeoJSON.
 *
 * Handles Polygon, MultiPolygon and FeatureCollection uploads so the map can
 * frame each supported GeoJSON shape.
 */
function boundsOf(geojson: GeoJSON.GeoJsonObject): [[number, number], [number, number]] | null {
  const positions: number[][] = [];
  const walk = (node: unknown): void => {
    if (!Array.isArray(node)) return;
    if (typeof node[0] === "number" && typeof node[1] === "number") {
      positions.push(node as number[]);
      return;
    }
    node.forEach(walk);
  };
  const collect = (g: unknown): void => {
    if (!g || typeof g !== "object") return;
    const geo = g as { type?: string; coordinates?: unknown; geometries?: unknown[]; features?: unknown[] };
    if (geo.type === "FeatureCollection" && Array.isArray(geo.features)) {
      geo.features.forEach(collect);
      return;
    }
    const geometry = (g as { geometry?: unknown }).geometry;
    if (geo.type === "Feature" && geometry) {
      collect(geometry);
      return;
    }
    walk(geo.coordinates);
  };
  collect(geojson);
  if (!positions.length) return null;
  // Avoid spread operator limit (~65k args) by using reduce instead of spread
  let minLat = Infinity, minLng = Infinity, maxLat = -Infinity, maxLng = -Infinity;
  for (const pos of positions) {
    const lat = pos[1], lng = pos[0];
    if (lat < minLat) minLat = lat;
    if (lat > maxLat) maxLat = lat;
    if (lng < minLng) minLng = lng;
    if (lng > maxLng) maxLng = lng;
  }
  return [
    [minLat, minLng],
    [maxLat, maxLng],
  ];
}

function isDrawCreatedEvent(event: L.LeafletEvent): event is L.DrawEvents.Created {
  return "layer" in event && "layerType" in event;
}

/** Leaflet.draw renders its toolbar as anchors with href="#".
 *
 * That is focusable but not a button: it has no role, no accessible name, and
 * Enter does not activate it. These are the only way to draw an area without a
 * mouse, so they are given the semantics and the key handling they lack.
 */
function makeDrawAccessible(control: L.Control.Draw): void {
  const container = control.getContainer();
  if (!container) return;
  container.querySelectorAll<HTMLAnchorElement>("a").forEach((anchor) => {
    anchor.setAttribute("role", "button");
    anchor.setAttribute("tabindex", "0");
    if (!anchor.getAttribute("aria-label")) {
      anchor.setAttribute("aria-label", anchor.title || "Draw a shape");
    }
    anchor.onkeydown = (event: KeyboardEvent) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        anchor.click();
      }
    };
  });
}

function MapController({
  uploadedGeoJSON,
  viewBounds,
  onBoundingBoxCreated,
  onSaveFeatures,
}: MapComponentProps) {
  const map = useMap();
  const drawControlRef = useRef<L.Control.Draw | null>(null);
  const drawnItemsRef = useRef<L.FeatureGroup | null>(null);

  // Function to convert Leaflet layers to GeoJSON
  const convertToGeoJSON = useCallback(() => {
    if (!drawnItemsRef.current) return null;
    
    const geojsonData: GeoJSON.FeatureCollection = {
      type: 'FeatureCollection',
      features: []
    };

    drawnItemsRef.current.eachLayer((layer: L.Layer) => {
      if (layer instanceof L.Polygon || layer instanceof L.Rectangle) {
        // Convert layer to GeoJSON using leaflet's built-in method
        const geoJsonFeature = layer.toGeoJSON();
        if (geoJsonFeature && geoJsonFeature.type === 'Feature') {
          geojsonData.features.push(geoJsonFeature);
        }
      }
    });

    return geojsonData;
  }, []);

  // Memoize event handlers to prevent recreation on every render
  const handleCreated = useCallback((e: L.LeafletEvent) => {
    if (!drawnItemsRef.current) return;
    if (!isDrawCreatedEvent(e)) return;
    
    const { layer } = e;
    if (!(layer instanceof L.Polygon || layer instanceof L.Rectangle)) return;
    // Add rather than replace. Clearing here meant a second shape silently
    // deleted the first, with nothing on screen to say so -- so drawing two
    // forest stands gave you whichever one you drew last and no explanation.
    // The backend dissolves a multi-part area to a MultiPolygon, and a
    // FeatureCollection of several is a real ask: two blocks either side of a
    // road, a reserve in two parcels.
    drawnItemsRef.current.addLayer(layer);

    const bounds = layer.getBounds();
    const bbox: BoundingBox = {
      north: bounds.getNorth(),
      south: bounds.getSouth(),
      east: bounds.getEast(),
      west: bounds.getWest(),
    };
    onBoundingBoxCreated(bbox);
    
    if (onSaveFeatures) {
      const geojsonData = convertToGeoJSON();
      if (geojsonData) {
        onSaveFeatures(geojsonData);
      }
    }
  }, [onBoundingBoxCreated, onSaveFeatures, convertToGeoJSON]);

  const handleDeleted = useCallback(() => {
    onBoundingBoxCreated(null);
    
    if (onSaveFeatures) {
      const geojsonData = convertToGeoJSON();
      if (geojsonData) {
        onSaveFeatures(geojsonData);
      }
    }
  }, [onBoundingBoxCreated, onSaveFeatures, convertToGeoJSON]);

  const handleEdited = useCallback(() => {
    if (onSaveFeatures) {
      const geojsonData = convertToGeoJSON();
      if (geojsonData) {
        onSaveFeatures(geojsonData);
      }
    }
  }, [onSaveFeatures, convertToGeoJSON]);

  // Move the map to an area that was named rather than drawn.
  //
  // Naming "Meru" set the outline and nothing else, so a county appeared
  // somewhere on the globe and the reader was left to find it -- on the one
  // place they had just asked about by name, which is the last thing anyone
  // should be expected to do. It also meant the offline offer, the size and the
  // boundary were all true and all off-screen.
  useEffect(() => {
    if (!uploadedGeoJSON) return;
    const bounds = boundsOf(uploadedGeoJSON);
    if (!bounds) return;
    map.fitBounds(bounds, { padding: [36, 36], maxZoom: 11 });
  }, [uploadedGeoJSON, map]);

  useEffect(() => {
    if (viewBounds) map.fitBounds(viewBounds, { padding: [36, 36], maxZoom: 13 });
  }, [viewBounds, map]);

  useEffect(() => {
    // Graticule lives in its own effect with no handler dependencies. It used to be
    // added in the draw-control effect, whose handlers change identity on every
    // parent render, so a fresh graticule was added and never removed.
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const graticule = (L as any).latlngGraticule({
      showLabel: true,
      opacity: 0.34,
      weight: 0.7,
      color: "#9cac9f",
      zoomInterval: [{ start: 2, end: 20, interval: 1 }],
    });
    graticule.addTo(map);
    return () => {
      map.removeLayer(graticule);
    };
  }, [map]);

  useEffect(() => {
    // Init drawn items
    if (!drawnItemsRef.current) {
      drawnItemsRef.current = new L.FeatureGroup();
      map.addLayer(drawnItemsRef.current);
    }

    // Add draw control
    if (!drawControlRef.current) {
      const drawControl = new L.Control.Draw({
        position: "topright",
        draw: {
          // Lines and circles serialize to unsupported GeoJSON geometries for
          // area analysis, so only offer geometry types the API can analyze.
          polyline: false,
          polygon: {
            shapeOptions: {
              // The signal hue, not Leaflet's default blue. The boundary you draw
              // is the one thing on screen that is unambiguously yours, so it is
              // the one thing allowed to use the accent.
              color: "#b9e84b",
              weight: 2,
              fillOpacity: 0.14,
            },
          },
          circle: false,
          marker: false,
          circlemarker: false,
          rectangle: {
            shapeOptions: {
              color: "#b9e84b",
              weight: 2,
              fillOpacity: 0.14,
            },
          },
        },
        edit: {
          featureGroup: drawnItemsRef.current,
          remove: true,
        },
      });

      drawControlRef.current = drawControl;
      map.addControl(drawControl);
      makeDrawAccessible(drawControl);

      map.on(L.Draw.Event.CREATED, handleCreated);
      map.on(L.Draw.Event.DELETED, handleDeleted);
      map.on(L.Draw.Event.EDITED, handleEdited);

      // Return cleanup function
      return () => {
        map.off(L.Draw.Event.CREATED, handleCreated);
        map.off(L.Draw.Event.DELETED, handleDeleted);
        map.off(L.Draw.Event.EDITED, handleEdited);
        if (drawControlRef.current) {
          map.removeControl(drawControlRef.current);
          drawControlRef.current = null;
        }
      };
    }
  }, [map, handleCreated, handleDeleted, handleEdited]);

  return null;
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
}: MapComponentProps) {
  // Visual clip: when raster overlays are on, dim everything outside the AOI.
  // A world polygon with the AOI rings as holes (fillRule evenodd) sits above
  // the tiles but below the AOI boundary and observation points. The raster is
  // still fetched, but the reader sees only the area they selected.
  const maskGeojson = useMemo(() => {
    if (!uploadedGeoJSON || !tileLayers?.length) return null;
    const rings: number[][][] = [];
    const collect = (geometry: GeoJSON.Geometry | null | undefined) => {
      if (!geometry) return;
      if (geometry.type === "Polygon") rings.push(...geometry.coordinates);
      else if (geometry.type === "MultiPolygon") {
        geometry.coordinates.forEach((polygon) => rings.push(...polygon));
      }
    };
    if (uploadedGeoJSON.type === "FeatureCollection") {
      const collection = uploadedGeoJSON as GeoJSON.FeatureCollection;
      collection.features.forEach((feature) => collect(feature.geometry));
    } else if (uploadedGeoJSON.type === "Feature") {
      const feature = uploadedGeoJSON as GeoJSON.Feature;
      collect(feature.geometry);
    } else if (uploadedGeoJSON.type === "Polygon" || uploadedGeoJSON.type === "MultiPolygon") {
      collect(uploadedGeoJSON as GeoJSON.Geometry);
    }
    if (rings.length === 0) return null;
    return {
      type: "Feature",
      properties: {},
      geometry: {
        type: "Polygon",
        coordinates: [[[-180, -85], [180, -85], [180, 85], [-180, 85], [-180, -85]], ...rings],
      },
    } as GeoJSON.Feature;
  }, [uploadedGeoJSON, tileLayers]);

  return (
    <div className="h-[440px] w-full relative sm:h-[540px] xl:h-[620px]">
      <style>{`.leaflet-draw-toolbar a:focus-visible{outline:2px solid #b9e84b;outline-offset:2px;}`}</style>
      <MapContainer
        center={[-1.275, 36.8219]} // Nairobi default
        zoom={11}
        style={{ height: "100%", width: "100%" }}
      >
        {/* Basemap: satellite imagery, no API key. */}
        <TileLayer
          attribution="Imagery &copy; Esri, Maxar, Earthstar Geographics"
          url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
          maxZoom={19}
        />

        {/* Labels over dark imagery. This was CARTO's light_only_labels, which
            now stamps carto.com/basemaps/apikey across the tiles. Esri's own
            boundary and place overlay is keyless and does the same job. */}
        <TileLayer
          attribution="Labels &copy; Esri, HERE, Garmin, OpenStreetMap contributors"
          url="https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"
          maxZoom={19}
        />

        <MapController
          uploadedGeoJSON={uploadedGeoJSON}
          viewBounds={viewBounds}
          onBoundingBoxCreated={onBoundingBoxCreated}
          onSaveFeatures={onSaveFeatures}
        />

        {points?.map((point) => (
          <CircleMarker
            key={point.id}
            center={[point.lat, point.lon]}
            radius={5}
            pathOptions={{ color: "#b9e84b", fillColor: "#b9e84b", fillOpacity: 0.8 }}
          >
            <Tooltip>{point.id}</Tooltip>
          </CircleMarker>
        ))}

        {/* Uploaded GeoJSON */}
        {uploadedGeoJSON && (
          <GeoJSON
            key={JSON.stringify(uploadedGeoJSON)}
            data={uploadedGeoJSON}
            style={{ color: "#b9e84b", weight: 2, fillOpacity: 0.14 }}
          />
        )}
        {/* Secondary outline, kept visible when a second boundary is added */}
        {maskGeojson && (
          <GeoJSON
            data={maskGeojson as GeoJSON.GeoJsonObject}
            style={{ fillColor: "#0B1020", fillOpacity: 0.92, stroke: false, fillRule: "evenodd" } as L.PathOptions}
          />
        )}
        {secondaryGeoJSON && (
          <GeoJSON
            key={JSON.stringify(secondaryGeoJSON)}
            data={secondaryGeoJSON}
            style={{ color: "#f59e0b", weight: 2, fillOpacity: 0.08, dashArray: "6 4" }}
          />
        )}

        {/* Real XYZ tiles from the rio-tiler endpoint. COG overviews make
            panning and zooming cheap; the old raw-COG hrefs could not render. */}
        {tileLayers?.map((layer) => (
          <TileLayer
            key={layer.id}
            url={layer.url}
            attribution={layer.attribution}
            opacity={layer.opacity}
            zIndex={layer.zIndex}
          />
        ))}
      </MapContainer>

      {(tileLayers?.length ?? 0) > 0 && onClearOverlay && (
        <button
          type="button"
          onClick={onClearOverlay}
          className="absolute bottom-4 right-4 z-[1000] rounded-full border border-line-2 bg-void/80 px-3 py-1.5 text-[0.6875rem] text-ink-2 backdrop-blur-md"
        >
          Clear overlay
        </button>
      )}

      {/* Bottom left, not top left: Leaflet's zoom control is anchored there and
          the hint was drawn straight over it, so the only way to zoom was to
          find the control underneath the thing telling you to use the map. */}
      <div
        className="pointer-events-none absolute bottom-4 left-4 z-[1000] flex items-center gap-2 rounded-full border border-line-2 bg-void/80 px-3 py-1.5 text-[0.6875rem] text-ink-2 backdrop-blur-md"
      >
        <span className="beat h-1.5 w-1.5 rounded-full bg-signal" />
        Draw a polygon or rectangle to select your area
      </div>
    </div>
  );
}
