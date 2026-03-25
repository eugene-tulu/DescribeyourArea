"use client";

import { useEffect, useRef, useCallback } from "react";
import { MapContainer, TileLayer, useMap, GeoJSON } from "react-leaflet";
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
  selectedLocation: { lat: number; lng: number } | null;
  onBoundingBoxCreated: (bbox: BoundingBox) => void;
  uploadedGeoJSON?: GeoJSON.GeoJsonObject | null;
  onSaveFeatures?: (features: GeoJSON.FeatureCollection) => void;
}

function MapController({
  selectedLocation,
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
      if (layer instanceof L.Polygon || layer instanceof L.Polyline || layer instanceof L.Circle || layer instanceof L.Rectangle) {
        // Convert layer to GeoJSON using leaflet's built-in method
        const geoJsonFeature = (layer as any).toGeoJSON();
        if (geoJsonFeature && geoJsonFeature.type === 'Feature') {
          geojsonData.features.push(geoJsonFeature);
        }
      }
    });

    return geojsonData;
  }, []);

  // Memoize event handlers to prevent recreation on every render
  const handleCreated = useCallback((e: any) => {
    if (!drawnItemsRef.current) return;
    
    const { layer } = e;
    drawnItemsRef.current.clearLayers();
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
    onBoundingBoxCreated({ north: 0, south: 0, east: 0, west: 0 });
    
    if (onSaveFeatures) {
      const geojsonData = convertToGeoJSON();
      if (geojsonData) {
        onSaveFeatures(geojsonData);
      }
    }
  }, [onBoundingBoxCreated, onSaveFeatures, convertToGeoJSON]);

  const handleEdited = useCallback((e: any) => {
    if (onSaveFeatures) {
      const geojsonData = convertToGeoJSON();
      if (geojsonData) {
        onSaveFeatures(geojsonData);
      }
    }
  }, [onSaveFeatures, convertToGeoJSON]);

  useEffect(() => {
    if (selectedLocation) {
      map.setView([selectedLocation.lat, selectedLocation.lng], 12);
    }
  }, [selectedLocation, map]);

  useEffect(() => {
    // Add graticule overlay
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const graticule = (L as any).latlngGraticule({
      showLabel: true,
      opacity: 0.6,
      weight: 0.8,
      color: "#999",
      zoomInterval: [{ start: 2, end: 20, interval: 1 }],
    });
    graticule.addTo(map);

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
          polyline: {
            shapeOptions: {
              color: "#3b82f6",
              weight: 2,
            },
          },
          polygon: {
            shapeOptions: {
              color: "#3b82f6",
              weight: 2,
              fillOpacity: 0.1,
            },
          },
          circle: {
            shapeOptions: {
              color: "#3b82f6",
              weight: 2,
              fillOpacity: 0.1,
            },
          },
          marker: false,
          circlemarker: false,
          rectangle: {
            shapeOptions: {
              color: "#3b82f6",
              weight: 2,
              fillOpacity: 0.1,
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
  selectedLocation,
  onBoundingBoxCreated,
  uploadedGeoJSON,
  onSaveFeatures,
}: MapComponentProps) {
  return (
    <div className="h-[600px] w-full relative">
      <MapContainer
        center={[-1.275, 36.8219]} // Nairobi default
        zoom={11}
        style={{ height: "100%", width: "100%" }}
        className="rounded-lg"
      >
        {/* Basemap satellite */}
        <TileLayer
          attribution="&copy; Esri & contributors"
          url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
          maxZoom={19}
        />

        {/* Labels layer */}
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/">OSM</a>'
          url="https://{s}.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}{r}.png"
          subdomains={["a", "b", "c", "d"]}
        />

        <MapController
          selectedLocation={selectedLocation}
          onBoundingBoxCreated={onBoundingBoxCreated}
          onSaveFeatures={onSaveFeatures}
        />

        {/* Uploaded GeoJSON */}
        {uploadedGeoJSON && (
          <GeoJSON
            data={uploadedGeoJSON}
            style={{ color: "yellow", weight: 2, fillOpacity: 0.1 }}
          />
        )}
      </MapContainer>

      <div className="absolute top-4 left-4 bg-black/70 text-white px-3 py-2 rounded-lg text-sm sm:text-xs backdrop-blur z-[1000]">
        Draw shape to select your area
      </div>
    </div>
  );
}
