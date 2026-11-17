export type BasemapId = 'satellite' | 'hybrid' | 'terrain' | 'streets' | 'light' | 'dark';

export interface BasemapSource {
  tiles: string[];
  attribution: string;
  maxzoom?: number;
}

export const BASEMAPS: Record<BasemapId, { label: string; sources: BasemapSource[] }> = {
  satellite: {
    label: 'Satellite',
    sources: [{
      tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'],
      attribution: 'Esri, Maxar, Earthstar Geographics',
      maxzoom: 19,
    }],
  },
  hybrid: {
    label: 'Hybrid',
    sources: [
      {
        tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}'],
        attribution: 'Esri, Maxar, Earthstar Geographics',
        maxzoom: 19,
      },
      {
        tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}'],
        attribution: 'Esri',
        maxzoom: 19,
      },
    ],
  },
  terrain: {
    label: 'Terrain',
    sources: [{
      tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}'],
      attribution: 'Esri, HERE, Garmin, USGS',
      maxzoom: 19,
    }],
  },
  streets: {
    label: 'Streets',
    sources: [{
      tiles: ['https://tile.openstreetmap.org/{z}/{x}/{y}.png'],
      attribution: '© OpenStreetMap contributors',
      maxzoom: 19,
    }],
  },
  light: {
    label: 'Light',
    sources: [
      {
        tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}'],
        attribution: 'Basemap © Esri',
        maxzoom: 16,
      },
      {
        tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Reference/MapServer/tile/{z}/{y}/{x}'],
        attribution: 'Labels © Esri',
        maxzoom: 16,
      },
    ],
  },
  dark: {
    label: 'Dark',
    sources: [
      {
        tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Base/MapServer/tile/{z}/{y}/{x}'],
        attribution: 'Basemap © Esri',
        maxzoom: 16,
      },
      {
        tiles: ['https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Dark_Gray_Reference/MapServer/tile/{z}/{y}/{x}'],
        attribution: 'Labels © Esri',
        maxzoom: 16,
      },
    ],
  },
};
