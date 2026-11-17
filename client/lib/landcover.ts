export const LANDCOVER_LABELS: Record<string, string> = {
  "1": "Water",
  "2": "Trees",
  "4": "Flooded Vegetation",
  "5": "Crops",
  "7": "Built Area",
  "8": "Bare Ground",
  "9": "Snow/Ice",
  "10": "Clouds",
  "11": "Rangeland",
};

// The Esri 10 m class palette, matching the raster the backend tiles.
export const LANDCOVER_COLOURS: Record<string, string> = {
  "1": "#0000ff",
  "2": "#005000",
  "4": "#009696",
  "5": "#966400",
  "7": "#c80032",
  "8": "#a0a0a0",
  "9": "#ffffff",
  "10": "#0000c8",
  "11": "#b4dc78",
};

export const LANDCOVER_NAME_TO_CODE: Record<string, string> = Object.fromEntries(
  Object.entries(LANDCOVER_LABELS).map(([code, label]) => [label.toLowerCase(), code]),
);
