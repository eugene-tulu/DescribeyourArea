from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, Union, List, Literal
import httpx
import rasterio
from rasterio.mask import mask
import numpy as np
import planetary_computer
import datetime
from collections import Counter
import pystac_client
from fastapi.responses import StreamingResponse
import asyncio
import json
from fastapi.middleware.cors import CORSMiddleware
import geopandas as gpd
from shapely.geometry import mapping, box
import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

app = FastAPI(title="GeoContext Generator API")

origins = [
    "https://describearea.vercel.app", 
    "http://localhost:3000",            # For local testing
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# ---------------- Schemas ---------------- #
class GeoJSONRequest(BaseModel):
    geojson: dict

class ContextResponse(BaseModel):
    summary: Dict[str, Any]
    narrative: Optional[str] = None

# ---------------- Landcover Lookup ---------------- #
ESA_WORLDCOVER_CLASSES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up areas",
    60: "Bare or sparse vegetation",
    70: "Snow and ice",
    80: "Permanent water bodies",
    90: "Herbaceous wetlands",
    95: "Mangroves",
    100: "Moss and lichen",
}

def label_landcover(percentages: Dict[str, float]) -> Dict[str, float]:
    labeled = {}
    for code, pct in percentages.items():
        label = ESA_WORLDCOVER_CLASSES.get(int(code), f"Unknown ({code})")
        labeled[label] = pct
    return labeled

# ---------------- Helpers ---------------- #
def normalize_geojson(geojson: dict) -> dict:
    if geojson.get("type") == "FeatureCollection":
        return geojson["features"][0]
    if geojson.get("type") == "Feature":
        return geojson
    raise HTTPException(400, "Unsupported GeoJSON type")

def compute_raster_stats(asset_href: str, geojson: dict) -> Dict[str, float]:
    try:
        signed_url = planetary_computer.sign(asset_href)
        with rasterio.open(signed_url) as src:
            clipped, _ = mask(src, [geojson["geometry"]], crop=True, nodata=src.nodata)
            arr = clipped[0].astype(float)
            arr[arr == src.nodata] = np.nan

            return {
                "mean": float(np.nanmean(arr)),
                "min": float(np.nanmin(arr)),
                "max": float(np.nanmax(arr)),
                "std": float(np.nanstd(arr)),
            }
    except Exception as e:
        return {"error": str(e)}

def interpret_terrain(dem: Dict[str, float]) -> Dict[str, Any]:
    if not dem or dem.get("mean") is None:
        return dem

    elevation_range = dem["max"] - dem["min"]

    if elevation_range < 50:
        terrain = "relatively flat"
    elif elevation_range < 300:
        terrain = "moderately undulating"
    else:
        terrain = "highly variable or mountainous"

    return {
        **dem,
        "elevation_range_m": round(elevation_range, 1),
        "terrain_type": terrain,
    }

def compute_landcover_percentages(asset_href: str, geojson: dict) -> Dict[str, Any]:
    try:
        signed_url = planetary_computer.sign(asset_href)
        with rasterio.open(signed_url) as src:
            clipped, _ = mask(src, [geojson["geometry"]], crop=True, nodata=src.nodata)
            arr = clipped[0].astype(int)
            arr = arr[arr != src.nodata]

            total = arr.size
            if total == 0:
                return {"error": "No valid landcover pixels"}

            counts = Counter(arr.flatten())
            percentages = {
                str(k): round((v / total) * 100, 2)
                for k, v in counts.items()
            }

            labeled = label_landcover(percentages)

            dominant_class = max(labeled, key=labeled.get)
            dominant_percentage = labeled[dominant_class]

            return {
                "classes": labeled,
                "dominant_class": dominant_class,
                "dominant_percentage": dominant_percentage,
            }

    except Exception as e:
        return {"error": str(e)}

# ---------------- Gemini ---------------- #
def load_prompt_template(name: str) -> str:
    with open(os.path.join("prompts", name), "r") as f:
        return f.read()

def generate_study_area_narrative(
    summary: Dict[str, Any],
    audience: Literal["academic", "investor", "farmer", "policy"] = "academic",
) -> str:

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "AI narrative generation unavailable: API key not configured"

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.5-flash")

    prompt = load_prompt_template("study_area_v1.txt").format(
        summary_data=json.dumps(summary, indent=2),
        audience=audience,
    )

    response = model.generate_content(prompt)
    return response.text.strip()

# ---------------- API ---------------- #
@app.post("/generate-context")
async def generate_context(
    request: GeoJSONRequest,
    include_narrative: bool = False,
    audience: str = "academic",
):

    geojson = normalize_geojson(request.geojson)
    coords = geojson["geometry"]["coordinates"][0]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    bbox = [min(xs), min(ys), max(xs), max(ys)]

    catalog = pystac_client.Client.open(
        STAC_URL, modifier=planetary_computer.sign_inplace
    )

    async def event_stream():
        # DEM
        dem_items = list(
            catalog.search(collections=["nasadem"], bbox=bbox, limit=1).items()
        )
        dem_href = dem_items[0].assets["elevation"].href if dem_items else None

        # NDVI (MODIS)
        year = datetime.date.today().year - 1
        months = {"Jan": "01", "Apr": "04", "Jul": "07", "Oct": "10"}
        ndvi_vals = []

        for m in months.values():
            search = catalog.search(
                collections=["modis-13A1-061"], bbox=bbox, datetime=f"{year}-{m}"
            )
            try:
                item = next(search.items())
                href = planetary_computer.sign(item.assets["500m_16_days_NDVI"].href)
                with rasterio.open(href) as src:
                    arr = src.read(1).astype(float)
                    arr[arr <= -2000] = np.nan
                    ndvi_vals.append(arr * 0.0001)
            except StopIteration:
                continue

        ndvi = (
            {
                "mean": float(np.nanmean(ndvi_vals)),
                "min": float(np.nanmin(ndvi_vals)),
                "max": float(np.nanmax(ndvi_vals)),
                "std": float(np.nanstd(ndvi_vals)),
            }
            if ndvi_vals
            else None
        )

        # Landcover
        lc_items = list(
            catalog.search(collections=["esa-worldcover"], bbox=bbox, limit=1).items()
        )
        lc_href = lc_items[0].assets["map"].href if lc_items else None

        raw_dem, landcover = await asyncio.gather(
            asyncio.to_thread(compute_raster_stats, dem_href, geojson),
            asyncio.to_thread(compute_landcover_percentages, lc_href, geojson),
        )

        dem = interpret_terrain(raw_dem)

        summary = {
            "dem": dem,
            "ndvi": ndvi,
            "landcover": landcover,
        }

        result = {"summary": summary}

        if include_narrative:
            result["narrative"] = generate_study_area_narrative(summary, audience)

        yield f"data: {json.dumps(result)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
