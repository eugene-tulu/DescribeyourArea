# --------------------------------------------------
# IMPORTS
# --------------------------------------------------
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, Union, List, Literal
import rasterio
from rasterio.mask import mask
import numpy as np
import planetary_computer
import datetime
from collections import Counter
import pystac_client
import asyncio
import json
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
import google.generativeai as genai
import sys
from shapely.geometry import shape

# Datacube imports (for median composite NDVI)
from odc.stac import load as stac_load
import xarray as xr

# --------------------------------------------------
# ENVIRONMENT
# --------------------------------------------------
load_dotenv()

# --------------------------------------------------
# APP CONFIGURATION
# --------------------------------------------------
app = FastAPI(title="GeoContext Generator API")

# CRITICAL FIX: Strip whitespace from origins to prevent CORS failures
origins = [
    "https://describearea.vercel.app  ",
    "http://localhost:3000",
]
origins = [origin.strip() for origin in origins if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CRITICAL FIX: Strip whitespace from STAC URL to prevent catalog failures
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1  ".strip()

# --------------------------------------------------
# SCHEMAS
# --------------------------------------------------
class GeoJSONRequest(BaseModel):
    geojson: dict

class ContextResponse(BaseModel):
    summary: Dict[str, Any]
    narrative: Optional[str] = None

# --------------------------------------------------
# LANDCOVER LOOKUP
# --------------------------------------------------
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
    return {
        ESA_WORLDCOVER_CLASSES.get(int(code), f"Unknown ({code})"): pct
        for code, pct in percentages.items()
    }

# --------------------------------------------------
# HELPERS
# --------------------------------------------------
def normalize_geojson(geojson: dict) -> dict:
    if geojson.get("type") == "FeatureCollection":
        return geojson["features"][0]
    if geojson.get("type") == "Feature":
        return geojson
    raise HTTPException(status_code=400, detail="Unsupported GeoJSON type")

def compute_raster_stats(asset_href: str, geojson: dict) -> Dict[str, float]:
    try:
        signed_url = planetary_computer.sign(asset_href)
        with rasterio.open(signed_url) as src:
            clipped, _ = mask(
                src,
                [geojson["geometry"]],
                crop=True,
                nodata=src.nodata,
            )
            arr = clipped[0].astype(float)
            arr[arr == src.nodata] = np.nan

            return {
                "mean": float(np.nanmean(arr)),
                "min": float(np.nanmin(arr)),
                "max": float(np.nanmax(arr)),
                "std": float(np.nanstd(arr)),
            }
    except Exception as e:
        print(f"DEM computation error: {str(e)}", file=sys.stderr)
        return {"error": str(e)}

def interpret_terrain(dem: Dict[str, float]) -> Dict[str, Any]:
    if not dem or "mean" not in dem:
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
            clipped, _ = mask(
                src,
                [geojson["geometry"]],
                crop=True,
                nodata=src.nodata,
            )
            arr = clipped[0].astype(int)
            arr = arr[arr != src.nodata]

            if arr.size == 0:
                return {"error": "No valid landcover pixels"}

            counts = Counter(arr.flatten())
            total = arr.size

            percentages = {
                str(k): round((v / total) * 100, 2)
                for k, v in counts.items()
            }

            labeled = label_landcover(percentages)
            dominant_class = max(labeled, key=labeled.get)

            return {
                "classes": labeled,
                "dominant_class": dominant_class,
                "dominant_percentage": labeled[dominant_class],
            }
    except Exception as e:
        print(f"Landcover computation error: {str(e)}", file=sys.stderr)
        return {"error": str(e)}

# --------------------------------------------------
# DATA CUBE HELPER (Safe Median Composite NDVI)
# --------------------------------------------------
async def compute_median_ndvi(
    bbox: list, 
    geojson_geom: dict,
    max_area_km2: float = 10.0,  # HARD LIMIT for Render free tier
    max_scenes: int = 8,         # Prevent memory explosion
    resolution_m: int = 20,      # Never go below 20m on free tier
) -> dict | None:
    """
    SAFE median composite NDVI for Render free tier.
    Returns None if constraints violated (fail gracefully).
    """
    try:
        # --- 1. Area guardrail (critical!) ---
        minx, miny, maxx, maxy = bbox
        # Approximate area in km² (rough calculation for bounding box)
        width_km = (maxx - minx) * 111.32  # degrees to km at equator
        height_km = (maxy - miny) * 111.32
        area_km2 = width_km * height_km
        
        if area_km2 > max_area_km2:
            print(f"⚠️ Skipping NDVI: area {area_km2:.1f}km² > {max_area_km2}km² limit", file=sys.stderr)
            return None

        # --- 2. Time window (last 90 days) ---
        end = datetime.datetime.utcnow()
        start = end - datetime.timedelta(days=90)
        time_window = f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"

        # --- 3. Search with strict limits ---
        catalog = pystac_client.Client.open(
            STAC_URL, 
            modifier=planetary_computer.sign_inplace
        )
        
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=time_window,
            query={"eo:cloud_cover": {"lt": 30}},
            limit=max_scenes  # CRITICAL: cap scenes
        )
        items = list(search.items())
        
        if len(items) < 2:
            print(f"⚠️ Skipping NDVI: found {len(items)} scenes (<2 required)", file=sys.stderr)
            return None

        # --- 4. Load with memory-safe parameters ---
        # Convert meters → degrees (approx for EPSG:4326)
        resolution_deg = resolution_m / 111320.0
        
        data = stac_load(
            items,
            bands=["B04", "B08", "SCL"],  # Only what we need
            crs="EPSG:4326",
            resolution=resolution_deg,
            chunks={"x": 512, "y": 512},  # SMALL chunks prevent OOM
            patch_url=planetary_computer.sign,
            bbox=bbox,
            dtype="uint16",
            groupby="solar_day",  # Deduplicate same-day scenes
            skip_broken=True,
        )
        
        # --- 5. Cloud masking (SCL band) ---
        # SCL values: 4=vegetation, 5=non-veg, 6=bare soil, 7=water, 11=snow
        valid_mask = data.SCL.isin([4, 5, 6, 7, 11])
        data = data.where(valid_mask)
        
        # --- 6. Median composite + NDVI ---
        median = data.median(dim="time")  # LAZY - no compute yet
        
        # Compute NDVI (still lazy)
        ndvi = (median.B08 - median.B04) / (median.B08 + median.B04 + 1e-8)
        
        # --- 7. Clip to polygon & compute stats (in thread with timeout) ---
        def _compute_stats():
            # Convert to rioxarray for clipping
            ndvi_rio = ndvi.rio.write_crs("EPSG:4326")
            clipped = ndvi_rio.rio.clip([geojson_geom], crs="EPSG:4326", all_touched=True)
            
            # Compute stats (triggers actual computation)
            arr = clipped.values.astype(float)
            arr = arr[~np.isnan(arr) & (arr > -1) & (arr < 1)]  # Valid NDVI range
            
            if arr.size == 0:
                return None
                
            return {
                "mean": float(np.mean(arr)),
                "min": float(np.min(arr)),
                "max": float(np.max(arr)),
                "std": float(np.std(arr)),
                "p25": float(np.percentile(arr, 25)),
                "p75": float(np.percentile(arr, 75)),
                "scene_count": len(items),
                "resolution_m": resolution_m,
                "method": "median_composite",
            }
        
        # Execute with timeout (prevents Render 30s kill)
        stats = await asyncio.wait_for(
            asyncio.to_thread(_compute_stats),
            timeout=15.0  # Must finish before 30s Render limit
        )
        
        return stats if stats and stats["mean"] is not None else None
        
    except (MemoryError, asyncio.TimeoutError) as e:
        print(f"⚠️ NDVI computation failed (safe fallback): {str(e)[:100]}", file=sys.stderr)
        return None
    except Exception as e:
        print(f"⚠️ NDVI error (non-critical): {str(e)[:100]}", file=sys.stderr)
        return None

# --------------------------------------------------
# GEMINI
# --------------------------------------------------
def load_prompt_template(name: str) -> str:
    path = os.path.join("prompts", name)
    with open(path, "r") as f:
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

# --------------------------------------------------
# API ENDPOINT
# --------------------------------------------------
@app.post("/generate-context", response_model=ContextResponse)
async def generate_context(
    request: GeoJSONRequest,
    include_narrative: bool = False,
    audience: str = "academic",
    include_ndvi: bool = True,  # NEW: explicit opt-in for NDVI
):
    try:
        # --- 1. Normalize input ---
        geojson = normalize_geojson(request.geojson)
        geom = geojson["geometry"]
        
        # Extract bounding box
        coords = geom["coordinates"][0]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        bbox = [min(xs), min(ys), max(xs), max(ys)]

        # --- 2. Open STAC catalog ---
        catalog = pystac_client.Client.open(
            STAC_URL,
            modifier=planetary_computer.sign_inplace,
        )

        # --- 3. Fetch DEM and Landcover items ---
        dem_items = list(
            catalog.search(collections=["nasadem"], bbox=bbox, limit=1).items()
        )
        lc_items = list(
            catalog.search(collections=["esa-worldcover"], bbox=bbox, limit=1).items()
        )
        
        if not dem_items or not lc_items:
            raise HTTPException(
                status_code=400, 
                detail="No elevation or landcover data available for this area"
            )

        dem_href = dem_items[0].assets["elevation"].href
        lc_href = lc_items[0].assets["map"].href

        # --- 4. Process DEM and Landcover in parallel ---
        try:
            raw_dem, landcover = await asyncio.wait_for(
                asyncio.gather(
                    asyncio.to_thread(compute_raster_stats, dem_href, geojson),
                    asyncio.to_thread(compute_landcover_percentages, lc_href, geojson),
                ),
                timeout=15.0
            )
        except asyncio.TimeoutError:
            raise HTTPException(
                status_code=504,
                detail="Processing timeout (Render free tier limit). Try smaller area."
            )
        except MemoryError:
            raise HTTPException(
                status_code=507,
                detail="Memory exceeded (Render free tier limit). Reduce area size."
            )

        # --- 5. Interpret terrain ---
        dem = interpret_terrain(raw_dem)

        # --- 6. Compute NDVI (optional, safe) ---
        ndvi_stats = None
        if include_ndvi:
            ndvi_stats = await compute_median_ndvi(
                bbox=bbox,
                geojson_geom=geom,
                max_area_km2=10.0,   # Render-safe limit
                max_scenes=8,
                resolution_m=20,     # Never 10m on free tier
            )
            # If median composite fails, fall back to MODIS (coarse but fast)
            if ndvi_stats is None:
                print("⚠️ Falling back to MODIS NDVI (coarse resolution)", file=sys.stderr)
                ndvi_stats = await compute_modis_ndvi_fallback(bbox)

        # --- 7. Build summary ---
        summary = {
            "dem": dem,
            "ndvi": ndvi_stats,  # Could be None (graceful degradation)
            "landcover": landcover,
        }

        result = {"summary": summary}

        # --- 8. Generate narrative (if requested) ---
        if include_narrative:
            try:
                narrative = await asyncio.wait_for(
                    asyncio.to_thread(generate_study_area_narrative, summary, audience),
                    timeout=8.0
                )
                result["narrative"] = narrative
            except asyncio.TimeoutError:
                result["narrative"] = "Narrative generation timed out (free tier limit)."
            except Exception as e:
                print(f"Narrative generation error: {str(e)}", file=sys.stderr)
                result["narrative"] = f"Narrative generation failed: {str(e)[:100]}"

        return result

    except HTTPException:
        raise
    except Exception as e:
        print(f"CRITICAL ERROR in /generate-context: {str(e)}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)[:150]}")

# --------------------------------------------------
# FALLBACK: MODIS NDVI (Coarse but Fast)
# --------------------------------------------------
async def compute_modis_ndvi_fallback(bbox: list) -> dict | None:
    """
    Fallback NDVI using MODIS when median composite fails.
    Coarse resolution (500m) but very fast and memory-safe.
    """
    try:
        catalog = pystac_client.Client.open(
            STAC_URL,
            modifier=planetary_computer.sign_inplace,
        )
        
        from datetime import date
        year = date.today().year - 1
        months = ["01", "04", "07", "10"]
        ndvi_vals = []

        for m in months:
            search = catalog.search(
                collections=["modis-13A1-061"],
                bbox=bbox,
                datetime=f"{year}-{m}",
            )
            try:
                item = next(search.items())
                href = planetary_computer.sign(
                    item.assets["500m_16_days_NDVI"].href
                )
                with rasterio.open(href) as src:
                    arr = src.read(1).astype(float)
                    arr[arr <= -2000] = np.nan
                    ndvi_vals.append(arr * 0.0001)
            except StopIteration:
                continue

        if not ndvi_vals:
            return None

        return {
            "mean": float(np.nanmean(ndvi_vals)),
            "min": float(np.nanmin(ndvi_vals)),
            "max": float(np.nanmax(ndvi_vals)),
            "std": float(np.nanstd(ndvi_vals)),
            "method": "modis_fallback",
            "resolution_m": 500,
            "warning": "Coarse resolution (500m) - use smaller areas for better accuracy",
        }
    except Exception as e:
        print(f"MODIS fallback error: {str(e)}", file=sys.stderr)
        return None

# --------------------------------------------------
# HEALTH CHECK ENDPOINT
# --------------------------------------------------
@app.get("/health")
async def health_check():
    """Health check endpoint for Render uptime monitoring"""
    return {
        "status": "healthy",
        "service": "GeoContext Generator API",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

# --------------------------------------------------
# VERSION ENDPOINT
# --------------------------------------------------
@app.get("/version")
async def get_version():
    """API version information"""
    return {
        "version": "1.0.0",
        "stac_provider": "Microsoft Planetary Computer",
        "ndvi_method": "Sentinel-2 median composite (fallback: MODIS)",
        "max_area_km2": 10.0,
        "resolution_m": 20
    }
