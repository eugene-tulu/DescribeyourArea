from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import httpx, rasterio
from rasterio.mask import mask
import numpy as np
import planetary_computer
import datetime
from collections import Counter
import pystac_client
import odc.stac
from fastapi.responses import StreamingResponse
import asyncio
import json
from fastapi.middleware.cors import CORSMiddleware
from typing import Union, List, Dict
# import sentry_sdk
import geopandas as gpd
from shapely.geometry import shape, mapping, box
import os
from dotenv import load_dotenv
from typing import Literal
import google.generativeai as genai

load_dotenv()

# sentry_sdk.init(
#     dsn="https://f3b3208c800a9df29c5e72da1b28fb1a@o4509989478596608.ingest.de.sentry.io/4509989482397776",
#     # Add data like request headers and IP for users,
#     # see https://docs.sentry.io/platforms/python/data-management/data-collected/ for more info
#     send_default_pii=True,
# )


app = FastAPI(title="GeoContext Generator API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"

# ----------- Schemas ------------ #
class GeoJSONRequest(BaseModel):
    geojson: dict

class ContextResponse(BaseModel):
    summary: Dict[str, Any]
    narrative: Optional[str] = None

# ----------- Helpers ------------ #
async def query_stac(collection: str, geojson: dict, limit: int = 1, time_range: str = None):
    """Query Planetary STAC and return item asset links.
    Arguments:
        collection (str): The STAC collection to query.
        geojson (dict): The GeoJSON geometry to intersect with.
        limit (int): The maximum number of items to return.
        time_range (str): The time range to filter items by.

    Returns:
        list: A list of STAC item asset links.
    """
    url = f"{STAC_URL}/search"
    payload = {
        "collections": [collection],
        "intersects": geojson["geometry"],
        "limit": limit,
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
    }
    if time_range:
        payload["datetime"] = time_range

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0)) as client:  # 60s timeout
            r = await client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
            return data.get("features", [])
    except httpx.ReadTimeout:
        raise HTTPException(status_code=504, detail=f"STAC query for {collection} timed out")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"STAC query failed: {str(e)}")

def compute_raster_stats(asset_href: Union[str, List[str]], geojson: dict) -> Dict[str, float]:
    try:
        if isinstance(asset_href, list):
            stats = []
            for href in asset_href:
                signed_url = planetary_computer.sign(href)
                with rasterio.open(signed_url) as src:
                    clipped, _ = mask(src, [geojson["geometry"]], crop=True, nodata=src.nodata)
                    arr = clipped[0].astype(float)
                    arr[arr == src.nodata] = np.nan
                    stats.append({
                        "mean": float(np.nanmean(arr)),
                        "min": float(np.nanmin(arr)),
                        "max": float(np.nanmax(arr)),
                        "std": float(np.nanstd(arr)),
                    })
            return {
                "per_scene": stats,
                "aggregate": {
                    "mean": float(np.nanmean([s["mean"] for s in stats])),
                    "min": float(np.nanmin([s["min"] for s in stats])),
                    "max": float(np.nanmax([s["max"] for s in stats])),
                    "std": float(np.nanmean([s["std"] for s in stats])),
                }
            }
        else:
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

def compute_landcover_percentages(asset_href: str, geojson: dict) -> Dict[str, float]:
    """Compute percentages of land cover classes inside bbox.
    Arguments:
        asset_href (str): The URL of the raster asset to download.
        geojson (dict): The GeoJSON geometry to clip the raster to.

    Returns:
        Dict[str, float]: A dictionary of land cover class percentages.
    """
    try:
        signed_url = planetary_computer.sign(asset_href)
        with rasterio.open(signed_url) as src:
            clipped, _ = mask(src, [geojson["geometry"]], crop=True, nodata=src.nodata)
            arr = clipped[0].astype(int)
            arr = arr[arr != src.nodata]
            total = arr.size
            counts = Counter(arr.flatten())
            return {str(k): round((v / total) * 100, 2) for k, v in counts.items()}
    except Exception as e:
        return {"error": str(e)}
    
def normalize_geojson(geojson: dict) -> dict:
    if geojson.get("type") == "FeatureCollection":
        return geojson["features"][0]
    elif geojson.get("type") == "Feature":
        return geojson
    raise HTTPException(400, "Unsupported GeoJSON type")

def safe_stats(arr: np.ndarray) -> Dict[str, float]:
    if arr.size == 0 or np.all(np.isnan(arr)):
        return {
            "mean": None,
            "min": None,
            "max": None,
            "std": None,
            "note": "All pixels nodata or AOI too small"
        }
    return {
        "mean": float(np.nanmean(arr)),
        "min": float(np.nanmin(arr)),
        "max": float(np.nanmax(arr)),
        "std": float(np.nanstd(arr)),
    }


def calculate_ndvi(red: np.ndarray, nir: np.ndarray) -> np.ndarray:
    ndvi = (nir - red) / (nir + red + 1e-6)  # add epsilon to avoid div by zero
    ndvi[np.isinf(ndvi)] = np.nan
    return ndvi

def compute_modis_ndvi(item, geom):
    """Compute NDVI stats for a MODIS MOD13A1 item clipped to AOI."""
    try:
        ndvi_href = planetary_computer.sign(item.assets["500m_16_days_NDVI"].href)

        with rasterio.open(ndvi_href) as src:
            # Reproject AOI to MODIS CRS
            raster_crs = src.crs
            gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326").to_crs(raster_crs)
            geom_reproj = mapping(gdf.iloc[0].geometry)

            # Clip raster
            clipped, _ = mask(src, [geom_reproj], crop=True)
            arr = clipped[0].astype("float32")
            if arr.size == 0 or np.all(np.isnan(arr)):
                return {"mean": None, "min": None, "max": None, "std": None}
            else:
                # MODIS NDVI: nodata = -3000, valid -2000 → 10000
                arr[arr <= -2000] = np.nan

                if np.all(np.isnan(arr)):
                    return {"mean": np.nan, "min": np.nan, "max": np.nan, "std": np.nan}

                # Apply scale factor
                ndvi = arr * 0.0001

                return {
                    "mean": float(np.nanmean(ndvi)),
                    "min": float(np.nanmin(ndvi)),
                    "max": float(np.nanmax(ndvi)),
                    "std": float(np.nanstd(ndvi)),
                }
    except Exception as e:
        return {"error": str(e)}
def compute_modis_lst(item, geom, to_celsius: bool = True) -> Dict[str, float]:
    """Compute MODIS LST stats (MOD11A1) for AOI."""
    try:
        lst_href = planetary_computer.sign(item.assets["LST_Day_1km"].href)
        with rasterio.open(lst_href) as src:
            raster_crs = src.crs
            gdf = gpd.GeoDataFrame(geometry=[geom], crs="EPSG:4326").to_crs(raster_crs)
            geom_reproj = mapping(gdf.iloc[0].geometry)

            clipped, _ = mask(src, [geom_reproj], crop=True)
            arr = clipped[0]

            # MOD11A1 scale factor: 0.02, values in Kelvin
            stats = safe_stats(arr, scale=0.02)

            if stats["mean"] is not None and to_celsius:
                # Convert from Kelvin to Celsius
                for k in ["mean", "min", "max"]:
                    if stats[k] is not None:
                        stats[k] = stats[k] - 273.15

            return stats
    except Exception as e:
        return {"error": str(e)}

def load_prompt_template(template_name: str) -> str:
    """Load a prompt template from the prompts directory."""
    prompt_path = os.path.join("prompts", template_name)
    with open(prompt_path, "r") as f:
        return f.read()

def generate_study_area_narrative(
    summary: Dict[str, Any],
    audience: Literal["academic", "investor", "farmer", "policy"] = "academic"
) -> str:
    """
    Takes precomputed geospatial statistics and returns
    a factual, non-speculative narrative description.
    """
    try:
        # Configure the API key (would typically come from environment)
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return "AI narrative generation unavailable: API key not configured"
        
        genai.configure(api_key=api_key)
        
        # Select the model
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # Load the prompt template
        prompt_template = load_prompt_template("study_area_v1.txt")
        
        # Format the prompt with the summary data and audience
        formatted_prompt = prompt_template.format(
            summary_data=str(summary),
            audience=audience
        )
        
        # Generate the content
        response = model.generate_content(formatted_prompt)
        
        # Extract the text from the response
        narrative = response.text.strip()
        
        return narrative
    
    except Exception as e:
        # Log the error in production
        print(f"Error generating narrative: {str(e)}")
        return ""


# ----------- API Endpoint ------------ #
@app.post("/generate-context")
async def generate_context(request: GeoJSONRequest, include_narrative: bool = False, audience: str = "academic"):
    """Generate geospatial context for a given GeoJSON polygon."""
    
    geojson = normalize_geojson(request.geojson)

    coords = geojson["geometry"]["coordinates"][0]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    bbox = [min(xs), min(ys), max(xs), max(ys)]

    current_year = datetime.date.today().year
    last_year = current_year - 1
    time_range = f"{last_year}-01-01/{last_year}-12-31"

    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)

    async def event_stream():
        # DEM
        dem_items = list(catalog.search(collections=["nasadem"], bbox=bbox, limit=1).items())
        dem_href = dem_items[0].assets["elevation"].href if dem_items else None

        # # LST
        # months = {
        #     "March": "03",
        #     "June": "06",
        #     "September": "09",
        #     "December": "12",
        # }
        # items = {}
        # for name, number in months.items():
        #     datetime = f"{last_year}-{number}"
        #     search = catalog.search(
        #         collections=["modis-11A1-061"],  # MOD11A1: Daily LST 1km
        #         bbox=bbox,
        #         datetime=datetime,
        #     )
        #     try:
        #         items[name] = next(search.items())
        #     except StopIteration:
        #         print(f"No MODIS LST found for {name} {last_year}")

        # results = {}
        # aoi_geom = box(*bbox)
        # for month, item in items.items():
        #     results[month] = compute_modis_lst(item, aoi_geom)

        # # Yearly average
        # valid = [r for r in results.values() if "mean" in r and r["mean"] is not None]
        # if valid:
        #     yearly_lst = {
        #         "mean": float(np.nanmean([r["mean"] for r in valid])),
        #         "min": float(np.nanmin([r["min"] for r in valid])),
        #         "max": float(np.nanmax([r["max"] for r in valid])),
        #         "std": float(np.nanmean([r["std"] for r in valid])),
        #     }
        # NDVI
        yield "data: Searching NDVI (Sentinel2)...\n\n"
        months = {
        "January": "01",
        "April": "04",
        "July": "07",
        "October": "10",
        }
        items = {}
        # Get 1 MODIS scene per chosen month
        for name, number in months.items():
            datetime = f"{last_year}-{number}"
            search = catalog.search(
                collections=["modis-13A1-061"],  # MOD13A1: 16-day NDVI, 500m
                bbox=bbox,
                datetime=datetime,
            )
            try:
                items[name] = next(search.items())  # first available scene in that month
            except StopIteration:
                print(f"No MODIS NDVI found for {name} {last_year}")

        # -------------------
        # Compute stats for each quarter
        results = {}
        aoi_geom = box(*bbox) # bounding box polygon
        for month, item in items.items():
            stats = compute_modis_ndvi(item, aoi_geom)
            results[month] = stats

        # Optional: compute yearly average stats
        valid = [r for r in results.values() if "mean" in r]
        if valid:
            yearly_stats = {
                "mean": float(np.nanmean([r["mean"] for r in valid])),
                "min": float(np.nanmin([r["min"] for r in valid])),
                "max": float(np.nanmax([r["max"] for r in valid])),
                "std": float(np.nanmean([r["std"] for r in valid])),
            }
        # Landcover
        yield "data: Analyzing landcover...\n\n"
        lc_items = list(catalog.search(collections=["esa-worldcover"], bbox=bbox, limit=1).items())
        lc_href = lc_items[0].assets["map"].href if lc_items else None
        dem_stats, lc_stats = await asyncio.gather(
            asyncio.to_thread(compute_raster_stats, dem_href, geojson),
            asyncio.to_thread(compute_landcover_percentages, lc_href, geojson)
        )


        # Final summary
        summary = {
            "dem": dem_stats,
            # "temperature": yearly_lst if yearly_lst else None,
            "ndvi": yearly_stats if yearly_stats else None,
            "landcover": lc_stats,
        }
        
        # Conditionally generate narrative
        result = {"summary": summary}
        if include_narrative:
            try:
                narrative = generate_study_area_narrative(summary, audience)
                result["narrative"] = narrative
            except Exception as e:
                # If narrative generation fails, continue with just the summary
                print(f"Narrative generation failed: {str(e)}")
                pass
        
        yield f"data: {json.dumps(result)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
