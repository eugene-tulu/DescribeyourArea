# --------------------------------------------------
# IMPORTS (CRITICAL FIX: rioxarray import)
# --------------------------------------------------
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional, Union, List, Literal
import rasterio as rio
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
import requests

# Datacube imports (CRITICAL: rioxarray registers .rio accessor)
from odc.stac import load as stac_load
import xarray as xr
import rioxarray  # MUST be imported to enable .rio methods

# Optional imports for external datasets
try:
    import overpy
    OVERPY_AVAILABLE = True
except ImportError:
    OVERPY_AVAILABLE = False

# --------------------------------------------------
# ADDITIONAL IMPORTS & EOPF SETUP
# --------------------------------------------------
import pandas as pd
import hashlib
import pickle
from pathlib import Path
import time

# EOPF optional imports
try:
    from pystac_client import Client as EopfClient
    EOPF_AVAILABLE = True
except ImportError:
    EOPF_AVAILABLE = False

# Caching configuration
CACHE_DIR = Path(".cache")
SCENE_CACHE = CACHE_DIR / "scenes"
SCENE_CACHE.mkdir(parents=True, exist_ok=True)

def get_scene_cache_key(collection: str, bbox: list, datetime_range: str, query: dict) -> str:
    """Generate deterministic cache key for scene search."""
    key_str = f"{collection}_{bbox}_{datetime_range}_{json.dumps(query, sort_keys=True)}"
    return hashlib.md5(key_str.encode()).hexdigest()[:16]

def get_or_search_scenes(
    catalog: pystac_client.Client,
    bbox: list,
    collection: str,
    datetime_range: str,
    query: dict,
    limit: int = 8,
) -> pd.DataFrame:
    """Get scene metadata with caching."""
    cache_key = get_scene_cache_key(collection, bbox, datetime_range, query)
    cache_file = SCENE_CACHE / f"{cache_key}.parquet"

    if cache_file.exists():
        try:
            df = pd.read_parquet(cache_file)
            print(f"[Cache] Loaded {len(df)} scenes from {cache_file.name}")
            return df
        except Exception as e:
            print(f"[Cache] Failed to read {cache_file}: {e}, refetching")

    search = catalog.search(
        collections=[collection],
        bbox=bbox,
        datetime=datetime_range,
        query=query,
        fields={"include": ["id", "geometry", "properties", "assets"]}
    )
    items = list(search.get_items())
    unique_items = {item.id: item for item in items}
    items = list(unique_items.values())

    records = []
    for item in items:
        records.append({
            "scene_id": item.id,
            "datetime": item.datetime,
            "geometry": item.geometry,
            "item_json": item.to_dict(),
            "cloud_cover": item.properties.get("eo:cloud_cover", 100),
            "collection": item.collection_id,
        })

    df = pd.DataFrame(records)
    if not df.empty:
        df.to_parquet(cache_file)
        print(f"[Fetch] Got {len(df)} scenes, cached to {cache_file.name}")

    return df

def select_best_scene(
    scenes_df: pd.DataFrame,
    aoi_geom,
    target_date=None,
    min_coverage: float = 0.5,
) -> Optional[dict]:
    """Score scenes by coverage, cloud, and date; return best or None."""
    aoi_area = aoi_geom.area
    scored = []

    for _, row in scenes_df.iterrows():
        try:
            scene_geom = shape(row["geometry"])
        except Exception:
            continue
        if not scene_geom.intersects(aoi_geom):
            continue
        try:
            intersection = scene_geom.intersection(aoi_geom)
            cover_pct = intersection.area / aoi_area
        except Exception:
            continue
        if cover_pct < min_coverage:
            continue

        date_score = 1.0
        if target_date is not None:
            scene_dt = row["datetime"] if row["datetime"].tzinfo else row["datetime"].tz_localize("UTC")
            date_diff = abs((scene_dt - target_date).total_seconds() / 86400)
            date_score = max(0.0, 1.0 - date_diff / 60.0)

        cloud_score = 1.0 - (row["cloud_cover"] / 100.0)
        score = 0.5 * cover_pct + 0.3 * date_score + 0.2 * cloud_score
        scored.append({**row.to_dict(), "cover_pct": cover_pct, "score": score})

    return max(scored, key=lambda x: x["score"]) if scored else None

def load_scene_data(
    item: pystac.Item,
    bbox: list,
    bands: list,
    resolution_deg: float,
) -> Optional[xr.Dataset]:
    """Load bands from a single STAC item, clipped to bbox."""
    try:
        ds = stac_load(
            [item],
            bands=bands,
            bbox=bbox,
            crs="EPSG:4326",
            resolution=resolution_deg,
            chunks={"x": 512, "y": 512},
            patch_url=planetary_computer.sign,
            dtype="uint16",
            groupby="solar_day",
            skip_broken=True,
        )
        if ds is None or len(ds.time) == 0:
            return None
        if "time" in ds.dims:
            ds = ds.isel(time=0).drop_vars("time")
        return ds
    except Exception as e:
        print(f"⚠️ Failed to load scene {item.id}: {e}", file=sys.stderr)
        return None

# --------------------------------------------------
# ENVIRONMENT
# --------------------------------------------------
load_dotenv()

# --------------------------------------------------
# APP CONFIGURATION
# --------------------------------------------------
app = FastAPI(title="GeoContext Generator API")

# CRITICAL FIX: Strip whitespace from origins
origins = [
    "https://describearea.vercel.app    ",
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

# CRITICAL FIX: Strip whitespace from STAC URL
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1    ".strip()

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
        with rio.open(signed_url) as src:
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
        with rio.open(signed_url) as src:
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
# NDVI COMPUTATION: PC OPTIMIZED + EOPF FALLBACK
# --------------------------------------------------

async def compute_ndvi_pc_optimized(
    bbox: list,
    geojson_geom: dict,
    max_area_km2: float = 100.0,
    max_scenes: int = 8,
    resolution_m: int = 20,
) -> dict | None:
    """Optimized PC NDVI with caching and best-scene selection."""
    try:
        minx, miny, maxx, maxy = bbox
        width_km = (maxx - minx) * 111.32
        height_km = (maxy - miny) * 111.32
        area_km2 = width_km * height_km
        if area_km2 > max_area_km2:
            print(f"⚠️ Skipping NDVI (PC): area {area_km2:.1f}km² > limit", file=sys.stderr)
            return None

        end = datetime.datetime.utcnow()
        start = end - datetime.timedelta(days=90)
        time_window = f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"

        catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
        query = {"eo:cloud_cover": {"lt": 30}}
        scenes_df = get_or_search_scenes(
            catalog=catalog,
            bbox=bbox,
            collection="sentinel-2-l2a",
            datetime_range=time_window,
            query=query,
            limit=max_scenes * 2,
        )

        if scenes_df.empty:
            print("⚠️ No scenes found in PC catalog", file=sys.stderr)
            return None

        best = select_best_scene(scenes_df, shape(geojson_geom), target_date=end, min_coverage=0.5)
        if best is None:
            print("⚠️ No single scene meets coverage threshold, trying composite", file=sys.stderr)
            return await _compute_ndvi_pc_composite(scenes_df.head(3), bbox, geojson_geom, resolution_m)

        print(f"✅ Selected PC scene: {best['scene_id'][:30]}... (cover={best['cover_pct']*100:.0f}%, cloud={best['cloud_cover']:.0f}%)", file=sys.stderr)

        item = pystac.Item.from_dict(best["item_json"])
        resolution_deg = resolution_m / 111320.0
        ds = await asyncio.to_thread(load_scene_data, item, bbox, ["B04", "B08", "SCL"], resolution_deg)
        if ds is None:
            return None

        ndvi = (ds.B08 - ds.B04) / (ds.B08 + ds.B04 + 1e-8)
        ndvi = ndvi.rio.write_crs("EPSG:4326")
        ndvi_clipped = ndvi.rio.clip([geojson_geom], crs="EPSG:4326", all_touched=True)
        valid_mask = ds.SCL.isin([4, 5, 6, 7, 11])
        ndvi_clipped = ndvi_clipped.where(valid_mask)

        def _stats():
            arr = ndvi_clipped.values.astype(float)
            arr = arr[~np.isnan(arr) & (arr > -1) & (arr < 1)]
            if arr.size == 0:
                return None
            return {
                "mean": float(np.mean(arr)),
                "min": float(np.nanmin(arr)),
                "max": float(np.nanmax(arr)),
                "std": float(np.nanstd(arr)),
                "p25": float(np.percentile(arr, 25)),
                "p75": float(np.percentile(arr, 75)),
                "scene_count": 1,
                "resolution_m": resolution_m,
                "method": "pc_single_scene",
                "scene_id": best["scene_id"],
                "scene_date": best["datetime"].isoformat() if hasattr(best["datetime"], "isoformat") else str(best["datetime"]),
                "scene_platform": best.get("platform", "sentinel-2"),
            }

        return await asyncio.wait_for(asyncio.to_thread(_stats), timeout=25.0)

    except asyncio.TimeoutError:
        print("⚠️ NDVI PC TIMEOUT", file=sys.stderr)
        return None
    except MemoryError:
        print("⚠️ NDVI PC MEMORY ERROR", file=sys.stderr)
        return None
    except Exception as e:
        print(f"⚠️ NDVI PC FAILED {type(e).__name__}: {str(e)[:150]}", file=sys.stderr)
        return None

async def _compute_ndvi_pc_composite(
    scenes_df: pd.DataFrame,
    bbox: list,
    geojson_geom: dict,
    resolution_m: int,
) -> dict | None:
    """Fallback composite from top N scenes."""
    try:
        resolution_deg = resolution_m / 111320.0
        items = []
        for _, row in scenes_df.iterrows():
            try:
                items.append(pystac.Item.from_dict(row["item_json"]))
            except Exception:
                continue
        if not items:
            return None

        data = stac_load(
            items,
            bands=["B04", "B08", "SCL"],
            bbox=bbox,
            crs="EPSG:4326",
            resolution=resolution_deg,
            chunks={"x": 512, "y": 512},
            patch_url=planetary_computer.sign,
            dtype="uint16",
            groupby="solar_day",
            skip_broken=True,
        )
        if data is None or len(data.time) == 0:
            return None

        valid_mask = data.SCL.isin([4, 5, 6, 7, 11])
        ndvi = (data.B08 - data.B04) / (data.B08 + data.B04 + 1e-8)
        ndvi = ndvi.rio.write_crs("EPSG:4326")
        ndvi_clipped = ndvi.rio.clip([geojson_geom], crs="EPSG:4326", all_touched=True)
        ndvi_clipped = ndvi_clipped.where(valid_mask)
        median_ndvi = ndvi_clipped.median(dim="time")

        def _stats():
            arr = median_ndvi.values.astype(float)
            arr = arr[~np.isnan(arr) & (arr > -1) & (arr < 1)]
            if arr.size == 0:
                return None
            return {
                "mean": float(np.mean(arr)),
                "min": float(np.nanmin(arr)),
                "max": float(np.nanmax(arr)),
                "std": float(np.nanstd(arr)),
                "p25": float(np.percentile(arr, 25)),
                "p75": float(np.percentile(arr, 75)),
                "scene_count": len(items),
                "resolution_m": resolution_m,
                "method": "pc_composite",
                "scene_ids": [it.id for it in items],
                "scene_dates": [it.datetime.isoformat() if hasattr(it.datetime, "isoformat") else str(it.datetime) for it in items],
            }

        return await asyncio.wait_for(asyncio.to_thread(_stats), timeout=60.0)

    except Exception as e:
        print(f"⚠️ NDVI PC COMPOSITE FAILED: {type(e).__name__}: {str(e)[:150]}", file=sys.stderr)
        return None

async def compute_ndvi_eopf(
    bbox: list,
    geojson_geom: dict,
    max_area_km2: float = 100.0,
    max_scenes: int = 8,
    resolution_m: int = 20,
) -> dict | None:
    """EOPF Sentinel Zarr NDVI."""
    if not EOPF_AVAILABLE:
        print("⚠️ EOPF not available (xarray-eopf missing)", file=sys.stderr)
        return None

    try:
        minx, miny, maxx, maxy = bbox
        width_km = (maxx - minx) * 111.32
        height_km = (maxy - miny) * 111.32
        area_km2 = width_km * height_km
        if area_km2 > max_area_km2:
            print(f"⚠️ Skipping NDVI (EOPF): area {area_km2:.1f}km² > limit", file=sys.stderr)
            return None

        catalog = EopfClient.open('https://stac.core.eopf.eodc.eu/')
        end = datetime.datetime.utcnow()
        start = end - datetime.timedelta(days=90)
        time_window = f"{start.strftime('%Y-%m-%d')}/{end.strftime('%Y-%m-%d')}"

        search = catalog.search(
            collections='sentinel-2-l2a',
            bbox=bbox,
            datetime=time_window,
            limit=max_scenes
        )
        items = list(search.items())
        if not items:
            print("⚠️ No scenes in EOPF catalog", file=sys.stderr)
            return None

        records = []
        for item in items:
            records.append({
                "scene_id": item.id,
                "datetime": item.datetime,
                "geometry": item.geometry,
                "cloud_cover": item.properties.get("eo:cloud_cover", 100),
            })
        scenes_df = pd.DataFrame(records)

        best = select_best_scene(scenes_df, shape(geojson_geom), target_date=end, min_coverage=0.5)
        if best is None:
            return await _compute_ndvi_eopf_composite(items[:3], bbox, geojson_geom, resolution_m)

        print(f"✅ EOPF selected scene: {best['scene_id'][:30]}... (cover={best['cover_pct']*100:.0f}%)", file=sys.stderr)

        item = next((i for i in items if i.id == best["scene_id"]), None)
        if item is None:
            return None

        zarr_asset = item.assets.get('product')
        if not zarr_asset:
            for key, asset in item.assets.items():
                media = getattr(asset, 'media_type', None)
                if media == 'application/vnd+zarr':
                    zarr_asset = asset
                    break
        if not zarr_asset:
            print("⚠️ No Zarr asset in EOPF item", file=sys.stderr)
            return None

        zarr_url = zarr_asset.href

        if not hasattr(compute_ndvi_eopf, "_zarr_cache"):
            compute_ndvi_eopf._zarr_cache = {}
        cache = compute_ndvi_eopf._zarr_cache

        if item.id not in cache:
            try:
                dt = xr.open_datatree(zarr_url, engine='eopf-zarr', chunks='auto')
                cache[item.id] = dt
            except Exception as e:
                print(f"⚠️ Failed to open Zarr: {e}", file=sys.stderr)
                return None
        else:
            dt = cache[item.id]

        try:
            ds = dt['/measurements/reflectance/r20m'].to_dataset()
        except Exception as e:
            print(f"⚠️ Cannot access reflectance: {e}", file=sys.stderr)
            return None

        # Determine CRS
        if hasattr(ds, 'rio') and ds.rio.crs is not None:
            dst_crs = ds.rio.crs
        else:
            epsg = item.properties.get('proj:epsg')
            if epsg:
                dst_crs = f"EPSG:{epsg}"
            else:
                print("⚠️ Cannot determine dataset CRS for EOPF", file=sys.stderr)
                return None

        from shapely.ops import transform
        import pyproj
        project = pyproj.Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True).transform
        aoi_utm = transform(project, shape(geojson_geom))
        minx, miny, maxx, maxy = aoi_utm.bounds

        x_sel = ds.x.sel(x=slice(minx, maxx))
        y_sel = ds.y.sel(y=slice(miny, maxy))
        if len(x_sel) == 0 or len(y_sel) == 0:
            print("⚠️ AOI out of EOPF dataset bounds", file=sys.stderr)
            return None

        subset = ds.sel(x=slice(minx, maxx), y=slice(miny, maxy))
        loaded = subset.compute()

        red = loaded['b04']
        nir = loaded['b8a']
        valid = (red > 0) & (nir > 0)
        ndvi = xr.where(valid, (nir - red) / (nir + red), np.nan)

        def _stats():
            arr = ndvi.values.astype(float)
            arr = arr[~np.isnan(arr) & (arr > -1) & (arr < 1)]
            if arr.size == 0:
                return None
            return {
                "mean": float(np.mean(arr)),
                "min": float(np.nanmin(arr)),
                "max": float(np.nanmax(arr)),
                "std": float(np.nanstd(arr)),
                "p25": float(np.percentile(arr, 25)),
                "p75": float(np.percentile(arr, 75)),
                "scene_count": 1,
                "resolution_m": resolution_m,
                "method": "eopf_zarr",
                "scene_id": item.id,
                "scene_date": item.datetime.isoformat() if hasattr(item.datetime, "isoformat") else str(item.datetime),
                "scene_platform": item.properties.get('platform', 'sentinel-2'),
            }

        return await asyncio.wait_for(asyncio.to_thread(_stats), timeout=25.0)

    except asyncio.TimeoutError:
        print("⚠️ NDVI EOPF TIMEOUT", file=sys.stderr)
        return None
    except MemoryError:
        print("⚠️ NDVI EOPF MEMORY ERROR", file=sys.stderr)
        return None
    except Exception as e:
        print(f"⚠️ NDVI EOPF FAILED {type(e).__name__}: {str(e)[:150]}", file=sys.stderr)
        return None

async def _compute_ndvi_eopf_composite(
    items: list,
    bbox: list,
    geojson_geom: dict,
    resolution_m: int,
) -> dict | None:
    """Future: multi-scene composite from EOPF Zarr."""
    return None

async def compute_median_ndvi(
    bbox: list,
    geojson_geom: dict,
    max_area_km2: float = 100.0,
    max_scenes: int = 8,
    resolution_m: int = 20,
) -> dict | None:
    """Orchestrator: try EOPF first, fall back to PC optimized."""
    if EOPF_AVAILABLE:
        result = await compute_ndvi_eopf(bbox, geojson_geom, max_area_km2, max_scenes, resolution_m)
        if result is not None:
            print("✅ NDVI via EOPF Zarr", file=sys.stderr)
            return result
        print("⚠️ EOPF failed, falling back to PC", file=sys.stderr)

    return await compute_ndvi_pc_optimized(bbox, geojson_geom, max_area_km2, max_scenes, resolution_m)

# --------------------------------------------------
# EXTERNAL DATASETS: Soils, Population, Climate, Hydrology
# --------------------------------------------------

def get_aoi_centroid(geojson: dict) -> tuple:
    """Return (lat, lon) of centroid."""
    from shapely.geometry import shape
    centroid = shape(geojson["geometry"]).centroid
    return centroid.y, centroid.x

async def fetch_soil_soc(bbox: list, geojson_geom: dict) -> dict | None:
    """Fetch Soil Organic Carbon (SOC) from OpenGeoHub STAC."""
    try:
        catalog = pystac_client.Client.open("https://stac.opengeohub.org/")
        collection = "biomass.soc_esacci.l4.cpool_go_landmetric"
        # Search for items (static, single date)
        search = catalog.search(
            collections=[collection],
            bbox=bbox,
            limit=1
        )
        items = list(search.items())
        if not items:
            print("⚠️ No SOC items found", file=sys.stderr)
            return None

        item = items[0]
        # Get COG asset (the one without qml/sld)
        cog_key = [k for k in item.assets.keys() if k.endswith('_go_epsg4326') or k.endswith('.tif')][0]
        href = item.assets[cog_key].href

        # Load with rasterio (public S3)
        with rio.open(href) as src:
            clipped, _ = mask(
                src,
                [geojson_geom],
                crop=True,
                nodata=src.nodata,
                all_touched=True
            )
            arr = clipped[0].astype(float)
            arr[arr == src.nodata] = np.nan
            valid = arr[~np.isnan(arr)]
            if valid.size == 0:
                return None
            mean_soc = float(np.mean(valid))
            return {
                "mean_soc_tC_ha": round(mean_soc, 2),
                "units": "tC/ha",
                "source": "OpenGeoHub",
                "collection": collection,
                "date": item.datetime.isoformat() if hasattr(item.datetime, "isoformat") else str(item.datetime),
            }
    except Exception as e:
        print(f"⚠️ SOC fetch failed: {type(e).__name__}: {str(e)[:100]}", file=sys.stderr)
        return None

async def fetch_population(bbox: list, geojson_geom: dict) -> dict | None:
    """Fetch population from GHS-POP via OpenGeoHub STAC."""
    try:
        catalog = pystac_client.Client.open("https://stac.opengeohub.org/")
        collection = "pop.count_ghs_go_landmetric"
        # Get items, pick most recent
        search = catalog.search(
            collections=[collection],
            bbox=bbox,
            limit=10  # get multiple years, sort later
        )
        items = list(search.items())
        if not items:
            print("⚠️ No POP items found", file=sys.stderr)
            return None

        # Pick latest datetime
        latest = max(items, key=lambda it: it.datetime)
        cog_key = [k for k in latest.assets.keys() if k.startswith('pop.') and not k.endswith('qml')][0]
        href = latest.assets[cog_key].href

        with rio.open(href) as src:
            clipped, _ = mask(
                src,
                [geojson_geom],
                crop=True,
                nodata=src.nodata,
                all_touched=True
            )
            arr = clipped[0].astype(float)
            arr[arr == src.nodata] = np.nan
            valid = arr[~np.isnan(arr)]
            total_pop = float(np.nansum(valid))
            # Compute area of AOI in km²
            from shapely.geometry import shape
            area_km2 = shape(geojson_geom).area * (111.32**2)  # approximate degrees to km²
            density = total_pop / area_km2 if area_km2 > 0 else None
            return {
                "total_pop": int(round(total_pop)),
                "density_per_km2": round(density, 1) if density else None,
                "year": latest.datetime.year if hasattr(latest.datetime, "year") else None,
                "source": "OpenGeoHub",
                "collection": collection,
            }
    except Exception as e:
        print(f"⚠️ Population fetch failed: {type(e).__name__}: {str(e)[:100]}", file=sys.stderr)
        return None

async def fetch_climate(geojson_geom: dict) -> dict | None:
    """Fetch climate normals from Open-Meteo."""
    try:
        lat, lon = get_aoi_centroid(geojson_geom)
        url = (
            "https://climate-api.open-meteo.com/v1/climate"
            f"?latitude={lat}&longitude={lon}"
            "&start_date=1991-01-01&end_date=2020-12-31"
            "&daily=temperature_2m_mean,precipitation_sum"
        )
        resp = await asyncio.to_thread(requests.get, url, timeout=15)
        if resp.status_code != 200:
            print(f"⚠️ Open-Meteo returned {resp.status_code}", file=sys.stderr)
            return None
        data = resp.json()
        daily = data.get("daily", {})
        temps = [t for t in daily.get("temperature_2m_mean", []) if t is not None]
        precips = [p for p in daily.get("precipitation_sum", []) if p is not None]
        if not temps or not precips:
            return None
        mean_temp = float(np.mean(temps))
        annual_precip = float(np.sum(precips)) / (len(precips) / 365.25)  # per year
        return {
            "mean_temp_c": round(mean_temp, 1),
            "annual_precip_mm": round(annual_precip, 0),
            "period": "1991-2020",
            "source": "Open-Meteo",
        }
    except Exception as e:
        print(f"⚠️ Climate fetch failed: {type(e).__name__}: {str(e)[:100]}", file=sys.stderr)
        return None

async def fetch_hydrology(bbox: list, geojson_geom: dict) -> dict | None:
    """Fetch water features from OpenStreetMap via Overpass."""
    try:
        from shapely.geometry import shape
        import overpy
        minx, miny, maxx, maxy = bbox
        # Expand bbox slightly to catch features on edges
        buffer = 0.001  # ~100m
        minx -= buffer; miny -= buffer; maxx += buffer; maxy += buffer

        query = f"""
        [out:json][timeout:25];
        (
          way["natural"="water"](bbox:{miny},{minx},{maxy},{maxx});
          relation["natural"="water"](bbox:{miny},{minx},{maxy},{maxx});
          way["waterway"~"^(river|stream|canal)$"](bbox:{miny},{minx},{maxy},{maxx});
        );
        out body;
        >;
        out skel qt;
        """

        api = overpy.Overpass()
        result = await asyncio.to_thread(api.query, query)

        # Calculate water area (polygons) and waterway length (lines)
        water_area_m2 = 0.0
        waterway_length_km = 0.0
        aoi_geom = shape(geojson_geom)

        for elem in result.ways + result.relations:
            tags = elem.tags
            # Build shapely geometry from nodes (simplified: use OSM polygon if available)
            # For MVP, use is_polygon flag
            if hasattr(elem, "geometry") and elem.geometry:
                try:
                    from shapely import wkt
                    geom = wkt.loads(elem.geometry)
                except Exception:
                    geom = None
                if geom and not geom.is_empty:
                    if geom.area > 0 and "natural" in tags and tags["natural"] == "water":
                        # Estimate area in WGS84 degrees -> m² (rough conversion)
                        area_deg2 = geom.area
                        # Approximate: 1 deg ≈ 111km, so m² = area_deg2 * (111320)^2
                        area_m2 = area_deg2 * (111320.0**2)
                        water_area_m2 += area_m2
                    elif geom.length > 0 and "waterway" in tags:
                        length_deg = geom.length
                        length_km = length_deg * 111.32  # rough
                        waterway_length_km += length_km

        water_area_km2 = water_area_m2 / 1e6
        water_cover_pct = (water_area_km2 / (aoi_geom.area * (111.32**2))) * 100 if aoi_geom.area > 0 else 0

        return {
            "water_area_km2": round(water_area_km2, 3),
            "water_cover_pct": round(water_cover_pct, 1),
            "waterway_length_km": round(waterway_length_km, 1),
            "source": "OpenStreetMap",
        }
    except Exception as e:
        print(f"⚠️ Hydrology fetch failed: {type(e).__name__}: {str(e)[:100]}", file=sys.stderr)
        return None

async def get_country_from_centroid(geojson_geom: dict) -> Optional[str]:
    """Return country name using Nominatim."""
    try:
        from shapely.geometry import shape
        centroid = shape(geojson_geom).centroid
        lat, lon = centroid.y, centroid.x

        resp = await asyncio.to_thread(
            requests.get,
            "https://nominatim.openstreetmap.org/reverse",
            params={
                "format": "json",
                "lat": lat,
                "lon": lon,
                "zoom": 4,
                "addressdetails": 1,
            },
            headers={"User-Agent": "GeoContextualize/1.0"},
            timeout=10.0,
        )
        if resp.status_code == 200:
            data = resp.json()
            return data.get("address", {}).get("country")
    except Exception as e:
        print(f"⚠️ Country lookup failed: {e}", file=sys.stderr)
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

    # Build country/regulatory context
    country = summary.get("country", "Unknown")
    admin1 = summary.get("admin_level1") or ""
    admin2 = summary.get("admin_level2") or ""
    reg_fw = summary.get("regulatory_framework")
    parts = [f"Country: {country}"]
    if admin1:
        parts.append(f"Primary administrative region: {admin1}")
    if admin2:
        parts.append(f"Secondary region: {admin2}")
    if reg_fw:
        parts.append(f"Relevant EIA framework: {reg_fw}")
    else:
        parts.append("Relevant EIA framework: International best practice")
    country_context = "\n".join(parts)

    # Build citations block
    ndvi = summary.get("ndvi", {})
    pop = summary.get("population", {})
    citations = f"""Elevation: NASA NASADEM (30 m). NASA/METI/AIST/Japan Spacesystems, 2024. Accessed via Microsoft Planetary Computer (CC-BY-4.0).
Land Cover: ESA WorldCover 2021 (10 m). © ESA WorldCover project 2021, processed by VITO. CC-BY-4.0.
Vegetation: Copernicus Sentinel-2 L2A (20 m). Scene date: {ndvi.get('scene_date','')}. Scene ID: {ndvi.get('scene_id','')}. Via {ndvi.get('source','Planetary Computer')}. CC-BY-4.0.
Soils: ESA CCI Soil Organic Carbon (100 m). Year: 2021. Via OpenGeoHub STAC. CC-BY-4.0.
Climate: Open-Meteo climate normals (1991-2020). Temperature and precipitation. https://open-meteo.com. CC-BY-4.0.
Hydrology: OpenStreetMap water features (natural=water, waterway=rivers/streams). © OpenStreetMap contributors, ODbL.
Population: GHS-POP (Global Human Settlement Layer) 100 m. Year: {pop.get('year','')}. Via OpenGeoHub STAC. CC-BY-4.0."""
    if reg_fw:
        citations += f"\nRegulatory framework: {reg_fw}"

    prompt = load_prompt_template("study_area_v2.txt").format(
        summary_data=json.dumps(summary, indent=2),
        country_context=country_context,
        citations=citations,
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
    include_ndvi: bool = True,
    regulatory_framework: Optional[str] = None,
):
    try:
        geojson = normalize_geojson(request.geojson)
        geom = geojson["geometry"]
        
        coords = geom["coordinates"][0]
        xs = [c[0] for c in coords]
        ys = [c[1] for c in coords]
        bbox = [min(xs), min(ys), max(xs), max(ys)]

        catalog = pystac_client.Client.open(
            STAC_URL,
            modifier=planetary_computer.sign_inplace,
        )

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

        # Process DEM + Landcover with timeout guardrail
        try:
            raw_dem, landcover = await asyncio.wait_for(
                asyncio.gather(
                    asyncio.to_thread(compute_raster_stats, dem_href, geojson),
                    asyncio.to_thread(compute_landcover_percentages, lc_href, geojson),
                ),
                timeout=20.0  # 20s buffer before Render 30s kill
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

        dem = interpret_terrain(raw_dem)

        # Compute NDVI (optimized pipeline)
        ndvi_stats = None
        if include_ndvi:
            ndvi_stats = await compute_median_ndvi(
                bbox=bbox,
                geojson_geom=geom,
                max_area_km2=100.0,
                max_scenes=8,
                resolution_m=20,
            )
            if ndvi_stats is None:
                print("⚠️ Falling back to MODIS NDVI (coarse resolution)", file=sys.stderr)
                ndvi_stats = await compute_modis_ndvi_fallback(bbox)

        # Fetch additional datasets in parallel
        try:
            soil_result, pop_result, climate_result, hydro_result = await asyncio.wait_for(
                asyncio.gather(
                    fetch_soil_soc(bbox, geom),
                    fetch_population(bbox, geom),
                    fetch_climate(geom),
                    fetch_hydrology(bbox, geom),
                ),
                timeout=30.0  # total extra datasets budget
            )
        except asyncio.TimeoutError:
            print("⚠️ Extra datasets timeout, proceeding without them", file=sys.stderr)
            soil_result = pop_result = climate_result = hydro_result = None

        # Get country
        country = await get_country_from_centroid(geom)

        # Extract scene metadata for citations
        scene_dates = {}
        scene_ids = {}
        if ndvi_stats:
            if "scene_date" in ndvi_stats:
                scene_dates["ndvi"] = ndvi_stats["scene_date"]
            if "scene_id" in ndvi_stats:
                scene_ids["ndvi"] = ndvi_stats["scene_id"]
            # Could also add from composite if multiple scenes
            if "scene_dates" in ndvi_stats:
                scene_dates["ndvi_composite"] = ", ".join(ndvi_stats["scene_dates"])
            if "scene_ids" in ndvi_stats:
                scene_ids["ndvi_composite"] = ", ".join(ndvi_stats["scene_ids"])

        summary = {
            "dem": dem,
            "ndvi": ndvi_stats,
            "landcover": landcover,
            "soils": soil_result,
            "population": pop_result,
            "climate": climate_result,
            "hydrology": hydro_result,
            "country": country,
            "admin_level1": "",  # TODO: fetch from OSM
            "admin_level2": "",  # TODO: fetch from OSM
            "regulatory_framework": regulatory_framework or "",
            "scene_dates": scene_dates,
            "scene_ids": scene_ids,
        }

        result = {"summary": summary}

        if include_narrative:
            try:
                narrative = await asyncio.wait_for(
                    asyncio.to_thread(generate_study_area_narrative, summary, audience),
                    timeout=10.0
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
        print(f"CRITICAL ERROR in /generate-context: {type(e).__name__} - {str(e)[:200]}", file=sys.stderr)
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)[:150]}")

# --------------------------------------------------
# FALLBACK: MODIS NDVI
# --------------------------------------------------
async def compute_modis_ndvi_fallback(bbox: list) -> dict | None:
    try:
        catalog = pystac_client.Client.open(
            STAC_URL,
            modifier=planetary_computer.sign_inplace,
        )
        
        year = datetime.date.today().year - 1
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
                with rio.open(href) as src:
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
        print(f"MODIS fallback error: {type(e).__name__} - {str(e)[:150]}", file=sys.stderr)
        return None

# --------------------------------------------------
# HEALTH CHECK
# --------------------------------------------------
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "GeoContext Generator API",
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "version": "1.1.0-optimized"
    }

@app.get("/version")
async def get_version():
    return {
        "version": "1.1.0",
        "optimizations": ["clip_before_median", "timeout_guardrails", "graceful_degradation"],
        "max_area_km2": 10.0,
        "ndvi_resolution_m": 20
    }
