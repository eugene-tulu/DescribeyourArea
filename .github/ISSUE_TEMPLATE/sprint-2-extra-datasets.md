# Sprint 2 — Extra Datasets (Soils, Climate, Hydrology, Population)

## Goal
Enrich baseline analysis with 4 additional environmental datasets: soils, climate normals, hydrology features, and population density.

---

## Target Datasets & Sources

### 1. Soils — Soil Organic Carbon (SOC)
- **Source:** OpenGeoHub STAC
- **STAC endpoint:** `https://stac.opengeohub.org/`
- **Collection:** `biomass.soc_esacci.l4.cpool_go_landmetric`
- **Resolution:** 100m
- **Temporal:** Annual composite 2021 (static)
- **License:** CC-BY-4.0 (commercial OK)
- **Asset:** COG (Cloud Optimized GeoTIFF) in S3
- **Implementation:** 
  - Add OpenGeoHub STAC client
  - Search collection for AOI
  - Load COG with rasterio (signed URL via Planetary Computer token not needed; S3 public)
  - Compute mean SOC (tC/ha) within AOI

### 2. Population — GHS-POP (Gridded Population)
- **Source:** OpenGeoHub STAC (same endpoint)
- **Collection:** `pop.count_ghs_go_landmetric`
- **Resolution:** 100m
- **Temporal:** 2000-01-01, 2005-01-01, 2010-01-01, 2015-01-01, 2020-01-01, 2025-01-01
- **License:** CC-BY-4.0
- **Implementation:**
  - Query latest available year (2025 if available else 2020)
  - Load COG, sum population counts within AOI
  - Compute density = total_pop / area_km2

### 3. Climate — Open-Meteo Climate Normals
- **Source:** Open-Meteo REST API
- **Endpoint:** `https://climate-api.open-meteo.com/v1/climate`
- **Variables:** `temperature_2m_mean`, `precipitation_sum`
- **Time:** 1991-01-01 to 2020-12-31 (30-year normals)
- **Spatial:** Point query at AOI centroid
- **License:** CC-BY-4.0
- **Implementation:**
  - Get centroid from GeoJSON
  - Request daily normals, compute annual averages
  - Return: mean annual temp (°C), annual precipitation (mm)

### 4. Hydrology — OSM Water Features
- **Source:** OpenStreetMap via Overpass API
- **Features:** 
  - `natural=water` (lakes, ponds)
  - `waterway=river`/`stream`/`canal`
- **Implementation:**
  - Query Overpass for bbox + water tags
  - Compute:
    - Total water area (polygons) within AOI
    - Total waterway length (linestrings) within AOI
    - Water cover % = water_area / AOI_area
  - Note: Coverage varies by region; add disclaimer if data sparse

### 5. Admin Boundaries — OSM + Nominatim (already implemented)
- Country via Nominatim reverse (centroid)
- Admin levels 1/2 via Overpass (deferred to later sprint)

---

## Implementation Tasks

- [ ] Add OpenGeoHub STAC client to `main.py` (reusable function)
- [ ] Implement `fetch_soil_soc(bbox, geom)` → dict with mean_soc_tC_ha
- [ ] Implement `fetch_population(bbox, geom)` → dict with total_pop, density_per_km2, year
- [ ] Implement `fetch_climate(centroid)` → dict with mean_temp_c, annual_precip_mm
- [ ] Implement `fetch_hydrology(bbox, geom)` → dict with water_area_km2, water_cover_pct, waterway_length_km
- [ ] Parallelize these calls in `/generate-context` (use `asyncio.gather`)
- [ ] Update `summary` dict schema to include new keys: `soils`, `population`, `climate`, `hydrology`
- [ ] Update `generate_study_area_narrative` prompt to v3 (include new sections)
- [ ] Test with diverse AOIs (different continents, urban/rural, coastal/inland)
- [ ] Document each dataset source, license, and limitation in code comments

---

## Dependencies

**New Python packages:** None over existing (use `requests` for Open-Meteo, `overpy` for OSM)

**Existing:** 
- `pystac-client` (already)
- `rasterio` (already)
- `shapely` (already)

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| OpenGeoHub rate limits | Cache results per AOI hash; respectful query rate |
| Overpass API timeout/rate limit | Set bounding box to AOI + small buffer; add retry logic |
| Climate API outage | Fallback placeholder values with "unavailable" message |
| STAC collection version changes | Pin to specific collection IDs; monitor updates |

---

## Success Criteria

- All 4 datasets return meaningful values for test AOIs in Kenya, Sweden, Brazil
- Combined added latency < 5s (parallel calls)
- No new large dependencies (keep Railway free tier compatible)
- All sources properly cited in narrative

---

## Out of Scope for Sprint 2

- Admin boundaries beyond country level (Overpass query for admin_level 4/6/8)
- Multi-year population trends (just latest year)
- Higher-resolution soil properties (texture, pH) — defer to later
- Hydrological connectivity (stream order, watershed boundaries)
