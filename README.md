# GeoContextualize

A geospatial analysis API that generates contextual information about any location on Earth using satellite imagery and geospatial datasets.

## Features

- Elevation analysis using NASADEM
- Landcover classification using ESA WorldCover
- NDVI (Normalized Difference Vegetation Index) analysis using Sentinel-2 data
- AI-powered narrative descriptions using Google Gemini
- Support for custom GeoJSON areas
- Render and Railway deployment ready

## Deployment

The API is deployed on both Render and Railway:

- **Railway**: https://describeyourarea-production.up.railway.app
- **Render**: (URL to be added)

## Backend (API)

Built with FastAPI, the backend provides:

- `/generate-context` - Main endpoint for generating geospatial context
- `/health` - Health check endpoint
- `/version` - Version information endpoint
- CORS support for web applications
- Memory and timeout constraints for free-tier hosting

## Frontend

The frontend is a Next.js application located in the `client/` directory that provides:

- Interactive map interface
- GeoJSON upload capability
- Visual feedback for analysis results
- Responsive design

## Technologies Used

### Backend
- FastAPI
- Rasterio
- PySTAC Client
- Microsoft Planetary Computer
- ODC STAC
- Google Generative AI
- XArray, RioxArray

### Frontend
- Next.js
- React
- Leaflet
- Tailwind CSS

## Setup

### Backend Setup

1. Clone the repository
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env to add your GEMINI_API_KEY
   ```
4. Run the server:
   ```bash
   uvicorn main:app --reload
   ```

### Frontend Setup

1. Navigate to the client directory:
   ```bash
   cd client
   ```
2. Install dependencies:
   ```bash
   npm install
   ```
3. Set environment variable to use the deployed backend:
   ```bash
   export NEXT_PUBLIC_BACKEND_URL=https://describeyourarea-production.up.railway.app
   ```
4. Run the development server:
   ```bash
   npm run dev
   ```

## API Endpoints

- `POST /generate-context` - Generate geospatial context for a GeoJSON area
- `GET /health` - Health check
- `GET /version` - Version information

## Parameters

The `/generate-context` endpoint accepts:
- `geojson`: GeoJSON object defining the area of interest
- `include_narrative`: Boolean to include AI-generated narrative
- `audience`: Target audience for narrative ("academic", "investor", "farmer", "policy")
- `include_ndvi`: Boolean to include NDVI analysis

## Architecture

The system leverages Microsoft Planetary Computer to access:
- NASADEM for elevation data
- ESA WorldCover for landcover classification
- Sentinel-2 L2A for NDVI analysis
- MODIS as fallback for NDVI when needed

## Constraints

The system includes several constraints for reliable operation on free-tier hosting:
- Maximum area size of 10 km² for NDVI analysis
- Timeout protection with 15-second limits
- Memory-safe processing with chunked operations
- Fallback mechanisms when constraints are exceeded