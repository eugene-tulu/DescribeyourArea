# GeoContext Frontend

This is the frontend for the GeoContext Generator API, built with Next.js.

## Configuration

To connect the frontend to the deployed backend, set the following environment variable:

```bash
NEXT_PUBLIC_BACKEND_URL=https://describeyourarea-production.up.railway.app
```

### Development

For local development with a locally running backend:

```bash
# Install dependencies
npm install

# Run the development server
npm run dev
```

The frontend will automatically connect to `http://127.0.0.1:8000` by default when no `NEXT_PUBLIC_BACKEND_URL` is set.

### Production

To run with the deployed Railway backend:

```bash
# Set the environment variable
export NEXT_PUBLIC_BACKEND_URL=https://describeyourarea-production.up.railway.app

# Install dependencies
npm install

# Build the application
npm run build

# Start the production server
npm start
```

## Features

- Interactive map for selecting study areas
- GeoJSON upload capability
- Elevation analysis
- NDVI (Normalized Difference Vegetation Index) analysis
- Landcover classification
- AI-powered narrative descriptions
- Responsive design for all devices

## Deployment

The frontend can be deployed to Vercel, Netlify, or any platform that supports Next.js applications.
