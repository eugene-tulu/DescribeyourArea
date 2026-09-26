# Python 3.12 has current security support and avoids a stale runtime image.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system deps for rasterio & GDAL
RUN apt-get update && apt-get install -y --no-install-recommends \
    gdal-bin \
    libgdal-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install dependencies
RUN pip install -r requirements.txt

# Copy project files
COPY . .

# The service has no reason to write application source or run as root.
RUN groupadd --system app && useradd --system --gid app app \
    && chown -R app:app /app
USER app

# Fetches the rainfall portfolio before serving; see the script for why a failed
# pull is not fatal.
RUN chmod +x /app/docker-entrypoint.sh

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI app
ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
