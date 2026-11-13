"""Connector adapters: turn a source into the pipeline's observation shape.

The pipeline is source-agnostic. Each connector knows how to authenticate,
query and map its fields; everything after `features_to_observations` is the
same enrich/evidence path. ArcGIS/Survey123 is first because it is the common
collector in the Mara and dMRV workflows; EarthRanger and COPA follow the same
interface.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlencode

import requests
from shapely.geometry import shape

from contract import Observation


class ConnectorError(Exception):
    pass


def arcgis_query_url(
    base_url: str,
    layer: str,
    bbox: list[float],
    where: Optional[str] = None,
    token: Optional[str] = None,
    limit: int = 2000,
) -> str:
    base = (base_url or "").strip().rstrip("/")
    if not base:
        raise ConnectorError("An ArcGIS service URL is required.")
    if len(bbox) != 4:
        raise ConnectorError("The AOI bbox must be [minLon, minLat, maxLon, maxLat].")

    if base.endswith("/query"):
        query_url = base
    elif re.search(r"/(MapServer|FeatureServer)(/\d+)?$", base):
        query_url = f"{base}/query" if re.search(r"/\d+$", base) else f"{base}/{layer}/query"
    else:
        query_url = f"{base}/{layer}/query"

    params = {
        "where": where or "1=1",
        "geometry": ",".join(str(value) for value in bbox),
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
        "outFields": "*",
        "returnGeometry": "true",
        "f": "geojson",
        "resultRecordCount": str(limit),
    }
    if token:
        params["token"] = token
    return f"{query_url}?{urlencode(params)}"


def _epoch_to_iso(value: Any) -> Optional[str]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value / 1000, tz=timezone.utc).isoformat()
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str) and value:
        return value
    return None


def features_to_observations(
    features: list[dict[str, Any]],
    id_field: Optional[str] = None,
    date_field: Optional[str] = None,
) -> list[Observation]:
    observations: list[Observation] = []
    for index, feature in enumerate(features):
        geometry = feature.get("geometry")
        if not geometry:
            continue
        try:
            shape_ = shape(geometry)
        except Exception:
            continue
        if shape_.is_empty:
            continue
        point = shape_.representative_point()
        properties = feature.get("properties") or {}
        identifier = properties.get(id_field) if id_field else properties.get("OBJECTID")
        timestamp = _epoch_to_iso(properties.get(date_field)) if date_field else None
        observer = properties.get("observer") or properties.get("observer_name") or properties.get("enumerator")
        type_ = properties.get("type") or properties.get("land_use") or properties.get("condition")
        attributes = {
            key: value
            for key, value in properties.items()
            if isinstance(value, (str, int, float, bool, type(None)))
        }
        observations.append(Observation(
            id=str(identifier) if identifier is not None else f"arcgis-{index + 1}",
            lon=float(point.x),
            lat=float(point.y),
            timestamp=timestamp,
            observer=str(observer) if observer is not None else None,
            type=str(type_) if type_ is not None else None,
            attributes=attributes,
        ))
    return observations


def fetch_arcgis(config: dict[str, Any]) -> tuple[list[Observation], list[str]]:
    try:
        url = arcgis_query_url(
            config.get("url", ""),
            str(config.get("layer_id", "")),
            config.get("bbox") or [],
            config.get("where"),
            config.get("token"),
            int(config.get("limit", 2000)),
        )
    except (ConnectorError, TypeError, ValueError) as exc:
        return [], [str(exc)]

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        payload = response.json()
    except Exception as exc:
        return [], [f"ArcGIS request failed: {exc}"]

    if isinstance(payload, dict) and payload.get("error"):
        error = payload["error"]
        message = error.get("message", error) if isinstance(error, dict) else error
        return [], [f"ArcGIS error: {message}"]
    features = payload.get("features") if isinstance(payload, dict) else None
    if features is None:
        return [], ["The ArcGIS response had no features."]
    return features_to_observations(
        features,
        config.get("id_field"),
        config.get("date_field"),
    ), []