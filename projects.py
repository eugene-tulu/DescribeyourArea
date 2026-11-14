"""File-backed project store.

A project is one pipeline configuration: an AOI, a window, the datasets to read,
and the connectors that feed observations. The store is deliberately small and
file-based, matching the job/artefact design: one JSON per project, written
atomically, readable by the backend and (later) by a worker.

The project type is a view selector, not a fork: dMRV, conservancy, cooperative
and custom projects all run through the same enrich/evidence pipeline.
"""

from __future__ import annotations

import json
import os
import secrets
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import rainfall

from contract import Project, ProjectCreate, ProjectSummary


def _dir() -> Path:
    return Path(os.environ.get("PROJECTS_DIR", "/app/.usage/projects"))


def _path(project_id: str) -> Path:
    if not project_id or not all(c.isalnum() for c in project_id):
        raise ValueError("A project id must be alphanumeric.")
    return _dir() / f"{project_id}.json"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _save(project: Project) -> None:
    directory = _dir()
    directory.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=directory, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(project.model_dump(), stream)
        os.replace(temporary, _path(project.id))
    except Exception:
        try:
            os.unlink(temporary)
        except OSError:
            pass
        raise


def _geometry(geojson: dict) -> Optional[dict]:
    if not isinstance(geojson, dict):
        return None
    if geojson.get("type") == "FeatureCollection":
        features = geojson.get("features") or []
        return features[0].get("geometry") if features else None
    if geojson.get("type") == "Feature":
        return geojson.get("geometry")
    return geojson


def create(data: ProjectCreate) -> Project:
    now = _now()
    geometry = _geometry(data.geojson)
    cache_key = rainfall.geometry_hash(geometry) if geometry else None
    project = Project(
        id=secrets.token_hex(8),
        created_at=now,
        updated_at=now,
        cache_key=cache_key,
        **data.model_dump(),
    )
    _save(project)
    return project


def get(project_id: str) -> Optional[Project]:
    path = _path(project_id)
    if not path.exists():
        return None
    return Project.model_validate_json(path.read_text(encoding="utf-8"))


def save_records(project_id: str, records: list, snapshot=None) -> Optional[Project]:
    project = get(project_id)
    if not project:
        return None
    updated = project.model_copy(update={
        "records": [record.model_dump() if hasattr(record, "model_dump") else record
                    for record in records],
        "snapshot": snapshot,
        "updated_at": _now(),
    })
    _save(updated)
    return updated


def list_projects() -> list[ProjectSummary]:
    directory = _dir()
    if not directory.exists():
        return []
    summaries: list[ProjectSummary] = []
    for path in sorted(directory.glob("*.json")):
        try:
            project = Project.model_validate_json(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        summaries.append(ProjectSummary(
            id=project.id,
            name=project.name,
            type=project.type,
            created_at=project.created_at,
            connectors=len(project.connectors),
        ))
    return summaries


def delete(project_id: str) -> bool:
    path = _path(project_id)
    if not path.exists():
        return False
    path.unlink()
    return True