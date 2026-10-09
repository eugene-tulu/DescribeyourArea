"""The published response contract.

The endpoint used to declare ``summary: Dict[str, Any]``, which made the schema
meaningless: FastAPI served an OpenAPI document that said nothing about ``dem``,
``ndvi``, ``landcover`` or ``rainfall``, and the frontend carried twelve
hand-written TypeScript interfaces that had to match those dicts with nothing
asserting it. That is how a duplicated literal drifted and shipped a stale limit
to production.

Two things live here, and the second is the point of the first.

**A declared schema.** ``/openapi.json`` is now a real contract a partner or a
client generator can use, and a test asserts a served response conforms to it.

**Epistemic status on every figure.** A number without a stated kind is an
assertion pretending to be a measurement. The distinction matters most for rainfall,
which is a *reanalysis* — a model output assimilating observations — and not a
gauge reading, even though it arrives looking like any other number. Terrain and
land cover are observed products; an index derived from them is derived. Where a
source cannot be verified, ``unconfirmed`` says so instead of the value being
quietly absent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

# What kind of thing a number is. Deliberately few, because a taxonomy nobody
# applies is worse than none.
#
#   observed   a measurement of the surface itself
#   derived    computed from observed values
#   modelled   produced by a model that assimilates data, such as a reanalysis
#   unconfirmed the source could not be verified, so the value is not claimed
EvidenceStatus = Literal["observed", "derived", "modelled", "unconfirmed"]

# What a module did. The success case belongs in the type: a contract that omits
# it is a contract that will be violated by the first successful response, which
# is exactly what happened when this landed.
UnavailableReason = Literal[
    "ok",
    "not_requested",
    "not_computed",
    "area_exceeded",
    "no_valid_pixels",
    "unavailable",
    "skipped",
    "error",
    "busy",
]


class ModuleEvidence(BaseModel):
    """How a figure came to exist, attached to the figure itself."""

    status: EvidenceStatus
    source: Optional[str] = None
    method: Optional[str] = None
    note: Optional[str] = None
    doi: Optional[str] = None
    license: Optional[str] = None
    retrieved: Optional[str] = None


class DemResult(BaseModel):
    status: UnavailableReason = "ok"
    mean: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    std: Optional[float] = None
    elevation_range_m: Optional[float] = None
    terrain_type: Optional[str] = None
    valid_pixel_count: Optional[int] = None
    valid_pixel_fraction: Optional[float] = None
    error: Optional[str] = None
    evidence: ModuleEvidence


class LandcoverResult(BaseModel):
    status: UnavailableReason = "ok"
    classes: Dict[str, float] = Field(default_factory=dict)
    dominant_class: Optional[str] = None
    dominant_percentage: Optional[float] = None
    valid_pixel_count: Optional[int] = None
    valid_pixel_fraction: Optional[float] = None
    error: Optional[str] = None
    evidence: ModuleEvidence


class SensorProvenance(BaseModel):
    id: Optional[str] = None
    label: Optional[str] = None
    collection: Optional[str] = None
    native_resolution_m: Optional[int] = None
    archive_start: Optional[str] = None
    ndvi: Optional[str] = None
    cloud_mask: Optional[str] = None
    scale: Optional[float] = None
    offset: Optional[float] = None
    revisit_days: Optional[int] = None
    mask_authoritative: Optional[bool] = None


class VegetationResult(BaseModel):
    status: UnavailableReason = "ok"
    mean: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    std: Optional[float] = None
    p25: Optional[float] = None
    p75: Optional[float] = None
    scene_count: Optional[int] = None
    scenes_examined: Optional[int] = None
    valid_pixel_count: Optional[int] = None
    valid_pixel_fraction: Optional[float] = None
    resolution_m: Optional[int] = None
    method: Optional[str] = None
    sensor: Optional[SensorProvenance] = None
    sensor_reason: Optional[str] = None
    window: Optional[Dict[str, Optional[str]]] = None
    scene_ids: List[str] = Field(default_factory=list)
    scene_dates: List[str] = Field(default_factory=list)
    warning: Optional[str] = None
    error: Optional[str] = None
    evidence: ModuleEvidence


class RainfallMonth(BaseModel):
    month: str
    precip_mm: float
    normal_mm: Optional[float] = None
    anomaly_mm: Optional[float] = None
    anomaly_pct: Optional[float] = None
    suspect: Optional[str] = None


class RainfallClimatology(BaseModel):
    standard: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    annual_mean_mm: Optional[float] = None
    monthly_mean_mm: Dict[str, float] = Field(default_factory=dict)


class RainfallResult(BaseModel):
    status: UnavailableReason = "not_computed"
    reason: Optional[str] = None
    message: Optional[str] = None
    label: Optional[str] = None
    cache_key: Optional[str] = None
    grid_cells: Optional[int] = None
    resolution_km: Optional[float] = None
    window: Optional[Dict[str, str]] = None
    climatology: Optional[RainfallClimatology] = None
    series: List[RainfallMonth] = Field(default_factory=list)
    summary: Optional[Dict[str, Any]] = None
    suspect_months: List[str] = Field(default_factory=list)
    evidence: ModuleEvidence


class AnalysisMetadata(BaseModel):
    bbox_area_km2: Optional[float] = None
    # The box itself, so a client can ask a boundary service what this outline
    # sits inside without measuring it a second time. Declared here because the
    # response is typed and Pydantic drops fields the model does not declare --
    # so adding it to the payload alone put it nowhere.
    bbox: Optional[List[float]] = None
    datasets: List[str] = Field(default_factory=list)
    mode: str = "synchronous"
    # The resolution actually used, which for a large area is not the native one.
    applied_resolution_m: Optional[Dict[str, int]] = None


class AdminContext(BaseModel):
    # The administrative chain the outline sits in, from the boundary service:
    # country, then region (ADM1), then district (ADM2). A country name alone
    # names a continent-sized unit; the levels under it name the place.
    country: Optional[str] = None
    admin1: Optional[str] = None
    admin2: Optional[str] = None


class ContextSummary(BaseModel):
    dem: Optional[DemResult] = None
    landcover: Optional[LandcoverResult] = None
    ndvi: Optional[VegetationResult] = None
    rainfall: Optional[RainfallResult] = None
    country: Optional[str] = None
    admin: Optional[AdminContext] = None
    scene_dates: Dict[str, str] = Field(default_factory=dict)
    scene_ids: Dict[str, str] = Field(default_factory=dict)
    analysis: Optional[AnalysisMetadata] = None
    # A uniform list of measure dicts, each with value/units/evidence/extent.
    # Built via measure.Measure.descripe(); see measure.py docstring. Absent when
    # no dynamic measures were produced.
    measures: Optional[List[Dict[str, Any]]] = None
    # Anything a reader should know before relying on the numbers above: a module
    # that was skipped and why, a repaired boundary, months worth a second look,
    # or a source that stops short of today.
    caveats: List[str] = Field(default_factory=list)


class ContextResponse(BaseModel):
    summary: ContextSummary
