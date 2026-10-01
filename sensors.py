"""Sensor registry and vegetation-index read path.

One code path serves three sensors, because they differ in ways that are easy to
get wrong in different ways. The most important of those is Landsat's radiometric
offset, which does **not** cancel in NDVI:

    Landsat C2 L2 on Planetary Computer publishes ``scale=2.75e-05, offset=-0.2``
    in reflectance units, and only in the STAC item — the file's own band tags are
    empty. Applying the offset moved a real scene's mean NDVI from 0.085 to 0.147,
    a 73% change. A naive port of the Sentinel-2 path, which is a plain ratio of
    raw uint16 values, therefore produces plausible wrong numbers.

MODIS is different in kind again: ``modis-13Q1-061`` ships NDVI as a finished
16-day product, already cloud-masked by NASA, with ``scale=0.0001`` and a fill of
-3000. There is no band arithmetic and no cloud mask to get wrong, which makes it
both the cheapest and the most defensible source at country scale.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

# Sentinel-2 Scene Classification: a pixel is unusable when it is nodata,
# defective, in shadow, cirrus, or probable cloud. Classes 4 (cloud) and 5 (bright
# cloud) are deliberately kept: their brightness test flags bright semi-arid
# ground and desert as cloud across whole tiles, and rejecting them empties the
# result for exactly the rangeland this service is built for. Residual cloud is
# removed by discarding implausible values instead.
SCL_REJECTED = frozenset({0, 1, 3, 8, 9, 10, 11})
NDVI_MIN_PLAUSIBLE = 0.0
# A month-scale flag, so implausible values stay visible rather than silently gone.
SUSPECT_INDEX_VALUE = -0.2

# Landsat Collection 2 QA_PIXEL bit flags.
QA_FILL = 1 << 0
QA_CLOUD_MASK = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4) | (1 << 5)  # cloud, cirrus, shadow, snow

# Beyond this bounding-box area the 0.25-degree MODIS product is the honest
# source: a 250 m grid is still far finer than a whole conservancy, and its
# cloud masking is done by the producer rather than by us.
MODIS_MIN_AREA_KM2 = 2_000.0


@dataclass(frozen=True)
class Sensor:
    id: str
    label: str
    collection: str
    native_res_m: int
    archive_start: str
    # Reflectance-to-value conversion, applied before any band arithmetic.
    scale: float = 1.0
    offset: float = 0.0
    red: Optional[str] = None
    nir: Optional[str] = None
    cloud: Optional[str] = None
    cloud_mask: str = "scl"          # "scl" | "qa_bits" | "product"
    ndvi_asset: Optional[str] = None  # set when NDVI is a product, not computed
    default_max_scenes: int = 4
    revisit_days: int = 5
    # Whether this sensor's own cloud mask is authoritative, or a deliberate
    # relaxation that trades accuracy for coverage. Measured cross-sensor over the
    # same 60 km2 study area and 90-day window: MODIS 0.418 (masked by NASA) and
    # Landsat 0.357 (QA_PIXEL bits) agree to within 0.06, while Sentinel-2 read
    # 0.161 — 0.20 low — because its SCL classes 4 and 5 are kept, and cloud drags
    # the median down. That relaxation is still right over bright desert, where
    # those same classes are a known false positive, but it means a Sentinel-2
    # composite is the least trustworthy of the three in cloudy conditions.
    mask_authoritative: bool = True

    @property
    def computes_ndvi(self) -> bool:
        return self.ndvi_asset is None

    def provenance(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "collection": self.collection,
            "native_resolution_m": self.native_res_m,
            "archive_start": self.archive_start,
            "ndvi": "computed from bands" if self.computes_ndvi else f"product ({self.ndvi_asset})",
            "cloud_mask": self.cloud_mask,
            "scale": self.scale,
            "offset": self.offset,
            "revisit_days": self.revisit_days,
        }


SENTINEL2 = Sensor(
    id="sentinel-2",
    label="Sentinel-2 L2A",
    collection="sentinel-2-l2a",
    native_res_m=10,
    archive_start="2015-06-27",
    scale=1.0,
    offset=0.0,
    red="B04",
    nir="B08",
    cloud="SCL",
    cloud_mask="scl",
    default_max_scenes=4,
    revisit_days=5,
    # See the note on mask_authoritative: the SCL relaxation costs about 0.2 NDVI
    # over the study area measured, so it is not the automatic choice.
    mask_authoritative=False,
)

LANDSAT = Sensor(
    id="landsat",
    label="Landsat Collection 2 Level 2",
    collection="landsat-c2-l2",
    native_res_m=30,
    archive_start="1982-08-22",
    # Published in the STAC item only, in reflectance units, and it does not
    # cancel in the NDVI ratio.
    scale=2.75e-05,
    offset=-0.2,
    red="red",
    nir="nir08",
    cloud="qa_pixel",
    cloud_mask="qa_bits",
    default_max_scenes=4,
    revisit_days=8,
)

MODIS = Sensor(
    id="modis",
    label="MODIS MOD13Q1 (250 m, 16-day)",
    collection="modis-13Q1-061",
    native_res_m=250,
    archive_start="2000-02-18",
    scale=0.0001,
    offset=0.0,
    cloud_mask="product",
    ndvi_asset="250m_16_days_NDVI",
    default_max_scenes=6,
    revisit_days=16,
)

SENSORS = {s.id: s for s in (SENTINEL2, LANDSAT, MODIS)}
DEFAULT_SENSOR = SENTINEL2


def get_sensor(sensor_id: Optional[str]) -> Sensor:
    if sensor_id in (None, "", "auto"):
        return DEFAULT_SENSOR
    try:
        return SENSORS[str(sensor_id).strip().lower()]
    except KeyError as exc:
        raise ValueError(
            f"unknown sensor {sensor_id!r}; choose from {sorted(SENSORS)} or 'auto'"
        ) from exc


def select_sensor(
    sensor_id: Optional[str],
    *,
    window_start: str,
    bbox_area_km2: float,
) -> tuple[Sensor, str]:
    """Choose a sensor, returning it with the reason it was chosen.

    Note that the MODIS step is unreachable from the synchronous path, where it
    looks like dead code and is not: ``select_sensor`` runs inside the vegetation
    handler, and the synchronous area cap has already refused everything past
    100 km2 by then. It is reachable through the worker and by naming the sensor.
    A reader comparing the ladder with the cap will find the third step
    unreachable and should not conclude the ladder is wrong.

    Sentinel-2 for recent windows: finest resolution and shortest revisit.
    Landsat for anything older: 30 m back to 1982, and the only source that can
    answer a pre-2015 question. MODIS beyond the synchronous area budget, where
    its 250 m product is far finer than the study area and its cloud masking is
    the producer's rather than ours.
    """
    if sensor_id not in (None, "", "auto"):
        return get_sensor(sensor_id), "explicitly requested"

    if bbox_area_km2 >= MODIS_MIN_AREA_KM2:
        return MODIS, (
            f"study area is {bbox_area_km2:,.0f} km2, at or beyond the "
            f"{MODIS_MIN_AREA_KM2:,.0f} km2 budget for a 10 m or 30 m source"
        )
    if window_start < SENTINEL2.archive_start:
        return LANDSAT, (
            f"requested window starts {window_start}, before Sentinel-2's "
            f"{SENTINEL2.archive_start} archive"
        )
    # Landsat over Sentinel-2 by default, despite being coarser: its QA_PIXEL mask
    # is authoritative while the SCL relaxation is not, and at 30 m a 100 km2 study
    # area gains nothing from 10 m. Sentinel-2 still wins for the days immediately
    # before a request, where Landsat has no scene yet.
    freshness_days = (
        datetime.date.today()
        - datetime.date.fromisoformat(window_start)
    ).days
    if freshness_days > LANDSAT.revisit_days:
        return LANDSAT, (
            f"window opens {freshness_days} days ago, beyond Landsat's "
            f"{LANDSAT.revisit_days}-day revisit, and its cloud mask is authoritative"
        )
    return SENTINEL2, (
        f"window opens {freshness_days} days ago, where only Sentinel-2 has a scene; "
        "its relaxed cloud mask may read low in cloudy conditions"
    )


# --------------------------------------------------------------------------
# Cloud masks
# --------------------------------------------------------------------------

def cloud_mask(sensor: Sensor, values: np.ndarray, cloud_values: np.ndarray) -> np.ndarray:
    """True where a pixel is usable for a vegetation index."""
    if sensor.cloud_mask == "scl":
        rejected = np.isin(
            np.nan_to_num(cloud_values, nan=-1.0).astype("int16"), tuple(SCL_REJECTED)
        )
        return ~rejected
    if sensor.cloud_mask == "qa_bits":
        bits = np.nan_to_num(cloud_values, nan=0.0).astype("int64")
        clear = (bits & QA_CLOUD_MASK) == 0
        fill = (bits & QA_FILL) != 0
        return clear & ~fill
    # "product": the producer already masked it.
    return np.isfinite(values)


def plausible(values: np.ndarray) -> np.ndarray:
    """Drop values no land surface produces, which is how residual cloud shows up."""
    keep = np.isfinite(values) & (values >= NDVI_MIN_PLAUSIBLE) & (values <= 1.0)
    return keep


# --------------------------------------------------------------------------
# Resolution policy
# --------------------------------------------------------------------------
# The area caps in main.py are a guard on the *request* budget, not a claim about
# what the data can do. What actually costs is pixels, and pixels are set by
# resolution. Measured on 2026-09-26:
#
#   worldcover 10 m, 5,500 km2   216M px  176 s   1,205 MB
#   worldcover 60 m, 5,500 km2   6.0M px  6.0 s
#   worldcover 100 m, 5,500 km2  2.2M px  1.3 s
#
# and a 2,462 km2 landscape already returns a vegetation index in 10.7 s from
# MODIS and 56.9 s from Landsat, both of which work today behind the cap.
#
# So the policy targets a pixel budget rather than an area budget, and a large
# study area is answered at a coarser resolution rather than refused. The
# resolution is reported with every result, because a 100 m land-cover
# composition is a different kind of claim from a 10 m one.
RESOLUTION_STEPS = (
    # (max bounding-box km2, metres, why)
    (100.0, 20, "native for the composite; a small area can afford the finest grid"),
    (1_000.0, 60, "100 km2 at 60 m is about 0.4M pixels, comfortably inside a request"),
    (10_000.0, 100, "10,000 km2 at 100 m is about 1M pixels; finer buys nothing a reader can see"),
)
COARSEST_RESOLUTION_M = 250

# A PIXEL_BUDGET used to be defined here and never read. The resolution ladder
# above replaced it as the policy: it asks what resolution the *reader* can
# resolve at this size, which is a question about the claim, whereas a pixel
# budget is a question about this server's memory. Those diverge -- the ladder
# says 100 m for 10,000 km2, and a 3M pixel budget at 100 m would refuse far
# smaller areas than that. Deleting it was the honest option; wiring it up would
# have introduced a second policy that contradicts the first.


def resolution_for_area(bbox_area_km2: float, *, native_m: int = 10) -> tuple[int, str]:
    """Metres per pixel for a study area of this size, and why.

    Returns the resolution to *request*, not the native one, so a caller can see
    what was actually analysed.
    """
    for limit, metres, reason in RESOLUTION_STEPS:
        if bbox_area_km2 <= limit:
            # Never ask for finer than the source publishes: a 250 m MODIS product
            # read at 20 m would be upsampling, which invents detail.
            chosen = max(metres, native_m)
            if chosen != metres:
                return chosen, (
                    f"the source's own resolution is {native_m} m, so no finer grid "
                    "is requested"
                )
            return metres, reason
    side_km = math.sqrt(bbox_area_km2)
    return COARSEST_RESOLUTION_M, (
        f"{bbox_area_km2:,.0f} km2 is beyond the synchronous budget, so it is read at "
        f"{COARSEST_RESOLUTION_M} m, about {side_km / (COARSEST_RESOLUTION_M / 1000):,.0f} "
        "pixels across"
    )
