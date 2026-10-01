"""Scale, decided once from the registry rather than per product.

There were four places deciding how big an area could be and how finely to read
it, and they disagreed. ``MAX_SYNC_BBOX_KM2`` guarded the whole request,
``MAX_LANDCOVER_BBOX_KM2`` guarded one module, ``sensors.RESOLUTION_STEPS`` chose
a vegetation resolution, and a fourth budget was read independently in two
modules. The registry now knows each product's native resolution and the area over
which its measure means anything, so this module derives the rest.

**The distinction that matters: affordable is not meaningful.** A product can be
affordable over far more ground than its measure is *about*. ERA5 is the sharp
case -- affordable indefinitely, meaningful from about one cell upward, and below
that the number describes a 774 km2 cell rather than the outline drawn inside it.
Conflating the two is how a cap set for memory gets quoted as a statement about
what the data can say.

So admission answers two separate questions and reports both:

- may we read this? -- pixels and time, a property of this server;
- does the number mean anything? -- the meaningful-area range, a property of the
  product, which no amount of server capacity can extend.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import registry


@dataclass(frozen=True)
class Admission:
    """Whether a read is allowed, and whether its result will mean anything."""

    allowed: bool
    reason: Optional[str] = None
    meaning: bool = True
    meaning_note: Optional[str] = None
    pixels: Optional[int] = None
    resolution_m: Optional[float] = None

    def describe(self) -> dict:
        out = {"allowed": self.allowed, "meaningful": self.meaning}
        if self.reason:
            out["reason"] = self.reason
        if self.meaning_note:
            out["meaningful_note"] = self.meaning_note
        if self.pixels is not None:
            out["pixels"] = self.pixels
        if self.resolution_m is not None:
            out["resolution_m"] = self.resolution_m
        return out


def pixels_for(area_km2: float, resolution_m: float) -> int:
    """Approximate pixel count, without pretending to be exact.

    A kilometre is a kilometre squared; the small error against a geodesic
    calculation is well under the difference between a resolution step and the
    next one, which is the only decision this number feeds.
    """
    side_km = resolution_m / 1000.0
    return int(max(1, (math.sqrt(max(area_km2, 0.0)) / side_km) ** 2))


def coarsen_to_meaningful(product_key: str, area_km2: float,
                          requested_m: float) -> float:
    """The resolution to read at, never finer than the product publishes.

    Reading a 250 m product at 20 m does not add information; it manufactures
    pixels. So the floor is the product's own resolution and the ladder only ever
    moves coarser than that.
    """
    product = registry.get(product_key)
    native = product.native_resolution_m if product else None
    return max(float(requested_m), float(native)) if native else float(requested_m)


def admit(product_key: str, area_km2: float, *,
          max_area_km2: float, resolution_m: Optional[float] = None) -> Admission:
    """Both questions, answered and both reported."""
    product = registry.get(product_key)
    if product is None:
        return Admission(False, reason=f"{product_key!r} is not a registered product")

    resolution = coarsen_to_meaningful(product_key, area_km2, resolution_m or 20.0)
    pixels = pixels_for(area_km2, resolution)

    if area_km2 > max_area_km2:
        return Admission(
            False,
            reason=(f"{product.label} is read inside a request up to "
                    f"{max_area_km2:,.0f} km²; this area is {area_km2:,.0f} km²"),
            resolution_m=resolution, pixels=pixels)

    floor = product.meaningful_min_km2
    if floor is not None and area_km2 < floor:
        return Admission(
            True, meaning=False,
            meaning_note=(
                f"{product.label} is a {product.resolution_km():,.0f} km product, so "
                f"one cell covers about {floor:,.0f} km². This outline is smaller, and "
                f"the figure is that cell's rather than the outline's."),
            resolution_m=resolution, pixels=pixels)

    return Admission(True, meaning=True, resolution_m=resolution, pixels=pixels)


def resolution_ladder(area_km2: float, native_m: float) -> tuple[float, str]:
    """Vegetation's ladder, expressed as a function of the area.

    Kept as a function rather than a table so the table and the reasoning cannot
    drift apart, and so a product's own native resolution is a floor.
    """
    import sensors

    for max_km2, metres, why in sensors.RESOLUTION_STEPS:
        if area_km2 <= max_km2:
            resolution = float(max(metres, native_m))
            if resolution != metres:
                why = (f"{why}, floored at the {native_m:g} m the source "
                       f"publishes rather than upsampled to {metres:g} m")
            return resolution, why
    return (float(max(sensors.COARSEST_RESOLUTION_M, native_m)),
            f"past the last step at {sensors.COARSEST_RESOLUTION_M} m")
