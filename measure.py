"""The result, as one object, whatever produced it.

Three products at three resolutions and two evidence classes are the same kind of
thing: a number, the ground it actually stands for, what kind of number it is,
how old it is, and how to take it away. That shape lived in the client, per card,
which is why the client is four hardcoded cards and why the two honesty fixes
that keep recurring -- state the cell count, state the series end -- have to be
remembered rather than enforced.

``Measure`` makes them required. A dynamic measure cannot be built without an
``observed_through`` date, and a gridded one cannot be built without its extent,
because the constructor derives them from the product rather than accepting them
as optional decoration.

The evidence class is inherited from the registry, not restated. That is the
whole design: there is exactly one place a product's properties are declared, and
a result cannot disagree with it.

An ``uncertain`` field exists because some answers genuinely are, and forcing
confidence we do not have is worse than admitting it -- a vegetation composite
over a persistently cloudy area reads low, and the honest response to that is a
caveat, not a number.
"""

from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import Any, Optional

import registry


def _utc_now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


class MeasureError(ValueError):
    """A measure that cannot honestly be built. Better than one that lies."""


@dataclass(frozen=True)
class Extent:
    """The ground a number actually stands for.

    Not the area the user drew. For a gridded product those differ, and the
    difference is the whole question: an ERA5 "area mean" over a 4 km2 paddock
    is one 774 km2 cell, and a reader who is not told that reads four square
    kilometres of implied precision into a number that came from far more ground
    than that.
    """

    requested_area_km2: Optional[float] = None
    covered_area_km2: Optional[float] = None
    cell_count: Optional[int] = None
    native_resolution_m: Optional[float] = None
    native_grid_degrees: Optional[float] = None
    window_start: Optional[str] = None
    window_end: Optional[str] = None

    @property
    def over_specified(self) -> bool:
        """Whether the request was finer than the product can support.

        True is the normal case for rainfall over a small area, and it is exactly
        what a reader needs to be told rather than left to infer.
        """
        if self.requested_area_km2 is None or self.covered_area_km2 is None:
            return False
        return self.covered_area_km2 > self.requested_area_km2 * 1.5

    def describe(self) -> dict:
        out = {
            "requested_area_km2": self.requested_area_km2,
            "covered_area_km2": self.covered_area_km2,
            "cell_count": self.cell_count,
            "native_resolution_m": self.native_resolution_m,
            "native_grid_degrees": self.native_grid_degrees,
            "over_specified": self.over_specified,
        }
        if self.window_start or self.window_end:
            out["window"] = {"start": self.window_start, "end": self.window_end}
        return out


@dataclass(frozen=True)
class Measure:
    """One number, with everything a reader needs in order to trust it."""

    product_key: str
    value: Any
    units: str
    extent: Extent = field(default_factory=Extent)
    observed_through: Optional[str] = None
    evidence: str = ""
    uncertainty: Optional[str] = None
    computed_at: str = field(default_factory=_utc_now)
    caveats: tuple[str, ...] = field(default_factory=tuple)
    extra: dict = field(default_factory=dict)

    def __post_init__(self):
        product = registry.get(self.product_key)
        if product is None:
            raise MeasureError(
                f"{self.product_key!r} is not a registered product; a measure for "
                f"it cannot be built, and building one would defeat the registry"
            )
        if not self.evidence:
            object.__setattr__(self, "evidence", product.evidence)
        if not self.caveats:
            object.__setattr__(self, "caveats", product.caveats)

        # A dynamic product with no observed-through date is the exact failure
        # this exists to prevent: four cards in a row, six months apart, none of
        # them saying so. Derived from the registry rather than remembered.
        if self.observed_through is None and product.latency_days:
            raise MeasureError(
                f"{self.product_key} is dynamic (latency "
                f"{product.latency_days} days) so a measure of it must carry "
                f"observed_through; that date is the freshness a reader needs"
            )

    @property
    def product(self) -> registry.Product:
        return registry.get(self.product_key)

    def describe(self) -> dict:
        product = self.product
        out = {
            "product": self.product_key,
            "measure": product.measure,
            "label": product.label,
            "value": self.value,
            "units": self.units,
            "evidence": self.evidence,
            "extent": self.extent.describe(),
            "observed_through": self.observed_through,
            "computed_at": self.computed_at,
            "source": product.source,
            "doi": product.doi,
            "caveats": list(self.caveats),
        }
        if self.uncertainty:
            out["uncertainty"] = self.uncertainty
        if self.extra:
            out["detail"] = self.extra
        return out


def measures_for(entries: list[Measure]) -> list[dict]:
    """The published form of a set of results, in one order, for a client.

    Deliberately a list of the same shape regardless of which products are
    present. A client that can render this can render a product that does not
    exist yet, which is the first of the two acceptance tests on the roadmap.
    """
    return [m.describe() for m in entries]


def headline(measure: Measure) -> str:
    """One sentence, in the reader's terms, with the honesty attached.

    Not a formatter. The point is that the qualification is not optional: a
    number whose ground is far larger than the outline cannot be rendered
    without saying so, whichever card is drawing it.
    """
    product = measure.product
    if measure.observed_through:
        fresh = f" as of {measure.observed_through[:10]}"
    else:
        fresh = ""
    if measure.extent.over_specified and measure.extent.covered_area_km2:
        return (
            f"{product.label} {measure.value} {measure.units}{fresh} — "
            f"{measure.evidence}, read over {measure.extent.covered_area_km2:,.0f} km² "
            f"of a single {product.resolution_km() or 0:,.0f} km grid cell for a "
            f"{measure.extent.requested_area_km2 or 0:,.0f} km² outline"
        )
    return f"{product.label} {measure.value} {measure.units}{fresh} — {measure.evidence}"
