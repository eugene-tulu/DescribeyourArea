"""The question, with the date range primary and everything else derived from it.

The interface used to offer 1, 3, 10 and 30 years and derive a date pair. That is
a lossy abstraction sitting in front of what the reader actually wants: nobody
asks for "ten years", they ask for "the 2019/20 drought" or "since the last
rains" or "the 1997/98 El Nino", and a preset control cannot express any of the
three. A researcher asking about 1997 had to pick 30 years and slice it.

So dates are primary. Everything the presets were doing is derived here instead,
which also makes the derivations visible rather than implicit:

- the **span**, and from it the **bin width** -- 390 monthly bars in a chart a
  person can read is not the same claim as 13 annual ones, and which of those you
  get should be a stated consequence of the dates rather than a silent default;
- **which sensors can answer at all**, from their archive starts, so a 1994
  request is answered with Landsat and a 1996 one is refused rather than silently
  answered with a sensor that began in 2015;
- the **comparison normal**, which for rainfall is the published 1991-2020 period
  and for vegetation is the years MOD13Q1 actually has;
- the **alert horizon**, so "when do I act" is a question about a date range
  rather than a hard-coded twelve months.

This module holds no data access. It is the part that decides what to ask for,
which is what makes the question layer a layer rather than four routes that each
re-derive their own idea of a window.
"""

from __future__ import annotations

import calendar
import datetime
from dataclasses import dataclass, field
from typing import Optional

import registry
import sensors

# Bin widths, in months. The threshold is a readability judgement and is stated
# as one: at more than ~120 monthly bars the individual columns stop being
# separable and the honest rendering is yearly.
MONTHLY_BIN_LIMIT = 120

DEFAULT_PRESETS_YEARS = (1, 3, 10, 30)


@dataclass(frozen=True)
class Window:
    """A date range, and everything that follows from it."""

    start: datetime.date
    end: datetime.date

    @property
    def days(self) -> int:
        return (self.end - self.start).days

    @property
    def months(self) -> int:
        return (self.end.year - self.start.year) * 12 + (self.end.month - self.start.month) + 1

    @property
    def bin_months(self) -> int:
        """How wide one bar should be, given the span.

        A consequence of the dates, not a choice offered alongside them.
        """
        return 1 if self.months <= MONTHLY_BIN_LIMIT else 12

    @property
    def label(self) -> str:
        """Described relative to its own end, never relative to today.

        An earlier version said "the past year" for anything between one and two
        years long, which is false for a 1997 window and false for a 2019 one.
        A label that is only correct for windows ending now is a label that is
        quietly wrong for the historical questions this design exists to allow.
        """
        if self.days == 0:
            return f"the month of {self.end:%Y-%m}"
        if self.days <= 45:
            return f"{self.days} days to {self.end:%Y-%m}"
        if self.months <= 12:
            return f"{self.months} months to {self.end:%Y-%m}"
        years = self.days / 365.25
        return f"{years:.1f} years to {self.end:%Y-%m}"

    @property
    def is_recent_enough_to_composite(self) -> bool:
        """Whether a window this short should be composited rather than binned.

        A composite averages the scenes inside the window into one number; a
        series bins them. One month is a composite, three years is a series.
        """
        return self.days <= 120

    def describe(self) -> dict:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat(),
            "days": self.days,
            "months": self.months,
            "bin_months": self.bin_months,
            "label": self.label,
            "composited": self.is_recent_enough_to_composite,
        }


def parse_window(start: Optional[str] = None, end: Optional[str] = None,
                 *, years: Optional[int] = None,
                 today: Optional[datetime.date] = None) -> Window:
    """Dates in, a ``Window`` out. Presets are a convenience on the way in.

    ``years`` is accepted because the existing control offers it, not because it
    is the primary interface: it is resolved to a concrete range immediately, so
    nothing downstream can tell the difference and nothing downstream depends on
    it.
    """
    today = today or datetime.datetime.now(datetime.UTC).date()
    end_date = _as_date(end, "end") if end else today
    if years is not None:
        if years <= 0:
            raise ValueError("a window of zero or fewer years is not a window")
        try:
            start_date = end_date.replace(year=end_date.year - years)
        except ValueError:      # 29 February into a non-leap year
            start_date = end_date.replace(year=end_date.year - years, day=28)
    elif start:
        start_date = _as_date(start, "start")
    else:
        # No dates at all: a year, which is the shortest span from which a
        # seasonal picture means anything. Better than a default of three months,
        # which reads as a green flash and explains nothing.
        start_date = end_date.replace(year=end_date.year - 1)
    if start_date > end_date:
        raise ValueError("the window starts after it ends")
    return Window(start_date, end_date)


def _as_date(value: str, which: str) -> datetime.date:
    try:
        return datetime.date.fromisoformat(str(value)[:10])
    except ValueError as exc:
        raise ValueError(f"the window {which} must be a date like 2019-01-01") from exc


def sensors_covering(window: Window) -> list[str]:
    """Which sensors have an archive that reaches back before the window starts.

    The honest question is not "which sensor is best" but "which sensors can
    answer this at all", and for a 1994 request the answer is Landsat alone.
    """
    out = []
    for sid, sensor in sensors.SENSORS.items():
        # archive_start is an ISO string in the sensor registry, so compare as
        # dates rather than lexically. ISO dates sort correctly as strings, but
        # only for the same format, and a locale-formatted one would silently
        # compare wrong rather than fail.
        try:
            begins = datetime.date.fromisoformat(str(sensor.archive_start)[:10])
        except ValueError:
            continue
        if window.start >= begins:
            out.append(sid)
    return out


def normal_window_for(product_key: str, window: Window) -> dict:
    """The period a comparison for this window is made against.

    Published for rainfall, derived for vegetation. That difference is real and
    the reason this exists as a function: MOD13Q1 begins in 2000-02, so a
    "1991-2020 normal" beside a vegetation series would be a claim about data
    that does not exist.
    """
    product = registry.get(product_key)
    if product is None:
        return {}
    if product_key == "vegetation_series":
        import vegetation_series as vs

        start_year = max(int(window.start.year) - vs.BASELINE_YEARS + 1, 2000)
        return {"start": f"{start_year}-01-01",
                "end": window.end.isoformat(),
                "years": window.end.year - start_year + 1,
                "basis": "the trailing years MODIS MOD13Q1 actually has, which "
                         "begins in 2000-02; a 1991-2020 label would be a "
                         "claim about data that does not exist"}
    import rainfall

    return {"start": rainfall.CLIMATOLOGY_START,
            "end": rainfall.CLIMATOLOGY_END,
            "years": 30,
            "basis": "the WMO 1991-2020 normal period"}


# The questions. Four, and the fifth that would be needed if a persona could not
# be served by one of them -- which is the test of whether this is a layer.
QUESTIONS = ("describe", "compare", "history", "watch")

# How many areas one comparison may carry. A planner's question is "which of my
# twenty wards is worst", and twenty synchronous raster reads is a different
# service from one. The cap is here rather than in the route so the question layer
# is the thing that decides, and so a client can ask what the limit is.
MAX_COMPARE_AREAS = 12

# What each question needs, and therefore what it costs. A question that can be
# answered from a cache is cheap regardless of which products it wants; one that
# needs a fresh raster read is not. This is the routing decision, made once,
# from the registry, rather than per route.
QUESTION_NEEDS: dict[str, tuple[str, ...]] = {
    # "what is this place" -- static properties and current condition
    "describe": ("dem", "landcover", "ndvi"),
    # "which of mine is worst" -- the same, over several areas
    "compare": ("dem", "landcover", "ndvi"),
    # "give me the record" -- series, which is never a live read
    "history": ("rainfall", "vegetation_series"),
    # "is it worsening, and when do I act" -- the recent record plus a threshold
    "watch": ("rainfall",),
}


@dataclass
class QuestionPlan:
    """What a question will need, and how it will be answered."""

    question: str
    window: Window
    products: tuple[str, ...]
    routing: str                      # "live" | "computed" | "mixed"
    sensors: tuple[str, ...] = ()
    notes: tuple[str, ...] = field(default_factory=tuple)

    def describe(self) -> dict:
        out = {
            "question": self.question,
            "window": self.window.describe(),
            "products": list(self.products),
            "routing": self.routing,
        }
        if self.sensors:
            out["sensors_covering_window"] = list(self.sensors)
        for key in self.products:
            normal = normal_window_for(key, self.window)
            if normal:
                out.setdefault("normals", {})[key] = normal
        if self.notes:
            out["notes"] = list(self.notes)
        return out


def plan_question(question: str, window: Window,
                  products: Optional[tuple[str, ...]] = None,
                  area_km2: Optional[float] = None) -> QuestionPlan:
    """Decide what to fetch and whether it can be done inside a request."""
    if question not in QUESTIONS:
        raise ValueError(f"unknown question {question!r}; expected one of {QUESTIONS}")
    wanted = tuple(products) if products else QUESTION_NEEDS[question]
    for key in wanted:
        if registry.get(key) is None:
            raise ValueError(f"{key!r} is not a registered product")

    # Routing from the registry, not from a table kept beside it. Every product
    # that can only be answered by a worker makes the question computed; every
    # product that can be answered live leaves it live.
    live = [k for k in wanted if registry.LIVE in registry.get(k).costs()]
    computed = [k for k in wanted if registry.COMPUTED in registry.get(k).costs()]
    if computed and not live:
        routing = "computed"
    elif live and not computed:
        routing = "live"
    else:
        routing = "mixed"

    notes: list[str] = []
    covering = tuple(sensors_covering(window))
    if wanted and any(k.startswith(("ndvi", "vegetation")) for k in wanted) and not covering:
        notes.append(
            f"No sensor archive reaches back to {window.start:%Y-%m}, so there is "
            f"no vegetation value for this window. Landsat begins "
            f"{sensors.LANDSAT.archive_start}, Sentinel-2 "
            f"{sensors.SENTINEL2.archive_start}.")
    if window.bin_months > 1:
        notes.append(
            f"{window.months} months at one bar each is not readable, so this "
            f"window is binned yearly.")
    if routing == "mixed":
        notes.append(
            f"{len(computed)} of {len(wanted)} products need the worker; the rest "
            f"answer inside the request. Expect the computed part to arrive later.")

    return QuestionPlan(question=question, window=window, products=wanted,
                        routing=routing, sensors=covering, notes=tuple(notes))


def plan_comparison(areas: list[dict], window: Window) -> dict:
    """A comparison plan, including the thing that makes comparison honest.

    Comparing areas is not comparing N independent numbers. For any product
    coarser than the outlines -- ERA5 above all -- two nearby areas can resolve to
    the *same* grid cell and therefore to the identical figure, and a ranking
    that puts one above the other is ranking noise. So the plan reports the cell
    count per area, which is what lets a caller notice when two entries are the
    same measurement rather than two measurements.

    It also states plainly which products can rank at all. A ranking is only
    meaningful across areas where the same measure, over the same ground, at the
    same resolution -- and ERA5 over small areas fails the second of those for
    reasons no amount of care here can fix.
    """
    if not areas:
        raise ValueError("a comparison needs at least one area")
    if len(areas) > MAX_COMPARE_AREAS:
        raise ValueError(
            f"{len(areas)} areas is more than one comparison carries "
            f"({MAX_COMPARE_AREAS}); narrow it or ask again in batches")

    products = QUESTION_NEEDS["compare"]
    rows = []
    for index, area in enumerate(areas):
        rows.append({
            "index": index,
            "id": area.get("id"),
            "name": area.get("name") or area.get("label") or f"area {index + 1}",
            "level": area.get("level"),
            "bbox_area_km2": area.get("area_km2"),
            "source": area.get("source"),
        })

    notes = [
        f"A comparison ranks {len(areas)} areas on the same products "
        f"({', '.join(products)}), over the window {window.label}.",
        "Elevation and land cover are single-date products, so a ranking across "
        "them compares places rather than times. Precipitation is a 0.25 degree "
        "reanalysis: two areas inside one 774 km2 cell produce the identical "
        "number, and the cell count below is how you can tell.",
    ]
    return {
        "question": "compare",
        "window": window.describe(),
        "products": list(products),
        "routing": "mixed",
        "areas": rows,
        "rankable_by": list(products),
        "notes": notes,
    }
