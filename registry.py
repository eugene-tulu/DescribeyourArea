"""One declaration per capability, and the answer to "what kind of number is this".

This module exists because the same question was being answered in six places
with five different answers. The resolution of ERA5 was ``27.8``, hardcoded in
``indicators.py`` while ``rainfall.py`` held ``ERA5_GRID_DEGREES`` and computed
it properly. The land-cover memory budget was defined twice with different
defaults. ``PIXEL_BUDGET`` was defined and never read. MODIS's native
resolution was 231.7 in the series path and 250 in the sensor registry, because
the sinusoidal grid really is 231.7 and the label says 250.

The cost of that is not tidiness. It is that a product's properties -- its
resolution, its latency, the area over which its measure means anything -- are
exactly the things a user needs in order to decide whether to believe a number,
and they were scattered through arithmetic.

So: declare once, derive everywhere. ``/version`` publishes this, the client can
render it, and adding a product is an entry here rather than a constant, a card,
a route and a branch.

Two properties are load-bearing and were previously implicit:

``meaningful_area_km2``
    The range over which the measure says something about the thing asked about.
    Distinct from the *affordable* range, which is a property of this server.
    ERA5 is the sharp case: at 0.25 degrees the measure is meaningful from
    roughly one cell upward, and below about 28 km2 the "area mean" is a single
    cell that says the same thing about a 10 km2 paddock and a 600 km2 landscape.

``latency_days``
    How old the newest available observation is. Four products currently sit in
    one row and are six months apart, and nothing said so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Evidence classes. The vocabulary is deliberately four and does not grow:
# a number without a stated kind is an assertion pretending to be a measurement.
OBSERVED = "observed"    # a product of a sensor, as published
DERIVED = "derived"      # an index or statistic computed from a product
MODELLED = "modelled"    # a model output, even one that assimilates observations

# Cost classes decide routing, not presentation. "live" answers inside a request;
# "computed" is a queued job. The question layer picks from these, so a user
# asking for a series and a user asking for a current state do not have to know
# which products happen to be slow.
#
# A product may serve *both*, and most of them do. NDVI answers live for a small
# area and is queued by the worker for a large one, because the area, not the
# product, decides whether the pixels fit in a request. So this is a set, not a
# flag: a single class would force the question layer to hard-code the exception.
LIVE = "live"
COMPUTED = "computed"


@dataclass(frozen=True)
class Product:
    """Everything that is true of a capability regardless of where it runs."""

    key: str
    """The stable id the API and client use. Never a display string."""

    label: str
    """What the reader calls it. Appears on a card."""

    measure: str
    """What is actually quantified, independent of the product that provides it.

    This is the field that makes products substitutable. Rainfall from ERA5 and
    rainfall from a 0.05 degree satellite-gauge blend are the same ``measure``
    with different values for resolution, latency and evidence, which is why one
    could replace the other without a new card.
    """

    units: str
    native_resolution_m: Optional[float] = None
    native_grid_degrees: Optional[float] = None
    meaningful_min_km2: Optional[float] = None
    meaningful_max_km2: Optional[float] = None
    latency_days: Optional[str] = None
    """A range, honestly, where it varies by update cadence: "1-2" for a monthly
    product or "0-1" for a daily one. A single integer here would be a claim
    about a cadence that does not exist."""

    evidence: str = OBSERVED
    cost: tuple[str, ...] = (LIVE,)
    seconds_per_month: Optional[float] = None
    """Measured wall-clock per month read for a queued series, on this host's
    predecessor. None where the product is answered live and has no series to
    wait for. This is a property of the product and the host, not of the area --
    cost here is request latency, not pixels -- so it belongs beside the rest of
    the declaration rather than in a cost table kept separately.

    ERA5 reads a whole time series from one asset and is fast per month.
    CHIRPS is a separate object per month and is not: measured at 1.04 s a month
    with eight concurrent reads, against ERA5's fraction of that. A CHIRPS job
    shown with ERA5's estimate promised 24 seconds and took 17 minutes.
    """
    source: str = ""
    doi: Optional[str] = None
    caveats: tuple[str, ...] = field(default_factory=tuple)
    fails_with: tuple[tuple[str, str], ...] = field(default_factory=tuple)
    """Conditions and the reason given, so a refusal is never a bare failure."""

    def costs(self) -> tuple[str, ...]:
        """The ways this product can be served, in a stable order."""
        return tuple(c for c in (LIVE, COMPUTED) if c in self.cost)

    def resolution_km(self) -> Optional[float]:
        if self.native_grid_degrees is not None:
            return round(self.native_grid_degrees * 111.32, 1)
        if self.native_resolution_m is not None:
            return round(self.native_resolution_m / 1000.0, 3)
        return None

    def describe(self) -> dict:
        """The published form. A partner reads this; it should be self-contained."""
        return {
            "key": self.key,
            "label": self.label,
            "measure": self.measure,
            "units": self.units,
            "native_resolution_m": self.native_resolution_m,
            "native_grid_degrees": self.native_grid_degrees,
            "resolution_km": self.resolution_km(),
            "meaningful_area_km2": (
                None if self.meaningful_min_km2 is None and self.meaningful_max_km2 is None
                else [self.meaningful_min_km2, self.meaningful_max_km2]
            ),
            "latency_days": self.latency_days,
            "evidence": self.evidence,
            "cost": list(self.costs()),
            "seconds_per_month": self.seconds_per_month,
            "source": self.source,
            "doi": self.doi,
            "caveats": list(self.caveats),
        }


# --- the products -----------------------------------------------------------
#
# Each entry's caveat is the honest version, not the reassuring one. The
# resolution ladder and the honesty copy already existed in main.py and
# sensors.py; this is those strings in one place, and deleting them from there
# changes what a user is told.

ELEVATION = Product(
    key="dem",
    label="Elevation",
    measure="surface_elevation",
    units="m",
    native_resolution_m=30.0,
    meaningful_min_km2=0.01,
    meaningful_max_km2=None,
    latency_days=None,
    evidence=OBSERVED,
    cost=(LIVE, COMPUTED),
    source="NASADEM (30 m, void-filled SRTM/3DEP)",
    doi="10.5066/F7PV9PCP",
    caveats=(
        "An observed product, not a field measurement of this polygon.",
        "A continuous elevation range is reported as three words -- flat, "
        "moderately undulating, or highly variable. The underlying range is "
        "reported to a tenth of a metre, which is a tenth of a pixel, so the "
        "banding is the honest summary and the number beneath it is not.",
    ),
    fails_with=(
        ("elevation_unavailable", "No elevation data covers this area."),
        ("no_valid_elevation_pixels", "Elevation exists for the region but not "
                                       "inside the boundary."),
    ),
)

LANDCOVER = Product(
    key="landcover",
    label="Land cover",
    measure="land_cover_share",
    units="%",
    native_resolution_m=10.0,
    meaningful_min_km2=0.01,
    meaningful_max_km2=1000.0,
    latency_days=None,
    evidence=OBSERVED,
    cost=(LIVE, COMPUTED),
    source="ESA WorldCover 10 m v200",
    doi="10.5061/DZYR-JMP",
    caveats=(
        "A single-date classification, so it reflects the scene, not a year.",
        "At 10 m this is the only module whose memory genuinely grows with area, "
        "so it has the largest budget of the four.",
        "Known gap: this reader takes its tiles at native resolution and does not "
        "decimate, so a large area queued for land cover is still read at 10 m. "
        "The resolution ladder applies to vegetation and not to this, and the "
        "interface says so rather than offering a coarser read that does not "
        "happen. Measured: 1,205 MB at 5,500 km2, which is why it is the product "
        "that sets the request budget.",
    ),
    fails_with=(
        ("landcover_unavailable", "No land-cover data covers this area."),
        ("landcover_area_exceeded",
         "Too large to read at 10 m on this server. Draw a smaller boundary, or "
         "process the area offline, which reads the same product more coarsely."),
    ),
)

VEGETATION = Product(
    key="ndvi",
    label="Vegetation",
    measure="vegetation_index",
    units="NDVI",
    native_resolution_m=None,        # varies by the sensor the ladder picks
    meaningful_min_km2=0.01,
    meaningful_max_km2=2000.0,
    latency_days="2-16",
    evidence=DERIVED,
    cost=(LIVE, COMPUTED),
    source="Landsat C2 L2, Sentinel-2 L2A or MODIS MOD13Q1, chosen per area",
    doi=None,
    caveats=(
        "An index derived from a surface-reflectance product, not a direct "
        "measurement of vegetation.",
        "A persistently cloudy area reads low. A figure below roughly 0.2 should "
        "be read as uncertain rather than as bare ground, and said so in "
        "anything assessment-facing.",
        "The resolution reported is the one actually read, not the one asked "
        "for: a 250 m product cannot be resampled to 20 m without inventing "
        "detail.",
    ),
    fails_with=(
        ("no_valid_ndvi_pixels",
         "Every candidate scene was flagged cloud, shadow or snow over the "
         "area, so no vegetation value is reported rather than reporting cloud."),
        ("sensor_archive_too_short",
         "The requested window starts before this sensor's archive does."),
        ("busy", "Vegetation analysis is busy. Terrain and land cover are "
                 "unaffected; retry shortly for a vegetation value."),
    ),
)

RAINFALL = Product(
    key="rainfall",
    label="Rainfall",
    measure="precipitation_total",
    units="mm",
    native_resolution_m=None,
    native_grid_degrees=0.25,
    # One cell, computed rather than remembered: 0.25 deg is 27.8 km on a side,
    # 774 km2 at the latitude this product is served for. It is seven times the
    # 100 km2 synchronous cap, which is the whole problem in one number -- every
    # area a user can analyse live is described by exactly one cell.
    meaningful_min_km2=774.0,
    meaningful_max_km2=None,
    latency_days="45-90",
    evidence=MODELLED,
    cost=COMPUTED,
    # 24 s for 195 months, one asset read for the whole series.
    seconds_per_month=0.12,
    source="ERA5 monthly precipitation, 0.25 degrees",
    doi="10.24381/cds.adbb2d47",
    caveats=(
        "A reanalysis: modelled output that assimilates observations, not a "
        "gauge reading. Treat a value near a threshold as uncertain.",
        "This is a landscape-to-regional product, not a field survey. At 0.25 "
        "degrees a study area smaller than one cell is described by that cell, "
        "which then says the same thing about a 10 km2 paddock and a 600 km2 "
        "landscape. The cell count travels with every result for that reason.",
        "Assignment is nearest-cell, not an area-weighted integral.",
    ),
    fails_with=(
        ("not_computed", "No series has been processed for this boundary yet. "
                         "It is computed offline because a single grid cell takes "
                         "about 20 seconds to read, longer than a whole request."),
    ),
)

VEGETATION_SERIES = Product(
    key="vegetation_series",
    label="Vegetation series",
    measure="vegetation_index",
    units="NDVI",
    native_resolution_m=231.7,       # the MODIS sinusoidal grid, not the 250 m label
    meaningful_min_km2=1.0,
    meaningful_max_km2=None,
    latency_days="30-60",
    evidence=DERIVED,
    cost=(COMPUTED,),
    # Read from the series module rather than restated, so the published cost and
    # the one an estimate uses cannot drift apart.
    seconds_per_month=__import__("vegetation_series").SECONDS_PER_READ,
    source="MODIS MOD13Q1 (250 m label, 231.7 m grid), 16-day",
    doi="10.5067/MODIS/MOD13Q1.061",
    caveats=(
        "An index derived from a surface-reflectance product, not a direct "
        "measurement of vegetation.",
        "Co-variation with rainfall is not attribution. Vegetation responds to "
        "rainfall with a lag that varies by season, and in semi-arid rangeland "
        "water is not always the limiting factor.",
        "MOD13Q1 begins in 2000-02, so the normal is formed from the years that "
        "exist rather than labelled 1991-2020.",
    ),
    fails_with=(
        ("no_modis_imagery", "No MODIS imagery covers this area for the period."),
    ),
)

# The second producer of `precipitation_total`. Two products in one measure is
# what makes the choice meaningful rather than a preference: a reader can see
# both, and the difference between them is the useful part.
RAINFALL_CHIRPS = Product(
    key="chirps",
    label="Rainfall (CHIRPS)",
    measure="precipitation_total",
    units="mm",
    native_grid_degrees=0.05,
    # 0.05 deg is about 31 km2 at this latitude, so unlike ERA5 the measure is
    # about a small area rather than about a cell that happens to contain one.
    meaningful_min_km2=1.0,
    meaningful_max_km2=None,
    latency_days="30-60",
    evidence=OBSERVED,
    cost=(COMPUTED,),
    # 3.4 s a month with eight concurrent reads, each month a separate object.
    # Refitted from a cold end-to-end job on the droplet: 681 s for 202 months. The
    # first figure, 1.04, came from a warm cell cache left by my own earlier read
    # and was 3x optimistic -- an estimate that understates the wait is the one
    # kind a reader cannot forgive.
    seconds_per_month=3.4,
    source="CHIRPS v2.0 (satellite and gauge, 0.05 degrees), via Digital Earth Africa",
    doi="10.1038/sdata.2015.66",
    caveats=(
        "A satellite-and-gauge blend rather than a reanalysis, so it earns an "
        "observed class where ERA5 is modelled.",
        "0.05 degrees, against ERA5's 0.25. A small study area is described by "
        "one CHIRPS cell rather than by a cell covering about 774 km2.",
        "The published archive has holes -- 2023-12, 2024-07 and 2024-08 are "
        "absent -- so months that could not be read are listed rather than "
        "omitted. A missing month is not a dry month.",
        "Measured against ERA5 over three Kenyan cells across 120 months, "
        "CHIRPS reads 0%, 16% and 24% higher, the difference growing as the land "
        "gets drier, while 92% of months agree on the sign of the anomaly. The "
        "two are complementary, not interchangeable.",
    ),
    fails_with=(
        ("not_computed", "No CHIRPS series has been processed for this boundary yet."),
    ),
)

PRODUCTS: dict[str, Product] = {
    p.key: p for p in (
        ELEVATION, LANDCOVER, VEGETATION, RAINFALL, RAINFALL_CHIRPS,
        VEGETATION_SERIES,
    )
}


def get(key: str) -> Optional[Product]:
    return PRODUCTS.get(key)


def describe_all() -> list[dict]:
    """Every product, published. Sorted so the output is diffable."""
    return [PRODUCTS[k].describe() for k in sorted(PRODUCTS)]


def measures() -> dict[str, list[str]]:
    """measure -> products providing it.

    The substitutability map. Two products in one list are interchangeable
    consumers of the same question, which is what makes "swap ERA5 for
    something at 0.05 degrees" a registry change rather than a rewrite.
    """
    out: dict[str, list[str]] = {}
    for product in PRODUCTS.values():
        out.setdefault(product.measure, []).append(product.key)
    return {k: sorted(v) for k, v in sorted(out.items())}
