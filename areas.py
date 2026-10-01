"""Turning "that ward" or "where I am" into a study area.

Every area this service has ever analysed entered as raw GeoJSON. That is the
friction the planner persona cannot get past: they think in administrative units
-- ward, sub-county, county -- and do not have those boundaries as files. The
conservancy manager has a boundary file and the ranger has nothing but a
location, and the researcher has a polygon in a script. Four ways of naming the
same thing, and only one of them was accepted.

So the resolver takes all of them and returns one ``Area``. It is a capability in
the same sense a product is, and deliberately *not* in ``registry.py``: a
product yields a ``Measure``, a resolver yields an ``Area``, and keeping the two
apart is what stops the registry becoming a bag of unrelated things.

Four modes, and the source is injectable so a test can run on a recorded
response from the real service while production talks to the real one. That
idiom already exists here -- ``rainfall.read_cell_monthly`` takes a ``source``
for exactly this reason -- so the pattern is the project's, not new.

**Degradation is explicit.** If the boundary service is unreachable, this raises
``ResolverUnavailable`` and the route reports "the boundary service is not
answering", not a 500 and not a silently empty map. A planner who cannot
resolve a ward needs to know whether the tool is broken or their input is, and
those demand different responses.

The fixtures in ``tests/fixtures`` are recorded responses from the live service,
not invented shapes, so the parsing is tested against what it will actually see.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Optional

DEFAULT_BASE_URL = os.getenv("GAUL_API_URL", "http://gaul-api:8000")
"""Default assumes a compose network. The droplet's copy is on loopback at
127.0.0.1:8002 and nginx does not publish it, so the two deployments configure
this differently -- which is why it is one setting rather than a constant."""


class ResolverError(Exception):
    """A request the resolver understood and cannot satisfy."""

    def __init__(self, message: str, *, reason: str = "unresolved"):
        super().__init__(message)
        self.reason = reason


class ResolverUnavailable(ResolverError):
    """The boundary service did not answer. Not the caller's fault."""


@dataclass(frozen=True)
class Area:
    """One canonical study area, however the caller arrived at it.

    ``source`` is the important field. The same geometry reached by hand, by an
    administrative id and by a GPS point are three different claims about how
    much precision is behind the outline, and a result that does not say which
    one it came from cannot warn about it.
    """

    geometry: dict
    bbox: tuple[float, float, float, float]
    area_km2: float
    name: Optional[str] = None
    level: Optional[int] = None
    id: Optional[str] = None
    parent: Optional[str] = None
    country: Optional[str] = None
    source: str = "supplied"
    resolver: Optional[str] = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def describe(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "level": self.level,
            "parent": self.parent,
            "country": self.country,
            "bbox": list(self.bbox),
            "area_km2": round(self.area_km2, 2),
            "source": self.source,
            "resolver": self.resolver,
            "notes": list(self.notes),
            "geometry": self.geometry,
        }


class GaulResolver:
    """A client for the boundary service. Injectable as any object with ``get``."""

    name = "gaul"

    def __init__(self, base_url: str = DEFAULT_BASE_URL, timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def get(self, path: str, params: dict) -> Any:
        query = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
        url = f"{self.base_url}/{path}?{query}" if query else f"{self.base_url}/{path}"
        try:
            with urllib.request.urlopen(url, timeout=self.timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = json.loads(exc.read()).get("detail", "")
            except Exception:
                pass
            raise ResolverError(
                detail or f"the boundary service refused this request ({exc.code})",
                reason="rejected",
            ) from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            # A service that is down is not a service that said no. Keeping them
            # apart is the difference between "try again" and "fix your input".
            raise ResolverUnavailable(
                f"the boundary service is not answering ({exc})", reason="unavailable"
            ) from exc

    def search(self, country: str, level: int, query: str) -> list[dict]:
        payload = self.get("boundary-index", {"country": country, "level": level, "q": query})
        return list(payload.get("items") or [])

    def by_id(self, boundary_id: str, level: int, simplify: Optional[float] = None) -> dict:
        payload = self.get("boundaries", {
            "ids": boundary_id, "level": level, "format": "geojson", "simplify": simplify,
        })
        features = payload.get("features") or []
        if not features:
            raise ResolverError(
                f"no boundary with id {boundary_id!r} at level {level}", reason="not_found")
        return features[0]

    def by_name(self, country: str, level: int, admin1: Optional[str] = None,
                admin2: Optional[str] = None, simplify: Optional[float] = None) -> dict:
        payload = self.get("boundaries", {
            "country": country, "level": level, "admin1": admin1, "admin2": admin2,
            "format": "geojson", "simplify": simplify,
        })
        features = payload.get("features") or []
        if not features:
            what = admin2 or admin1 or country
            raise ResolverError(
                f"no administrative area named {what!r} at level {level}", reason="not_found")
        if len(features) > 1:
            raise ResolverError(
                f"{len(features)} areas match {admin2 or admin1 or country!r} at level "
                f"{level}; name a more specific one, or search for the id",
                reason="ambiguous")
        return features[0]

    def containing(self, lat: float, lon: float) -> dict:
        return self.get("containing", {"lat": lat, "lon": lon})


def resolve_by_name(resolver, query: str, *,
                    simplify: Optional[float] = None) -> Area:
    """Resolve whatever a person types into a boundary.

    The boundary service narrows from a country: ``country=137&admin1=Narok``.
    Handing it a bare area name as the country finds nothing, so a caller that
    simply says "Narok" gets a silent miss. The translation belongs here rather
    than in a client, so that "Narok", "Kenya, Narok" and "137, Narok" all mean
    the same thing to every caller.

    Two forms are accepted, and the second is the one worth documenting: a bare
    name is ambiguous across ~200 countries, so it is matched against country
    names first and otherwise refused with the form that would work.
    """
    query = (query or "").strip()
    if not query:
        raise ResolverError("name an area", reason="malformed_request")

    parts = [p.strip() for p in query.split(",") if p.strip()]
    country, name = (parts[0], parts[1]) if len(parts) > 1 else (None, parts[0])

    if country is None:
        # A bare name: it may be a country in its own right.
        try:
            return _area_from_feature(
                resolver.by_name(name, 0, simplify=simplify),
                source="name", resolver=getattr(resolver, "name", "gaul"))
        except ResolverError:
            pass
        raise ResolverError(
            f"no country named {name!r}. Administrative areas are named "
            f"'Country, Area' -- try 'Kenya, {name}'.",
            reason="needs_country")

    feature = resolver.by_name(country, 1, admin1=name, simplify=simplify)
    return _area_from_feature(feature, source="name",
                              resolver=getattr(resolver, "name", "gaul"),
                              notes=(f"Resolved as the {name} administrative area "
                                     f"of {country}; its outline is the whole unit.",
                                     ))


def _area_from_feature(feature: dict, *, source: str, resolver: str,
                       notes: tuple[str, ...] = ()) -> Area:
    import main

    properties = feature.get("properties") or {}
    geometry = feature.get("geometry")
    if not geometry:
        raise ResolverError("the boundary came back with no geometry", reason="malformed")
    # validate_for_lookup, not validate_aoi: it keeps the payload and geometry
    # checks and drops the area cap, which is what an administrative boundary
    # needs. Kenya's admin0 is about 580,000 km2 and would be refused outright by
    # the synchronous admission path this resolver exists to avoid.
    aoi = main.validate_for_lookup({"type": "Feature", "properties": {},
                                    "geometry": geometry})

    # Finest level present in the properties is this feature's level. Written
    # out rather than as `a and b or c`, which reads as a trick and silently
    # collapses to 0 whenever a code is legitimately zero.
    if "gaul2_code" in properties or "gaul2_name" in properties:
        level = 2
    elif "gaul1_code" in properties or "gaul1_name" in properties:
        level = 1
    else:
        level = 0
    name = (properties.get("gaul2_name") or properties.get("gaul1_name")
            or properties.get("gaul0_name"))
    return Area(
        geometry={"type": "Feature", "properties": {}, "geometry": aoi["feature"]["geometry"]},
        bbox=tuple(aoi["bbox"]),
        area_km2=aoi["bbox_area_km2"],
        name=name,
        level=level,
        id=feature.get("id"),
        parent=properties.get("gaul0_name") if level else None,
        country=properties.get("gaul0_name"),
        source=source,
        resolver=resolver,
        notes=notes,
    )


def resolve_area(resolver, *, id: Optional[str] = None, level: Optional[int] = None,
                 country: Optional[str] = None, admin1: Optional[str] = None,
                 admin2: Optional[str] = None, lat: Optional[float] = None,
                 lon: Optional[float] = None, bbox: Optional[str] = None,
                 simplify: Optional[float] = None) -> Area:
    """One area from whichever way the caller can name it.

    The modes are mutually exclusive on purpose. A request carrying an id *and* a
    point is ambiguous, and silently preferring one of them is how a user ends up
    analysing somewhere other than where they think.
    """
    named = [bool(v) for v in (id, admin2 or admin1 or (country if level is not None else None),
                               lat is not None and lon is not None, bbox)]
    if sum(named) != 1:
        raise ResolverError(
            "name the area one way: an administrative id, a country and level, "
            "a latitude and longitude, or a bounding box",
            reason="ambiguous_request")

    if id:
        if level is None:
            # The id carries its own level, so read it rather than making the
            # caller repeat it. Asking for the wrong level is a 422, and the
            # message says which level the id actually is.
            try:
                level = int(str(id).split(":")[1])
            except (IndexError, ValueError) as exc:
                raise ResolverError(
                    f"{id!r} is not a boundary id; they look like 137:1:1385",
                    reason="malformed_request") from exc
        feature = resolver.by_id(id, level, simplify=simplify)
        return _area_from_feature(feature, source="admin_id", resolver=getattr(resolver, "name", "gaul"))

    if lat is not None and lon is not None:
        payload = resolver.containing(lat, lon)
        levels = payload.get("boundaries") or {}
        # Finest first. A point is in exactly one admin2, and returning the
        # district is the useful answer; a planner asking about a county should
        # ask for it by name rather than get a smaller unit by accident.
        for key, source_level in (("l2", 2), ("l1", 1), ("l0", 0)):
            entry = levels.get(key) or {}
            feature = entry.get("feature")
            if feature:
                notes = ("Resolved from a point, so the outline is the whole "
                         "administrative unit and not a smaller thing you drew.")
                return _area_from_feature(feature, source="point", resolver=getattr(resolver, "name", "gaul"),
                                          notes=(notes,))
        raise ResolverError(f"no administrative boundary contains {lat},{lon}",
                            reason="not_found")

    if bbox:
        try:
            west, south, east, north = (float(v) for v in bbox.split(","))
        except ValueError as exc:
            raise ResolverError("bbox must be minLon,minLat,maxLon,maxLat",
                                reason="malformed_request") from exc
        import main

        geometry = {"type": "Polygon", "coordinates": [[
            [west, south], [east, south], [east, north], [west, north], [west, south]]]}
        aoi = main.validate_for_lookup({"type": "Feature", "properties": {},
                                        "geometry": geometry})
        return Area(
            geometry={"type": "Feature", "properties": {}, "geometry": geometry},
            bbox=(west, south, east, north), area_km2=aoi["bbox_area_km2"],
            source="bbox", resolver=getattr(resolver, "name", "gaul"),
            notes=("A bounding box is a rectangle, not the area you had in mind. "
                   "Its figures are the rectangle's.",),
        )

    feature = resolver.by_name(country, level, admin1=admin1, admin2=admin2,
                               simplify=simplify)
    return _area_from_feature(feature, source="name", resolver=getattr(resolver, "name", "gaul"))
