"""Planting and harvest windows from FAO's crop calendar for Chile, by
agro-ecological zone."""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel

FAO_CROP_CALENDAR = "https://api-cropcalendar.apps.fao.org/api/v1/countries/CL/cropCalendar"
# FAO splits Chile into agro-ecological zones; these map how people name the
# regions onto them. Central Chile's climate is Mediterranean.
REGION_ZONES = [
    (("north", "norte", "atacama", "arid"), "Arid Zone"),
    (("central", "centro", "santiago", "maule", "higgins", "valparaíso", "valparaiso", "mediterranean"),
     "Mediterranean"),
    (("south", "sur", "araucan", "los lagos", "los ríos", "los rios", "biobío", "biobio", "temperate"),
     "Moist Temperate"),
]  # fmt: skip
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


class RegionNotCoveredError(LookupError):
    """The region maps to no FAO zone. Raised before any network call."""


class CropNotInCalendarError(LookupError):
    def __init__(self, crop: str, region: str, crops_available: list[str]) -> None:
        super().__init__(crop, region)
        self.crop, self.region, self.crops_available = crop, region, crops_available


class Window(BaseModel):
    zone: str
    sowing: str
    harvest: str


class CropCalendar(BaseModel):
    crop: str
    region: str
    windows: list[Window]
    source: str


def zones_for_region(region: str) -> set[str]:
    lowered = region.lower()
    zones = {zone for keywords, zone in REGION_ZONES if any(k in lowered for k in keywords)}
    if not zones and "chile" in lowered:
        zones = {zone for _, zone in REGION_ZONES}  # "Chile" alone: every zone
    return zones


def normalize_crop(name: str) -> str:
    name = name.strip().lower()
    for suffix in ("oes", "es", "s"):  # potatoes -> potat(o), beans -> bean
        if name.endswith(suffix) and len(name) > len(suffix) + 2:
            return name[: -len(suffix)]
    return name


def _window(start: dict[str, Any], end: dict[str, Any]) -> str:
    # An en dash, not a hyphen: it's what the agent's tool output has always
    # contained, and the model-facing output must not change.
    return (
        f"{int(start['day'])} {_MONTHS[start['month'] - 1]} – {int(end['day'])} {_MONTHS[end['month'] - 1]}"  # noqa: RUF001
    )


def crop_calendar(crop: str, region: str) -> CropCalendar:
    """Raises RegionNotCoveredError, CropNotInCalendarError, or httpx.HTTPError /
    ValueError when FAO's service is unreachable or malformed."""
    zones = zones_for_region(region)
    if not zones:
        raise RegionNotCoveredError(region)
    resp = httpx.get(FAO_CROP_CALENDAR, params={"language": "en"}, timeout=15)
    resp.raise_for_status()
    entries: list[dict[str, Any]] = resp.json()

    wanted = normalize_crop(crop)
    matches = [
        e
        for e in entries
        if normalize_crop(e["crop"]["name"]).startswith(wanted) and e["aez"]["name"] in zones
    ]
    if not matches:
        raise CropNotInCalendarError(crop, region, sorted({e["crop"]["name"] for e in entries}))
    windows = [
        Window(
            zone=e["aez"]["name"],
            sowing=_window(s["early_sowing"], s["later_sowing"]),
            harvest=_window(s["early_harvest"], s["late_harvest"]),
        )
        for e in matches
        for s in e["sessions"]
    ]
    return CropCalendar(
        crop=matches[0]["crop"]["name"],
        region=region,
        windows=windows,
        source=f"FAO Crop Calendar, Chile (updated {matches[0]['lastUpdated']})",
    )
