"""Weather forecasts from Open-Meteo, with frost-risk and rain days flagged in
code rather than left to whoever reads the numbers."""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel
from sqlalchemy import func, select

from agrios.db import engine, fields_table

# Air-temperature minimum at or below which a day counts as a frost risk;
# ground frost can form a couple of degrees above 0 °C air temperature.
FROST_RISK_C = 2.0
# Daily total at or above which a day counts as a rain day (the usual
# meteorological "wet day" threshold), so drizzle-level amounts don't.
RAIN_DAY_MM = 1.0
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
MAX_FORECAST_DAYS = 16  # Open-Meteo forecasts at most 16 days

# int | float, not float: a value Open-Meteo sends as 21 must stay 21, not
# become 21.0 -- the agent's tool output is these values serialised as-is.
Number = int | float


class LocationNotFoundError(LookupError):
    pass


class DayForecast(BaseModel):
    date: str
    temp_max_c: Number | None
    temp_min_c: Number | None
    rain_mm: Number | None
    frost_risk: bool


class Forecast(BaseModel):
    location: str
    resolved_to: str
    rain_days: list[str]
    total_rain_mm: Number
    frost_risk_days: list[str]
    forecast: list[DayForecast]
    source: str = "open-meteo.com"


class Place(BaseModel):
    name: str
    latitude: float
    longitude: float


def resolve_location(location: str) -> Place | None:
    """A field_id resolves to that field's stored coordinates; anything else
    is looked up as a place name."""
    with engine.connect() as conn:
        row = conn.execute(
            select(fields_table.c.field_id, fields_table.c.latitude, fields_table.c.longitude).where(
                func.lower(fields_table.c.field_id) == location.strip().lower()
            )
        ).fetchone()
    if row is not None and row.latitude is not None:
        return Place(name=row.field_id, latitude=row.latitude, longitude=row.longitude)
    resp = httpx.get(OPEN_METEO_GEOCODE, params={"name": location, "count": 1}, timeout=10)
    resp.raise_for_status()
    results: list[dict[str, Any]] = resp.json().get("results") or []
    if not results:
        return None
    hit = results[0]
    name = ", ".join(p for p in (hit.get("name"), hit.get("admin1"), hit.get("country")) if p)
    return Place(name=name, latitude=hit["latitude"], longitude=hit["longitude"])


def forecast(
    location: str, days: int = 7, *, frost_risk_c: float = FROST_RISK_C, rain_day_mm: float = RAIN_DAY_MM
) -> Forecast:
    """Raises LocationNotFoundError, or httpx.HTTPError / KeyError / ValueError when
    Open-Meteo is unreachable or answers with something unexpected."""
    days = max(1, min(int(days), MAX_FORECAST_DAYS))
    place = resolve_location(location)
    if place is None:
        raise LocationNotFoundError(location)
    resp = httpx.get(
        OPEN_METEO_FORECAST,
        timeout=10,
        params={
            "latitude": place.latitude,
            "longitude": place.longitude,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
            "forecast_days": days,
            "timezone": "auto",
        },
    )
    resp.raise_for_status()
    daily = resp.json()["daily"]
    # The model misjudged thresholds on its own (it called 10 °C a frost
    # risk), so the forecast flags frost-risk days itself.
    days_out = [
        DayForecast(
            date=d,
            temp_max_c=hi,
            temp_min_c=lo,
            rain_mm=rain,
            frost_risk=lo is not None and lo <= frost_risk_c,
        )
        for d, hi, lo, rain in zip(
            daily["time"],
            daily["temperature_2m_max"],
            daily["temperature_2m_min"],
            daily["precipitation_sum"],
            strict=False,
        )
    ]
    # Spelled out because the model misread the daily list once ("no rain"
    # with rain in the forecast); reading a ready answer is harder to get wrong.
    return Forecast(
        location=location,
        resolved_to=place.name,
        rain_days=[d.date for d in days_out if (d.rain_mm or 0) >= rain_day_mm],
        total_rain_mm=round(sum(d.rain_mm or 0 for d in days_out), 1),
        frost_risk_days=[d.date for d in days_out if d.frost_risk],
        forecast=days_out,
    )
