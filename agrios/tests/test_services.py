"""The service layer's own contract -- typed results and typed errors -- as the
console will use it. What the agent's tools make of it is pinned separately
by test_tools_golden.py."""

from typing import Any

import httpx
import pytest

from agrios.services import calendar, farm, weather


class _Resp:
    def __init__(self, payload: object, status: int = 200) -> None:
        self.payload, self.status = payload, status

    def json(self) -> object:
        return self.payload

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise httpx.HTTPStatusError("boom", request=None, response=None)  # type: ignore[arg-type]


@pytest.fixture
def calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, dict[str, Any]]]:
    """Fakes every outbound request and records it; nothing reaches the network."""
    seen: list[tuple[str, dict[str, Any]]] = []
    daily = {"time": ["d1", "d2"], "temperature_2m_max": [20, 18.5],
             "temperature_2m_min": [3, 1.5], "precipitation_sum": [0.5, 2]}  # fmt: skip

    def get(url: str, params: dict[str, Any] | None = None, timeout: float | None = None) -> _Resp:
        seen.append((url, params or {}))
        if url == weather.OPEN_METEO_GEOCODE:
            return _Resp({})  # no place matches
        return _Resp({"daily": daily})

    monkeypatch.setattr(httpx, "get", get)
    return seen


def test_logged_tasks_get_ids_and_show_as_pending() -> None:
    a = farm.log_task("field-1", "service test a")
    b = farm.log_task("field-1", "service test b")
    assert b.id > a.id
    pending = {t.id: t for t in farm.pending_tasks()}
    assert pending[a.id].description == "service test a"
    assert pending[b.id].status == "pending"


def test_unknown_field_is_none_not_an_error() -> None:
    assert farm.get_field("field-99") is None
    field = farm.get_field("field-1")
    assert field is not None
    assert field.crop == "potato"


def test_forecast_flags_by_the_thresholds_it_is_given(calls: list[tuple[str, dict[str, Any]]]) -> None:
    default = weather.forecast("field-1", days=99)
    assert calls[-1][1]["forecast_days"] == weather.MAX_FORECAST_DAYS
    assert default.frost_risk_days == ["d2"]  # 1.5 <= 2.0; 3 isn't
    assert default.rain_days == ["d2"]  # 2 >= 1.0; 0.5 isn't
    # Control: the thresholds are parameters, not constants baked in. The
    # console's settings pass them through here.
    stricter = weather.forecast("field-1", frost_risk_c=0.0, rain_day_mm=0.1)
    assert stricter.frost_risk_days == []
    assert stricter.rain_days == ["d1", "d2"]
    assert default.forecast[0].temp_max_c == 20  # an integer stays an integer
    assert isinstance(default.forecast[0].temp_max_c, int)


def test_forecast_errors_are_typed(calls: list[tuple[str, dict[str, Any]]]) -> None:
    with pytest.raises(weather.LocationNotFoundError):
        weather.forecast("Atlantis")


def test_calendar_refuses_uncovered_regions_before_calling_fao(
    calls: list[tuple[str, dict[str, Any]]],
) -> None:
    with pytest.raises(calendar.RegionNotCoveredError):
        calendar.crop_calendar("potato", "Midwest USA")
    assert calls == []
    assert calendar.zones_for_region("Chile") == {"Arid Zone", "Mediterranean", "Moist Temperate"}
