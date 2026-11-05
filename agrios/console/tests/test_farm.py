"""The Farm screen: tasks logged and completed through the page, who may do
that, and the weather view."""

import re
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text

from agrios.console import auth
from agrios.console.tests.conftest import PASSWORD, T0, page_csrf, sign_in
from agrios.db import engine, init_db
from agrios.services import farm, settings, weather


def log_via_page(
    client: TestClient, field_id: str = "field-1", description: str = "Irrigate north rows"
) -> httpx.Response:
    return client.post(
        "/farm/tasks",
        data={"field_id": field_id, "description": description, "csrf_token": page_csrf(client, "/farm")},
    )


def test_a_task_logged_on_the_page_is_pending_with_its_time(signed_in):
    response = log_via_page(signed_in, description="  Irrigate north rows  ")
    assert response.status_code == 303  # post/redirect/get: a reload won't log it twice
    [task] = farm.pending_tasks()
    assert (task.field_id, task.description, task.created_at) == ("field-1", "Irrigate north rows", T0)
    assert "Irrigate north rows" in signed_in.get("/farm").text


@pytest.mark.parametrize(
    ("field_id", "description", "message"),
    [
        ("field-99", "Irrigate", "Choose one of the fields"),
        ("field-1", "   ", "Describe the task"),
        ("field-1", "x" * 501, "under 500 characters"),
    ],
)
def test_a_bad_task_is_refused_and_what_was_typed_kept(signed_in, field_id, description, message):
    response = log_via_page(signed_in, field_id, description)
    assert response.status_code == 422
    assert message in response.text
    assert farm.pending_tasks() == []
    if description.strip():
        assert f'value="{description}"' in response.text


def test_completing_a_task_moves_it_to_recently_done(signed_in, clock):
    task = farm.log_task("field-2", "Plough", now=T0)
    clock.advance(600)  # inside the 30-minute idle timeout
    csrf = page_csrf(signed_in, "/farm")
    assert signed_in.post(f"/farm/tasks/{task.id}/done", data={"csrf_token": csrf}).status_code == 303
    assert farm.pending_tasks() == []
    [done] = farm.recently_completed()
    assert (done.id, done.status, done.completed_at) == (task.id, "done", T0 + 600)
    # A second click (or a second person) changes nothing.
    clock.advance(60)
    assert signed_in.post(f"/farm/tasks/{task.id}/done", data={"csrf_token": csrf}).status_code == 303
    assert farm.recently_completed()[0].completed_at == T0 + 600
    assert not farm.complete_task(task.id)
    page = signed_in.get("/farm").text
    assert "Recently done" in page
    assert "Plough" in page


def test_a_task_logged_in_chat_is_the_same_record(signed_in):
    from agrios import agent

    agent.log_task.invoke({"field_id": "field-1", "description": "Check the drip lines"})
    assert "Check the drip lines" in signed_in.get("/farm").text


def test_completing_needs_the_csrf_token(signed_in):
    task = farm.log_task("field-1", "Spray", now=T0)
    assert signed_in.post(f"/farm/tasks/{task.id}/done").status_code == 403
    assert [t.id for t in farm.pending_tasks()] == [task.id]


def test_a_viewer_sees_the_farm_but_cannot_change_tasks(client, clock):
    auth.create_user("viewer", PASSWORD, "viewer", now=clock())
    sign_in(client, username="viewer")
    task = farm.log_task("field-1", "Spray", now=T0)
    page = client.get("/farm")
    assert page.status_code == 200
    assert "Spray" in page.text
    assert 'action="/farm/tasks"' not in page.text  # no form they can't use
    assert "/done" not in page.text
    csrf = page_csrf(client, "/farm")
    assert log_via_page(client).status_code == 403
    assert client.post(f"/farm/tasks/{task.id}/done", data={"csrf_token": csrf}).status_code == 403
    assert [t.description for t in farm.pending_tasks()] == ["Spray"]


def test_an_operator_can_change_tasks(client, clock):
    auth.create_user("operator", PASSWORD, "operator", now=clock())
    sign_in(client, username="operator")
    assert log_via_page(client).status_code == 303
    assert len(farm.pending_tasks()) == 1


# --- weather ----------------------------------------------------------------


FAKE = weather.Forecast(
    location="field-1",
    resolved_to="field-1",
    rain_days=["2026-11-06"],
    total_rain_mm=4.2,
    frost_risk_days=["2026-11-05"],
    forecast=[
        weather.DayForecast(date="2026-11-05", temp_max_c=18, temp_min_c=-0.5, rain_mm=0, frost_risk=True),
        weather.DayForecast(date="2026-11-06", temp_max_c=21.1, temp_min_c=6, rain_mm=4.2, frost_risk=False),
    ],
)


@pytest.fixture
def forecasts(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []

    def fake(location, days=7, **thresholds):
        calls.append({"location": location, "days": days, **thresholds})
        return FAKE

    monkeypatch.setattr(weather, "forecast", fake)
    return calls


def test_the_farm_page_loads_each_forecast_after_the_page(signed_in, forecasts):
    page = signed_in.get("/farm").text
    assert 'hx-get="/farm/weather/field-1"' in page
    assert 'hx-get="/farm/weather/field-2"' in page
    assert forecasts == []  # the page itself never waits on the weather service


def test_weather_is_a_fragment_for_htmx_and_a_page_without_it(signed_in, forecasts):
    settings.update("farm", {"frost_risk_c": "3", "rain_day_mm": "2.5"}, username="owner")
    fragment = signed_in.get("/farm/weather/field-1", headers={"HX-Request": "true"}).text
    page = signed_in.get("/farm/weather/field-1").text
    assert "<html" not in fragment
    assert "<html" in page
    assert forecasts[0] == {"location": "field-1", "days": 7, "frost_risk_c": 3.0, "rain_day_mm": 2.5}
    for html in (fragment, page):
        assert "frost risk" in html
        assert "rain day" in html
        assert "Days with frost risk: 1" in html
        assert "4.2 mm in total" in html
        assert "-0.5 °C" in html
        assert "Thu 05 Nov" in html  # 2026-11-05


def test_a_field_without_coordinates_never_reaches_the_weather_service(signed_in, forecasts):
    with engine.begin() as conn:
        conn.execute(text("INSERT INTO fields (field_id, crop) VALUES ('field-9', 'oats')"))
    try:
        html = signed_in.get("/farm/weather/field-9", headers={"HX-Request": "true"}).text
    finally:
        with engine.begin() as conn:
            conn.execute(text("DELETE FROM fields WHERE field_id = 'field-9'"))
    assert "no coordinates" in html
    assert forecasts == []


def test_a_weather_outage_is_explained(signed_in, monkeypatch):
    def down(*args, **kwargs):
        raise httpx.ConnectError("no route")

    monkeypatch.setattr(weather, "forecast", down)
    response = signed_in.get("/farm/weather/field-1", headers={"HX-Request": "true"})
    assert response.status_code == 200
    assert "answer. Try again in a minute" in response.text


def test_weather_for_an_unknown_field_is_404(signed_in, forecasts):
    assert signed_in.get("/farm/weather/field-99").status_code == 404


# --- schema ---------------------------------------------------------------


def test_init_db_adds_the_task_timestamps_to_an_older_database(tmp_path):
    db = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    with db.begin() as conn:
        conn.execute(
            text("CREATE TABLE tasks (id INTEGER PRIMARY KEY, field_id TEXT, description TEXT, status TEXT)")
        )
        conn.execute(text("INSERT INTO tasks VALUES (1, 'field-1', 'old task', 'pending')"))
    init_db(db)
    init_db(db)  # a second run must be a no-op
    with db.connect() as conn:
        rows = conn.execute(text("SELECT id, created_at, completed_at FROM tasks")).fetchall()
    # An old task's time isn't known, so it's left empty rather than invented.
    assert [tuple(r) for r in rows] == [(1, None, None)]


def test_the_page_shows_no_time_for_a_task_without_one(signed_in):
    with engine.begin() as conn:
        conn.execute(
            text("INSERT INTO tasks (field_id, description, status) VALUES ('field-1', 'legacy', 'pending')")
        )
    page = signed_in.get("/farm").text
    item = re.search(r"legacy.*?</li>", page, flags=re.DOTALL)
    assert item
    assert "logged" not in item.group(0)


def test_a_boosted_click_on_the_forecast_link_gets_the_whole_page(signed_in, forecasts):
    # Boosted requests are htmx requests too, but they replace the whole body.
    page = signed_in.get("/farm/weather/field-1", headers={"HX-Request": "true", "HX-Boosted": "true"})
    assert "<header" in page.text
    assert 'href="/farm/weather/field-1" hx-boost="false"' in signed_in.get("/farm").text
