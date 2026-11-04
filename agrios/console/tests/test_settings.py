"""The settings service and the forms built from it."""

import json
from typing import Any

import pytest
from pydantic import BaseModel, ValidationError
from sqlalchemy import insert

from agrios.console import forms
from agrios.console.tests.conftest import page_csrf
from agrios.db import engine, settings_table
from agrios.services import settings, weather
from agrios.services.settings import FarmSettings


def test_defaults_when_nothing_is_stored():
    assert settings.load() == settings.Settings()


def test_update_saves_and_logs_only_what_changed():
    settings.update("farm", {"frost_risk_c": "0.5", "rain_day_mm": "1.0"}, username="owner", now=100.0)
    assert settings.load().farm.frost_risk_c == 0.5
    log = settings.changes()
    assert [(c.setting, c.old_value, c.new_value, c.username, c.at) for c in log] == [
        ("farm.frost_risk_c", 2.0, 0.5, "owner", 100.0)
    ]
    settings.update("farm", {"frost_risk_c": "0.5"}, username="owner")  # same value: nothing new
    assert len(settings.changes()) == 1


def test_invalid_value_saves_nothing():
    with pytest.raises(ValidationError):
        settings.update("farm", {"rain_day_mm": "3", "frost_risk_c": "99"}, username="owner")
    assert settings.load().farm == FarmSettings()  # the valid field in the same update wasn't saved either
    assert settings.changes() == []


@pytest.mark.parametrize(
    ("group", "values"),
    [("farm", {"k_enter": 1}), ("safety", {"failsafe_ms": 1000}), ("access", {"exposure ": "lan"})],
)
def test_names_that_are_not_settings_are_refused(group, values):
    with pytest.raises(settings.UnknownSettingError):
        settings.update(group, values, username="owner")


def test_reset_restores_the_default_and_logs_it():
    settings.update("general", {"farm_name": "Fundo Los Robles"}, username="owner")
    settings.reset("general", "farm_name", username="owner")
    assert settings.load().general.farm_name == "My farm"
    assert [c.new_value for c in settings.changes("general")] == ["My farm", "Fundo Los Robles"]


def test_a_stored_value_that_no_longer_validates_falls_back_alone():
    # e.g. a constraint tightened in a later version: that field reverts to its
    # default, the rest of the group is kept, and loading never fails.
    stored = {"frost_risk_c": 99.0, "rain_day_mm": 4.0}
    with engine.begin() as conn:
        conn.execute(insert(settings_table).values(group_name="farm", value=json.dumps(stored)))
    assert settings.load().farm == FarmSettings(rain_day_mm=4.0)


@pytest.mark.parametrize(
    ("networks", "ok"),
    [
        (["192.168.1.0/24"], True),
        (["fd00::/8", "10.0.0.0/8"], True),
        (["192.168.1.5/24"], False),
        (["nope"], False),
    ],
)
def test_allowed_networks_are_validated(networks, ok):
    if ok:
        settings.update("access", {"allowed_networks": networks}, username="owner")
        assert [str(n) for n in settings.load().access.allowed_networks] == networks
    else:
        with pytest.raises(ValidationError):
            settings.update("access", {"allowed_networks": networks}, username="owner")


def test_the_assistant_reads_the_farm_thresholds(monkeypatch):
    import main  # imported here: heavy, and only this test needs the agent module

    seen: dict[str, Any] = {}

    def fake_forecast(location, days=7, **thresholds):
        seen.update(thresholds)
        raise weather.LocationNotFoundError(location)

    # main calls weather.forecast through the module, so patching the module reaches it.
    monkeypatch.setattr(weather, "forecast", fake_forecast)
    settings.update("farm", {"frost_risk_c": "4.5", "rain_day_mm": "2"}, username="owner")
    main.get_weather_forecast.invoke({"location": "field-1"})
    assert seen == {"frost_risk_c": 4.5, "rain_day_mm": 2.0}


# --- forms ----------------------------------------------------------------


def test_form_limits_come_from_the_model():
    spec = {s.name: s for s in forms.field_specs(FarmSettings)}["frost_risk_c"]
    assert (spec.widget, spec.minimum, spec.maximum, spec.default) == ("number", -10, 10, "2.0")
    access = {s.name: s for s in forms.field_specs(settings.AccessSettings)}
    assert access["exposure"].options == ("local", "lan", "custom")
    assert access["allowed_networks"].widget == "lines"
    assert access["port"].applies == "restart"


def test_every_setting_has_a_widget():
    for model in settings.GROUPS.values():
        assert forms.field_specs(model)


def test_a_field_without_a_widget_is_an_error_not_a_blank():
    class Unsupported(BaseModel):
        mapping: dict[str, int] = {}

    with pytest.raises(TypeError, match="no form widget"):
        forms.field_specs(Unsupported)


def test_settings_page_saves_and_shows_history(signed_in):
    page = signed_in.get("/settings/farm")
    assert page.status_code == 200
    assert 'min="-10"' in page.text
    assert 'max="10"' in page.text
    response = signed_in.post(
        "/settings/farm", data={"frost_risk_c": "1.5", "rain_day_mm": "1.0", "csrf_token": csrf(page)}
    )
    assert response.status_code == 200
    assert "Saved." in response.text
    assert "1.5" in response.text.split("Recent changes")[1]
    assert settings.load().farm.frost_risk_c == 1.5


def csrf(page: Any) -> str:
    return str(page.text.split('name="csrf_token" value="')[1].split('"')[0])


def test_settings_page_explains_errors_and_keeps_what_was_typed(signed_in):
    data = {"frost_risk_c": "99", "rain_day_mm": "5", "csrf_token": page_csrf(signed_in, "/settings/farm")}
    response = signed_in.post("/settings/farm", data=data)
    assert response.status_code == 422
    assert "Not saved." in response.text
    assert 'aria-invalid="true"' in response.text
    assert "less than or equal to 10" in response.text
    assert 'value="99"' in response.text
    assert 'value="5"' in response.text
    assert settings.load().farm == FarmSettings()


def test_restart_settings_say_so_when_saved(signed_in):
    data = {
        "model": "llama3.2:3b",
        "ollama_url": "",
        "csrf_token": page_csrf(signed_in, "/settings/assistant"),
    }
    response = signed_in.post("/settings/assistant", data=data)
    assert "Takes effect after a restart: Model." in response.text


def test_reset_button_and_header_follow_the_saved_name(signed_in):
    token = page_csrf(signed_in, "/settings/general")
    response = signed_in.post(
        "/settings/general", data={"farm_name": "Fundo Los Robles", "csrf_token": token}
    )
    assert "<title>Settings · General · Fundo Los Robles</title>" in response.text
    reset = signed_in.post("/settings/general/reset/farm_name", data={"csrf_token": token})
    assert reset.status_code == 200
    assert settings.load().general.farm_name == "My farm"


def test_unknown_group_or_field_is_not_found(signed_in):
    token = page_csrf(signed_in)
    assert signed_in.get("/settings/safety").status_code == 404
    assert signed_in.post("/settings/safety", data={"csrf_token": token}).status_code == 404
    assert signed_in.post("/settings/farm/reset/k_enter", data={"csrf_token": token}).status_code == 404


def test_extra_form_fields_cannot_reach_the_service(signed_in):
    data = {
        "frost_risk_c": "2.0",
        "rain_day_mm": "1.0",
        "k_enter": "1",
        "csrf_token": page_csrf(signed_in, "/settings/farm"),
    }
    assert signed_in.post("/settings/farm", data=data).status_code == 200
    assert settings.changes() == []


def test_setting_text_is_escaped_not_trusted(signed_in):
    # The assistant's help text contains "<name>". Passed through _() it was
    # marked safe and reached the page as a raw tag; it must arrive escaped.
    html = signed_in.get("/settings/assistant").text
    assert "ollama pull &lt;name&gt;" in html
    assert "ollama pull <name>" not in html


@pytest.mark.parametrize("raw", ["{not json", "[1, 2]", "null"])
def test_unreadable_stored_group_falls_back_to_defaults(signed_in, raw):
    with engine.begin() as conn:
        conn.execute(insert(settings_table).values(group_name="general", value=raw))
    assert settings.load().general == settings.GeneralSettings()
    assert signed_in.get("/settings/general").status_code == 200  # the UI still loads, to save over it


@pytest.mark.parametrize(
    ("names", "ok"),
    [(["farm-pc.local"], True), (["Farm-PC.Local "], True), (["evil.example/x"], False), (["-bad"], False)],
)
def test_hostnames_are_validated(names, ok):
    if ok:
        settings.update("access", {"hostnames": names}, username="owner")
        assert settings.load().access.hostnames == ("farm-pc.local",)
    else:
        with pytest.raises(ValidationError):
            settings.update("access", {"hostnames": names}, username="owner")
