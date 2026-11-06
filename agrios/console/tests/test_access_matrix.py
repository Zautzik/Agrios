"""Every signed-in route, requested as each role, against a table of who may
use it. The table must list every route: one added without a decision about
who can reach it fails here. (Signed-out access is checked route by route in
test_access.py: only the public pages answer.)"""

from typing import Any

import pytest
from fastapi import APIRouter, Depends, FastAPI

from agrios.console import auth, conversations
from agrios.console.app import PUBLIC_PATHS, create_app
from agrios.console.tests.conftest import PASSWORD, T0, Clock, make_client, page_csrf, sign_in
from agrios.console.tests.test_access import routes
from agrios.console.web import SignedIn, signed_in
from agrios.services import farm, weather

ROLES = ("viewer", "operator", "owner")
EVERYONE = frozenset(ROLES)
OPERATORS = frozenset({"operator", "owner"})
OWNER = frozenset({"owner"})

EXPECTED: dict[tuple[str, str], frozenset[str]] = {
    ("GET", "/"): EVERYONE,
    ("POST", "/logout"): EVERYONE,
    ("GET", "/farm"): EVERYONE,
    ("GET", "/farm/weather/{field_id}"): EVERYONE,
    ("POST", "/farm/tasks"): OPERATORS,
    ("POST", "/farm/tasks/{task_id}/done"): OPERATORS,
    # The assistant can log tasks, so it's for those who may change them.
    ("GET", "/assistant"): OPERATORS,
    ("GET", "/assistant/{thread_id}"): OPERATORS,
    ("POST", "/assistant/stream"): OPERATORS,
    ("POST", "/assistant/send"): OPERATORS,
    ("POST", "/assistant/{thread_id}/delete"): OPERATORS,
    ("GET", "/settings"): OWNER,
    ("GET", "/settings/data"): OWNER,
    ("GET", "/settings/about"): OWNER,
    ("GET", "/settings/{group}"): OWNER,
    ("POST", "/settings/{group}"): OWNER,
    ("POST", "/settings/{group}/reset/{field}"): OWNER,
}

# What each route is sent: a request that reaches the handler without side
# effects beyond the test database (empty messages are refused before any
# model runs; the forecast is faked below).
FORM: dict[tuple[str, str], dict[str, str]] = {
    ("POST", "/farm/tasks"): {"field_id": "field-1", "description": "matrix check"},
    ("POST", "/assistant/stream"): {"message": ""},
    ("POST", "/assistant/send"): {"message": ""},
    ("POST", "/settings/{group}"): {"farm_name": "Matrix farm"},
}


def signed_in_routes(app: FastAPI) -> list[tuple[str, str]]:
    return [(m, p) for m, p in routes(app) if p not in PUBLIC_PATHS]


def check_matrix(
    app: FastAPI, clock: Clock, expected: dict[tuple[str, str], frozenset[str]]
) -> tuple[list[str], int]:
    """Every way the app differs from `expected`: routes missing from the
    table, and each (route, role) whose answer doesn't match it. A refused
    request gets 403; anything else means the handler was reached. Also
    returns how many requests were checked, so an empty run can't pass."""
    found = signed_in_routes(app)
    problems = [f"not in the table: {m} {p}" for m, p in found if (m, p) not in expected]
    problems += [f"in the table but gone: {m} {p}" for m, p in expected if (m, p) not in found]
    checked = 0
    for role in ROLES:
        user = auth.create_user(f"matrix-{role}", PASSWORD, role, now=clock())  # type: ignore[arg-type]
        task = farm.log_task("field-2", f"matrix task for {role}", now=T0)
        thread = conversations.create(user.id, "matrix", now=clock()).thread_id
        values = {"{group}": "general", "{field}": "farm_name", "{task_id}": str(task.id),
                  "{field_id}": "field-1", "{thread_id}": thread}  # fmt: skip
        with make_client(app) as client:
            sign_in(client, username=user.username)
            csrf = page_csrf(client, "/")
            # Sign-out last: it ends the session the other requests use.
            for method, path in sorted(found, key=lambda r: r == ("POST", "/logout")):
                if (method, path) not in expected:
                    continue
                url = path
                for name, value in values.items():
                    url = url.replace(name, value)
                data = {**FORM.get((method, path), {}), "csrf_token": csrf} if method == "POST" else None
                status = client.request(method, url, data=data).status_code
                checked += 1
                allowed = role in expected[(method, path)]
                if (status == 403) == allowed or status >= 500:
                    problems.append(
                        f"{role} {method} {path}: {status}, expected {'allowed' if allowed else '403'}"
                    )
    return problems, checked


@pytest.fixture
def fake_forecast(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(location: str, days: int = 7, **thresholds: Any) -> weather.Forecast:
        return weather.Forecast(
            location=location,
            resolved_to=location,
            rain_days=[],
            total_rain_mm=0,
            frost_risk_days=[],
            forecast=[],
        )

    monkeypatch.setattr(weather, "forecast", fake)


def test_every_role_reaches_exactly_what_the_table_says(app, clock, fake_forecast):
    problems, checked = check_matrix(app, clock, EXPECTED)
    assert problems == []
    assert checked == len(EXPECTED) * len(ROLES) == 51


def test_the_matrix_catches_a_wrong_guard_and_an_unlisted_route(clock, fake_forecast):
    # Negative control: an owner-only route guarded only by sign-in, and a
    # route nobody decided about, must both be reported.
    app = create_app(clock=clock)
    extra = APIRouter(dependencies=[Depends(signed_in)])

    def handler(session: SignedIn) -> str:
        return "secret"

    extra.add_api_route("/owners-only", handler, methods=["GET"])
    extra.add_api_route("/undecided", handler, methods=["GET"])
    app.include_router(extra)
    problems, _ = check_matrix(app, clock, {**EXPECTED, ("GET", "/owners-only"): OWNER})
    assert "not in the table: GET /undecided" in problems
    assert "viewer GET /owners-only: 200, expected 403" in problems
    assert "operator GET /owners-only: 200, expected 403" in problems
