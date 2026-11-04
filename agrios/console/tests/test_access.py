"""Who can reach what: the public route set, roles, network exposure and
response headers."""

import re

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from agrios.console import auth
from agrios.console.app import PUBLIC_PATHS
from agrios.console.security import client_allowed
from agrios.console.tests.conftest import PASSWORD, make_client, sign_in
from agrios.services import settings
from agrios.services.settings import AccessSettings


def routes(app: FastAPI) -> list[tuple[str, str]]:
    """Every (method, path), from the OpenAPI schema: a public API that lists
    routes from included routers too. (The schema isn't served, openapi_url is
    None, but the app can still build it.)"""
    found = [(method.upper(), path) for path, item in app.openapi()["paths"].items() for method in item]
    assert len(found) >= 10, "fewer routes than expected: is the enumeration still finding them?"
    return found


def unguarded_routes(app: FastAPI, client: TestClient) -> set[str]:
    """Routes that answer a signed-out request with anything other than the
    sign-in guard's redirect, which is the only response that carries ?next=."""
    unguarded = set()
    for method, path in routes(app):
        url = path.replace("{group}", "general").replace("{field}", "farm_name")
        response = client.request(method, url)
        if not (response.status_code == 303 and response.headers["location"].startswith("/login?next=")):
            unguarded.add(path)
    return unguarded


def test_public_routes_are_exactly_the_listed_ones(app, client, owner):
    assert unguarded_routes(app, client) == PUBLIC_PATHS


def test_route_check_catches_a_forgotten_guard(app: FastAPI, client: TestClient, owner: object) -> None:
    # Negative control: a route added straight onto the app, outside the
    # protected router, must show up as unguarded.
    @app.get("/leak")
    def leak() -> str:
        return "farm data"

    assert unguarded_routes(app, client) == PUBLIC_PATHS | {"/leak"}


@pytest.mark.parametrize("role", ["operator", "viewer"])
def test_settings_are_owner_only(client, clock, role):
    auth.create_user("someone", PASSWORD, role, now=clock())
    sign_in(client, username="someone")
    assert client.get("/").status_code == 200
    page = client.get("/")
    assert "/settings" not in page.text  # no link to a page they can't open
    assert client.get("/settings/general").status_code == 403
    token = page.text.split('name="csrf_token" value="')[1].split('"')[0]
    assert client.post("/settings/general", data={"farm_name": "X", "csrf_token": token}).status_code == 403
    assert settings.load().general.farm_name == "My farm"


# --- network exposure -----------------------------------------------------


def access(**values: object) -> AccessSettings:
    return AccessSettings.model_validate(values)


@pytest.mark.parametrize(
    ("host", "settings_values", "allowed"),
    [
        ("127.0.0.1", {}, True),
        ("::1", {}, True),
        ("192.168.1.20", {}, False),  # the default is this machine only
        ("192.168.1.20", {"exposure": "lan"}, True),
        ("::ffff:192.168.1.20", {"exposure": "lan"}, True),  # IPv4 on a dual-stack socket
        ("fd12:3456::1", {"exposure": "lan"}, True),
        ("8.8.8.8", {"exposure": "lan"}, False),
        ("100.64.0.7", {"exposure": "lan"}, False),
        ("100.64.0.7", {"exposure": "custom", "allowed_networks": ["100.64.0.0/10"]}, True),
        ("192.168.1.20", {"exposure": "custom", "allowed_networks": ["100.64.0.0/10"]}, False),
        ("127.0.0.1", {"exposure": "custom", "allowed_networks": []}, True),  # can't lock yourself out
        ("testclient", {"exposure": "lan"}, False),  # not an address: refused, not guessed at
        (None, {"exposure": "lan"}, False),
    ],
)
def test_client_allowed(host, settings_values, allowed):
    assert client_allowed(host, access(**settings_values)) is allowed


def test_network_guard_covers_pages_and_static_files(app, owner):
    with make_client(app, client=("192.168.1.20", 50000)) as lan_client:
        for path in ("/login", "/static/app.css", "/healthz"):
            assert lan_client.get(path).status_code == 403, path
        settings.update("access", {"exposure": "lan"}, username="owner")
        for path in ("/login", "/static/app.css", "/healthz"):
            assert lan_client.get(path).status_code == 200, path


def test_forwarded_header_does_not_change_the_client(app):
    with make_client(app, client=("192.168.1.20", 50000)) as lan_client:
        assert lan_client.get("/healthz", headers={"x-forwarded-for": "127.0.0.1"}).status_code == 403


# --- headers --------------------------------------------------------------


def test_security_headers_on_every_page(client: TestClient, owner: object) -> None:
    response = client.get("/login")
    csp = response.headers["content-security-policy"]
    assert "script-src 'self'" in csp
    assert "frame-ancestors 'none'" in csp
    assert "unsafe-inline" not in csp
    assert "unsafe-eval" not in csp
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"


def test_no_inline_script_or_style_in_pages(signed_in):
    # The CSP above would block them; this catches one before a browser does.
    for path in ("/", "/settings/general", "/settings/access", "/settings/about", "/settings/data"):
        html = signed_in.get(path).text
        # Any <script> without src is inline, whatever its other attributes.
        assert not re.search(r"<script(?![^>]*\bsrc=)[^>]*>", html), path
        for forbidden in ("<style", " style=", " onclick=", "hx-on"):
            assert forbidden not in html, (path, forbidden)
