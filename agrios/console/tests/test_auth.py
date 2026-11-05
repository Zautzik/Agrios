"""First-run setup, sign-in, backoff, sessions and CSRF, each with the case
that must be refused next to the case that must work."""

import hashlib
import urllib.parse

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import Response
from sqlalchemy import func, select

from agrios.console import auth
from agrios.console.app import create_app, safe_next
from agrios.console.security import CSRF_COOKIE
from agrios.console.tests.conftest import (
    PASSWORD,
    SETUP_CODE,
    Clock,
    csrf_from,
    make_client,
    page_csrf,
    sign_in,
)
from agrios.console.web import SESSION_COOKIE
from agrios.db import engine, sessions_table, users_table
from agrios.services import settings


def user_count() -> int:
    with engine.connect() as conn:
        return int(conn.execute(select(func.count()).select_from(users_table)).scalar() or 0)


def setup(
    client: TestClient,
    code: str = SETUP_CODE,
    username: str = "owner",
    password: str = PASSWORD,
    confirm: str | None = None,
) -> Response:
    page = client.get("/setup")
    data = {
        "setup_code": code,
        "username": username,
        "password": password,
        "confirm": password if confirm is None else confirm,
        "csrf_token": csrf_from(page.text),
    }
    return client.post("/setup", data=data)


# --- first run ------------------------------------------------------------


def test_first_run_sends_everything_to_setup(client):
    for path in ("/", "/settings/general", "/login"):
        response = client.get(path)
        assert response.status_code == 303
        assert response.headers["location"].startswith("/setup")


def test_setup_needs_the_printed_code(client):
    assert setup(client, code="guess").status_code == 403
    assert user_count() == 0
    response = setup(client)
    assert response.status_code == 303
    assert client.get("/").status_code == 200  # signed straight in
    assert user_count() == 1


def test_setup_closes_after_the_first_owner(client):
    setup(client)
    client.cookies.clear()
    assert client.get("/setup").headers["location"] == "/login"
    # Posting the form directly, with the code that worked once, still can't add an owner.
    data = {
        "setup_code": SETUP_CODE,
        "username": "intruder",
        "password": PASSWORD,
        "confirm": PASSWORD,
        "csrf_token": page_csrf(client, "/login"),
    }
    response = client.post("/setup", data=data)
    assert response.status_code == 303
    assert response.headers["location"] == "/login"
    assert user_count() == 1


def test_setup_disabled_without_a_code(clock):
    with make_client(create_app(setup_token=None, clock=clock)) as client:
        assert "python -m agrios.console" in client.get("/setup").text
        # No form is shown, but a hand-made post with a valid CSRF token and any code is still refused.
        data = {
            "setup_code": "any-guess",
            "username": "owner",
            "password": PASSWORD,
            "confirm": PASSWORD,
            "csrf_token": client.cookies[CSRF_COOKIE],
        }
        assert client.post("/setup", data=data).status_code == 403
        assert user_count() == 0


@pytest.mark.parametrize(
    ("username", "password", "confirm"),
    [("owner", "short", None), ("Not Valid!", PASSWORD, None), ("owner", PASSWORD, PASSWORD + "x")],
)
def test_setup_rejects_bad_credentials(client, username, password, confirm):
    assert setup(client, username=username, password=password, confirm=confirm).status_code == 422
    assert user_count() == 0


# --- sign-in --------------------------------------------------------------


def test_wrong_password_and_unknown_user_look_the_same(client, owner):
    wrong_password = sign_in(client, password="wrong password, long enough")
    unknown_user = sign_in(client, username="nobody")
    assert wrong_password.status_code == unknown_user.status_code == 401
    assert "Wrong username or password." in wrong_password.text
    assert "Wrong username or password." in unknown_user.text


def test_backoff_refuses_even_the_right_password_until_it_expires(app, client, owner, clock):
    for _ in range(auth.FREE_ATTEMPTS):
        assert sign_in(client, password="not the password at all").status_code == 401
    locked = sign_in(client)
    assert locked.status_code == 429
    assert int(locked.headers["retry-after"]) == int(auth.FIRST_LOCK_S) + 1
    # A different machine isn't locked out by guesses from this one.
    with make_client(app, client=("127.0.0.2", 50000)) as other:
        assert sign_in(other).status_code == 303
    clock.advance(auth.FIRST_LOCK_S + 1)
    assert sign_in(client).status_code == 303


def test_lock_doubles_and_is_capped():
    assert auth.lock_remaining(auth.FREE_ATTEMPTS - 1, 0.0, 0.0) == 0.0
    assert auth.lock_remaining(auth.FREE_ATTEMPTS, 0.0, 0.0) == auth.FIRST_LOCK_S
    assert auth.lock_remaining(auth.FREE_ATTEMPTS + 1, 0.0, 0.0) == 2 * auth.FIRST_LOCK_S
    assert auth.lock_remaining(auth.FREE_ATTEMPTS + 20, 0.0, 0.0) == auth.MAX_LOCK_S


def test_next_parameter_cannot_leave_the_site(client, owner):
    assert sign_in(client, next="/settings/farm").headers["location"] == "/settings/farm"
    client.cookies.clear()
    assert sign_in(client, next="//evil.example/").headers["location"] == "/"
    for target in ("https://evil.example", "/\\evil.example", "", None):
        assert safe_next(target) == "/"


# --- sessions -------------------------------------------------------------


def test_session_cookie_flags_and_only_a_hash_stored(client, owner):
    response = sign_in(client)
    cookie = response.headers["set-cookie"].lower()
    assert "httponly" in cookie
    assert "samesite=strict" in cookie
    token = client.cookies[SESSION_COOKIE]
    with engine.connect() as conn:
        stored = conn.execute(select(sessions_table.c.token_hash)).scalars().all()
    assert stored == [hashlib.sha256(token.encode()).hexdigest()]
    assert token not in stored


def test_every_sign_in_gets_a_new_token(client, owner):
    sign_in(client)
    first = client.cookies[SESSION_COOKIE]
    sign_in(client)
    assert client.cookies[SESSION_COOKIE] != first


def test_idle_timeout_follows_the_setting(signed_in, clock):
    idle = settings.load().access.session_idle_minutes * 60
    clock.advance(idle - 60)
    assert signed_in.get("/").status_code == 200  # activity resets the idle timer
    clock.advance(idle - 60)
    assert signed_in.get("/").status_code == 200
    clock.advance(idle + 1)
    response = signed_in.get("/")
    assert response.status_code == 303
    assert response.headers["location"].startswith("/login")


def test_absolute_timeout_ends_even_an_active_session(signed_in, clock):
    max_s = settings.load().access.session_max_hours * 3600
    step = 20 * 60
    for _ in range(int(max_s // step) - 1):
        clock.advance(step)
        assert signed_in.get("/").status_code == 200
    clock.advance(2 * step)
    assert signed_in.get("/").status_code == 303


def test_sign_out_ends_the_session_server_side(signed_in):
    token = signed_in.cookies[SESSION_COOKIE]
    assert signed_in.post("/logout", data={"csrf_token": page_csrf(signed_in)}).status_code == 303
    signed_in.cookies.set(SESSION_COOKIE, token)  # a copied cookie is no use afterwards
    assert signed_in.get("/").status_code == 303


# --- CSRF -----------------------------------------------------------------


def farm_name() -> str:
    return settings.load().general.farm_name


def test_csrf_token_required_on_changes(signed_in):
    base = {"farm_name": "Changed"}
    assert signed_in.post("/settings/general", data=base).status_code == 403
    assert signed_in.post("/settings/general", data={**base, "csrf_token": "forged"}).status_code == 403
    assert farm_name() == "My farm"
    good = signed_in.post("/settings/general", data={**base, "csrf_token": page_csrf(signed_in)})
    assert good.status_code == 200
    assert farm_name() == "Changed"


@pytest.mark.parametrize(
    "headers", [{"origin": "http://evil.example"}, {"sec-fetch-site": "cross-site"}, {"origin": "null"}]
)
def test_cross_site_requests_refused_even_with_the_token(signed_in, headers):
    data = {"farm_name": "Changed", "csrf_token": page_csrf(signed_in)}
    assert signed_in.post("/settings/general", data=data, headers=headers).status_code == 403
    assert farm_name() == "My farm"
    same_site = {"origin": "http://localhost", "sec-fetch-site": "same-origin"}
    assert signed_in.post("/settings/general", data=data, headers=same_site).status_code == 200


def test_csrf_token_header_works_for_scripts(signed_in):
    token = page_csrf(signed_in)
    response = signed_in.post(
        "/settings/general", data={"farm_name": "Via header"}, headers={"x-csrf-token": token}
    )
    assert response.status_code == 200
    assert farm_name() == "Via header"


def test_sign_in_form_has_csrf_too(client, owner):
    response = client.post("/login", data={"username": "owner", "password": PASSWORD})
    assert response.status_code == 403
    assert SESSION_COOKIE not in client.cookies


def test_sessions_are_shared_by_app_instances(owner, clock):
    # Two apps over the same database share sessions, as two workers would.
    app_a: FastAPI = create_app(clock=clock)
    app_b: FastAPI = create_app(clock=Clock())
    with make_client(app_a) as a:
        sign_in(a)
        token = a.cookies[SESSION_COOKIE]
    with make_client(app_b) as b:
        b.cookies.set(SESSION_COOKIE, token)
        assert b.get("/").status_code == 200


# --- found in review: concurrency, rebinding, the remaining negatives ------


def test_concurrent_guesses_cannot_skip_the_backoff(owner, clock):
    # Before the fix, 40 parallel wrong passwords were all checked and none
    # locked out. Now at most FREE_ATTEMPTS are checked; the rest are refused.
    from collections import Counter
    from concurrent.futures import ThreadPoolExecutor

    def guess(i: int) -> str:
        result = auth.authenticate("owner", f"wrong guess number {i:04d}", client="10.0.0.9", now=clock())
        return type(result).__name__

    with ThreadPoolExecutor(max_workers=40) as pool:
        outcomes = Counter(pool.map(guess, range(40)))
    assert outcomes == Counter({"NoneType": auth.FREE_ATTEMPTS, "LockedOut": 40 - auth.FREE_ATTEMPTS})


def test_rebinding_host_is_refused_before_anything_else(app, owner):
    # A page on evil.example rebinds its name to 127.0.0.1: the browser connects
    # from loopback with matching Host and Origin. The Host check is what stops it.
    from fastapi.testclient import TestClient as Client

    evil = {"origin": "http://evil.example:8000", "sec-fetch-site": "same-origin"}
    with Client(app, client=("127.0.0.1", 50000), base_url="http://evil.example:8000") as rebound:
        assert rebound.get("/login", headers=evil).status_code == 421
        assert rebound.get("/healthz").status_code == 421
    for base in ("http://localhost:8000", "http://127.0.0.1:8000", "http://[::1]:8000"):
        with Client(app, client=("127.0.0.1", 50000), base_url=base) as direct:
            assert direct.get("/healthz").status_code == 200, base
    settings.update("access", {"hostnames": ["farm-pc.local"]}, username="owner")
    with Client(app, client=("127.0.0.1", 50000), base_url="http://farm-pc.local:8000") as named:
        assert named.get("/healthz").status_code == 200


def test_sign_out_needs_the_csrf_token(signed_in):
    # A cross-site page can't sign you out (or anything else) without the token.
    assert signed_in.post("/logout").status_code == 403
    assert signed_in.get("/").status_code == 200


def test_sign_in_csrf_cookie_must_match_the_form(client, owner):
    client.get("/login")  # sets the cookie
    data = {"username": "owner", "password": PASSWORD, "csrf_token": "not-the-cookie-value"}
    assert client.post("/login", data=data).status_code == 403
    assert SESSION_COOKIE not in client.cookies


def test_unknown_user_takes_as_long_as_a_wrong_password(owner, clock):
    # The claim is about time, so measure time: medians of 7 each, one argon2
    # verification on both paths. A missing dummy verify would be ~100x faster.
    import statistics
    import time as _time

    def median_s(username: str) -> float:
        samples = []
        for i in range(7):
            start = _time.perf_counter()
            auth.authenticate(
                username, "wrong password here", client=f"10.1.{i}.{len(username)}", now=clock()
            )
            samples.append(_time.perf_counter() - start)
        return statistics.median(samples)

    known, unknown = median_s("owner"), median_s("nobody-at-all")
    assert 0.5 < unknown / known < 2.0, (known, unknown)


def test_a_very_long_lockout_does_not_overflow():
    assert auth.lock_remaining(5000, 0.0, 0.0) == auth.MAX_LOCK_S


@pytest.mark.parametrize(
    ("referer", "comes_back_to"),
    [
        ("http://localhost/farm", "/farm"),  # the page the form was on
        ("http://localhost/assistant/console-abc?x=1", "/assistant/console-abc?x=1"),
        ("https://evil.example/farm", "/"),  # another site's page is never a target
        (None, "/"),
    ],
)
def test_a_form_posted_after_the_session_expired_comes_back_to_its_page(
    client, owner, clock, referer, comes_back_to
):
    # The form's own address (/farm/tasks) has no page: coming back to it after
    # signing in would be a 405.
    sign_in(client)
    csrf = page_csrf(client, "/farm")
    clock.advance(31 * 60)  # past the idle timeout
    headers = {"referer": referer} if referer else {}
    response = client.post("/farm/tasks", data={"csrf_token": csrf, "description": "x"}, headers=headers)
    assert response.status_code == 303
    assert response.headers["location"] == "/login?next=" + urllib.parse.quote(comes_back_to)


def test_a_page_request_after_the_session_expired_still_comes_back_to_itself(client, owner, clock):
    sign_in(client)
    clock.advance(31 * 60)
    assert client.get("/farm").headers["location"] == "/login?next=/farm"
