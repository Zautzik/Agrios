import re
from collections.abc import Iterator

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from httpx import Response
from sqlalchemy import delete

from agrios.console import auth
from agrios.console.app import create_app
from agrios.db import (
    assistant_threads_table,
    engine,
    init_db,
    login_attempts_table,
    sessions_table,
    settings_changes_table,
    settings_table,
    tasks_table,
    users_table,
)

LOCAL = ("127.0.0.1", 50000)
PASSWORD = "correct horse battery staple"
SETUP_CODE = "setup-code-for-tests"
T0 = 1_800_000_000.0


class Clock:
    """A clock the tests move by hand, so timeouts are tested without sleeping."""

    def __init__(self) -> None:
        self.t = T0

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _clean() -> None:
    with engine.begin() as conn:
        for table in (
            assistant_threads_table,
            tasks_table,
            sessions_table,
            login_attempts_table,
            users_table,
            settings_changes_table,
            settings_table,
        ):
            conn.execute(delete(table))


@pytest.fixture(autouse=True)
def _clean_console_tables() -> Iterator[None]:
    # Cleaned before and after, so settings saved here never reach other tests
    # sharing the database (the CI Postgres job runs the whole suite on one).
    init_db()
    _clean()
    yield
    _clean()


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def app(clock: Clock) -> FastAPI:
    return create_app(setup_token=SETUP_CODE, clock=clock)


def make_client(app: FastAPI, client: tuple[str, int] = LOCAL) -> TestClient:
    # base_url sets the Host header; "testserver", TestClient's default, is a
    # name the console doesn't answer to (see the DNS-rebinding check).
    return TestClient(app, client=client, base_url="http://localhost", follow_redirects=False)


@pytest.fixture
def client(app: FastAPI) -> Iterator[TestClient]:
    with make_client(app) as c:
        yield c


@pytest.fixture
def owner(clock: Clock) -> auth.User:
    return auth.create_user("owner", PASSWORD, "owner", now=clock())


def csrf_from(html: str) -> str:
    match = re.search(r'name="csrf_token" value="([^"]+)"', html)
    assert match, "page has no CSRF token"
    return match.group(1)


def sign_in(client: TestClient, username: str = "owner", password: str = PASSWORD, **extra: str) -> Response:
    page = client.get("/login")
    return client.post(
        "/login",
        data={"username": username, "password": password, "csrf_token": csrf_from(page.text), **extra},
    )


@pytest.fixture
def signed_in(client: TestClient, owner: auth.User) -> TestClient:
    assert sign_in(client).status_code == 303
    return client


def page_csrf(client: TestClient, path: str = "/") -> str:
    return csrf_from(client.get(path).text)
