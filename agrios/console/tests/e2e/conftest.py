"""Browser tests: the console served by a real uvicorn on a free port, driven
by Chromium through Playwright.

Everything a page does in a browser that the TestClient can't see is checked
here: Content-Security-Policy violations, script errors, layout at phone width,
keyboard focus, streaming. The assistant answers from a script (no Ollama in
CI) and the forecast is faked (no network), so a run is the same every time.

Run with `uv run pytest -m e2e`. A browser is needed: `uv run playwright
install chromium`, or set AGRIOS_E2E_CHANNEL=msedge to use an installed Edge.
"""

import os
import socket
import threading
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from playwright.sync_api import Browser, ConsoleMessage, Error, Page, sync_playwright

from agrios import agent
from agrios.console import auth
from agrios.console.app import create_app
from agrios.console.tests.conftest import PASSWORD
from agrios.console.tests.scripted import ScriptedModel
from agrios.services import weather

DESKTOP = {"width": 1280, "height": 900}
PHONE = {"width": 390, "height": 844}

# Recorded in the page before any of its own scripts run. An init script is
# injected by the browser, not by the page, so the page's CSP doesn't apply to it.
WATCH = """
document.addEventListener('securitypolicyviolation', e =>
  console.error('CSP violation: ' + e.violatedDirective + ' at ' + e.sourceFile + ':' + e.lineNumber));
"""

# A page with one of each problem the checks below must catch: an inline
# script and an inline style (both refused by the CSP), a line too long for a
# phone, and a text box with no label.
BROKEN_PAGE = (
    """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>broken</title></head>
<body><script>console.log("inline")</script><p style="color:red">styled</p>
<p>"""
    + "x" * 400
    + """</p><input name="unlabelled"></body></html>"""
)


def forecast(location: str, days: int = 7, **thresholds: Any) -> weather.Forecast:
    return weather.Forecast(
        location=location,
        resolved_to=location,
        rain_days=["2026-11-06"],
        total_rain_mm=4.2,
        frost_risk_days=["2026-11-05"],
        forecast=[
            weather.DayForecast(
                date="2026-11-05", temp_max_c=18, temp_min_c=-0.5, rain_mm=0, frost_risk=True
            ),
            weather.DayForecast(
                date="2026-11-06", temp_max_c=21.1, temp_min_c=6, rain_mm=4.2, frost_risk=False
            ),
        ],
    )


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port: int = s.getsockname()[1]
        return port


@dataclass
class LiveConsole:
    url: str
    app: FastAPI
    model: ScriptedModel


@pytest.fixture(scope="session")
def live() -> Iterator[LiveConsole]:
    patch = pytest.MonkeyPatch()
    model = ScriptedModel()
    patch.setattr(agent, "model", model)
    patch.setattr(agent, "model_with_tools", model)
    patch.setattr(agent, "summarizer_model", model)
    patch.setattr(weather, "forecast", forecast)
    agent._model_with.cache_clear()

    app = create_app()

    @app.get("/__e2e/broken", response_class=HTMLResponse)
    def broken() -> str:
        # Test-only, on this test app: the negative control for the page checks.
        return BROKEN_PAGE

    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 15
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("the console didn't start")
        time.sleep(0.05)
    # 127.0.0.1, not localhost: the console answers to IP literals (see the Host check).
    yield LiveConsole(url=f"http://127.0.0.1:{port}", app=app, model=model)
    server.should_exit = True
    thread.join(timeout=10)
    patch.undo()
    agent._model_with.cache_clear()


@pytest.fixture(scope="session")
def browser() -> Iterator[Browser]:
    channel = os.environ.get("AGRIOS_E2E_CHANNEL") or None
    with sync_playwright() as p:
        try:
            launched = p.chromium.launch(channel=channel, headless=True)
        except Error as e:
            if os.environ.get("CI"):
                raise  # in CI a missing browser is a failure, never a silent skip
            pytest.skip(f"no browser for Playwright ({e.message.splitlines()[0]})")
        yield launched
        launched.close()


@dataclass
class Watched:
    """A page plus everything that went wrong in it."""

    page: Page
    problems: list[str] = field(default_factory=list)


def open_page(browser: Browser, viewport: dict[str, int]) -> Watched:
    page = browser.new_context(viewport=viewport).new_page()  # type: ignore[arg-type]
    watched = Watched(page)
    page.add_init_script(WATCH)

    def on_console(message: ConsoleMessage) -> None:
        if message.type == "error":
            watched.problems.append(f"console: {message.text}")

    page.on("console", on_console)
    page.on("pageerror", lambda e: watched.problems.append(f"script error: {e}"))
    return watched


@pytest.fixture
def users() -> dict[str, auth.User]:
    # Function-scoped: the console tests' autouse fixture empties the user
    # table around every test.
    return {
        role: auth.create_user(role, PASSWORD, role, now=time.time())
        for role in ("owner", "operator", "viewer")
    }


def sign_in(watched: Watched, base: str, username: str) -> None:
    page = watched.page
    page.goto(base + "/login")
    page.fill("#username", username)
    page.fill("#password", PASSWORD)
    page.click("button.primary")
    page.wait_for_url(base + "/")


def screenshot(page: Page, name: str) -> None:
    """Saved when AGRIOS_SCREENSHOTS names a directory (CI uploads it)."""
    folder = os.environ.get("AGRIOS_SCREENSHOTS")
    if folder:
        Path(folder).mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(Path(folder) / f"{name}.png"), full_page=True)
