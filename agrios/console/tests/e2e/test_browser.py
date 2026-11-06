"""The console in a real browser: every screen clean at desktop and phone
widths, the main workflows end to end, and the checks' negative control."""

import threading

import pytest
from langchain_core.messages import AIMessage
from playwright.sync_api import Browser, Request, expect

from agrios.console import auth
from agrios.console.tests.e2e.checks import keyboard, overflow, unnamed_controls
from agrios.console.tests.e2e.conftest import DESKTOP, PHONE, LiveConsole, open_page, screenshot, sign_in
from agrios.console.tests.scripted import call
from agrios.services import farm

pytestmark = pytest.mark.e2e

OWNER_PAGES = [
    "/",
    "/farm",
    "/assistant",
    "/settings/general",
    "/settings/assistant",
    "/settings/farm",
    "/settings/access",
    "/settings/data",
    "/settings/about",
]


@pytest.mark.parametrize(("size", "viewport"), [("desktop", DESKTOP), ("phone", PHONE)])
def test_every_screen_loads_without_errors_and_fits(
    live: LiveConsole, browser: Browser, users: dict[str, auth.User], size: str, viewport: dict[str, int]
) -> None:
    farm.log_task("field-1", "Inspect potato rows for late blight", now=0)
    watched = open_page(browser, viewport)
    page = watched.page
    page.goto(live.url + "/login")
    screenshot(page, f"{size}-login")
    sign_in(watched, live.url, "owner")
    problems: list[str] = []
    for path in OWNER_PAGES:
        page.goto(live.url + path)
        page.wait_for_load_state("networkidle")
        # keyboard() tabs from the top, so it runs before anything is clicked.
        focus_problems, _ = keyboard(page)
        found = focus_problems + overflow(page) + unnamed_controls(page)
        problems += [f"{path}: {p}" for p in found]
        screenshot(page, f"{size}-{path.strip('/').replace('/', '-') or 'today'}")
    assert problems + watched.problems == []


def test_tabbing_the_farm_page_reaches_what_a_task_needs(
    live: LiveConsole, browser: Browser, users: dict[str, auth.User]
) -> None:
    farm.log_task("field-1", "Spray", now=0)
    watched = open_page(browser, DESKTOP)
    sign_in(watched, live.url, "owner")
    watched.page.goto(live.url + "/farm")
    watched.page.wait_for_load_state("networkidle")
    problems, reached = keyboard(watched.page)
    assert problems == []
    joined = "\n".join(reached)
    # The skip link first, then everything needed to log and complete a task.
    assert "skip-link" in reached[0]
    for needed in (
        'href="/farm"',
        'aria-label="Mark done: Spray"',
        'name="field_id"',
        'name="description"',
        "Log task",
    ):
        assert needed in joined, needed


def test_a_task_is_logged_and_completed(
    live: LiveConsole, browser: Browser, users: dict[str, auth.User]
) -> None:
    watched = open_page(browser, DESKTOP)
    page = watched.page
    sign_in(watched, live.url, "operator")
    page.click("nav a[href='/farm']")
    page.select_option("#task-field", "field-2")
    page.fill("#task-description", "Plough before sowing")
    page.click("button:has-text('Log task')")
    pending = page.locator("li.task", has_text="Plough before sowing")
    expect(pending).to_have_count(1)
    pending.locator("button").click()
    expect(page.locator("li.task", has_text="Plough before sowing")).to_have_count(0)
    expect(page.locator("#done-heading ~ ul li", has_text="Plough before sowing")).to_have_count(1)
    # The forecasts arrived after the page, frost and rain in words.
    expect(page.locator("table.forecast")).to_have_count(2)
    expect(page.locator(".flag.frost").first).to_contain_text("frost risk")
    assert watched.problems == []


def test_the_assistant_streams_an_answer_with_its_steps(
    live: LiveConsole, browser: Browser, users: dict[str, auth.User]
) -> None:
    farm.log_task("field-1", "Spray", now=0)
    live.model.replies = [call("get_pending_tasks"), AIMessage("One task is pending: spray field-1.")]
    watched = open_page(browser, DESKTOP)
    page = watched.page
    sign_in(watched, live.url, "operator")
    page.click("nav a[href='/assistant']")
    page.fill("#message", "What's pending?")
    page.press("#message", "Enter")
    expect(page.locator("#conversation")).to_have_attribute("aria-busy", "false", timeout=20_000)
    expect(page.locator(".turn .step")).to_have_text("✔ Listed the pending tasks")
    expect(page.locator(".turn .answer")).to_contain_text("One task is pending: spray field-1.")
    assert "/assistant/console-" in page.url  # the new conversation has its own address
    screenshot(page, "desktop-assistant-answered")
    page.reload()
    expect(page.locator(".turn .step.ok")).to_have_count(1)
    expect(page.locator(".turn .answer")).to_contain_text("One task is pending: spray field-1.")
    expect(page.locator(".threads .list li")).to_have_count(1)
    assert watched.problems == []


def test_enter_twice_while_answering_sends_one_message(
    live: LiveConsole, browser: Browser, users: dict[str, auth.User]
) -> None:
    gate = threading.Event()
    live.model.gate = gate
    live.model.replies = [AIMessage("Hello.")]
    watched = open_page(browser, DESKTOP)
    page = watched.page
    sent: list[str] = []

    def on_request(request: Request) -> None:
        if request.method == "POST" and request.url.endswith("/assistant/stream"):
            sent.append(request.url)

    page.on("request", on_request)
    try:
        sign_in(watched, live.url, "operator")
        page.goto(live.url + "/assistant")
        page.fill("#message", "Hi")
        page.press("#message", "Enter")
        page.fill("#message", "Hi again")
        page.press("#message", "Enter")
        page.wait_for_timeout(300)
        assert len(sent) == 1
        expect(page.locator(".turn")).to_have_count(1)
    finally:
        live.model.gate = None
        gate.set()
    expect(page.locator(".turn .answer")).to_contain_text("Hello.")
    expect(page.locator("button:has-text('Send')")).to_be_enabled()
    assert watched.problems == []


def test_a_viewer_sees_the_farm_read_only(
    live: LiveConsole, browser: Browser, users: dict[str, auth.User]
) -> None:
    watched = open_page(browser, PHONE)
    page = watched.page
    sign_in(watched, live.url, "viewer")
    expect(page.locator("nav a[href='/assistant']")).to_have_count(0)
    page.click("nav a[href='/farm']")
    expect(page.locator("#task-description")).to_have_count(0)
    assert overflow(page) == []
    assert watched.problems == []


def test_the_checks_catch_a_broken_page(
    live: LiveConsole, browser: Browser, users: dict[str, auth.User]
) -> None:
    # Negative control: a page with an inline script and style, a line too
    # long for a phone, an unlabelled text box, a button whose focus ring is
    # removed, and a script that throws. Every check must fire.
    watched = open_page(browser, PHONE)
    page = watched.page
    page.goto(live.url + "/__e2e/broken")
    page.wait_for_load_state("networkidle")
    violations = [p for p in watched.problems if "CSP violation" in p]
    assert any("script-src" in v for v in violations), watched.problems
    assert any("style-src" in v for v in violations), watched.problems
    assert any("planted script error" in p for p in watched.problems if p.startswith("script error"))
    assert overflow(page) != []
    unnamed = unnamed_controls(page)
    assert any("unlabelled" in p for p in unnamed)
    assert any("icon-only" in p for p in unnamed)  # hidden text isn't a name
    assert not any('value="Save"' in p for p in unnamed)  # a submit's value is
    focus_problems, reached = keyboard(page)
    assert any("no-ring" in p for p in focus_problems), reached
