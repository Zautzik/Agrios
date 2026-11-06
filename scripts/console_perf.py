"""Console performance, measured: what each page costs the server, what it
costs a browser to show, and how long the Assistant takes to start answering
when the model itself takes no time.

Runs the console on a throwaway database with the forecast faked and the
assistant answering from a script, so the numbers are the console's own:
not Open-Meteo's, not Ollama's. Like every latency figure in this repo, the
result is only meaningful next to its machine stamp, and isn't gated in CI.

    uv run python scripts/console_perf.py                 # Markdown to stdout
    uv run python scripts/console_perf.py --json out.json
    AGRIOS_E2E_CHANNEL=msedge uv run python scripts/console_perf.py   # use installed Edge
"""

import argparse
import json
import os
import socket
import sys
import tempfile
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

# A throwaway database, before agrios.db reads DATABASE_URL at import.
os.environ["DATABASE_URL"] = f"sqlite:///{Path(tempfile.mkdtemp(prefix='agrios-perf-')) / 'agrios.db'}"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
import uvicorn
from langchain_core.messages import AIMessage

from agrios import agent
from agrios.console import auth
from agrios.console.app import create_app
from agrios.console.tests.e2e.conftest import forecast
from agrios.console.tests.scripted import ScriptedModel, call
from agrios.db import init_db
from agrios.services import farm, weather
from vision.eval.metrics import LatencyReport, machine_stamp

PAGES = ["/", "/farm", "/farm/weather/field-1", "/assistant", "/settings/general", "/settings/access"]
USER, PASSWORD = "perf", "a long password for perf runs"


def serve() -> tuple[str, uvicorn.Server]:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    return f"http://127.0.0.1:{port}", server


@contextmanager
def signed_in_client(base: str) -> Iterator[httpx.Client]:
    with httpx.Client(base_url=base, follow_redirects=False) as client:
        page = client.get("/login").text
        token = page.split('name="csrf_token" value="')[1].split('"')[0]
        response = client.post("/login", data={"username": USER, "password": PASSWORD, "csrf_token": token})
        assert response.status_code == 303, response.status_code
        yield client


def server_side(base: str, n: int) -> dict[str, LatencyReport]:
    """Time to the full response, from a client on the same machine."""
    reports = {}
    with signed_in_client(base) as client:
        for path in PAGES:
            for _ in range(10):  # warm-up: templates compiled, caches filled
                client.get(path)
            samples = []
            for _ in range(n):
                start = time.perf_counter_ns()
                response = client.get(path)
                samples.append(time.perf_counter_ns() - start)
                assert response.status_code == 200, (path, response.status_code)
            reports[path] = LatencyReport.from_ns(samples)
    return reports


def first_answer(base: str, model: ScriptedModel, n: int) -> LatencyReport:
    """Send to first streamed event of the answer, with a model that answers
    at once: the console's share of the wait (checkpoint load, graph, stream)."""
    samples = []
    with signed_in_client(base) as client:
        token = client.get("/assistant").text.split('name="csrf_token" value="')[1].split('"')[0]
        for _ in range(n):
            model.replies = [call("get_pending_tasks"), AIMessage("Nothing pending today.")]
            start = time.perf_counter_ns()
            with client.stream(
                "POST", "/assistant/stream", data={"message": "pending?"}, headers={"x-csrf-token": token}
            ) as response:
                first_token = None
                for line in response.iter_lines():
                    if line == "event: token" and first_token is None:
                        first_token = time.perf_counter_ns() - start
            assert first_token is not None
            samples.append(first_token)
    return LatencyReport.from_ns(samples)


def browser_side(base: str, n: int) -> dict[str, LatencyReport]:
    """Navigation Timing in Chromium: request start to the load event."""
    from playwright.sync_api import sync_playwright

    reports = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=os.environ.get("AGRIOS_E2E_CHANNEL") or None, headless=True)
        page = browser.new_page()
        page.goto(base + "/login")
        page.fill("#username", USER)
        page.fill("#password", PASSWORD)
        page.click("button.primary")
        page.wait_for_url(base + "/")
        for path in [p for p in PAGES if not p.startswith("/farm/weather")]:
            samples = []
            for _ in range(n):
                page.goto(base + path, wait_until="load")
                ms = page.evaluate(
                    "() => { const t = performance.getEntriesByType('navigation')[0];"
                    " return t.loadEventEnd - t.startTime; }"
                )
                samples.append(int(ms * 1e6))
            reports[path] = LatencyReport.from_ns(samples)
        browser.close()
    return reports


def table(title: str, reports: dict[str, LatencyReport]) -> str:
    rows = [
        f"**{title}**",
        "",
        "| Page | n | p50 ms [95% CI] | p95 | p99 | max |",
        "|---|---|---|---|---|---|",
    ]
    for path, r in reports.items():
        lo, hi = r.p50_ci95_ms
        rows.append(
            f"| `{path}` | {r.n} | {r.p50_ms:.2f} [{lo:.2f}-{hi:.2f}] "
            f"| {r.p95_ms:.2f} | {r.p99_ms:.2f} | {r.max_ms:.2f} |"
        )
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--n", type=int, default=300, help="requests per page (server side)")
    parser.add_argument("--browser-n", type=int, default=30, help="page loads per page (browser)")
    parser.add_argument("--json", type=Path, help="also write the raw report here")
    args = parser.parse_args()

    init_db()
    auth.create_user(USER, PASSWORD, "owner", now=time.time())
    for field_id, text in (("field-1", "Inspect rows for blight"), ("field-2", "Plough before sowing")):
        farm.log_task(field_id, text)
    model = ScriptedModel()
    for name in ("model", "model_with_tools", "summarizer_model"):
        setattr(agent, name, model)
    agent._model_with.cache_clear()
    weather.forecast = forecast  # the console's cost, not Open-Meteo's

    base, server = serve()
    try:
        server_reports = server_side(base, args.n)
        stream_report = first_answer(base, model, max(20, args.n // 10))
        browser_reports = browser_side(base, args.browser_n)
    finally:
        server.should_exit = True

    stamp = machine_stamp()
    cpu, cpus, system, python = stamp["cpu"], stamp["logical_cpus"], stamp["os"], stamp["python"]
    print(f"Machine: {cpu}, {cpus} logical CPUs, {system}, Python {python}\n")
    print(table("Server: request to full response", server_reports) + "\n")
    print(table("Browser (Chromium): navigation start to load event", browser_reports) + "\n")
    lo, hi = stream_report.p50_ci95_ms
    print(
        f"**Assistant, scripted model:** send to first answer token, n={stream_report.n}: "
        f"p50 {stream_report.p50_ms:.1f} ms [{lo:.1f}-{hi:.1f}], p95 {stream_report.p95_ms:.1f}, "
        f"max {stream_report.max_ms:.1f}"
    )
    if args.json:
        raw: dict[str, Any] = {
            "machine": stamp,
            "server": {k: asdict(v) for k, v in server_reports.items()},
            "browser": {k: asdict(v) for k, v in browser_reports.items()},
            "assistant_first_token": asdict(stream_report),
        }
        args.json.write_text(json.dumps(raw, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
