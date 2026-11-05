"""python -m agrios.console [--demo] [--host H] [--port P]

The address and port come from Settings > Network & access; --host and --port
override them for one run, e.g. to recover from a setting that stops the
console starting.
"""

from __future__ import annotations

import argparse
import os
import secrets
import time
from pathlib import Path

# The demo's own database, under the gitignored data/ so it's never committed.
# It's recreated on every --demo start, so the demo always opens in the same state.
DEMO_DB = Path(__file__).resolve().parents[2] / "data" / "demo" / "agrios-demo.db"
DEMO_USER = ("demo", "agrios-demo-password")
DEMO_TASKS = (
    ("field-1", "Inspect potato rows for late blight after the rain"),
    ("field-1", "Fertilize with 40 kg/ha nitrogen"),
    ("field-2", "Plough before the wheat sowing window"),
)


def _use_demo_database() -> None:
    DEMO_DB.parent.mkdir(parents=True, exist_ok=True)
    # The demo's conversations are in checkpoints.db beside it
    # (db.default_checkpoint_path), so they reset with it.
    for name in (DEMO_DB.name, "checkpoints.db"):
        for suffix in ("", "-wal", "-shm"):
            DEMO_DB.with_name(name + suffix).unlink(missing_ok=True)
    # Must happen before agrios.db is imported: it builds its engine from
    # DATABASE_URL at import, the same way the CLI and the tests configure it.
    os.environ["DATABASE_URL"] = f"sqlite:///{DEMO_DB}"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m agrios.console", description="Run the Agrios console.")
    parser.add_argument("--demo", action="store_true", help="separate demo database with sample data")
    parser.add_argument("--host", help="override the listen address from Settings")
    parser.add_argument("--port", type=int, help="override the port from Settings")
    args = parser.parse_args(argv)
    if args.demo:
        _use_demo_database()

    import uvicorn

    from agrios.console import auth
    from agrios.console.app import create_app
    from agrios.console.security import bind_host
    from agrios.db import init_db
    from agrios.services import farm, settings

    init_db()
    access = settings.load().access
    host = args.host or bind_host(access)
    port = args.port or access.port
    url = f"http://{'localhost' if host == '127.0.0.1' else host}:{port}"

    if args.demo:
        auth.create_user(*DEMO_USER, "owner", now=time.time())
        for field_id, description in DEMO_TASKS:
            farm.log_task(field_id, description)
        app = create_app(demo=True, demo_credentials=DEMO_USER)
        print(f"Agrios console (demo data) at {url}  sign in as {DEMO_USER[0]} / {DEMO_USER[1]}")
    elif not auth.has_users():
        code = secrets.token_urlsafe(12)
        app = create_app(setup_token=code)
        print(f"Agrios console at {url}\nFirst run: open {url}/setup and enter this setup code: {code}")
    else:
        app = create_app()
        print(f"Agrios console at {url}")
    # proxy_headers off: there's no reverse proxy, so X-Forwarded-For must not
    # be able to change the client address the network guard checks.
    uvicorn.run(app, host=host, port=port, proxy_headers=False, log_level="info")


if __name__ == "__main__":
    main()
