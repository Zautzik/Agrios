"""The console web app. create_app builds it; python -m agrios.console runs it.

Every route is in a router: `public` (sign-in, first-run setup, health
check), or one behind the sign-in guard (web.signed_in), which also checks
CSRF on every state-changing request: `protected` here, and one per screen in
pages/. A test requests every route signed out and asserts that only the
public set answers.
"""

from __future__ import annotations

import gettext
import secrets
import time
import urllib.parse
from collections.abc import Awaitable, Callable
from datetime import date, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from jinja2 import Environment, FileSystemLoader
from pydantic import ValidationError
from sqlalchemy import make_url

from agrios.console import auth, forms
from agrios.console.pages import assistant as assistant_pages
from agrios.console.pages import farm as farm_pages
from agrios.console.security import CSRF_COOKIE, client_allowed, host_allowed, parse_client
from agrios.console.web import (
    SAFE_METHODS,
    SESSION_COOKIE,
    CsrfError,
    ForbiddenError,
    NotSignedInError,
    Owner,
    SetupRequiredError,
    SignedIn,
    anonymous_csrf,
    current_settings,
    now,
    redirect,
    render,
    signed_in,
)
from agrios.db import BASE_DIR, DATABASE_URL
from agrios.services import farm
from agrios.services import settings as settings_service

HERE = Path(__file__).resolve().parent
PUBLIC_PATHS = frozenset({"/login", "/setup", "/healthz"})  # plus /static, mounted separately

GROUP_LABELS = {
    "general": "General",
    "assistant": "Assistant",
    "farm": "Farm",
    "access": "Network & access",
    "data": "Data",
    "about": "About",
}

# Content Security Policy: scripts, styles and images only from this server,
# no inline script, no framing. htmx is configured (base.html) to need neither
# eval nor inline styles, so this holds without exceptions.
CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "object-src 'none'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'"
)
SECURITY_HEADERS = {
    "Content-Security-Policy": CSP,
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "same-origin",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}


def _jinja_env() -> Environment:
    env = Environment(
        loader=FileSystemLoader(HERE / "templates"),
        autoescape=True,
        trim_blocks=True,
        lstrip_blocks=True,
        extensions=["jinja2.ext.i18n"],
    )
    # Every UI string goes through gettext from the start; Spanish becomes a
    # translation catalogue later rather than a rewrite of the templates.
    env.install_gettext_translations(gettext.NullTranslations(), newstyle=True)  # type: ignore[attr-defined]
    env.filters["when"] = lambda at: datetime.fromtimestamp(at).strftime("%Y-%m-%d %H:%M")
    # "Thu 05 Nov" from "2026-11-05": short enough for a phone-width forecast table.
    env.filters["day"] = lambda iso: date.fromisoformat(iso).strftime("%a %d %b")
    return env


def safe_next(target: str | None) -> str:
    """Only same-site paths: a ?next= pointing elsewhere would make the sign-in
    page an open redirect."""
    if target and target.startswith("/") and not target.startswith("//") and "\\" not in target:
        return target
    return "/"


def _return_path(request: Request) -> str:
    """Where to come back to after signing in. A GET comes back to itself. A
    form post can't (most have no page at their address: /farm/tasks would be
    a 405), so it comes back to the page the form was on, from Referer when
    that's this site, else the home page."""
    if request.method in SAFE_METHODS:
        return request.url.path
    referer = urllib.parse.urlsplit(request.headers.get("referer", ""))
    if referer.netloc == request.url.netloc:
        return safe_next(referer.path + (f"?{referer.query}" if referer.query else ""))
    return "/"


def create_app(
    *,
    demo: bool = False,
    setup_token: str | None = None,
    demo_credentials: tuple[str, str] | None = None,
    clock: Callable[[], float] = time.time,
) -> FastAPI:
    """setup_token: the one-time code printed on the console machine that the
    first-run setup asks for, so the first person to find the page on the
    network can't make themselves owner. None disables setup."""
    # No interactive API docs: every route here is a page, and the docs would
    # be another unauthenticated surface.
    app = FastAPI(title="Agrios Console", docs_url=None, redoc_url=None, openapi_url=None)
    app.state.demo = demo
    app.state.setup_token = setup_token
    app.state.demo_credentials = demo_credentials
    app.state.clock = clock
    app.state.templates = Jinja2Templates(env=_jinja_env())

    @app.middleware("http")
    async def network_guard(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        # Applied before anything else, static files included: a refused
        # client learns nothing beyond how to change the setting.
        host = request.client.host if request.client else None
        access = current_settings(request).access
        if not host_allowed(request.headers.get("host"), access):
            return PlainTextResponse(
                "This console only answers to its own addresses. If you reached it by a network name, "
                "add that name under Settings > Network & access on the console machine.",
                status_code=421,
            )
        if not client_allowed(host, access):
            return PlainTextResponse(
                f"This console doesn't accept connections from {host}. To change that, open Settings > "
                "Network & access on the console machine itself.",
                status_code=403,
            )
        response = await call_next(request)
        for name, value in SECURITY_HEADERS.items():
            response.headers.setdefault(name, value)
        if request.url.path not in ("/healthz",) and not request.url.path.startswith("/static/"):
            response.headers.setdefault("Cache-Control", "no-store")  # pages can show farm data
        return response

    @app.exception_handler(SetupRequiredError)
    async def _setup_required(request: Request, _: Exception) -> Response:
        return redirect(request, "/setup")

    @app.exception_handler(NotSignedInError)
    async def _not_signed_in(request: Request, _: Exception) -> Response:
        response = redirect(request, "/login?next=" + urllib.parse.quote(_return_path(request)))
        response.delete_cookie(SESSION_COOKIE)
        return response

    @app.exception_handler(ForbiddenError)
    async def _forbidden(request: Request, _: Exception) -> Response:
        return render(
            request,
            "error.html",
            {"title": "Not allowed", "message": "Your account's role doesn't include this page."},
            status=403,
        )

    @app.exception_handler(CsrfError)
    async def _csrf(request: Request, _: Exception) -> Response:
        return PlainTextResponse(
            "This form expired or came from another site. Go back, reload the page and try again.",
            status_code=403,
        )

    public = APIRouter(dependencies=[Depends(anonymous_csrf)])
    protected = APIRouter(dependencies=[Depends(signed_in)])

    @public.get("/healthz")
    def healthz() -> JSONResponse:
        return JSONResponse({"status": "ok"})

    @public.get("/setup", response_class=HTMLResponse)
    def setup_page(request: Request) -> Response:
        if auth.has_users():
            return RedirectResponse("/login", status_code=303)
        return render(request, "setup.html", {"setup_enabled": app.state.setup_token is not None})

    @public.post("/setup", response_class=HTMLResponse)
    def setup_submit(
        request: Request,
        setup_code: Annotated[str, Form()],
        username: Annotated[str, Form()],
        password: Annotated[str, Form()],
        confirm: Annotated[str, Form()],
    ) -> Response:
        if auth.has_users():
            return RedirectResponse("/login", status_code=303)
        token: str | None = app.state.setup_token
        context: dict[str, Any] = {"setup_enabled": token is not None, "username": username}
        if token is None or not secrets.compare_digest(setup_code.strip().encode(), token.encode()):
            return render(
                request,
                "setup.html",
                {
                    **context,
                    "error": "That setup code isn't right. It's "
                    "printed in the window where the console was started.",
                },
                status=403,
            )
        if password != confirm:
            return render(request, "setup.html", {**context, "error": "The two passwords don't match."}, 422)
        try:
            user = auth.create_first_owner(username, password, now=now(request))
        except auth.InvalidCredentialsError as e:
            return render(request, "setup.html", {**context, "error": str(e)}, status=422)
        except auth.SetupClosedError:
            return RedirectResponse("/login", status_code=303)
        app.state.setup_token = None  # one use
        return _signed_in_response(request, user)

    @public.get("/login", response_class=HTMLResponse)
    def login_page(request: Request, next: str = "/") -> Response:
        if not auth.has_users():
            return RedirectResponse("/setup", status_code=303)
        return render(request, "login.html", {"next": safe_next(next), "demo_credentials": demo_credentials})

    @public.post("/login", response_class=HTMLResponse)
    def login_submit(
        request: Request,
        username: Annotated[str, Form()],
        password: Annotated[str, Form()],
        next: Annotated[str, Form()] = "/",
    ) -> Response:
        context = {"next": safe_next(next), "username": username, "demo_credentials": demo_credentials}
        # Normalised, so ::ffff:a.b.c.d and a.b.c.d share one backoff counter.
        address = parse_client(request.client.host if request.client else None)
        client = str(address) if address else "unknown"
        result = auth.authenticate(username, password, client=client, now=now(request))
        if isinstance(result, auth.LockedOut):
            wait = int(result.retry_after_s) + 1
            response = render(
                request,
                "login.html",
                {**context, "error": f"Too many failed attempts. Try again in {wait} seconds."},
                status=429,
            )
            response.headers["Retry-After"] = str(wait)
            return response
        if result is None:
            return render(request, "login.html", {**context, "error": "Wrong username or password."}, 401)
        return _signed_in_response(request, result, safe_next(next))

    def _signed_in_response(request: Request, user: auth.User, target: str = "/") -> Response:
        token, _session = auth.start_session(user, now=now(request))
        response = RedirectResponse(target, status_code=303)
        # No max-age: the cookie ends with the browser, and the server enforces
        # the idle and absolute timeouts regardless.
        response.set_cookie(
            SESSION_COOKIE, token, httponly=True, samesite="strict", secure=request.url.scheme == "https"
        )
        response.delete_cookie(CSRF_COOKIE)
        return response

    @protected.post("/logout")
    def logout(request: Request) -> Response:
        token = request.cookies.get(SESSION_COOKIE)
        if token:
            auth.end_session(token)
        response = RedirectResponse("/login", status_code=303)
        response.delete_cookie(SESSION_COOKIE)
        return response

    @protected.get("/", response_class=HTMLResponse)
    def today(request: Request, session: SignedIn) -> Response:
        return render(request, "today.html", {"fields": farm.list_fields(), "tasks": farm.pending_tasks()})

    @protected.get("/settings")
    def settings_home(session: Owner) -> Response:
        return RedirectResponse("/settings/general", status_code=303)

    @protected.get("/settings/data", response_class=HTMLResponse)
    def settings_data(request: Request, session: Owner) -> Response:
        database = make_url(DATABASE_URL).render_as_string(hide_password=True)
        return render(
            request,
            "settings_data.html",
            {"groups": GROUP_LABELS, "group": "data", "database": database, "base_dir": BASE_DIR},
        )

    @protected.get("/settings/about", response_class=HTMLResponse)
    def settings_about(request: Request, session: Owner) -> Response:
        return render(
            request, "settings_about.html", {"groups": GROUP_LABELS, "group": "about", "auth": auth}
        )

    def _group_page(
        request: Request,
        group: str,
        *,
        values: dict[str, str] | None = None,
        errors: dict[str, str] | None = None,
        saved: list[str] | None = None,
        status: int = 200,
    ) -> Response:
        model = settings_service.GROUPS[group]
        current = getattr(settings_service.load(), group)
        specs = forms.field_specs(model)
        shown = {s.name: forms.display(getattr(current, s.name)) for s in specs}
        restart = [s.title for s in specs if s.name in (saved or []) and s.applies == "restart"]
        return render(
            request,
            "settings_group.html",
            {
                "groups": GROUP_LABELS,
                "group": group,
                "specs": specs,
                "values": {**shown, **(values or {})},
                "errors": errors or {},
                "saved": saved,
                "restart": restart,
                "changes": settings_service.changes(group, limit=20),
                "change_value": forms.change_value,
            },
            status=status,
        )

    def _known_group(group: str) -> str:
        if group not in settings_service.GROUPS:
            raise HTTPException(status_code=404)
        return group

    @protected.get("/settings/{group}", response_class=HTMLResponse)
    def settings_group(request: Request, group: str, session: Owner) -> Response:
        return _group_page(request, _known_group(group))

    def _change(request: Request, group: str, apply: Callable[[], settings_service.Settings]) -> Any:
        """Applies a settings change. For the access group, sessions that
        expired under the timeouts in force go first, so a longer timeout
        can't bring one back (refreshes sweep only once a minute). The
        timeouts are read under the settings lock, in the same step as the
        change: a save that started earlier can't sweep with stale values."""
        with settings_service.exclusive():
            if group == "access":
                access = settings_service.load().access
                auth.sweep_expired(
                    now=now(request),
                    idle_s=access.session_idle_minutes * 60.0,
                    max_s=access.session_max_hours * 3600.0,
                )
            return getattr(apply(), group)

    @protected.post("/settings/{group}", response_class=HTMLResponse)
    async def settings_save(request: Request, group: str, session: Owner) -> Response:
        model = settings_service.GROUPS[_known_group(group)]
        form = {k: v for k, v in (await request.form()).items() if isinstance(v, str)}
        values = forms.parse_form(model, form)
        before = getattr(settings_service.load(), group)
        try:
            after = _change(
                request, group, lambda: settings_service.update(group, values, username=session.user.username)
            )
        except ValidationError as e:
            # Re-show what was typed, so a mistake in one field doesn't lose the others.
            typed = {name: form[name] for name in values}
            return _group_page(request, group, values=typed, errors=forms.errors_by_field(e), status=422)
        request.state.settings = None  # the header shows the farm name just saved
        saved = [n for n in model.model_fields if getattr(before, n) != getattr(after, n)]
        return _group_page(request, group, saved=saved)

    @protected.post("/settings/{group}/reset/{field}", response_class=HTMLResponse)
    def settings_reset(request: Request, group: str, field: str, session: Owner) -> Response:
        _known_group(group)
        try:
            _change(
                request, group, lambda: settings_service.reset(group, field, username=session.user.username)
            )
        except settings_service.UnknownSettingError:
            raise HTTPException(status_code=404) from None
        request.state.settings = None
        return _group_page(request, group, saved=[field])

    app.include_router(public)
    app.include_router(protected)
    app.include_router(farm_pages.router)
    app.include_router(assistant_pages.router)
    app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")
    return app
