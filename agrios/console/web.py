"""What every page module shares: the sign-in and role guards, CSRF checks,
and rendering. app.py builds the app from these; the pages/ modules define
their routes with them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any

from fastapi import Depends, Request
from fastapi.responses import RedirectResponse, Response
from fastapi.templating import Jinja2Templates

from agrios.console import auth
from agrios.console.security import (
    CSRF_COOKIE,
    CSRF_FIELD,
    CSRF_HEADER,
    cross_site,
    csrf_valid,
    new_csrf_token,
)
from agrios.services import settings as settings_service

SESSION_COOKIE = "agrios_session"
SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})


class NotSignedInError(Exception):
    pass


class SetupRequiredError(Exception):
    pass


class ForbiddenError(Exception):
    pass


class CsrfError(Exception):
    pass


def now(request: Request) -> float:
    clock: Callable[[], float] = request.app.state.clock
    return clock()


def current_settings(request: Request) -> settings_service.Settings:
    cached = getattr(request.state, "settings", None)
    if cached is None:
        cached = settings_service.load()
        request.state.settings = cached
    return cached


def current_session(request: Request) -> auth.Session:
    if not auth.has_users():
        raise SetupRequiredError
    token = request.cookies.get(SESSION_COOKIE)
    access = current_settings(request).access
    session = (
        auth.resolve_session(
            token,
            now=now(request),
            idle_s=access.session_idle_minutes * 60.0,
            max_s=access.session_max_hours * 3600.0,
        )
        if token
        else None
    )
    if session is None:
        raise NotSignedInError
    request.state.session = session
    return session


async def check_csrf(request: Request, expected: str | None) -> None:
    if cross_site(request):
        raise CsrfError
    submitted = request.headers.get(CSRF_HEADER)
    if submitted is None:
        value = (await request.form()).get(CSRF_FIELD)
        submitted = value if isinstance(value, str) else None
    if not csrf_valid(submitted, expected):
        raise CsrfError


async def signed_in(
    request: Request, session: Annotated[auth.Session, Depends(current_session)]
) -> auth.Session:
    """The protected routers' guard: a live session, and for anything that
    changes state, that session's CSRF token."""
    if request.method not in SAFE_METHODS:
        await check_csrf(request, session.csrf_token)
    return session


async def anonymous_csrf(request: Request) -> None:
    """CSRF for the forms used before signing in (double-submit cookie)."""
    if request.method not in SAFE_METHODS:
        await check_csrf(request, request.cookies.get(CSRF_COOKIE))


def require_owner(session: Annotated[auth.Session, Depends(signed_in)]) -> auth.Session:
    if session.user.role != "owner":
        raise ForbiddenError
    return session


def require_operator(session: Annotated[auth.Session, Depends(signed_in)]) -> auth.Session:
    """Owner or operator: anyone allowed to change farm records."""
    if session.user.role not in ("owner", "operator"):
        raise ForbiddenError
    return session


SignedIn = Annotated[auth.Session, Depends(signed_in)]
Owner = Annotated[auth.Session, Depends(require_owner)]
Operator = Annotated[auth.Session, Depends(require_operator)]


def is_htmx(request: Request) -> bool:
    return request.headers.get("hx-request") == "true"


def redirect(request: Request, url: str) -> Response:
    # htmx follows HX-Redirect with a full page load, which a plain 303 inside
    # an htmx request wouldn't do for a sign-in page.
    if is_htmx(request):
        return Response(status_code=200, headers={"HX-Redirect": url})
    return RedirectResponse(url, status_code=303)


def render(request: Request, name: str, context: dict[str, Any] | None = None, status: int = 200) -> Response:
    templates: Jinja2Templates = request.app.state.templates
    session: auth.Session | None = getattr(request.state, "session", None)
    csrf = session.csrf_token if session else request.cookies.get(CSRF_COOKIE) or new_csrf_token()
    base = {
        "demo": request.app.state.demo,
        "session": session,
        "csrf_token": csrf,
        "farm_name": current_settings(request).general.farm_name,
        "path": request.url.path,
    }
    response = templates.TemplateResponse(request, name, {**base, **(context or {})}, status_code=status)
    if session is None and request.cookies.get(CSRF_COOKIE) != csrf:
        response.set_cookie(
            CSRF_COOKIE, csrf, httponly=True, samesite="strict", secure=request.url.scheme == "https"
        )
    return response
