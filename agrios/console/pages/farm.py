"""The Farm screen: fields, tasks, and each field's 7-day weather.

Everyone signed in can see it; changing tasks needs the operator or owner
role. The functions called here are the ones the assistant's tools call, so a
task logged on this page and one logged in chat are the same record.
"""

from __future__ import annotations

from typing import Annotated

import httpx
from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response

from agrios.console.web import Operator, SignedIn, current_settings, is_htmx, now, render, signed_in
from agrios.services import farm, weather

MAX_TASK_LENGTH = 500
FORECAST_DAYS = 7

router = APIRouter(dependencies=[Depends(signed_in)])


def _page(request: Request, *, error: str | None = None, typed: dict[str, str] | None = None) -> Response:
    return render(
        request,
        "farm.html",
        {
            "fields": farm.list_fields(),
            "tasks": farm.pending_tasks(),
            "done": farm.recently_completed(),
            "error": error,
            "typed": typed or {},
            "max_task_length": MAX_TASK_LENGTH,
        },
        status=422 if error else 200,
    )


@router.get("/farm", response_class=HTMLResponse)
def farm_page(request: Request, session: SignedIn) -> Response:
    return _page(request)


@router.post("/farm/tasks", response_class=HTMLResponse)
def add_task(
    request: Request,
    session: Operator,
    field_id: Annotated[str, Form()] = "",
    description: Annotated[str, Form()] = "",
) -> Response:
    typed = {"field_id": field_id, "description": description}
    text = description.strip()
    if farm.get_field(field_id) is None:
        return _page(request, error="Choose one of the fields in the list.", typed=typed)
    if not text:
        return _page(request, error="Describe the task, for example “Irrigate the north rows”.", typed=typed)
    if len(text) > MAX_TASK_LENGTH:
        return _page(request, error=f"Keep the task under {MAX_TASK_LENGTH} characters.", typed=typed)
    farm.log_task(field_id, text, now=now(request))
    # Post/redirect/get: reloading the page afterwards doesn't log it twice.
    return RedirectResponse("/farm", status_code=303)


@router.post("/farm/tasks/{task_id}/done")
def complete_task(request: Request, task_id: int, session: Operator) -> Response:
    # Already done or missing both end on the farm page: the result the person
    # wanted (that task not pending) holds either way.
    farm.complete_task(task_id, now=now(request))
    return RedirectResponse("/farm", status_code=303)


@router.get("/farm/weather/{field_id}", response_class=HTMLResponse)
def field_weather(request: Request, field_id: str, session: SignedIn) -> Response:
    field = farm.get_field(field_id)
    if field is None:
        raise HTTPException(status_code=404)
    thresholds = current_settings(request).farm
    error = None
    result = None
    if field.latitude is None or field.longitude is None:
        # Checked here: without coordinates, forecast() would look the field's
        # id up as a place name, and could return some town's weather.
        error = "This field has no coordinates, so there's no forecast for it."
    else:
        try:
            result = weather.forecast(
                field_id,
                FORECAST_DAYS,
                frost_risk_c=thresholds.frost_risk_c,
                rain_day_mm=thresholds.rain_day_mm,
            )
        except (weather.LocationNotFoundError, httpx.HTTPError, KeyError, ValueError):
            error = "The weather service (open-meteo.com) didn't answer. Try again in a minute."
    context = {"field": field, "forecast": result, "error": error, "thresholds": thresholds}
    # Loaded into the Farm page by htmx; opened directly it's a page of its own.
    # A boosted link click is htmx too, but it replaces the whole page, so it
    # gets the whole page.
    fragment = is_htmx(request) and request.headers.get("hx-boosted") != "true"
    return render(request, "_weather.html" if fragment else "farm_weather.html", context)
