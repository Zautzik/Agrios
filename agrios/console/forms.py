"""Settings forms built from the settings models, so the UI can't drift from
what the service validates: the input type, limits, default and help text all
come from the model's field definitions."""

from __future__ import annotations

import typing
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ValidationError
from pydantic.fields import FieldInfo

Widget = Literal["text", "number", "select", "lines"]


@dataclass(frozen=True)
class FieldSpec:
    name: str
    title: str
    description: str
    widget: Widget
    applies: str
    default: str  # as shown in the form
    options: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    step: str | None = None
    min_length: int | None = None
    max_length: int | None = None
    pattern: str | None = None


def _constraint(info: FieldInfo, name: str) -> Any:
    return next((getattr(m, name) for m in info.metadata if getattr(m, name, None) is not None), None)


def display(value: Any) -> str:
    """A value as it appears in a form input."""
    if isinstance(value, tuple | list):
        return "\n".join(str(v) for v in value)
    return str(value)


def field_specs(model: type[BaseModel]) -> list[FieldSpec]:
    specs = []
    for name, info in model.model_fields.items():
        annotation = info.annotation
        options: tuple[str, ...] = ()
        step = None
        if typing.get_origin(annotation) is Literal:
            widget: Widget = "select"
            options = tuple(str(o) for o in typing.get_args(annotation))
        elif typing.get_origin(annotation) in (tuple, list):
            widget = "lines"
        elif annotation in (int, float):
            widget, step = "number", ("1" if annotation is int else "any")
        elif annotation is str:
            widget = "text"
        else:
            raise TypeError(f"no form widget for {model.__name__}.{name}: {annotation}")
        extra = info.json_schema_extra if isinstance(info.json_schema_extra, dict) else {}
        specs.append(
            FieldSpec(
                name=name,
                title=info.title or name,
                description=info.description or "",
                widget=widget,
                applies=str(extra.get("applies", "now")),
                default=display(info.default),
                options=options,
                minimum=_constraint(info, "ge"),
                maximum=_constraint(info, "le"),
                step=step,
                min_length=_constraint(info, "min_length"),
                max_length=_constraint(info, "max_length"),
                pattern=_constraint(info, "pattern"),
            )
        )
    return specs


def parse_form(model: type[BaseModel], form: Mapping[str, str]) -> dict[str, Any]:
    """Form strings to values for the model. Only the model's own fields are
    read; anything else posted is ignored here and can't reach the service."""
    values: dict[str, Any] = {}
    for spec in field_specs(model):
        if spec.name not in form:
            continue
        raw = form[spec.name]
        if spec.widget == "lines":
            values[spec.name] = [line.strip() for line in raw.splitlines() if line.strip()]
        else:
            values[spec.name] = raw.strip()
    return values


def errors_by_field(error: ValidationError) -> dict[str, str]:
    """The first message per field, for showing beside its input."""
    messages: dict[str, str] = {}
    for err in error.errors():
        if err["loc"]:
            messages.setdefault(str(err["loc"][0]), err["msg"])
    return messages


def change_value(value: Any) -> str:
    """A logged value for the change history, where a blank must be visible."""
    shown = display(value)
    return shown if shown else "(blank)"
