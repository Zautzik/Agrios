"""Settings the owner can change from the console: typed, validated, stored in
the database, and every change logged with who made it and when.

Each group is a Pydantic model, and the console builds its forms from these
models, so a setting added here appears in the UI with its validation, default
and help text, with no template changes. The safety invariants (fail-safe
timing, the minimum frames per alert) are deliberately not settings: they live
in code and the console shows them read-only.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Mapping
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    IPvAnyNetwork,
    StringConstraints,
    ValidationError,
)
from sqlalchemy import Engine, delete, insert, select

from agrios.db import engine, settings_changes_table, settings_table

logger = logging.getLogger(__name__)
# Saves are serialised so two first saves of a group can't both delete-then-insert
# its row (a primary-key error on Postgres). Per process, like the sign-in lock.
_update_lock = threading.Lock()

# When a change takes effect: "now" on the next request or tool call, or
# "restart" once the console (or the CLI) is started again. Shown beside each
# setting so nobody waits for a change that needs a restart.
Applies = Literal["now", "restart"]


def _setting(
    default: Any, *, title: str, description: str, applies: Applies = "now", **constraints: Any
) -> Any:
    return Field(
        default, title=title, description=description, json_schema_extra={"applies": applies}, **constraints
    )


class _Group(BaseModel):
    # extra="ignore": a setting removed in a later version is dropped from the
    # stored JSON on the next save instead of making the whole group unloadable.
    model_config = ConfigDict(frozen=True, extra="ignore")


class GeneralSettings(_Group):
    farm_name: str = _setting(
        "My farm",
        title="Farm name",
        description="Shown in the page header and the browser tab.",
        min_length=1,
        max_length=60,
    )


class AssistantSettings(_Group):
    model: str = _setting(
        "qwen2.5:7b",
        title="Model",
        description=(
            "The Ollama model the assistant runs; pull it first with 'ollama pull <name>'. "
            "The assistant's eval suite was measured on qwen2.5:7b, so other models are unmeasured."
        ),
        applies="restart",
        min_length=1,
        max_length=100,
        pattern=r"^[\w.:/-]+$",
    )
    ollama_url: str = _setting(
        "",
        title="Ollama address",
        description="Leave blank to use OLLAMA_HOST, or http://localhost:11434 if that isn't set.",
        applies="restart",
        max_length=200,
        pattern=r"^(https?://\S+)?$",
    )


class FarmSettings(_Group):
    frost_risk_c: float = _setting(
        2.0,
        title="Frost risk below (°C)",
        description="A forecast day whose minimum temperature is below this is flagged as frost risk.",
        ge=-10,
        le=10,
    )
    rain_day_mm: float = _setting(
        1.0,
        title="Rain day from (mm)",
        description="A forecast day with at least this much precipitation counts as a rain day.",
        ge=0.1,
        le=100,
    )


Exposure = Literal["local", "lan", "custom"]
# Normalised before the pattern is checked: StringConstraints' own strip and
# lowercase run after it, so "Farm-PC.Local " would otherwise be rejected.
Hostname = Annotated[
    str,
    BeforeValidator(lambda v: v.strip().lower() if isinstance(v, str) else v),
    StringConstraints(pattern=r"^[a-z0-9]([a-z0-9.-]{0,251}[a-z0-9])?$"),
]


class AccessSettings(_Group):
    exposure: Exposure = _setting(
        "local",
        title="Who can reach the console",
        description=(
            "local: this machine only. lan: devices on the private network (192.168.x.x, 10.x.x.x and "
            "similar). custom: only the networks listed below. Narrowing applies at once; opening up from "
            "local needs a restart, because the console then has to listen on the network. Without HTTPS, "
            "anyone on the network can read the traffic, so prefer local until a certificate is set up."
        ),
        applies="restart",
    )
    allowed_networks: tuple[IPvAnyNetwork, ...] = _setting(
        (),
        title="Allowed networks",
        description=(
            "Used when the setting above is custom. One per line, e.g. 192.168.1.0/24 or 100.64.0.0/10 "
            "for a Headscale VPN. This machine is always allowed, so you can't lock yourself out. "
            "The console listens on IPv4; IPv6 entries apply when it's started with --host ::."
        ),
    )
    hostnames: tuple[Hostname, ...] = _setting(
        (),
        title="Names for this machine",
        description=(
            "Network names other devices use to reach the console, e.g. farm-pc.local, one per line. "
            "Addresses typed as numbers and localhost always work. Other names are refused, which stops "
            "a web page you visit from reaching the console through your own browser (DNS rebinding)."
        ),
        max_length=20,
    )
    port: int = _setting(
        8000,
        title="Port",
        description="The port the console listens on.",
        applies="restart",
        ge=1024,
        le=65535,
    )
    session_idle_minutes: int = _setting(
        30,
        title="Sign out after inactivity (minutes)",
        description="A session unused for this long ends, and the next visit asks for the password.",
        ge=5,
        le=1440,
    )
    session_max_hours: int = _setting(
        12,
        title="Sign in again after (hours)",
        description="Even an active session ends this long after signing in.",
        ge=1,
        le=168,
    )


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True)

    general: GeneralSettings = GeneralSettings()
    assistant: AssistantSettings = AssistantSettings()
    farm: FarmSettings = FarmSettings()
    access: AccessSettings = AccessSettings()


GROUPS: dict[str, type[_Group]] = {
    "general": GeneralSettings,
    "assistant": AssistantSettings,
    "farm": FarmSettings,
    "access": AccessSettings,
}


class UnknownSettingError(KeyError):
    """A group or field that isn't a setting, e.g. a typo, or an attempt to set
    a safety invariant, which is deliberately not one."""


class SettingChange(BaseModel):
    model_config = ConfigDict(frozen=True)

    at: float
    username: str
    setting: str
    old_value: Any
    new_value: Any


def _load_group(cls: type[_Group], raw: str | None) -> _Group:
    if raw is None:
        return cls()
    try:
        stored = json.loads(raw)
    except json.JSONDecodeError:
        stored = None
    if not isinstance(stored, dict):
        # Unreadable rather than merely outdated: the whole group falls back to
        # its defaults, so every page (and the network guard) still loads and the
        # owner can save over it from the UI.
        logger.warning("ignoring unreadable stored settings for %s", cls.__name__)
        return cls()
    try:
        return cls.model_validate(stored)
    except ValidationError as e:
        # A stored value that no longer validates (a constraint tightened in a
        # later version) falls back to its default; the rest of the group stays.
        bad = {str(err["loc"][0]) for err in e.errors() if err["loc"]}
        logger.warning("ignoring invalid stored settings %s.%s", cls.__name__, sorted(bad))
        return cls.model_validate({k: v for k, v in stored.items() if k not in bad})


def load(db_engine: Engine = engine) -> Settings:
    with db_engine.connect() as conn:
        rows = dict(conn.execute(select(settings_table.c.group_name, settings_table.c.value)).tuples().all())
    return Settings.model_validate({name: _load_group(cls, rows.get(name)) for name, cls in GROUPS.items()})


def update(
    group: str,
    values: Mapping[str, Any],
    *,
    username: str,
    db_engine: Engine = engine,
    now: float | None = None,
) -> Settings:
    """Set some fields of one group. Raises UnknownSettingError for a name that
    isn't a setting, and pydantic.ValidationError (nothing saved) for an
    invalid value. Unchanged values are not logged."""
    cls = GROUPS.get(group)
    if cls is None:
        raise UnknownSettingError(group)
    unknown = set(values) - set(cls.model_fields)
    if unknown:
        raise UnknownSettingError(f"{group}.{sorted(unknown)[0]}")
    at = time.time() if now is None else now
    with _update_lock, db_engine.begin() as conn:
        raw = conn.execute(
            select(settings_table.c.value).where(settings_table.c.group_name == group)
        ).scalar()
        current = _load_group(cls, raw)
        new = cls.model_validate({**current.model_dump(), **values})
        old_json, new_json = current.model_dump(mode="json"), new.model_dump(mode="json")
        changed = [name for name in cls.model_fields if old_json[name] != new_json[name]]
        if changed:
            conn.execute(delete(settings_table).where(settings_table.c.group_name == group))
            conn.execute(insert(settings_table).values(group_name=group, value=json.dumps(new_json)))
            conn.execute(
                insert(settings_changes_table),
                [
                    {
                        "at": at,
                        "username": username,
                        "setting": f"{group}.{name}",
                        "old_value": json.dumps(old_json[name]),
                        "new_value": json.dumps(new_json[name]),
                    }
                    for name in changed
                ],
            )
    return load(db_engine)


def reset(group: str, field: str, *, username: str, db_engine: Engine = engine) -> Settings:
    cls = GROUPS.get(group)
    if cls is None or field not in cls.model_fields:
        raise UnknownSettingError(f"{group}.{field}")
    return update(group, {field: cls.model_fields[field].default}, username=username, db_engine=db_engine)


def changes(group: str | None = None, *, limit: int = 50, db_engine: Engine = engine) -> list[SettingChange]:
    query = select(settings_changes_table).order_by(settings_changes_table.c.id.desc()).limit(limit)
    if group is not None:
        query = query.where(settings_changes_table.c.setting.startswith(f"{group}.", autoescape=True))
    with db_engine.connect() as conn:
        rows = conn.execute(query).mappings().all()
    return [
        SettingChange(
            at=r["at"],
            username=r["username"],
            setting=r["setting"],
            old_value=json.loads(r["old_value"]),
            new_value=json.loads(r["new_value"]),
        )
        for r in rows
    ]
