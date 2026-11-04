"""Fields and tasks. Errors from the database (SQLAlchemyError) propagate:
the caller decides how a failure is shown."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict
from sqlalchemy import insert, select

from agrios.db import engine, fields_table, tasks_table


class Field(BaseModel):
    # Declaration order is output order: the agent's tool serialises this
    # model as-is, and what the model sees must not change.
    model_config = ConfigDict(frozen=True)

    field_id: str
    crop: str | None
    status: str | None
    region: str | None
    latitude: float | None
    longitude: float | None


class Task(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: int
    field_id: str
    description: str
    status: str


def get_field(field_id: str) -> Field | None:
    with engine.connect() as conn:
        row = (
            conn.execute(select(fields_table).where(fields_table.c.field_id == field_id))
            .mappings()
            .fetchone()
        )
    return None if row is None else Field.model_validate(dict(row))


def log_task(field_id: str, description: str) -> Task:
    with engine.begin() as conn:
        result = conn.execute(
            insert(tasks_table).values(field_id=field_id, description=description, status="pending")
        )
        key = result.inserted_primary_key
    if key is None or key[0] is None:
        raise RuntimeError("the database did not return an id for the new task")
    return Task(id=int(key[0]), field_id=field_id, description=description, status="pending")


def pending_tasks() -> list[Task]:
    with engine.connect() as conn:
        rows = (
            conn.execute(
                select(
                    tasks_table.c.id, tasks_table.c.field_id, tasks_table.c.description, tasks_table.c.status
                ).where(tasks_table.c.status == "pending")
            )
            .mappings()
            .fetchall()
        )
    return [Task.model_validate(dict(r)) for r in rows]


def list_fields() -> list[Field]:
    with engine.connect() as conn:
        rows = conn.execute(select(fields_table).order_by(fields_table.c.field_id)).mappings().fetchall()
    return [Field.model_validate(dict(r)) for r in rows]
