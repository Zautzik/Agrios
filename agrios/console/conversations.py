"""Whose conversation is whose. The messages live in the agent's checkpointer
under thread_id; this records the owner and a title, so the Assistant screen
can list a person's conversations and refuse anyone else's."""

from __future__ import annotations

import secrets
from dataclasses import dataclass

from sqlalchemy import delete, insert, select, update

from agrios.db import assistant_threads_table, engine

TITLE_LENGTH = 60
# The CLI's threads are named by hand ("1", "farm"); the console's get a
# prefix, so the two can share a checkpointer without colliding.
THREAD_PREFIX = "console-"


@dataclass(frozen=True)
class Conversation:
    thread_id: str
    title: str
    created_at: float
    updated_at: float


def title_for(first_message: str) -> str:
    text = " ".join(first_message.split())
    return text if len(text) <= TITLE_LENGTH else text[: TITLE_LENGTH - 1].rstrip() + "…"


def create(user_id: int, first_message: str, *, now: float) -> Conversation:
    conversation = Conversation(
        thread_id=THREAD_PREFIX + secrets.token_urlsafe(12),
        title=title_for(first_message),
        created_at=now,
        updated_at=now,
    )
    with engine.begin() as conn:
        conn.execute(
            insert(assistant_threads_table).values(
                thread_id=conversation.thread_id,
                user_id=user_id,
                title=conversation.title,
                created_at=now,
                updated_at=now,
            )
        )
    return conversation


def get(thread_id: str, user_id: int) -> Conversation | None:
    """The conversation if it exists and belongs to user_id. Someone else's
    and a missing one look the same, so ids can't be probed."""
    with engine.connect() as conn:
        row = conn.execute(
            select(assistant_threads_table).where(
                assistant_threads_table.c.thread_id == thread_id, assistant_threads_table.c.user_id == user_id
            )
        ).fetchone()
    if row is None:
        return None
    return Conversation(row.thread_id, row.title, row.created_at, row.updated_at)


def exists(thread_id: str) -> bool:
    with engine.connect() as conn:
        row = conn.execute(
            select(assistant_threads_table.c.thread_id).where(
                assistant_threads_table.c.thread_id == thread_id
            )
        ).fetchone()
    return row is not None


def list_for(user_id: int, limit: int = 50) -> list[Conversation]:
    with engine.connect() as conn:
        rows = conn.execute(
            select(assistant_threads_table)
            .where(assistant_threads_table.c.user_id == user_id)
            .order_by(assistant_threads_table.c.updated_at.desc())
            .limit(limit)
        ).fetchall()
    return [Conversation(r.thread_id, r.title, r.created_at, r.updated_at) for r in rows]


def touch(thread_id: str, *, now: float) -> None:
    with engine.begin() as conn:
        conn.execute(
            update(assistant_threads_table)
            .where(assistant_threads_table.c.thread_id == thread_id)
            .values(updated_at=now)
        )


def remove(thread_id: str) -> None:
    with engine.begin() as conn:
        conn.execute(delete(assistant_threads_table).where(assistant_threads_table.c.thread_id == thread_id))
