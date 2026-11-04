"""Database engine, schema, seed data and migrations, plus the conversation
checkpointer that follows the same DATABASE_URL."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from sqlalchemy import (
    Column,
    Engine,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    Table,
    Text,
    create_engine,
    func,
    insert,
    inspect,
    make_url,
    select,
    text,
    update,
)
from sqlalchemy.pool import QueuePool

# .env is the one configuration source for every entry point (the CLI, the
# console), so it's read where the configuration is first needed.
load_dotenv()

# The project root, not this package's directory: agrios.db and checkpoints.db
# have always lived next to main.py, and existing installs keep finding them.
BASE_DIR = Path(__file__).resolve().parents[1]
# DATABASE_URL lets the same code run against Postgres (e.g.
# postgresql+psycopg://user:pass@localhost/agrios); SQLite is the default.
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{BASE_DIR}/agrios.db")
engine = create_engine(DATABASE_URL, poolclass=QueuePool, pool_size=5, max_overflow=10)


@contextmanager
def open_checkpointer(
    database_url: str = DATABASE_URL, sqlite_path: Path = BASE_DIR / "checkpoints.db"
) -> Iterator[BaseCheckpointSaver[str]]:
    """Conversation history follows DATABASE_URL: on Postgres it lives in the
    same database as the farm data; otherwise in its own SQLite file."""
    url = make_url(database_url)
    if url.get_backend_name() == "postgresql":
        from langgraph.checkpoint.postgres import PostgresSaver

        # PostgresSaver takes a plain libpq URL, without SQLAlchemy's "+driver".
        conn_string = url.set(drivername="postgresql").render_as_string(hide_password=False)
        with PostgresSaver.from_conn_string(conn_string) as saver:
            saver.setup()  # creates or migrates its tables; safe to repeat
            yield saver
    else:
        with SqliteSaver.from_conn_string(str(sqlite_path)) as saver:
            yield saver


# Schema as SQLAlchemy Core tables rather than raw DDL, so each dialect emits
# its own auto-increment (SQLite: INTEGER PRIMARY KEY, Postgres: SERIAL).
metadata = MetaData()
fields_table = Table(
    "fields",
    metadata,
    Column("field_id", Text, primary_key=True),
    Column("crop", Text),
    Column("status", Text),
    Column("region", Text),
    Column("latitude", Float),
    Column("longitude", Float),
)
tasks_table = Table(
    "tasks",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("field_id", Text),
    Column("description", Text),
    Column("status", Text, server_default="pending"),
)

# Console tables. Timestamps are Unix seconds as floats: dialect-neutral, and the
# comparisons they serve (idle timeout, lockout) are plain arithmetic.
users_table = Table(
    "users",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("username", Text, nullable=False, unique=True),
    Column("password_hash", Text, nullable=False),  # argon2id, PHC string format
    Column("role", Text, nullable=False),
    Column("created_at", Float, nullable=False),
)
# Sessions live server-side so signing out, or deleting a user, ends them at
# once; a signed cookie would stay valid until it expired. Only a hash of the
# token is stored, so a copy of the database can't be used to sign in.
sessions_table = Table(
    "sessions",
    metadata,
    Column("token_hash", Text, primary_key=True),
    Column("user_id", Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    Column("csrf_token", Text, nullable=False),
    Column("created_at", Float, nullable=False),
    Column("last_seen", Float, nullable=False),
)
login_attempts_table = Table(
    "login_attempts",
    metadata,
    Column("username", Text, primary_key=True),
    Column("client", Text, primary_key=True),
    Column("failures", Integer, nullable=False),
    Column("last_failure", Float, nullable=False),
)
# One JSON document per settings group; only values that differ from the
# defaults need to be present, so new settings appear with their defaults.
settings_table = Table(
    "settings",
    metadata,
    Column("group_name", Text, primary_key=True),
    Column("value", Text, nullable=False),
)
settings_changes_table = Table(
    "settings_changes",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("at", Float, nullable=False),
    Column("username", Text, nullable=False),
    Column("setting", Text, nullable=False),  # "group.field"
    Column("old_value", Text, nullable=False),  # JSON
    Column("new_value", Text, nullable=False),
)

SEED_FIELDS = [
    {"field_id": "field-1", "crop": "potato", "status": "growing", "region": "central Chile",
     "latitude": -34.17, "longitude": -70.74},
    {"field_id": "field-2", "crop": "wheat", "status": "fallow", "region": "central Chile",
     "latitude": -34.58, "longitude": -70.99},
]  # fmt: skip
# Columns added after the first release, with the SQL type to add them as.
ADDED_COLUMNS = {"region": "TEXT", "latitude": "FLOAT", "longitude": "FLOAT"}


def init_db(db_engine: Engine = engine) -> None:
    metadata.create_all(db_engine)
    with db_engine.begin() as conn:
        # create_all doesn't alter existing tables: older databases get each
        # missing column added here, and the seed rows backfilled.
        columns = {c["name"] for c in inspect(conn).get_columns("fields")}
        for name, sql_type in ADDED_COLUMNS.items():
            if name in columns:
                continue
            conn.execute(text(f"ALTER TABLE fields ADD COLUMN {name} {sql_type}"))
            column = fields_table.c[name]
            for row in SEED_FIELDS:
                conn.execute(
                    update(fields_table)
                    .where(fields_table.c.field_id == row["field_id"], column.is_(None))
                    .values({name: row[name]})
                )
        count = conn.execute(select(func.count()).select_from(fields_table)).scalar()
        if count == 0:
            conn.execute(insert(fields_table), SEED_FIELDS)
