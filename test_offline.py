"""Fast tests that need no live model: tool output shape, schema portability,
and history trimming. Run by a plain `pytest`."""
import json

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, trim_messages
from sqlalchemy import create_engine, text
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.schema import CreateTable

import main


def test_tools_return_json():
    field = json.loads(main.query_field_status.invoke({"field_id": "field-1"}))
    assert field == {"field_id": "field-1", "crop": "potato", "status": "growing", "region": "central Chile"}

    missing = json.loads(main.query_field_status.invoke({"field_id": "field-99"}))
    assert "error" in missing

    logged = json.loads(main.log_task.invoke({"field_id": "field-2", "description": 'irrigate "north" rows'}))
    assert logged["logged"] is True

    pending = json.loads(main.get_pending_tasks.invoke({}))
    assert {"field_id": "field-2", "description": 'irrigate "north" rows'} in pending

    # Quotes in arguments used to produce invalid JSON via % formatting.
    weather = json.loads(main.get_weather_forecast.invoke({"location": 'Bob\'s "field"'}))
    assert weather["location"] == 'Bob\'s "field"'
    crop = json.loads(main.lookup_crop_calendar.invoke({"crop": "potato", "region": "Chile"}))
    assert crop["crop"] == "potato"


def test_schema_compiles_for_postgres_and_sqlite():
    pg = str(CreateTable(main.tasks_table).compile(dialect=postgresql.dialect()))
    assert "SERIAL" in pg and "AUTOINCREMENT" not in pg
    lite = str(CreateTable(main.tasks_table).compile(dialect=sqlite.dialect()))
    assert "id INTEGER NOT NULL" in lite


def test_init_db_adds_region_to_a_pre_region_database(tmp_path):
    db = create_engine(f"sqlite:///{tmp_path / 'old.db'}")
    with db.begin() as conn:
        conn.execute(text("CREATE TABLE fields (field_id TEXT PRIMARY KEY, crop TEXT, status TEXT)"))
        conn.execute(text("INSERT INTO fields VALUES ('field-1', 'potato', 'growing'), ('field-7', 'corn', 'growing')"))

    main.init_db(db)
    main.init_db(db)  # second run must be a no-op

    with db.connect() as conn:
        rows = dict(conn.execute(text("SELECT field_id, region FROM fields")).fetchall())
    assert rows == {"field-1": "central Chile", "field-7": None}  # only seed rows are backfilled


def _trim(messages):
    return trim_messages(messages, token_counter=len, max_tokens=main.MAX_HISTORY_MESSAGES,
                         strategy="last", start_on="human")


def test_history_trimming_keeps_recent_turns_and_whole_tool_exchanges():
    history = []
    for i in range(30):
        history += [
            HumanMessage(f"q{i}"),
            AIMessage("", tool_calls=[{"name": "get_pending_tasks", "args": {}, "id": f"c{i}"}]),
            ToolMessage("[]", tool_call_id=f"c{i}"),
            AIMessage(f"a{i}"),
        ]
    trimmed = _trim(history)
    assert len(trimmed) <= main.MAX_HISTORY_MESSAGES
    assert isinstance(trimmed[0], HumanMessage)  # never starts mid tool exchange
    assert trimmed[-1].content == "a29"


def test_short_history_is_untouched():
    history = [HumanMessage("hi"), AIMessage("hello")]
    assert _trim(history) == history
