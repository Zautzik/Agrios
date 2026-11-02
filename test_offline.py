"""Fast tests that need no live model: tool output shape, schema portability,
history trimming and checkpointing. Run by a plain `pytest`."""
import json
import uuid

import pytest
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


def _round_trip_history(checkpointer):
    thread = {"configurable": {"thread_id": f"test-{uuid.uuid4()}"}}
    app = main.graph.compile(checkpointer=checkpointer)
    # update_state writes a checkpoint without running any node, so no model is needed.
    app.update_state(thread, {"messages": [HumanMessage("remember me"), AIMessage("noted")]}, as_node="agent")
    return [m.content for m in app.get_state(thread).values["messages"]]


def test_sqlite_checkpointer_keeps_history(tmp_path):
    url = f"sqlite:///{tmp_path / 'agrios.db'}"
    with main.open_checkpointer(url, tmp_path / "checkpoints.db") as saver:
        assert type(saver).__name__ == "SqliteSaver"
        assert _round_trip_history(saver) == ["remember me", "noted"]


@pytest.mark.skipif(not main.DATABASE_URL.startswith("postgresql"), reason="DATABASE_URL is not Postgres")
def test_postgres_checkpointer_keeps_history_across_connections():
    thread = {"configurable": {"thread_id": f"test-{uuid.uuid4()}"}}
    with main.open_checkpointer() as saver:
        assert type(saver).__name__ == "PostgresSaver"
        main.graph.compile(checkpointer=saver).update_state(
            thread, {"messages": [HumanMessage("remember me"), AIMessage("noted")]}, as_node="agent")
    # A fresh connection, as after a CLI restart, must see the same thread.
    with main.open_checkpointer() as saver:
        messages = main.graph.compile(checkpointer=saver).get_state(thread).values["messages"]
    assert [m.content for m in messages] == ["remember me", "noted"]


def _run_tool_turn(*calls):
    msg = AIMessage("", tool_calls=[{"name": n, "args": a, "id": f"c{i}"} for i, (n, a) in enumerate(calls)])
    out = main.tool_node({"messages": [HumanMessage("q"), msg]})["messages"]
    assert [m.tool_call_id for m in out] == [f"c{i}" for i in range(len(calls))]  # one reply per call
    return [json.loads(m.content) for m in out]


def test_guard_defers_calendar_called_alongside_field_lookup():
    field, calendar = _run_tool_turn(
        ("query_field_status", {"field_id": "field-1"}),
        ("lookup_crop_calendar", {"crop": "wheat", "region": "Midwest USA"}),
    )
    assert field["crop"] == "potato"
    assert "query_field_status" in calendar["error"]


def test_guard_rejects_empty_and_unknown_calls():
    empty, unknown = _run_tool_turn(
        ("lookup_crop_calendar", {"crop": "", "region": " "}),
        ("delete_everything", {}),
    )
    assert "crop, region must not be empty" in empty["error"]
    assert "Unknown tool" in unknown["error"]


def test_guard_lets_valid_calls_through():
    (calendar,) = _run_tool_turn(("lookup_crop_calendar", {"crop": "potato", "region": "central Chile"}))
    assert calendar["crop"] == "potato"


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
