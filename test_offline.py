"""Fast tests that need no live model: tool output shape, schema portability,
history trimming and checkpointing. Run by a plain `pytest`."""
import json
import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, trim_messages
from sqlalchemy import create_engine, text
from sqlalchemy.dialects import postgresql, sqlite
from sqlalchemy.schema import CreateTable

import httpx

import main
import migrate_checkpoints


class FakeResponse:
    def __init__(self, payload, status=200):
        self.payload, self.status = payload, status

    def json(self):
        return self.payload

    def raise_for_status(self):
        if self.status >= 400:
            raise httpx.HTTPStatusError("boom", request=None, response=None)


FORECAST = {"daily": {"time": ["2026-11-02", "2026-11-03"], "temperature_2m_max": [21.1, 18.2],
                      "temperature_2m_min": [6.0, -0.5], "precipitation_sum": [0.0, 4.2]}}


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    """Offline tests never reach Open-Meteo; a test that needs it fakes it."""
    def refuse(*args, **kwargs):
        raise AssertionError(f"unexpected network call: {args} {kwargs}")
    monkeypatch.setattr(main.httpx, "get", refuse)


# Trimmed from FAO's real Chile response (api-cropcalendar.apps.fao.org), same shape.
FAO_CL = [
    {"lastUpdated": "26/02/2025", "crop": {"name": "Potato"}, "aez": {"name": "Mediterranean"},
     "sessions": [{"early_sowing": {"day": 15, "month": 11}, "later_sowing": {"day": 15, "month": 2},
                   "early_harvest": {"day": 15, "month": 3}, "late_harvest": {"day": 15, "month": 6}}]},
    {"lastUpdated": "26/02/2025", "crop": {"name": "Potato"}, "aez": {"name": "Moist Temperate"},
     "sessions": [{"early_sowing": {"day": 15, "month": 9}, "later_sowing": {"day": 15, "month": 11},
                   "early_harvest": {"day": 15, "month": 3}, "late_harvest": {"day": 15, "month": 5}}]},
    {"lastUpdated": "26/02/2025", "crop": {"name": "Wheat, bread"}, "aez": {"name": "Mediterranean"},
     "sessions": [{"early_sowing": {"day": "1", "month": 5}, "later_sowing": {"day": "31", "month": 7},
                   "early_harvest": {"day": 1, "month": 10}, "late_harvest": {"day": 31, "month": 1}}]},
]


@pytest.fixture
def fake_open_meteo(monkeypatch):
    calls = []
    def get(url, params=None, timeout=None):
        calls.append((url, params))
        if url == main.FAO_CROP_CALENDAR:
            return FakeResponse(FAO_CL)
        if url == main.OPEN_METEO_GEOCODE:
            hits = [] if params["name"] == "Atlantis" else [
                {"name": "Talca", "admin1": "Maule", "country": "Chile", "latitude": -35.43, "longitude": -71.67}]
            return FakeResponse({"results": hits} if hits else {})
        return FakeResponse(FORECAST)
    monkeypatch.setattr(main.httpx, "get", get)
    return calls


def test_tools_return_json(fake_open_meteo):
    field = json.loads(main.query_field_status.invoke({"field_id": "field-1"}))
    assert field == {"field_id": "field-1", "crop": "potato", "status": "growing", "region": "central Chile",
                     "latitude": -34.17, "longitude": -70.74}

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
    assert crop["crop"] == "Potato"


def test_weather_for_a_field_uses_its_stored_coordinates(fake_open_meteo):
    weather = json.loads(main.get_weather_forecast.invoke({"location": "Field-1", "days": 2}))
    assert weather["resolved_to"] == "field-1"
    assert weather["forecast"][1] == {"date": "2026-11-03", "temp_max_c": 18.2, "temp_min_c": -0.5, "rain_mm": 4.2,
                                      "frost_risk": True}
    assert weather["forecast"][0]["frost_risk"] is False  # 6.0 °C minimum
    assert weather["rain_days"] == ["2026-11-03"] and weather["total_rain_mm"] == 4.2
    assert weather["frost_risk_days"] == ["2026-11-03"]
    (url, params), = fake_open_meteo  # no geocoding call for a known field
    assert url == main.OPEN_METEO_FORECAST
    assert (params["latitude"], params["longitude"], params["forecast_days"]) == (-34.17, -70.74, 2)


def test_weather_for_a_place_name_geocodes_it(fake_open_meteo):
    weather = json.loads(main.get_weather_forecast.invoke({"location": "Talca", "days": 99}))
    assert weather["resolved_to"] == "Talca, Maule, Chile"
    assert fake_open_meteo[1][1]["forecast_days"] == 16  # capped to what Open-Meteo serves


def test_weather_reports_errors_instead_of_inventing_data(fake_open_meteo, monkeypatch):
    unknown = json.loads(main.get_weather_forecast.invoke({"location": "Atlantis"}))
    assert "Couldn't find" in unknown["error"]

    monkeypatch.setattr(main.httpx, "get", lambda *a, **k: FakeResponse({}, status=503))
    down = json.loads(main.get_weather_forecast.invoke({"location": "field-1"}))
    assert "Weather service unavailable" in down["error"]


def test_crop_calendar_uses_fao_windows_for_the_zone(fake_open_meteo):
    cal = json.loads(main.lookup_crop_calendar.invoke({"crop": "potatoes", "region": "central Chile"}))
    assert cal["crop"] == "Potato" and cal["source"].startswith("FAO Crop Calendar")
    assert cal["windows"] == [{"zone": "Mediterranean", "sowing": "15 Nov – 15 Feb", "harvest": "15 Mar – 15 Jun"}]

    wheat = json.loads(main.lookup_crop_calendar.invoke({"crop": "wheat", "region": "Chile"}))
    assert wheat["windows"] == [{"zone": "Mediterranean", "sowing": "1 May – 31 Jul", "harvest": "1 Oct – 31 Jan"}]

    south = json.loads(main.lookup_crop_calendar.invoke({"crop": "potato", "region": "southern Chile"}))
    assert [w["zone"] for w in south["windows"]] == ["Moist Temperate"]


def test_crop_calendar_refuses_regions_and_crops_it_has_no_data_for(fake_open_meteo):
    invented = json.loads(main.lookup_crop_calendar.invoke({"crop": "potato", "region": "Midwest USA"}))
    assert "Chile only" in invented["error"]
    assert not fake_open_meteo  # refused before calling FAO

    unknown = json.loads(main.lookup_crop_calendar.invoke({"crop": "kiwi", "region": "central Chile"}))
    assert "No FAO calendar entry" in unknown["error"] and "Potato" in unknown["crops_available"]


def test_fallow_field_points_at_the_crop_calendar():
    field = json.loads(main.query_field_status.invoke({"field_id": "field-2"}))
    assert field["status"] == "fallow" and "lookup_crop_calendar" in field["note"]


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
        rows = {r[0]: tuple(r[1:]) for r in conn.execute(text("SELECT field_id, region, latitude, longitude FROM fields"))}
    # only seed rows are backfilled
    assert rows == {"field-1": ("central Chile", -34.17, -70.74), "field-7": (None, None, None)}


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


class FakeModel:
    def __init__(self, reply):
        self.reply, self.seen = reply, []

    def invoke(self, messages):
        self.seen.append(messages)
        return AIMessage(self.reply)


def _long_thread(turns):
    messages = []
    for i in range(turns):
        messages += [
            HumanMessage(f"q{i}", id=f"h{i}"),
            AIMessage("", tool_calls=[{"name": "get_pending_tasks", "args": {}, "id": f"c{i}"}], id=f"a{i}"),
            ToolMessage("[]", tool_call_id=f"c{i}", id=f"t{i}"),
            AIMessage(f"a{i}", id=f"r{i}"),
        ]
    return messages


def test_long_thread_routes_to_summarize_and_short_one_ends():
    assert main.should_continue({"messages": _long_thread(6)}) == "summarize"  # 24 messages
    assert main.should_continue({"messages": _long_thread(2)}) == main.END


class FakeNotesModel(FakeModel):
    def invoke(self, messages):
        self.seen.append(messages)
        return self.reply


def test_summarize_folds_old_turns_and_removes_them(monkeypatch):
    notes = main.ConversationNotes(fields_and_places=["field-2"], tasks_logged=["irrigate field-2"],
                                   open_questions=["q3 unanswered"])
    fake = FakeNotesModel(notes)
    monkeypatch.setattr(main, "summarizer_model", fake)
    out = main.summarize_node({"messages": _long_thread(6), "summary": "Earlier: hello."})

    # Rendered from the structure: empty sections are left out, nothing else is added.
    assert out["summary"] == ("Fields and places: field-2\nTasks logged: irrigate field-2\n"
                              "Open questions: q3 unanswered")
    removed = [m.id for m in out["messages"]]
    assert removed == [f"{p}{i}" for i in range(4) for p in "hatr"]  # 4 whole turns, oldest first
    rules, prompt = fake.seen[0][0], fake.seen[0][1].content
    assert rules.type == "system"
    assert "Earlier: hello." in prompt and "human: q0" in prompt and "q4" not in prompt


def test_agent_sees_the_summary(monkeypatch):
    fake = FakeModel("ok")
    monkeypatch.setattr(main, "model_with_tools", fake)
    main.agent_node({"messages": [HumanMessage("what did we do?")], "summary": "Logged irrigation for field-2."})
    first = fake.seen[0][0]
    assert first.type == "system" and "Logged irrigation for field-2." in first.content


def _seed_threads(saver, n=2):
    app = main.graph.compile(checkpointer=saver)
    threads = []
    for i in range(n):
        thread = {"configurable": {"thread_id": f"mig-{uuid.uuid4()}"}}
        app.update_state(thread, {"messages": [HumanMessage(f"hello {i}"), AIMessage(f"hi {i}")]}, as_node="agent")
        app.update_state(thread, {"messages": [HumanMessage(f"again {i}"), AIMessage(f"yes {i}")]}, as_node="agent")
        threads.append(thread)
    return threads


def _history(saver, thread):
    return [m.content for m in main.graph.compile(checkpointer=saver).get_state(thread).values["messages"]]


def test_copy_checkpoints_preserves_every_thread(tmp_path):
    with main.open_checkpointer(f"sqlite:///{tmp_path / 'x.db'}", tmp_path / "src.db") as src, \
         main.open_checkpointer(f"sqlite:///{tmp_path / 'x.db'}", tmp_path / "dst.db") as dst:
        threads = _seed_threads(src)
        copied = migrate_checkpoints.copy_checkpoints(src, dst)
        assert copied == len(list(src.list(None)))
        for thread in threads:
            assert _history(dst, thread) == _history(src, thread)
        migrate_checkpoints.copy_checkpoints(src, dst)  # re-running must not duplicate
        assert len(list(dst.list(None))) == copied


@pytest.mark.skipif(not main.DATABASE_URL.startswith("postgresql"), reason="DATABASE_URL is not Postgres")
def test_copy_checkpoints_from_sqlite_into_postgres(tmp_path):
    with main.open_checkpointer(f"sqlite:///{tmp_path / 'x.db'}", tmp_path / "src.db") as src, \
         main.open_checkpointer() as dst:
        threads = _seed_threads(src)
        migrate_checkpoints.copy_checkpoints(src, dst)
        assert _history(dst, threads[0]) == ["hello 0", "hi 0", "again 0", "yes 0"]
        for thread in threads:
            assert _history(dst, thread) == _history(src, thread)


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


def test_guard_lets_valid_calls_through(fake_open_meteo):
    (calendar,) = _run_tool_turn(("lookup_crop_calendar", {"crop": "potato", "region": "central Chile"}))
    assert calendar["crop"] == "Potato"


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
