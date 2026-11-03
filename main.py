from langchain_ollama import ChatOllama
import httpx
from ollama import ResponseError
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START
from langgraph.graph import END
from langchain_core.messages import (ToolMessage, AIMessageChunk, HumanMessage, SystemMessage,
                                     RemoveMessage, trim_messages)
from typing import TypedDict, Annotated, NotRequired
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv
from langfuse.langchain import CallbackHandler
from langgraph.errors import GraphRecursionError
from sqlalchemy import create_engine, make_url, select, func, insert, update, inspect, text, MetaData, Table, Column, Float, Integer, Text
from sqlalchemy.pool import QueuePool
from sqlalchemy.exc import SQLAlchemyError
import argparse
from contextlib import contextmanager
import json
import logging
import os
from pathlib import Path

from spatial_tools import submit_spatial_intent, read_joint_states

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
# DATABASE_URL lets the same code run against Postgres (e.g.
# postgresql+psycopg://user:pass@localhost/agrios); SQLite is the default.
DATABASE_URL = os.environ.get("DATABASE_URL", f"sqlite:///{BASE_DIR}/agrios.db")
engine = create_engine(DATABASE_URL, poolclass=QueuePool, pool_size=5, max_overflow=10)

@contextmanager
def open_checkpointer(database_url: str = DATABASE_URL, sqlite_path: Path = BASE_DIR / "checkpoints.db"):
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
    "fields", metadata,
    Column("field_id", Text, primary_key=True),
    Column("crop", Text),
    Column("status", Text),
    Column("region", Text),
    Column("latitude", Float),
    Column("longitude", Float),
)
tasks_table = Table(
    "tasks", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("field_id", Text),
    Column("description", Text),
    Column("status", Text, server_default="pending"),
)

# Only the most recent turns are sent to the model; the full history stays in
# the checkpoint. Without this, every turn replays the whole thread.
MAX_HISTORY_MESSAGES = 20

SEED_FIELDS = [
    {"field_id": "field-1", "crop": "potato", "status": "growing", "region": "central Chile",
     "latitude": -34.17, "longitude": -70.74},
    {"field_id": "field-2", "crop": "wheat", "status": "fallow", "region": "central Chile",
     "latitude": -34.58, "longitude": -70.99},
]
# Columns added after the first release, with the SQL type to add them as.
ADDED_COLUMNS = {"region": "TEXT", "latitude": "FLOAT", "longitude": "FLOAT"}

def init_db(db_engine=engine):
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
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]
    summary: NotRequired[str]
    summary_tasks: NotRequired[list[str]]  # tasks folded into the summary so far

# Once a thread passes SUMMARIZE_AFTER messages, everything but the most recent
# KEEP_RECENT is folded into a running summary and removed from state, so old
# turns are condensed instead of silently dropped by the trim below.
SUMMARIZE_AFTER = MAX_HISTORY_MESSAGES
KEEP_RECENT = 10

def agent_node(state: AgentState) -> dict:
    logger.info("agent_node: %d messages in", len(state["messages"]))
    history = trim_messages(
        state["messages"],
        token_counter=len,
        max_tokens=MAX_HISTORY_MESSAGES,
        strategy="last",
        start_on="human",
    )
    if state.get("summary"):
        history = [SystemMessage(f"Summary of the earlier conversation: {state['summary']}")] + history
    response = model_with_tools.invoke(history)
    logger.info("agent_node: tool_calls=%s", bool(response.tool_calls))
    return {"messages": [response]}

def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tools"
    if len(state["messages"]) > SUMMARIZE_AFTER:
        return "summarize"
    return END

TASKS_PREFIX = "Tasks logged: "

def logged_tasks(messages: list) -> list[str]:
    """Tasks saved by log_task in these messages, read from its tool results
    rather than left to the model, which filed its own suggestions as tasks.
    Older histories hold the pre-JSON reply ("Logged task for field-1: ...")."""
    tasks = []
    for m in messages:
        if not (isinstance(m, ToolMessage) and m.name == "log_task"):
            continue
        try:
            result = json.loads(m.content)
        except ValueError:
            if m.content.startswith("Logged task for "):
                tasks.append(m.content.removeprefix("Logged task for "))
            continue
        if isinstance(result, dict) and result.get("logged"):
            tasks.append(f"{result['field_id']}: {result['description']}")
    return tasks

def summarize_node(state: AgentState) -> dict:
    messages = state["messages"]
    recent = trim_messages(messages, token_counter=len, max_tokens=KEEP_RECENT,
                           strategy="last", start_on="human")
    dropped = messages[:len(messages) - len(recent)]
    if not dropped:
        return {}
    transcript = "\n".join(f"{m.type}: {m.content}" for m in dropped if m.content)
    # The tasks line is maintained in code, so the model never sees or rewrites it.
    previous = "\n".join(line for line in state.get("summary", "").splitlines()
                         if not line.startswith(TASKS_PREFIX))
    rules = SystemMessage(
        "You write the running summary of a farm-operations chat, as notes for the assistant, "
        "not a reply to the user. Record only what was said: fields, places, crops, dates, "
        "forecasts, decisions, and open questions, as short bullet points. Leave out tasks "
        "logged; they are tracked separately. Do not give advice, next steps, greetings, or "
        "offers of help."
    )
    prompt = f"Current summary: {previous or '(none)'}\n\nNew messages:\n{transcript}"
    tasks = state.get("summary_tasks", []) + logged_tasks(dropped)
    summary = summarizer_model.invoke([rules, HumanMessage(prompt)]).content.strip()
    if tasks:
        summary += f"\n{TASKS_PREFIX}{'; '.join(tasks)}"
    logger.info("summarize_node: folded %d messages into the summary", len(dropped))
    return {"summary": summary, "summary_tasks": tasks,
            "messages": [RemoveMessage(id=m.id) for m in dropped]}


init_db()

# Tools whose arguments should come from query_field_status's result. When the
# model calls one in the same turn as query_field_status, it hasn't seen that
# result yet and is guessing, so the call is sent back instead of run.
FIELD_DEPENDENT_TOOLS = {"lookup_crop_calendar"}
# Arguments that must not be empty; the model sometimes sends "" for them.
REQUIRED_ARGS = {
    "lookup_crop_calendar": ("crop", "region"),
    "get_weather_forecast": ("location",),
    "query_field_status": ("field_id",),
    "log_task": ("field_id", "description"),
}

def rejected_call_reason(call: dict, names_this_turn: set[str]) -> str | None:
    """Why a tool call shouldn't run as given, or None if it should."""
    name = call["name"]
    if name not in tools_by_name:
        return f"Unknown tool {name!r}."
    if name in FIELD_DEPENDENT_TOOLS and "query_field_status" in names_this_turn:
        return (f"Not run: wait for query_field_status's result, then call {name} again "
                "with the crop and region it returned.")
    empty = [a for a in REQUIRED_ARGS.get(name, ()) if not str(call["args"].get(a, "")).strip()]
    if empty:
        return f"Not run: {', '.join(empty)} must not be empty."
    return None

def tool_node(state: AgentState):
    last_message = state["messages"][-1]
    names_this_turn = {c["name"] for c in last_message.tool_calls}
    results = []
    for call in last_message.tool_calls:
        # Every tool call still gets a ToolMessage, so the model sees why a
        # rejected call didn't run and can retry it on its next turn.
        reason = rejected_call_reason(call, names_this_turn)
        if reason:
            logger.warning("tool_node: rejected %s(%s): %s", call["name"], call["args"], reason)
            output = json.dumps({"error": reason})
        else:
            logger.info("tool_node: calling %s(%s)", call["name"], call["args"])
            output = tools_by_name[call["name"]].invoke(call["args"])
        results.append(
            ToolMessage(content=str(output), tool_call_id=call["id"], name=call["name"])
        )
    return {"messages": results}

@tool
def get_weather_forecast(location: str, days:int=7) -> str:
    """Returns the weather forecast (daily temperature, rain, and a
    frost_risk flag per day) for a field or place over the next N days.
    location can be a field_id such
    as "field-1" or a place name. Always call this tool for any question
    about weather, forecasts, rain, or frost risk; never answer weather
    from memory."""
    days = max(1, min(int(days), 16))  # Open-Meteo forecasts at most 16 days
    try:
        place = _resolve_location(location)
        if place is None:
            return json.dumps({"error": f"Couldn't find a field or place called {location!r}."})
        resp = httpx.get(OPEN_METEO_FORECAST, timeout=10, params={
            "latitude": place["latitude"], "longitude": place["longitude"],
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
            "forecast_days": days, "timezone": "auto",
        })
        resp.raise_for_status()
        daily = resp.json()["daily"]
        # The model misjudged thresholds on its own (it called 10 °C a frost
        # risk), so the forecast flags frost-risk days itself.
        forecast = [
            {"date": d, "temp_max_c": hi, "temp_min_c": lo, "rain_mm": rain,
             "frost_risk": lo is not None and lo <= FROST_RISK_C}
            for d, hi, lo, rain in zip(daily["time"], daily["temperature_2m_max"],
                                       daily["temperature_2m_min"], daily["precipitation_sum"])
        ]
        # Spelled out because the model misread the daily list once ("no rain"
        # with rain in the forecast); reading a ready answer is harder to get wrong.
        rain_days = [day["date"] for day in forecast if (day["rain_mm"] or 0) >= RAIN_DAY_MM]
        return json.dumps({"location": location, "resolved_to": place["name"],
                           "rain_days": rain_days,
                           "total_rain_mm": round(sum(day["rain_mm"] or 0 for day in forecast), 1),
                           "frost_risk_days": [day["date"] for day in forecast if day["frost_risk"]],
                           "forecast": forecast, "source": "open-meteo.com"})
    except (httpx.HTTPError, KeyError, ValueError) as e:
        logger.exception("get_weather_forecast failed for %s", location)
        return json.dumps({"error": f"Weather service unavailable for {location!r}: {e}"})

# Air-temperature minimum at or below which a day counts as a frost risk;
# ground frost can form a couple of degrees above 0 °C air temperature.
FROST_RISK_C = 2.0
# Daily total at or above which a day counts as a rain day (the usual
# meteorological "wet day" threshold), so drizzle-level amounts don't.
RAIN_DAY_MM = 1.0
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"

def _resolve_location(location: str) -> dict | None:
    """A field_id resolves to that field's stored coordinates; anything else
    is looked up as a place name."""
    with engine.connect() as conn:
        row = conn.execute(
            select(fields_table.c.field_id, fields_table.c.latitude, fields_table.c.longitude)
            .where(func.lower(fields_table.c.field_id) == location.strip().lower())
        ).fetchone()
    if row is not None and row.latitude is not None:
        return {"name": row.field_id, "latitude": row.latitude, "longitude": row.longitude}
    resp = httpx.get(OPEN_METEO_GEOCODE, params={"name": location, "count": 1}, timeout=10)
    resp.raise_for_status()
    results = resp.json().get("results") or []
    if not results:
        return None
    hit = results[0]
    name = ", ".join(p for p in (hit.get("name"), hit.get("admin1"), hit.get("country")) if p)
    return {"name": name, "latitude": hit["latitude"], "longitude": hit["longitude"]}

@tool
def lookup_crop_calendar(crop: str, region: str) -> str:
    """Returns planting and harvest windows for a crop in a given region.
    Use when the user asks when to plant or harvest a specific crop."""
    zones = _zones_for_region(region)
    if not zones:
        return json.dumps({"error": f"No crop calendar for region {region!r}: the FAO calendar used here "
                                    "covers Chile only (northern, central or southern Chile)."})
    try:
        resp = httpx.get(FAO_CROP_CALENDAR, params={"language": "en"}, timeout=15)
        resp.raise_for_status()
        entries = resp.json()
    except (httpx.HTTPError, ValueError) as e:
        logger.exception("lookup_crop_calendar failed for %s/%s", crop, region)
        return json.dumps({"error": f"Crop calendar service unavailable: {e}"})

    wanted = _normalize_crop(crop)
    matches = [e for e in entries
               if _normalize_crop(e["crop"]["name"]).startswith(wanted) and e["aez"]["name"] in zones]
    if not matches:
        known = sorted({e["crop"]["name"] for e in entries})
        return json.dumps({"error": f"No FAO calendar entry for {crop!r} in {region!r}.", "crops_available": known})
    windows = [
        {"zone": e["aez"]["name"], "sowing": _window(s["early_sowing"], s["later_sowing"]),
         "harvest": _window(s["early_harvest"], s["late_harvest"])}
        for e in matches for s in e["sessions"]
    ]
    return json.dumps({"crop": matches[0]["crop"]["name"], "region": region, "windows": windows,
                       "source": f"FAO Crop Calendar, Chile (updated {matches[0]['lastUpdated']})"})

FAO_CROP_CALENDAR = "https://api-cropcalendar.apps.fao.org/api/v1/countries/CL/cropCalendar"
# FAO splits Chile into agro-ecological zones; these map how people name the
# regions onto them. Central Chile's climate is Mediterranean.
_REGION_ZONES = [
    (("north", "norte", "atacama", "arid"), "Arid Zone"),
    (("central", "centro", "santiago", "maule", "higgins", "valparaíso", "valparaiso", "mediterranean"),
     "Mediterranean"),
    (("south", "sur", "araucan", "los lagos", "los ríos", "los rios", "biobío", "biobio", "temperate"),
     "Moist Temperate"),
]
_MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

def _zones_for_region(region: str) -> set[str]:
    text = region.lower()
    zones = {zone for keywords, zone in _REGION_ZONES if any(k in text for k in keywords)}
    if not zones and "chile" in text:
        zones = {zone for _, zone in _REGION_ZONES}  # "Chile" alone: every zone
    return zones

def _normalize_crop(name: str) -> str:
    name = name.strip().lower()
    for suffix in ("oes", "es", "s"):  # potatoes -> potat(o), beans -> bean
        if name.endswith(suffix) and len(name) > len(suffix) + 2:
            return name[: -len(suffix)]
    return name

def _window(start: dict, end: dict) -> str:
    return f"{int(start['day'])} {_MONTHS[start['month'] - 1]} – {int(end['day'])} {_MONTHS[end['month'] - 1]}"

@tool
def query_field_status(field_id: str) -> str:
    """Returns the crop, status and region for a given field_id. field_id must match
the format "field-<n>", e.g. "field-1" or "field-2"."""
    try:
        with engine.connect() as conn:
            row = conn.execute(
                select(fields_table).where(fields_table.c.field_id == field_id)
            ).mappings().fetchone()
        if row is None:
            return json.dumps({"error": f"No field found with id {field_id}"})
        result = dict(row)
        if result["status"] == "fallow":
            # Without this the model often decides an empty field has nothing
            # to look up, and skips the calendar even when asked for it.
            result["note"] = ("Fallow: crop is the field's planned crop. For planting or harvest "
                              "timing, call lookup_crop_calendar with this crop and region.")
        return json.dumps(result)
    except SQLAlchemyError as e:
        logger.exception("query_field_status failed for %s", field_id)
        return json.dumps({"error": f"Error looking up field {field_id}: {e}"})


@tool
def log_task(field_id: str, description: str) -> str:
    """Logs a pending task for a field. Use when the user wants to record
    a future action like irrigating, fertilizing, or inspecting for a given field_id. field_id must match
the format "field-<n>", e.g. "field-1" or "field-2"."""
    try:
        with engine.begin() as conn:
            conn.execute(
                insert(tasks_table).values(field_id=field_id, description=description, status="pending")
            )
        return json.dumps({"logged": True, "field_id": field_id, "description": description})
    except SQLAlchemyError as e:
        logger.exception("log_task failed for %s", field_id)
        return json.dumps({"error": f"Error logging task for {field_id}: {e}"})

@tool
def get_pending_tasks() -> str:
    """Returns all tasks currently pending across all fields."""
    try:
        with engine.connect() as conn:
            rows = conn.execute(
                select(tasks_table.c.field_id, tasks_table.c.description)
                .where(tasks_table.c.status == "pending")
            ).mappings().fetchall()
        return json.dumps([dict(r) for r in rows])
    except SQLAlchemyError as e:
        logger.exception("get_pending_tasks failed")
        return json.dumps({"error": f"Error retrieving pending tasks: {e}"})

model = ChatOllama(model="qwen2.5:7b")
langfuse_handler = CallbackHandler()
tools = [get_weather_forecast, lookup_crop_calendar, query_field_status, log_task, get_pending_tasks,
         submit_spatial_intent, read_joint_states]
model_with_tools = model.bind_tools(tools)
summarizer_model = model  # plain model, no tools: summaries are text
tools_by_name = {t.name: t for t in tools}
graph = StateGraph(AgentState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.add_node("summarize", summarize_node)

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "summarize": "summarize", END: END})
graph.add_edge("tools", "agent")
graph.add_edge("summarize", END)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agrios farm-ops agent CLI")
    parser.add_argument("--thread", default="1",
                        help="conversation thread id; each id keeps its own history (default: 1)")
    args = parser.parse_args()
    config = {"configurable": {"thread_id": args.thread}}

    with open_checkpointer() as memory:
        app = graph.compile(checkpointer=memory)
        while True:
            user_input = input("You: ")
            if user_input.lower() == "exit":
                break
            try:
                # stream_mode="messages" yields the model's tokens as they are
                # generated; only the agent node's text is user-facing.
                for chunk, meta in app.stream(
                    {"messages": [("user", user_input)]},
                    config={**config, "callbacks": [langfuse_handler], "recursion_limit": 10},
                    stream_mode="messages",
                ):
                    if (isinstance(chunk, AIMessageChunk) and meta.get("langgraph_node") == "agent"
                            and isinstance(chunk.content, str)):
                        print(chunk.content, end="", flush=True)
                print()
            except GraphRecursionError:
                logger.error("Hit recursion limit (10) for thread %s", config["configurable"]["thread_id"])
                print("Sorry, I got stuck looping on that one — try rephrasing your request.")
            except (ResponseError, ConnectionError, httpx.ConnectError) as e:
                logger.exception("Ollama call failed")
                print(f"Couldn't get a response from the model: {e}")
