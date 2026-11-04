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
from langfuse.langchain import CallbackHandler
from langgraph.errors import GraphRecursionError
from sqlalchemy.exc import SQLAlchemyError
import argparse
import json
import logging

# The farm logic lives in the shared service layer (agrios/), which the web
# console calls too; the tools below only translate its results and errors
# into what the model sees. Names re-exported here keep existing imports and
# tests (main.engine, main.init_db, main.FAO_CROP_CALENDAR, ...) working.
from agrios.db import (BASE_DIR, DATABASE_URL, ADDED_COLUMNS, SEED_FIELDS, engine, fields_table,  # noqa: F401
                       init_db, metadata, open_checkpointer, tasks_table)
from agrios.services import calendar, farm, weather
from agrios.services import settings as app_settings
from agrios.services.calendar import FAO_CROP_CALENDAR  # noqa: F401
from agrios.services.weather import (FROST_RISK_C, OPEN_METEO_FORECAST, OPEN_METEO_GEOCODE,  # noqa: F401
                                     RAIN_DAY_MM)
from spatial_tools import submit_spatial_intent, read_joint_states

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

# Only the most recent turns are sent to the model; the full history stays in
# the checkpoint. Without this, every turn replays the whole thread.
MAX_HISTORY_MESSAGES = 20

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
    # Read per call, so a threshold changed in the console applies to the next
    # question. If the settings can't be read, the defaults still give an answer.
    try:
        farm_settings = app_settings.load().farm
    except SQLAlchemyError:
        logger.exception("couldn't read settings; using the default thresholds")
        farm_settings = app_settings.FarmSettings()
    try:
        result = weather.forecast(location, days, frost_risk_c=farm_settings.frost_risk_c,
                                  rain_day_mm=farm_settings.rain_day_mm)
    except weather.LocationNotFoundError:
        return json.dumps({"error": f"Couldn't find a field or place called {location!r}."})
    except (httpx.HTTPError, KeyError, ValueError) as e:
        logger.exception("get_weather_forecast failed for %s", location)
        return json.dumps({"error": f"Weather service unavailable for {location!r}: {e}"})
    # json.dumps of the dump, not model_dump_json: the model has always seen
    # json.dumps's separators and ASCII escaping, and must keep seeing them.
    return json.dumps(result.model_dump())

@tool
def lookup_crop_calendar(crop: str, region: str) -> str:
    """Returns planting and harvest windows for a crop in a given region.
    Use when the user asks when to plant or harvest a specific crop."""
    try:
        result = calendar.crop_calendar(crop, region)
    except calendar.RegionNotCoveredError:
        return json.dumps({"error": f"No crop calendar for region {region!r}: the FAO calendar used here "
                                    "covers Chile only (northern, central or southern Chile)."})
    except calendar.CropNotInCalendarError as e:
        return json.dumps({"error": f"No FAO calendar entry for {crop!r} in {region!r}.",
                           "crops_available": e.crops_available})
    except (httpx.HTTPError, ValueError) as e:
        logger.exception("lookup_crop_calendar failed for %s/%s", crop, region)
        return json.dumps({"error": f"Crop calendar service unavailable: {e}"})
    return json.dumps(result.model_dump())

@tool
def query_field_status(field_id: str) -> str:
    """Returns the crop, status and region for a given field_id. field_id must match
the format "field-<n>", e.g. "field-1" or "field-2"."""
    try:
        field = farm.get_field(field_id)
    except SQLAlchemyError as e:
        logger.exception("query_field_status failed for %s", field_id)
        return json.dumps({"error": f"Error looking up field {field_id}: {e}"})
    if field is None:
        return json.dumps({"error": f"No field found with id {field_id}"})
    result = field.model_dump()
    if field.status == "fallow":
        # Without this the model often decides an empty field has nothing
        # to look up, and skips the calendar even when asked for it.
        result["note"] = ("Fallow: crop is the field's planned crop. For planting or harvest "
                          "timing, call lookup_crop_calendar with this crop and region.")
    return json.dumps(result)


@tool
def log_task(field_id: str, description: str) -> str:
    """Logs a pending task for a field. Use when the user wants to record
    a future action like irrigating, fertilizing, or inspecting for a given field_id. field_id must match
the format "field-<n>", e.g. "field-1" or "field-2"."""
    try:
        farm.log_task(field_id, description)
    except SQLAlchemyError as e:
        logger.exception("log_task failed for %s", field_id)
        return json.dumps({"error": f"Error logging task for {field_id}: {e}"})
    return json.dumps({"logged": True, "field_id": field_id, "description": description})

@tool
def get_pending_tasks() -> str:
    """Returns all tasks currently pending across all fields."""
    try:
        tasks = farm.pending_tasks()
    except SQLAlchemyError as e:
        logger.exception("get_pending_tasks failed")
        return json.dumps({"error": f"Error retrieving pending tasks: {e}"})
    return json.dumps([{"field_id": t.field_id, "description": t.description} for t in tasks])

# The model and Ollama address are console settings, read once at startup;
# a blank address leaves ChatOllama to use OLLAMA_HOST or its localhost default.
_assistant = app_settings.load().assistant
model = ChatOllama(model=_assistant.model, base_url=_assistant.ollama_url or None)
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
