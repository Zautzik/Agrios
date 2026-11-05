"""Agrios farm-ops agent CLI. The agent itself lives in agrios/agent.py, which
the web console runs too; this file is the terminal front end."""

import httpx
from ollama import ResponseError
from langchain_core.messages import AIMessageChunk
from langfuse.langchain import CallbackHandler
from langgraph.errors import GraphRecursionError
import argparse
import logging

# Re-exported so existing imports and tests (main.engine, main.init_db,
# main.FAO_CROP_CALENDAR, main.graph, ...) keep working after the move.
# Tests that patch the agent's model patch agrios.agent, where it's read.
from agrios.agent import (END, FARM_TOOLS, KEEP_RECENT, MAX_HISTORY_MESSAGES, SUMMARIZE_AFTER,  # noqa: F401
                          TASKS_PREFIX, AgentState, agent_node, get_pending_tasks, get_weather_forecast,
                          graph, log_task, logged_tasks, lookup_crop_calendar, model, model_with_tools,
                          query_field_status, rejected_call_reason, should_continue, summarize_node,
                          summarizer_model, tool_node, tools, tools_by_name)
from agrios.db import (BASE_DIR, DATABASE_URL, ADDED_COLUMNS, SEED_FIELDS, engine, fields_table,  # noqa: F401
                       init_db, metadata, open_checkpointer, tasks_table)
from agrios.services.calendar import FAO_CROP_CALENDAR  # noqa: F401
from agrios.services.weather import (FROST_RISK_C, OPEN_METEO_FORECAST, OPEN_METEO_GEOCODE,  # noqa: F401
                                     RAIN_DAY_MM)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agrios farm-ops agent CLI")
    parser.add_argument("--thread", default="1",
                        help="conversation thread id; each id keeps its own history (default: 1)")
    args = parser.parse_args()
    config = {"configurable": {"thread_id": args.thread}}
    langfuse_handler = CallbackHandler()

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
