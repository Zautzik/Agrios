"""The Assistant screen: the farm agent in the browser.

A reply streams as Server-Sent Events over the POST that sends the message
(FastAPI's EventSourceResponse), read by static/assistant.js. Without
JavaScript the same form posts to /assistant/send, which waits for the whole
reply and redirects back to the conversation. Both run the same generator.

Each tool call the agent makes is shown in plain words ("Checked the
forecast for field-1"), with whether it worked, so an answer can be traced
to what it was based on.

The console runs the agent with the farm tools only (agrios.agent.FARM_TOOLS):
the robot-arm tools stay in the CLI until the Hardware screen adds a
confirmation step.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from collections.abc import Callable, Generator, Iterator
from dataclasses import dataclass, field
from typing import Annotated, Any, cast

import httpx
from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.sse import EventSourceResponse, ServerSentEvent
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, ToolMessage
from langgraph.errors import GraphRecursionError
from ollama import ResponseError

from agrios.console import conversations
from agrios.console.web import Operator, current_settings, now, render, signed_in
from agrios.db import open_checkpointer

logger = logging.getLogger(__name__)

MAX_MESSAGE_LENGTH = 4000
# Between the parts of an answer that a tool call interrupted; assistant.js uses the same.
ANSWER_BREAK = "\n\n"
# The same cap as the CLI: a tool loop that hasn't answered by then won't.
RECURSION_LIMIT = 10

router = APIRouter(dependencies=[Depends(signed_in)])

# Threads with a reply in progress. A second message to the same thread while
# one is running would interleave two runs in one checkpoint. Per process,
# like the console's other locks (one uvicorn worker; see auth.py).
_busy: set[str] = set()
_busy_lock = threading.Lock()


def describe_call(name: str, args: dict[str, Any]) -> str:
    """A tool call in plain words. Arguments are shown as given, so a wrong
    one (the wrong field, a misspelt place) is visible."""
    match name:
        case "get_weather_forecast":
            return f"Checked the forecast for {args.get('location', '?')}"
        case "lookup_crop_calendar":
            return f"Looked up the crop calendar for {args.get('crop', '?')} in {args.get('region', '?')}"
        case "query_field_status":
            return f"Looked up {args.get('field_id', '?')}"
        case "log_task":
            return f"Logged a task for {args.get('field_id', '?')}: {args.get('description', '')}"
        case "get_pending_tasks":
            return "Listed the pending tasks"
        case _:
            return f"Used {name}"


def tool_error(message: ToolMessage) -> str | None:
    """The error a tool reported, if any: every tool returns {"error": ...} on failure."""
    try:
        result = json.loads(message.content) if isinstance(message.content, str) else None
    except ValueError:
        return None
    return str(result["error"]) if isinstance(result, dict) and "error" in result else None


@dataclass
class ToolStep:
    call_id: str
    label: str
    error: str | None = None
    finished: bool = False


@dataclass
class Turn:
    """One exchange as the screen shows it: what was asked, the tools used, the answer."""

    question: str
    steps: list[ToolStep] = field(default_factory=list)
    answer: str = ""


def turns_from(messages: list[BaseMessage]) -> list[Turn]:
    turns: list[Turn] = []
    steps: dict[str, ToolStep] = {}
    for m in messages:
        if isinstance(m, HumanMessage):
            turns.append(Turn(question=str(m.content)))
        elif not turns:
            continue  # the start of the thread was folded into the summary
        elif isinstance(m, AIMessage):
            for call in m.tool_calls:
                step = ToolStep(call_id=call["id"] or "", label=describe_call(call["name"], call["args"]))
                steps[step.call_id] = step
                turns[-1].steps.append(step)
            if isinstance(m.content, str) and m.content:
                # Text before a tool call is part of the answer too, joined the
                # way assistant.js shows it while streaming (a blank line per break).
                turn = turns[-1]
                turn.answer = f"{turn.answer}{ANSWER_BREAK}{m.content}" if turn.answer else m.content
        elif isinstance(m, ToolMessage) and m.tool_call_id in steps:
            steps[m.tool_call_id].error = tool_error(m)
            steps[m.tool_call_id].finished = True
    return turns


def _config(thread_id: str) -> dict[str, Any]:
    from agrios.agent import ALLOWED_TOOLS_KEY, FARM_TOOLS

    return {
        "configurable": {"thread_id": thread_id, ALLOWED_TOOLS_KEY: sorted(FARM_TOOLS)},
        "recursion_limit": RECURSION_LIMIT,
    }


def _graph() -> Any:
    # Imported on first use, not at startup: it builds the model client and is
    # only needed once someone opens the assistant.
    from agrios.agent import graph

    return graph


def _model_unreachable(request: Request) -> str:
    assistant = current_settings(request).assistant
    where = assistant.ollama_url or "this machine"
    return (
        f"Couldn't get an answer from the model {assistant.model} at {where}. Is Ollama running, "
        f"and has the model been downloaded (ollama pull {assistant.model})? Both can be changed "
        "in Settings > Assistant."
    )


# What went wrong, by kind: the streaming screen shows the message at once;
# without JavaScript the kind travels in the redirect (?error=) and the page
# looks the message up here, so no text from the URL is ever shown.
ERRORS = {
    "busy": "This conversation is still answering. Wait for it to finish, then try again.",
    "loop": "I got stuck going round in circles on that one. Try rephrasing it.",
}


def error_message(request: Request, kind: str) -> str | None:
    if kind == "model":
        return _model_unreachable(request)
    return ERRORS.get(kind)


def _error(kind: str, messages: dict[str, str | None]) -> ServerSentEvent:
    return ServerSentEvent(event="error", data={"kind": kind, "message": messages.get(kind)})


def run_turn(request: Request, thread_id: str, message: str) -> Iterator[ServerSentEvent]:
    """Runs one message through the agent, yielding what the screen shows:
    `token` (answer text as it's generated), `tool` (a call starting),
    `tool_done`, `error`, and finally `done`.

    It may run on another thread after the request is gone (see
    in_background), so everything it needs from the request is read here,
    before the first step."""
    clock = request.app.state.clock
    messages = {kind: error_message(request, kind) for kind in ("busy", "loop", "model")}
    return _run_turn(thread_id, message, clock, messages)


def _run_turn(
    thread_id: str, message: str, clock: Callable[[], float], messages: dict[str, str | None]
) -> Generator[ServerSentEvent, None, None]:
    with _busy_lock:
        if thread_id in _busy:
            yield _error("busy", messages)
            yield ServerSentEvent(event="done", data={})
            return
        _busy.add(thread_id)
    try:
        with open_checkpointer() as saver:
            app = _graph().compile(checkpointer=saver)
            stream = app.stream(
                {"messages": [HumanMessage(message)]},
                config=_config(thread_id),
                stream_mode=["messages", "updates"],
            )
            for mode, payload in stream:
                yield from _events(mode, payload)
    except GraphRecursionError:
        logger.error("assistant: recursion limit hit on %s", thread_id)
        yield _error("loop", messages)
    except (ResponseError, httpx.HTTPError, ConnectionError):
        logger.exception("assistant: model call failed on %s", thread_id)
        yield _error("model", messages)
    finally:
        if conversations.exists(thread_id):
            conversations.touch(thread_id, now=clock())
        else:
            # Deleted while answering: each checkpoint holds the whole state, so
            # the run's last write would have brought the history back, owned by
            # nobody. Delete refuses while a run is going; this closes the gap
            # between that check and the delete.
            with open_checkpointer() as saver:
                saver.delete_thread(thread_id)
        with _busy_lock:
            _busy.discard(thread_id)
    yield ServerSentEvent(event="done", data={})


_FINISHED = object()


def in_background[T](events: Iterator[T]) -> Generator[T, None, None]:
    """Runs events to the end on a thread of its own and passes them on.

    Streamed inline, a turn would stop wherever it was when the browser went
    away (a reload, a closed tab, a dropped connection): FastAPI closes the
    response's generator at its next yield, leaving a tool call without its
    result saved. Here the browser leaving only stops the passing on; the turn
    finishes and is saved, and the page shows it on the next load."""
    found: queue.Queue[object] = queue.Queue()

    def work() -> None:
        try:
            for event in events:
                found.put(event)
        except Exception:
            logger.exception("assistant: turn failed")
        finally:
            found.put(_FINISHED)

    threading.Thread(target=work, name="assistant-turn", daemon=True).start()
    while (item := found.get()) is not _FINISHED:
        yield cast(T, item)


def _events(mode: str, payload: Any) -> Iterator[ServerSentEvent]:
    if mode == "messages":
        chunk, meta = payload
        # Only the agent's own text is the answer; the summarizer's isn't.
        if (
            isinstance(chunk, AIMessageChunk)
            and meta.get("langgraph_node") == "agent"
            and isinstance(chunk.content, str)
            and chunk.content
        ):
            yield ServerSentEvent(event="token", data=chunk.content)
        return
    for node, update in (payload or {}).items():
        for m in (update or {}).get("messages", []):
            if node == "agent" and isinstance(m, AIMessage):
                for call in m.tool_calls:
                    label = describe_call(call["name"], call["args"])
                    yield ServerSentEvent(event="tool", data={"id": call["id"], "label": label})
            elif node == "tools" and isinstance(m, ToolMessage):
                yield ServerSentEvent(event="tool_done", data={"id": m.tool_call_id, "error": tool_error(m)})


class MessageRejectedError(ValueError):
    """A message the form can't send, with the reason to show."""


def _clean_message(message: str) -> str:
    text = message.strip()
    if not text:
        raise MessageRejectedError("Type a message first.")
    if len(text) > MAX_MESSAGE_LENGTH:
        raise MessageRejectedError(f"Keep messages under {MAX_MESSAGE_LENGTH} characters.")
    return text


@router.get("/assistant", response_class=HTMLResponse)
def assistant_home(request: Request, session: Operator, q: str = "") -> Response:
    # ?q= comes from the Today page's box: it only fills in the message, the
    # person still presses Send, so a link can't make the assistant act.
    return _page(request, session, None, typed=q[:MAX_MESSAGE_LENGTH])


@router.get("/assistant/{thread_id}", response_class=HTMLResponse)
def conversation_page(request: Request, thread_id: str, session: Operator, error: str = "") -> Response:
    conversation = conversations.get(thread_id, session.user.id)
    if conversation is None:
        raise HTTPException(status_code=404)
    return _page(request, session, conversation, error=error_message(request, error))


def _page(
    request: Request,
    session: Operator,
    conversation: conversations.Conversation | None,
    *,
    error: str | None = None,
    typed: str = "",
    status: int = 200,
) -> Response:
    turns: list[Turn] = []
    summarised = False
    if conversation is not None:
        with open_checkpointer() as saver:
            state = _graph().compile(checkpointer=saver).get_state(_config(conversation.thread_id))
        turns = turns_from(state.values.get("messages", []))
        summarised = bool(state.values.get("summary"))
    return render(
        request,
        "assistant.html",
        {
            "conversation": conversation,
            "conversations": conversations.list_for(session.user.id),
            "turns": turns,
            "summarised": summarised,
            "error": error,
            "typed": typed,
            "max_message_length": MAX_MESSAGE_LENGTH,
            "model_name": current_settings(request).assistant.model,
        },
        status=status,
    )


@router.post("/assistant/stream", response_class=EventSourceResponse)
def send_streaming(
    request: Request,
    session: Operator,
    message: Annotated[str, Form()] = "",
    thread_id: Annotated[str, Form()] = "",
) -> Iterator[ServerSentEvent]:
    # This body runs once the response has started, so problems are reported
    # as error events rather than as an HTTP status.
    try:
        text = _clean_message(message)
    except MessageRejectedError as e:
        yield ServerSentEvent(event="error", data={"kind": "message", "message": str(e)})
        yield ServerSentEvent(event="done", data={})
        return
    if thread_id and conversations.get(thread_id, session.user.id) is None:
        yield ServerSentEvent(
            event="error", data={"kind": "missing", "message": "That conversation doesn't exist."}
        )
        yield ServerSentEvent(event="done", data={})
        return
    thread = thread_id or conversations.create(session.user.id, text, now=now(request)).thread_id
    # First, so a new conversation's page address is known before the answer.
    yield ServerSentEvent(event="thread", data={"id": thread, "url": f"/assistant/{thread}"})
    yield from in_background(run_turn(request, thread, text))


@router.post("/assistant/send", response_class=HTMLResponse)
def send_waiting(
    request: Request,
    session: Operator,
    message: Annotated[str, Form()] = "",
    thread_id: Annotated[str, Form()] = "",
) -> Response:
    existing = conversations.get(thread_id, session.user.id) if thread_id else None
    if thread_id and existing is None:
        raise HTTPException(status_code=404)
    try:
        text = _clean_message(message)
    except MessageRejectedError as e:
        return _page(request, session, existing, error=str(e), typed=message, status=422)
    thread = (
        existing.thread_id
        if existing
        else conversations.create(session.user.id, text, now=now(request)).thread_id
    )
    # Run to the end (list, not any()), so the turn finishes and is saved.
    events = list(run_turn(request, thread, text))
    kinds = [e.data["kind"] for e in events if e.event == "error"]
    return RedirectResponse(f"/assistant/{thread}" + (f"?error={kinds[0]}" if kinds else ""), status_code=303)


@router.post("/assistant/{thread_id}/delete")
def delete_conversation(thread_id: str, session: Operator) -> Response:
    if conversations.get(thread_id, session.user.id) is None:
        raise HTTPException(status_code=404)
    if thread_id in _busy:
        # The running turn would write its checkpoint back after the delete.
        return RedirectResponse(f"/assistant/{thread_id}?error=busy", status_code=303)
    # The messages too, not just the entry in the list: deleting a
    # conversation should leave nothing of it behind.
    with open_checkpointer() as saver:
        saver.delete_thread(thread_id)
    conversations.remove(thread_id)
    return RedirectResponse("/assistant", status_code=303)
