"""The Assistant screen, run against a scripted chat model instead of Ollama:
streamed replies, tool calls shown in words, the farm-tools-only rule,
conversations kept per account, and what happens when the model is down."""

import json
import time
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from agrios import agent
from agrios.console import auth, conversations
from agrios.console.pages import assistant
from agrios.console.tests.conftest import PASSWORD, T0, page_csrf, sign_in
from agrios.console.tests.scripted import ScriptedModel, call
from agrios.db import open_checkpointer
from agrios.services import farm


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> Iterator[ScriptedModel]:
    scripted = ScriptedModel()
    monkeypatch.setattr(agent, "model", scripted)
    monkeypatch.setattr(agent, "model_with_tools", scripted)
    monkeypatch.setattr(agent, "summarizer_model", scripted)
    agent._model_with.cache_clear()  # it caches per tool set, over whichever model was there
    yield scripted
    agent._model_with.cache_clear()


def events(body: str) -> list[tuple[str, Any]]:
    """The (event, data) pairs of a text/event-stream body."""
    found = []
    for block in body.split("\n\n"):
        lines = [line for line in block.splitlines() if not line.startswith(":")]
        name = next((line[7:] for line in lines if line.startswith("event: ")), "message")
        data = [line[6:] for line in lines if line.startswith("data: ")]
        if data:
            found.append((name, json.loads("\n".join(data))))
    return found


def stream(
    client: TestClient, message: str, thread_id: str = "", csrf: str | None = None
) -> list[tuple[str, Any]]:
    token = csrf if csrf is not None else page_csrf(client, "/assistant")
    response = client.post(
        "/assistant/stream",
        data={"message": message, "thread_id": thread_id},
        headers={"x-csrf-token": token},
    )
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    return events(response.text)


def history(thread_id: str) -> list[BaseMessage]:
    with open_checkpointer() as saver:
        state = agent.graph.compile(checkpointer=saver).get_state({"configurable": {"thread_id": thread_id}})
    return list(state.values.get("messages", []))


# --- streaming -------------------------------------------------------------


def test_a_reply_streams_with_each_tool_call_in_plain_words(signed_in, model):
    farm.log_task("field-1", "Spray")
    model.replies = [call("get_pending_tasks"), AIMessage("You have one task: spray field-1.")]
    got = stream(signed_in, "What's pending?")

    kinds = [name for name, _ in got]
    assert kinds[0] == "thread"
    assert kinds[-1] == "done"
    assert ("tool", {"id": "c1", "label": "Listed the pending tasks"}) in got
    assert ("tool_done", {"id": "c1", "error": None}) in got
    tokens = [data for name, data in got if name == "token"]
    assert len(tokens) > 1  # arrived in pieces, not as one block at the end
    assert "".join(tokens) == "You have one task: spray field-1."
    assert kinds.index("tool") < kinds.index("tool_done") < kinds.index("token")

    thread = got[0][1]["id"]
    assert got[0][1]["url"] == f"/assistant/{thread}"
    assert [m.type for m in history(thread)] == ["human", "ai", "tool", "ai"]


def test_the_conversation_page_shows_question_steps_and_answer(signed_in, model):
    model.replies = [call("query_field_status", field_id="field-2"), AIMessage("field-2 is fallow.")]
    thread = stream(signed_in, "How is field-2?")[0][1]["id"]
    page = signed_in.get(f"/assistant/{thread}").text
    assert "How is field-2?" in page
    assert "Looked up field-2" in page
    assert "field-2 is fallow." in page
    assert page.count('class="step ok"') == 1
    assert "How is field-2?" in signed_in.get("/assistant").text  # listed by its first message


def test_a_failed_tool_is_shown_as_failed(signed_in, model):
    model.replies = [call("query_field_status", field_id="field-99"), AIMessage("I couldn't find field-99.")]
    got = stream(signed_in, "How is field-99?")
    [done] = [data for name, data in got if name == "tool_done"]
    assert done["error"] == "No field found with id field-99"
    page = signed_in.get(f"/assistant/{got[0][1]['id']}").text
    assert 'class="step failed"' in page
    assert "No field found with id field-99" in page


def test_the_second_message_continues_the_same_conversation(signed_in, owner, model):
    model.replies = [AIMessage("Hello."), AIMessage("Still here.")]
    thread = stream(signed_in, "Hi")[0][1]["id"]
    assert stream(signed_in, "Are you there?", thread)[0][1]["id"] == thread
    assert [m.content for m in history(thread)] == ["Hi", "Hello.", "Are you there?", "Still here."]
    assert [c.thread_id for c in conversations.list_for(owner.id)] == [thread]


# --- the farm-tools-only rule ------------------------------------------------


def test_the_console_gives_the_model_only_the_farm_tools(signed_in, model):
    model.replies = [AIMessage("ok")]
    stream(signed_in, "hello")
    assert [set(names) for names in model.bound] == [agent.FARM_TOOLS]


def test_a_robot_call_from_the_console_is_refused_not_run(signed_in, model, monkeypatch):
    ran: list[Any] = []
    monkeypatch.setitem(agent.tools_by_name, "submit_spatial_intent", _Recorder(ran))
    model.replies = [
        call("submit_spatial_intent", tvec=[0.1, 0.2, 0.3], rvec=[0, 0, 0]),
        AIMessage("I can't move the arm from here."),
    ]
    got = stream(signed_in, "Move the arm")
    [done] = [data for name, data in got if name == "tool_done"]
    assert done["error"] == "Not run: submit_spatial_intent isn't available here."
    assert ran == []


def test_the_cli_still_has_every_tool():
    # Negative control for the two tests above: without the console's
    # restriction the same guard lets a robot call through.
    assert agent.rejected_call_reason({"name": "submit_spatial_intent", "args": {}}, set()) is None
    assert agent.rejected_call_reason({"name": "submit_spatial_intent", "args": {}}, set(), agent.FARM_TOOLS)


class _Recorder:
    name = "submit_spatial_intent"

    def __init__(self, ran: list[Any]) -> None:
        self.ran = ran

    def invoke(self, args: Any) -> str:
        self.ran.append(args)
        return "moved"


# --- without JavaScript ------------------------------------------------------


def test_without_javascript_the_form_waits_then_shows_the_conversation(signed_in, model):
    model.replies = [AIMessage("Frost is unlikely.")]
    csrf = page_csrf(signed_in, "/assistant")
    response = signed_in.post("/assistant/send", data={"message": "Frost?", "csrf_token": csrf})
    assert response.status_code == 303
    location = response.headers["location"]
    assert location.startswith("/assistant/console-")
    assert "error" not in location
    assert "Frost is unlikely." in signed_in.get(location).text


def test_an_empty_message_is_refused_without_starting_a_conversation(signed_in, owner, model):
    csrf = page_csrf(signed_in, "/assistant")
    response = signed_in.post("/assistant/send", data={"message": "   ", "csrf_token": csrf})
    assert response.status_code == 422
    assert "Type a message first." in response.text
    assert [name for name, _ in stream(signed_in, "  ")] == ["error", "done"]
    assert conversations.list_for(owner.id) == []
    assert model.seen == []


def test_the_today_box_only_fills_in_the_message(signed_in, model):
    page = signed_in.get("/assistant", params={"q": "Frost on field-1?"}).text
    assert ">Frost on field-1?</textarea>" in page
    assert model.seen == []  # nothing was sent


# --- the model being unavailable ---------------------------------------------


def test_ollama_down_is_explained_with_what_to_do(signed_in, model):
    model.replies = [httpx.ConnectError("connection refused")]
    got = stream(signed_in, "Hi")
    [error] = [data for name, data in got if name == "error"]
    assert error["kind"] == "model"
    assert "Is Ollama running" in error["message"]
    assert "ollama pull qwen2.5:7b" in error["message"]
    assert got[-1][0] == "done"


def test_ollama_down_without_javascript(signed_in, model):
    model.replies = [httpx.ConnectError("connection refused")]
    csrf = page_csrf(signed_in, "/assistant")
    location = signed_in.post("/assistant/send", data={"message": "Hi", "csrf_token": csrf}).headers[
        "location"
    ]
    assert location.endswith("?error=model")
    assert "Is Ollama running" in signed_in.get(location).text


def test_an_error_kind_in_the_url_only_selects_a_known_message(signed_in, model):
    model.replies = [AIMessage("ok")]
    thread = stream(signed_in, "Hi")[0][1]["id"]
    page = signed_in.get(f"/assistant/{thread}", params={"error": "<b>injected</b>"}).text
    assert "injected" not in page


def test_a_second_message_while_one_is_answering_is_refused(signed_in, model):
    model.replies = [AIMessage("ok")]
    thread = stream(signed_in, "Hi")[0][1]["id"]
    assistant._busy.add(thread)
    try:
        got = stream(signed_in, "Again", thread)
    finally:
        assistant._busy.discard(thread)
    assert [d["kind"] for n, d in got if n == "error"] == ["busy"]
    assert len(history(thread)) == 2  # the refused message never reached the agent


# --- whose conversation ----------------------------------------------------------


def test_one_account_cannot_reach_anothers_conversation(app, model, clock):
    from agrios.console.tests.conftest import make_client

    auth.create_user("owner", PASSWORD, "owner", now=clock())
    auth.create_user("other", PASSWORD, "operator", now=clock())
    with make_client(app) as a, make_client(app) as b:
        sign_in(a)
        sign_in(b, username="other")
        model.replies = [AIMessage("private answer")]
        thread = stream(a, "my question")[0][1]["id"]

        assert b.get(f"/assistant/{thread}").status_code == 404
        assert "my question" not in b.get("/assistant").text
        got = stream(b, "let me in", thread)
        assert [d["kind"] for n, d in got if n == "error"] == ["missing"]
        csrf = page_csrf(b, "/assistant")
        assert (
            b.post(
                "/assistant/send", data={"message": "x", "thread_id": thread, "csrf_token": csrf}
            ).status_code
            == 404
        )
        assert b.post(f"/assistant/{thread}/delete", data={"csrf_token": csrf}).status_code == 404
    assert [m.content for m in history(thread)] == ["my question", "private answer"]


def test_deleting_a_conversation_removes_its_messages(signed_in, model):
    model.replies = [AIMessage("noted")]
    thread = stream(signed_in, "remember this")[0][1]["id"]
    csrf = page_csrf(signed_in, "/assistant")
    assert signed_in.post(f"/assistant/{thread}/delete", data={"csrf_token": csrf}).status_code == 303
    assert history(thread) == []
    assert signed_in.get(f"/assistant/{thread}").status_code == 404


def test_streaming_needs_the_csrf_token(signed_in, model):
    response = signed_in.post("/assistant/stream", data={"message": "Hi"}, headers={"x-csrf-token": "wrong"})
    assert response.status_code == 403
    assert model.seen == []


def test_a_viewer_has_no_assistant(client, clock, model):
    auth.create_user("viewer", PASSWORD, "viewer", now=clock())
    sign_in(client, username="viewer")
    assert 'href="/assistant"' not in client.get("/").text
    assert client.get("/assistant").status_code == 403
    csrf = page_csrf(client, "/")
    response = client.post("/assistant/stream", data={"message": "Hi"}, headers={"x-csrf-token": csrf})
    assert response.status_code == 403
    assert model.seen == []


def test_what_the_model_writes_is_shown_as_text(signed_in, model):
    model.replies = [AIMessage('<script>alert("x")</script> <img src=x onerror=alert(1)>')]
    thread = stream(signed_in, "<b>bold?</b>")[0][1]["id"]
    page = signed_in.get(f"/assistant/{thread}").text
    assert "<script>alert" not in page
    assert "<img src=x" not in page
    assert "<b>bold?</b>" not in page
    assert "&lt;script&gt;" in page
    assert "&lt;b&gt;bold?&lt;/b&gt;" in page


# --- reading a history back --------------------------------------------------------


def test_turns_skip_what_the_summary_replaced_and_pair_results_with_calls():
    messages: list[BaseMessage] = [
        AIMessage("an answer whose question was summarised away"),
        HumanMessage("q"),
        AIMessage("", tool_calls=[{"name": "get_pending_tasks", "args": {}, "id": "a"}]),
        ToolMessage(json.dumps({"error": "db down"}), tool_call_id="a", name="get_pending_tasks"),
        AIMessage("the answer"),
    ]
    [turn] = assistant.turns_from(messages)
    assert turn.question == "q"
    assert turn.answer == "the answer"
    [step] = turn.steps
    assert (step.label, step.error, step.finished) == ("Listed the pending tasks", "db down", True)


# --- found in review ----------------------------------------------------------------


def _wait_until_idle(thread_id: str, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while thread_id in assistant._busy and time.monotonic() < deadline:
        time.sleep(0.02)
    assert thread_id not in assistant._busy, "the turn never finished"


def test_a_turn_finishes_and_is_saved_after_the_browser_leaves(owner, model):
    model.replies = [call("get_pending_tasks"), AIMessage("Nothing pending.")]
    thread = conversations.create(owner.id, "pending?", now=T0).thread_id
    events = assistant.in_background(assistant._run_turn(thread, "pending?", lambda: T0, {}))
    next(events)  # the first event reaches the browser...
    events.close()  # ...which then reloads, closes the tab or loses the network
    _wait_until_idle(thread)
    assert [m.type for m in history(thread)] == ["human", "ai", "tool", "ai"]
    assert history(thread)[-1].content == "Nothing pending."


def _counting(seen: list[int]) -> Generator[int, None, None]:
    for i in range(3):
        seen.append(i)
        yield i


@pytest.mark.parametrize(("background", "expected"), [(True, [0, 1, 2]), (False, [0])])
def test_only_in_background_does_a_closed_stream_keep_going(background, expected):
    # The control for the test above, on a plain generator: consumed inline,
    # closing after the first event stops it there. (Done with a real turn,
    # closing inline abandons LangGraph mid-write while the checkpointer
    # closes, a native-level race; the app no longer does that, so the test
    # doesn't either.)
    seen: list[int] = []
    events = assistant.in_background(_counting(seen)) if background else _counting(seen)
    next(events)
    events.close()
    deadline = time.monotonic() + 5
    while len(seen) < len(expected) and time.monotonic() < deadline:
        time.sleep(0.01)
    time.sleep(0.05)
    assert seen == expected


def test_delete_waits_while_a_conversation_is_answering(signed_in, model):
    model.replies = [AIMessage("ok")]
    thread = stream(signed_in, "Hi")[0][1]["id"]
    csrf = page_csrf(signed_in, "/assistant")
    assistant._busy.add(thread)
    try:
        response = signed_in.post(f"/assistant/{thread}/delete", data={"csrf_token": csrf})
    finally:
        assistant._busy.discard(thread)
    assert response.status_code == 303
    assert response.headers["location"] == f"/assistant/{thread}?error=busy"
    assert "still answering" in signed_in.get(response.headers["location"]).text
    assert [m.content for m in history(thread)] == ["Hi", "ok"]


def test_a_conversation_deleted_mid_answer_leaves_no_history(owner, model):
    # The gap between delete's busy check and the delete itself: the run's
    # last checkpoint would bring the history back, owned by nobody.
    model.replies = [AIMessage("secret answer")]
    thread = conversations.create(owner.id, "hello", now=T0).thread_id
    conversations.remove(thread)  # deleted after the run was let through
    list(assistant._run_turn(thread, "hello", lambda: T0, {}))
    assert history(thread) == []


def test_text_before_a_tool_call_stays_in_the_saved_answer():
    messages: list[BaseMessage] = [
        HumanMessage("pending?"),
        AIMessage("Let me check. ", tool_calls=[{"name": "get_pending_tasks", "args": {}, "id": "a"}]),
        ToolMessage("[]", tool_call_id="a", name="get_pending_tasks"),
        AIMessage("Nothing pending."),
    ]
    [turn] = assistant.turns_from(messages)
    # The same join assistant.js makes while streaming (ANSWER_BREAK after a tool call).
    assert turn.answer == "Let me check. " + assistant.ANSWER_BREAK + "Nothing pending."
    assert "const ANSWER_BREAK = " + json.dumps(assistant.ANSWER_BREAK) in ASSISTANT_JS


ASSISTANT_JS = (Path(assistant.__file__).parents[1] / "static" / "assistant.js").read_text(encoding="utf-8")
