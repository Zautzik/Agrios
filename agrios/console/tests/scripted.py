"""A chat model that answers from a script, for tests that must run without
Ollama: the Assistant tests and the browser tests."""

import json
import re
import threading
from collections.abc import Iterator
from typing import Any

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from pydantic import Field as PydanticField


class ScriptedModel(BaseChatModel):
    """Answers with the next scripted reply. Tool calls stream as tool-call
    chunks and text word by word, the way ChatOllama streams them, because
    the Assistant shows the stream, not only the final message."""

    replies: list[AIMessage | Exception] = PydanticField(default_factory=list)
    # When set, each reply waits for it first: lets a test act while a reply is in progress.
    gate: threading.Event | None = None
    bound: list[list[str]] = PydanticField(default_factory=list)
    seen: list[list[BaseMessage]] = PydanticField(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedModel":
        self.bound.append([t.name for t in tools])
        return self

    def _next(self, messages: list[BaseMessage]) -> AIMessage:
        self.seen.append(messages)
        if self.gate is not None:
            self.gate.wait(timeout=30)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kw: Any
    ) -> ChatResult:
        return ChatResult(generations=[ChatGeneration(message=self._next(messages))])

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: Any = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kw: Any,
    ) -> Iterator[ChatGenerationChunk]:
        reply = self._next(messages)
        if reply.tool_calls:
            chunks = [
                {"name": c["name"], "args": json.dumps(c["args"]), "id": c["id"], "index": i}
                for i, c in enumerate(reply.tool_calls)
            ]
            yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=chunks))
            return
        for word in re.split(r"(\s)", str(reply.content)):
            chunk = ChatGenerationChunk(message=AIMessageChunk(content=word))
            if run_manager:
                run_manager.on_llm_new_token(word, chunk=chunk)
            yield chunk


def call(name: str, call_id: str = "c1", **args: Any) -> AIMessage:
    return AIMessage("", tool_calls=[{"name": name, "args": args, "id": call_id}])
