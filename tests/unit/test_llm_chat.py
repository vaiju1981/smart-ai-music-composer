"""The chat surface: the types, the wire, and both adapters that satisfy it.

`ChatClient` is the layer under `LLMClient.parse` — messages and tool
schemas in, tool calls out — and the session conductor is its only caller.
Nothing here touches the parse path, which stays covered by
`test_parser.py`.

Two conventions, both load-bearing:

- Every wire assertion is made against a recorded request body rather than
  against the adapter's own helpers. The helpers are the thing under test:
  `_wire_message` agreeing with `_read_tool_call` would prove only that one
  module is self-consistent.
- The async calls are driven with `asyncio.run` from sync tests, as
  `test_llm_config.py` does, and the HTTP client is closed in a `finally`
  rather than through an `async with`. Both keep the event loop's teardown
  inside the test. An `@asynccontextmanager` leaves its async generator to
  be finalized after the loop is closed, and with
  `filterwarnings = ["error"]` the resulting ResourceWarning fails
  whichever *later* test happens to trigger collection — this module's
  first version broke `test_job_stages.py` exactly that way.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from saimc.llm.base import (
    ChatClient,
    ChatRequest,
    ChatResult,
    LLMClient,
    LLMError,
    Message,
    ToolCall,
    ToolSpec,
)
from saimc.llm.ollama import OllamaAdapter

_BASE_URL = "http://ollama.invalid:11434"
_MODEL = "a-model"

_TOOL = ToolSpec(
    name="set_tempo",
    description="Set the tempo in BPM.",
    parameters={
        "type": "object",
        "properties": {"bpm": {"type": "integer"}},
        "required": ["bpm"],
    },
)


def _reply(payload: dict[str, Any]) -> httpx.Response:
    return httpx.Response(200, json=payload)


def _assistant(content: str = "", tool_calls: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"message": message, "done": True}


class _Recorder:
    """An httpx transport that records the request bodies it is handed."""

    def __init__(self, *responses: httpx.Response) -> None:
        self.bodies: list[dict[str, Any]] = []
        self.paths: list[str] = []
        self._responses = list(responses)

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.paths.append(request.url.path)
        self.bodies.append(json.loads(request.content))
        if not self._responses:
            raise AssertionError(f"transport exhausted at call {len(self.bodies)}")
        return self._responses.pop(0)


async def _open(recorder: _Recorder) -> tuple[OllamaAdapter, httpx.AsyncClient]:
    http = httpx.AsyncClient(transport=httpx.MockTransport(recorder.handler))
    return OllamaAdapter(base_url=_BASE_URL, model=_MODEL, http_client=http), http


async def _chat(recorder: _Recorder, request: ChatRequest) -> ChatResult:
    """One chat turn over a recording transport, client closed afterwards."""
    adapter, http = await _open(recorder)
    try:
        return await adapter.chat(request)
    finally:
        await http.aclose()


async def _chat_with_adapter(recorder: _Recorder, request: ChatRequest) -> tuple[Any, ChatResult]:
    """The same, for the cases that need the adapter itself in hand."""
    adapter, http = await _open(recorder)
    try:
        return adapter, await adapter.chat(request)
    finally:
        await http.aclose()


def _noop(monkeypatch: pytest.MonkeyPatch) -> Any:
    """The fallback client, reached the way the worker reaches it.

    `load_llm_config` does not raise on an *unset* environment — it returns
    built-in defaults — so the no-op path has to be provoked rather than
    assumed, or the test would be handed a real adapter pointed at a real
    host.
    """
    from saimc.jobs import worker

    def boom() -> None:
        raise ValueError("bad config")

    monkeypatch.setattr("saimc.llm.config.load_llm_config", boom)
    return worker._build_default_llm_client()


def _go() -> ChatRequest:
    return ChatRequest(messages=(Message(role="user", content="go"),))


class TestTheChatTypesRefuseWhatTheWireCannotCarry:
    """A `Message` that cannot exist on the wire is refused at construction.

    Ollama reads a tool result by `tool_name` and correlates tool calls by
    name and order. A message built without those is not a message the host
    can interpret, so it fails here rather than being sent and misread.
    """

    def test_a_tool_message_must_name_the_tool_it_answers(self) -> None:
        with pytest.raises(ValueError, match="must name the tool"):
            Message(role="tool", content="ok")

    def test_only_a_tool_message_may_carry_a_tool_name(self) -> None:
        with pytest.raises(ValueError, match="only a tool message"):
            Message(role="user", content="hi", tool_name="set_tempo")

    def test_only_an_assistant_message_may_carry_tool_calls(self) -> None:
        with pytest.raises(ValueError, match="only an assistant message"):
            Message(role="user", content="hi", tool_calls=(ToolCall("set_tempo", {"bpm": 90}),))

    def test_an_assistant_tool_call_turn_may_have_empty_content(self) -> None:
        """The normal shape of a tool-calling turn, and it is re-sent in the
        history — so it has to be constructible, not merely tolerated."""
        message = Message(role="assistant", tool_calls=(ToolCall("set_tempo", {"bpm": 90}),))
        assert message.content == ""

    def test_a_chat_request_needs_at_least_one_message(self) -> None:
        with pytest.raises(ValueError, match="at least one message"):
            ChatRequest(messages=())

    def test_a_failed_result_cannot_also_carry_a_reply(self) -> None:
        with pytest.raises(ValueError, match="cannot also carry a reply"):
            ChatResult(content="sure", error=LLMError("llm_unreachable", "x"))
        with pytest.raises(ValueError, match="cannot also carry a reply"):
            ChatResult(tool_calls=(ToolCall("set_tempo", {}),), error=LLMError("x", "y"))

    def test_a_reply_with_no_content_and_no_calls_is_not_an_error(self) -> None:
        """An empty-but-successful reply is a fact about the model, not a
        transport failure, and the conductor is what judges it."""
        assert ChatResult().ok
        assert ChatResult(error=LLMError("llm_unreachable", "x")).ok is False


class TestTheAdapterSpeaksOllamasToolWire:
    def test_the_request_carries_model_messages_and_tools(self) -> None:
        recorder = _Recorder(_reply(_assistant("hello")))
        asyncio.run(
            _chat(
                recorder,
                ChatRequest(
                    messages=(Message(role="system", content="You are a conductor."),),
                    tools=(_TOOL,),
                ),
            )
        )
        body = recorder.bodies[0]
        assert body["model"] == _MODEL
        assert body["stream"] is False
        assert body["messages"] == [{"role": "system", "content": "You are a conductor."}]

    def test_a_tool_goes_on_the_wire_as_a_function_declaration(self) -> None:
        recorder = _Recorder(_reply(_assistant("hello")))
        asyncio.run(_chat(recorder, ChatRequest(messages=_go().messages, tools=(_TOOL,))))
        assert recorder.bodies[0]["tools"] == [
            {
                "type": "function",
                "function": {
                    "name": "set_tempo",
                    "description": "Set the tempo in BPM.",
                    "parameters": {
                        "type": "object",
                        "properties": {"bpm": {"type": "integer"}},
                        "required": ["bpm"],
                    },
                },
            }
        ]

    def test_no_tools_means_no_tools_key(self) -> None:
        """Ollama reads an empty `tools` array as a declared tool surface of
        nothing, which is not the same request as a plain chat."""
        recorder = _Recorder(_reply(_assistant("hello")))
        asyncio.run(_chat(recorder, _go()))
        assert "tools" not in recorder.bodies[0]

    def test_the_system_message_travels_with_the_callers_messages(self) -> None:
        """The conductor's own convention, the opposite of `ParseRequest`'s:
        a system message is the first entry of `messages`, not adapter state.
        """
        recorder = _Recorder(_reply(_assistant("hello")))
        asyncio.run(
            _chat(
                recorder,
                ChatRequest(
                    messages=(
                        Message(role="system", content="policy"),
                        Message(role="user", content="brief"),
                    )
                ),
            )
        )
        assert recorder.bodies[0]["messages"] == [
            {"role": "system", "content": "policy"},
            {"role": "user", "content": "brief"},
        ]

    def test_history_echoes_an_assistant_tool_call_turn_shape_for_shape(self) -> None:
        """Ollama's documented history entry carries `function.name` and
        `function.arguments` and nothing else — no `type`, no call id. An
        echoed `type` would be a field the host never sent.
        """
        recorder = _Recorder(_reply(_assistant("done")))
        asyncio.run(
            _chat(
                recorder,
                ChatRequest(
                    messages=(
                        Message(role="user", content="go"),
                        Message(role="assistant", tool_calls=(ToolCall("set_tempo", {"bpm": 90}),)),
                        Message(role="tool", content="tempo set", tool_name="set_tempo"),
                    )
                ),
            )
        )
        messages = recorder.bodies[0]["messages"]
        assert messages[1] == {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "set_tempo", "arguments": {"bpm": 90}}}],
        }
        assert messages[2] == {"role": "tool", "content": "tempo set", "tool_name": "set_tempo"}


class TestTheAdapterReadsTheReply:
    def test_tool_calls_are_read_by_name_and_arguments(self) -> None:
        recorder = _Recorder(
            _reply(
                _assistant(
                    "",
                    [
                        {"function": {"name": "set_tempo", "arguments": {"bpm": 90}}},
                        {"function": {"name": "set_mood", "arguments": {"mood": "calming"}}},
                    ],
                )
            )
        )
        result = asyncio.run(_chat(recorder, _go()))
        assert result.ok
        assert result.tool_calls == (
            ToolCall("set_tempo", {"bpm": 90}),
            ToolCall("set_mood", {"mood": "calming"}),
        )

    def test_arguments_sent_as_a_json_string_are_read(self) -> None:
        """The documented shape is an object, but a model emitting a string
        is common enough that reading only the object form would drop its
        calls on the floor with no error anywhere."""
        recorder = _Recorder(
            _reply(
                _assistant(
                    "",
                    [{"function": {"name": "set_tempo", "arguments": '{"bpm": 90}'}}],
                )
            )
        )
        result = asyncio.run(_chat(recorder, _go()))
        assert result.tool_calls == (ToolCall("set_tempo", {"bpm": 90}),)

    def test_omitted_arguments_become_an_empty_mapping(self) -> None:
        """A tool that takes no arguments is a real tool, so its call is
        real; there is nothing unreadable about an absent key."""
        recorder = _Recorder(_reply(_assistant("", [{"function": {"name": "compare"}}])))
        result = asyncio.run(_chat(recorder, _go()))
        assert result.tool_calls == (ToolCall("compare", {}),)

    def test_a_prose_reply_with_no_tool_calls_is_a_success(self) -> None:
        """A model that answers in words when tools were offered is a normal
        outcome. Reporting it as a transport error would hide the prose the
        conductor has to decide about."""
        recorder = _Recorder(_reply(_assistant("I would rather explain my plan.")))
        result = asyncio.run(_chat(recorder, ChatRequest(messages=_go().messages, tools=(_TOOL,))))
        assert result.ok
        assert result.content == "I would rather explain my plan."
        assert result.tool_calls == ()

    def test_content_and_calls_can_arrive_together(self) -> None:
        recorder = _Recorder(
            _reply(
                _assistant(
                    "Setting that now.",
                    [{"function": {"name": "set_tempo", "arguments": {"bpm": 90}}}],
                )
            )
        )
        result = asyncio.run(_chat(recorder, _go()))
        assert result.content == "Setting that now."
        assert result.tool_calls == (ToolCall("set_tempo", {"bpm": 90}),)

    def test_the_result_records_what_a_benchmark_needs(self) -> None:
        recorder = _Recorder(_reply(_assistant("hello")))
        result = asyncio.run(_chat(recorder, _go()))
        assert result.extra["model_identifier"] == _MODEL
        assert int(result.extra["latency_ms"]) >= 0


class TestTheAdapterRefusesWhatItCannotRead:
    """An unreadable call is refused, never defaulted.

    Turning bad arguments into `{}` would run a real tool with wrong values
    — a wrong action rather than a missing one, which is exactly the silent
    no-op the plan's delta rules forbid.
    """

    def test_arguments_that_are_not_json_are_refused(self) -> None:
        recorder = _Recorder(
            _reply(_assistant("", [{"function": {"name": "set_tempo", "arguments": "{bpm: 90"}}]))
        )
        result = asyncio.run(_chat(recorder, _go()))
        assert result.error is not None
        assert result.error.error_code == "tool_call_malformed"
        assert result.tool_calls == ()

    def test_arguments_that_are_a_json_array_are_refused(self) -> None:
        recorder = _Recorder(
            _reply(_assistant("", [{"function": {"name": "set_tempo", "arguments": "[90]"}}]))
        )
        result = asyncio.run(_chat(recorder, _go()))
        assert result.error is not None
        assert result.error.error_code == "tool_call_malformed"

    def test_a_call_with_no_name_is_refused(self) -> None:
        recorder = _Recorder(_reply(_assistant("", [{"function": {"arguments": {"bpm": 90}}}])))
        result = asyncio.run(_chat(recorder, _go()))
        assert result.error is not None
        assert result.error.error_code == "tool_call_malformed"

    def test_an_http_failure_is_named_not_swallowed(self) -> None:
        recorder = _Recorder(httpx.Response(500, text="boom"))
        result = asyncio.run(_chat(recorder, _go()))
        assert result.error is not None
        assert result.error.error_code == "llm_unreachable"
        assert result.ok is False

    def test_a_body_that_is_not_json_is_named(self) -> None:
        recorder = _Recorder(httpx.Response(200, text="<html>not json</html>"))
        result = asyncio.run(_chat(recorder, _go()))
        assert result.error is not None
        assert result.error.error_code == "llm_invalid_response"


class TestAChatTurnDoesNotProbeStructuredOutput:
    """`parse` probes the host for `format=` support because it has to choose
    a body shape. `chat` has no second mode to choose between, so probing
    would spend a network call on an answer it does not use — and would
    report a capability the chat path never exercised.
    """

    def test_the_probe_is_never_issued_and_the_flag_stays_unset(self) -> None:
        recorder = _Recorder(_reply(_assistant("hello")))
        adapter, result = asyncio.run(_chat_with_adapter(recorder, _go()))
        assert result.ok
        assert adapter.structured_output is None
        assert recorder.paths == ["/api/chat"]
        assert len(recorder.bodies) == 1
        assert "format" not in recorder.bodies[0]


class TestBothAdaptersSatisfyBothProtocols:
    """Structural typing, checked rather than assumed.

    `isinstance` against a `runtime_checkable` Protocol passes only if every
    method is present, so this fails on a renamed or missing `chat` — which
    is the failure a duck-typed codebase otherwise discovers at runtime, in
    production, on the turn that needed the model.
    """

    def test_the_ollama_adapter_satisfies_both(self) -> None:
        adapter, _ = asyncio.run(_chat_with_adapter(_Recorder(_reply(_assistant("hi"))), _go()))
        assert isinstance(adapter, LLMClient)
        assert isinstance(adapter, ChatClient)

    def test_the_noop_fallback_satisfies_both(self, monkeypatch: pytest.MonkeyPatch) -> None:
        client = _noop(monkeypatch)
        assert isinstance(client, LLMClient)
        assert isinstance(client, ChatClient)

    def test_the_noop_refuses_by_name_rather_than_answering_emptily(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An empty reply would read as a model that chose to say nothing,
        which is a different situation from one that is not there. This is
        what makes "with the LLM disabled every deterministic path still
        works, and only the conductor's turns fail" true.
        """
        client = _noop(monkeypatch)
        result = asyncio.run(client.chat(_go()))
        assert result.ok is False
        assert result.error is not None
        assert result.error.error_code == "llm_not_configured"
        assert result.tool_calls == ()
