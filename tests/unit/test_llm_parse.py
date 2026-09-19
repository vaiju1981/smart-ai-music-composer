"""`OllamaAdapter.parse` over the wire: the probe, the four error branches, the reply.

C2 recorded this as an owed gap and named the phase that would pay it: the
adapter's parse path had **no HTTP-level test at all**, so its five failure
branches were exercised by nothing — the one test that touched this class
asserted only that it was constructed from a config file. `chat` has had all
of its branches covered since C2, which is what made the asymmetry stark.

Two conventions, both inherited from `test_llm_chat.py` and load-bearing:

- Every assertion is made against a recorded request body rather than against
  the adapter's own helpers. The helpers are the thing under test.
- **The probe is scripted into every recorder, because `parse` issues it
  first.** `_structured_output` starts `None`, so the first `parse` spends a
  request asking whether the host enforces a schema — and whether it does
  changes the *body* of the request that follows. A recorder with one scripted
  reply would be measuring a probe, not a parse.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from saimc.llm.base import ParseRequest
from saimc.llm.ollama import (
    PROMPTED_JSON_MARKER,
    SCHEMA_ENFORCED_MARKER,
    OllamaAdapter,
)
from saimc.spec import CompositionSpec, Mood

_BASE_URL = "http://ollama.invalid:11434"
_MODEL = "a-model"

_SPEC = CompositionSpec(mood=Mood.CALMING, duration_seconds=180)

_PROBE_SCHEMA = {
    "type": "object",
    "properties": {"pong": {"type": "string"}},
    "required": ["pong"],
}


def _reply(payload: dict[str, Any]) -> httpx.Response:
    return httpx.Response(200, json=payload)


def _assistant(content: str) -> dict[str, Any]:
    return {"message": {"role": "assistant", "content": content}, "done": True}


def _probe_pong() -> httpx.Response:
    """The reply that reports native schema enforcement."""
    return _reply(_assistant(json.dumps({"pong": "pong"})))


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


def _request(prompt: str = "calming piano", *, previous_error: str | None = None) -> ParseRequest:
    return ParseRequest(prompt=prompt, request_id="t1", previous_error=previous_error)


async def _parse(recorder: _Recorder, request: ParseRequest | None = None) -> Any:
    """One parse over a recording transport, client closed afterwards."""
    http = httpx.AsyncClient(transport=httpx.MockTransport(recorder.handler))
    adapter = OllamaAdapter(base_url=_BASE_URL, model=_MODEL, http_client=http)
    try:
        return await adapter.parse(request or _request())
    finally:
        await http.aclose()


async def _parse_twice(recorder: _Recorder) -> tuple[OllamaAdapter, Any, Any]:
    """Two parses on one adapter, for the cases about the probe's lifetime."""
    http = httpx.AsyncClient(transport=httpx.MockTransport(recorder.handler))
    adapter = OllamaAdapter(base_url=_BASE_URL, model=_MODEL, http_client=http)
    try:
        first = await adapter.parse(_request())
        second = await adapter.parse(_request())
        return adapter, first, second
    finally:
        await http.aclose()


class TestTheProbeComesFirst:
    """`parse` asks the host what it can do before it decides what to send."""

    def test_the_first_request_is_the_probe_and_the_reply_is_schema_enforced(self) -> None:
        recorder = _Recorder(_probe_pong(), _reply(_assistant(json.dumps(_SPEC.model_dump(mode="json")))))
        result = asyncio.run(_parse(recorder))

        assert recorder.paths == ["/api/chat", "/api/chat"]
        probe, parse_body = recorder.bodies
        assert probe["format"] == _PROBE_SCHEMA
        assert probe["messages"][0]["content"] == "Respond with the single word: pong"
        # The parse body carries the *spec's* schema, which is what enforcement
        # means — and it is a different schema from the probe's.
        assert parse_body["format"] == CompositionSpec.model_json_schema()
        assert result.structured_output is True
        assert result.capability_probe == SCHEMA_ENFORCED_MARKER

    def test_the_probe_is_asked_once_per_adapter(self) -> None:
        """A second parse reuses the answer rather than re-asking."""
        recorder = _Recorder(
            _probe_pong(),
            _reply(_assistant(json.dumps(_SPEC.model_dump(mode="json")))),
            _reply(_assistant(json.dumps(_SPEC.model_dump(mode="json")))),
        )
        _, first, second = asyncio.run(_parse_twice(recorder))
        assert len(recorder.bodies) == 3
        assert first.spec == second.spec == _SPEC

    def test_a_host_that_does_not_enforce_schema_gets_the_schema_in_the_prompt(self) -> None:
        """The fallback mode, which changes the body rather than the answer.

        A probe reply that is not a `pong` is the documented way a host
        reports it ignored `format`, and the adapter then has to reach the
        model through prose. Both halves are asserted: the parameter is gone
        from the request, and the schema arrived in its place.
        """
        recorder = _Recorder(
            _reply(_assistant("Sure! What kind of music would you like?")),
            _reply(_assistant(json.dumps(_SPEC.model_dump(mode="json")))),
        )
        result = asyncio.run(_parse(recorder))
        parse_body = recorder.bodies[1]
        assert "format" not in parse_body
        assert "MUST match this schema exactly" in parse_body["messages"][0]["content"]
        assert result.structured_output is False
        assert result.capability_probe == PROMPTED_JSON_MARKER
        assert result.spec == _SPEC


class TestASuccessfulParse:
    def test_the_spec_comes_back_and_the_extra_records_what_a_benchmark_needs(self) -> None:
        recorder = _Recorder(
            _probe_pong(), _reply(_assistant(json.dumps(_SPEC.model_dump(mode="json"))))
        )
        result = asyncio.run(_parse(recorder))

        assert result.error is None
        assert result.spec == _SPEC
        assert result.parser_source == "llm"
        assert result.attempts == 1
        assert result.extra["model_identifier"] == _MODEL
        assert result.extra["request_id"] == "t1"
        assert result.extra["latency_ms"].isdigit()
        # The adapter does not get to speak for the repair loop: `first_attempt`
        # is what *attempt one* read, and only the loop that made the attempt
        # can witness it.
        assert result.first_attempt is None

    def test_a_repair_carries_the_previous_error_into_the_prompt(self) -> None:
        """The fold `parse_prompt` relies on, asserted at the wire."""
        recorder = _Recorder(
            _probe_pong(), _reply(_assistant(json.dumps(_SPEC.model_dump(mode="json"))))
        )
        result = asyncio.run(
            _parse(recorder, _request("calming piano", previous_error="duration_seconds: bad"))
        )
        system = recorder.bodies[1]["messages"][0]["content"]
        assert "duration_seconds: bad" in system
        assert system.rstrip().endswith("fixes exactly that problem.")
        assert result.spec == _SPEC


class TestEveryFailureBranchIsNamedAndTimed:
    """Four branches, four codes, and all of them carry a latency reading.

    Each is a different *kind* of failure and therefore a different code: the
    host did not answer, the host's answer was not a JSON body, the body's
    content was not JSON, and the JSON was not a spec. Collapsing any two
    would lose the distinction a reader needs — and the fourth is the only one
    whose `stage` is `validating`, because it is the only one where a model
    answered with something well-formed.
    """

    def _result_and_extra(self, recorder: _Recorder, ticks: list[float]) -> tuple[Any, str]:
        """One parse under a fake clock, so the latency is a reading and not a shrug."""
        import saimc.llm.ollama as ollama

        original = ollama.time.perf_counter
        clock = iter(ticks)
        ollama.time.perf_counter = lambda: next(clock)  # type: ignore[assignment]
        try:
            result = asyncio.run(_parse(recorder))
        finally:
            ollama.time.perf_counter = original
        return result, result.extra["latency_ms"]

    def test_an_unreachable_host_is_named_and_timed(self) -> None:
        """The branch that had no `latency_ms` until this commit.

        Fired under a fake clock rather than asserting the key exists: "an
        unreachable request still took time" is the claim, and a key-presence
        check would pass for an empty string.
        """
        recorder = _Recorder(_probe_pong(), httpx.Response(500, text="boom"))
        result, latency_ms = self._result_and_extra(recorder, [10.0, 12.25])
        assert result.error is not None
        assert result.error.error_code == "llm_unreachable"
        assert result.error.stage == "parsing"
        assert result.spec is None
        assert latency_ms == "2250"

    def test_a_body_that_is_not_json_is_named_and_timed(self) -> None:
        recorder = _Recorder(_probe_pong(), httpx.Response(200, text="<html>not json</html>"))
        result, latency_ms = self._result_and_extra(recorder, [10.0, 10.5])
        assert result.error is not None
        assert result.error.error_code == "llm_invalid_response"
        assert result.error.stage == "parsing"
        assert latency_ms == "500"

    def test_content_that_is_not_json_is_named_and_timed(self) -> None:
        recorder = _Recorder(_probe_pong(), _reply(_assistant("I'm sorry, I can't help.")))
        result, latency_ms = self._result_and_extra(recorder, [10.0, 11.0])
        assert result.error is not None
        assert result.error.error_code == "llm_invalid_json"
        assert result.error.stage == "parsing"
        assert latency_ms == "1000"

    def test_json_that_is_not_a_spec_is_named_and_timed(self) -> None:
        recorder = _Recorder(_probe_pong(), _reply(_assistant('{"mood": "euphoric"}')))
        result, latency_ms = self._result_and_extra(recorder, [10.0, 10.25])
        assert result.error is not None
        assert result.error.error_code == "schema_invalid"
        assert result.error.stage == "validating"
        assert result.spec is None
        assert latency_ms == "250"
