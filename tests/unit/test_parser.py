"""Unit tests for the parser orchestration (LLM + fallback)."""

from __future__ import annotations

import dataclasses

import pytest

from saimc.llm.base import LLMClient, ParseRequest, ParseResult
from saimc.parser import MAX_LLM_ATTEMPTS, parse_prompt
from saimc.spec import CompositionSpec, Mood, SpecError


class _ScriptedClient:
    """Test double for LLMClient.

    `responses` is a list of ParseResults returned in order; the client
    raises StopIteration if the list is exhausted (the parser should never
    exceed MAX_LLM_ATTEMPTS calls).
    """

    def __init__(self, responses: list[ParseResult], *, latency_ms: int = 10) -> None:
        self._responses = list(responses)
        self.calls = 0
        self._latency_ms = latency_ms
        self.requests: list[ParseRequest] = []

    async def parse(self, request: ParseRequest) -> ParseResult:
        self.requests.append(request)
        self.calls += 1
        if not self._responses:
            raise AssertionError(
                f"_ScriptedClient exhausted at call {self.calls}; parser asked for too many attempts"
            )
        return dataclasses.replace(
            self._responses.pop(0),
            extra={**self._responses[0].extra, "latency_ms": str(self._latency_ms)}
            if self._responses
            else {},
        )

    async def aclose(self) -> None:
        return None


def _spec(mood: str = "calming", duration: int = 180) -> CompositionSpec:
    return CompositionSpec(mood=Mood(mood), duration_seconds=duration)


def _err(code: str = "schema_invalid", stage: str = "validating") -> SpecError:
    return SpecError(error_code=code, message="x", stage=stage)


@pytest.mark.asyncio
async def test_llm_first_attempt_succeeds() -> None:
    client: LLMClient = _ScriptedClient(
        [ParseResult(parser_source="llm", spec=_spec(), extra={"latency_ms": "5"})]
    )
    result = await parse_prompt(client, "calming piano music", request_id="r1")
    assert result.spec is not None
    assert result.parser_source == "llm"
    assert result.attempts == 1


@pytest.mark.asyncio
async def test_llm_repairs_succeed_on_second_attempt() -> None:
    client: LLMClient = _ScriptedClient(
        [
            ParseResult(parser_source="llm", error=_err("schema_invalid")),
            ParseResult(parser_source="llm", spec=_spec(), extra={"latency_ms": "5"}),
        ]
    )
    result = await parse_prompt(client, "calming piano music", request_id="r2")
    assert result.spec is not None
    assert result.attempts == 2


@pytest.mark.asyncio
async def test_fallback_used_after_three_attempts() -> None:
    client: LLMClient = _ScriptedClient(
        [
            ParseResult(parser_source="llm", error=_err()),
            ParseResult(parser_source="llm", error=_err()),
            ParseResult(parser_source="llm", error=_err()),
        ]
    )
    result = await parse_prompt(client, "calming piano music", request_id="r3")
    assert result.spec is not None
    assert result.parser_source == "hybrid"
    assert result.attempts == MAX_LLM_ATTEMPTS
    assert result.extra.get("fallback_used") == "true"


@pytest.mark.asyncio
async def test_out_of_vocabulary_surfaces_fallback_message() -> None:
    """When the LLM fails schema validation and fallback rejects the
    prompt as out-of-vocabulary, the user gets the vocabulary hint —
    not the LLM plumbing error."""
    client: LLMClient = _ScriptedClient(
        [
            ParseResult(parser_source="llm", error=_err()),
            ParseResult(parser_source="llm", error=_err()),
            ParseResult(parser_source="llm", error=_err()),
        ]
    )
    result = await parse_prompt(client, "an angry piano piece", request_id="r4")
    assert result.spec is None
    assert result.error is not None
    assert result.error.error_code == "out_of_vocabulary"
    assert "calming" in result.error.message  # the vocabulary hint
    assert result.extra.get("llm_error_code") == "schema_invalid"


@pytest.mark.asyncio
async def test_transport_error_not_replaced_by_fallback_rejection() -> None:
    """A transport-level LLM failure (llm_unreachable) stays surfaced —
    the fallback vocabulary hint would misstate the cause."""
    client: LLMClient = _ScriptedClient(
        [
            ParseResult(parser_source="llm", error=_err("llm_unreachable", "parsing")),
            ParseResult(parser_source="llm", error=_err("llm_unreachable", "parsing")),
            ParseResult(parser_source="llm", error=_err("llm_unreachable", "parsing")),
        ]
    )
    result = await parse_prompt(client, "an angry piano piece", request_id="r6")
    assert result.error is not None
    assert result.error.error_code == "llm_unreachable"
    assert result.extra.get("fallback_error_code") == "out_of_vocabulary"


@pytest.mark.asyncio
async def test_repair_attempts_carry_previous_error() -> None:
    """The §6 repair loop must tell the model what went wrong."""
    client: LLMClient = _ScriptedClient(
        [
            ParseResult(parser_source="llm", error=_err("schema_invalid")),
            ParseResult(parser_source="llm", spec=_spec(), extra={"latency_ms": "5"}),
        ]
    )
    result = await parse_prompt(client, "calming piano music", request_id="r7")
    assert result.spec is not None
    assert result.attempts == 2
    # Second call's request carried the first attempt's error.
    second_request = client.requests[1]
    assert second_request.previous_error is not None
    assert "schema_invalid" in second_request.previous_error
    # First attempt had no feedback to give.
    assert client.requests[0].previous_error is None


@pytest.mark.asyncio
async def test_does_not_exceed_max_attempts() -> None:
    client: LLMClient = _ScriptedClient(
        [
            ParseResult(parser_source="llm", error=_err()),
            ParseResult(parser_source="llm", error=_err()),
            ParseResult(parser_source="llm", error=_err()),
        ]
    )
    await parse_prompt(client, "calming piano music", request_id="r5")
    assert client.calls == MAX_LLM_ATTEMPTS
