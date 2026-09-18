"""Unit tests for the parser orchestration (LLM + fallback).

The async calls are driven with `asyncio.run` from sync tests, as
`test_llm_chat.py` and `test_llm_config.py` do, rather than through
`@pytest.mark.asyncio`. The marker is not the simpler spelling: it is the
one that leaks. `pytest_asyncio`'s `event_loop` fixture ends by installing
a fresh loop for the next `get_event_loop()` caller and never closing it,
so every marked test leaves a loop — and its AF_UNIX self-pipe — for the
cyclic collector, which under `filterwarnings = ["error"]` reports the
`ResourceWarning` inside whichever *later* test happens to trigger that
collection. This module was the last user of the marker; with it gone the
plugin is inert, which is why it is no longer a dev dependency.
"""

from __future__ import annotations

import ast
import asyncio
import dataclasses
from pathlib import Path

from saimc.llm.base import LLMClient, ParseRequest, ParseResult
from saimc.parser import MAX_LLM_ATTEMPTS, NO_COMPLETION_ERROR_CODES, parse_prompt
from saimc.spec import CompositionSpec, Mood, SpecError

_REPO_ROOT = Path(__file__).resolve().parents[2]


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


def _parse(client: LLMClient, prompt: str, *, request_id: str) -> ParseResult:
    """Drive one parse to completion, on a loop closed before this returns."""
    return asyncio.run(parse_prompt(client, prompt, request_id=request_id))


def test_llm_first_attempt_succeeds() -> None:
    client: LLMClient = _ScriptedClient(
        [ParseResult(parser_source="llm", spec=_spec(), extra={"latency_ms": "5"})]
    )
    result = _parse(client, "calming piano music", request_id="r1")
    assert result.spec is not None
    assert result.parser_source == "llm"
    assert result.attempts == 1


def test_llm_repairs_succeed_on_second_attempt() -> None:
    client: LLMClient = _ScriptedClient(
        [
            ParseResult(parser_source="llm", error=_err("schema_invalid")),
            ParseResult(parser_source="llm", spec=_spec(), extra={"latency_ms": "5"}),
        ]
    )
    result = _parse(client, "calming piano music", request_id="r2")
    assert result.spec is not None
    assert result.attempts == 2


def test_fallback_used_after_three_attempts() -> None:
    client: LLMClient = _ScriptedClient(
        [
            ParseResult(parser_source="llm", error=_err()),
            ParseResult(parser_source="llm", error=_err()),
            ParseResult(parser_source="llm", error=_err()),
        ]
    )
    result = _parse(client, "calming piano music", request_id="r3")
    assert result.spec is not None
    assert result.parser_source == "hybrid"
    assert result.attempts == MAX_LLM_ATTEMPTS
    assert result.extra.get("fallback_used") == "true"


def test_out_of_vocabulary_surfaces_fallback_message() -> None:
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
    result = _parse(client, "an angry piano piece", request_id="r4")
    assert result.spec is None
    assert result.error is not None
    assert result.error.error_code == "out_of_vocabulary"
    assert "calming" in result.error.message  # the vocabulary hint
    assert result.extra.get("llm_error_code") == "schema_invalid"


def test_transport_error_not_replaced_by_fallback_rejection() -> None:
    """A transport-level LLM failure (llm_unreachable) stays surfaced —
    the fallback vocabulary hint would misstate the cause."""
    client: LLMClient = _ScriptedClient(
        [
            ParseResult(parser_source="llm", error=_err("llm_unreachable", "parsing")),
            ParseResult(parser_source="llm", error=_err("llm_unreachable", "parsing")),
            ParseResult(parser_source="llm", error=_err("llm_unreachable", "parsing")),
        ]
    )
    result = _parse(client, "an angry piano piece", request_id="r6")
    assert result.error is not None
    assert result.error.error_code == "llm_unreachable"
    assert result.extra.get("fallback_error_code") == "out_of_vocabulary"


def test_repair_attempts_carry_previous_error() -> None:
    """The §6 repair loop must tell the model what went wrong."""
    client: LLMClient = _ScriptedClient(
        [
            ParseResult(parser_source="llm", error=_err("schema_invalid")),
            ParseResult(parser_source="llm", spec=_spec(), extra={"latency_ms": "5"}),
        ]
    )
    result = _parse(client, "calming piano music", request_id="r7")
    assert result.spec is not None
    assert result.attempts == 2
    # Second call's request carried the first attempt's error.
    second_request = client.requests[1]
    assert second_request.previous_error is not None
    assert "schema_invalid" in second_request.previous_error
    # First attempt had no feedback to give.
    assert client.requests[0].previous_error is None


def test_does_not_exceed_max_attempts() -> None:
    client: LLMClient = _ScriptedClient(
        [
            ParseResult(parser_source="llm", error=_err()),
            ParseResult(parser_source="llm", error=_err()),
            ParseResult(parser_source="llm", error=_err()),
        ]
    )
    _parse(client, "calming piano music", request_id="r5")
    assert client.calls == MAX_LLM_ATTEMPTS


class TestWhatTheFirstAttemptRead:
    """§8's first bar, which only this loop can witness.

    Every case here is about *attempt one's* result rather than the run's, so
    the fixture values are chosen to make the two differ wherever they can.
    """

    def test_a_spec_on_attempt_one_is_valid(self) -> None:
        client: LLMClient = _ScriptedClient(
            [ParseResult(parser_source="llm", spec=_spec(), extra={"latency_ms": "5"})]
        )
        result = _parse(client, "calming piano music", request_id="fa1")
        assert result.first_attempt == "valid"

    def test_a_bad_reply_on_attempt_one_is_invalid_even_when_a_later_one_parses(self) -> None:
        """The case the `last_result` reading gets wrong.

        Attempt one answered and the answer did not parse; attempt three did.
        A field written from the result the loop *exits* with would say
        `"valid"` here and score this prompt as a clean first pass, which is
        the whole defect the field exists to remove. So the run ends with a
        spec and the first attempt is still `"invalid"`.
        """
        client: LLMClient = _ScriptedClient(
            [
                ParseResult(parser_source="llm", error=_err("schema_invalid")),
                ParseResult(parser_source="llm", error=_err("schema_invalid")),
                ParseResult(parser_source="llm", spec=_spec(), extra={"latency_ms": "5"}),
            ]
        )
        result = _parse(client, "calming piano music", request_id="fa2")
        assert result.spec is not None
        assert result.attempts == MAX_LLM_ATTEMPTS
        assert result.first_attempt == "invalid"

    def test_a_run_where_no_model_answered_is_neither(self) -> None:
        """`None`, not `"invalid"` — and the run still resolves.

        The host is unreachable for all three attempts and the fallback
        answers, so the result carries a spec and `first_attempt` is `None`.
        A two-state field would have to call attempt one *invalid* for a run
        in which no model output was ever read, which charges the model for
        the network's failure.
        """
        client: LLMClient = _ScriptedClient(
            [
                ParseResult(parser_source="llm", error=_err("llm_unreachable", "parsing")),
                ParseResult(parser_source="llm", error=_err("llm_unreachable", "parsing")),
                ParseResult(parser_source="llm", error=_err("llm_unreachable", "parsing")),
            ]
        )
        result = _parse(client, "calming piano music", request_id="fa3")
        assert result.spec is not None
        assert result.parser_source == "hybrid"
        assert result.first_attempt is None

    def test_a_run_that_ends_refused_still_reports_what_attempt_one_read(self) -> None:
        """The field describes attempt one even when there is no spec at all."""
        client: LLMClient = _ScriptedClient(
            [
                ParseResult(parser_source="llm", error=_err("schema_invalid")),
                ParseResult(parser_source="llm", error=_err("schema_invalid")),
                ParseResult(parser_source="llm", error=_err("schema_invalid")),
            ]
        )
        result = _parse(client, "an angry piano piece", request_id="fa4")
        assert result.spec is None
        assert result.first_attempt == "invalid"

    def test_an_adapter_does_not_get_to_set_the_field(self) -> None:
        """A client that claims a first attempt is overwritten by the loop.

        `ParseResult` is documented as the *attempt* an adapter returns, and
        an adapter has no way to know whether its call was the first one. A
        fixture that sets the field to something else proves the loop is the
        writer rather than a forwarder.
        """
        client: LLMClient = _ScriptedClient(
            [
                ParseResult(
                    parser_source="llm",
                    spec=_spec(),
                    first_attempt="invalid",
                    extra={"latency_ms": "5"},
                )
            ]
        )
        result = _parse(client, "calming piano music", request_id="fa5")
        assert result.first_attempt == "valid"


class TestTheNoCompletionVocabulary:
    """The ratchet on `NO_COMPLETION_ERROR_CODES`.

    The set is the rule §8's first bar rests on, and its correctness is a
    property of code outside this module — the adapter's branches — so it is
    checked against the code that mints those codes rather than restated. A
    fifth branch added to the parse adapter fails
    `test_every_code_the_adapters_mint_is_classified` until it is classified
    one way or the other, which is the difference between a deliberate
    decision and a branch that silently reports 0%.
    """

    # The codes that mean the model *did* answer. Not a constant in `parser.py`
    # because nothing there branches on them — the only consumer is the set
    # complement — so the partition is spelled out here, where a new code has
    # to be placed.
    JUDGEABLE: frozenset[str] = frozenset({"llm_invalid_json", "schema_invalid"})

    @staticmethod
    def _parse_path_codes(path: Path, *, within: str | None = None) -> set[str]:
        """Every `SpecError(error_code="…")` literal in `path` (or in one class).

        Both spellings of the callee are matched — `SpecError(...)` and
        `module.SpecError(...)` — because the worker's `_NoopLLM` builds its
        error through a deferred `__import__` rather than a local import, and a
        scanner that only saw the bare name would quietly find nothing there
        and pass.
        """
        tree = ast.parse(path.read_text(encoding="utf-8"))
        scope: list[ast.AST] = [tree]
        if within is not None:
            classes = [
                n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == within
            ]
            assert len(classes) == 1, f"expected exactly one class {within!r} in {path}"
            scope = classes
        codes: set[str] = set()
        for node in scope:
            for call in ast.walk(node):
                if not isinstance(call, ast.Call):
                    continue
                func = call.func
                is_spec_error = (isinstance(func, ast.Name) and func.id == "SpecError") or (
                    isinstance(func, ast.Attribute) and func.attr == "SpecError"
                )
                if not is_spec_error:
                    continue
                for kw in call.keywords:
                    if kw.arg == "error_code":
                        assert isinstance(kw.value, ast.Constant), f"non-literal code in {path}"
                        assert isinstance(kw.value.value, str)
                        codes.add(kw.value.value)
        return codes

    def _minted_codes(self) -> set[str]:
        return self._parse_path_codes(
            _REPO_ROOT / "src" / "saimc" / "llm" / "ollama.py"
        ) | self._parse_path_codes(
            _REPO_ROOT / "src" / "saimc" / "jobs" / "worker.py", within="_NoopLLM"
        )

    def test_the_adapters_mint_exactly_the_codes_this_suite_knows(self) -> None:
        """The fixture's own premise, so the ratchet below cannot pass vacuously."""
        assert self._minted_codes() == {
            "llm_unreachable",
            "llm_invalid_response",
            "llm_invalid_json",
            "schema_invalid",
            "llm_not_configured",
        }

    def test_every_code_the_adapters_mint_is_classified(self) -> None:
        minted = self._minted_codes()
        classified = NO_COMPLETION_ERROR_CODES | self.JUDGEABLE
        assert minted - classified == set(), (
            "unclassified parse-path error codes: "
            f"{sorted(minted - classified)}. Add each to NO_COMPLETION_ERROR_CODES "
            "(no model output was read) or to JUDGEABLE above (the model answered), "
            "then say which in a docstring."
        )

    def test_the_two_sides_do_not_overlap(self) -> None:
        """A code in both would make the third state unreachable for it."""
        assert NO_COMPLETION_ERROR_CODES.isdisjoint(self.JUDGEABLE)

    def test_the_transport_codes_are_the_ones_that_are_not_a_judgement(self) -> None:
        """Named explicitly, because the split is a claim about the adapter.

        `llm_invalid_json` and `schema_invalid` are the two the model can only
        produce by answering, and they are absent from the set for that reason
        — asserted rather than left to the complement above, which would still
        hold if the set grew to swallow them.
        """
        assert "llm_invalid_json" not in NO_COMPLETION_ERROR_CODES
        assert "schema_invalid" not in NO_COMPLETION_ERROR_CODES
