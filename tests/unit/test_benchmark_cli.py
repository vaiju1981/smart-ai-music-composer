"""Tests for the benchmark CLI — the offline path and the live sweep.

The live path is driven through `_run_with_client` directly with a stub
client, and through `main` with `build_ollama_adapter` patched out, because
neither case may touch a host: this suite runs with no Ollama reachable, and a
test that reached one would be measuring the network rather than the runner.
The async call is driven with `asyncio.run` from sync tests, as
`test_parser.py` does and for the reason recorded there.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Literal

import pytest
from typer.testing import CliRunner

from saimc.benchmark import ParseObservation
from saimc.benchmark_cli import _observe, _run_with_client, app
from saimc.benchmark_corpus import BenchmarkRecord
from saimc.llm.base import ParseRequest, ParseResult
from saimc.parser import MAX_LLM_ATTEMPTS
from saimc.spec import CompositionSpec, Mood, SpecError

runner = CliRunner()


class _StubClient:
    """A scripted `LLMClient` that records whether it was closed."""

    def __init__(self, responses: list[ParseResult]) -> None:
        self._responses = list(responses)
        self.requests: list[ParseRequest] = []
        self.closed = False

    async def parse(self, request: ParseRequest) -> ParseResult:
        self.requests.append(request)
        if not self._responses:
            raise AssertionError("stub client exhausted: the runner asked for too many attempts")
        return self._responses.pop(0)

    async def aclose(self) -> None:
        self.closed = True


def _spec() -> CompositionSpec:
    return CompositionSpec(mood=Mood.CALMING, duration_seconds=180)


def _err(code: str = "schema_invalid", stage: str = "validating") -> SpecError:
    return SpecError(error_code=code, message="x", stage=stage)


def _records(n: int, *, expected: str = "accepted") -> list[BenchmarkRecord]:
    return [
        BenchmarkRecord(
            id=f"r{i}",
            category="supported_paraphrase" if expected == "accepted" else "unsupported",
            prompt=f"prompt-{i}",
            expected_outcome=expected,
            expected_spec=None,
            expected_error=None if expected == "accepted" else "out_of_vocabulary",
            label_rationale="cli test record",
        )
        for i in range(n)
    ]


def test_cli_fallback_only_runs(tmp_path: Path) -> None:
    """The fallback parser is expected to miss corpus thresholds.

    The CLI must still run cleanly and write the report; the failing exit
    code reflects the (expected) miss, not a bug.
    """
    out = tmp_path / "report.json"
    result = runner.invoke(
        app,
        ["--fallback-only", "--label", "fallback", "--output", str(out)],
    )
    payload = json.loads(out.read_text())
    assert payload["client_label"] == "fallback"
    assert payload["corpus_size"] >= 20
    assert payload["latency_p95_s"] < 5.0
    assert payload["passed"] is False
    assert result.exit_code == 1


def test_cli_verbose_lists_records(tmp_path: Path) -> None:
    out = tmp_path / "report.json"
    result = runner.invoke(
        app,
        ["--fallback-only", "--label", "fallback", "--output", str(out), "--verbose"],
    )
    assert "supported_001" in result.stdout
    assert "field_exact" in result.stdout


def test_offline_run_reports_the_first_pass_as_unmeasured(tmp_path: Path) -> None:
    """The offline path asks no model, so it has no first pass to report.

    This is the whole of G3's headline defect: the flag used to be a
    hardcoded `True` on this path, so the only runnable benchmark reported a
    first-pass validity of 1.0 for a run that called nothing.
    """
    out = tmp_path / "report.json"
    runner.invoke(app, ["--fallback-only", "--output", str(out)])
    payload = json.loads(out.read_text())
    assert payload["first_pass_validity"] is None
    assert any("not measured" in r for r in payload["failure_reasons"])


def test_the_run_prints_the_corpus_hash(tmp_path: Path) -> None:
    """`docs/parser-benchmark.md` requires it recorded beside a score."""
    from saimc.benchmark_corpus import DEFAULT_CORPUS_PATH, corpus_sha256

    out = tmp_path / "report.json"
    result = runner.invoke(app, ["--fallback-only", "--output", str(out)])
    assert corpus_sha256(DEFAULT_CORPUS_PATH) in result.stdout


def test_the_corpus_option_is_honoured(tmp_path: Path) -> None:
    """`--corpus` was accepted, marked required, and then discarded.

    The loader was called with no path at all, so a run against a different
    eval set silently scored the canonical one — and the option's declaration
    claimed the caller had to supply it. Both halves are fixed together here
    because either one alone leaves the option a lie.
    """
    small = tmp_path / "small.jsonl"
    small.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": f"s{i}",
                    "category": "supported_paraphrase",
                    "prompt": "calming piano music",
                    "expected_outcome": "accepted",
                    "expected_spec": None,
                    "expected_error": None,
                    "label_rationale": "cli test record",
                }
            )
            for i in range(2)
        )
        + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "report.json"
    result = runner.invoke(app, ["--fallback-only", "--corpus", str(small), "--output", str(out)])
    assert "Loaded 2 records" in result.stdout
    assert json.loads(out.read_text())["corpus_size"] == 2


def test_a_live_run_with_no_configuration_refuses_by_name(tmp_path: Path, monkeypatch) -> None:
    """The one case where there is no adapter, and it must not read as a run."""
    monkeypatch.setattr("saimc.benchmark_cli.build_ollama_adapter", lambda **_: None)
    out = tmp_path / "report.json"
    result = runner.invoke(app, ["--label", "anything", "--output", str(out)])
    assert result.exit_code == 1
    combined = result.stdout + (result.stderr or "")
    assert "no LLM configuration" in combined
    assert not out.exists()


class TestTheLiveSweep:
    def test_one_observation_per_record_and_the_client_is_closed(self) -> None:
        records = _records(3)
        client = _StubClient(
            [ParseResult(parser_source="llm", spec=_spec(), extra={"latency_ms": "5"})] * 3
        )
        report = asyncio.run(_run_with_client(records, "stub", client))
        assert report.corpus_size == 3
        assert len(client.requests) == 3
        assert client.closed, "the sweep owns the client it is handed and must close it"
        assert report.first_pass_validity == 1.0

    def test_a_recovered_prompt_is_not_a_clean_first_pass(self) -> None:
        """The end-to-end shape of G3's field: a spec after repairs, `invalid` on attempt one."""
        client = _StubClient(
            [
                ParseResult(parser_source="llm", error=_err()),
                ParseResult(parser_source="llm", spec=_spec(), extra={"latency_ms": "5"}),
            ]
        )
        report = asyncio.run(_run_with_client(_records(1), "stub", client))
        assert report.first_pass_validity == 0.0
        assert report.resolution_rate == 1.0
        assert report.outcomes[0].attempts == 2
        assert report.outcomes[0].matched

    def test_a_run_that_never_reached_the_host_reports_not_measured(self) -> None:
        client = _StubClient(
            [
                ParseResult(parser_source="llm", error=_err("llm_unreachable", "parsing")),
            ]
            * MAX_LLM_ATTEMPTS
        )
        report = asyncio.run(_run_with_client(_records(1), "stub", client))
        assert report.first_pass_validity is None
        assert not report.passed
        assert any("not measured" in r for r in report.failure_reasons)

    def test_latency_is_the_whole_request_not_the_last_call(self, monkeypatch) -> None:
        """Three attempts must measure as the slower request the user waits for.

        The stub's own `extra` says 5 ms, and the fake clock says 3 s, so the
        recorded figure can only have come from the runner's own reading —
        which is what makes this a test of *where* the timer sits rather than
        of arithmetic. A runner that read the adapter's diagnostic would fail
        it, and so would one that timed only the final attempt.
        """
        ticks = iter([0.0, 3.0])
        monkeypatch.setattr("saimc.benchmark_cli.time.perf_counter", lambda: next(ticks))
        client = _StubClient(
            [
                ParseResult(parser_source="llm", error=_err()),
                ParseResult(parser_source="llm", error=_err()),
                ParseResult(parser_source="llm", spec=_spec(), extra={"latency_ms": "5"}),
            ]
        )
        report = asyncio.run(_run_with_client(_records(1), "stub", client))
        assert report.outcomes[0].latency_ms == 3000

    def test_a_closed_client_is_closed_even_when_the_sweep_raises(self) -> None:
        """The `finally`, which a passing run cannot distinguish from a happy path."""
        client = _StubClient([])
        with pytest.raises(AssertionError, match="exhausted"):
            asyncio.run(_run_with_client(_records(1), "stub", client))
        assert client.closed


def test_a_live_run_labels_itself_with_the_tag_the_host_served(
    tmp_path: Path, monkeypatch
) -> None:
    """The default label is the adapter's own resolved tag, not the requested one.

    A proxy's `*-cloud` namespace need not be the tag that was asked for, and
    the identifier `MODELS.md` records is the one this host served — so the
    default is read back off the client rather than re-derived from
    configuration, which is a second answer that can disagree with the first.
    """
    corpus = tmp_path / "two.jsonl"
    corpus.write_text(
        "\n".join(
            json.dumps(
                {
                    "id": f"s{i}",
                    "category": "supported_paraphrase",
                    "prompt": "calming piano music",
                    "expected_outcome": "accepted",
                    "expected_spec": None,
                    "expected_error": None,
                    "label_rationale": "cli test record",
                }
            )
            for i in range(2)
        )
        + "\n",
        encoding="utf-8",
    )

    class _StubAdapter(_StubClient):
        model_identifier = "served-tag:latest"

    monkeypatch.setattr(
        "saimc.benchmark_cli.build_ollama_adapter",
        lambda **_: _StubAdapter(
            [ParseResult(parser_source="llm", spec=_spec(), extra={"latency_ms": "5"})] * 2
        ),
    )
    out = tmp_path / "report.json"
    runner.invoke(app, ["--corpus", str(corpus), "--output", str(out)])
    assert json.loads(out.read_text())["client_label"] == "served-tag:latest"


class TestTheThreeStatesOnTheWire:
    """`ParseResult.first_attempt` onto `ParseObservation.raw_first_response_valid`."""

    @staticmethod
    def _result(first_attempt: Literal["valid", "invalid"] | None) -> ParseResult:
        return ParseResult(parser_source="llm", spec=_spec(), first_attempt=first_attempt)

    def test_valid_is_true(self) -> None:
        obs = _observe(self._result("valid"), 12)
        assert obs.raw_first_response_valid is True
        assert obs.latency_ms == 12

    def test_invalid_is_false(self) -> None:
        assert _observe(self._result("invalid"), 12).raw_first_response_valid is False

    def test_none_stays_none(self) -> None:
        """Not `False`: the same distinction the field carries, one layer up."""
        assert _observe(self._result(None), 12).raw_first_response_valid is None

    def test_cost_is_recorded_as_unpriced(self) -> None:
        """Nothing meters a call here, and `0.0` would read as free."""
        assert _observe(self._result("valid"), 12).cost_usd is None

    def test_an_error_result_carries_its_code(self) -> None:
        obs = _observe(
            ParseResult(parser_source="llm", error=_err("llm_invalid_json", "parsing")),
            12,
        )
        assert obs.error_code == "llm_invalid_json"
        assert obs.spec is None


def test_the_observation_is_the_shape_the_scorer_reads() -> None:
    """A guard against `_observe` drifting from the type it builds."""
    assert isinstance(_observe(ParseResult(parser_source="llm", spec=_spec()), 1), ParseObservation)
