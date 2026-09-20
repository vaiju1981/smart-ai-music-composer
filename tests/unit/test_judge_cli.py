"""`saimc-judge`, driven through typer the way the console script runs it.

The runner goes through `app()` rather than calling `main` directly, for
`pyproject.toml`'s reason: calling the decorated function hands the body the
raw `typer.Option(...)` defaults and proves nothing about the entry point.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

import saimc.judge_cli as judge_cli
from saimc.judge_cli import app, build_grid
from saimc.llm.base import ChatRequest, ChatResult

runner = CliRunner()


class _Listener:
    """A judge that always prefers whichever piece it was shown second."""

    model_identifier = "scripted-listener"

    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, request: ChatRequest) -> ChatResult:
        self.calls += 1
        return ChatResult(content='{"prefers": "second", "reason": "it goes somewhere"}')

    async def aclose(self) -> None:
        return None


class TestTheGrid:
    def test_every_cell_is_composed_and_measured(self) -> None:
        pieces = build_grid(["calming"], [30], [1, 2])
        assert [p.label for p in pieces] == ["calming:30s:seed1", "calming:30s:seed2"]
        # The measurement rides with the piece, taken from the same notation
        # the judge will be shown.
        assert all(p.quality.as_dict()["step_ratio"] is not None for p in pieces)

    def test_a_seed_composes_the_same_piece_twice(self) -> None:
        first = build_grid(["calming"], [30], [7])[0]
        second = build_grid(["calming"], [30], [7])[0]
        assert first.notation.compute_hash() == second.notation.compute_hash()


class TestDryRun:
    def test_it_prints_the_grid_and_asks_nothing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _no_model(**_: Any) -> None:
            raise AssertionError("a dry run must not build an adapter")

        monkeypatch.setattr(judge_cli, "build_ollama_adapter", _no_model)
        result = runner.invoke(
            app,
            ["--dry-run", "--mood", "calming", "--duration", "30", "--seed", "1", "--seed", "2"],
        )
        assert result.exit_code == 0, result.output
        assert "Composed 2 pieces into 1 pairs" in result.output
        assert "scorecard prefers:" in result.output
        assert "Melody:" in result.output

    def test_it_refuses_to_report_an_agreement_rate_it_did_not_measure(self) -> None:
        result = runner.invoke(
            app, ["--dry-run", "--mood", "sleep", "--duration", "30", "--seed", "1", "--seed", "2"]
        )
        assert "no agreement rate, because nothing was asked" in result.output
        assert "agreement_rate" not in result.output


class TestALiveRun:
    def test_no_configuration_fails_with_the_thing_to_set(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(judge_cli, "build_ollama_adapter", lambda **_: None)
        result = runner.invoke(
            app, ["--mood", "calming", "--duration", "30", "--seed", "1", "--seed", "2"]
        )
        assert result.exit_code == 1
        assert "OLLAMA_MODEL" in result.output
        assert "--dry-run" in result.output

    def test_a_sweep_reports_the_two_rates_and_the_model_it_asked(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        listener = _Listener()
        monkeypatch.setattr(judge_cli, "build_ollama_adapter", lambda **_: listener)
        out = tmp_path / "judged.json"
        result = runner.invoke(
            app,
            ["--mood", "calming", "--duration", "30", "--seed", "1", "--seed", "2", "-o", str(out)],
        )
        assert result.exit_code == 1, result.output
        # Two calls for the one pair: forward, then swapped.
        assert listener.calls == 2
        assert '"model": "scripted-listener"' in result.output
        assert out.exists()

    def test_a_judge_that_only_ever_says_second_is_reported_as_no_evidence(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Always-second is position bias, and the run fails rather than scoring it.

        The report has no quality bar to fail — there is no measured floor yet
        — but a judge that never survived a swap measured itself, and a green
        exit over that would be the run claiming something it did not find.
        """
        monkeypatch.setattr(judge_cli, "build_ollama_adapter", lambda **_: _Listener())
        result = runner.invoke(
            app, ["--mood", "calming", "--duration", "30", "--seed", "1", "--seed", "2"]
        )
        assert result.exit_code == 1
        assert "position bias" in result.output
