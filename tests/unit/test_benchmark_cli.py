"""Smoke tests for the benchmark CLI (offline path)."""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from saimc.benchmark_cli import app

runner = CliRunner()


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


def test_cli_live_mode_is_stubbed(tmp_path: Path) -> None:
    """Live-Ollama execution is a release-gate task and not implemented yet.

    The CLI must surface the `NotImplementedError` message rather than
    silently producing a report.
    """
    out = tmp_path / "report.json"
    result = runner.invoke(
        app,
        ["--label", "anything", "--output", str(out)],
    )
    assert result.exit_code == 1
    combined = (result.stdout + (result.stderr or "")).lower()
    assert "not implemented" in combined
