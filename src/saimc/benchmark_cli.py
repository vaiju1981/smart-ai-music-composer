"""Phase 1 model-selection benchmark CLI.

Per `docs/roadmap.md` §10 #3:

1. Query the configured Ollama host for currently available text models.
2. Discard any model that fails the license/acceptable-use gate (operator-supplied
   allow-list or per-model `MODELS.md` review).
3. Run the 100-prompt benchmark corpus against each surviving model.
4. The least expensive model that meets every §8 quality and latency threshold
   becomes `OLLAMA_MODEL`; ties are broken by lower p95 latency.
5. Record the model identifier, service date, benchmark results, license,
   terms URL, review date, and structured-output capability result in
   `MODELS.md`.

This CLI is the operator-facing entry point. It is intentionally read-only
with respect to `MODELS.md` (it prints a candidate summary; committing the
final entry to `MODELS.md` is a manual review step per the same §10 #3 gate).

Real-Ollama execution requires `OLLAMA_BASE_URL`, `OLLAMA_MODEL`, and
`OLLAMA_API_KEY` to be set. The `--fallback-only` mode runs the benchmark
against the deterministic fallback parser only — useful in CI and for
verifying the runner without a live Ollama host.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import typer

from saimc.benchmark import (
    BenchmarkReport,
    ParseObservation,
    score_corpus,
)
from saimc.benchmark_corpus import BenchmarkRecord, load_corpus
from saimc.llm.fallback import parse_fallback
from saimc.spec import CompositionSpec

app = typer.Typer(
    help="Run the Phase 1 parser benchmark against an Ollama host or the fallback parser."
)


def _run_fallback_only(records: list[BenchmarkRecord], client_label: str) -> BenchmarkReport:
    """Run the corpus against the deterministic fallback parser only."""
    observations: list[ParseObservation] = []
    for rec in records:
        t0 = time.perf_counter()
        out = parse_fallback(rec.prompt)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        if isinstance(out, CompositionSpec):
            observations.append(
                ParseObservation(
                    spec=out,
                    error_code=None,
                    parser_source="fallback",
                    attempts=0,
                    structured_output=False,
                    latency_ms=elapsed_ms,
                    raw_first_response_valid=True,
                    cost_usd=0.0,
                )
            )
        else:
            observations.append(
                ParseObservation(
                    spec=None,
                    error_code=out.error_code,
                    parser_source="fallback",
                    attempts=0,
                    structured_output=False,
                    latency_ms=elapsed_ms,
                    raw_first_response_valid=True,
                    cost_usd=0.0,
                )
            )
    return score_corpus(client_label, records, observations)


async def _run_with_client(
    records: list[BenchmarkRecord],
    client_label: str,
    client: object = None,
) -> BenchmarkReport:
    """Run the corpus against an LLM client.

    Live-Ollama execution is a release-gate task and intentionally not
    implemented yet — wire the real adapter here once an Ollama Cloud
    endpoint and API key are configured for development. The signature
    takes `client: object` so the function signature is stable while the
    body is stubbed.
    """
    raise NotImplementedError(
        "Live-Ollama execution is a release-gate task and intentionally "
        "not implemented yet. Use --fallback-only for offline benchmarking."
    )


@app.command()
def main(
    corpus: Path = typer.Option(
        ..., "--corpus", exists=True, readable=True, help="Path to parser_benchmark.jsonl"
    ),
    fallback_only: bool = typer.Option(
        False, "--fallback-only", help="Run against the deterministic fallback parser only."
    ),
    client_label: str = typer.Option(
        "fallback", "--label", help="Label written into the report (model name for LLM runs)."
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write report JSON to this path."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Per-record progress."),
) -> None:
    """Run the parser benchmark and emit a report."""
    # The corpus path argument is here so future live runs can pass a
    # different corpus (e.g. a model-specific eval set). The default
    # loader already points at the canonical one.
    del corpus  # the loader uses the canonical path; parameter reserved for future use

    records, stats = load_corpus()
    typer.echo(
        f"Loaded {stats.total} records "
        f"(accepted={stats.accepted_expected}, rejected={stats.rejected_expected})"
    )

    if fallback_only:
        report = _run_fallback_only(records, client_label)
    else:
        try:
            report = asyncio.run(_run_with_client(records, client_label))
        except NotImplementedError as exc:
            typer.echo(f"FAILED: {exc}", err=True)
            raise typer.Exit(code=1) from None

    typer.echo(json.dumps({k: v for k, v in report.to_dict().items() if k != "outcomes"}, indent=2))
    if verbose:
        for o in report.outcomes:
            typer.echo(
                f"{o.record_id:20} {o.category:20} expected={o.expected_outcome:8} "
                f"actual={o.actual_outcome:8} matched={o.matched} "
                f"field_exact={o.field_exact_match} error={o.error_code}"
            )

    if output is not None:
        output.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        typer.echo(f"Wrote {output}")

    if not report.passed:
        typer.echo(f"FAILED: {'; '.join(report.failure_reasons)}", err=True)
        raise typer.Exit(code=1)


def _entry() -> None:
    """Console-script entry."""
    app()


if __name__ == "__main__":
    _entry()


__all__ = ["app", "main"]
