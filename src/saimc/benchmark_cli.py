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

A live run reads its host and model from the same configuration the app uses
(`saimc.toml` / `OLLAMA_*` environment / built-in defaults, resolved by
`saimc.llm.config`), so the sweep measures the host the product would actually
talk to. `--model` selects a different tag for one run, which is §10 #3 step 3
("run the corpus against each surviving model"); `--fallback-only` runs the
deterministic fallback parser alone, needs no host, and is what CI takes.

The corpus hash is printed with every run. `docs/parser-benchmark.md` requires
it recorded beside a score so the score can be invalidated when the corpus
moves, and this is the reader `corpus_sha256` was added for.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from saimc.benchmark import (
    BenchmarkReport,
    ParseObservation,
    score_corpus,
)
from saimc.benchmark_corpus import (
    DEFAULT_CORPUS_PATH,
    BenchmarkRecord,
    corpus_sha256,
    load_corpus,
)
from saimc.llm.config import build_ollama_adapter
from saimc.llm.fallback import parse_fallback
from saimc.parser import parse_prompt
from saimc.spec import CompositionSpec

if TYPE_CHECKING:
    from saimc.llm.base import LLMClient, ParseResult

app = typer.Typer(
    help="Run the Phase 1 parser benchmark against an Ollama host or the fallback parser."
)


def _run_fallback_only(records: list[BenchmarkRecord], client_label: str) -> BenchmarkReport:
    """Run the corpus against the deterministic fallback parser only.

    Every observation reports `raw_first_response_valid=None`, not `True`: this
    path asks no model, so there is no first response to have been valid, and a
    hardcoded `True` here is what made the only runnable benchmark report a
    first-pass validity of 1.0 for a run that called nothing. `cost_usd=0.0` is
    a genuine zero by contrast — no call was made, so there is a price and it is
    zero, which is not the same statement as `None`.
    """
    observations: list[ParseObservation] = []
    for rec in records:
        t0 = time.perf_counter()
        out = parse_fallback(rec.prompt)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        spec: CompositionSpec | None
        error_code: str | None
        if isinstance(out, CompositionSpec):
            spec, error_code = out, None
        else:
            spec, error_code = None, out.error_code
        observations.append(
            ParseObservation(
                spec=spec,
                error_code=error_code,
                parser_source="fallback",
                attempts=0,
                structured_output=False,
                latency_ms=elapsed_ms,
                raw_first_response_valid=None,
                cost_usd=0.0,
            )
        )
    return score_corpus(client_label, records, observations)


def _observe(result: ParseResult, latency_ms: int) -> ParseObservation:
    """One observation from one parsed record.

    `first_attempt`'s three states map straight through: `"valid" -> True`,
    `"invalid" -> False`, and `None -> None`. That last arm is the reason the
    field has a third state at all — a run against an unreachable host reports
    *not measured* rather than 0%, because no model output was read and §8's bar
    is about a model's first response.

    `cost_usd` is `None`: nothing in this codebase meters or prices a call, and
    the configured dev host is a local proxy to a cloud provider whose inference
    is billed to an account the app cannot read. See `ParseObservation.cost_usd`.
    """
    return ParseObservation(
        spec=result.spec,
        error_code=result.error.error_code if result.error is not None else None,
        parser_source=result.parser_source,
        attempts=result.attempts,
        structured_output=result.structured_output,
        latency_ms=latency_ms,
        raw_first_response_valid=(
            None if result.first_attempt is None else result.first_attempt == "valid"
        ),
        cost_usd=None,
    )


async def _run_with_client(
    records: list[BenchmarkRecord],
    client_label: str,
    client: LLMClient,
) -> BenchmarkReport:
    """Sweep the corpus against `client`, and close it when the sweep ends.

    It takes ownership of the client it is handed: the caller builds one adapter
    for the whole corpus, and a protocol whose implementations hold an HTTP
    client and cannot be closed is the leak `aclose` exists to prevent. One
    coroutine closes it on its own event loop rather than a second `asyncio.run`
    doing it on a fresh one.

    **The latency here is measured runner-side, around the whole `parse_prompt`
    call** — so a prompt the model took three attempts to answer is measured as
    the slower request it is. The adapter's own per-call `latency_ms` stays a
    diagnostic in `ParseResult.extra` and is deliberately not what §8's p95 bar
    reads, because that bar is about the request the user waits for.
    """
    observations: list[ParseObservation] = []
    try:
        for rec in records:
            t0 = time.perf_counter()
            result = await parse_prompt(client, rec.prompt, f"benchmark:{rec.id}")
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            observations.append(_observe(result, elapsed_ms))
    finally:
        await client.aclose()
    return score_corpus(client_label, records, observations)


@app.command()
def main(
    corpus: Path | None = typer.Option(
        None,
        "--corpus",
        exists=True,
        readable=True,
        help="Corpus to sweep. Defaults to the canonical tests/fixtures/parser_benchmark.jsonl.",
    ),
    fallback_only: bool = typer.Option(
        False, "--fallback-only", help="Run against the deterministic fallback parser only."
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="Model tag to sweep. Defaults to the configured one.",
    ),
    client_label: str | None = typer.Option(
        None,
        "--label",
        help="Label written into the report. Defaults to the model tag the host served.",
    ),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write report JSON to this path."
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Per-record progress."),
) -> None:
    """Run the parser benchmark and emit a report."""
    corpus_path = corpus if corpus is not None else DEFAULT_CORPUS_PATH
    records, stats = load_corpus(corpus_path)
    typer.echo(
        f"Loaded {stats.total} records "
        f"(accepted={stats.accepted_expected}, rejected={stats.rejected_expected}) "
        f"from {corpus_path}"
    )
    typer.echo(f"corpus_sha256 {corpus_sha256(corpus_path)}")

    if fallback_only:
        report = _run_fallback_only(records, client_label or "fallback")
    else:
        client = build_ollama_adapter(model=model)
        if client is None:
            typer.echo(
                "FAILED: no LLM configuration, so there is no model to sweep. Set "
                "OLLAMA_MODEL (or a saimc.toml), or use --fallback-only to run offline.",
                err=True,
            )
            raise typer.Exit(code=1)
        # The label defaults to the tag the adapter actually resolved, read
        # back off the client rather than re-derived from configuration: the
        # identifier §10 #3 requires recorded is the one this host served, and
        # a proxy's tag namespace need not be the one that was asked for.
        report = asyncio.run(
            _run_with_client(records, client_label or client.model_identifier, client)
        )

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
