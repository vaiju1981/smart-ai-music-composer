"""Musical-quality scorecard CLI.

Composes a matrix of specs, measures each piece with `saimc.quality`, and
prints the corpus report — optionally as JSON, so the output can feed a
repair loop, a tuning sweep, or a dashboard.

The default matrix spans the moods at a fixed duration so a regression in
the generator shows up as a moved number rather than a subjective
complaint. Pass `--spec` to measure a specific payload instead.

Exit code is 1 when the corpus misses a threshold, so this is usable as a
CI step or a pre-release check once the generator clears the bars; today
it exits 1, which is the honest report.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from saimc.compose.engine import compose
from saimc.quality import (
    QUALITY_THRESHOLDS,
    PieceQuality,
    QualityReport,
    QualityThreshold,
    score_corpus,
    score_piece,
)
from saimc.spec import CompositionSpec

app = typer.Typer(
    help="Score the musical quality of generated pieces and report what to fix."
)

# The default matrix: every mood, one scalar and one full ensemble, at a
# fixed duration and seed so runs are comparable.
DEFAULT_MATRIX: tuple[dict[str, object], ...] = (
    {"mood": "calming", "instrumentation": "piano", "seed": 1},
    {"mood": "electrifying", "instrumentation": "piano", "seed": 2},
    {"mood": "sleep", "instrumentation": "piano", "seed": 3},
    {
        "mood": "electrifying",
        "instrumentation": [
            {"role": "melody", "instrument": "violin"},
            {"role": "harmony", "instrument": "strings"},
            {"role": "bass", "instrument": "contrabass"},
            {"role": "percussion", "instrument": "drum_set"},
        ],
        "seed": 4,
    },
)

DEFAULT_DURATION_SECONDS = 180


def _default_specs() -> list[CompositionSpec]:
    return [
        CompositionSpec.model_validate({"duration_seconds": DEFAULT_DURATION_SECONDS, **payload})
        for payload in DEFAULT_MATRIX
    ]


def _load_spec(path: Path) -> CompositionSpec:
    """Load one spec from a JSON file (a raw spec, or a job's input_spec block)."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "input_spec" in payload:
        payload = payload["input_spec"]["spec"]  # a manifest.json
    return CompositionSpec.model_validate(payload)


def _measure(specs: list[CompositionSpec]) -> list[PieceQuality]:
    return [
        score_piece(compose(spec).notation_score, piece=_label(index, spec))
        for index, spec in enumerate(specs)
    ]


def _label(index: int, spec: CompositionSpec) -> str:
    """A stable, readable name for one matrix entry."""
    roles = "+".join(entry.instrument.value for entry in spec.instrumentation)
    return f"{index}:{spec.mood.value}:{spec.duration_seconds}s:{roles}"


@app.command()
def main(
    spec: Path | None = typer.Option(
        None, "--spec", exists=True, readable=True, help="Measure one spec JSON instead of the matrix."
    ),
    label: str = typer.Option("matrix", "--label", help="Label written into the report."),
    output: Path | None = typer.Option(
        None, "--output", "-o", help="Write the report JSON to this path."
    ),
    json_out: bool = typer.Option(False, "--json", help="Print the report as JSON."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Per-piece metrics."),
) -> None:
    """Measure the matrix and report every threshold the music misses."""
    specs = [_load_spec(spec)] if spec is not None else _default_specs()
    pieces = _measure(specs)
    report = score_corpus(label, pieces)

    if json_out:
        typer.echo(json.dumps(report.to_dict(), indent=2))
    else:
        _print_report(report)

    if verbose:
        for piece in pieces:
            typer.echo(f"\n{piece.piece}")
            for name, value in piece.as_dict().items():
                shown = "n/a" if value is None else f"{value:.3f}"
                typer.echo(f"  {name:32} {shown}")

    if output is not None:
        output.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        typer.echo(f"Wrote {output}")

    if not report.passed:
        raise typer.Exit(code=1)


def _print_report(report: QualityReport) -> None:
    typer.echo(f"Scored {report.piece_count} piece(s) against {len(QUALITY_THRESHOLDS)} thresholds")
    typer.echo("")
    typer.echo(f"{'metric':34} {'mean':>10}   bar")
    for threshold in QUALITY_THRESHOLDS:
        value = report.metrics.get(threshold.metric)
        shown = "n/a" if value is None else f"{value:.3f}"
        typer.echo(f"{threshold.metric:34} {shown:>10}   {_bar_text(threshold)}")
    if report.failure_reasons:
        typer.echo("")
        typer.echo("Missed:")
        for reason in report.failure_reasons:
            typer.echo(f"  {reason}")


def _bar_text(threshold: QualityThreshold) -> str:
    """The threshold as a readable bar: one-sided, or a band."""
    if threshold.minimum is not None and threshold.maximum is not None:
        return f"in [{threshold.minimum:.2f}, {threshold.maximum:.2f}]"
    if threshold.minimum is not None:
        return f">= {threshold.minimum:.2f}"
    assert threshold.maximum is not None  # the table guarantees a direction
    return f"<= {threshold.maximum:.2f}"


def _entry() -> None:
    """Console-script entry."""
    app()


if __name__ == "__main__":
    _entry()


__all__ = ["app", "main"]
