"""Run the listening judge over a grid the engine composes, and report the gap.

`saimc-benchmark` scores the parser and `saimc-quality` scores the composition
against its own thirteen bars. This is the third reading and the only one that
comes from outside: it composes a grid, pairs the pieces, asks a model which of
each pair is the better music without showing it a single measurement, and
reports how often that answer and the arbiter's order are the same answer.

    saimc-judge --dry-run                 # compose, pair, print — no model
    saimc-judge --model qwen3:8b -o out.json

`--dry-run` is the mode CI can take: it exercises the grid, the pairing and the
scorecard's own order, which is everything except the part that needs a host.
The agreement rate is the one number it cannot produce, and it says so rather
than printing a zero that would read as total disagreement.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import TYPE_CHECKING

import typer

from saimc.compose.engine import compose
from saimc.judge import (
    JudgedPiece,
    JudgeReport,
    Pairing,
    describe_piece,
    judge_corpus,
    pairs_from,
)
from saimc.llm.config import build_ollama_adapter
from saimc.quality import score_piece
from saimc.spec import CompositionSpec, Mood

if TYPE_CHECKING:  # pragma: no cover
    pass

app = typer.Typer(add_completion=False, help=__doc__)

DEFAULT_DURATIONS: tuple[int, ...] = (30, 120)
DEFAULT_SEEDS: tuple[int, ...] = (1, 2, 3, 4)
"""The grid's default axes.

Small on purpose: every cell is a real composition and every *pair* is two
model calls, so a default that swept the whole mood x duration x seed space
would be an expensive accident for anyone who typed the command to see what it
did. `--durations` and `--seeds` widen it deliberately.
"""


def build_grid(moods: list[str], durations: list[int], seeds: list[int]) -> list[JudgedPiece]:
    """Compose the grid, measuring each piece as it lands.

    The measurement is taken here and carried on the piece rather than looked
    up later, so the number the report compares against is the one taken from
    the same notation the judge read — not a recomposition that a later engine
    change could make differ.
    """
    pieces: list[JudgedPiece] = []
    for mood in moods:
        for duration in durations:
            for seed in seeds:
                spec = CompositionSpec(mood=Mood(mood), duration_seconds=duration, seed=seed)
                out = compose(spec)
                pieces.append(
                    JudgedPiece(
                        label=f"{mood}:{duration}s:seed{seed}",
                        notation=out.notation_score,
                        quality=score_piece(out.notation_score),
                    )
                )
    return pieces


def _print_dry_run(pairings: list[Pairing]) -> None:
    for pairing in pairings:
        typer.echo(f"\n=== {pairing.pair_id} ===")
        typer.echo(f"scorecard prefers: {pairing.scorecard_prefers()}")
        typer.echo(f"\n--- {pairing.left.label} ---\n{describe_piece(pairing.left.notation)}")
        typer.echo(f"\n--- {pairing.right.label} ---\n{describe_piece(pairing.right.notation)}")


@app.command()
def main(
    moods: list[str] = typer.Option(
        [m.value for m in Mood], "--mood", help="Moods to sweep. Repeatable."
    ),
    durations: list[int] = typer.Option(
        list(DEFAULT_DURATIONS), "--duration", help="Durations in seconds. Repeatable."
    ),
    seeds: list[int] = typer.Option(list(DEFAULT_SEEDS), "--seed", help="Seeds. Repeatable."),
    pair_seed: int = typer.Option(0, "--pair-seed", help="Seed for the pairing shuffle."),
    max_pairs: int = typer.Option(0, "--max-pairs", help="Cap the pairs judged. 0 is no cap."),
    model: str | None = typer.Option(
        None, "--model", help="Model tag. Defaults to the configured one."
    ),
    label: str | None = typer.Option(None, "--label", help="Label written into the report."),
    output: Path | None = typer.Option(None, "--output", "-o", help="Write report JSON here."),
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Compose and pair, print what a judge would read, call nothing."
    ),
) -> None:
    """Compose a grid, pair it, and measure the listener against the scorecard."""
    pieces = build_grid(moods, durations, seeds)
    pairings = pairs_from(pieces, seed=pair_seed)
    if max_pairs > 0:
        pairings = pairings[:max_pairs]
    typer.echo(f"Composed {len(pieces)} pieces into {len(pairings)} pairs")

    if dry_run:
        _print_dry_run(pairings)
        typer.echo(
            "\nDry run: no agreement rate, because nothing was asked. "
            "Give --model (or configure one) to measure it."
        )
        return

    client = build_ollama_adapter(model=model)
    if client is None:
        typer.echo(
            "FAILED: no LLM configuration, so there is no listener to ask. Set "
            "OLLAMA_MODEL (or a saimc.toml), or use --dry-run to see the grid.",
            err=True,
        )
        raise typer.Exit(code=1)

    served = getattr(client, "model_identifier", "unknown")
    try:
        report: JudgeReport = asyncio.run(
            judge_corpus(client, pairings, label=label or served, model=served)
        )
    finally:
        asyncio.run(client.aclose())

    typer.echo(json.dumps(report.to_dict(), indent=2))
    if output is not None:
        output.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
        typer.echo(f"Wrote {output}")

    # No exit code on the rate, because there is no bar to fail. `judge.py`
    # says why: nothing has measured what a reasonable agreement rate is, and
    # a threshold invented before the first sweep would be one chosen to pass.
    # A judge that could not answer at all is a different matter.
    if report.pairs and report.consistent_pairs == 0:
        typer.echo(
            "FAILED: the judge was never consistent across a swapped pair, so it "
            "measured its own position bias and nothing about the music.",
            err=True,
        )
        raise typer.Exit(code=1)


def _entry() -> None:
    """Console-script entry."""
    app()


if __name__ == "__main__":
    _entry()


__all__ = ["app", "build_grid", "main"]
