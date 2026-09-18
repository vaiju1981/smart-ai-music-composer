"""`saimc-preferences` — print what the preference log says about the repair table.

The report is a report: it writes nothing, it retunes nothing, and exit status
stays 0 for any log it can read. Retuning the repair defaults is a change every
user feels, so a person reads this and edits `repairs._REPAIRS` — the same
division `saimc-benchmark` keeps with `MODELS.md`, and the one `gate_session_
publication` keeps with the publish path.

The whole of the reading is `retune.proposal`, which is where the counting rule
and the ranking rule are written down. This module is the two things a CLI owes
on top of that: the log is opened from a path or from the environment, and the
proposal is laid out so a person can see the diff it is asking them to approve.
The current order of every bar is printed beside the proposed one for that
reason — a report that named only the winner would be asking for a decision
about a change the reader cannot see.
"""

from __future__ import annotations

from pathlib import Path

import typer

from saimc.session.deltas import Delta
from saimc.session.preferences import PreferenceLog
from saimc.session.retune import Candidate, Proposal, proposal

app = typer.Typer(
    help="Report what the preference log says about the repair table. Writes nothing."
)


def _order(deltas: tuple[Delta, ...]) -> str:
    """A bar's requests as one line, in the order given."""
    return ", ".join(delta.describe() for delta in deltas)


def _rate(candidate: Candidate) -> str:
    """A candidate's acceptance, or the reason there is no number.

    "no evidence" rather than a percentage, because a request no user has ever
    been served has no rate and `0%` would read as an opinion about it.
    """
    if candidate.acceptance is None:
        return "no evidence"
    return f"{candidate.acceptance:.0%} ({candidate.likes} like, {candidate.dislikes} dislike)"


def _evidence(candidates: tuple[Candidate, ...], *, indent: str) -> list[str]:
    """Candidate lines with their rates in one column.

    The requests are different lengths and the number is what a reader is
    comparing down the block, so the column is what makes the block scannable
    rather than the order alone.
    """
    width = max((len(candidate.delta.describe()) for candidate in candidates), default=0)
    return [
        f"{indent}{candidate.delta.describe().ljust(width)}  {_rate(candidate)}"
        for candidate in candidates
    ]


def _plural(count: int, noun: str) -> str:
    """A count and its noun.

    A report that says "1 judgements" reads like a bug in the tool rather than
    like a log with one opinion in it, and this report is read by a person
    deciding whether to edit a table.
    """
    return f"{count} {noun}" if count == 1 else f"{count} {noun}s"


def _lines(report: Proposal, *, path: Path, exists: bool) -> tuple[str, ...]:
    """The report, as the lines a reader is handed."""
    lines = [f"Log: {path}" + ("" if exists else " (absent: nothing has been recorded)")]
    lines.append(f"{_plural(report.judgements, 'judgement')} over {_plural(report.requests, 'request')}.")
    lines.append("")
    lines.append("Every bar of the repair table, with the evidence the log holds for it.")
    for metric in report.metrics:
        lines.append("")
        lines.append(f"{metric.metric}{'   [reordered]' if metric.moved else ''}")
        lines.append(f"  table:    {_order(metric.tabled)}")
        lines.append(f"  proposed: {_order(metric.proposed)}")
        lines.extend(_evidence(metric.ranked, indent="    "))

    unheld = report.unheld
    lines.append("")
    lines.append(f"Requests a repair served that the table does not hold ({len(unheld)}):")
    lines.extend(_evidence(unheld, indent="    ") or ["    (none)"])

    unproven = report.unproven
    lines.append("")
    lines.append(f"Table entries no judgement carries ({len(unproven)}):")
    lines.extend(f"    {metric}: {delta.describe()}" for metric, delta in unproven)
    if not unproven:
        lines.append("    (none)")

    return tuple(lines)


def _report(log: PreferenceLog) -> Proposal:
    """The proposal for `log`, with a log that cannot be read reported, not traced.

    Two refusals are met here, and both are sentences a person can act on: a line
    that is not a row (`PreferenceLog.rows` names the path and the line) and a
    judgement whose rows disagree about its verdict (`retune.proposal` names the
    draft and the moment), which is what a hand-edited log produces. Both are
    refusals about the log's own contents rather than about this program, so a
    traceback around either would be noise, and the exit status is the only thing
    this adds.
    """
    try:
        return proposal(log.rows())
    except ValueError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def main(
    log: Path | None = typer.Option(
        None, "--log", help="The preferences directory. Defaults to SAIMC_PREFERENCES_DIR."
    ),
) -> None:
    """Report what the preference log says about the repair table, and exit."""
    store = PreferenceLog(log)
    report = _report(store)
    for line in _lines(report, path=store.path, exists=store.path.exists()):
        typer.echo(line)


def _entry() -> None:
    """Console-script entry."""
    app()


if __name__ == "__main__":
    _entry()


__all__ = ["app", "main"]
