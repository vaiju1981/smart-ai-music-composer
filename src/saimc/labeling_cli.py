"""The second-pass labeling tool for the parser benchmark corpus.

`docs/roadmap.md` §10 #3 selects the parser model from a benchmark run, and
`docs/model-fine-tuning.md` requires the corpus to be labeled twice,
independently, with at least 90% pre-adjudication field-level agreement
before the first candidate is scored — "before the first candidate is
scored, not after". No code existed for that second pass; this is it.

Three decisions shape the tool, and each is the reason for something a
reader would otherwise want to change:

1. **It shows the prompt and nothing else.** Showing the primary labeling
   would anchor the reviewer, and the rate would then measure agreement with
   the screen rather than between two readings of the request. `category` is
   withheld for the same reason: it names the answer's shape.
2. **It offers no default for a label field, and every field must be
   answered.** A default makes Enter-through the fast path, and a reviewer
   who accepts every default produces a file that agrees by construction —
   a 100% rate measuring the tool's own skeleton. Typing each answer is the
   price of the bar meaning something.
3. **It writes the review file in the corpus's own record schema** and
   appends one flushed line per record, so the pass can be done in sittings
   and is validated by the corpus's own loader.

The report it prints at the end is a report: exit status stays 0. The gate
that fails a release is a test (`tests/acceptance/test_release_gates.py`),
not this CLI — the same division `saimc-benchmark` keeps with `MODELS.md`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypeVar

import typer
from pydantic import ValidationError

from saimc.benchmark_corpus import (
    DEFAULT_CORPUS_PATH,
    DEFAULT_REVIEW_PATH,
    REJECTION_ERROR_CODES,
    BenchmarkRecord,
    load_corpus,
)
from saimc.labels import FIXED_BY_SCHEMA, compare_labelings
from saimc.spec import SPEC_SCHEMA_VERSION, CompositionSpec, RequestKind

app = typer.Typer(
    help="Label the parser benchmark corpus a second time, independently, and score the agreement."
)

_UNSET = "-"
"""How an optional field is answered. A literal rather than a blank line,
because a blank line is how a reviewer skips something by accident."""


# --- The spec's own schema, as the source of every offered value ------------


def _spec_schema() -> dict[str, Any]:
    """`CompositionSpec`'s JSON schema.

    Every vocabulary and bound this tool offers is read from here rather than
    restated, so the prompts cannot describe a spec the model has moved past.
    """
    return CompositionSpec.model_json_schema()


def _deref(node: Any, defs: Any) -> Any:
    """Follow `$ref`s into `$defs` until the node describes itself.

    Depth-bounded: a self-referential schema would otherwise spin, and this
    tool must not be the thing that hangs on a malformed schema.
    """
    for _ in range(10):
        if not isinstance(node, dict) or "$ref" not in node:
            return node
        node = defs.get(str(node["$ref"]).rsplit("/", 1)[-1])
    return node


def _is_null(node: Any, defs: Any) -> bool:
    resolved = _deref(node, defs)
    return isinstance(resolved, dict) and resolved.get("type") == "null"


def _value_shape(node: Any, defs: Any) -> tuple[Any, bool]:
    """A field's value node, plus whether `null` is also allowed.

    Optional fields are spelled as `anyOf: [{...}, {type: null}]`, so the
    interesting node is the first alternative that is not the null one.
    """
    resolved = _deref(node, defs)
    if isinstance(resolved, dict) and isinstance(resolved.get("anyOf"), list):
        alternatives = [alt for alt in resolved["anyOf"] if isinstance(alt, dict)]
        nullable = any(_is_null(alt, defs) for alt in alternatives)
        for alt in alternatives:
            if not _is_null(alt, defs):
                return _deref(alt, defs), nullable
        return resolved, nullable
    return resolved, False


def _choices(node: Any, defs: Any) -> tuple[str, ...] | None:
    """The closed vocabulary a node names, if it names one.

    Resolves through `anyOf` the way `_value_shape` does, so a nullable enum
    reads the same as a bare one. That is not tidiness: `key` is spelled
    `anyOf: [WesternKey, null]`, and a `_choices` that returned `None` for it
    would offer a menu with no entries and then re-ask every answer for ever —
    a prompt that cannot be satisfied is worse than one that cannot be read.
    """
    resolved = _value_shape(node, defs)[0]
    if not isinstance(resolved, dict):
        return None
    values = resolved.get("enum")
    if isinstance(values, list):
        return tuple(str(value) for value in values)
    if "const" in resolved:
        return (str(resolved["const"]),)
    return None


def _bounds(node: Any) -> tuple[int | None, int | None]:
    """A node's inclusive integer bounds, if it declares them."""
    if not isinstance(node, dict):
        return None, None
    low = node.get("minimum")
    high = node.get("maximum")
    return (
        low if isinstance(low, int) else None,
        high if isinstance(high, int) else None,
    )


def _range_text(low: int | None, high: int | None) -> str:
    if low is not None and high is not None:
        return f"{low}-{high}"
    if low is not None:
        return f">= {low}"
    if high is not None:
        return f"<= {high}"
    return "any integer"


# --- Prompts ----------------------------------------------------------------


_Choice = TypeVar("_Choice", bound=str)


def _ask_choice(
    label: str, values: tuple[_Choice, ...], *, allow_unset: bool
) -> _Choice | None:
    """Ask for one value from a closed vocabulary, numbered.

    Returns `None` only when the vocabulary allows it and `-` was typed. Generic
    over the vocabulary's own element type, so a menu built from a `Literal`
    hands back that `Literal` rather than a `str`: the writer of a record with a
    closed field should not have to assert what its own menu just guaranteed.
    """
    typer.echo(f"\n  {label}:")
    for index, value in enumerate(values, start=1):
        typer.echo(f"    {index:>2}) {value}")
    if allow_unset:
        typer.echo(f"     {_UNSET}) (unset)")
    hint = f"1-{len(values)}" + (f" or {_UNSET}" if allow_unset else "")
    while True:
        raw = typer.prompt(f"  {label} [{hint}]").strip()
        if allow_unset and raw == _UNSET:
            return None
        if raw.isdigit() and 1 <= int(raw) <= len(values):
            return values[int(raw) - 1]
        typer.echo(f"    enter a number from 1 to {len(values)}" + (
            f", or {_UNSET} for unset" if allow_unset else ""
        ))


def _ask_int(label: str, *, low: int | None, high: int | None, allow_unset: bool) -> int | None:
    """Ask for an integer inside a schema-declared range."""
    while True:
        raw = typer.prompt(f"  {label} [{_range_text(low, high)}, or {_UNSET} to leave unset]").strip()
        if allow_unset and raw == _UNSET:
            return None
        if raw.lstrip("-").isdigit():
            value = int(raw)
            if (low is None or value >= low) and (high is None or value <= high):
                return value
        typer.echo(f"    enter an integer in {_range_text(low, high)}")


def _parse_entries(raw: str) -> list[dict[str, str]]:
    """Parse `role=instrument,role=instrument` into spec entries.

    Raises `ValueError` on a chunk that is not a pair, so a typo is re-asked
    rather than silently dropped — the same "refuse, never no-op" rule the
    delta vocabulary holds. An empty answer is the first chunk being empty,
    which is why there is no separate "no entries" guard: it could not be
    reached, and a guard that cannot fail is worse than none.
    """
    entries: list[dict[str, str]] = []
    for chunk in raw.split(","):
        role, separator, instrument = chunk.partition("=")
        if not separator or not role.strip() or not instrument.strip():
            raise ValueError(f"expected role=instrument, got {chunk.strip()!r}")
        entries.append({"role": role.strip(), "instrument": instrument.strip()})
    return entries


def _ask_instrumentation(props: Any, defs: Any) -> list[dict[str, str]]:
    """Ask for `role=instrument` pairs, re-asking a pair that is not one.

    The roles are printed with the legend: there are four, they are closed, and a
    reviewer who must type one should be able to read the ones that exist. The 66
    instruments are not printed — they would be a wall — and a bad one is answered
    by the spec's own refusal, which names it. The vocabulary is read off the
    entry's `role`, not off the array's `items`: the roles belong to the entry, so
    a `_choices` on the array node can only ever return `None`.
    """
    entry = _deref(_deref(props["instrumentation"], defs).get("items", {}), defs)
    roles = _choices(entry.get("properties", {}).get("role", {}), defs) or ()
    typer.echo(
        "\n  instrumentation: comma-separated role=instrument pairs"
        f"\n    roles: {', '.join(roles)}"
        " (roles you omit take the mood's own defaults)"
    )
    while True:
        raw = typer.prompt("  instrumentation").strip()
        try:
            return _parse_entries(raw)
        except ValueError as exc:
            typer.echo(f"    {exc}")


def _ask_rationale() -> str:
    return typer.prompt("  rationale (blank to skip)", default="", show_default=False).strip()


# --- One record -------------------------------------------------------------


def _label_spec() -> dict[str, object]:
    """Ask for every spec field, then validate the result against the spec.

    The two constant fields are written rather than asked: `schema_version` is
    this build's version by definition, and `request_kind` has one legal value
    in Phase 1. Asking would be a prompt with one answer.
    """
    schema = _spec_schema()
    defs = schema.get("$defs", {})
    props = schema.get("properties", {})

    mood_node = props["mood"]
    mood = _ask_choice("mood", _choices(mood_node, defs) or (), allow_unset=False)
    duration_node, _ = _value_shape(props["duration_seconds"], defs)
    tempo_node, tempo_nullable = _value_shape(props["tempo_bpm"], defs)
    key_node, key_nullable = _value_shape(props["key"], defs)
    key_values = _choices(key_node, defs) or ()
    signature = _ask_choice("time_signature", _choices(props["time_signature"], defs) or (), allow_unset=False)
    seed_node, seed_nullable = _value_shape(props["seed"], defs)
    humanization = _ask_choice(
        "humanization", _choices(props["humanization"], defs) or (), allow_unset=False
    )

    payload: dict[str, object] = {
        "schema_version": SPEC_SCHEMA_VERSION,
        "request_kind": RequestKind.MOOD_GENERATION.value,
        "mood": mood,
        "duration_seconds": _ask_int(
            "duration_seconds", low=_bounds(duration_node)[0], high=_bounds(duration_node)[1],
            allow_unset=False,
        ),
        "tempo_bpm": _ask_int(
            "tempo_bpm", low=_bounds(tempo_node)[0], high=_bounds(tempo_node)[1],
            allow_unset=tempo_nullable,
        ),
        "key": _ask_choice("key", key_values, allow_unset=key_nullable),
        "time_signature": signature,
        "instrumentation": _ask_instrumentation(props, defs),
        "seed": _ask_int(
            "seed", low=_bounds(seed_node)[0], high=_bounds(seed_node)[1], allow_unset=seed_nullable
        ),
        "humanization": humanization,
    }
    return CompositionSpec.model_validate(payload).model_dump(mode="json")


def _label_record(record: BenchmarkRecord, *, position: int, total: int) -> BenchmarkRecord:
    """Ask for one record's labels, re-asking the spec if it does not validate.

    The record's own id is *not* shown, though it is written into the file. The
    ids are named for their category — `unsupported_007`, `boundary_012` — so
    printing one tells the reviewer the answer's shape before they have read the
    request, which is the anchoring this tool withholds `category` to avoid. The
    position is what a person needs in order to know how far along they are.
    """
    typer.echo(f"\n[{position}/{total}]")
    typer.echo(f"  prompt: {record.prompt!r}")
    outcome = _ask_choice("expected_outcome", ("accepted", "rejected"), allow_unset=False)
    if outcome == "rejected":
        return BenchmarkRecord(
            id=record.id,
            category=record.category,
            prompt=record.prompt,
            expected_outcome="rejected",
            expected_spec=None,
            expected_error=_ask_choice(
                "expected_error", tuple(sorted(REJECTION_ERROR_CODES)), allow_unset=False
            ),
            label_rationale=_ask_rationale(),
        )
    while True:
        try:
            spec = _label_spec()
        except ValidationError as exc:
            typer.echo("\n  that spec does not validate; answering again:")
            typer.echo(f"    {exc.errors()[0]['msg']} at {exc.errors()[0]['loc']}")
            continue
        return BenchmarkRecord(
            id=record.id,
            category=record.category,
            prompt=record.prompt,
            expected_outcome="accepted",
            expected_spec=spec,
            expected_error=None,
            label_rationale=_ask_rationale(),
        )


# --- The file ---------------------------------------------------------------


def _existing_ids(path: Path) -> set[str]:
    """The ids already in the review file, so the pass resumes rather than restarts."""
    if not path.exists():
        return set()
    records, _ = load_corpus(path)
    return {record.id for record in records}


def _append(path: Path, record: BenchmarkRecord) -> None:
    """Append one labeled record and flush it, so an interrupted pass keeps its work."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(record.model_dump_json() + "\n")
        fh.flush()


def _report(corpus_path: Path, review_path: Path) -> None:
    """Print the agreement between the two passes, or why it cannot be scored."""
    if not review_path.exists():
        typer.echo("\nNo review file yet, so there is nothing to compare.")
        return
    primary, _ = load_corpus(corpus_path)
    review, _ = load_corpus(review_path)
    primary_ids = {record.id for record in primary}
    review_ids = {record.id for record in review}
    missing = sorted(primary_ids - review_ids)
    extra = sorted(review_ids - primary_ids)
    if missing or extra:
        typer.echo(
            f"\n{len(missing)} record(s) not yet reviewed, "
            f"{len(extra)} reviewed but not in the corpus — agreement is not scored yet."
        )
        return
    try:
        report = compare_labelings(primary, review)
    except ValueError as exc:
        typer.echo(f"\nCannot compare the two labelings: {exc}", err=True)
        raise typer.Exit(code=1) from None
    typer.echo(
        f"\nField-level agreement {report.rate:.2%} over {report.field_count} fields "
        f"({report.matched_fields} matched; {report.disagreeing_records} record(s) differ)"
    )
    typer.echo(
        "  by field: "
        + ", ".join(f"{entry.field} {entry.rate:.1%} ({entry.compared})" for entry in report.by_field)
    )
    typer.echo(
        f"  not compared (one value by construction): {', '.join(sorted(FIXED_BY_SCHEMA))}"
    )
    for reason in report.failure_reasons:
        typer.echo(f"  {reason}")
    for disagreement in report.disagreements:
        typer.echo(
            f"  {disagreement.record_id:16} {disagreement.field:20} "
            f"primary={disagreement.primary!r} review={disagreement.review!r}"
        )
    if report.passed:
        typer.echo("  The two labelings agree; commit the review file with the corpus.")


@app.command()
def main(
    corpus: Path | None = typer.Option(
        None, "--corpus", exists=True, readable=True, help="Path to parser_benchmark.jsonl."
    ),
    review: Path | None = typer.Option(
        None, "--review", help="Where the second labeling is written and read from."
    ),
    report_only: bool = typer.Option(
        False, "--report-only", help="Print the agreement and exit without asking anything."
    ),
) -> None:
    """Label the corpus a second time, one record at a time, resumably."""
    corpus_path = corpus if corpus is not None else DEFAULT_CORPUS_PATH
    review_path = review if review is not None else DEFAULT_REVIEW_PATH

    records, stats = load_corpus(corpus_path)
    done = _existing_ids(review_path)
    pending = [record for record in records if record.id not in done]
    # Only claim a write when there is going to be one. `--report-only` is the
    # mode that is safe to run anywhere precisely because it writes nothing,
    # and a run with nothing pending writes nothing either.
    writing = not report_only and bool(pending)
    typer.echo(
        f"{stats.total} records; {len(done)} already labeled; {len(pending)} to go. "
        + (f"Writing {review_path}." if writing else f"Review file: {review_path}.")
    )

    if not report_only and pending:
        for index, record in enumerate(pending, start=1):
            labeled = _label_record(record, position=len(done) + index, total=stats.total)
            _append(review_path, labeled)
            typer.echo(f"  wrote {labeled.id}")

    _report(corpus_path, review_path)


def _entry() -> None:
    """Console-script entry."""
    app()


if __name__ == "__main__":
    _entry()


__all__ = ["app", "main"]
