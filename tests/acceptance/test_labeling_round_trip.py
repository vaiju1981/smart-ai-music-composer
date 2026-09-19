"""The second-labeling tool, run end to end over the real corpus.

What this proves, and what it deliberately does not — the difference is the
whole reason the file is written this way.

It proves the **tool** works. Nothing here is mocked: the real console-script
`app` runs over the real 100-record fixture, the answers are typed through the
wizard's own prompts, the file comes back through the corpus's own loader, and
`compare_labelings` scores it. So a wizard that offered the wrong menu, wrote a
record the schema rejects, dropped a record, or produced a file the comparison
refuses to read fails here.

It does **not** discharge `docs/model-fine-tuning.md`'s second-labeling
precondition. The keystrokes are generated *from the primary labels*, so this
review file is the primary labeling re-encoded and its agreement is 100% by
construction. §10 #3 needs a second labeling reached **independently** of the
first, and `labeling_cli`'s own docstring names the failure mode this would be —
a file that agrees "by construction — a 100% rate measuring the tool's own
skeleton". That is why the review file lands in a tmp dir and is never
committed, and why no release gate is asserted against it: that gate belongs to
the owner's labeling pass, over a file a person produced.

So the weight falls on the tests that stop this from being a tautology shaped
like an assertion. A round trip that can only agree proves nothing, and "the tool
coproduced its own input" is exactly the shape the harness is for: the rate has
to move when a single label moves, by exactly one field, and it has to fall below
the bar when enough of them do. Both are checked against the same file, so the
100% above is a measurement rather than a fixed point.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import pytest
from typer.testing import CliRunner

from saimc.benchmark_corpus import (
    REJECTION_ERROR_CODES,
    BenchmarkRecord,
    load_corpus,
)
from saimc.labeling_cli import _choices, app
from saimc.labels import MIN_FIELD_AGREEMENT, compare_labelings
from saimc.spec import CompositionSpec, Mood, TimeSignature, WesternKey

_CORPUS = Path(__file__).resolve().parents[1] / "fixtures" / "parser_benchmark.jsonl"

runner = CliRunner()

_SCHEMA = CompositionSpec.model_json_schema()
_PROPERTIES = _SCHEMA["properties"]
_DEFS = _SCHEMA["$defs"]

# Must match `_label_record`'s own menu, which is this pair in this order.
_OUTCOMES = ("accepted", "rejected")

_UNSET = "-"


# --- Typing the answers -----------------------------------------------------


def _menu(field: str) -> tuple[str, ...]:
    """The vocabulary the wizard offers for a spec field, in the order it prints it."""
    return _choices(_PROPERTIES[field], _DEFS) or ()


def _error_menu() -> tuple[str, ...]:
    """`_label_record` offers the rejection codes sorted, so this does too."""
    return tuple(sorted(REJECTION_ERROR_CODES))


def _pick(field: str, value: str) -> str:
    """The keystroke selecting `value` from `field`'s menu, 1-based as the wizard prints it."""
    return str(_menu(field).index(value) + 1)


def _keys_for(record: BenchmarkRecord) -> str:
    """The keystrokes a reviewer types to record this record's own labels.

    Every menu position is *looked up* rather than written down, so a vocabulary
    that gains or loses a value moves this with it instead of silently
    mis-indexing every answer after the one that changed. That is the same rule
    the wizard itself follows in reading its menus off the schema.
    """
    answers = [str(_OUTCOMES.index(record.expected_outcome) + 1)]
    if record.expected_outcome == "rejected":
        code = record.expected_error
        assert code is not None, "a rejected record without an error code is not a record"
        answers.append(str(_error_menu().index(code) + 1))
    else:
        spec = record.expected_spec
        assert spec is not None, "an accepted record without a spec is not a record"
        answers.append(_pick("mood", spec["mood"]))
        answers.append(_pick("time_signature", spec["time_signature"]))
        answers.append(_pick("humanization", spec["humanization"]))
        answers.append(str(spec["duration_seconds"]))
        answers.append(_UNSET if spec["tempo_bpm"] is None else str(spec["tempo_bpm"]))
        answers.append(_UNSET if spec["key"] is None else _pick("key", spec["key"]))
        # The wizard parses `role=instrument,role=instrument`, and the order it
        # is given is the order it writes, so the record's own order is kept.
        answers.append(",".join(f"{e['role']}={e['instrument']}" for e in spec["instrumentation"]))
        answers.append(_UNSET if spec["seed"] is None else str(spec["seed"]))
    # The rationale is not a compared field, so a blank answer costs nothing.
    return "\n".join(answers) + "\n\n"


# --- Retouching a labeling, the way a disagreement would --------------------


def _retouch(record: BenchmarkRecord, **spec_changes: object) -> BenchmarkRecord:
    """The record with spec fields changed, validated as the corpus validates it."""
    payload = record.model_dump()
    spec = dict(payload["expected_spec"] or {})
    spec.update(spec_changes)
    payload["expected_spec"] = spec
    return BenchmarkRecord.model_validate(payload)


def _relabel_error(record: BenchmarkRecord, code: str) -> BenchmarkRecord:
    payload = record.model_dump()
    payload["expected_error"] = code
    return BenchmarkRecord.model_validate(payload)


def _other_error_code(code: str) -> str:
    return next(candidate for candidate in _error_menu() if candidate != code)


# --- Fixtures ---------------------------------------------------------------


@pytest.fixture(scope="module")
def primary() -> list[BenchmarkRecord]:
    records, stats = load_corpus(_CORPUS)
    assert stats.total == len(records), "the loader's count and its records disagree"
    return list(records)


@pytest.fixture(scope="module")
def review(primary: list[BenchmarkRecord], tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Drive the real wizard over the real corpus once, and hand back the file it wrote.

    This is the end-to-end step: no prompt is monkeypatched and no answer is
    short-circuited, so the fixture fails loudly if the wizard cannot be driven
    from beginning to end through its own interface.
    """
    path = tmp_path_factory.mktemp("second-labeling") / "parser_benchmark_review.jsonl"
    typed = "".join(_keys_for(record) for record in primary)
    result = runner.invoke(app, ["--corpus", str(_CORPUS), "--review", str(path)], input=typed)
    assert result.exit_code == 0, f"the wizard exited {result.exit_code}:\n{result.output}"
    return path


@pytest.fixture(scope="module")
def reviewed(review: Path) -> list[BenchmarkRecord]:
    records, _ = load_corpus(review)
    return list(records)


# --- The round trip ---------------------------------------------------------


class TestTheWizardLabelsTheWholeCorpus:
    """The file it writes is the corpus's schema, and it covers the corpus."""

    def test_the_review_file_loads_as_a_corpus(
        self, review: Path, reviewed: list[BenchmarkRecord]
    ) -> None:
        """`load_corpus` is the corpus's own validator, so this is the schema check."""
        assert review.exists(), "the wizard claimed to write a file it did not write"
        assert len(reviewed) == 100, f"expected 100 labeled records, got {len(reviewed)}"

    def test_the_review_covers_exactly_the_corpuss_records(
        self, primary: list[BenchmarkRecord], reviewed: list[BenchmarkRecord]
    ) -> None:
        """Asserted here so a mismatch reads as this failure and not a comparison error.

        `compare_labelings` raises on a mismatched id set, so without this premise
        the drift guard would report the symptom from one step away.
        """
        assert [r.id for r in reviewed] == [r.id for r in primary]

    def test_the_outcomes_are_split_as_the_corpus_is(
        self, reviewed: list[BenchmarkRecord]
    ) -> None:
        """The generated answers really are read off the labels, not emitted flat."""
        outcomes = [r.expected_outcome for r in reviewed]
        counts = {"accepted": outcomes.count("accepted"), "rejected": outcomes.count("rejected")}
        assert counts == {"accepted": 60, "rejected": 40}, counts


class TestTheTwoLabelingsAgree:
    """Scored by the same function the release gate will call."""

    def test_the_agreement_is_complete(
        self, primary: list[BenchmarkRecord], reviewed: list[BenchmarkRecord]
    ) -> None:
        report = compare_labelings(primary, reviewed)
        assert report.matched_fields == report.field_count, report.disagreements
        assert report.rate == 1.0, report.disagreements
        assert report.passed is True, report.failure_reasons

    def test_the_agreement_is_scored_over_every_compared_field(
        self, primary: list[BenchmarkRecord], reviewed: list[BenchmarkRecord]
    ) -> None:
        """A rate of 1.0 over three fields would be a true statement about nothing."""
        report = compare_labelings(primary, reviewed)
        assert report.field_count > 0
        assert report.record_count == 100


# --- The teeth --------------------------------------------------------------


class TestTheComparisonCanDisagree:
    """A round trip that could only agree would make the three tests above vacuous."""

    def test_one_mutated_field_costs_exactly_one_field_of_agreement(
        self, primary: list[BenchmarkRecord], reviewed: list[BenchmarkRecord]
    ) -> None:
        """The `1 - 1/n` shape, computed from the report rather than assumed."""
        clean = compare_labelings(primary, reviewed)
        target = next(r for r in reviewed if r.expected_outcome == "accepted")
        duration = target.expected_spec["duration_seconds"]  # type: ignore[index]
        mutated = [_retouch(r, duration_seconds=duration + 1) if r.id == target.id else r
                   for r in reviewed]

        report = compare_labelings(primary, mutated)
        assert report.field_count == clean.field_count, "a mutation changed the field count"
        assert report.matched_fields == clean.matched_fields - 1
        assert report.rate == pytest.approx(1.0 - 1.0 / report.field_count)
        assert report.disagreeing_records == 1
        assert report.passed is True, "one field of 620 should not fail a 90% bar"

    def test_a_rejection_that_disagrees_is_caught_as_its_own_field(
        self, primary: list[BenchmarkRecord], reviewed: list[BenchmarkRecord]
    ) -> None:
        """`expected_error` is one compared field, and it is the whole of a rejection."""
        target = next(r for r in reviewed if r.expected_outcome == "rejected")
        code = target.expected_error or ""
        mutated = [_relabel_error(r, _other_error_code(code)) if r.id == target.id else r
                   for r in reviewed]

        report = compare_labelings(primary, mutated)
        assert report.disagreeing_records == 1
        assert [d.field for d in report.disagreements] == ["expected_error"]
        assert report.matched_fields == report.field_count - 1

    def test_a_labeling_wrong_about_enough_fields_fails_the_bar(
        self, primary: list[BenchmarkRecord], reviewed: list[BenchmarkRecord]
    ) -> None:
        """The bar has to be reachable, or `MIN_FIELD_AGREEMENT` is decoration."""
        mutated = [
            _retouch(r, duration_seconds=r.expected_spec["duration_seconds"] + 1)  # type: ignore[index]
            if r.expected_outcome == "accepted"
            else _relabel_error(r, _other_error_code(r.expected_error or ""))
            for r in reviewed
        ]

        report = compare_labelings(primary, mutated)
        assert report.rate < MIN_FIELD_AGREEMENT, f"the mutation left the rate at {report.rate}"
        assert report.passed is False
        assert report.failure_reasons, "a failed comparison has to say why"
        assert report.disagreeing_records == 100


class TestTheMenusAreTheDeclaredVocabulary:
    """The cancellation the generator would otherwise hide.

    `_keys_for` picks each answer by looking its value up in the wizard's own
    menu, which is the right way to type a review file and the wrong way to
    *test* one: a `_choices` that sorted, reversed or dropped a value would move
    the menu and the keystroke together, so every answer would still land on the
    value it names and the round trip would report 100%. These cases read the
    vocabulary from the document that declares it instead — the enum, and the
    literal — so the two sides can disagree. Fired as a sabotage on `_choices`
    (a `sorted(...)` in its return), the pin fails while the round trip stays
    green, which is exactly the hole it exists to close. The mood case is the
    exception in that firing, because its three values happen to sort into their
    declaration order; the other two carry it.
    """

    @pytest.mark.parametrize(
        ("field", "enum"),
        [("mood", Mood), ("time_signature", TimeSignature), ("key", WesternKey)],
    )
    def test_a_closed_menu_is_its_enum_in_declaration_order(
        self, field: str, enum: type[StrEnum]
    ) -> None:
        assert _menu(field) == tuple(member.value for member in enum)

    def test_the_literal_menu_is_declared_in_this_order(self) -> None:
        """`humanization` is a `Literal` in `spec.py`, so the order is the literal's.

        Written out rather than derived, as the order pins elsewhere in this repo
        are: it cannot be read from the annotation without re-implementing the
        schema generator, and a pin that re-derived its subject would reintroduce
        the cancellation these tests exist to break.
        """
        assert _menu("humanization") == ("none", "light", "expressive")

    def test_the_error_menu_is_the_sorted_codes(self) -> None:
        """The wizard sorts them, so this restates that rule rather than the wizard's output."""
        assert _error_menu() == tuple(sorted(REJECTION_ERROR_CODES))


class TestTheTypedAnswersFollowTheLabels:
    """The generator's own premise: the keystrokes are a function of the record."""

    def test_a_changed_label_changes_the_keystrokes(self, primary: list[BenchmarkRecord]) -> None:
        record = next(r for r in primary if r.expected_outcome == "accepted")
        other_mood = next(m for m in _menu("mood") if m != record.expected_spec["mood"])  # type: ignore[index]
        assert _keys_for(_retouch(record, mood=other_mood)) != _keys_for(record)

    def test_different_moods_are_typed_as_different_menu_picks(
        self, primary: list[BenchmarkRecord]
    ) -> None:
        """A generator that always typed the same number would pass every test above."""
        moods = _menu("mood")
        picks = [
            _pick("mood", mood)
            for mood in moods
            if any(r.expected_spec and r.expected_spec["mood"] == mood for r in primary)
        ]
        assert len(picks) > 1, "the corpus uses one mood, so this test asserts nothing"
        assert len(set(picks)) == len(picks), "two moods share a keystroke"

    def test_an_accepted_record_is_asked_for_a_spec_and_a_rejected_one_is_not(
        self, primary: list[BenchmarkRecord]
    ) -> None:
        accepted = next(r for r in primary if r.expected_outcome == "accepted")
        rejected = next(r for r in primary if r.expected_outcome == "rejected")
        assert len(_keys_for(accepted).splitlines()) > len(_keys_for(rejected).splitlines())
