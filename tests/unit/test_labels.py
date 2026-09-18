"""Unit tests for the second-labeling agreement measure.

`compare_labelings` decides whether §10 #3's precondition is met, which is the
one thing standing between the corpus and the first candidate model scored
against it — so every test here is written to fail when the *measure* moves,
not merely when the code stops running.
"""

from __future__ import annotations

import pytest

from saimc.benchmark_corpus import (
    DEFAULT_CORPUS_PATH,
    REJECTION_ERROR_CODES,
    BenchmarkRecord,
    ExpectedErrorCode,
    load_corpus,
)
from saimc.labels import (
    FIXED_BY_SCHEMA,
    MIN_FIELD_AGREEMENT,
    FieldAgreement,
    compare_labelings,
)
from saimc.spec import SPEC_SCHEMA_VERSION, CompositionSpec, RequestKind

_PROMPT = "5 min of calming piano music"


def _spec(**overrides: object) -> dict[str, object]:
    """A valid spec's wire form, built through the spec rather than beside it."""
    payload: dict[str, object] = {
        "mood": "calming",
        "duration_seconds": 300,
        "instrumentation": [{"role": "melody", "instrument": "piano"}],
    }
    payload.update(overrides)
    return CompositionSpec.model_validate(payload).model_dump(mode="json")


def _accepted(record_id: str, prompt: str = _PROMPT, **spec: object) -> BenchmarkRecord:
    return BenchmarkRecord(
        id=record_id,
        category="supported_paraphrase",
        prompt=prompt,
        expected_outcome="accepted",
        expected_spec=_spec(**spec),
        expected_error=None,
        label_rationale="reads as calming and 5 min",
    )


def _rejected(
    record_id: str,
    prompt: str = "write me a saxophone solo",
    error: ExpectedErrorCode = "out_of_vocabulary",
) -> BenchmarkRecord:
    return BenchmarkRecord(
        id=record_id,
        category="unsupported",
        prompt=prompt,
        expected_outcome="rejected",
        expected_spec=None,
        expected_error=error,
        label_rationale="no mood to map it to",
    )


def _mutated(record: BenchmarkRecord, **spec: object) -> BenchmarkRecord:
    """The same record with one or more spec keys changed, as a reviewer's would be."""
    assert record.expected_spec is not None
    return record.model_copy(
        update={"expected_spec": {**record.expected_spec, **spec}}
    )


class TestTheComparableFields:
    """Which fields the rate is taken over, and which it deliberately is not."""

    def test_a_labeling_agrees_with_itself(self) -> None:
        first = _accepted("a", mood="calming")
        second = _accepted("b", duration_seconds=420, mood="sleep")
        refused = _rejected("c")
        report = compare_labelings([first, second, refused], [first, second, refused])

        assert report.passed
        assert report.rate == 1.0
        assert report.disagreements == ()
        assert report.record_count == 3
        # Two accepted records at (outcome, error, 8 spec keys) and one rejected
        # at (outcome, error, whole spec), with the two fixed keys left out.
        assert report.field_count == 2 * 10 + 3
        assert report.matched_fields == report.field_count

    def test_the_fixed_keys_are_absent_from_the_breakdown(self) -> None:
        record = _accepted("a")
        report = compare_labelings([record], [record])
        compared = {entry.field for entry in report.by_field}

        assert compared & FIXED_BY_SCHEMA == set()
        assert "mood" in compared
        assert "expected_outcome" in compared

    def test_the_keys_the_prompts_fix_are_still_compared(self) -> None:
        """A one-valued key is only excluded when the *schema* fixes it.

        `time_signature`, `humanization` and `key` are one value across the whole
        corpus, and they are decisions — what is constant there is a property of
        these prompts, which is the thing a second reading tests. Excluding them
        would be excluding the measurement, so a reviewer who disagrees on one
        has to move the rate.
        """
        record = _accepted("a")
        report = compare_labelings([record], [_mutated(record, time_signature="3/4")])

        assert [d.field for d in report.disagreements] == ["time_signature"]
        assert report.rate == pytest.approx(0.9)

    def test_the_two_fixed_keys_admit_one_value_each(self) -> None:
        """The premise behind the exclusion, asserted rather than believed.

        `request_kind`'s field offers one value. `schema_version`'s offers
        three, which is *why* the drift refusal exists beside the exclusion —
        and why the corpus is checked to hold the one this build writes.
        """
        schema = CompositionSpec.model_json_schema()
        properties = schema["properties"]

        assert properties["request_kind"]["enum"] == [RequestKind.MOOD_GENERATION.value]
        assert properties["schema_version"]["enum"] != [SPEC_SCHEMA_VERSION]

        records, _ = load_corpus(DEFAULT_CORPUS_PATH)
        written = {
            record.expected_spec["schema_version"]
            for record in records
            if record.expected_spec is not None
        }
        assert written == {SPEC_SCHEMA_VERSION}

    def test_every_rejected_record_names_a_code_the_parser_can_emit(self) -> None:
        """The vocabulary is closed, so a typo cannot enter the corpus as a label."""
        records, _ = load_corpus(DEFAULT_CORPUS_PATH)
        rejected = [record for record in records if record.expected_outcome == "rejected"]

        assert rejected
        assert {record.expected_error for record in rejected} <= REJECTION_ERROR_CODES
        assert all(record.expected_spec is None for record in rejected)
        assert all(record.expected_error is None for record in records if record.expected_spec)

    def test_a_spec_compared_whole_is_one_field_and_not_ten(self) -> None:
        """The branch where one side is a rejection: one absence, not ten misses."""
        accepted = _accepted("a")
        report = compare_labelings(
            [accepted], [_rejected("a", prompt=accepted.prompt)]
        )

        fields = {entry.field for entry in report.by_field}
        assert "expected_spec" in fields
        assert "mood" not in fields
        assert {d.field for d in report.disagreements} == {
            "expected_outcome",
            "expected_error",
            "expected_spec",
        }


class TestTheRate:
    """The number itself, and the two ways it can be wrong."""

    def test_one_mutated_field_moves_the_rate_by_exactly_one_field(self) -> None:
        records = [_accepted(letter) for letter in "abc"]
        report = compare_labelings(records, records)
        mutated = [_mutated(records[0], mood="sleep"), *records[1:]]
        moved = compare_labelings(records, mutated)

        assert len(moved.disagreements) == 1
        assert moved.matched_fields == report.matched_fields - 1
        assert moved.field_count == report.field_count
        assert moved.rate == pytest.approx(
            (report.matched_fields - 1) / report.field_count
        )
        assert moved.disagreeing_records == 1

    def test_wrongness_on_every_decision_fails_however_the_padding_is_counted(self) -> None:
        """The sabotage the exclusion exists for, fired at the real corpus.

        Every accepted record's `mood` and `duration_seconds` — the two keys the
        prompts vary most — read as *a different value from the one it carries*,
        so all 120 of those fields disagree. Under a rate taken over all ten
        spec keys that is 120 wrong of 840; with the two the writer fixes left
        out it is 120 of 720. Both fail, and the *breakdown* is what says which
        two fields did it, which is the difference between a number and a
        diagnosis.
        """
        records, _ = load_corpus(DEFAULT_CORPUS_PATH)
        mutated = [
            _mutated(
                record,
                mood=next(m for m in ("calming", "sleep", "electrifying") if m != record.expected_spec["mood"]),
                duration_seconds=600 if record.expected_spec["duration_seconds"] != 600 else 30,
            )
            if record.expected_spec is not None
            else record
            for record in records
        ]
        report = compare_labelings(records, mutated)

        assert not report.passed
        assert report.rate < MIN_FIELD_AGREEMENT
        assert report.field_count == 720
        assert len(report.disagreements) == 2 * 60
        assert [entry.field for entry in report.by_field if entry.matched < entry.compared] == [
            "duration_seconds",
            "mood",
        ]
        assert report.failure_reasons

    def test_the_breakdown_is_the_total_read_another_way(self) -> None:
        """One quantity, two readings — so the two cannot disagree."""
        records, _ = load_corpus(DEFAULT_CORPUS_PATH)
        report = compare_labelings(records, records)

        assert sum(entry.compared for entry in report.by_field) == report.field_count
        assert sum(entry.matched for entry in report.by_field) == report.matched_fields
        assert report.field_count == 720
        assert report.rate == 1.0

    def test_each_breakdown_entry_reports_its_own_rate(self) -> None:
        entry = FieldAgreement(field="mood", compared=8, matched=6)
        assert entry.rate == pytest.approx(0.75)
        assert FieldAgreement(field="mood", compared=0, matched=0).rate == 0.0
        assert entry.to_dict()["field"] == "mood"

    def test_the_breakdown_reads_the_record_before_the_spec(self) -> None:
        """The order is declared, not the order the fields were first met in.

        A single-outcome corpus cannot tell the two apart: a file whose records
        all carry a spec inserts the spec's keys first and never reaches the
        whole-document branch, and a file whose records are all rejections
        inserts the three record fields in the declared order by accident. The
        corpus has both, so the two readings disagree from the third entry on —
        which is the only place this test can catch a breakdown that fell back
        on dict insertion.
        """
        records, _ = load_corpus(DEFAULT_CORPUS_PATH)
        report = compare_labelings(records, records)

        assert [entry.field for entry in report.by_field][:3] == [
            "expected_outcome",
            "expected_error",
            "expected_spec",
        ]
        assert compare_labelings([_rejected("a")], [_rejected("a")]).by_field[0].field == (
            "expected_outcome"
        )

    def test_an_empty_comparison_fails_rather_than_passing_at_zero(self) -> None:
        report = compare_labelings([], [])

        assert not report.passed
        assert report.rate == 0.0
        assert "no comparable label fields" in report.failure_reasons[0]

    def test_the_bar_is_met_at_exactly_the_minimum(self) -> None:
        """§8 says "at least 90%", so 0.90 passes and a hair above it does not."""
        record = _accepted("a")
        mutated = [_mutated(record, mood="sleep")]  # 9 of 10 fields agree
        assert compare_labelings([record], mutated).rate == pytest.approx(MIN_FIELD_AGREEMENT)
        assert compare_labelings([record], mutated).passed is True
        assert compare_labelings([record], mutated, minimum=0.95).passed is False


class TestDrift:
    """Differences that are not disagreements about a label, and must not be scored."""

    def test_the_two_files_must_cover_the_same_records(self) -> None:
        with pytest.raises(ValueError, match="different records"):
            compare_labelings([_accepted("a")], [_accepted("b")])

    def test_a_missing_record_is_named_in_the_refusal(self) -> None:
        with pytest.raises(ValueError, match="labeled but not reviewed"):
            compare_labelings([_accepted("a"), _accepted("b")], [_accepted("a")])

    def test_a_mismatch_of_many_records_is_truncated_rather_than_dumped(self) -> None:
        """A refusal has to stay diagnosable without becoming a listing."""
        many = [_accepted(letter) for letter in "abcdefg"]

        with pytest.raises(ValueError, match=r"6 reviewed but not labeled \(b, c, d, e, f, …"):
            compare_labelings([_accepted("a")], many)

    def test_a_reworded_prompt_is_drift_and_not_a_disagreement(self) -> None:
        with pytest.raises(ValueError, match="corpus drift"):
            compare_labelings([_accepted("a")], [_accepted("a", prompt="different")])

    def test_a_duplicate_id_is_refused_in_the_file_that_repeats_it(self) -> None:
        with pytest.raises(ValueError, match="the review labeling lists a more than once"):
            compare_labelings([_accepted("a")], [_accepted("a"), _accepted("a")])

    def test_a_spec_version_difference_is_refused_rather_than_scored(self) -> None:
        """A version bump is a corpus version, not a reader's mistake."""
        record = _accepted("a")
        older = record.model_copy(
            update={"expected_spec": {**record.expected_spec, "schema_version": 2}}
        )

        with pytest.raises(ValueError, match="more than one spec version"):
            compare_labelings([record], [older])

    def test_a_version_that_is_not_a_number_is_reported_rather_than_coerced(self) -> None:
        """The value comes off a file, so `"3"` is a version this cannot place."""
        record = _accepted("a")
        text = record.model_copy(
            update={"expected_spec": {**record.expected_spec, "schema_version": "3"}}
        )

        with pytest.raises(ValueError, match="more than one spec version"):
            compare_labelings([record], [text])

    def test_two_rejections_agree_without_naming_a_version(self) -> None:
        """No spec means no version to compare, which is not a mismatch."""
        assert compare_labelings([_rejected("a")], [_rejected("a")]).passed

    def test_the_whole_report_serialises(self) -> None:
        records, _ = load_corpus(DEFAULT_CORPUS_PATH)
        payload = compare_labelings(records, records).to_dict()

        assert payload["passed"] is True
        assert payload["rate"] == 1.0
        assert payload["disagreeing_records"] == 0
        assert len(payload["by_field"]) == 11  # outcome, error, spec, 8 keys
        assert set(payload) == {
            "record_count",
            "field_count",
            "matched_fields",
            "rate",
            "disagreeing_records",
            "passed",
            "failure_reasons",
            "by_field",
            "disagreements",
        }
