"""Tests for the second-labeling wizard.

The tool's job is to produce a file that is *evidence* about the prompts, so the
tests that matter are the ones that would let it produce evidence about itself:
a defaulted field, an anchored reviewer, or a label that never got validated.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from saimc.benchmark_corpus import BenchmarkRecord, load_corpus
from saimc.labeling_cli import (
    _bounds,
    _choices,
    _deref,
    _parse_entries,
    _range_text,
    _value_shape,
    app,
)
from saimc.spec import SPEC_SCHEMA_VERSION, CompositionSpec

runner = CliRunner()

_CORPUS = Path(__file__).resolve().parents[1] / "fixtures" / "parser_benchmark.jsonl"

# The wizard asks, in this order: the outcome; then the spec's mood, meter and
# humanization; then its duration, tempo, key, instrumentation and seed; then a
# rationale. Two of those are nullable, spelled `-`. The trailing blank line is
# the rationale's answer, and it is not optional: a prompt with no input left
# aborts rather than defaulting, and an abort after every answer still loses the
# record.
_ACCEPTED_ANSWERS = (
    "\n".join(
        [
            "1",  # expected_outcome -> accepted
            "1",  # mood -> calming
            "1",  # time_signature -> 4/4
            "2",  # humanization -> light
            "300",  # duration_seconds
            "-",  # tempo_bpm -> unset
            "-",  # key -> unset
            "melody=piano",  # instrumentation
            "-",  # seed -> unset
        ]
    )
    + "\n\n"  # rationale, blank
)

_REJECTED_ANSWERS = (
    "\n".join(
        [
            "2",  # expected_outcome -> rejected
            "2",  # expected_error -> out_of_vocabulary
        ]
    )
    + "\n\n"  # rationale, blank
)


def _one_record_corpus(tmp_path: Path, record: BenchmarkRecord) -> Path:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text(record.model_dump_json() + "\n", encoding="utf-8")
    return corpus


def _record(**overrides: object) -> BenchmarkRecord:
    payload: dict[str, object] = {
        "id": "unsupported_007",
        "category": "unsupported",
        "prompt": "write me a saxophone solo",
        "expected_outcome": "rejected",
        "expected_spec": None,
        "expected_error": "out_of_vocabulary",
        "label_rationale": "no mood to map a saxophone solo to",
    }
    payload.update(overrides)
    return BenchmarkRecord.model_validate(payload)


def _accepted_record(**overrides: object) -> BenchmarkRecord:
    spec = CompositionSpec.model_validate(
        {
            "mood": "sleep",
            "duration_seconds": 420,
            "instrumentation": [{"role": "melody", "instrument": "music_box"}],
        }
    ).model_dump(mode="json")
    payload: dict[str, object] = {
        "id": "boundary_012",
        "category": "boundary",
        "prompt": "seven minutes of sleep music",
        "expected_outcome": "accepted",
        "expected_spec": spec,
        "expected_error": None,
        "label_rationale": "sleep plus seven minutes",
    }
    payload.update(overrides)
    return BenchmarkRecord.model_validate(payload)


class TestTheSchemaMenus:
    """Every offered value is read from the spec, so the wizard cannot drift."""

    def test_a_nullable_enum_offers_its_values_and_not_an_empty_menu(self) -> None:
        """`key` is `anyOf: [WesternKey, null]`, and an empty menu re-asks for ever."""
        properties = CompositionSpec.model_json_schema()["properties"]
        defs = CompositionSpec.model_json_schema()["$defs"]

        keys = _choices(properties["key"], defs)
        assert keys is not None
        assert "C" in keys
        assert "Bbm" in keys
        assert _value_shape(properties["key"], defs)[1] is True

    def test_a_bare_vocabulary_is_read_through_its_reference(self) -> None:
        schema = CompositionSpec.model_json_schema()
        moods = _choices(schema["properties"]["mood"], schema["$defs"])

        assert moods == ("calming", "electrifying", "sleep")

    def test_a_field_with_no_vocabulary_offers_none(self) -> None:
        schema = CompositionSpec.model_json_schema()
        assert _choices(schema["properties"]["seed"], schema["$defs"]) is None

    def test_the_bounds_and_their_wording_come_from_the_schema(self) -> None:
        schema = CompositionSpec.model_json_schema()
        defs = schema["$defs"]
        duration, _ = _value_shape(schema["properties"]["duration_seconds"], defs)
        seed, _ = _value_shape(schema["properties"]["seed"], defs)

        assert _bounds(duration) == (30, 600)
        assert _bounds(seed) == (0, None)
        assert _range_text(30, 600) == "30-600"
        assert _range_text(0, None) == ">= 0"
        assert _range_text(None, 240) == "<= 240"
        assert _range_text(None, None) == "any integer"

    def test_a_reference_chain_is_followed_and_a_cycle_does_not_spin(self) -> None:
        assert _deref({"$ref": "#/$defs/Mood"}, {"Mood": {"type": "string"}}) == {
            "type": "string"
        }
        looping = {"$ref": "#/$defs/A"}
        assert _deref(looping, {"A": looping}) is looping

    def test_the_role_vocabulary_lives_on_the_entry_and_not_the_array(self) -> None:
        """The premise the legend's roles are read through, asserted.

        The array's `items` node is an object schema, so a `_choices` on it can
        only ever return `None` — which is what a legend that read the roles from
        there would print, while saying nothing about it.
        """
        schema = CompositionSpec.model_json_schema()
        props, defs = schema["properties"], schema["$defs"]
        item = _deref(props["instrumentation"], defs).get("items", {})

        assert _choices(item, defs) is None
        assert _choices(_deref(item, defs)["properties"]["role"], defs) == (
            "melody",
            "harmony",
            "bass",
            "percussion",
        )

    def test_the_spellings_this_spec_does_not_use_are_still_read(self) -> None:
        """`const`, a null-first alternative, and a node that is not a mapping.

        The spec spells its nullable fields null-last and its one `const` field
        with an `enum` beside it, so none of these branches is reached by *this*
        spec — they are exercised against synthetic nodes the way the
        reference-chain test is, because a reader of the schema that cannot read
        a spelling the schema may use is a reader that fails silently.
        """
        assert _choices({"const": "mood_generation"}, {}) == ("mood_generation",)
        # A null alternative written first: the loop skips it and keeps looking.
        assert _value_shape({"anyOf": [{"type": "null"}, {"enum": ["calming"]}]}, {}) == (
            {"enum": ["calming"]},
            True,
        )
        # Nothing but nulls: still nullable, and the node is its own value shape.
        assert _value_shape({"anyOf": [{"type": "null"}]}, {}) == (
            {"anyOf": [{"type": "null"}]},
            True,
        )
        # A boolean schema names no vocabulary, and `_bounds` has no mapping to read.
        assert _choices(True, {}) is None
        assert _choices({"type": "string"}, {}) is None
        assert _bounds("not a node") == (None, None)

    def test_an_ensemble_entry_that_is_not_a_pair_is_refused(self) -> None:
        assert _parse_entries("melody=piano, bass=cello") == [
            {"role": "melody", "instrument": "piano"},
            {"role": "bass", "instrument": "cello"},
        ]
        with pytest.raises(ValueError, match="role=instrument"):
            _parse_entries("piano")
        with pytest.raises(ValueError, match="role=instrument"):
            _parse_entries("melody=")

    def test_an_empty_ensemble_is_refused_as_its_first_chunk(self) -> None:
        """There is no separate "no entries" path: this is the only way in."""
        with pytest.raises(ValueError, match="expected role=instrument, got ''"):
            _parse_entries("")


class TestTheWizard:
    """One record, start to finish, through the same surface a person uses."""

    def test_an_accepted_record_is_written_in_the_corpus_schema(self, tmp_path: Path) -> None:
        corpus = _one_record_corpus(tmp_path, _accepted_record())
        review = tmp_path / "review.jsonl"

        result = runner.invoke(
            app, ["--corpus", str(corpus), "--review", str(review)], input=_ACCEPTED_ANSWERS
        )

        assert result.exit_code == 0, result.output
        records, stats = load_corpus(review)
        assert stats.total == 1
        written = records[0]
        # The id, the prompt and the category are copied, never asked for: the
        # prompt is the input the label is about, and the other two are what
        # joins the two files.
        assert written.id == "boundary_012"
        assert written.category == "boundary"
        assert written.prompt == "seven minutes of sleep music"
        assert written.expected_outcome == "accepted"
        assert written.expected_error is None
        assert written.expected_spec is not None
        assert written.expected_spec["mood"] == "calming"
        assert written.expected_spec["duration_seconds"] == 300
        assert written.expected_spec["schema_version"] == SPEC_SCHEMA_VERSION
        assert written.expected_spec["request_kind"] == "mood_generation"
        assert written.expected_spec["tempo_bpm"] is None
        assert written.expected_spec["instrumentation"][0] == {
            "role": "melody",
            "instrument": "piano",
        }
        # The legend names the roles the reviewer has to choose one of, read off
        # the entry rather than the array — a legend reading the array prints
        # nothing, which is what it did before this assertion existed.
        assert "roles: melody, harmony, bass, percussion" in result.output

    def test_a_rejected_record_carries_a_code_from_the_closed_vocabulary(
        self, tmp_path: Path
    ) -> None:
        corpus = _one_record_corpus(tmp_path, _record())
        review = tmp_path / "review.jsonl"

        runner.invoke(
            app, ["--corpus", str(corpus), "--review", str(review)], input=_REJECTED_ANSWERS
        )

        written = load_corpus(review)[0][0]
        assert written.expected_outcome == "rejected"
        assert written.expected_spec is None
        assert written.expected_error == "out_of_vocabulary"

    def test_the_reviewer_never_sees_the_first_labeling(self, tmp_path: Path) -> None:
        """No id, no category, no rationale — the rate must measure two readings.

        The ids *are* named for their category (`unsupported_007`), so showing
        one would anchor the reviewer exactly as showing the label would. This
        is the guard on the whole gate: if the second pass can see the first,
        the agreement is between the reviewer and the screen. The claim is about
        what is shown *while asking* — the confirmation line afterwards names
        the id it appended, which is after the answer and cannot anchor it.
        """
        record = _record()
        corpus = _one_record_corpus(tmp_path, record)

        result = runner.invoke(
            app,
            ["--corpus", str(corpus), "--review", str(tmp_path / "review.jsonl")],
            input=_REJECTED_ANSWERS,
        )

        asked = result.output.split("  wrote ")[0]
        assert record.prompt in asked
        # The three strings that are *only* reachable from the primary labeling,
        # checked against what was shown while asking. `id` and `category` are
        # one leak rather than two — the ids carry the category as their prefix,
        # which is exactly why the id is not printed — so both are asserted
        # where it is shown. `expected_error` is deliberately not among them:
        # the codes are the menu the tool offers, so "the code appears" would
        # assert nothing, and the label being one of its members is the whole of
        # what a rejection means.
        assert record.id not in asked
        assert record.category not in asked
        assert record.label_rationale not in asked

    def test_a_resumed_pass_skips_what_it_already_labeled(self, tmp_path: Path) -> None:
        first, second = _record(id="unsupported_007"), _record(id="unsupported_008")
        corpus = tmp_path / "corpus.jsonl"
        corpus.write_text(
            first.model_dump_json() + "\n" + second.model_dump_json() + "\n", encoding="utf-8"
        )
        review = tmp_path / "review.jsonl"

        runner.invoke(
            app, ["--corpus", str(corpus), "--review", str(review)], input=_REJECTED_ANSWERS
        )
        result = runner.invoke(
            app, ["--corpus", str(corpus), "--review", str(review)], input=_REJECTED_ANSWERS
        )

        assert "1 already labeled; 1 to go" in result.output
        records, _ = load_corpus(review)
        assert [record.id for record in records] == ["unsupported_007", "unsupported_008"]

    def test_a_spec_that_does_not_validate_is_asked_again(self, tmp_path: Path) -> None:
        """Out-of-range input is re-asked, not written as a corrupt record."""
        corpus = _one_record_corpus(tmp_path, _accepted_record())
        review = tmp_path / "review.jsonl"
        # A duration of 5 is below the schema's minimum: the wizard re-asks.
        answers = _ACCEPTED_ANSWERS.replace("\n300\n", "\n5\n300\n")

        result = runner.invoke(
            app, ["--corpus", str(corpus), "--review", str(review)], input=answers
        )

        assert result.exit_code == 0, result.output
        assert "enter an integer in 30-600" in result.output
        assert load_corpus(review)[0][0].expected_spec["duration_seconds"] == 300

    def test_an_ensemble_typo_is_re_asked_rather_than_dropped(self, tmp_path: Path) -> None:
        corpus = _one_record_corpus(tmp_path, _accepted_record())
        answers = _ACCEPTED_ANSWERS.replace("melody=piano", "piano\nmelody=piano")

        result = runner.invoke(
            app,
            ["--corpus", str(corpus), "--review", str(tmp_path / "review.jsonl")],
            input=answers,
        )

        assert "expected role=instrument" in result.output
        assert load_corpus(tmp_path / "review.jsonl")[0][0].expected_spec is not None

    def test_an_answer_outside_a_menu_or_a_range_is_re_asked(self, tmp_path: Path) -> None:
        """Both loops' guards, which are what make the menus closed.

        `9` is outside the mood's three entries and `forty` is not an integer at
        all; each is answered by the loop that asked, so a record cannot carry a
        value the menu never offered.
        """
        corpus = _one_record_corpus(tmp_path, _accepted_record())
        answers = _ACCEPTED_ANSWERS.replace(
            "1\n1\n1\n2\n300\n", "1\n9\n1\n1\n2\nforty\n300\n"
        )

        result = runner.invoke(
            app, ["--corpus", str(corpus), "--review", str(tmp_path / "review.jsonl")],
            input=answers,
        )

        assert result.exit_code == 0, result.output
        assert "enter a number from 1 to 3" in result.output
        assert "enter an integer in 30-600" in result.output
        written = load_corpus(tmp_path / "review.jsonl")[0][0]
        assert written.expected_spec is not None
        assert written.expected_spec["mood"] == "calming"
        assert written.expected_spec["duration_seconds"] == 300

    def test_a_role_outside_the_vocabulary_is_re_asked_by_the_spec(self, tmp_path: Path) -> None:
        """`_parse_entries` checks a pair's shape, not its names.

        An unknown role gets past the parser and is refused by `CompositionSpec`,
        so the whole spec is asked again rather than a record being written that
        the corpus's own loader would refuse. The refusal is printed with its
        location, which is what tells the reviewer *which* field to answer.
        """
        corpus = _one_record_corpus(tmp_path, _accepted_record())
        first_pass = "1\n1\n1\n2\n300\n-\n-\nbanjo=piano\n-\n"

        result = runner.invoke(
            app,
            ["--corpus", str(corpus), "--review", str(tmp_path / "review.jsonl")],
            input=first_pass + _ACCEPTED_ANSWERS,
        )

        assert result.exit_code == 0, result.output
        assert "at ('instrumentation', 0, 'role')" in result.output
        written = load_corpus(tmp_path / "review.jsonl")[0][0]
        assert written.expected_spec is not None
        # The leading entry is what the reviewer typed, which is the whole claim:
        # the spec expands a partial ensemble from the mood's own defaults, so the
        # list beyond it is the mood's and asserting on it would assert nothing
        # about the second pass.
        assert written.expected_spec["instrumentation"][0] == {
            "role": "melody",
            "instrument": "piano",
        }


class TestTheReport:
    """What the tool says when it is not asking anything."""

    def test_a_pass_with_no_review_file_says_so(self, tmp_path: Path) -> None:
        result = runner.invoke(
            app,
            [
                "--corpus",
                str(_CORPUS),
                "--review",
                str(tmp_path / "absent.jsonl"),
                "--report-only",
            ],
        )

        assert result.exit_code == 0
        assert "No review file yet" in result.output

    def test_a_partial_pass_is_not_scored(self, tmp_path: Path) -> None:
        """Scoring a half-finished pass would report the reviewer's progress."""
        review = tmp_path / "review.jsonl"
        review.write_text(_record().model_dump_json() + "\n", encoding="utf-8")

        result = runner.invoke(
            app, ["--corpus", str(_CORPUS), "--review", str(review), "--report-only"]
        )

        assert result.exit_code == 0
        assert "not yet reviewed" in result.output

    def test_a_complete_labeling_scores_against_the_corpus(self, tmp_path: Path) -> None:
        """The round trip: the corpus relabeled as itself agrees at 100%."""
        review = tmp_path / "review.jsonl"
        records, _ = load_corpus(_CORPUS)
        review.write_text(
            "".join(record.model_dump_json() + "\n" for record in records), encoding="utf-8"
        )

        result = runner.invoke(
            app, ["--corpus", str(_CORPUS), "--review", str(review), "--report-only"]
        )

        assert result.exit_code == 0, result.output
        assert "Field-level agreement 100.00% over 720 fields" in result.output
        assert "not compared (one value by construction): request_kind, schema_version" in (
            result.output
        )
        assert "commit the review file with the corpus" in result.output

    def test_a_disagreeing_labeling_is_reported_with_its_fields(self, tmp_path: Path) -> None:
        """Wrong on two of the varying keys, so the bar is missed and it says why."""
        records, _ = load_corpus(_CORPUS)
        mutated = [
            record.model_copy(
                update={
                    "expected_spec": {
                        **record.expected_spec,
                        "mood": next(
                            m
                            for m in ("calming", "sleep", "electrifying")
                            if m != record.expected_spec["mood"]
                        ),
                        "duration_seconds": 600
                        if record.expected_spec["duration_seconds"] != 600
                        else 30,
                    }
                }
            )
            if record.expected_spec is not None
            else record
            for record in records
        ]
        review = tmp_path / "review.jsonl"
        review.write_text(
            "".join(record.model_dump_json() + "\n" for record in mutated), encoding="utf-8"
        )

        result = runner.invoke(
            app, ["--corpus", str(_CORPUS), "--review", str(review), "--report-only"]
        )

        assert result.exit_code == 0
        assert "Field-level agreement 83.33% over 720 fields" in result.output
        assert "field-level agreement 83.33% < 90%" in result.output
        assert "by field: expected_outcome 100.0% (100), expected_error 100.0% (100)" in (
            result.output
        )
        assert "duration_seconds 0.0% (60)" in result.output
        assert "duration_seconds" in result.output
        assert "The two labelings agree" not in result.output

    def test_a_review_file_of_the_wrong_corpus_is_refused_with_a_reason(
        self, tmp_path: Path
    ) -> None:
        review = tmp_path / "review.jsonl"
        review.write_text(_record(prompt="a different request").model_dump_json(), encoding="utf-8")
        corpus = _one_record_corpus(tmp_path, _record())

        result = runner.invoke(
            app, ["--corpus", str(corpus), "--review", str(review), "--report-only"]
        )

        assert result.exit_code == 1
        assert "Cannot compare the two labelings" in result.output

    def test_the_report_is_valid_json_when_dumped(self) -> None:
        """The report's own shape, so a caller can read it as well as a person."""
        from saimc.labels import compare_labelings

        records, _ = load_corpus(_CORPUS)
        payload = json.loads(json.dumps(compare_labelings(records, records).to_dict()))

        assert payload["field_count"] == 720
        assert payload["by_field"][0]["field"] == "expected_outcome"
