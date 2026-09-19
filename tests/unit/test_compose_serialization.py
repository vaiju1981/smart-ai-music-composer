"""Tests for engine-output sidecar persistence."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from saimc.compose.duration import DurationArrangement
from saimc.compose.engine import SIDECAR_FORMAT, EngineOutput, compose
from saimc.compose.forms import ChordSlot, ChordTemplate
from saimc.compose.plan import PLAN_FORMAT, CompositionPlan, default_plan
from saimc.compose.score import (
    KeySignature,
    Measure,
    NotationScore,
    NoteEvent,
    PerformanceNoteEvent,
    PerformancePlan,
    TempoMap,
)
from saimc.compose.serialization import read_engine_output, write_engine_output
from saimc.spec import CompositionSpec, Mood


def _make_engine_output() -> EngineOutput:
    note = NoteEvent(voice_id=0, pitch_midi=60, tick=0, duration_ticks=480, velocity=64)
    measure = Measure(index=0, start_tick=0, end_tick=1920, time_signature="4/4")
    notation = NotationScore.make(
        ppq=480,
        key=KeySignature(root="C", mode="major"),
        time_signature="4/4",
        tempo_bpm=80.0,
        measures=[measure],
        notes=[note],
    )
    perf_note = PerformanceNoteEvent(
        voice_id=0,
        pitch_midi=60,
        start_us=0,
        duration_us=500_000,
        velocity=64,
    )
    performance = PerformancePlan.make(sample_rate=44100, notes=[perf_note])
    arrangement = DurationArrangement(
        form_bars=1,
        template=ChordTemplate(name="stub_1bar", bars=1, chords=(ChordSlot(0, 1),)),
        repetition_count=1,
        total_bars=1,
        tempo_bpm=80.0,
        coda_bars=0,
    )
    return EngineOutput(
        notation_score=notation,
        performance_plan=performance,
        arrangement=arrangement,
        key=KeySignature(root="C", mode="major"),
        time_signature="4/4",
    )


def test_round_trip_preserves_all_fields(tmp_path: Path) -> None:
    """to_sidecar -> write -> read -> from_sidecar reproduces the output."""
    out = _make_engine_output()
    path = tmp_path / "engine_output.json"
    write_engine_output(path, out)
    assert path.exists()

    back = read_engine_output(path)
    assert back == out


def test_round_trip_for_real_engine_output(tmp_path: Path) -> None:
    """The real Phase 1 engine round-trips through the sidecar."""
    spec = CompositionSpec(mood=Mood.CALMING, seed=42, duration_seconds=180)
    out = compose(spec)
    path = tmp_path / "engine_output.json"
    write_engine_output(path, out)
    back = read_engine_output(path)
    # Compare the canonical hashes (a strong equality check).
    assert back.notation_score.compute_hash() == out.notation_score.compute_hash()
    assert back.performance_plan.compute_hash() == out.performance_plan.compute_hash()
    assert back.arrangement == out.arrangement
    assert back.key == out.key
    assert back.time_signature == out.time_signature


def test_read_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_engine_output(tmp_path / "nope.json")


def test_coda_bars_round_trips(tmp_path: Path) -> None:
    """`coda_bars=2` survives the round trip."""
    out = _make_engine_output()
    # Rebuild with a non-zero coda. Coda must be < form_bars, so use a
    # 4-bar form with a 2-bar coda.
    arrangement = DurationArrangement(
        form_bars=4,
        template=ChordTemplate(name="stub_4bar", bars=4, chords=(ChordSlot(0, 2), ChordSlot(5, 2))),
        repetition_count=1,
        total_bars=4,
        tempo_bpm=80.0,
        coda_bars=2,
    )
    out2 = EngineOutput(
        notation_score=out.notation_score,
        performance_plan=out.performance_plan,
        arrangement=arrangement,
        key=out.key,
        time_signature=out.time_signature,
    )
    path = tmp_path / "engine_output.json"
    write_engine_output(path, out2)
    back = read_engine_output(path)
    assert back.arrangement.coda_bars == 2


def test_write_is_canonical_json(tmp_path: Path) -> None:
    """The on-disk file is canonical (sorted keys, no extra whitespace)."""
    out = _make_engine_output()
    path = tmp_path / "engine_output.json"
    write_engine_output(path, out)
    text = path.read_text(encoding="utf-8")
    # No insignificant whitespace.
    assert ": " not in text
    assert ", " not in text


def test_to_sidecar_does_not_include_tempo_map_inline() -> None:
    """TempoMap is included under notation_score, not at the top level."""
    out = _make_engine_output()
    sd = out.to_sidecar()
    assert "tempo" in sd["notation_score"]
    assert isinstance(sd["notation_score"]["tempo"], dict)
    assert sd["notation_score"]["tempo"]["bpm"] == 80.0
    assert sd["notation_score"]["tempo"]["ppq"] == 480
    assert "tempo" not in sd  # top-level
    # Sanity: TempoMap is its own dataclass.
    assert isinstance(out.notation_score.tempo, TempoMap)


def test_sidecar_round_trips_controllers_and_pitch_bends(tmp_path: Path) -> None:
    """The expression layer survives the sidecar (S4)."""
    out = _make_engine_output()
    from saimc.compose.score import ControllerEvent, PitchBendEvent

    plan_with_expression = replace(
        out.performance_plan,
        controllers=(
            ControllerEvent(voice_id=0, control=11, value=96, start_us=0),
            ControllerEvent(voice_id=0, control=64, value=127, start_us=250_000),
        ),
        pitch_bends=(PitchBendEvent(voice_id=0, start_us=100_000, bend=-2048),),
    )
    out = replace(out, performance_plan=plan_with_expression)
    path = tmp_path / "engine_output.json"
    write_engine_output(path, out)
    back = read_engine_output(path)
    assert back == out
    assert back.performance_plan.controllers == plan_with_expression.controllers
    assert back.performance_plan.pitch_bends == plan_with_expression.pitch_bends


def test_tempo_changes_round_trip(tmp_path: Path) -> None:
    """The outro ritardando's tempo points survive the sidecar."""
    spec = CompositionSpec(mood=Mood.CALMING, seed=42, duration_seconds=45)
    out = compose(spec)
    assert out.notation_score.tempo.changes, "expected a coda ritardando"
    path = tmp_path / "engine_output.json"
    write_engine_output(path, out)
    back = read_engine_output(path)
    assert back.notation_score.tempo.changes == out.notation_score.tempo.changes
    assert back.arrangement.ritardando_factor == out.arrangement.ritardando_factor
    assert back.arrangement.intro_bars == out.arrangement.intro_bars


def test_chord_bars_round_trip(tmp_path: Path) -> None:
    """The per-bar chord pitch classes survive the sidecar for the gates."""
    spec = CompositionSpec(mood=Mood.CALMING, seed=42, duration_seconds=45)
    out = compose(spec)
    assert out.chord_bars, "expected per-bar chord pcs"
    path = tmp_path / "engine_output.json"
    write_engine_output(path, out)
    back = read_engine_output(path)
    assert back.chord_bars == out.chord_bars


_DRUMS = [
    {"role": "melody", "instrument": "piano"},
    {"role": "harmony", "instrument": "strings"},
    {"role": "bass", "instrument": "contrabass"},
    {"role": "percussion", "instrument": "drum_set"},
]
"""A piece with a rhythm section, so the plan's percussion fields are not
the defaults by accident and the round trip has them to lose."""


def _composed() -> EngineOutput:
    return compose(
        CompositionSpec.model_validate(
            {
                "mood": "electrifying",
                "duration_seconds": 45,
                "seed": 9,
                "instrumentation": _DRUMS,
            }
        )
    )


class TestTheSidecarRecordsThePlan:
    """The plan goes into the sidecar, versioned and readable.

    Without it a sidecar is a record of the notes and not of the decisions
    that produced them: the render stages read the sidecar rather than
    re-composing, so a plan that is not in it is a plan no later stage and
    no reader can see.
    """

    def test_the_composed_output_carries_the_resolved_plan(self) -> None:
        """`compose` resolves the default rather than leaving the field
        empty, so a caller who never heard of a plan still gets the record
        of what the piece was composed under."""
        spec = CompositionSpec.model_validate(
            {
                "mood": "electrifying",
                "duration_seconds": 45,
                "seed": 9,
                "instrumentation": _DRUMS,
            }
        )
        out = compose(spec)
        assert out.plan is not None
        assert out.plan == default_plan(spec)
        assert out.plan.format == PLAN_FORMAT

    def test_a_supplied_plan_is_the_one_the_sidecar_records(self) -> None:
        """The record is of the plan the piece was composed under, not of
        what the defaults would have been: a reader replaying the sidecar
        has to get the piece that was rendered."""
        spec = CompositionSpec.model_validate(
            {
                "mood": "electrifying",
                "duration_seconds": 45,
                "seed": 9,
                "instrumentation": _DRUMS,
            }
        )
        supplied = replace(default_plan(spec), harmony_pad_velocity=42)
        out = compose(spec, plan=supplied)
        assert out.plan == supplied
        assert out.plan != default_plan(spec)

    def test_the_plan_survives_the_sidecar(self, tmp_path: Path) -> None:
        out = _composed()
        path = tmp_path / "engine_output.json"
        write_engine_output(path, out)
        back = read_engine_output(path)
        assert back.plan == out.plan
        assert back.plan is not None
        assert back.plan.compute_hash() == out.plan.compute_hash()

    def test_the_plan_in_the_sidecar_is_the_canonical_document(
        self, tmp_path: Path
    ) -> None:
        """The block written is the document the plan's own hash covers, so
        a reader can verify it rather than take it on trust."""
        out = _composed()
        path = tmp_path / "engine_output.json"
        write_engine_output(path, out)
        document = json.loads(path.read_text(encoding="utf-8"))["plan"]
        assert document["format"] == PLAN_FORMAT
        reloaded = CompositionPlan.from_canonical_dict(document)
        assert reloaded == out.plan

    def test_a_sidecar_without_a_plan_still_loads(self, tmp_path: Path) -> None:
        """The backward tolerance every other sidecar field has.

        A sidecar written before the plan existed has no `plan` key, and
        the read is a `.get()` for exactly that reason. It is *not*
        resolved from the spec: a resolved plan would describe this build's
        defaults, not the ones the piece was composed under, and a
        provenance record that guesses is worse than one that says nothing.
        """
        out = _composed()
        payload = out.to_sidecar()
        del payload["plan"]
        back = EngineOutput.from_sidecar(payload)
        assert back.plan is None
        assert back.notation_score.compute_hash() == out.notation_score.compute_hash()

    def test_no_plan_is_omitted_rather_than_nulled(self) -> None:
        """A pre-plan output re-serialized must not invent a key that reads
        as "a plan was considered here"."""
        payload = _make_engine_output().to_sidecar()
        assert payload["format"] == SIDECAR_FORMAT
        assert "plan" not in payload

    def test_the_sidecar_carries_its_own_version_tag(self, tmp_path: Path) -> None:
        """§6's rule for a document that wraps others: the container names
        its own shape, because each nested document names only its own."""
        out = _composed()
        path = tmp_path / "engine_output.json"
        write_engine_output(path, out)
        assert json.loads(path.read_text(encoding="utf-8"))["format"] == SIDECAR_FORMAT

    @pytest.mark.parametrize("written", ["EngineOutput:2", "EngineOutput", "NotationScore:1"])
    def test_a_sidecar_of_another_version_is_refused(
        self, tmp_path: Path, written: str
    ) -> None:
        """Refused rather than coerced: a newer sidecar may carry state this
        build would drop on the floor while still reporting success."""
        payload = _composed().to_sidecar()
        payload["format"] = written
        with pytest.raises(ValueError) as exc_info:
            EngineOutput.from_sidecar(payload)
        assert SIDECAR_FORMAT in str(exc_info.value)

    def test_a_sidecar_with_no_tag_at_all_still_loads(self) -> None:
        """Every sidecar on disk today predates the tag, so its absence has
        to mean "the version this tag was introduced at" — not a refusal."""
        payload = _composed().to_sidecar()
        del payload["format"]
        assert EngineOutput.from_sidecar(payload).plan is not None

    def test_a_future_plan_version_inside_the_sidecar_is_refused(self) -> None:
        """The nested document is versioned too, and the refusal is the
        plan's own — the sidecar does not get to accept what the plan type
        would not."""
        from saimc.compose.plan import PLAN_FORMAT_PREFIX, PLAN_SCHEMA_VERSION, PlanError

        payload = _composed().to_sidecar()
        # Derived, so "the next version" stays the next one: written out it
        # silently becomes this build's own tag, and the test stops firing.
        payload["plan"]["format"] = f"{PLAN_FORMAT_PREFIX}:{PLAN_SCHEMA_VERSION + 1}"
        with pytest.raises(PlanError):
            EngineOutput.from_sidecar(payload)
