"""Unit tests for the theory linter."""

from __future__ import annotations

from saimc.compose.linter import (
    PIANO_MAX_MIDI,
    PIANO_MIN_MIDI,
    LintCode,
    lint,
)
from saimc.compose.score import (
    PPQ,
    KeySignature,
    Measure,
    NotationScore,
    NoteEvent,
)


def _build_score(
    *,
    time_signature: str = "4/4",
    key: KeySignature | None = None,
    notes: list[NoteEvent] | None = None,
    measures: list[Measure] | None = None,
) -> NotationScore:
    if measures is None:
        measures = [Measure(index=0, start_tick=0, end_tick=1920, time_signature="4/4")]
    if notes is None:
        notes = [NoteEvent(voice_id=0, pitch_midi=60, tick=0, duration_ticks=480)]
    return NotationScore.make(
        ppq=PPQ,
        key=key or KeySignature(root="C", mode="major"),
        time_signature=time_signature,
        tempo_bpm=80.0,
        measures=measures,
        notes=notes,
    )


class TestLintPasses:
    def test_minimal_score_passes(self) -> None:
        report = lint(_build_score())
        assert report.passed
        assert report.issues == ()


class TestLintFailsOnRange:
    def test_pitch_below_range(self) -> None:
        score = _build_score(
            notes=[NoteEvent(voice_id=0, pitch_midi=PIANO_MIN_MIDI - 1, tick=0, duration_ticks=480)]
        )
        report = lint(score)
        assert not report.passed
        assert any(i.code == LintCode.NOTE_OUT_OF_RANGE for i in report.issues)

    def test_pitch_above_range(self) -> None:
        score = _build_score(
            notes=[NoteEvent(voice_id=0, pitch_midi=PIANO_MAX_MIDI + 1, tick=0, duration_ticks=480)]
        )
        report = lint(score)
        assert not report.passed
        assert any(i.code == LintCode.NOTE_OUT_OF_RANGE for i in report.issues)


class TestLintFailsOnMeasureCompleteness:
    def test_short_measure_fails(self) -> None:
        # 4/4 measure should be 1920 ticks; we declare 1000.
        score = _build_score(
            measures=[Measure(index=0, start_tick=0, end_tick=1000, time_signature="4/4")]
        )
        report = lint(score)
        assert not report.passed
        assert any(i.code == LintCode.MEASURE_INCOMPLETE for i in report.issues)


class TestLintFailsOnTimeSignatureMismatch:
    def test_measure_time_signature_mismatch(self) -> None:
        score = _build_score(
            time_signature="4/4",
            measures=[
                Measure(index=0, start_tick=0, end_tick=1920, time_signature="4/4"),
                Measure(index=1, start_tick=1920, end_tick=3360, time_signature="3/4"),
            ],
        )
        report = lint(score)
        assert not report.passed
        assert any(i.code == LintCode.TIME_SIGNATURE_MISMATCH for i in report.issues)


class TestLintFailsOnEmpty:
    def test_empty_score_fails(self) -> None:
        score = _build_score(notes=[], measures=[])
        report = lint(score)
        assert not report.passed
        assert any(i.code == LintCode.EMPTY_SCORE for i in report.issues)


class TestLintFailsOnDuplicateNotes:
    def test_duplicate_at_tick_fails(self) -> None:
        score = _build_score(
            notes=[
                NoteEvent(voice_id=0, pitch_midi=60, tick=0, duration_ticks=480),
                NoteEvent(voice_id=0, pitch_midi=60, tick=0, duration_ticks=480),
            ]
        )
        report = lint(score)
        assert not report.passed
        assert any(i.code == LintCode.DUPLICATE_NOTE_AT_TICK for i in report.issues)

    def test_same_pitch_different_voice_ok(self) -> None:
        # Same pitch at the same tick but different voices is fine.
        score = _build_score(
            notes=[
                NoteEvent(voice_id=0, pitch_midi=60, tick=0, duration_ticks=480),
                NoteEvent(voice_id=1, pitch_midi=60, tick=0, duration_ticks=480),
            ]
        )
        report = lint(score)
        assert report.passed


class TestLintFailsOnSimultaneous:
    def test_too_many_simultaneous_fails(self) -> None:
        # 9 notes at tick=0 should fail (limit is 8).
        notes = [
            NoteEvent(voice_id=0, pitch_midi=60 + i, tick=0, duration_ticks=480) for i in range(9)
        ]
        score = _build_score(notes=notes)
        report = lint(score)
        assert not report.passed
        assert any(i.code == LintCode.TOO_MANY_SIMULTANEOUS_NOTES for i in report.issues)
