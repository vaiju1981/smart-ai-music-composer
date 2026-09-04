"""Theory linter for the NotationScore.

Per `docs/roadmap.md` §8 (Composition correctness):

- All notes within instrument range.
- All measures complete (no dropped beats).
- No unresolved voice-leading collisions flagged by the theory linter.
- Generated MusicXML validates against the MusicXML schema.
- Generated MIDI is well-formed.

The linter runs as the `validating` job stage. Failures are
structured: every issue has a stable `code` so the operator (and the
acceptance tests) can branch on it. The linter does NOT auto-fix
issues; the engine is responsible for producing a score that passes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from saimc.compose.duration import bar_ticks
from saimc.compose.score import NotationScore, NoteEvent

# Phase 1 is piano-only. The piano range MIDI is 21 (A0) to 108 (C8).
# We give a 1-note margin at each end to allow idiomatic voicings
# without forcing the engine to reach the physical extremes.
PIANO_MIN_MIDI: int = 22
PIANO_MAX_MIDI: int = 107

# Maximum simultaneous note count. Piano can't physically play more
# than 10 notes at once; the engine caps at 8 to leave headroom for
# the humanizer.
MAX_SIMULTANEOUS_NOTES: int = 8


class LintCode(StrEnum):
    """Stable error codes for linter findings."""

    NOTE_OUT_OF_RANGE = "note_out_of_range"
    MEASURE_INCOMPLETE = "measure_incomplete"
    VOICE_LEADING_COLLISION = "voice_leading_collision"
    TOO_MANY_SIMULTANEOUS_NOTES = "too_many_simultaneous_notes"
    TIME_SIGNATURE_MISMATCH = "time_signature_mismatch"
    EMPTY_SCORE = "empty_score"
    DUPLICATE_NOTE_AT_TICK = "duplicate_note_at_tick"


@dataclass(frozen=True)
class LintIssue:
    """A single linter finding.

    `measure_index` is the 0-based measure index where the issue
    occurs, or None if it spans the whole score.
    """

    code: LintCode
    message: str
    measure_index: int | None = None
    tick: int | None = None
    voice_id: int | None = None
    pitch_midi: int | None = None


@dataclass(frozen=True)
class LintReport:
    """Result of linting a NotationScore."""

    score_hash: str
    issues: tuple[LintIssue, ...]
    passed: bool


def lint(score: NotationScore) -> LintReport:
    """Run every linter check and return a structured report."""
    issues: list[LintIssue] = []
    issues.extend(_check_range(score))
    issues.extend(_check_measures_complete(score))
    issues.extend(_check_simultaneous_notes(score))
    issues.extend(_check_time_signature_consistency(score))
    issues.extend(_check_duplicate_notes(score))
    issues.extend(_check_nonempty(score))

    return LintReport(
        score_hash=score.compute_hash(),
        issues=tuple(issues),
        passed=not issues,
    )


def _check_range(score: NotationScore) -> list[LintIssue]:
    issues: list[LintIssue] = []
    for note in score.notes:
        if note.pitch_midi < PIANO_MIN_MIDI or note.pitch_midi > PIANO_MAX_MIDI:
            issues.append(
                LintIssue(
                    code=LintCode.NOTE_OUT_OF_RANGE,
                    message=(
                        f"pitch {note.pitch_midi} is outside the piano range "
                        f"[{PIANO_MIN_MIDI}, {PIANO_MAX_MIDI}]"
                    ),
                    tick=note.tick,
                    voice_id=note.voice_id,
                    pitch_midi=note.pitch_midi,
                )
            )
    return issues


def _check_measures_complete(score: NotationScore) -> list[LintIssue]:
    issues: list[LintIssue] = []
    for measure in score.measures:
        expected_ticks = bar_ticks(measure.time_signature)
        actual_ticks = measure.end_tick - measure.start_tick
        if actual_ticks != expected_ticks:
            issues.append(
                LintIssue(
                    code=LintCode.MEASURE_INCOMPLETE,
                    message=(
                        f"measure {measure.index} is {actual_ticks} ticks but "
                        f"time signature {measure.time_signature!r} requires {expected_ticks}"
                    ),
                    measure_index=measure.index,
                )
            )
    return issues


def _check_simultaneous_notes(score: NotationScore) -> list[LintIssue]:
    """Flag measures where more than MAX_SIMULTANEOUS_NOTES start at the same tick."""
    issues: list[LintIssue] = []
    by_tick: dict[int, list[NoteEvent]] = {}
    for note in score.notes:
        by_tick.setdefault(note.tick, []).append(note)
    for tick, notes_at_tick in by_tick.items():
        if len(notes_at_tick) > MAX_SIMULTANEOUS_NOTES:
            issues.append(
                LintIssue(
                    code=LintCode.TOO_MANY_SIMULTANEOUS_NOTES,
                    message=(
                        f"{len(notes_at_tick)} notes start at tick {tick}; "
                        f"max is {MAX_SIMULTANEOUS_NOTES}"
                    ),
                    tick=tick,
                )
            )
    return issues


def _check_time_signature_consistency(score: NotationScore) -> list[LintIssue]:
    """Every measure's time signature must match the score's."""
    issues: list[LintIssue] = []
    for measure in score.measures:
        if measure.time_signature != score.time_signature:
            issues.append(
                LintIssue(
                    code=LintCode.TIME_SIGNATURE_MISMATCH,
                    message=(
                        f"measure {measure.index} has time signature "
                        f"{measure.time_signature!r} but score is "
                        f"{score.time_signature!r}"
                    ),
                    measure_index=measure.index,
                )
            )
    return issues


def _check_duplicate_notes(score: NotationScore) -> list[LintIssue]:
    """A given (tick, voice, pitch) triple must appear at most once."""
    issues: list[LintIssue] = []
    seen: set[tuple[int, int, int]] = set()
    for note in score.notes:
        key = (note.tick, note.voice_id, note.pitch_midi)
        if key in seen:
            issues.append(
                LintIssue(
                    code=LintCode.DUPLICATE_NOTE_AT_TICK,
                    message=(
                        f"duplicate note at tick={note.tick}, voice={note.voice_id}, "
                        f"pitch={note.pitch_midi}"
                    ),
                    tick=note.tick,
                    voice_id=note.voice_id,
                    pitch_midi=note.pitch_midi,
                )
            )
        seen.add(key)
    return issues


def _check_nonempty(score: NotationScore) -> list[LintIssue]:
    if not score.notes:
        return [
            LintIssue(
                code=LintCode.EMPTY_SCORE,
                message="NotationScore has no notes",
            )
        ]
    return []


__all__ = [
    "MAX_SIMULTANEOUS_NOTES",
    "PIANO_MAX_MIDI",
    "PIANO_MIN_MIDI",
    "LintCode",
    "LintIssue",
    "LintReport",
    "lint",
]
