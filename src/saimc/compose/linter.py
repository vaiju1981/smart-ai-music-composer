"""Theory linter for the NotationScore.

Per `docs/roadmap.md` §8 (Composition correctness):

- All notes within instrument range.
- All measures complete (no dropped beats).
- The melody resolves: its final note is the tonic or its third.
- Generated MusicXML validates against the MusicXML schema.
- Generated MIDI is well-formed.

The linter runs as the `validating` job stage. Failures are
structured: every issue has a stable `code` so the operator (and the
acceptance tests) can branch on it. The linter does NOT auto-fix
issues; the engine is responsible for producing a score that passes.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterator
from dataclasses import dataclass
from enum import StrEnum

from saimc.compose.duration import bar_ticks
from saimc.compose.forms import PHRASE_BARS, STEP_MAX_SEMITONES, bar_diatonic_pcs, key_root_midi
from saimc.compose.score import (
    VOICE_MELODY,
    VOICE_PERCUSSION,
    NotationScore,
    NoteEvent,
)

# Phase 1 is piano-only. The piano range MIDI is 21 (A0) to 108 (C8).
# We give a 1-note margin at each end to allow idiomatic voicings
# without forcing the engine to reach the physical extremes.
PIANO_MIN_MIDI: int = 22
PIANO_MAX_MIDI: int = 107

# Maximum simultaneous note count. Piano can't physically play more
# than 10 notes at once; the engine caps at 8 to leave headroom for
# the humanizer.
MAX_SIMULTANEOUS_NOTES: int = 8

# Close-position dissonances between simultaneously sounding voices:
# minor seconds and major sevenths. (Tritones and wider clashes are
# colours the harmony may intend; a rubbed m2/M7 is a bug.)
DISSONANT_INTERVALS: frozenset[int] = frozenset((1, 11))


class LintCode(StrEnum):
    """Stable error codes for linter findings."""

    NOTE_OUT_OF_RANGE = "note_out_of_range"
    MEASURE_INCOMPLETE = "measure_incomplete"
    TOO_MANY_SIMULTANEOUS_NOTES = "too_many_simultaneous_notes"
    TIME_SIGNATURE_MISMATCH = "time_signature_mismatch"
    EMPTY_SCORE = "empty_score"
    DUPLICATE_NOTE_AT_TICK = "duplicate_note_at_tick"
    ENDS_OFF_TONIC = "ends_off_tonic"
    CHORD_TONE_VIOLATION = "chord_tone_violation"
    DISSONANT_COLLISION = "dissonant_collision"
    PHRASE_GAP_MISSING = "phrase_gap_missing"


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


def lint(score: NotationScore, *, chord_bars: tuple[tuple[int, ...], ...] | None = None) -> LintReport:
    """Run every linter check and return a structured report.

    `chord_bars` carries the chord pitch classes sounding in each bar
    (bar order). When given, the harmony-aware checks run: every pitched
    note must be a chord tone of its bar, an anacrusis anticipation, or a
    legal passing/neighbour tone (see `legal_non_chord_tone`), and close
    m2/M7 overlaps between voices are flagged unless both tones belong to
    the sounding chord (a maj7 voicing is intended, a rubbed second is
    not). Without it the two harmony checks are skipped — a chord-less
    score cannot state its intent.
    """
    issues: list[LintIssue] = []
    issues.extend(_check_range(score))
    issues.extend(_check_measures_complete(score))
    issues.extend(_check_simultaneous_notes(score))
    issues.extend(_check_time_signature_consistency(score))
    issues.extend(_check_duplicate_notes(score))
    issues.extend(_check_nonempty(score))
    issues.extend(_check_ends_on_tonic(score))
    if chord_bars is not None:
        issues.extend(_check_chord_tones(score, chord_bars))
        issues.extend(_check_dissonant_collisions(score, chord_bars))
    issues.extend(_check_phrase_gaps(score))

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
    """Flag measures where more than MAX_SIMULTANEOUS_NOTES start at the same tick.

    The cap models a pianist's two hands: it applies to the melodic
    voices. The percussion kit is one drum machine — a crash, kick, and
    hat can all fire on the same downbeat without straining anything —
    so its notes don't count toward the cap.
    """
    issues: list[LintIssue] = []
    by_tick: dict[int, list[NoteEvent]] = {}
    for note in score.notes:
        if note.voice_id == VOICE_PERCUSSION:
            continue
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


def _check_ends_on_tonic(score: NotationScore) -> list[LintIssue]:
    """The melody's final note must resolve to the tonic or its third.

    A piece that ends on an arbitrary chord tone sounds unfinished;
    the engine's cadence lands the last note at home. The check allows
    the third because a piece may close on the more coloursome
    mediant; anything else (fifth included) fails.
    """
    melody_notes = [n for n in score.notes if n.voice_id == VOICE_MELODY]
    if not melody_notes:
        return []
    last = max(melody_notes, key=lambda n: (n.tick, n.pitch_midi))
    tonic_pc = key_root_midi(score.key) % 12
    third_pc = (tonic_pc + (4 if score.key.mode == "major" else 3)) % 12
    if last.pitch_midi % 12 not in (tonic_pc, third_pc):
        return [
            LintIssue(
                code=LintCode.ENDS_OFF_TONIC,
                message=(
                    f"final melody note pitch {last.pitch_midi} at tick {last.tick} "
                    f"does not resolve to the tonic or its third "
                    f"(key {score.key.root} {score.key.mode})"
                ),
                tick=last.tick,
                voice_id=last.voice_id,
                pitch_midi=last.pitch_midi,
            )
        ]
    return []


def _measure_starts(score: NotationScore) -> list[int]:
    """The measures' start ticks, in order — for tick-to-bar lookups."""
    return [m.start_tick for m in score.measures]


def _bar_of_tick(starts: list[int], tick: int) -> int:
    """Return the bar index a tick falls in, or -1 when past the last measure."""
    index = bisect_right(starts, tick) - 1
    if index < 0 or index >= len(starts):
        return -1
    return index


def legal_non_chord_tone(
    note: NoteEvent,
    *,
    prev: NoteEvent | None,
    nxt: NoteEvent | None,
    bar_start_tick: int,
    ppq: int,
    diatonic_pcs: frozenset[int],
) -> bool:
    """Whether a non-chord tone is a legitimate passing or neighbour tone.

    This is the licence that lets a melody move by step between chord
    tones, and it is the *only* licence: every other non-chord tone is a
    harmony bug. All four conditions must hold.

    - **Unaccented.** It does not begin on the bar's downbeat. A
      non-chord tone on the beat is a suspension or an appoggiatura, and
      the engine models neither device.
    - **Short.** It lasts no longer than a quarter note (`ppq` ticks). A
      brief tone is a passing tone by definition; a long unprepared
      dissonance is the unmodelled device again.
    - **Diatonic.** It belongs to the scale the bar's harmony belongs to
      (`forms.bar_diatonic_pcs`), so it cannot clash chromatically with
      the prevailing harmony.
    - **Approached and left by step, with no rest between.** Both
      neighbours exist in the same voice, each abuts this note exactly,
      and each lies within `STEP_MAX_SEMITONES` of it.

    Direction is deliberately *not* constrained. A passing tone is
    entered and left by step in the same direction (C-D-E) and a
    neighbour tone in the opposite one (C-D-C), so demanding either
    direction would reject the other figure. The adjacency requirement
    is what keeps the rule strict: a stepwise tone followed by a rest
    does not resolve, so it does not qualify.

    `prev` and `nxt` are this note's neighbours *in its own voice*, which
    the caller derives — a step across a rest or from another voice is
    not a step. Either being `None` rejects the note.
    """
    if note.tick == bar_start_tick:
        return False
    if note.duration_ticks > ppq:
        return False
    if note.pitch_midi % 12 not in diatonic_pcs:
        return False
    if prev is None or nxt is None:
        return False
    previous_abuts = prev.tick + prev.duration_ticks == note.tick
    next_abuts = nxt.tick == note.tick + note.duration_ticks
    if not (previous_abuts and next_abuts):
        return False
    previous_step = abs(note.pitch_midi - prev.pitch_midi)
    next_step = abs(note.pitch_midi - nxt.pitch_midi)
    return 0 < previous_step <= STEP_MAX_SEMITONES and 0 < next_step <= STEP_MAX_SEMITONES


def _voice_neighbours(
    score: NotationScore,
) -> dict[int, tuple[NoteEvent | None, NoteEvent | None]]:
    """Map each note's position in `score.notes` to its same-voice neighbours.

    Neighbours are the adjacent entries in tick order within a voice —
    the note before it and the note after it. Whether they *abut* it is
    the licence's business, not this lookup's, so a rest between them
    simply fails there.

    Positions, not notes, key the result: two notes of one voice could be
    equal in every field the score records, and a mapping keyed on the
    note itself would silently give them each other's neighbours.
    """
    by_voice: dict[int, list[int]] = {}
    for position, note in enumerate(score.notes):
        by_voice.setdefault(note.voice_id, []).append(position)
    neighbours: dict[int, tuple[NoteEvent | None, NoteEvent | None]] = {}
    for positions in by_voice.values():
        positions.sort(key=lambda p: (score.notes[p].tick, score.notes[p].pitch_midi))
        for order, position in enumerate(positions):
            before = score.notes[positions[order - 1]] if order > 0 else None
            after = score.notes[positions[order + 1]] if order + 1 < len(positions) else None
            neighbours[position] = (before, after)
    return neighbours


def _check_chord_tones(
    score: NotationScore,
    chord_bars: tuple[tuple[int, ...], ...],
) -> list[LintIssue]:
    """Every pitched note must sound a chord tone of its bar, or be a legal one.

    Percussion notes are GM kit keys, not pitches, and are exempt. Two
    licences apply. The anacrusis: a pickup in the bar's final eighth may
    anticipate the next chord (that is what a pickup is for). And the
    passing/neighbour tone, per `legal_non_chord_tone` — an unaccented,
    brief, diatonic tone entered and left by step between chord tones,
    which is how a melody moves by step at all.

    Anything else is a harmony bug: the engine composes from chord-tone
    tables plus that one stepwise figure, so a tone outside both is
    wrong rather than expressive.
    """
    issues: list[LintIssue] = []
    starts = _measure_starts(score)
    neighbours = _voice_neighbours(score)
    # Per bar, not per note: the scale a bar's harmony belongs to is a
    # property of the bar, and recovering it scans 24 candidate scales.
    diatonic = [bar_diatonic_pcs(tuple(bar), score.key) for bar in chord_bars]
    for position, note in enumerate(score.notes):
        if note.voice_id == VOICE_PERCUSSION:
            continue
        bar = _bar_of_tick(starts, note.tick)
        if bar < 0 or bar >= len(chord_bars):
            continue
        bar_pcs = chord_bars[bar]
        pc = note.pitch_midi % 12
        if pc in bar_pcs:
            continue
        # Anacrusis zone: the bar's final eighth may anticipate the
        # next chord's tones.
        measure = score.measures[bar]
        in_anticipation = note.tick >= measure.end_tick - score.ppq // 2
        next_pcs = chord_bars[bar + 1] if bar + 1 < len(chord_bars) else ()
        if in_anticipation and pc in next_pcs:
            continue
        prev, nxt = neighbours[position]
        if legal_non_chord_tone(
            note,
            prev=prev,
            nxt=nxt,
            bar_start_tick=measure.start_tick,
            ppq=score.ppq,
            diatonic_pcs=diatonic[bar],
        ):
            continue
        issues.append(
            LintIssue(
                code=LintCode.CHORD_TONE_VIOLATION,
                message=(
                    f"pitch {note.pitch_midi} at tick {note.tick} is not a chord "
                    f"tone of bar {bar} (pcs {list(bar_pcs)}) and is not a legal "
                    f"passing or neighbour tone"
                ),
                measure_index=bar,
                tick=note.tick,
                voice_id=note.voice_id,
                pitch_midi=note.pitch_midi,
            )
        )
    return issues


def _check_dissonant_collisions(
    score: NotationScore,
    chord_bars: tuple[tuple[int, ...], ...],
) -> list[LintIssue]:
    """Flag close m2/M7 overlaps between different pitched voices.

    Both tones belonging to the bar's chord is a voicing, not a
    collision — a maj7 chord (F A C E) legitimately carries an M7 a
    semitone-wide inversion apart when the melody and bass land on
    neighbouring chord tones. What must never sound is a rubbed
    second: a non-chord tone grinding against a chord tone, or two
    non-chord tones a semitone apart. Percussion is exempt (GM kit
    keys are not pitch classes).
    """
    issues: list[LintIssue] = []
    starts = _measure_starts(score)
    pitched = [n for n in score.notes if n.voice_id != VOICE_PERCUSSION]
    for a, b in _overlapping_pairs(pitched):
        distance = abs(a.pitch_midi - b.pitch_midi)
        if distance not in DISSONANT_INTERVALS:
            continue
        bar = _bar_of_tick(starts, max(a.tick, b.tick))
        if 0 <= bar < len(chord_bars):
            bar_pcs = set(chord_bars[bar])
            if a.pitch_midi % 12 in bar_pcs and b.pitch_midi % 12 in bar_pcs:
                continue
        lower, higher = sorted((a, b), key=lambda n: n.pitch_midi)
        issues.append(
            LintIssue(
                code=LintCode.DISSONANT_COLLISION,
                message=(
                    f"voices {lower.voice_id} and {higher.voice_id} sound pitches "
                    f"{lower.pitch_midi}/{higher.pitch_midi} a {distance}-semitone "
                    f"interval apart at tick {max(a.tick, b.tick)}"
                ),
                measure_index=bar if bar >= 0 else None,
                tick=max(a.tick, b.tick),
                voice_id=a.voice_id,
                pitch_midi=a.pitch_midi,
            )
        )
    return issues


def _overlapping_pairs(
    notes: list[NoteEvent],
) -> Iterator[tuple[NoteEvent, NoteEvent]]:
    """Yield each pair of different-voice notes whose sounding times overlap."""
    for a_index, a in enumerate(notes):
        for b in notes[a_index + 1 :]:
            if b.voice_id == a.voice_id:
                continue
            if a.tick < b.tick + b.duration_ticks and b.tick < a.tick + a.duration_ticks:
                yield a, b


def _check_phrase_gaps(score: NotationScore) -> list[LintIssue]:
    """The melody must breathe: no continuous span longer than a phrase.

    A span is continuous while consecutive melody notes touch or tie;
    a rest (however short) breaks it. Anything beyond `PHRASE_BARS`
    bars of unbroken melody reads as a never-breathing line, the
    mechanical arpeggio this engine exists to avoid. Anacrusis pickups
    (the last-eighth lead-in notes) are excluded — they attach to the
    following phrase, and the rest before them is the breath.
    """
    issues: list[LintIssue] = []
    ticks_per_bar = bar_ticks(score.time_signature)
    melody_notes = sorted(
        (
            n
            for n in score.notes
            if n.voice_id == VOICE_MELODY
            and not _is_anacrusis_pickup(n, ticks_per_bar, score.ppq)
        ),
        key=lambda n: (n.tick, n.pitch_midi),
    )
    if len(melody_notes) < 2:
        return issues
    max_span_ticks = PHRASE_BARS * ticks_per_bar
    group_start = melody_notes[0].tick
    prev = melody_notes[0]
    for note in melody_notes[1:]:
        if note.tick > prev.tick + prev.duration_ticks:
            _flag_long_span(issues, group_start, prev, max_span_ticks, ticks_per_bar)
            group_start = note.tick
        prev = note
    _flag_long_span(issues, group_start, prev, max_span_ticks, ticks_per_bar)
    return issues


def _flag_long_span(
    issues: list[LintIssue],
    group_start: int,
    last_note: NoteEvent,
    max_span_ticks: int,
    ticks_per_bar: int,
) -> None:
    span_ticks = last_note.tick + last_note.duration_ticks - group_start
    if span_ticks > max_span_ticks:
        issues.append(
            LintIssue(
                code=LintCode.PHRASE_GAP_MISSING,
                message=(
                    f"melody runs {span_ticks / ticks_per_bar:.1f} bars without a "
                    f"breath from tick {group_start} (max is {max_span_ticks} ticks)"
                ),
                tick=group_start,
                voice_id=VOICE_MELODY,
            )
        )


def _is_anacrusis_pickup(note: NoteEvent, ticks_per_bar: int, ppq: int) -> bool:
    """A pickup note: an eighth in the bar's final eighth-note slot."""
    return (
        note.tick % ticks_per_bar == ticks_per_bar - ppq // 2
        and note.duration_ticks <= ppq // 2
    )


__all__ = [
    "MAX_SIMULTANEOUS_NOTES",
    "PIANO_MAX_MIDI",
    "PIANO_MIN_MIDI",
    "LintCode",
    "LintIssue",
    "LintReport",
    "legal_non_chord_tone",
    "lint",
]
