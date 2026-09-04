"""Phase 1 composition engine.

Takes a CompositionSpec and returns (NotationScore, PerformancePlan).
The engine is rule-based, deterministic, and piano-only. It uses
music21 only at generation time (chord-tone lookup, key transposition,
voice-range checks); the output is our canonical format, not a
music21 stream.

Stages (per `docs/roadmap.md` §2 step 3):

1. Resolve key + time signature from the spec.
2. Pick a mood-appropriate chord template (form).
3. Apply the §10 #1 duration policy to find the form, repetition
   count, and tempo that fit the spec's `duration_seconds` within
   ±2% tolerance.
4. Generate one melody voice + one bass voice over the chord
   progression, applying per-section seed-derived variation when
   the arrangement has repeats.
5. Theory-lint the resulting NotationScore.
6. Build the PerformancePlan with integer-microsecond timestamps.

The engine raises `CompositionEngineError` (with a stable code) for
any failure the calling layer should react to. Lint failures become
a `lint_failed` error before the engine returns.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import StrEnum

from saimc.compose.duration import (
    DurationArrangement,
    DurationUnfulfillableError,
    arrange_for_duration,
    bar_ticks,
    section_seed,
)
from saimc.compose.forms import (
    ChordTemplate,
    key_root_midi,
    key_signature_from_spec,
)
from saimc.compose.linter import LintIssue, lint
from saimc.compose.score import (
    DEFAULT_VELOCITY,
    PPQ,
    KeySignature,
    Measure,
    NotationScore,
    NoteEvent,
    PerformanceNoteEvent,
    PerformancePlan,
    ticks_to_microseconds,
)
from saimc.spec import CompositionSpec


class EngineErrorCode(StrEnum):
    """Stable error codes emitted by the engine."""

    DURATION_UNFULFILLABLE = "duration_unfulfillable"
    LINT_FAILED = "lint_failed"
    INVALID_SPEC = "invalid_spec"


class CompositionEngineError(Exception):
    """Raised when the engine cannot produce a score for the spec."""

    def __init__(
        self, code: EngineErrorCode, message: str, *, lint_issues: tuple[LintIssue, ...] = ()
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.lint_issues = lint_issues


@dataclass(frozen=True)
class EngineOutput:
    """The engine's output: NotationScore + PerformancePlan + arrangement metadata."""

    notation_score: NotationScore
    performance_plan: PerformancePlan
    arrangement: DurationArrangement
    key: KeySignature
    time_signature: str


def compose(spec: CompositionSpec) -> EngineOutput:
    """Run the full composition pipeline against the spec."""
    key = key_signature_from_spec(spec)
    time_signature = spec.time_signature.value

    try:
        arrangement = arrange_for_duration(
            mood=spec.mood.value,
            target_duration_seconds=float(spec.duration_seconds),
            time_signature=time_signature,
        )
    except DurationUnfulfillableError as exc:
        raise CompositionEngineError(
            code=EngineErrorCode.DURATION_UNFULFILLABLE,
            message=str(exc),
        ) from exc

    score = _build_score(spec, key, time_signature, arrangement)
    lint_report = lint(score)
    if not lint_report.passed:
        raise CompositionEngineError(
            code=EngineErrorCode.LINT_FAILED,
            message=f"score failed lint: {[i.code for i in lint_report.issues]}",
            lint_issues=lint_report.issues,
        )

    performance = _build_performance_plan(score)
    return EngineOutput(
        notation_score=score,
        performance_plan=performance,
        arrangement=arrangement,
        key=key,
        time_signature=time_signature,
    )


# ---------------------------------------------------------------------------
# Score generation
# ---------------------------------------------------------------------------


def _build_score(
    spec: CompositionSpec,
    key: KeySignature,
    time_signature: str,
    arrangement: DurationArrangement,
) -> NotationScore:
    """Build the NotationScore from the spec + arrangement.

    Generates one melody voice + one bass voice per section, with
    per-section seed-derived variation when the arrangement has
    multiple repetitions. If the arrangement has a coda, an extra
    coda-length tail is appended using a coda-flavored seed so the
    variation rules from §10 #10 still apply.
    """
    measures: list[Measure] = []
    notes: list[NoteEvent] = []
    section_starts: list[int] = []  # start_tick of each section
    cursor_tick = 0

    rng_base_seed = spec.seed if spec.seed is not None else 0

    for section_idx in range(arrangement.repetition_count):
        section_rng = random.Random(section_seed(spec.seed, section_idx))
        section_starts.append(cursor_tick)
        section_notes = _generate_section(
            key=key,
            time_signature=time_signature,
            template=arrangement.template,
            section_start_tick=cursor_tick,
            rng=section_rng,
            seed_for_variation=rng_base_seed + section_idx,
        )
        notes.extend(section_notes)
        cursor_tick += arrangement.form_bars * bar_ticks(time_signature)

    # Optional coda: append a coda-length tail using the same chord
    # template (truncated to coda_bars). The coda gets its own RNG
    # seed (a stable offset from the spec seed) so it sounds distinct
    # from the body, per §10 #10.
    if arrangement.coda_bars > 0:
        coda_template = _truncate_template_for_coda(arrangement.template, arrangement.coda_bars)
        coda_rng = random.Random(section_seed(spec.seed, arrangement.repetition_count))
        coda_notes = _generate_section(
            key=key,
            time_signature=time_signature,
            template=coda_template,
            section_start_tick=cursor_tick,
            rng=coda_rng,
            seed_for_variation=rng_base_seed + arrangement.repetition_count,
        )
        notes.extend(coda_notes)
        cursor_tick += arrangement.coda_bars * bar_ticks(time_signature)

    # Build measures with strict 1-based bar indexing, including any
    # coda bars after the full repetitions.
    total_bars = arrangement.total_bars_with_coda
    for bar_idx in range(total_bars):
        start = bar_idx * bar_ticks(time_signature)
        measures.append(
            Measure(
                index=bar_idx,
                start_tick=start,
                end_tick=start + bar_ticks(time_signature),
                time_signature=time_signature,
            )
        )

    return NotationScore.make(
        ppq=PPQ,
        key=key,
        time_signature=time_signature,
        tempo_bpm=arrangement.tempo_bpm,
        measures=measures,
        notes=notes,
    )


def _generate_section(
    *,
    key: KeySignature,
    time_signature: str,
    template: ChordTemplate,
    section_start_tick: int,
    rng: random.Random,
    seed_for_variation: int,
) -> list[NoteEvent]:
    """Generate the bass + melody notes for one section.

    The bass voice plays the chord root on beat 1 of every bar (one
    whole-note-per-bar voice that the listener can hear as a drone).
    The melody voice arpegiates through the chord tones, with rhythm
    variation seeded by `seed_for_variation` so repeated sections
    sound different without changing the harmonic content.
    """
    notes: list[NoteEvent] = []
    tonic_midi = key_root_midi(key)
    bar_ticks_count = bar_ticks(time_signature)

    cursor = 0
    for degree, dur in template.chords:
        chord_root = tonic_midi + _scale_degree_to_semitones(degree, key.mode)
        chord_tones = _chord_tones_midi(degree, key)
        chord_root_tick = section_start_tick + cursor
        bar_length_ticks = dur * bar_ticks_count

        # Bass voice: chord root on beat 1, held for the whole chord.
        notes.append(
            NoteEvent(
                voice_id=0,
                pitch_midi=_octave_down(chord_root, octaves=1),
                tick=chord_root_tick,
                duration_ticks=bar_length_ticks,
                velocity=_velocity_for_section(seed_for_variation, base=58, jitter=8),
            )
        )

        # Melody voice: arpeggiate the chord tones across the chord
        # duration, with seed-derived rhythm variation.
        melody_notes = _arpeggiate_chord(
            chord_root=chord_root,
            chord_tones=chord_tones,
            start_tick=chord_root_tick,
            duration_ticks=bar_length_ticks,
            rng=rng,
            key_mode=key.mode,
            seed_for_variation=seed_for_variation,
        )
        notes.extend(melody_notes)
        cursor += bar_length_ticks

    return notes


def _truncate_template_for_coda(template: ChordTemplate, coda_bars: int) -> ChordTemplate:
    """Return a coda-sized prefix of `template`.

    The coda is a sub-form: a coda_bars-bar prefix of the form's
    template, using the first chord cycles that fit. Coda length is
    always strictly less than the form's full length. Zero-duration
    chord entries are dropped.
    """
    kept: list[tuple[int, int]] = []
    consumed = 0
    for degree, dur in template.chords:
        remaining = coda_bars - consumed
        if remaining <= 0:
            break
        if dur > remaining:
            kept.append((degree, remaining))
            consumed += remaining
        else:
            kept.append((degree, dur))
            consumed += dur
    return ChordTemplate(
        name=f"{template.name}_coda{coda_bars}",
        bars=coda_bars,
        chords=tuple(kept),
    )


def _scale_degree_to_semitones(degree: int, mode: str) -> int:
    """Map a 0-based scale degree to its semitone offset from the tonic."""
    if mode == "major":
        major_scale = (0, 2, 4, 5, 7, 9, 11)
        return major_scale[degree % 7]
    minor_scale = (0, 2, 3, 5, 7, 8, 10)
    return minor_scale[degree % 7]


def _chord_tones_midi(degree: int, key: KeySignature) -> tuple[int, ...]:
    """Return the chord-tone MIDI offsets for a diatonic chord.

    Major key: I, ii, iii, IV, V, vi, vii -> major, minor, minor, major,
    major, minor, dim. Minor key: i, ii°, III, iv, v, VI, VII -> minor,
    dim, major, minor, minor, major, major.
    """
    major_triads = (
        (0, 4, 7),
        (2, 5, 9),
        (4, 7, 11),
        (5, 9, 12),
        (7, 11, 14),
        (9, 12, 16),
        (11, 14, 17),
    )
    minor_triads = (
        (0, 3, 7),
        (2, 5, 8),
        (3, 7, 10),
        (5, 8, 12),
        (7, 10, 14),
        (8, 12, 15),
        (10, 14, 17),
    )
    table = major_triads if key.mode == "major" else minor_triads
    return table[degree % 7]


def _octave_down(midi: int, *, octaves: int) -> int:
    """Shift a MIDI note down by `octaves` octaves, floored at 21."""
    return max(21, midi - 12 * octaves)


def _velocity_for_section(seed: int, *, base: int, jitter: int) -> int:
    """Deterministically vary the velocity by ±`jitter` using `seed`."""
    rng = random.Random(seed * 31 + 7)
    return max(1, min(127, base + rng.randint(-jitter, jitter)))


def _arpeggiate_chord(
    *,
    chord_root: int,
    chord_tones: tuple[int, ...],
    start_tick: int,
    duration_ticks: int,
    rng: random.Random,
    key_mode: str,
    seed_for_variation: int,
) -> list[NoteEvent]:
    """Generate an arpeggio over one chord.

    Phase 1 keeps it simple: 4 quarter notes per bar at 4/4 (or the
    equivalent subdivision), cycling through the chord tones starting
    on a seed-derived index. The exact rhythm is a seed-driven
    choice between quarter and eighth notes for variety.
    """
    notes: list[NoteEvent] = []
    rhythm_choices = (PPQ, PPQ // 2)  # quarter or eighth notes
    note_length = rng.choice(rhythm_choices)
    beat_count = duration_ticks // note_length
    if beat_count < 1:
        beat_count = 1
    starting_tone_index = seed_for_variation % len(chord_tones)
    for i in range(beat_count):
        tone_index = (starting_tone_index + i) % len(chord_tones)
        pitch = chord_root + chord_tones[tone_index]
        # Cap melody at piano range.
        if pitch > 107:
            pitch -= 12
        if pitch < 22:
            pitch += 12
        notes.append(
            NoteEvent(
                voice_id=1,
                pitch_midi=pitch,
                tick=start_tick + i * note_length,
                duration_ticks=note_length,
                velocity=_velocity_for_section(
                    seed_for_variation + i, base=DEFAULT_VELOCITY, jitter=12
                ),
            )
        )
    return notes


# ---------------------------------------------------------------------------
# Performance plan: convert tick-level notation into microsecond timestamps.
# ---------------------------------------------------------------------------


def _build_performance_plan(score: NotationScore) -> PerformancePlan:
    events: list[PerformanceNoteEvent] = []
    for note in score.notes:
        start_us = ticks_to_microseconds(note.tick, score.tempo.bpm, ppq=score.ppq)
        duration_us = ticks_to_microseconds(note.duration_ticks, score.tempo.bpm, ppq=score.ppq)
        events.append(
            PerformanceNoteEvent(
                voice_id=note.voice_id,
                pitch_midi=note.pitch_midi,
                start_us=start_us,
                duration_us=duration_us,
                velocity=note.velocity,
                tie=note.tie,
            )
        )
    return PerformancePlan.make(sample_rate=44100, notes=events)


__all__ = [
    "CompositionEngineError",
    "EngineErrorCode",
    "EngineOutput",
    "compose",
]
