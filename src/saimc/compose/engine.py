"""Phase 1 composition engine.

Takes a CompositionSpec and returns (NotationScore, PerformancePlan).
The engine is rule-based, deterministic, and piano-only; chord tones
and key handling come from pure interval tables in this module and
`saimc.compose.forms` — no music21 dependency at generation time.

Stages (per `docs/roadmap.md` §2 step 3):

1. Resolve key + time signature from the spec.
2. Pick a mood-appropriate chord template (form).
3. Apply the §10 #1 duration policy to find the form, repetition
   count, and tempo that fit the spec's `duration_seconds` within
   ±2% tolerance.
4. Generate one melody voice + one bass voice over the chord
   progression:
   - the left hand plays a root-fifth broken pattern instead of a
     held drone,
   - the melody sits an octave above the bass to keep the registers
     separate,
   - rhythm, arpeggio direction, and starting tone vary per bar
     (seeded),
   - velocity follows an arch across the section with beat accents.
   Repeated sections rotate through the mood's chord-template
   variants (A/B form) and the coda closes on the tonic.
5. Theory-lint the resulting NotationScore.
6. Build the PerformancePlan with integer-microsecond timestamps.

The engine raises `CompositionEngineError` (with a stable code) for
any failure the calling layer should react to. Lint failures become
a `lint_failed` error before the engine returns.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from saimc.compose.duration import (
    DurationArrangement,
    DurationUnfulfillableError,
    arrange_for_duration,
    bar_ticks,
    section_seed,
)
from saimc.compose.forms import (
    ChordTemplate,
    get_template_for_form,
    key_root_midi,
    key_signature_from_spec,
)
from saimc.compose.linter import LintIssue, lint
from saimc.compose.percussion import (
    MOOD_VELOCITY_SCALE,
    PERCUSSION_NOTE_TICKS,
    PERCUSSION_VELOCITY_MAX,
    style_for,
)
from saimc.compose.score import (
    DEFAULT_VELOCITY,
    PPQ,
    VOICE_BASS,
    VOICE_MELODY,
    VOICE_PERCUSSION,
    KeySignature,
    Measure,
    NotationScore,
    NoteEvent,
    PerformanceNoteEvent,
    PerformancePlan,
    TempoMap,
    ticks_to_microseconds,
)
from saimc.spec import CompositionSpec, Instrument


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

    def to_sidecar(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict for the sidecar file.

        The compose types are plain `@dataclass(frozen=True)`, not
        Pydantic, so we use `dataclasses.asdict` for the conversion.
        """
        from dataclasses import asdict

        return {
            "notation_score": asdict(self.notation_score),
            "performance_plan": asdict(self.performance_plan),
            "arrangement": asdict(self.arrangement),
            "key": asdict(self.key),
            "time_signature": self.time_signature,
        }

    @classmethod
    def from_sidecar(cls, payload: dict[str, Any]) -> EngineOutput:
        """Reconstruct from the sidecar JSON dict.

        Nested dataclasses (`NotationScore`, `PerformancePlan`,
        `DurationArrangement`, `KeySignature`, `ChordTemplate`,
        `Measure`, `NoteEvent`, `PerformanceNoteEvent`) are rebuilt
        with their constructors by name; `asdict` collapses them
        into plain `dict`s, so we rehydrate each one explicitly.
        """
        score_payload = payload["notation_score"]
        plan_payload = payload["performance_plan"]
        arrangement_payload = payload["arrangement"]
        score = NotationScore(
            format=score_payload["format"],
            ppq=score_payload["ppq"],
            key=KeySignature(**score_payload["key"]),
            time_signature=score_payload["time_signature"],
            tempo=TempoMap(**score_payload["tempo"]),
            measures=tuple(Measure(**m) for m in score_payload["measures"]),
            notes=tuple(NoteEvent(**n) for n in score_payload["notes"]),
        )
        plan = PerformancePlan(
            format=plan_payload["format"],
            sample_rate=plan_payload["sample_rate"],
            notes=tuple(PerformanceNoteEvent(**n) for n in plan_payload["notes"]),
        )
        arrangement = DurationArrangement(
            form_bars=arrangement_payload["form_bars"],
            template=ChordTemplate(
                name=arrangement_payload["template"]["name"],
                bars=arrangement_payload["template"]["bars"],
                chords=tuple(tuple(chord) for chord in arrangement_payload["template"]["chords"]),
            ),
            repetition_count=arrangement_payload["repetition_count"],
            total_bars=arrangement_payload["total_bars"],
            tempo_bpm=arrangement_payload["tempo_bpm"],
            coda_bars=arrangement_payload.get("coda_bars", 0),
        )
        return cls(
            notation_score=score,
            performance_plan=plan,
            arrangement=arrangement,
            key=KeySignature(**payload["key"]),
            time_signature=payload["time_signature"],
        )


def compose(spec: CompositionSpec) -> EngineOutput:
    """Run the full composition pipeline against the spec."""
    key = key_signature_from_spec(spec)
    time_signature = spec.time_signature.value

    try:
        arrangement = arrange_for_duration(
            mood=spec.mood.value,
            target_duration_seconds=float(spec.duration_seconds),
            time_signature=time_signature,
            tempo_bpm=float(spec.tempo_bpm) if spec.tempo_bpm is not None else None,
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
        # Rotate the chord-template variants across repetitions: the
        # first section carries the arrangement's template, later ones
        # cycle the mood's remaining variants (an A/B form) so repeats
        # differ harmonically, not just in surface rhythm.
        if section_idx == 0:
            section_template = arrangement.template
        else:
            section_template = get_template_for_form(
                spec.mood.value,
                arrangement.form_bars,
                variant_index=section_idx,
            )
        section_notes = _generate_section(
            key=key,
            time_signature=time_signature,
            template=section_template,
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

    # Drum set: when the piece is written for the kit, the piano stays
    # as the accompaniment and a percussion voice plays the mood's
    # rhythm pattern in every bar (voice 2, GM channel-10 keys). Styles
    # with A/B variants rotate across sections, matching the chord
    # templates' variation rule. Melodic meters the library does not
    # cover (5/4, 7/8) get no percussion rather than a wrong pattern.
    if spec.instrumentation == Instrument.DRUM_SET:
        notes.extend(
            _generate_percussion(
                mood=spec.mood.value,
                time_signature=time_signature,
                form_bars=arrangement.form_bars,
                repetition_count=arrangement.repetition_count,
                total_bars=arrangement.total_bars_with_coda,
                seed=rng_base_seed,
            )
        )

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

    The left hand plays a root-fifth broken pattern (root on the
    downbeat, fifth at the bar's midpoint) instead of a held drone.
    The melody is an octave above the bass and arpeggiates the chord
    tones with per-bar rhythm and direction variation. Velocity
    follows an arch across the section with a slight accent on
    downbeats. Everything is derived from `seed_for_variation`, so
    repeated sections sound different but stay deterministic.
    """
    notes: list[NoteEvent] = []
    tonic_midi = key_root_midi(key)
    ticks_per_bar = bar_ticks(time_signature)
    section_ticks = template.bars * ticks_per_bar

    cursor = 0
    bar_index = 0
    for degree, dur in template.chords:
        chord_root = tonic_midi + _scale_degree_to_semitones(degree, key.mode)
        chord_tones = _chord_tones_midi(degree, key)
        chord_root_tick = section_start_tick + cursor

        for _bar in range(dur):
            bar_tick = chord_root_tick + _bar * ticks_per_bar
            bar_pos = (cursor + _bar * ticks_per_bar) / max(1, section_ticks)
            bass_root = _octave_down(chord_root, octaves=1)
            bass_fifth = min(107, bass_root + 7)

            # Left hand: root on the downbeat, fifth at the midpoint.
            half = ticks_per_bar // 2
            notes.append(
                NoteEvent(
                    voice_id=VOICE_BASS,
                    pitch_midi=bass_root,
                    tick=bar_tick,
                    duration_ticks=half,
                    velocity=_shaped_velocity(
                        base=56,
                        position=bar_pos,
                        tick=bar_tick,
                        ticks_per_bar=ticks_per_bar,
                        rng_seed=seed_for_variation,
                    ),
                )
            )
            notes.append(
                NoteEvent(
                    voice_id=VOICE_BASS,
                    pitch_midi=bass_fifth,
                    tick=bar_tick + half,
                    duration_ticks=ticks_per_bar - half,
                    velocity=_shaped_velocity(
                        base=50,
                        position=bar_pos,
                        tick=bar_tick + half,
                        ticks_per_bar=ticks_per_bar,
                        rng_seed=seed_for_variation,
                    ),
                )
            )

            # Melody voice: one bar of arpeggio, rhythm and starting
            # tone re-chosen each bar.
            notes.extend(
                _arpeggiate_bar(
                    chord_root=chord_root + 12,
                    chord_tones=chord_tones,
                    start_tick=bar_tick,
                    bar_ticks=ticks_per_bar,
                    rng=rng,
                    position=bar_pos,
                    ticks_per_bar=ticks_per_bar,
                    seed_for_variation=seed_for_variation + bar_index * 101,
                )
            )
            bar_index += 1

        cursor += dur * ticks_per_bar

    # Canonical order: the bass and melody interleave within a bar, so
    # sort by (tick, voice, pitch) rather than relying on append order.
    return sorted(notes, key=lambda n: (n.tick, n.voice_id, n.pitch_midi))


def _truncate_template_for_coda(template: ChordTemplate, coda_bars: int) -> ChordTemplate:
    """Return a coda-sized prefix of `template`, closing on the tonic.

    The coda is a sub-form: a coda_bars-bar prefix of the form's
    template, using the first chord cycles that fit. Its final chord
    is forced to the tonic (degree 0) so the piece ends with a
    cadence home rather than on whatever chord the truncation lands
    on. Coda length is always strictly less than the form's full
    length. Zero-duration chord entries are dropped.
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
    if kept:
        last_degree, last_dur = kept[-1]
        kept[-1] = (0, last_dur) if last_degree != 0 else (last_degree, last_dur)
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


def _velocity_arc(position: float) -> float:
    """A gentle dynamic arch: quiet entrances and exits, fuller middle.

    `position` is 0..1 across the section. The arc spans roughly 0.8x
    to 1.2x so phrases breathe without any note becoming extreme.
    """
    pos = min(1.0, max(0.0, position))
    return 0.8 + 0.4 * math.sin(math.pi * pos)


def _shaped_velocity(
    *, base: int, position: float, tick: int, ticks_per_bar: int, rng_seed: int
) -> int:
    """Base velocity shaped by the section arch + downbeat accent + jitter.

    Deterministic: the jitter draws from a per-tick-seeded RNG so the
    same seed reproduces the exact same performance.
    """
    accent = 6 if tick % ticks_per_bar == 0 else 0
    jitter = random.Random(rng_seed * 31 + tick).randint(-4, 4)
    shaped = base * _velocity_arc(position) + accent + jitter
    return max(1, min(127, round(shaped)))


def _arpeggiate_bar(
    *,
    chord_root: int,
    chord_tones: tuple[int, ...],
    start_tick: int,
    bar_ticks: int,
    rng: random.Random,
    position: float,
    ticks_per_bar: int,
    seed_for_variation: int,
) -> list[NoteEvent]:
    """Generate one bar of melody arpeggio.

    Rhythm (quarter vs. eighth subdivision), direction, and starting
    tone are all re-chosen per bar from the section RNG, so a long
    chord keeps moving instead of cycling a fixed figure. The melody
    sits an octave above the chord root handed in by the caller.
    """
    notes: list[NoteEvent] = []
    rhythm_choices = (PPQ, PPQ // 2)  # quarter or eighth notes
    note_length = rng.choice(rhythm_choices)
    step = 1 if rng.random() < 0.7 else -1  # mostly rising figures
    count = bar_ticks // note_length
    if count < 1:
        count = 1
    starting_tone_index = rng.randrange(len(chord_tones))
    for i in range(count):
        tone_index = (starting_tone_index + step * i) % len(chord_tones)
        pitch = chord_root + chord_tones[tone_index]
        # Cap melody at piano range.
        if pitch > 107:
            pitch -= 12
        if pitch < 22:
            pitch += 12
        notes.append(
            NoteEvent(
                voice_id=VOICE_MELODY,
                pitch_midi=pitch,
                tick=start_tick + i * note_length,
                duration_ticks=note_length,
                velocity=_shaped_velocity(
                    base=DEFAULT_VELOCITY + 8,
                    position=position,
                    tick=start_tick + i * note_length,
                    ticks_per_bar=ticks_per_bar,
                    rng_seed=seed_for_variation + i,
                ),
            )
        )
    return notes


def _generate_percussion(
    *,
    mood: str,
    time_signature: str,
    form_bars: int,
    repetition_count: int,
    total_bars: int,
    seed: int,
) -> list[NoteEvent]:
    """Generate the percussion voice for a drum-set piece.

    The style comes from the mood + meter (`style_for`); its variants
    rotate across sections the way the chord templates do, with the
    coda treated as one more section. A per-bar seeded jitter of a few
    velocity points keeps repeated bars from sounding machine-stamped.
    """
    style = style_for(mood, time_signature)
    if style is None:
        return []
    ticks_per_bar = bar_ticks(time_signature)
    mood_scale = MOOD_VELOCITY_SCALE.get(mood, 1.0)
    notes: list[NoteEvent] = []
    for bar in range(total_bars):
        in_body = bar < repetition_count * form_bars
        section_idx = bar // form_bars if in_body else repetition_count
        pattern = style.pattern(time_signature, section_idx)
        if pattern is None:
            continue
        bar_rng = random.Random(seed + bar)
        bar_start = bar * ticks_per_bar
        for hit in pattern:
            jitter = bar_rng.uniform(0.92, 1.06)
            velocity = round(hit.velocity * style.velocity_scale * mood_scale * jitter)
            notes.append(
                NoteEvent(
                    voice_id=VOICE_PERCUSSION,
                    pitch_midi=hit.key,
                    tick=bar_start + hit.offset_ticks,
                    duration_ticks=PERCUSSION_NOTE_TICKS,
                    velocity=min(PERCUSSION_VELOCITY_MAX, max(1, velocity)),
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
