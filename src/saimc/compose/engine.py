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
from dataclasses import dataclass, replace
from enum import StrEnum
from itertools import pairwise
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
    apply_final_cadence,
    get_template_for_form,
    key_root_midi,
    key_signature_from_spec,
)
from saimc.compose.linter import LintIssue, lint
from saimc.compose.motif import MotifVariant, generate_motif, vary_motif
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
    ControllerEvent,
    KeySignature,
    Measure,
    NotationScore,
    NoteEvent,
    PerformanceNoteEvent,
    PerformancePlan,
    PitchBendEvent,
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
            controllers=tuple(
                ControllerEvent(**c) for c in plan_payload.get("controllers", ())
            ),
            pitch_bends=tuple(
                PitchBendEvent(**b) for b in plan_payload.get("pitch_bends", ())
            ),
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
    try:
        key = key_signature_from_spec(spec)
    except ValueError as exc:
        raise CompositionEngineError(
            code=EngineErrorCode.INVALID_SPEC,
            message=str(exc),
        ) from exc
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

    performance = _build_performance_plan(
        score,
        instrumentation=spec.instrumentation.value,
        humanization=spec.humanization,
        seed=spec.seed,
    )
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
        if section_idx == arrangement.repetition_count - 1:
            # The last repetition must land at home: rewrite its last
            # two bars as the mood's cadence (earlier sections may end
            # open — their V resolves into the next section's I).
            section_template = apply_final_cadence(section_template, spec.mood.value)
        section_notes = _generate_section(
            key=key,
            time_signature=time_signature,
            template=section_template,
            section_start_tick=cursor_tick,
            rng=section_rng,
            seed_for_variation=rng_base_seed + section_idx,
            is_final_section=section_idx == arrangement.repetition_count - 1,
        )
        notes.extend(section_notes)
        cursor_tick += arrangement.form_bars * bar_ticks(time_signature)

    # Optional coda: append a coda-length tail using the same chord
    # template (truncated to coda_bars). The coda gets its own RNG
    # seed (a stable offset from the spec seed) so it sounds distinct
    # from the body, per §10 #10. It is also the piece's true ending,
    # so it carries the final cadence.
    if arrangement.coda_bars > 0:
        coda_template = _truncate_template_for_coda(arrangement.template, arrangement.coda_bars)
        coda_template = apply_final_cadence(coda_template, spec.mood.value)
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
    is_final_section: bool = False,
) -> list[NoteEvent]:
    """Generate the bass + melody notes for one section.

    The left hand plays a root-fifth broken pattern (root on the
    downbeat, fifth at the bar's midpoint) instead of a held drone.
    The melody is an octave above the bass and develops the section's
    motif: every bar replays the motif through one classic operation
    (repetition, transposition, sequence, inversion, truncation,
    ornament) onto that bar's chord. Velocity follows an arch across
    the section with a slight accent on downbeats. Everything is
    derived from `seed_for_variation`, so repeated sections sound
    different but stay deterministic.

    Phrase shape: one bar per section is the melodic apex (raised an
    octave-portion above the line, near the 60% mark); bars ending a
    4-bar phrase lift off early into a breath — on the dominant's root
    when the chord there is the V (a half cadence), otherwise on a
    shortened chord tone. When `is_final_section` is set the last bar
    resolves onto the tonic or its third, held to the bar line.
    """
    notes: list[NoteEvent] = []
    tonic_midi = key_root_midi(key)
    ticks_per_bar = bar_ticks(time_signature)
    section_ticks = template.bars * ticks_per_bar
    # The melodic apex sits near the 60% mark, never on the final bar.
    apex_bar = min(int(template.bars * 0.6), template.bars - 2)
    motif = generate_motif(rng, bar_ticks=ticks_per_bar)

    cursor = 0
    bar_index = 0
    total_bars = template.bars
    for degree, dur in template.chords:
        chord_root = tonic_midi + _scale_degree_to_semitones(degree, key.mode)
        chord_tones = _chord_intervals(degree, key)
        chord_root_tick = section_start_tick + cursor

        for _bar in range(dur):
            bar_tick = chord_root_tick + _bar * ticks_per_bar
            bar_pos = (cursor + _bar * ticks_per_bar) / max(1, section_ticks)
            bass_root = _octave_down(chord_root, octaves=1)
            # The upper voice of the open fifth follows the triad quality:
            # a perfect fifth on major/minor chords, a diminished fifth
            # (6 semitones) on the ii°/vii° chords the templates use.
            bass_fifth = min(107, bass_root + chord_tones[2] - chord_tones[0])

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

            # Melody voice: one bar derived from the section's motif.
            # The last bar of the piece resolves at home; a
            # phrase-ending bar over the V chord is a half cadence; a
            # random bar lifts off early into a breath.
            is_final_bar = is_final_section and bar_index == total_bars - 1
            is_apex = bar_index == apex_bar
            is_half_cadence = degree == 4 and bar_index % 4 == 3 and not is_final_bar
            breathe = (
                not is_final_bar and not is_half_cadence and rng.random() < 0.18
            )
            anchor = _downbeat_anchor(rng, len(chord_tones))
            if is_final_bar or is_half_cadence or is_apex or breathe:
                variant = MotifVariant(motif=motif)
            else:
                variant = vary_motif(motif, rng)
            notes.extend(
                _melody_bar(
                    variant=variant,
                    chord_root=chord_root + 12,
                    chord_tones=chord_tones,
                    anchor=anchor,
                    start_tick=bar_tick,
                    bar_ticks=ticks_per_bar,
                    rng=rng,
                    position=bar_pos,
                    ticks_per_bar=ticks_per_bar,
                    seed_for_variation=seed_for_variation + bar_index * 101,
                    is_final_bar=is_final_bar,
                    half_cadence=is_half_cadence,
                    apex=is_apex,
                    breathe=breathe,
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


def _chord_intervals(degree: int, key: KeySignature) -> tuple[int, ...]:
    """Return the root-relative chord-tone intervals for a diatonic chord.

    Major key: I, ii, iii, IV, V, vi, vii -> major, minor, minor, major,
    major, minor, dim. Minor key: i, ii°, III, iv, v, VI, VII -> minor,
    dim, major, minor, minor, major, major. Each entry is an interval
    above the chord root (e.g. (0, 3, 7) is a minor triad), so callers
    add these to the chord root — not to the scale-degree root — to get
    absolute pitches.
    """
    major_triads = (
        (0, 4, 7),
        (0, 3, 7),
        (0, 3, 7),
        (0, 4, 7),
        (0, 4, 7),
        (0, 3, 7),
        (0, 3, 6),
    )
    minor_triads = (
        (0, 3, 7),
        (0, 3, 6),
        (0, 4, 7),
        (0, 3, 7),
        (0, 3, 7),
        (0, 4, 7),
        (0, 4, 7),
    )
    table = major_triads if key.mode == "major" else minor_triads
    return table[degree % 7]


def _octave_down(midi: int, *, octaves: int) -> int:
    """Shift a MIDI note down by `octaves` octaves, floored at 21."""
    return max(21, midi - 12 * octaves)


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


def _downbeat_anchor(rng: random.Random, tone_count: int) -> int:
    """Choose a bar's starting chord tone: root (50%), third (30%), fifth (20%)."""
    roll = rng.random()
    if roll < 0.5:
        return 0
    if roll < 0.8:
        return 1 % tone_count
    return 2 % tone_count


def _melody_bar(
    *,
    variant: MotifVariant,
    chord_root: int,
    chord_tones: tuple[int, ...],
    anchor: int,
    start_tick: int,
    bar_ticks: int,
    rng: random.Random,
    position: float,
    ticks_per_bar: int,
    seed_for_variation: int,
    is_final_bar: bool = False,
    half_cadence: bool = False,
    apex: bool = False,
    breathe: bool = False,
) -> list[NoteEvent]:
    """Render one bar of melody from a motif variant.

    The motif is walked in chord-tone index space starting at `anchor`,
    so the same shape lands correctly on every chord. When `repeat` is
    set (the sequence operation) the motif keeps replaying from the top
    — anchor advancing one tone per cycle — until the bar is full.

    Phrase shape (carried over from the arpeggio walk):
    - an `apex` bar is lifted an octave (capped to range) with a
      velocity lift — the section's melodic peak;
    - a `half_cadence` bar ends early on the chord's root, leaving a
      rest (the phrase breathes on the V);
    - a breathing bar shortens its last note into a rest;
    - the `is_final_bar` of the piece resolves onto the tonic or its
      third, held to the bar line.
    """
    # Collect (offset, duration, tone_index) slots first, then resolve
    # pitches — the bar's last note can be replaced wholesale by the
    # cadence/breath shape.
    slots: list[tuple[int, int, int]] = []
    offset = 0
    cycle_anchor = anchor % len(chord_tones)
    while offset < bar_ticks:
        tone_index = cycle_anchor
        for cell in variant.motif:
            if offset >= bar_ticks:
                break
            duration = min(cell.length_ticks, bar_ticks - offset)
            slots.append((offset, duration, tone_index))
            offset += cell.length_ticks
            tone_index += cell.step
        if not variant.repeat:
            break
        cycle_anchor += 1

    notes: list[NoteEvent] = []
    for slot_index, (bar_offset, duration, tone_index) in enumerate(slots):
        tick = start_tick + bar_offset
        if slot_index == len(slots) - 1 and is_final_bar:
            # The piece ends at home: tonic or its third, held to the
            # bar line.
            pitch = chord_root if rng.random() < 0.6 else chord_root + chord_tones[1]
            duration = bar_ticks - bar_offset
        else:
            pitch = chord_root + chord_tones[tone_index % len(chord_tones)]
            if slot_index == len(slots) - 1 and half_cadence:
                # Land on the chord's root, lifted early so a rest
                # follows.
                pitch = chord_root + chord_tones[0]
                duration = duration // 2
            elif slot_index == len(slots) - 1 and breathe:
                duration = duration // 2
            if apex:
                pitch += 12
            # Cap melody at piano range.
            if pitch > 107:
                pitch -= 12
            if pitch < 22:
                pitch += 12
        notes.append(
            NoteEvent(
                voice_id=VOICE_MELODY,
                pitch_midi=pitch,
                tick=tick,
                duration_ticks=duration,
                velocity=_shaped_velocity(
                    base=DEFAULT_VELOCITY + 8 + (10 if apex else 0),
                    position=position,
                    tick=tick,
                    ticks_per_bar=ticks_per_bar,
                    rng_seed=seed_for_variation + tick,
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
# Performance plan: convert tick-level notation into microsecond timestamps,
# then lay the expression layer on top (the engraved score stays on-grid).
# ---------------------------------------------------------------------------

# Sustained instruments: their notes blur into one another, so the
# performance layer lets each melody note ring slightly into the next
# (legato). Percussive, plucked, and mallet instruments keep their
# notated durations — that attack gap IS their articulation.
SUSTAINED_INSTRUMENTS: frozenset[str] = frozenset(
    {
        "violin", "viola", "cello", "contrabass", "fiddle", "strings",
        "tremolo_strings", "flute", "piccolo", "recorder", "pan_flute",
        "ocarina", "oboe", "english_horn", "bassoon", "clarinet",
        "soprano_sax", "alto_sax", "tenor_sax", "baritone_sax",
        "french_horn", "brass_section", "trumpet", "muted_trumpet",
        "trombone", "tuba", "choir", "pipe_organ", "accordion",
        "harmonica", "sitar", "harmonium", "bansuri", "sarangi",
        "rudra_veena", "sarasvati_veena", "qanoon", "ud", "kora",
        "shakuhachi", "shanai", "bagpipe",
    }
)

# Keyboard instruments that read a sustain pedal; organ voices sustain
# by themselves and gain nothing from CC64.
PEDAL_INSTRUMENTS: frozenset[str] = frozenset(
    {"piano", "harpsichord", "celesta", "music_box"}
)

# How far a legato note rings past its written end (never past the next
# note's start: a same-pitch retrigger would re-attack the line).
LEGATO_OVERLAP_US: int = 35_000

# Pedal lifts a moment before each bar line so chords do not blur across
# the bar; the lift is a fixed 40 ms before the next downbeat.
PEDAL_RELEASE_LEAD_US: int = 40_000

# Humanization profiles (spec.humanization): timing scatter on the
# realized timestamps and how far velocities spread around their neutral
# 64 centre. 'none' applies neither.
HUMANIZE_TIMING_US: dict[str, int] = {"light": 10_000, "expressive": 25_000}
HUMANIZE_VELOCITY_SPAN: dict[str, float] = {"light": 1.15, "expressive": 1.35}
PERCUSSION_TIMING_US: dict[str, int] = {"light": 5_000, "expressive": 15_000}

# Percussion ghost notes: quiet extra hits that make the kit feel played
# rather than sequenced.
GHOST_NOTE_PROBABILITY: float = 0.08
GHOST_NOTE_VELOCITY_RANGE: tuple[int, int] = (20, 35)

# CC11 (expression) rides the dynamic arch so phrases swell and relax
# even inside a held chord. 96 is near-full expression at the arch peak.
EXPRESSION_BASE: int = 96


def _build_performance_plan(
    score: NotationScore,
    *,
    instrumentation: str,
    humanization: str,
    seed: int | None,
) -> PerformancePlan:
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

    melody_sorted_idx = sorted(
        (i for i, e in enumerate(events) if e.voice_id == VOICE_MELODY),
        key=lambda i: events[i].start_us,
    )
    has_melody = bool(melody_sorted_idx)

    # Legato: sustained instruments let each melody note ring a little
    # past the next attack so the release tail blurs into the next note.
    # A same-pitch neighbour is capped at its start: a late note_off on
    # the same key would re-attack or cut the line.
    legato_extended: set[int] = set()
    if instrumentation in SUSTAINED_INSTRUMENTS:
        for a, b in pairwise(melody_sorted_idx):
            prev, curr = events[a], events[b]
            gap = curr.start_us - prev.start_us
            limit = gap if prev.pitch_midi == curr.pitch_midi else gap + LEGATO_OVERLAP_US
            extended = min(prev.duration_us + LEGATO_OVERLAP_US, limit)
            if extended > prev.duration_us:
                legato_extended.add(a)
                events[a] = PerformanceNoteEvent(
                    voice_id=prev.voice_id,
                    pitch_midi=prev.pitch_midi,
                    start_us=prev.start_us,
                    duration_us=extended,
                    velocity=prev.velocity,
                    tie=prev.tie,
                )

    controllers: list[ControllerEvent] = []
    if has_melody:
        # Expression swells ride the same arch as the velocities, one
        # value per bar, so phrases breathe in the controller layer too.
        total_ticks = max(1, score.total_ticks())
        for measure in score.measures:
            position = measure.start_tick / total_ticks
            value = min(127, round(EXPRESSION_BASE * _velocity_arc(position)))
            controllers.append(
                ControllerEvent(
                    voice_id=VOICE_MELODY,
                    control=11,
                    value=value,
                    start_us=ticks_to_microseconds(
                        measure.start_tick, score.tempo.bpm, ppq=score.ppq
                    ),
                )
            )
        if instrumentation in PEDAL_INSTRUMENTS:
            # Per-bar pedaling: press on the downbeat, lift just before
            # the next one so chords do not wash across the bar line.
            for i, measure in enumerate(score.measures):
                press_us = ticks_to_microseconds(
                    measure.start_tick, score.tempo.bpm, ppq=score.ppq
                )
                controllers.append(
                    ControllerEvent(
                        voice_id=VOICE_MELODY, control=64, value=127, start_us=press_us
                    )
                )
                if i + 1 < len(score.measures):
                    next_downbeat = ticks_to_microseconds(
                        score.measures[i + 1].start_tick, score.tempo.bpm, ppq=score.ppq
                    )
                    controllers.append(
                        ControllerEvent(
                            voice_id=VOICE_MELODY,
                            control=64,
                            value=0,
                            start_us=max(0, next_downbeat - PEDAL_RELEASE_LEAD_US),
                        )
                    )

    # Humanization touches only this plan — the NotationScore (and the
    # engraved sheet) keeps its grid-perfect timing.
    if humanization != "none":
        rng = random.Random(((seed or 0) * 2654435761 + 11) % (2**31))
        timing_us = HUMANIZE_TIMING_US.get(humanization, HUMANIZE_TIMING_US["light"])
        velocity_span = HUMANIZE_VELOCITY_SPAN.get(
            humanization, HUMANIZE_VELOCITY_SPAN["light"]
        )
        perc_timing_us = PERCUSSION_TIMING_US.get(humanization, PERCUSSION_TIMING_US["light"])
        humanized: list[PerformanceNoteEvent] = []
        for event in events:
            if event.voice_id == VOICE_MELODY:
                offset = round(rng.uniform(-1.0, 1.0) * timing_us)
                velocity = max(1, min(127, round(64 + (event.velocity - 64) * velocity_span)))
                humanized.append(
                    PerformanceNoteEvent(
                        voice_id=event.voice_id,
                        pitch_midi=event.pitch_midi,
                        start_us=max(0, event.start_us + offset),
                        duration_us=event.duration_us,
                        velocity=velocity,
                        tie=event.tie,
                    )
                )
            elif event.voice_id == VOICE_PERCUSSION:
                offset = round(rng.uniform(-1.0, 1.0) * perc_timing_us)
                humanized.append(
                    PerformanceNoteEvent(
                        voice_id=event.voice_id,
                        pitch_midi=event.pitch_midi,
                        start_us=max(0, event.start_us + offset),
                        duration_us=event.duration_us,
                        velocity=event.velocity,
                        tie=event.tie,
                    )
                )
            else:
                humanized.append(event)
        events = humanized

        # Timing scatter can push one note past its neighbour's start on
        # back-to-back melody lines; trim the release so the line never
        # overlaps itself. Percussion hits are left alone — a few
        # milliseconds of overlap between drum voices is inaudible.
        melody_idx = sorted(
            (i for i, e in enumerate(events) if e.voice_id == VOICE_MELODY),
            key=lambda i: events[i].start_us,
        )
        for a, b in pairwise(melody_idx):
            prev, curr = events[a], events[b]
            gap = curr.start_us - prev.start_us
            if a not in legato_extended and 0 < gap < prev.duration_us:
                events[a] = replace(prev, duration_us=max(1, gap))

        if humanization == "expressive":
            # Repeated melody notes shorten into a light staccato
            # instead of two identical full-length hits.
            melody_events = sorted(
                (e for e in events if e.voice_id == VOICE_MELODY), key=lambda e: e.start_us
            )
            shortened = {
                id(prev)
                for prev, curr in pairwise(melody_events)
                if prev.pitch_midi == curr.pitch_midi
                and curr.start_us - prev.start_us < prev.duration_us * 2
            }
            events = [
                (
                    PerformanceNoteEvent(
                        voice_id=e.voice_id,
                        pitch_midi=e.pitch_midi,
                        start_us=e.start_us,
                        duration_us=round(e.duration_us * 0.8),
                        velocity=e.velocity,
                        tie=e.tie,
                    )
                    if id(e) in shortened
                    else e
                )
                for e in events
            ]

        # Ghost notes: a quiet extra hit a 16th after some percussion
        # notes, skipped when a real hit already occupies the slot.
        percussion = [e for e in events if e.voice_id == VOICE_PERCUSSION]
        if percussion:
            sixteenth_us = round(60_000_000 / score.tempo.bpm / 4)
            ghosts: list[PerformanceNoteEvent] = []
            for event in percussion:
                if rng.random() >= GHOST_NOTE_PROBABILITY:
                    continue
                ghost_start = event.start_us + sixteenth_us
                if any(
                    other.pitch_midi == event.pitch_midi
                    and abs(other.start_us - ghost_start) < 20_000
                    for other in percussion
                ):
                    continue
                ghosts.append(
                    PerformanceNoteEvent(
                        voice_id=VOICE_PERCUSSION,
                        pitch_midi=event.pitch_midi,
                        start_us=ghost_start,
                        duration_us=event.duration_us,
                        velocity=rng.randint(*GHOST_NOTE_VELOCITY_RANGE),
                        tie=False,
                    )
                )
            events.extend(ghosts)

    # Canonical order for the plan: realized time, then voice, then pitch.
    events.sort(key=lambda e: (e.start_us, e.voice_id, e.pitch_midi))
    controllers.sort(key=lambda c: (c.start_us, c.control, c.voice_id))
    return PerformancePlan.make(
        sample_rate=44100, notes=events, controllers=controllers, pitch_bends=[]
    )


__all__ = [
    "CompositionEngineError",
    "EngineErrorCode",
    "EngineOutput",
    "compose",
]
