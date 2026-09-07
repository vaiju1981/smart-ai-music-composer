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
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from itertools import pairwise
from typing import Any

from saimc.compose.duration import (
    ARRANGEMENT_ARC_MIN_REPS,
    RITARDANDO_BARS,
    DurationArrangement,
    DurationUnfulfillableError,
    arrange_for_duration,
    bar_ticks,
    section_seed,
)
from saimc.compose.forms import (
    PHRASE_BARS,
    ChordSlot,
    ChordTemplate,
    apply_final_cadence,
    chord_intervals,
    get_template_for_form,
    key_root_midi,
    key_signature_from_spec,
)
from saimc.compose.linter import LintIssue, lint
from saimc.compose.motif import (
    BarSlot,
    MotifVariant,
    apply_rhythm,
    generate_motif,
    vary_motif,
)
from saimc.compose.percussion import (
    DRUM_CRASH,
    DRUM_KICK,
    MOOD_VELOCITY_SCALE,
    PERCUSSION_NOTE_TICKS,
    PERCUSSION_VELOCITY_MAX,
    SECTION_CRASH_VELOCITY,
    rotation_index,
    style_for,
)
from saimc.compose.score import (
    DEFAULT_VELOCITY,
    PPQ,
    VOICE_BASS,
    VOICE_HARMONY,
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
    TempoPoint,
    microseconds_at_tick,
)
from saimc.compose.ensemble import Ensemble, resolve_ensemble
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
class VoiceInstrument:
    """Which instrument renders one engine voice (a sidecar row).

    A list of these (not a dict keyed by voice id) because JSON object
    keys are strings — the int voice ids would not round-trip.
    """

    voice_id: int
    instrument: str


@dataclass(frozen=True)
class EngineOutput:
    """The engine's output: NotationScore + PerformancePlan + arrangement metadata.

    `chord_bars` carries the chord pitch classes sounding in each bar
    (bar order) so the release gates can re-run the linter's
    chord-tone check without regenerating the harmony. `voice_instruments`
    maps each engine voice to the instrument that renders it — the
    renderers' source of truth for per-voice programs and channels.
    """

    notation_score: NotationScore
    performance_plan: PerformancePlan
    arrangement: DurationArrangement
    key: KeySignature
    time_signature: str
    chord_bars: tuple[tuple[int, ...], ...] = ()
    voice_instruments: tuple[VoiceInstrument, ...] = ()

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
            "chord_bars": [list(bar) for bar in self.chord_bars],
            "voice_instruments": [asdict(v) for v in self.voice_instruments],
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
        tempo_payload = score_payload["tempo"]
        score = NotationScore(
            format=score_payload["format"],
            ppq=score_payload["ppq"],
            key=KeySignature(**score_payload["key"]),
            time_signature=score_payload["time_signature"],
            tempo=TempoMap(
                bpm=tempo_payload["bpm"],
                ppq=tempo_payload["ppq"],
                changes=tuple(TempoPoint(**c) for c in tempo_payload.get("changes", ())),
            ),
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
                chords=tuple(ChordSlot(*chord) for chord in arrangement_payload["template"]["chords"]),
            ),
            repetition_count=arrangement_payload["repetition_count"],
            total_bars=arrangement_payload["total_bars"],
            tempo_bpm=arrangement_payload["tempo_bpm"],
            coda_bars=arrangement_payload.get("coda_bars", 0),
            intro_bars=arrangement_payload.get("intro_bars", 0),
            ritardando_factor=arrangement_payload.get("ritardando_factor", 1.0),
        )
        return cls(
            notation_score=score,
            performance_plan=plan,
            arrangement=arrangement,
            key=KeySignature(**payload["key"]),
            time_signature=payload["time_signature"],
            chord_bars=tuple(tuple(bar) for bar in payload.get("chord_bars", ())),
            # Jobs composed before the sidecar carried voice instruments
            # render with the legacy single-instrument fallback.
            voice_instruments=tuple(
                VoiceInstrument(**v) for v in payload.get("voice_instruments", ())
            ),
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

    ensemble = resolve_ensemble(spec)
    score, chord_bars = _build_score(spec, key, time_signature, arrangement, ensemble)
    lint_report = lint(score, chord_bars=chord_bars or None)
    if not lint_report.passed:
        raise CompositionEngineError(
            code=EngineErrorCode.LINT_FAILED,
            message=f"score failed lint: {[i.code for i in lint_report.issues]}",
            lint_issues=lint_report.issues,
        )

    # The legacy drum-kit layout (scalar drum_set spec) keys its plan
    # off "drum_set" exactly as it did before the ensemble landed, so
    # its render stays byte-identical: no melody legato or pedal, kit
    # humanization only.
    drum_set_legacy = ensemble.percussion == "drum_set" and ensemble.melody == "piano"
    performance = _build_performance_plan(
        score,
        voice_instruments=ensemble.voice_instruments(),
        humanization=spec.humanization,
        seed=spec.seed,
        arrangement=arrangement,
        drum_set_legacy=drum_set_legacy,
    )
    return EngineOutput(
        notation_score=score,
        performance_plan=performance,
        arrangement=arrangement,
        key=key,
        time_signature=time_signature,
        chord_bars=chord_bars,
        voice_instruments=tuple(
            VoiceInstrument(voice_id=voice_id, instrument=instrument)
            for voice_id, instrument in sorted(ensemble.voice_instruments().items())
        ),
    )


# ---------------------------------------------------------------------------
# Score generation
# ---------------------------------------------------------------------------


def _build_score(
    spec: CompositionSpec,
    key: KeySignature,
    time_signature: str,
    arrangement: DurationArrangement,
    ensemble: Ensemble,
) -> tuple[NotationScore, tuple[tuple[int, ...], ...]]:
    """Build the NotationScore from the spec + arrangement.

    Generates one melody voice + one bass voice per section, with
    per-section seed-derived variation when the arrangement has
    multiple repetitions. If the arrangement has a coda, an extra
    coda-length tail is appended using a coda-flavored seed so the
    variation rules from §10 #10 still apply.

    Also returns the chord pitch classes sounding in each bar, in bar
    order — the linter's chord-tone gate consumes them.
    """
    measures: list[Measure] = []
    notes: list[NoteEvent] = []
    chord_bars: list[tuple[int, ...]] = []
    section_starts: list[int] = []  # start_tick of each section
    cursor_tick = 0
    prev_bass: int | None = None
    bars_since_breath = 0

    rng_base_seed = spec.seed if spec.seed is not None else 0
    long_piece = arrangement.repetition_count >= ARRANGEMENT_ARC_MIN_REPS

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
        is_final_section = section_idx == arrangement.repetition_count - 1
        if is_final_section:
            # The last repetition must land at home: rewrite its last
            # two bars as the mood's cadence (earlier sections may end
            # open — their V resolves into the next section's I).
            section_template = apply_final_cadence(section_template, spec.mood.value)
        # Long pieces lift the final repetition a whole step — the
        # piece ends in the new key, so the coda (which follows it)
        # stays lifted too and the ending keeps its cadence.
        key_offset = MODULATION_OFFSET if is_final_section and long_piece else 0
        section_result = _generate_section(
            key=key,
            time_signature=time_signature,
            template=section_template,
            section_start_tick=cursor_tick,
            rng=section_rng,
            seed_for_variation=rng_base_seed + section_idx,
            mood=spec.mood.value,
            prev_bass=prev_bass,
            is_final_section=is_final_section,
            key_offset=key_offset,
            melody_from_bar=arrangement.intro_bars if section_idx == 0 else 0,
            bars_since_breath=bars_since_breath,
            harmony=ensemble.harmony is not None,
        )
        section_notes, section_chord_bars, bars_since_breath = section_result
        chord_bars.extend(section_chord_bars)
        # Terraced dynamics: the section's whole dynamic sits at its
        # step of the arc rather than drifting continuously.
        velocity_scale = _section_velocity_scale(section_idx, arrangement.repetition_count)
        if velocity_scale != 1.0:
            section_notes = [
                replace(
                    note,
                    velocity=max(1, min(127, round(note.velocity * velocity_scale))),
                )
                for note in section_notes
            ]
        notes.extend(section_notes)
        bass_notes = [n for n in section_notes if n.voice_id == VOICE_BASS]
        if bass_notes:
            prev_bass = max(bass_notes, key=lambda n: n.tick).pitch_midi
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
        coda_result = _generate_section(
            key=key,
            time_signature=time_signature,
            template=coda_template,
            section_start_tick=cursor_tick,
            rng=coda_rng,
            seed_for_variation=rng_base_seed + arrangement.repetition_count,
            mood=spec.mood.value,
            prev_bass=prev_bass,
            # The outro thins out: the coda opens bass alone, and a
            # long-piece modulation stays lifted through the ending.
            key_offset=MODULATION_OFFSET if long_piece else 0,
            melody_from_bar=1 if arrangement.coda_bars >= 2 else 0,
            # The coda is the piece's true ending: its final bar must
            # force the tonic resolution the way a last section does.
            is_final_section=True,
            bars_since_breath=bars_since_breath,
            harmony=ensemble.harmony is not None,
        )
        coda_notes, coda_chord_bars, _ = coda_result
        chord_bars.extend(coda_chord_bars)
        velocity_scale = _section_velocity_scale(
            arrangement.repetition_count, arrangement.repetition_count
        )
        if velocity_scale != 1.0:
            coda_notes = [
                replace(
                    note,
                    velocity=max(1, min(127, round(note.velocity * velocity_scale))),
                )
                for note in coda_notes
            ]
        notes.extend(coda_notes)
        cursor_tick += arrangement.coda_bars * bar_ticks(time_signature)

    # Drum set: when the piece is written for the kit, the piano stays
    # as the accompaniment and a percussion voice plays the mood's
    # rhythm pattern in every bar (voice 2, GM channel-10 keys). Styles
    # with A/B variants rotate across sections, matching the chord
    # templates' variation rule. Melodic meters the library does not
    # cover (5/4, 7/8) get no percussion rather than a wrong pattern.
    # Long pieces rest the kit during the bass-alone intro bars and one
    # mid-piece section, so the texture has a hole before it refills.
    if ensemble.percussion == "drum_set":
        rest_bars: set[int] = set()
        if long_piece:
            rest_bars.update(range(arrangement.intro_bars))
            rest_bars.update(
                range(
                    PERCUSSION_REST_SECTION * arrangement.form_bars,
                    (PERCUSSION_REST_SECTION + 1) * arrangement.form_bars,
                )
            )
        notes.extend(
            _generate_percussion(
                mood=spec.mood.value,
                time_signature=time_signature,
                form_bars=arrangement.form_bars,
                repetition_count=arrangement.repetition_count,
                total_bars=arrangement.total_bars_with_coda,
                seed=rng_base_seed,
                rest_bars=frozenset(rest_bars),
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

    # The outro ritardando: a coda'd piece slows across its whole coda;
    # a long piece without a coda eases in over its final cadence bars.
    # The arrangement's duration math already included the slowdown, so
    # the tempo map and the realised duration agree.
    tempo_changes: tuple[TempoPoint, ...] = ()
    if arrangement.ritardando_factor < 1.0:
        if arrangement.coda_bars > 0:
            change_tick = arrangement.repetition_count * arrangement.form_bars * bar_ticks(
                time_signature
            )
        else:
            change_tick = (
                arrangement.total_bars - RITARDANDO_BARS
            ) * bar_ticks(time_signature)
        tempo_changes = (
            TempoPoint(
                tick=change_tick,
                bpm=round(arrangement.tempo_bpm * arrangement.ritardando_factor, 1),
            ),
        )

    return NotationScore.make(
        ppq=PPQ,
        key=key,
        time_signature=time_signature,
        tempo_bpm=arrangement.tempo_bpm,
        measures=measures,
        notes=notes,
        tempo_changes=tempo_changes,
    ), tuple(chord_bars)


def _generate_section(
    *,
    key: KeySignature,
    time_signature: str,
    template: ChordTemplate,
    section_start_tick: int,
    rng: random.Random,
    seed_for_variation: int,
    mood: str,
    prev_bass: int | None = None,
    is_final_section: bool = False,
    key_offset: int = 0,
    melody_from_bar: int = 0,
    bars_since_breath: int = 0,
    harmony: bool = False,
) -> tuple[list[NoteEvent], tuple[tuple[int, ...], ...], int]:
    """Generate the bass + melody notes for one section.

    The left hand walks: each chord's bass lands on the chord tone
    nearest the previous chord's bass (root position when there is no
    previous bass, or wherever the template pins `bass_degree`), and
    the bar's midpoint sounds the next chord tone above it — so the
    bass line moves stepwise through inversions instead of jumping
    root to root, and the walk carries across section boundaries via
    `prev_bass`. The melody is an octave above the chord root and
    develops the section's motif: every bar replays the motif through
    one classic operation (repetition, transposition, sequence,
    inversion, truncation, ornament) onto that bar's chord. Velocity
    follows an arch across the section with a slight accent on
    downbeats. Everything is derived from `seed_for_variation`, so
    repeated sections sound different but stay deterministic.

    Phrase shape: one bar per section is the melodic apex (raised an
    octave-portion above the line, near the 60% mark); bars ending a
    4-bar phrase lift off early into a breath — on the dominant's root
    when the chord there is the V (a half cadence), otherwise on a
    shortened chord tone. The breath is guaranteed, not just likely:
    a bar whose distance from the last gap reaches `PHRASE_BARS` is
    forced to breathe, with `bars_since_breath` carrying the previous
    section's trailing run across the boundary. When `is_final_section`
    is set the last bar resolves onto the tonic or its third, held to
    the bar line.

    When `harmony` is set, a harmony voice (pad for the calm moods,
    broken-chord arpeggio for the energetic one) is generated from the
    same resolved chords, cleared of any note that crowds the melody.

    Returns the section's notes, the chord pitch classes sounding in
    each bar (for the linter's chord-tone gate), and the breath
    deficit the next section inherits.
    """
    notes: list[NoteEvent] = []
    melody_notes: list[NoteEvent] = []
    bar_pcs: list[tuple[int, ...]] = []
    # A breath (or half cadence) at bar g means the melody runs
    # continuously for bar g+1 onward; the deficit inherited from the
    # previous section counts against this one's bars. A fresh section
    # (deficit 0) starts with the virtual gap just before its first
    # bar, so its opening run can still reach at most PHRASE_BARS.
    last_gap_bar = -bars_since_breath - 1
    tonic_midi = key_root_midi(key)
    ticks_per_bar = bar_ticks(time_signature)
    section_ticks = template.bars * ticks_per_bar
    # The melodic apex sits near the 60% mark, never on the final bar.
    apex_bar = min(int(template.bars * 0.6), template.bars - 2)
    motif = generate_motif(rng, bar_ticks=ticks_per_bar)

    # Pre-resolve each slot's chord so a bar can pick up into the next
    # chord's register (the anacrusis needs to know what it leads to).
    # `key_offset` transposes the section (the modulation lift on a
    # long piece's final repetition) but exempts the final cadence —
    # the piece lifts and then comes home for its close.
    chords: list[tuple[int, tuple[int, ...], int]] = []
    for slot_index, slot in enumerate(template.chords):
        slot_offset = 0 if slot_index >= len(template.chords) - 2 else key_offset
        degree = slot.degree
        dur = slot.bars
        root_offset = _scale_degree_to_semitones(degree, key.mode)
        if slot.borrowed and key.mode == "major" and degree % 7 in (2, 5, 6):
            # bIII/bVI/bVII: the borrowed roots sit a semitone below the
            # diatonic scale degrees (Bb, not B, in C major).
            root_offset -= 1
        chord_root = tonic_midi + slot_offset + root_offset
        chord_tones = _chord_intervals(degree, key, seventh=slot.seventh, borrowed=slot.borrowed)
        chords.append((chord_root, chord_tones, dur))
        # Every bar of the slot sounds the same pitch classes; the
        # linter checks melody and bass against this set.
        bar_pcs.extend(
            tuple(sorted({(chord_root + tone) % 12 for tone in chord_tones}))
            for _ in range(dur)
        )

    cursor = 0
    bar_index = 0
    total_bars = template.bars
    for slot_index, slot in enumerate(template.chords):
        chord_root, chord_tones, dur = chords[slot_index]

        # Walking bass: the pinned bass degree wins; otherwise the
        # chord tone nearest the previous bass (root on the first
        # chord). The midpoint sounds the next chord tone above.
        if slot.bass_degree is not None:
            bass_pitch = _octave_down(
                tonic_midi
                + slot_offset
                + _scale_degree_to_semitones(slot.bass_degree, key.mode),
                octaves=1,
            )
            prev_bass = bass_pitch
        else:
            # Two octaves of candidates keep the walk inside the bass
            # register even in sharp minor keys whose chord roots sit
            # above the middle of the keyboard.
            candidates = [
                _octave_down(chord_root + tone, octaves=octaves)
                for tone in chord_tones
                for octaves in (1, 2)
            ]
            candidates = [c for c in candidates if 21 <= c <= 60]
            if not candidates:
                bass_pitch = _octave_down(chord_root, octaves=2)
            elif prev_bass is None:
                bass_pitch = candidates[0]
            else:
                last_bass = prev_bass
                bass_pitch = min(candidates, key=lambda c: abs(c - last_bass))
        above = sorted(
            chord_root + tone
            for tone in chord_tones
            if chord_root + tone > bass_pitch
            and (chord_root + tone) % 12 != bass_pitch % 12
        )
        bass_fifth = above[0] if above else bass_pitch + 12
        while bass_fifth - bass_pitch > 12:
            bass_fifth -= 12
        # Stay near the bass register — an octave shift keeps the note
        # a chord tone (a hard clamp would not be).
        if bass_fifth > 67:
            bass_fifth -= 12
        if bass_fifth <= bass_pitch:
            bass_fifth += 12
        while bass_fifth - bass_pitch > 12:
            bass_fifth -= 12

        chord_root_tick = section_start_tick + cursor
        for _bar in range(dur):
            bar_tick = chord_root_tick + _bar * ticks_per_bar
            bar_pos = (cursor + _bar * ticks_per_bar) / max(1, section_ticks)

            # Left hand: the walking tone on the downbeat, the next
            # chord tone above it at the midpoint.
            half = ticks_per_bar // 2
            notes.append(
                NoteEvent(
                    voice_id=VOICE_BASS,
                    pitch_midi=bass_pitch,
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
            # random bar lifts off early into a breath. An intro
            # (`melody_from_bar` > 0) keeps these bars bass alone.
            if bar_index < melody_from_bar:
                bar_index += 1
                continue
            is_final_bar = is_final_section and bar_index == total_bars - 1
            is_apex = bar_index == apex_bar
            is_half_cadence = degree == 4 and bar_index % 4 == 3 and not is_final_bar
            breathe = (
                not is_final_bar and not is_half_cadence and rng.random() < 0.18
            )
            # The breath is a guarantee, not a coin toss: when the
            # melody has run `PHRASE_BARS` bars without a gap (counting
            # the previous section's trailing run), this bar must lift
            # off early. A half cadence already leaves the rest.
            if not is_final_bar and not is_half_cadence and bar_index - last_gap_bar >= PHRASE_BARS:
                breathe = True
            if is_half_cadence or breathe:
                last_gap_bar = bar_index
            anchor = _downbeat_anchor(rng, len(chord_tones))
            if is_final_bar or is_half_cadence or is_apex or breathe:
                variant = MotifVariant(motif=motif)
            else:
                variant = vary_motif(motif, rng)
            # Anacrusis: when the next bar exists, the pickup leads into
            # it from the pickup chord's tones — root first, but a tone
            # that would sound a close m2/M7 against the bar's sounding
            # bass is skipped, and if every candidate clashes the
            # pickup is dropped. Zero-length slots (a template
            # truncation artifact) are skipped — they never sound, so
            # anticipating them would be wrong.
            pickup_pitch: int | None = None
            pickup_root: int | None = None
            pickup_tones: tuple[int, ...] = ()
            if not is_final_bar:
                if _bar < dur - 1:
                    pickup_root, pickup_tones = chord_root, chord_tones
                else:
                    next_slot = next(
                        (c for c in chords[slot_index + 1 :] if c[2] > 0), None
                    )
                    if next_slot is not None:
                        pickup_root, pickup_tones = next_slot[0], next_slot[1]
                if pickup_root is not None:
                    pickup_pitch = next(
                        (
                            candidate
                            for candidate in (
                                min(107, pickup_root + 12 + tone) for tone in pickup_tones
                            )
                            if all(
                                abs(candidate - bass) not in (1, 11)
                                for bass in (bass_pitch, bass_fifth)
                            )
                        ),
                        None,
                    )
            melody_notes.extend(
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
                    mood=mood,
                    is_final_bar=is_final_bar,
                    half_cadence=is_half_cadence,
                    apex=is_apex,
                    breathe=breathe,
                    pickup_pitch=pickup_pitch,
                )
            )
            bar_index += 1

        if slot.bass_degree is None:
            prev_bass = bass_pitch
        cursor += dur * ticks_per_bar

    # Ties hold a repeated pitch across a bar line: when a bar's last
    # melody note and the next bar's first share the pitch and touch,
    # the first is marked tied (the performance layer plays them as one
    # sound; the engraving shows the tie).
    tie_probability = TIE_PROBABILITY.get(mood, 0.25)
    for index, (a, b) in enumerate(pairwise(melody_notes)):
        if (
            not a.tie
            and a.pitch_midi == b.pitch_midi
            and a.tick + a.duration_ticks == b.tick
            and rng.random() < tie_probability
        ):
            melody_notes[index] = replace(a, tie=True)

    notes.extend(melody_notes)

    # Harmony voice: generated from the same resolved `chords` so its
    # pitch classes cannot drift from the ones the melody and bass
    # were written against.
    if harmony:
        notes.extend(
            _generate_harmony_section(
                chords=chords,
                section_start_tick=section_start_tick,
                ticks_per_bar=ticks_per_bar,
                mood=mood,
                rng=rng,
                melody_notes=melody_notes,
                melody_from_bar=melody_from_bar,
                seed_for_variation=seed_for_variation,
            )
        )

    # Canonical order: the bass and melody interleave within a bar, so
    # sort by (tick, voice, pitch) rather than relying on append order.
    ordered = sorted(notes, key=lambda n: (n.tick, n.voice_id, n.pitch_midi))
    # The breath deficit the next section inherits: one more than the
    # trailing run of continuously sounding melody bars, so its first
    # bar continues that run (a silence — an intro or coda pickup bar —
    # breaks it and leaves nothing to inherit).
    if melody_from_bar >= total_bars:
        trailing = 0
    elif last_gap_bar >= melody_from_bar:
        trailing = max(0, total_bars - 1 - last_gap_bar)
    else:
        trailing = total_bars - melody_from_bar
    return ordered, tuple(bar_pcs), trailing + 1


# The harmony voice's constants. Its bed sits between the bass and the
# melody: low enough to stay out of the tune's register, high enough to
# clear the bass walk.
HARMONY_MIN_MIDI: int = 48
HARMONY_MAX_MIDI: int = 84
# A harmony note is cleared away from the melody when it sits a rubbed
# second (or unison) — or a major-seventh inversion of one — from a
# simultaneously sounding melody note: shifted an octave or dropped.
# These are exactly the linter's dissonant intervals plus the unison
# (a doubled tune line is the melody's job, not the pad's).
HARMONY_MELODY_CLEARANCE: int = 2
HARMONY_CROWD_INTERVALS: frozenset[int] = frozenset({0, 1, 2, 10, 11})
HARMONY_PAD_VELOCITY: int = 46
HARMONY_ARPEGGIO_VELOCITY: int = 52


def _into_harmony_register(pitch: int) -> int:
    """Octave-shift a chord tone into the harmony bed's register.

    An octave shift keeps the pitch class (and so the chord tone); a
    clamp would not.
    """
    while pitch > HARMONY_MAX_MIDI:
        pitch -= 12
    while pitch < HARMONY_MIN_MIDI:
        pitch += 12
    return pitch


def _generate_harmony_section(
    *,
    chords: list[tuple[int, tuple[int, ...], int]],
    section_start_tick: int,
    ticks_per_bar: int,
    mood: str,
    rng: random.Random,
    melody_notes: list[NoteEvent],
    melody_from_bar: int,
    seed_for_variation: int,
) -> list[NoteEvent]:
    """Generate the harmony voice for one section from the resolved chords.

    Two textures, chosen by mood: a sustained pad for the calm moods
    (two chord tones held across each bar, the voicing rotating through
    root position and inversions) and a broken-chord arpeggio for the
    energetic one (eighth notes walking the chord tones). Intro bars
    stay silent — the harmony enters with the melody.

    The bed is written before the melody-to-harmony clearance pass so
    the RNG draws stay independent of what the tune happens to do.
    """
    notes: list[NoteEvent] = []
    pad = mood != "electrifying"
    # Which chord tone sits lowest: the rotation (not the bar) decides
    # it, so the section's voicing stays stable instead of churning.
    rotation = rng.randrange(3)
    eighth = PPQ // 2
    arpeggio_steps = max(1, ticks_per_bar // eighth)

    cursor = 0
    bar_index = 0
    total_bars = sum(dur for _, _, dur in chords)
    for chord_root, chord_tones, dur in chords:
        for _bar in range(dur):
            bar_tick = section_start_tick + cursor + _bar * ticks_per_bar
            if bar_index < melody_from_bar:
                bar_index += 1
                continue
            position = bar_index / max(1, total_bars)
            if pad:
                low = (bar_index + rotation) % len(chord_tones)
                pair = (chord_tones[low], chord_tones[(low + 2) % len(chord_tones)])
                for tone in pair:
                    notes.append(
                        NoteEvent(
                            voice_id=VOICE_HARMONY,
                            pitch_midi=_into_harmony_register(chord_root + tone),
                            tick=bar_tick,
                            duration_ticks=ticks_per_bar,
                            velocity=_shaped_velocity(
                                base=HARMONY_PAD_VELOCITY,
                                position=position,
                                tick=bar_tick,
                                ticks_per_bar=ticks_per_bar,
                                rng_seed=seed_for_variation + bar_tick,
                            ),
                        )
                    )
            else:
                for step in range(arpeggio_steps):
                    tone = chord_tones[(step + rotation) % len(chord_tones)]
                    notes.append(
                        NoteEvent(
                            voice_id=VOICE_HARMONY,
                            pitch_midi=_into_harmony_register(chord_root + tone),
                            tick=bar_tick + step * eighth,
                            duration_ticks=eighth,
                            velocity=_shaped_velocity(
                                base=HARMONY_ARPEGGIO_VELOCITY,
                                position=position,
                                tick=bar_tick + step * eighth,
                                ticks_per_bar=ticks_per_bar,
                                rng_seed=seed_for_variation + bar_tick * 101 + step,
                            ),
                        )
                    )
            bar_index += 1
        cursor += dur * ticks_per_bar

    return _clear_harmony_of_melody(notes, melody_notes)


def _crowds_melody(note: NoteEvent, melody_notes: list[NoteEvent], pitch: int) -> bool:
    """Does a harmony pitch collide with any simultaneously sounding melody note?"""
    return any(
        melody.tick < note.tick + note.duration_ticks
        and note.tick < melody.tick + melody.duration_ticks
        and abs(melody.pitch_midi - pitch) in HARMONY_CROWD_INTERVALS
        for melody in melody_notes
    )


def _clear_harmony_of_melody(
    notes: list[NoteEvent], melody_notes: list[NoteEvent]
) -> list[NoteEvent]:
    """Shift or drop harmony notes that crowd the melody.

    A harmony note a rubbed second (or its major-seventh inversion, or
    a unison) from a simultaneously sounding melody note is moved an
    octave — the pitch class, and so the chord tone, survives — when
    that keeps it in register and out of trouble; otherwise it is
    dropped. The pass runs after generation so the pad and arpeggio
    logic stay register-agnostic.
    """
    kept: list[NoteEvent] = []
    for note in notes:
        if not _crowds_melody(note, melody_notes, note.pitch_midi):
            kept.append(note)
            continue
        alternatives = (
            shifted
            for shift in (12, -12)
            if HARMONY_MIN_MIDI <= (shifted := note.pitch_midi + shift) <= HARMONY_MAX_MIDI
            and not _crowds_melody(note, melody_notes, shifted)
        )
        clear = next(alternatives, None)
        if clear is None:
            continue
        kept.append(replace(note, pitch_midi=clear))
    return kept


def _truncate_template_for_coda(template: ChordTemplate, coda_bars: int) -> ChordTemplate:
    """Return a coda-sized prefix of `template`, closing on the tonic.

    The coda is a sub-form: a coda_bars-bar prefix of the form's
    template, using the first chord cycles that fit. Its final chord
    is forced to the tonic (degree 0) so the piece ends with a
    cadence home rather than on whatever chord the truncation lands
    on. Coda length is always strictly less than the form's full
    length. Zero-duration chord entries are dropped.
    """
    kept: list[ChordSlot] = []
    consumed = 0
    for slot in template.chords:
        remaining = coda_bars - consumed
        if remaining <= 0:
            break
        if slot.bars > remaining:
            kept.append(slot._replace(bars=remaining))
            consumed += remaining
        else:
            kept.append(slot)
            consumed += slot.bars
    if kept:
        last = kept[-1]
        kept[-1] = ChordSlot(0, last.bars, seventh=last.seventh, bass_degree=0)
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


def _chord_intervals(
    degree: int,
    key: KeySignature,
    *,
    seventh: bool = False,
    borrowed: bool = False,
) -> tuple[int, ...]:
    """Root-relative chord-tone intervals for a diatonic chord.

    Thin wrapper over `forms.chord_intervals`, the single source of
    truth for the mode-aware triad/seventh/borrowed tables.
    """
    return chord_intervals(degree, key, seventh=seventh, borrowed=borrowed)


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
    mood: str,
    is_final_bar: bool = False,
    half_cadence: bool = False,
    apex: bool = False,
    breathe: bool = False,
    pickup_pitch: int | None = None,
) -> list[NoteEvent]:
    """Render one bar of melody from a motif variant.

    The motif is walked in chord-tone index space starting at `anchor`,
    so the same shape lands correctly on every chord. When `repeat` is
    set (the sequence operation) the motif keeps replaying from the top
    — anchor advancing one tone per cycle — until the bar is full. The
    bar's slots are then re-voiced through the mood's rhythm library
    (`motif.apply_rhythm`): dotted figures, 16th subdivisions, ties.

    Phrase shape (carried over from the arpeggio walk):
    - an `apex` bar is lifted an octave (capped to range) with a
      velocity lift — the section's melodic peak;
    - a `half_cadence` bar ends early on the chord's root, leaving a
      rest (the phrase breathes on the V);
    - a breathing bar shortens its last note into a rest;
    - the `is_final_bar` of the piece resolves onto the tonic or its
      third, held to the bar line;
    - when the bar leaves at least an eighth of space at its end and
      `pickup_pitch` is given (the next bar's anchor tone), an anacrusis
      pickup note sounds on the last eighth, leading into the next bar.
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
    rhythm_slots: list[BarSlot]
    if is_final_bar:
        # The closing bar keeps the motif's own rhythm: the resolution
        # is the one event that should not be dressed up.
        rhythm_slots = [(o, d, t, False) for o, d, t in slots]
    else:
        rhythm_slots = apply_rhythm(slots, rng=rng, mood=mood)

    notes: list[NoteEvent] = []
    for slot_index, slot in enumerate(rhythm_slots):
        bar_offset, duration, tone_index, tie = slot
        tick = start_tick + bar_offset
        if slot_index == len(rhythm_slots) - 1 and is_final_bar:
            # The piece ends at home: tonic or its third, held to the
            # bar line.
            pitch = chord_root if rng.random() < 0.6 else chord_root + chord_tones[1]
            duration = bar_ticks - bar_offset
        else:
            pitch = chord_root + chord_tones[tone_index % len(chord_tones)]
            if slot_index == len(rhythm_slots) - 1 and half_cadence:
                # Land on the chord's root, lifted early so a rest
                # follows.
                pitch = chord_root + chord_tones[0]
                duration = duration // 2
            elif slot_index == len(rhythm_slots) - 1 and breathe:
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
                tie=bool(tie),
            )
        )

    # Anacrusis: the bar left room at its end, so an eighth-note pickup
    # on the next chord's anchor tone leads into the next downbeat.
    if (
        pickup_pitch is not None
        and not is_final_bar
        and notes
        and notes[-1].tick + notes[-1].duration_ticks <= start_tick + bar_ticks - PPQ // 2
    ):
        notes.append(
            NoteEvent(
                voice_id=VOICE_MELODY,
                pitch_midi=pickup_pitch,
                tick=start_tick + bar_ticks - PPQ // 2,
                duration_ticks=PPQ // 2,
                velocity=max(1, _shaped_velocity(
                    base=DEFAULT_VELOCITY + 8,
                    position=position,
                    tick=start_tick + bar_ticks - PPQ // 2,
                    ticks_per_bar=ticks_per_bar,
                    rng_seed=seed_for_variation + start_tick,
                ) - 8),
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
    rest_bars: frozenset[int] = frozenset(),
) -> list[NoteEvent]:
    """Generate the percussion voice for a drum-set piece.

    The style comes from the mood + meter (`style_for`); its variants
    rotate across sections on a longer cycle than plain A/B
    (`percussion.rotation_index`), with the coda treated as one more
    section. A section's last bar hands off to the next through the
    style's fill (never on the piece's final bar, which must resolve),
    and every section downbeat is marked with a crash cymbal — plus a
    kick when the pattern does not already open with one. A per-bar
    seeded jitter of a few velocity points keeps repeated bars from
    sounding machine-stamped. Bars in `rest_bars` (the intro and one
    mid-piece section on long pieces) are silent, and the sections
    that do play follow the terraced dynamic arc.
    """
    style = style_for(mood, time_signature)
    if style is None:
        return []
    ticks_per_bar = bar_ticks(time_signature)
    mood_scale = MOOD_VELOCITY_SCALE.get(mood, 1.0)
    notes: list[NoteEvent] = []
    for bar in range(total_bars):
        if bar in rest_bars:
            continue
        in_body = bar < repetition_count * form_bars
        section_idx = bar // form_bars if in_body else repetition_count
        section_start = bar % form_bars == 0
        is_final_bar = bar == total_bars - 1
        terrace = _section_velocity_scale(section_idx, repetition_count)
        if bar % form_bars == form_bars - 1 and not is_final_bar:
            pattern = style.fill(time_signature, section_idx)
        else:
            pattern = None
        if pattern is None:
            pattern = style.pattern(
                time_signature, rotation_index(section_idx, len(style.variants.get(time_signature, ())))
            )
        if pattern is None:
            continue
        bar_rng = random.Random(seed + bar)
        bar_start = bar * ticks_per_bar
        for hit in pattern:
            jitter = bar_rng.uniform(0.92, 1.06)
            velocity = round(
                hit.velocity * style.velocity_scale * mood_scale * terrace * jitter
            )
            notes.append(
                NoteEvent(
                    voice_id=VOICE_PERCUSSION,
                    pitch_midi=hit.key,
                    tick=bar_start + hit.offset_ticks,
                    duration_ticks=PERCUSSION_NOTE_TICKS,
                    velocity=min(PERCUSSION_VELOCITY_MAX, max(1, velocity)),
                )
            )
        if section_start:
            # The section downbeat is marked: crash always, and a kick
            # underneath it when the groove does not open with one.
            crash_velocity = round(
                SECTION_CRASH_VELOCITY * style.velocity_scale * mood_scale * terrace
            )
            notes.append(
                NoteEvent(
                    voice_id=VOICE_PERCUSSION,
                    pitch_midi=DRUM_CRASH,
                    tick=bar_start,
                    duration_ticks=PERCUSSION_NOTE_TICKS,
                    velocity=min(PERCUSSION_VELOCITY_MAX, max(1, crash_velocity)),
                )
            )
            if not any(h.offset_ticks == 0 and h.key == DRUM_KICK for h in pattern):
                notes.append(
                    NoteEvent(
                        voice_id=VOICE_PERCUSSION,
                        pitch_midi=DRUM_KICK,
                        tick=bar_start,
                        duration_ticks=PERCUSSION_NOTE_TICKS,
                        velocity=min(PERCUSSION_VELOCITY_MAX, max(1, round(84 * mood_scale))),
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

# Cross-bar ties: when two adjacent bars share a pitch at the boundary,
# the first is marked tied with this probability (calmer moods hold
# more; electrifying keeps its attacks).
TIE_PROBABILITY: dict[str, float] = {
    "electrifying": 0.18,
    "calming": 0.28,
    "sleep": 0.35,
}

# Arrangement arc (S8): long pieces lift their final repetition a whole
# step (the piece ends in the new key — the lift IS the ending), drop
# the drums for one mid-piece section to give the texture a hole, and
# step the dynamics per section instead of arching continuously.
MODULATION_OFFSET: int = 2
PERCUSSION_REST_SECTION: int = 1


def _section_velocity_scale(section_idx: int, repetition_count: int) -> float:
    """Terraced dynamics: the arc is stepped per section, not continuous.

    The opening sits back, the penultimate section peaks, and the
    final one settles slightly for the cadence home. A single-section
    piece has nowhere to move and plays at full.
    """
    if repetition_count < 2:
        return 1.0
    if section_idx == 0:
        return 0.82
    if section_idx == repetition_count - 1:
        return 0.95
    if section_idx == repetition_count - 2:
        return 1.12
    return 1.0


def _phrase_swell(position_in_phrase: float) -> float:
    """One rise-and-fall per phrase, for the controller layer.

    `position_in_phrase` is 0..1 across a `PHRASE_BARS`-bar phrase; the
    swell spans roughly 0.85x to 1.0x so each phrase breathes once.
    """
    pos = min(1.0, max(0.0, position_in_phrase))
    return 0.85 + 0.15 * math.sin(math.pi * pos)


def _merged_tie_runs(
    notes: tuple[NoteEvent, ...] | list[NoteEvent],
) -> tuple[set[int], dict[int, int]]:
    """Which tied notes fold into their predecessor, and the merged spans.

    A note with `tie=True` connects to the next same-voice, same-pitch,
    contiguous note — and that continuation may itself be tied onward,
    so a run of tied noteheads collapses into one sounded note. Returned
    as (indices to skip, duration in ticks per surviving index). The
    walk stays on the tick grid, where contiguity is exact.
    """
    by_voice: dict[int, list[int]] = {}
    for idx, note in enumerate(notes):
        by_voice.setdefault(note.voice_id, []).append(idx)

    skip: set[int] = set()
    durations: dict[int, int] = {}
    for voice_indices in by_voice.values():
        k = 0
        while k < len(voice_indices):
            head = voice_indices[k]
            if head in skip:
                k += 1
                continue
            span = notes[head].duration_ticks
            cur = head
            j = k + 1
            while (
                j < len(voice_indices)
                and notes[cur].tie
                and notes[voice_indices[j]].pitch_midi == notes[cur].pitch_midi
                and notes[cur].tick + notes[cur].duration_ticks == notes[voice_indices[j]].tick
            ):
                span += notes[voice_indices[j]].duration_ticks
                skip.add(voice_indices[j])
                cur = voice_indices[j]
                j += 1
            durations[head] = span
            k = j
    return skip, durations


def _build_performance_plan(
    score: NotationScore,
    *,
    voice_instruments: Mapping[int, str],
    humanization: str,
    seed: int | None,
    arrangement: DurationArrangement,
    drum_set_legacy: bool = False,
) -> PerformancePlan:
    """Lay the expression layer on the notated surface, per voice.

    `voice_instruments` maps engine voice ids to instrument names; the
    plan's legato, CC11 swells, sustain pedal, and humanization are
    chosen per voice from that instrument's family (sustained vs
    percussive vs pedal-reading). The bass is the engine's own walking
    line and is deliberately left un-humanized. `drum_set_legacy` pins
    the scalar drum-set spec's plan to its pre-ensemble behaviour: the
    melody voice is treated as the kit's lead ("drum_set") — no legato,
    no pedal — so its render is byte-identical.
    """
    # Ties play as one sound: a tied note's continuation never
    # re-attacks — the predecessor rings through it. Runs are resolved
    # on the notation grid first (contiguity in ticks is exact, and a
    # chain of tied noteheads collapses into a single sounded note);
    # humanization then scatters the surviving events freely.
    tie_skips, tie_spans = _merged_tie_runs(score.notes)
    events: list[PerformanceNoteEvent] = []
    for idx, note in enumerate(score.notes):
        if idx in tie_skips:
            continue
        start_us = microseconds_at_tick(note.tick, score.tempo)
        duration_us = microseconds_at_tick(
            tie_spans.get(idx, note.duration_ticks) + note.tick, score.tempo
        ) - start_us
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
    melody_instrument = "drum_set" if drum_set_legacy else voice_instruments.get(
        VOICE_MELODY, "piano"
    )
    harmony_instrument = voice_instruments.get(VOICE_HARMONY)
    harmony_sustained = (
        harmony_instrument is not None and harmony_instrument in SUSTAINED_INSTRUMENTS
    )

    # Legato: sustained instruments let each note of a line ring a
    # little past the next attack so the release tail blurs into the
    # next note. A same-pitch neighbour is capped at its start: a late
    # note_off on the same key would re-attack or cut the line.
    legato_extended: set[int] = set()
    legato_voices = [
        voice
        for voice, instrument in (
            (VOICE_MELODY, melody_instrument),
            (VOICE_HARMONY, harmony_instrument),
        )
        if instrument is not None and instrument in SUSTAINED_INSTRUMENTS
    ]
    for voice in legato_voices:
        voice_sorted_idx = sorted(
            (i for i, e in enumerate(events) if e.voice_id == voice),
            key=lambda i: events[i].start_us,
        )
        for a, b in pairwise(voice_sorted_idx):
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
        # On top of the piece-long arch, each 4-bar phrase swells once
        # (rise into its middle, relax at its end) and the section
        # terracing sits the whole step of the arc. The melody always
        # swells; the harmony voice follows when its instrument is
        # sustained (a pad wants to breathe; a plucked arpeggio has no
        # sustain to shape).
        total_ticks = max(1, score.total_ticks())
        ticks_per_bar = max(1, score.measures[0].end_tick - score.measures[0].start_tick)
        phrase_ticks = PHRASE_BARS * ticks_per_bar
        swell_voices = [VOICE_MELODY]
        if harmony_sustained:
            swell_voices.append(VOICE_HARMONY)
        for measure in score.measures:
            position = measure.start_tick / total_ticks
            phrase_position = (measure.start_tick % phrase_ticks) / phrase_ticks
            section_idx = (measure.start_tick // (arrangement.form_bars * ticks_per_bar)) if (
                arrangement.form_bars > 0
            ) else 0
            value = min(
                127,
                round(
                    EXPRESSION_BASE
                    * _velocity_arc(position)
                    * _phrase_swell(phrase_position)
                    * _section_velocity_scale(section_idx, arrangement.repetition_count)
                ),
            )
            for voice in swell_voices:
                controllers.append(
                    ControllerEvent(
                        voice_id=voice,
                        control=11,
                        value=value,
                        start_us=microseconds_at_tick(measure.start_tick, score.tempo),
                    )
                )
        # Per-bar pedaling: press on the downbeat, lift just before the
        # next one so chords do not wash across the bar line. Applied
        # per voice whose instrument reads a pedal.
        pedal_voices = [
            voice
            for voice, instrument in (
                (VOICE_MELODY, melody_instrument),
                (VOICE_HARMONY, harmony_instrument),
            )
            if instrument is not None and instrument in PEDAL_INSTRUMENTS
        ]
        for voice in pedal_voices:
            for i, measure in enumerate(score.measures):
                press_us = microseconds_at_tick(measure.start_tick, score.tempo)
                controllers.append(
                    ControllerEvent(voice_id=voice, control=64, value=127, start_us=press_us)
                )
                if i + 1 < len(score.measures):
                    next_downbeat = microseconds_at_tick(
                        score.measures[i + 1].start_tick, score.tempo
                    )
                    controllers.append(
                        ControllerEvent(
                            voice_id=voice,
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
            if event.voice_id in (VOICE_MELODY, VOICE_HARMONY):
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
    "VoiceInstrument",
    "compose",
]
