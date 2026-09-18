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
   - the left hand states each chord with a figure drawn from the
     mood's vocabulary instead of a held drone,
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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from itertools import pairwise
from typing import Any, Final

from saimc.compose.duration import (
    DEFAULT_SECTION_ARC,
    HARMONY_TEXTURE_CYCLE,
    HARMONY_TEXTURE_FIRST,
    HARMONY_TEXTURE_REST,
    DurationArrangement,
    DurationUnfulfillableError,
    SectionArc,
    arrange_for_duration,
    bar_ticks,
    section_seed,
)
from saimc.compose.ensemble import Ensemble, resolve_ensemble
from saimc.compose.forms import (
    LEAP_MIN_SEMITONES,
    PHRASE_BARS,
    STEP_MAX_SEMITONES,
    ChordSlot,
    ChordTemplate,
    apply_final_cadence,
    apply_harmonic_rhythm,
    apply_section_close,
    bar_scale_intervals,
    chord_intervals,
    chord_root_offset,
    chord_tone_degrees,
    chosen_key,
    get_template_for_form,
    key_root_midi,
    scale_walk,
    transposed_key,
)
from saimc.compose.linter import DISSONANT_INTERVALS, LintIssue, lint
from saimc.compose.motif import (
    DEFAULT_MELODY_SHAPE,
    PLAIN_BASS_FIGURE,
    BarSlot,
    BassFigure,
    MelodyShape,
    MotifVariant,
    apply_rhythm,
    draw_bass_figures,
    generate_motif,
    vary_motif,
)
from saimc.compose.percussion import (
    DEFAULT_DRUM_KIT,
    DRUM_CRASH,
    DRUM_KICK,
    PERCUSSION_NOTE_TICKS,
    PERCUSSION_VELOCITY_MAX,
    DrumKit,
    rotation_index,
)
from saimc.compose.plan import CompositionPlan, resolve_plan
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
from saimc.compose.voices import (
    DEFAULT_HARMONY_VOICES,
    HARMONY_MELODY_CLEARANCE,
    HARMONY_STAB_INSTRUMENTS,
    HarmonyVoices,
)
from saimc.instruments import (
    LINE_BAND_SEMITONES,
    BedRegisters,
    MelodyBand,
    bed_registers,
    bed_window,
    melody_band,
    range_for,
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
class VoiceInstrument:
    """Which instrument renders one engine voice (a sidecar row).

    A list of these (not a dict keyed by voice id) because JSON object
    keys are strings — the int voice ids would not round-trip.
    """

    voice_id: int
    instrument: str


SIDECAR_SCHEMA_VERSION: Final[int] = 1
"""Bump when the sidecar's own key set or its nesting changes.

It tracks this *container's* shape rather than the canonical-JSON encoding
rules (`CANONICAL_FORMAT_VERSION`), for `PLAN_SCHEMA_VERSION`'s reason: the
sidecar is the only canonical document that wraps others, and each wrapped
document carries its own tag — the score's and the performance plan's are
the encoding version, the composition plan's is its own schema. This one is
about the sidecar alone, so a reader can tell a document written by an
older build from one written by a newer one instead of silently taking what
it recognises and dropping the rest.
"""

SIDECAR_FORMAT: Final[str] = f"EngineOutput:{SIDECAR_SCHEMA_VERSION}"
"""The tag every engine-output sidecar carries, in §6's `{kind}:{version}` form."""


@dataclass(frozen=True)
class EngineOutput:
    """The engine's output: NotationScore + PerformancePlan + arrangement metadata.

    `chord_bars` carries the chord pitch classes sounding in each bar
    (bar order) so the release gates can re-run the linter's
    chord-tone check without regenerating the harmony, and `bar_keys`
    carries the key each of those bars belongs to — the same key unless a
    modulation moved it — so the passing-tone licence reads the bar the
    way the walk wrote it. `voice_instruments` maps each engine voice to
    the instrument that renders it — the renderers' source of truth for
    per-voice programs and channels.

    `plan` is the plan the engine composed under, resolved: the record
    that makes a sidecar replayable without re-deriving anything. It is
    `None` only for a sidecar written before the plan existed — every
    output `compose` returns carries one, because `compose` resolves the
    default rather than leaving the field empty.
    """

    notation_score: NotationScore
    performance_plan: PerformancePlan
    arrangement: DurationArrangement
    key: KeySignature
    time_signature: str
    chord_bars: tuple[tuple[int, ...], ...] = ()
    bar_keys: tuple[KeySignature, ...] = ()
    voice_instruments: tuple[VoiceInstrument, ...] = ()
    plan: CompositionPlan | None = None

    def to_sidecar(self) -> dict[str, Any]:
        """Serialize to a JSON-friendly dict for the sidecar file.

        The compose types are plain `@dataclass(frozen=True)`, not
        Pydantic, so we use `dataclasses.asdict` for the conversion. The
        plan is written as its canonical document rather than as an
        `asdict` of the dataclass, so the block in the sidecar is the same
        document its own hash is taken over.
        """
        from dataclasses import asdict

        payload: dict[str, Any] = {
            "format": SIDECAR_FORMAT,
            "notation_score": asdict(self.notation_score),
            "performance_plan": asdict(self.performance_plan),
            "arrangement": asdict(self.arrangement),
            "key": asdict(self.key),
            "time_signature": self.time_signature,
            "chord_bars": [list(bar) for bar in self.chord_bars],
            "bar_keys": [asdict(k) for k in self.bar_keys],
            "voice_instruments": [asdict(v) for v in self.voice_instruments],
        }
        # Omitted, not nulled, when there is none: a sidecar written before
        # the plan existed has no key either, and re-serializing one should
        # not invent a field that reads as "a plan was considered here".
        if self.plan is not None:
            payload["plan"] = self.plan.to_canonical_dict()
        return payload

    @classmethod
    def from_sidecar(cls, payload: dict[str, Any]) -> EngineOutput:
        """Reconstruct from the sidecar JSON dict.

        Nested dataclasses (`NotationScore`, `PerformancePlan`,
        `DurationArrangement`, `KeySignature`, `ChordTemplate`,
        `Measure`, `NoteEvent`, `PerformanceNoteEvent`) are rebuilt
        with their constructors by name; `asdict` collapses them
        into plain `dict`s, so we rehydrate each one explicitly. The
        plan is the exception — it is rehydrated from its canonical
        document, which is how it was written.

        The document's own `format` tag is read with a default, because
        sidecars written before it existed carry no tag; a tag of any
        other version is refused rather than coerced, since a newer
        sidecar may carry state this build would drop on the floor.
        """
        written_format = payload.get("format")
        if written_format is not None and written_format != SIDECAR_FORMAT:
            raise ValueError(
                f"engine-output sidecar names format {written_format!r}, but this build "
                f"writes and understands {SIDECAR_FORMAT!r}; re-compose the job with a "
                "build that matches, or delete its sidecar and job directory."
            )
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
            # Jobs composed before the sidecar carried per-bar keys re-lint
            # against the score's own key, which is what the licence read
            # before the modulation was published.
            bar_keys=tuple(KeySignature(**k) for k in payload.get("bar_keys", ())),
            # Jobs composed before the sidecar carried voice instruments
            # render with the legacy single-instrument fallback.
            voice_instruments=tuple(
                VoiceInstrument(**v) for v in payload.get("voice_instruments", ())
            ),
            # Jobs composed before the sidecar carried the plan have none.
            # Deliberately *not* resolved from the spec: a resolved plan
            # would describe this build's defaults, not the ones the piece
            # was actually composed under, and a provenance record that
            # guesses is worse than one that says nothing.
            plan=(
                CompositionPlan.from_canonical_dict(plan_payload)
                if (plan_payload := payload.get("plan")) is not None
                else None
            ),
        )


def compose(
    spec: CompositionSpec, *, plan: CompositionPlan | None = None
) -> EngineOutput:
    """Run the full composition pipeline against the spec and its plan.

    `plan=None` resolves to this spec's default plan, which is a
    description of the engine as it behaved before the plan existed, so
    a caller that does not know about plans gets byte-identical output.
    A supplied plan is honoured as written: the engine looks its musical
    decisions up in the artifact rather than in module tables, which is
    what makes `(plan, seed) -> notes` a claim about the artifact.
    """
    resolved = resolve_plan(spec, plan)
    knobs = resolved.arrangement_knobs()
    try:
        key = chosen_key(spec.key, pool=resolved.key_pool, seed=spec.seed)
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
            knobs=knobs,
        )
    except DurationUnfulfillableError as exc:
        raise CompositionEngineError(
            code=EngineErrorCode.DURATION_UNFULFILLABLE,
            message=str(exc),
        ) from exc

    ensemble = resolve_ensemble(spec)
    score, chord_bars, bar_keys = _build_score(
        spec,
        key,
        time_signature,
        arrangement,
        ensemble,
        plan=resolved,
    )
    lint_report = lint(
        score,
        chord_bars=chord_bars or None,
        bar_keys=bar_keys or None,
        voice_instruments=ensemble.voice_instruments(),
    )
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
        arc=resolved.section_arc(),
        drum_set_legacy=drum_set_legacy,
    )
    return EngineOutput(
        notation_score=score,
        performance_plan=performance,
        arrangement=arrangement,
        key=key,
        time_signature=time_signature,
        chord_bars=chord_bars,
        bar_keys=bar_keys,
        voice_instruments=tuple(
            VoiceInstrument(voice_id=voice_id, instrument=instrument)
            for voice_id, instrument in sorted(ensemble.voice_instruments().items())
        ),
        # The resolved plan, not the argument: a caller who supplied none
        # still gets the record of what the piece was composed under, so a
        # sidecar replays from its own contents rather than from a default
        # re-derived at read time (see the module docstring of `plan.py`).
        plan=resolved,
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
    *,
    plan: CompositionPlan,
) -> tuple[NotationScore, tuple[tuple[int, ...], ...], tuple[KeySignature, ...]]:
    """Build the NotationScore from the spec + arrangement.

    Generates one melody voice + one bass voice per section, with
    per-section seed-derived variation when the arrangement has
    multiple repetitions. If the arrangement has a coda, an extra
    coda-length tail is appended using a coda-flavored seed so the
    variation rules from §10 #10 still apply.

    Also returns the chord pitch classes sounding in each bar and the key
    each of those bars belongs to, in bar order — the linter's chord-tone
    and passing-tone gates consume them.
    """
    # The window the melody is written in belongs to the instrument that
    # carries it — raised, if it has to be, to leave the accompaniment a
    # register underneath — and it is read once for the piece: a band
    # that moved between sections would make the tessitura a per-section
    # fact and the register the whole walk is quantised to would not
    # hold.
    shape = plan.melody_shape()
    voices = plan.harmony_voices()
    band = _melody_band_for(
        melody=ensemble.melody,
        bed=ensemble.harmony,
        band_semitones=shape.line_band_semitones,
        clearance=voices.melody_clearance,
    )
    measures: list[Measure] = []
    notes: list[NoteEvent] = []
    chord_bars: list[tuple[int, ...]] = []
    bar_keys: list[KeySignature] = []
    section_starts: list[int] = []  # start_tick of each section
    cursor_tick = 0
    prev_bass: int | None = None
    # The melody's last sounding pitch and the interval that arrived
    # there, carried across section boundaries exactly as the bass walk
    # is: a section boundary is a bar line, and the bar after it has to
    # join the line the way any other bar does. Without this the first
    # bar of every section was placed against nothing, which is where
    # the widest entrances and the unanswerable ones came from.
    prev_melody: int | None = None
    prev_melody_leap: int | None = None
    bars_since_breath = 0

    rng_base_seed = spec.seed if spec.seed is not None else 0
    knobs = plan.arrangement_knobs()
    arc = plan.section_arc()
    long_piece = arrangement.repetition_count >= knobs.arc_min_reps
    bass_figures = plan.bass_figures
    harmony_voices = tuple(
        (voice_id, instrument)
        for voice_id, instrument in ensemble.voice_instruments().items()
        if voice_id >= VOICE_HARMONY
    )

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
        # The plan's rate, cut in before anything rewrites the section's
        # ending: a close is two one-bar chords, and re-cutting the bars
        # after it would overwrite the cadence the plan asked for.
        section_template = _with_harmonic_rhythm(section_template, plan)
        if is_final_section:
            # The last repetition must land at home: rewrite its last
            # two bars as the plan's cadence (earlier sections may end
            # open — their V resolves into the next section's I).
            section_template = apply_final_cadence(
                section_template,
                cadence_degree=plan.cadence_degree,
                seventh=plan.cadence_seventh,
            )
        else:
            # Every earlier section closes the way the plan says, which by
            # default is a half cadence: a phrase that stops mid-thought
            # leaves the bar's harmony unstated, and the resolution the
            # comment above describes had nothing to resolve. The piece's
            # own ending is not governed by this field — a piece that does
            # not land at home is a different request than this vocabulary
            # has a name for.
            section_template = apply_section_close(
                section_template,
                close=plan.section_close,
                cadence_degree=plan.cadence_degree,
                seventh=plan.cadence_seventh,
            )
        # Long pieces lift the final repetition by the plan's offset —
        # the piece ends in the new key, so the coda (which follows it)
        # stays lifted too and the ending keeps its cadence.
        key_offset = plan.modulation_offset if is_final_section and long_piece else 0
        section_result = _generate_section(
            key=key,
            time_signature=time_signature,
            template=section_template,
            section_start_tick=cursor_tick,
            rng=section_rng,
            seed_for_variation=rng_base_seed + section_idx,
            band=band,
            shape=shape,
            voices=voices,
            figures=bass_figures,
            bass_root_motion=plan.bass_root_motion,
            prev_bass=prev_bass,
            prev_melody=prev_melody,
            prev_melody_leap=prev_melody_leap,
            is_final_section=is_final_section,
            key_offset=key_offset,
            melody_from_bar=arrangement.intro_bars if section_idx == 0 else 0,
            bars_since_breath=bars_since_breath,
            harmony_voices=_active_harmony_voices(
                harmony_voices,
                section_index=section_idx,
                long_piece=long_piece,
                cycle=arc.texture_cycle,
            ),
        )
        section_notes, section_chord_bars, section_bar_keys, bars_since_breath = section_result
        chord_bars.extend(section_chord_bars)
        bar_keys.extend(section_bar_keys)
        # Terraced dynamics: the section's whole dynamic sits at its
        # step of the arc rather than drifting continuously.
        velocity_scale = _section_velocity_scale(
            section_idx, arrangement.repetition_count, arc
        )
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
            # The walk carries the *landing* tone across the seam, not
            # the last note heard: a figure decorates above its landing
            # tone, so the last note of a bar is wherever the figure
            # climbed to, and joining the next chord to it would move the
            # walk by the figure's reach rather than by the harmony's
            # step. The landing tone is the last bar's lowest bass note —
            # every rung is resolved at or above it, and rung 0 sounds it
            # on the bar line.
            ticks = bar_ticks(time_signature)
            last_bar = max(n.tick for n in bass_notes) // ticks
            prev_bass = min(
                n.pitch_midi for n in bass_notes if n.tick // ticks == last_bar
            )
        melody_notes = [n for n in section_notes if n.voice_id == VOICE_MELODY]
        if melody_notes:
            prev_melody = melody_notes[-1].pitch_midi
            prev_melody_leap = (
                melody_notes[-1].pitch_midi - melody_notes[-2].pitch_midi
                if len(melody_notes) >= 2
                else None
            )
        cursor_tick += arrangement.form_bars * bar_ticks(time_signature)

    # Optional coda: append a coda-length tail using the same chord
    # template (truncated to coda_bars). The coda gets its own RNG
    # seed (a stable offset from the spec seed) so it sounds distinct
    # from the body, per §10 #10. It is also the piece's true ending,
    # so it carries the final cadence.
    if arrangement.coda_bars > 0:
        coda_template = _truncate_template_for_coda(arrangement.template, arrangement.coda_bars)
        # The coda is a section of the piece, so it changes chords at the
        # piece's rate too — and its own cadence is cut in after, for the
        # same reason the body's sections' are.
        coda_template = _with_harmonic_rhythm(coda_template, plan)
        coda_template = apply_final_cadence(
            coda_template,
            cadence_degree=plan.cadence_degree,
            seventh=plan.cadence_seventh,
        )
        coda_rng = random.Random(section_seed(spec.seed, arrangement.repetition_count))
        coda_result = _generate_section(
            key=key,
            time_signature=time_signature,
            template=coda_template,
            section_start_tick=cursor_tick,
            rng=coda_rng,
            seed_for_variation=rng_base_seed + arrangement.repetition_count,
            band=band,
            shape=shape,
            voices=voices,
            figures=bass_figures,
            bass_root_motion=plan.bass_root_motion,
            prev_bass=prev_bass,
            prev_melody=prev_melody,
            prev_melody_leap=prev_melody_leap,
            # The outro thins out: the coda opens bass alone, and a
            # long-piece modulation stays lifted through the ending.
            key_offset=plan.modulation_offset if long_piece else 0,
            melody_from_bar=1 if arrangement.coda_bars >= 2 else 0,
            # The coda is the piece's true ending: its final bar must
            # force the tonic resolution the way a last section does.
            is_final_section=True,
            bars_since_breath=bars_since_breath,
            harmony_voices=harmony_voices,
        )
        coda_notes, coda_chord_bars, coda_bar_keys, _ = coda_result
        chord_bars.extend(coda_chord_bars)
        bar_keys.extend(coda_bar_keys)
        velocity_scale = _section_velocity_scale(
            arrangement.repetition_count, arrangement.repetition_count, arc
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

    # The bed's register is a property of the finished piece, not of the
    # section that wrote it: the melody's floor and ceiling are one
    # number each for the whole piece, so the pad is settled against them
    # once, here. It has to wait until every section exists — the tune's
    # band is not known until the tune is — which is the same reason it
    # is a post-pass at all rather than a bound inside
    # `_generate_harmony_section`.
    melody_pitches = [
        note.pitch_midi for note in notes if note.voice_id == VOICE_MELODY
    ]
    if melody_pitches:
        notes = _settle_harmony_register(
            notes,
            melody_floor=min(melody_pitches),
            melody_ceiling=max(melody_pitches),
            registers={voice_id: bed_registers(name) for voice_id, name in harmony_voices},
            clearance=voices.melody_clearance,
        )
        # The pass moves notes after the sections sorted them, and a moved
        # note can land under another note of its own voice at the same
        # tick (a pad's fifth folded below the root that stayed). The
        # score's canonical order is what the engraving and the plan read
        # it back by, so it is restored here rather than left to the call
        # order of the pass.
        notes.sort(key=lambda note: (note.tick, note.voice_id, note.pitch_midi))

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
                    plan.percussion_rest_section * arrangement.form_bars,
                    (plan.percussion_rest_section + 1) * arrangement.form_bars,
                )
            )
        notes.extend(
            _generate_percussion(
                kit=plan.drum_kit(),
                time_signature=time_signature,
                form_bars=arrangement.form_bars,
                repetition_count=arrangement.repetition_count,
                total_bars=arrangement.total_bars_with_coda,
                seed=rng_base_seed,
                rest_bars=frozenset(rest_bars),
                arc=arc,
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
                arrangement.total_bars - knobs.ritardando_bars
            ) * bar_ticks(time_signature)
        tempo_changes = (
            TempoPoint(
                tick=change_tick,
                bpm=round(arrangement.tempo_bpm * arrangement.ritardando_factor, 1),
            ),
        )

    return (
        NotationScore.make(
            ppq=PPQ,
            key=key,
            time_signature=time_signature,
            tempo_bpm=arrangement.tempo_bpm,
            measures=measures,
            notes=notes,
            tempo_changes=tempo_changes,
        ),
        tuple(chord_bars),
        tuple(bar_keys),
    )


# The bass voice's ceiling. A figure's upper rungs climb from the bar's
# landing tone; a rung that reaches past this is taken an octave lower,
# where it is still the same chord tone.
BASS_HIGH_MIDI: int = 67

# The register the walking bass lands in. Both arms of the walk — the one
# that picks the nearest chord tone and the one a template's `bass_degree`
# pins — keep their candidates inside it, so neither can land a bar above
# the point where the figure has nothing left to resolve onto.
BASS_WALK_LOW_MIDI: int = 21
BASS_WALK_HIGH_MIDI: int = 60


def _bass_ladder(
    anchor: int, *, chord_root: int, chord_tones: tuple[int, ...]
) -> tuple[int, ...]:
    """The chord's own tones rising from the bar's landing tone.

    Rung 0 is the landing tone itself and every later rung is the next
    chord tone above the one before, so a figure written in rungs lands
    as the chord's own intervals wherever the walk put it: a
    root-third-fifth figure is a root-third-fifth over every chord of the
    progression, and the same figure reaches the seventh on a seventh
    chord. A triad fills a rung within every octave, so rungs 0 to 3
    exist for any chord a mood can draw.
    """
    pitch_classes = {(chord_root + tone) % 12 for tone in chord_tones}
    ladder = [anchor]
    pitch = anchor
    for _ in range(len(pitch_classes)):
        pitch = next(p for p in range(pitch + 1, pitch + 13) if p % 12 in pitch_classes)
        ladder.append(pitch)
    return tuple(ladder)


def _bass_figure_pitches(
    figure: BassFigure, *, anchor: int, chord_root: int, chord_tones: tuple[int, ...]
) -> tuple[tuple[int, int, int], ...]:
    """Resolve a figure's rungs into the bass register above the anchor.

    Returns `(start, length, pitch)` per sounding note, still in
    sixteenths of the bar. A rung that would climb past the bass ceiling
    is taken an octave lower, and one that still lands below the bar's
    landing tone is dropped: the figure is written for the harmony, not
    for the register the walk happened to land in, and a note under the
    tone the bar opened on is a different figure. Dropping a rung rather
    than rewriting it keeps every remaining onset where the figure put
    it. The landing tone itself is always rung 0 at the bar line, so no
    figure can leave a bar without its harmony on the downbeat.
    """
    ladder = _bass_ladder(anchor, chord_root=chord_root, chord_tones=chord_tones)
    resolved: list[tuple[int, int, int]] = []
    for start, length, rung in figure:
        pitch = ladder[min(rung, len(ladder) - 1)]
        while pitch > BASS_HIGH_MIDI:
            pitch -= 12
        if pitch >= anchor:
            resolved.append((start, length, pitch))
    return tuple(resolved)


def _generate_section(
    *,
    key: KeySignature,
    time_signature: str,
    template: ChordTemplate,
    section_start_tick: int,
    rng: random.Random,
    seed_for_variation: int,
    band: MelodyBand,
    figures: tuple[BassFigure, ...],
    bass_root_motion: bool,
    voices: HarmonyVoices = DEFAULT_HARMONY_VOICES,
    shape: MelodyShape = DEFAULT_MELODY_SHAPE,
    prev_bass: int | None = None,
    prev_melody: int | None = None,
    prev_melody_leap: int | None = None,
    is_final_section: bool = False,
    key_offset: int = 0,
    melody_from_bar: int = 0,
    bars_since_breath: int = 0,
    harmony_voices: tuple[tuple[int, str], ...] = (),
) -> tuple[
    list[NoteEvent],
    tuple[tuple[int, ...], ...],
    tuple[KeySignature, ...],
    int,
]:
    """Generate the bass + melody notes for one section.

    The left hand walks: each chord's bass lands on the chord tone
    nearest the previous chord's bass (root position when there is no
    previous bass, or wherever the template pins `bass_degree`), and the
    slot's figure — drawn from the mood's vocabulary in `motif` — is
    played over that landing tone for every bar of the chord, so the
    bass line moves stepwise through inversions instead of jumping
    root to root, and the walk carries across section boundaries via
    `prev_bass`. `bass_root_motion` narrows those landing candidates to
    the chord's root, so the walk states the harmony where it changes
    and still joins the last bar by the smallest step the octave grid
    allows; the stepwise-through-inversions rule above is what `False`
    keeps. The melody carries too, through `prev_melody` and
    `prev_melody_leap`: a section opens on the line the last one left
    off rather than restarting it. The melody develops the section's
    motif: every bar
    replays it through one classic operation (repetition, transposition,
    sequence, inversion, truncation, ornament) onto that bar's chord,
    walked in the chord's own scale and placed in the tessitura band, so
    a bar joins the one before it by step wherever the chord allows.
    Velocity follows an arch across the section with a slight accent on
    downbeats. Everything is derived from `seed_for_variation`, so
    repeated sections sound different but stay deterministic.

    Phrase shape: one bar per section is the melodic apex (placed at the
    top of the band, near the 60% mark); bars ending a
    4-bar phrase lift off early into a breath — on the dominant's root
    when the chord there is the V (a half cadence), otherwise on a
    shortened chord tone. The breath is guaranteed, not just likely:
    a bar whose distance from the last gap reaches `PHRASE_BARS` is
    forced to breathe, with `bars_since_breath` carrying the previous
    section's trailing run across the boundary. When `is_final_section`
    is set the last bar resolves onto the tonic or its third, held to
    the bar line.

    Each entry in `harmony_voices` gets an instrument-aware texture
    generated from the same resolved chords and cleared of any note
    that crowds the melody.

    Returns the section's notes, the chord pitch classes sounding in
    each bar (for the linter's chord-tone gate), the key each of those
    bars belongs to, and the breath deficit the next section inherits.
    """
    notes: list[NoteEvent] = []
    melody_notes: list[NoteEvent] = []
    bar_pcs: list[tuple[int, ...]] = []
    bar_keys: list[KeySignature] = []
    # The melody's last sounding pitch, carried bar to bar so each bar is
    # placed where it continues the line rather than restarting it, and
    # the interval that arrived there — a bar entered after a leap is
    # the bar that has to answer it. Both are seeded from the previous
    # section's last melody note, so the seam between sections is
    # weighed like any other bar line.
    prev_melody_pitch: int | None = prev_melody
    last_melody_leap: int | None = prev_melody_leap
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
    apex_bar = min(int(template.bars * shape.apex_position), template.bars - 2)
    motif = generate_motif(rng, bar_ticks=ticks_per_bar, shape=shape)

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
        root_offset = chord_root_offset(degree, key, borrowed=slot.borrowed)
        chord_root = tonic_midi + slot_offset + root_offset
        chord_tones = _chord_intervals(degree, key, seventh=slot.seventh, borrowed=slot.borrowed)
        chords.append((chord_root, chord_tones, dur))
        # The key this slot's harmony belongs to — the score's, or the
        # lifted one the modulation carries it to. Every bar of the slot
        # sounds the same pitch classes; the linter checks melody and bass
        # against this set and reads the passing-tone licence against this
        # key, so the two travel together.
        bar_key = transposed_key(key, slot_offset)
        bar_pcs.extend(
            tuple(sorted({(chord_root + tone) % 12 for tone in chord_tones}))
            for _ in range(dur)
        )
        bar_keys.extend([bar_key] * dur)

    cursor = 0
    bar_index = 0
    total_bars = template.bars
    # One figure per chord slot, not per bar: the left hand states a
    # figure for as long as its harmony lasts and changes it when the
    # harmony changes. Drawn from a stream of its own — the bass picking
    # a figure must not shift the melody's draws, or this would be a
    # melody rewrite as well, and nobody asked for one.
    slot_figures = draw_bass_figures(
        figures,
        rng=random.Random(seed_for_variation * 31 + 17),
        count=len(template.chords),
    )
    for slot_index, slot in enumerate(template.chords):
        chord_root, chord_tones, dur = chords[slot_index]

        # A close rewrites a template's last two bars, and `_truncate_template`
        # leaves the chord it broke on behind at zero bars. That slot writes
        # nothing, so it must not move the walk either: `prev_bass` is the last
        # tone that *sounded*, and letting a phantom step set it made the next
        # landing — the cadence's pinned root above all — jump an octave to join
        # a note nobody heard.
        if dur == 0:
            continue

        # Walking bass: the pinned bass degree wins; otherwise the
        # chord tone nearest the previous bass (root on the first
        # chord). That tone is the bar's landing tone — the figure's
        # rung 0 and the walk's own step.
        if slot.bass_degree is not None:
            # The cadence pins the degree, not the octave: the pinned
            # tone keeps its pitch class and takes the register nearest
            # the line it joins, the way every other chord's landing tone
            # is drawn from the register nearest the walk. Without this
            # the fixed octave makes the cadence the one place the bass
            # leaps, since the walk has no say in where the pin lands.
            pinned = _octave_down(
                tonic_midi
                + slot_offset
                # A pinned degree belongs to the chord's own table, so a
                # borrowed chord would pin the other mode's degree.
                # `test_a_pinned_bass_degree_is_never_a_borrowed_chord`
                # sweeps the tables and holds that no pin is ever borrowed —
                # the argument is inert today and correct if that changes.
                + chord_root_offset(slot.bass_degree, key, borrowed=slot.borrowed),
                octaves=1,
            )
            if prev_bass is None:
                bass_pitch = pinned
            else:
                # The pin is chosen the way the walk's own landing is, and
                # that includes whose register it may take it in: an octave
                # above the walk's ceiling leaves the figure every rung
                # clamped below its own anchor, and the bar loses its bass.
                in_register = [
                    p
                    for p in (pinned - 12, pinned, pinned + 12)
                    if BASS_WALK_LOW_MIDI <= p <= BASS_WALK_HIGH_MIDI
                ]
                last_bass = prev_bass
                bass_pitch = min(
                    in_register or [pinned], key=lambda p: (abs(p - last_bass), p)
                )
        else:
            # Which of the chord's tones the landing may choose from. The
            # plan's root motion narrows it to the root, so a chord change
            # is stated where it happens; the alternative is every chord
            # tone, which lets the walk join the last bar by the smallest
            # step and land on an inversion — or on a tone the two chords
            # share, and then the bass does not move at all.
            tones = (0,) if bass_root_motion else chord_tones
            # Two octaves of candidates keep the walk inside the bass
            # register even in sharp minor keys whose chord roots sit
            # above the middle of the keyboard.
            candidates = [
                _octave_down(chord_root + tone, octaves=octaves)
                for tone in tones
                for octaves in (1, 2)
            ]
            candidates = [
                c for c in candidates if BASS_WALK_LOW_MIDI <= c <= BASS_WALK_HIGH_MIDI
            ]
            if not candidates:
                bass_pitch = _octave_down(chord_root, octaves=2)
            elif prev_bass is None:
                bass_pitch = candidates[0]
            else:
                last_bass = prev_bass
                bass_pitch = min(candidates, key=lambda c: abs(c - last_bass))
        # The slot's figure, its rungs resolved onto this chord and this
        # landing tone. Every bar of the slot plays it, which is what
        # makes the left hand an accompaniment rather than a new idea
        # every bar.
        bar_figure = _bass_figure_pitches(
            slot_figures[slot_index],
            anchor=bass_pitch,
            chord_root=chord_root,
            chord_tones=chord_tones,
        )

        chord_root_tick = section_start_tick + cursor
        for _bar in range(dur):
            bar_tick = chord_root_tick + _bar * ticks_per_bar
            bar_pos = (cursor + _bar * ticks_per_bar) / max(1, section_ticks)
            is_final_bar = is_final_section and bar_index == total_bars - 1

            # Left hand: the slot's figure, whose rung 0 states the walk's
            # landing tone on the bar line. A close is stated, not
            # decorated, so the piece's last bar plays the plain figure
            # whatever its slot drew — and the cadence's pinned degree
            # stays where the final bar's harmony already is.
            figure = bar_figure
            if is_final_bar:
                figure = _bass_figure_pitches(
                    PLAIN_BASS_FIGURE,
                    anchor=bass_pitch,
                    chord_root=chord_root,
                    chord_tones=chord_tones,
                )
            for figure_index, (offset16, length16, pitch) in enumerate(figure):
                onset = offset16 * ticks_per_bar // 16
                notes.append(
                    NoteEvent(
                        voice_id=VOICE_BASS,
                        pitch_midi=pitch,
                        tick=bar_tick + onset,
                        duration_ticks=length16 * ticks_per_bar // 16,
                        velocity=_shaped_velocity(
                            # The bar's first note carries the weight;
                            # the ones after it are answered, not stated.
                            base=56 if figure_index == 0 else 50,
                            position=bar_pos,
                            tick=bar_tick + onset,
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
            is_apex = bar_index == apex_bar
            # The bar's own chord, not the template's last one: reading
            # the loop-leaked `degree` here meant a half cadence could
            # only ever fall on a form whose final slot was the dominant.
            is_half_cadence = slot.degree == 4 and bar_index % 4 == 3 and not is_final_bar
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
                variant = vary_motif(motif, rng, shape=shape)
            # Anacrusis: when the next bar exists, the pickup leads into
            # it from the pickup chord's tones — the one nearest the note
            # it follows, skipping any candidate that would sound a close
            # m2/M7 against the bar's sounding bass. If every candidate
            # clashes the pickup is dropped. Zero-length slots (a
            # template truncation artifact) are skipped — they never
            # sound, so anticipating them would be wrong.
            pickup: tuple[int, tuple[int, ...]] | None = None
            if not is_final_bar:
                if _bar < dur - 1:
                    pickup = (chord_root, chord_tones)
                else:
                    next_slot = next(
                        (c for c in chords[slot_index + 1 :] if c[2] > 0), None
                    )
                    if next_slot is not None:
                        pickup = (next_slot[0], next_slot[1])
            bar_melody = _melody_bar(
                band=band,
                variant=variant,
                chord_root=chord_root,
                chord_tones=chord_tones,
                scale=bar_scale_intervals(slot.degree, key, borrowed=slot.borrowed),
                anchor=anchor,
                prev_pitch=prev_melody_pitch,
                start_tick=bar_tick,
                bar_ticks=ticks_per_bar,
                rng=rng,
                position=bar_pos,
                ticks_per_bar=ticks_per_bar,
                seed_for_variation=seed_for_variation + bar_index * 101,
                shape=shape,
                is_final_bar=is_final_bar,
                half_cadence=is_half_cadence,
                apex=is_apex,
                breathe=breathe,
                pickup=pickup,
                bass_pitches=tuple(pitch for _offset, _length, pitch in figure),
                prev_leap=last_melody_leap,
            )
            melody_notes.extend(bar_melody)
            prev_melody_pitch = bar_melody[-1].pitch_midi
            last_melody_leap = (
                bar_melody[-1].pitch_midi - bar_melody[-2].pitch_midi
                if len(bar_melody) >= 2
                else None
            )
            bar_index += 1

        # The walk steps from the tone this slot actually stated, whichever
        # arm chose it — the pin outranks the motion policy, not the other
        # way round, and both are the same line here.
        prev_bass = bass_pitch
        cursor += dur * ticks_per_bar

    # Ties hold a repeated pitch across a bar line: when a bar's last
    # melody note and the next bar's first share the pitch and touch,
    # the first is marked tied (the performance layer plays them as one
    # sound; the engraving shows the tie).
    tie_probability = shape.tie_probability
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
    for voice_id, instrument in harmony_voices:
        notes.extend(
            _generate_harmony_section(
                chords=chords,
                section_start_tick=section_start_tick,
                ticks_per_bar=ticks_per_bar,
                rng=rng,
                melody_from_bar=melody_from_bar,
                seed_for_variation=seed_for_variation,
                voice_id=voice_id,
                instrument=instrument,
                window=bed_window(instrument),
                layer_index=voice_id - VOICE_HARMONY,
                voices=voices,
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
    return ordered, tuple(bar_pcs), tuple(bar_keys), trailing + 1


# The harmony voice's constants. The texture it states, the figure that
# texture steps on, the level each sounds at and the clearance between
# the bed and the tune all moved to `saimc.compose.voices`, because the
# plan carries them and the plan cannot import this module. There is no
# register window among this module's remaining constants: the bed is
# folded into its *instrument's* window (`instruments.bed_window`, the
# comfortable range), because "where this instrument sounds like itself"
# does not depend on which role it is playing. Until it moved, the bed
# was folded into a module constant, MIDI 48-84, for every pad
# instrument, so a celesta (lowest note C4 = 60, lives 72-96) had its
# pad written 48-59, a twelfth below the instrument — and the range gate
# could not see it. Which side of the tune the bed settles on is decided
# over the finished piece by `_settle_harmony_register`, against the
# tune's band and the room the accompaniment's own instrument has for
# it.
HARMONY_CROWD_INTERVALS: frozenset[int] = frozenset({0, 1, 2, 10, 11})
# The room the accompaniment needs underneath the tune: one octave, so
# that every one of the twelve pitch classes has an octave of the bed's
# own to be folded into. Narrower than this and the bed can sound some
# chord tones under the tune and not others — the pad answers a D with a
# D two octaves up against a C it answers two octaves down, which is a
# hole in the voicing and a leap in a voice that never leaps. The line
# is raised inside its own instrument to leave this room, and a bed with
# less than it goes above the tune instead; see `_melody_band_for`.
BED_OCTAVE_SEMITONES: int = 12
# The least room a bed can be written in and still be a bed: half an
# octave, the span that covers seven of the twelve pitch classes. A
# window with less than this on the side the bed has chosen cannot voice
# a chord there — every chord tone with no octave is dropped — so the
# settle pass reaches for the instrument's whole compass instead; see
# `_can_clear_the_tune`.
BED_MIN_ROOM_SEMITONES: int = BED_OCTAVE_SEMITONES // 2


def _active_harmony_voices(
    voices: tuple[tuple[int, str], ...],
    *,
    section_index: int,
    long_piece: bool,
    cycle: tuple[str, ...] = HARMONY_TEXTURE_CYCLE,
) -> tuple[tuple[int, str], ...]:
    """Shape long-form density instead of looping one wall of sound.

    The cycle names a group of voices per phase of the arc, repeating
    every four sections: with two harmony colors the opening presents the
    first, the third section becomes a contrasting breakdown led by the
    second, and the intervening sections combine them.

    A group, not an index list, because how many harmony voices a piece
    has is the ensemble's decision — "the leading one" and "every voice
    but it" mean the same thing at two voices and at five.
    """
    if not long_piece or len(voices) < 2:
        return voices
    group = cycle[section_index % len(cycle)]
    if group == HARMONY_TEXTURE_FIRST:
        return voices[:1]
    if group == HARMONY_TEXTURE_REST:
        return voices[1:]
    return voices


def _melody_band_for(
    *,
    melody: str,
    bed: str | None,
    band_semitones: int = LINE_BAND_SEMITONES,
    clearance: int = HARMONY_MELODY_CLEARANCE,
) -> MelodyBand:
    """The window the tune is written in, given what plays underneath it.

    An instrument's own band is where its melody would go if it played
    alone. A piece is not that: the accompaniment needs a register of its
    own, and the register it needs is an octave of its instrument's
    comfortable range, under the tune. So the tune is raised inside its
    own range until that room exists — a piano tune over a string pad is
    an F3-C5 band raised to an E♭4-C6 one, which is where the two
    instruments can both be heard — and it keeps the band's width, so the
    line still has the twelfth the walk needs.

    The raise is bounded by the tune instrument's own comfort, and that
    bound is the point rather than a detail. A tuba has nothing above a
    string pad's floor to be raised into, and a cello cannot get clear of
    a piano's; in those pairs the raise does not happen, the bed's
    instrument has no octave underneath the tune, and
    `_settle_harmony_register` writes the bed *above* it instead. Which
    side of the tune the bed ends up on is therefore decided here, once,
    by what the two instruments can do — not per bar, which would move a
    held pad note an octave mid-chord.

    With no accompaniment voice the tune sits in its instrument's own
    band, which is what a solo piece is.

    The clearance is an argument for the reason the band's width is one:
    both are the plan's, and neither is a fact about the instrument, so
    this function is handed them rather than reaching for a table.
    """
    band = melody_band(melody, band_semitones=band_semitones)
    if bed is None:
        return band
    window = bed_window(bed)
    floor = window.low_midi + BED_OCTAVE_SEMITONES + clearance
    if floor <= band.low_midi:
        return band
    span = range_for(melody)
    if floor > span.tessitura_high - band_semitones:
        return band
    return MelodyBand(low_midi=floor, high_midi=floor + band_semitones)


def _octaves_in_window(pitch: int, window: MelodyBand) -> list[int]:
    """Every octave of `pitch` inside `window`, lowest first.

    An octave shift is the only move a bed note has: it keeps the pitch
    class, and so the chord tone the note was written as. A window
    narrower than an octave can miss a pitch class entirely, and an
    empty list is the honest answer there — the caller drops the note
    rather than inventing a pitch the instrument cannot sound.

    Only `bed_window` windows are passed, and those span an instrument's
    comfortable range, so every pitch class has an octave in one and the
    empty case is for an instrument whose whole compass is narrower than
    an octave — for which there is no honest note to write.
    """
    octave = pitch
    while octave > window.high_midi:
        octave -= 12
    while octave < window.low_midi:
        octave += 12
    if octave > window.high_midi:
        return []
    while octave - 12 >= window.low_midi:
        octave -= 12
    octaves: list[int] = []
    while octave <= window.high_midi:
        octaves.append(octave)
        octave += 12
    return octaves


def _into_harmony_register(pitch: int, *, window: MelodyBand) -> int:
    """Octave-shift a chord tone into the harmony bed's register.

    The fold moves in the direction it has to: down while the pitch is
    over the window, then up while it is under it, which is what the
    fixed 48-84 fold did when the window was those two constants. An
    octave shift keeps the pitch class (and so the chord tone); a clamp
    would not. A pitch class the window does not contain cannot be
    folded at all and comes back out of the window untouched —
    `_settle_harmony_register` is what drops it, because the note is
    unplayable either way and only one of the two is honest about it.
    """
    while pitch > window.high_midi:
        pitch -= 12
    while pitch < window.low_midi:
        pitch += 12
    return pitch


def _generate_harmony_section(
    *,
    chords: list[tuple[int, tuple[int, ...], int]],
    section_start_tick: int,
    ticks_per_bar: int,
    rng: random.Random,
    melody_from_bar: int,
    seed_for_variation: int,
    voice_id: int = VOICE_HARMONY,
    instrument: str = "piano",
    window: MelodyBand,
    layer_index: int = 0,
    voices: HarmonyVoices = DEFAULT_HARMONY_VOICES,
) -> list[NoteEvent]:
    """Generate the harmony voice for one section from the resolved chords.

    The plan's texture decides the shape: a sustained pad, a broken-chord
    ostinato, or — for an instrument that does that well — spacious chord
    accents. A broken chord is stated by the leading harmony layer only;
    additional harmony colors form a quieter sustained bed. Intro bars
    stay silent — harmony enters with the melody.

    The bed is written inside `window` — the instrument's own, from
    `saimc.instruments` — so a pad is written where the instrument that
    plays it can sound. Where that window is relative to the tune is
    settled later, over the finished piece, by
    `_settle_harmony_register`; nothing here reads the tune, so the RNG
    draws do not depend on what it happens to do.
    """
    notes: list[NoteEvent] = []
    pad = not voices.broken_chord or layer_index > 0
    stabs = voices.broken_chord and instrument in HARMONY_STAB_INSTRUMENTS
    # Which chord tone sits lowest: the rotation (not the bar) decides
    # it, so the section's voicing stays stable instead of churning.
    rotation = rng.randrange(3)
    step_ticks = voices.arpeggio_step_ticks
    arpeggio_steps = max(1, ticks_per_bar // step_ticks)

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
            if stabs:
                pulse_duration = max(PPQ // 2, min(PPQ, ticks_per_bar // 4))
                for pulse_index, pulse_tick in enumerate((0, ticks_per_bar // 2)):
                    low = (bar_index + rotation + pulse_index) % len(chord_tones)
                    pair = (chord_tones[low], chord_tones[(low + 2) % len(chord_tones)])
                    for tone in pair:
                        notes.append(
                            NoteEvent(
                                voice_id=voice_id,
                                pitch_midi=_into_harmony_register(chord_root + tone, window=window),
                                tick=bar_tick + pulse_tick,
                                duration_ticks=pulse_duration,
                                velocity=_shaped_velocity(
                                    base=voices.stab_velocity,
                                    position=position,
                                    tick=bar_tick + pulse_tick,
                                    ticks_per_bar=ticks_per_bar,
                                    rng_seed=seed_for_variation + bar_tick * 103 + pulse_index,
                                ),
                            )
                        )
            elif pad:
                low = (bar_index + rotation) % len(chord_tones)
                pair = (chord_tones[low], chord_tones[(low + 2) % len(chord_tones)])
                for tone in pair:
                    notes.append(
                        NoteEvent(
                            voice_id=voice_id,
                            pitch_midi=_into_harmony_register(chord_root + tone, window=window),
                            tick=bar_tick,
                            duration_ticks=ticks_per_bar,
                            velocity=_shaped_velocity(
                                base=voices.pad_velocity - layer_index * 4,
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
                            voice_id=voice_id,
                            pitch_midi=_into_harmony_register(chord_root + tone, window=window),
                            tick=bar_tick + step * step_ticks,
                            duration_ticks=step_ticks,
                            velocity=_shaped_velocity(
                                base=voices.arpeggio_velocity,
                                position=position,
                                tick=bar_tick + step * step_ticks,
                                ticks_per_bar=ticks_per_bar,
                                rng_seed=seed_for_variation + bar_tick * 101 + step,
                            ),
                        )
                    )
            bar_index += 1
        cursor += dur * ticks_per_bar

    # Register and clearance are settled over the finished piece, by
    # `_settle_harmony_register`. The bed is written where the instrument
    # sounds and left alone here: a pass in this loop would be fixing a
    # register without knowing the tune it has to clear.
    return notes


def _crowds_melody(note: NoteEvent, melody_notes: list[NoteEvent], pitch: int) -> bool:
    """Does a harmony pitch collide with any simultaneously sounding melody note?"""
    return any(
        melody.tick < note.tick + note.duration_ticks
        and note.tick < melody.tick + melody.duration_ticks
        and abs(melody.pitch_midi - pitch) in HARMONY_CROWD_INTERVALS
        for melody in melody_notes
    )


def _settle_harmony_register(
    notes: list[NoteEvent],
    *,
    melody_floor: int,
    melody_ceiling: int,
    registers: Mapping[int, BedRegisters],
    clearance: int = HARMONY_MELODY_CLEARANCE,
) -> list[NoteEvent]:
    """Settle every harmony voice inside its instrument's window, clear of the tune.

    Two constraints, and the first is not negotiable: a note must be
    inside its instrument's window, because a celesta cannot sound a C3
    and the gate that used to check this was the piano's compass applied
    to every voice. The second is that it should stay `clearance` clear
    of the tune's band — a bed note inside the melody's register is what
    `tessitura_overlap_semitones` measures and what a hot pot of
    instruments sounds like.

    Each voice is settled on *one* side of the tune, and which side is
    the voice's decision rather than each note's. Under the tune is
    where a bed belongs, and it is where every voice goes whose window
    can hold an octave down there — an octave being what gives each
    chord tone a place to fold to. A window that cannot (a celesta's
    comfortable range starts at C5, and a cello's melody sits too low
    for the strings under it to reach below) puts the bed above the
    tune instead of losing it. `_melody_band_for` is what makes the
    first case the common one — the tune is raised so that an octave
    underneath exists — and this pass is what handles the pairs where
    it cannot be raised.

    The side is the voice's rather than the note's because a bed with
    notes either side of the tune is not a bed: it is the register this
    pass exists to keep out of, and it is what a per-note choice
    produced the moment the octave under the tune happened to rub a
    melody note — that one note jumped over the tune, held there for a
    bar, and left the voice's range closed around the melody's.

    A note the chosen side has no octave for that clears the tune is
    dropped, which is the same accepted outcome the crowding pass has.
    Keeping it instead would put a pad note in the tune's own band, and
    a bed that shares the tune's register is the one thing this pass
    exists to prevent; it is also, measured over the gate matrix, one or
    two notes a piece rather than a hole in the texture, because the
    arrangement has already left the bed an octave to fold into.

    The bounds are the piece's, not the bar's, and deliberately so. The
    tune's floor and ceiling are one number each for the whole piece, so
    a bar whose melody happens to sit high cannot admit a high bed note
    that is inside the melody's band for the piece as a whole. It also
    keeps the bed in one register: a bar-local bound would move one pad
    note an octave between two bars — a leap in a voice that never
    leaps, in the middle of a held chord.

    A note that moves is asked the crowding question again, because an
    octave shift can land it a seventh under a melody note it never met
    where it was.

    It runs over the finished piece, after every section and the coda,
    for the same reason the crowding pass runs after generation: the
    draws must not depend on what the tune did. It changes pitches only —
    the arpeggio's onsets, the pad's held bars and the stabs' pulses are
    all left where they were.
    """
    below_ceiling = melody_floor - clearance
    above_floor = melody_ceiling + clearance
    melody = [note for note in notes if note.voice_id == VOICE_MELODY]
    settled = [note for note in notes if note.voice_id not in registers]
    for voice_id, voice_registers in registers.items():
        settled.extend(
            _settle_voice(
                [note for note in notes if note.voice_id == voice_id],
                registers=voice_registers,
                melody=melody,
                below_ceiling=below_ceiling,
                above_floor=above_floor,
            )
        )
    return settled


def _settle_voice(
    voice_notes: list[NoteEvent],
    *,
    registers: BedRegisters,
    melody: list[NoteEvent],
    below_ceiling: int,
    above_floor: int,
) -> list[NoteEvent]:
    """Settle one harmony voice on one side of the tune.

    The side is decided once, for the voice, by `_bed_goes_above`; the
    notes then fold to the octave of that side nearest the tune that does
    not rub a simultaneously sounding melody note. A note the side has no
    such octave for is dropped — see `_settle_harmony_register`.

    The comfortable range is tried first and the whole compass only if the
    tune leaves it no register clear of itself; a voice with neither is
    kept where the harmony pass wrote it rather than emptied.
    """
    for window in (registers.comfortable, registers.compass):
        above = _bed_goes_above(window, below_ceiling=below_ceiling, above_floor=above_floor)
        if _can_clear_the_tune(window, below_ceiling=below_ceiling, above_floor=above_floor, above=above):
            break
    else:
        # Neither window has a register clear of the tune — a compass
        # forty semitones wide with a tune filling twenty-one of them
        # leaves nowhere to fold to. The notes are kept where the harmony
        # pass wrote them, because a bed in the tune's register is worth
        # more than no bed at all; the ones that rub the tune still go,
        # since a clash is never what keeping the register is for.
        return [
            note for note in voice_notes if not _crowds_melody(note, melody, note.pitch_midi)
        ]
    settled: list[NoteEvent] = []
    for note in voice_notes:
        if (
            not above
            and note.pitch_midi <= below_ceiling
            and window.contains(note.pitch_midi)
            and not _crowds_melody(note, melody, note.pitch_midi)
        ):
            # Already under the tune, inside the bed's window, and clear
            # of it: there is nothing to move it to.
            settled.append(note)
            continue
        octaves = _octaves_in_window(note.pitch_midi, window)
        candidates = [
            octave
            for octave in octaves
            if (octave >= above_floor if above else octave <= below_ceiling)
        ]
        # Nearest the tune first — the highest octave under it, the
        # lowest over it — so the bed sits as close to the tune as its
        # instrument allows without either entering it.
        pitch = next(
            (
                octave
                for octave in (candidates if above else reversed(candidates))
                if not _crowds_melody(note, melody, octave)
            ),
            None,
        )
        if pitch is not None:
            settled.append(note if pitch == note.pitch_midi else replace(note, pitch_midi=pitch))
    return settled


def _bed_goes_above(window: MelodyBand, *, below_ceiling: int, above_floor: int) -> bool:
    """Whether this voice's accompaniment sits above the tune rather than under it.

    Under, whenever the window holds a whole octave there: a bed belongs
    under a tune, and an octave is what gives every chord tone a place to
    fold to. `BED_OCTAVE_SEMITONES` is the same octave `_melody_band_for`
    raises the tune to leave room for, so a pairing that could be raised
    lands here with exactly an octave of room and goes under.

    Only a window that cannot fit an octave underneath is asked which
    side it prefers, and then it is the roomier one. That is the celesta
    over a low tune — its comfortable range starts above the tune's top,
    so it has no room underneath at all and a bed there sounds high,
    which is where the instrument lives anyway.
    """
    under_room = below_ceiling - window.low_midi
    over_room = window.high_midi - above_floor
    if under_room >= BED_OCTAVE_SEMITONES:
        return False
    return over_room >= BED_OCTAVE_SEMITONES or over_room > under_room


def _can_clear_the_tune(
    window: MelodyBand, *, below_ceiling: int, above_floor: int, above: bool
) -> bool:
    """Whether a bed written in this window has room to clear the tune.

    `BED_MIN_ROOM_SEMITONES` on the side the bed has chosen, which is the
    span that covers seven of the twelve pitch classes. Below that the
    window cannot voice a chord: every chord tone with no octave there is
    dropped, so what is written is not a smaller bed but a bed with holes
    in it, sounding the three or four pitch classes the sliver holds.

    A window with no such room is not used — `_settle_voice` tries the
    instrument's compass next, and keeps the voice where the harmony pass
    wrote it if that has no room either.
    """
    room = below_ceiling - window.low_midi if not above else window.high_midi - above_floor
    return room >= BED_MIN_ROOM_SEMITONES


def _with_harmonic_rhythm(template: ChordTemplate, plan: CompositionPlan) -> ChordTemplate:
    """The plan's rate cut into a template, or the template itself.

    `None` is the plan's spelling for the templates' own rhythm — the rate
    the hand-coded progression already has — and this is where that name is
    read, so the body's sections and the coda cannot disagree about what it
    means or forget it between them.
    """
    if plan.harmonic_rhythm is None:
        return template
    return apply_harmonic_rhythm(template, plan.harmonic_rhythm)


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


# The melody's tessitura is the melody instrument's, and it is looked up
# per piece: `saimc.instruments.melody_band` turns the instrument into
# the window `_place_bar` places every bar inside. There is no module
# constant any more. Until it was removed this was `MELODY_LOW_MIDI = 64`
# to `MELODY_HIGH_MIDI = 84` (E4 to C6) for all sixty-six instruments, so
# a tuba, a piccolo and a piano were written the same twenty semitones
# and `compose()` returned a byte-identical score for each.
#
# What the band is *for* is unchanged: every bar is placed inside it by
# `_place_bar`, which is what bounds the piece's range — a bar is moved
# to a register, never clamped note by note into one. It is wide enough
# for a phrase peak and for a line that moves, and narrow enough that the
# accompaniment has a register of its own below it. How wide that has to
# be is `instruments.LINE_BAND_SEMITONES`, and the window is placed at
# the middle of the tessitura there; this is the same constraint seen
# from the other end.
#
# The floor and ceiling are only reachable by octave placement, and a
# line wider than the band divided by 12 rotations is not guaranteed a
# register inside it: a twelve-semitone line sits in 9 of the 12 octaves
# it could be written in, so 3 of them have no register inside the band.
# A bar against them has nowhere left to go — so it sounds where the
# placement put it, up to a tone or two past the edge, rather than being
# displaced note by note into a tear. `_place_bar` ranks placements by
# how many notes each leaves outside, so a bar that *can* fit does; the
# few that cannot are the price of an octave-quantised register, paid at
# the edge by a tone. A narrow instrument pays it more often, which is
# the honest cost of writing for it rather than for the piano.
# How far a bar's walk may reach from its anchor before it is folded
# back an octave. Ten degrees is at most 18 semitones in either diatonic
# mode, so a bar that stays inside this window fits the band above and
# is never folded for reasons the motif did not already imply.
_WALK_REACH_DEGREES: int = 5
# How far from the drawn anchor a bar may be restated, in scale degrees:
# an octave each way. The band is narrower than that in every direction
# that matters, so a wider lattice would only offer placements the
# tessitura refuses — but a narrower one would leave the octave grid
# (`_place_bar`) as the only way to move a line, and an octave step is
# twelve semitones when the entrance wanted three.
_START_REACH_DEGREES: int = 8
# The widest interval a bar may be entered on. A leap is recovered by the
# step that follows it, and past an octave there is no answer the ear
# accepts: the line is lost before the recovery arrives. It binds in one
# place — a bar whose line fits the band in a single register and whose
# every restatement rubs the bass has no way in but a leap, and this says
# which leap. Inside the octave, size is still the caller's tiebreak.
# Which rank position it occupies is what makes it bind, and the apex
# needs its own answer: the bound sits behind the register everywhere
# else (the band is a fact about the instrument and a rub is a refusal)
# but ahead of it at the apex, which is the one bar per section that was
# otherwise free to ignore it.
_MAX_ENTRANCE_SEMITONES: int = 12
# Which field of `_place_bar`'s rank tuple counts the notes left rubbing
# the bass. It is the second field of both rank shapes — the apex's and
# every other bar's — and it is read by `_melody_bar`, which retries a bar
# whose winner rubs.
_RANK_RUBBING: int = 1


def _bound_walk(degrees: list[int], centre_degree: int) -> list[int]:
    """Fold a bar's degree walk into one octave of where it starts.

    A sequence climbs a chord tone per replay, and a long bar replays the
    motif many times, so without this the line walks out of the
    instrument. Folding is an octave displacement — what a sequence does
    at its seam anyway — and the leap-answering pass that follows treats
    it as the leap it is.
    """
    low = centre_degree - _WALK_REACH_DEGREES
    high = centre_degree + _WALK_REACH_DEGREES
    folded: list[int] = []
    for degree in degrees:
        while degree > high:
            degree -= 7
        while degree < low:
            degree += 7
        folded.append(degree)
    return folded


def _answer_leaps(
    degrees: list[int],
    *,
    fixed_tail: int = 0,
    remainders: tuple[int, ...] = (0, 2, 4),
    shape: MelodyShape = DEFAULT_MELODY_SHAPE,
) -> list[int]:
    """Answer every leap in a bar's degree walk with a turn back.

    A leap that is not answered is the fault a listener notices first,
    and the answer has to survive the licence pass to be heard at all: a
    single step back lands on a non-chord tone, and the licence covers
    that only when it is entered *and* left by a step. So the answer is
    the whole figure — the step back, and the step that leaves it, which
    is a passing tone between the two chord tones either side of it.

    The turn comes back the way the leap went, so a phrase that has
    climbed answers downward and one that has dived answers up.

    The leap itself is kept whenever it can be: a leap the licence can
    cover lands *on* the harmony — three degrees from the chord's fifth
    is its root — and a leap that lands off it cannot be licensed at all,
    because a non-chord tone is entered by a step or not at all. So a leap
    whose landing is off the harmony is re-aimed at it rather than undone.
    The chord scale puts a chord tone on either side of any landing, the
    side further along the leap is a degree wider than the motif drew it,
    and a degree is the whole of the change — where the motif drew a
    fourth the bar sounds a fifth, and the leap it drew is heard. Only a
    landing with no chord tone beyond it narrows, and one with neither is
    undone as it always was: the landing steps back to where the line came
    from, which leaves the bar's contour intact and one leap poorer rather
    than one dissonance richer.

    Re-aiming happens here rather than where the motif is drawn, and the
    reason is the frame: `_draw_step` chooses a step from the motif's own
    start, and the chord tones it can see are the ones congruent to *that*
    degree, while this pass reads a landing against the bar's anchor — a
    different chord tone, so a landing the draw called consonant is one
    this pass calls a dissonance. Constraining the draw to chord tones
    measured at half the leaps drawn and *fewer* of them rendered, because
    the two frames disagree in one case in three.

    `fixed_tail` is how many of the bar's last slots the answer may not
    move — the closing gesture's note, which has to land where it lands.
    A leap into one of those is answered from the other side instead:
    the note *before* the landing takes the step, which is how a cadence
    is approached in the first place. Rewriting the note before a
    landing can leave a leap before *that* one, so the loop walks back
    over the slots it has rewritten.

    A slot the pass has written as the *approach* to its pair is settled,
    and settled slots are never written again. Without that, two repairs
    that face each other undo one another on every pass and the walk never
    finishes: a bar whose answer has to come from before the landing can
    be re-leapt by the repair of the pair before it — the engine hung on
    exactly that. Every walk back settles one more slot at a lower index
    than the last, so the walk cannot go round, and a leap whose both
    sides are settled is left to the bar after this one, whose entrance
    answers it (`_entry_answer`).

    A leap into the bar's *last* slot is the same case one step along,
    and it is the case this pass used to get wrong. There is no room
    after such a landing for any answer — the turn needs three slots —
    so the landing was treated as a fault and the leap undone. But a leap
    the bar cannot answer is a leap the bar does not have to answer when
    the bar after it can: `_entry_answer` turns that bar's opening step
    back the way the leap came, and the placement ranks an answered
    entrance above an unanswered one. So the leap is kept, re-aimed at
    the harmony if it needs to be, and handed over — which is what makes
    the seam's own machinery reachable at all. Measured over the 840-piece
    grid, the pass keeps 58% of the leaps the motif draws where it kept
    39%, the pieces whose melody never leaps fall from thirteen to nine,
    and the pieces carrying an interval wider than an octave fall from
    78 to 51.
    """
    out = list(degrees)
    last_mutable = len(out) - fixed_tail - 1
    settled = [False] * len(out)
    index = 0
    while index + 1 < len(out):
        leap = out[index + 1] - out[index]
        if abs(leap) < shape.leap_degrees:
            index += 1
            continue
        back = -1 if leap > 0 else 1
        landing = index + 1
        if not settled[landing]:
            # Room for the whole answer, or room for none of it — but the
            # bar's last slot is not room *less*: it is the one landing the
            # bar after this one answers, so the seam's turn counts here.
            seam = fixed_tail == 0 and landing == last_mutable
            room = seam or (
                index + 3 <= last_mutable
                and not settled[index + 2]
                and not settled[index + 3]
            )
            if out[landing] % 7 not in remainders:
                # The landing needs the licence and cannot have it where it
                # sits: a step is the only way into a non-chord tone. Re-aim
                # it at the harmony, one degree out, and the leap is kept —
                # but only where the answer that licenses it has room to
                # follow, because a leap the bar cannot answer is the fault
                # this pass exists to remove, and keeping one here would be
                # that fault rather than a repair of it.
                aimed = _aim_leap(out[landing], back, remainders=remainders) if room else None
                if aimed is None or abs(aimed - out[index]) < shape.leap_degrees:
                    # Neither side of the landing is a chord tone the leap
                    # survives on — a third from the chord's third is the
                    # case — so the line keeps its shape and loses the leap.
                    out[landing] = out[index] + back
                    if index + 2 <= last_mutable and not settled[index + 2]:
                        out[index + 2] = out[landing] + back
                    index += 1
                    continue
                out[landing] = aimed
            if room:
                if seam:
                    # Nothing left in this bar to write the answer on, and
                    # the bar after it owes the turn.
                    index += 1
                    continue
                # Landing on a chord tone, with room for the whole answer.
                out[index + 2] = out[landing] + back
                out[index + 3] = out[index + 2] + back
                index += 3
                continue
        if settled[index]:
            # Nothing left on this pair that may be written, so the leap
            # stands and the next bar's entrance is what answers it.
            index += 1
            continue
        # No room after the landing for the turn: the landing is where
        # the bar has to be, so the note before it steps into it.
        out[index] = out[landing] + back
        settled[index] = True
        index = max(index - 1, 0)
    return out


def _aim_leap(landing: int, back: int, *, remainders: tuple[int, ...]) -> int | None:
    """The chord degree a leap is re-aimed at, or `None` if there is none.

    A chord built in thirds puts a chord tone on either side of any degree,
    so this is a choice between two and not a search. `back` is the
    direction the line came from, so `landing - back` is the side further
    along the leap and is always the one a leap wants: it is a degree
    wider than the motif drew, so the leap survives by construction. The
    nearer side is the fallback, and it is the caller's to check — one
    degree in can shorten a fourth to a third, and a leap that is no
    longer a leap is the fault this pass exists to answer.
    """
    wide = landing - back
    if wide % 7 in remainders:
        return wide
    narrow = landing + back
    return narrow if narrow % 7 in remainders else None


def _licit_line(
    degrees: list[int],
    *,
    remainders: tuple[int, ...],
    fixed_tail: int = 0,
    shape: MelodyShape = DEFAULT_MELODY_SHAPE,
) -> list[int]:
    """Reshape a walk so every non-chord tone is a passing or neighbour tone.

    A non-chord tone owes the licence a single degree on each side: it is
    entered by a step and left by a step, and one degree is a semitone or
    a whole tone in every diatonic mode. That makes the licence a
    statement about the *walk* — a line that arrives at a non-chord tone
    by a third has already broken the rule, and one that leaves by a
    third breaks it again — and it makes the repair a local one: the
    offending note moves by a degree, the way it was already going.

    Only the notes around a non-chord tone move, and they move by a
    single degree, so a bar's rhythm is untouched and its contour barely
    shifts. Of the two degrees the moved note could take, the one nearer
    the note on its far side wins, which is what keeps the repair from
    opening a gap of its own.

    An earlier version of this repair snapped the non-chord tone onto the
    chord instead. That is the right answer for a note with nowhere to
    go, but as a general repair it moves the *wrong* note: it puts the
    decorated tone on the harmony and leaves the harmony's own tones a
    third or a fourth apart, which is the very leap the licence exists to
    prevent. Bending the line around the non-chord tone keeps the music
    and removes the fault.

    `fixed_tail` is the bar's closing gesture, which has to land where it
    lands; a line that cannot bend toward it is left for the snap pass,
    which is the one repair allowed to move a note the bar has pinned.
    """
    out = list(degrees)
    count = len(out)
    last_mutable = count - fixed_tail - 1
    for index in range(1, count):
        if out[index] % 7 in remainders and out[index - 1] % 7 in remainders:
            continue
        gap = out[index] - out[index - 1]
        if abs(gap) == 1:
            continue
        # A repeat is as unwalkable here as a leap: the non-chord tone
        # has to be left by a degree, and staying still is not one.
        upward = gap > 0 if gap else _was_rising(out, index)
        far = out[index + 1] if index + 1 < count else None
        if index <= last_mutable:
            out[index] = _bent_step(out[index - 1], upward, far, shape=shape)
        elif index - 1 <= last_mutable:
            out[index - 1] = _bent_step(
                out[index], not upward, out[index - 2] if index > 1 else None, shape=shape
            )
    return out


def _bent_step(
    anchor: int, upward: bool, far: int | None, *, shape: MelodyShape = DEFAULT_MELODY_SHAPE
) -> int:
    """One degree from `anchor`, in the direction the line was going.

    A bend is the licence's repair, and a leap is the fault that licence
    exists to prevent, so a candidate that leaves `far` a step away
    outranks one that is merely nearer to it — a bend that closes the gap
    it was called for and opens a fourth on the far side has repaired
    nothing. Where neither is a step away, the other candidate wins only
    when it leaves `far` strictly nearer, which is the older rule and
    still the one that keeps a line's intervals from widening.
    """
    up, down = anchor + 1, anchor - 1
    if far is not None:
        ordered = (up, down) if upward else (down, up)
        for step in ordered:
            if abs(far - step) <= shape.chord_tone_degrees:
                return step
        if abs(far - ordered[1]) < abs(far - ordered[0]):
            return ordered[1]
    return up if upward else down


def _was_rising(degrees: list[int], index: int) -> bool:
    """Whether the line was rising before a repeated note."""
    for position in range(index - 1, 0, -1):
        step = degrees[position] - degrees[position - 1]
        if step:
            return step > 0
    return True


def _snap_to_chord(
    degree: int,
    *,
    tone_count: int,
    prefer_up: bool,
    neighbours: tuple[int, ...] = (),
    shape: MelodyShape = DEFAULT_MELODY_SHAPE,
) -> int:
    """The bar's nearest chord degree to `degree`, preferring one direction.

    A chord scale's tones sit on every other degree, so a tone that is
    not a chord tone always has one within two degrees — which is what
    makes this a snap and not a search.

    `neighbours` are the degrees the snapped note sits between, and a
    snap decides the intervals *they* are heard on, so what the chord
    tone costs them is weighed before how far the note itself moves:

    - A neighbour the licence covers is a step away, and it has to stay
      one. Snapping a note onto a chord tone a third from such a
      neighbour leaves the neighbour a note the licence cannot cover any
      more, so the pass snaps that one too — and two snaps facing away
      from each other widen a step into a leap, which is the fault the
      whole line is written to avoid.
    - A chord-tone neighbour needs no licence, but a leap between two
      chord tones is still a leap: the line has to answer it, and the
      room to answer it may not be there. So a leap behind the note
      costs less than a stranded neighbour and more than neither.

    Only among chord tones that are equally kind to the neighbours does
    the smaller move win, and then the direction the line was going.
    """
    remainders = chord_tone_degrees(tone_count)
    candidates = [
        degree + offset
        for offset in (1, -1, 2, -2)
        if (degree + offset) % 7 in remainders
    ]

    def cost(candidate: int) -> tuple[int, int]:
        """How many neighbours the move leaves stranded, and how many leaping."""
        stranded = leapt = 0
        for neighbour in neighbours:
            gap = abs(neighbour - candidate)
            if neighbour % 7 in remainders:
                leapt += gap >= shape.leap_degrees
            else:
                stranded += gap != 1
        return stranded, leapt

    return min(
        candidates,
        key=lambda candidate: (
            *cost(candidate),
            abs(candidate - degree),
            0 if (candidate > degree) == prefer_up else 1,
        ),
    )


def _hold_tied_pairs(slots: list[BarSlot]) -> list[BarSlot]:
    """Write a tied pair as the one pitch it is.

    A tie joins two noteheads into a single sound, so the second of them
    has to carry the first's degree. The passes between the rhythm
    library and the licence each rewrite a note of the line without
    knowing which notes are tied to their neighbour — the entry turn
    writes the bar's opening two moves outright — and a pair left at two
    pitches is not a tie at all: the engraver draws a slur between
    different heights, and the performance layer, which plays a tied
    continuation as nothing, drops a pitch the line meant to sound.

    Both halves are chord tones of the bar's chord scale when the rhythm
    library drew the tie, so holding the pair together invents no
    dissonance; it also cannot open a leap, since the tie's own pitch is
    the one already there.
    """
    out = list(slots)
    for index, slot in enumerate(out[:-1]):
        if slot[3]:
            offset, duration, _degree, tie = out[index + 1]
            out[index + 1] = (offset, duration, slot[2], tie)
    return out


def _legal_slots(
    slots: list[BarSlot],
    *,
    tone_count: int,
    chord_root: int,
    scale: tuple[int, ...],
    avoid_pcs: frozenset[int] = frozenset(),
    shape: MelodyShape = DEFAULT_MELODY_SHAPE,
) -> list[BarSlot]:
    """Snap every slot the passing-tone licence cannot cover to a chord tone.

    A slot sounding a chord tone of the bar (`forms.chord_tone_degrees`)
    needs no licence. A slot that does not is a non-chord tone, and the
    licence covers it only when it is unaccented, no longer than a
    quarter, and entered and left by a step that abuts it on both sides
    (`linter.legal_non_chord_tone`). Whatever fails those is snapped to
    the nearest chord degree in the direction the line was already
    moving, so the bar keeps its rhythm and its contour and the note
    sounds the chord tone it was decorating instead of one the harmony
    forbids.

    `avoid_pcs` are the pitch classes of the bar's other sounding
    voices, and a non-chord tone a semitone from one of them is snapped
    whatever its surroundings: the collision check flags a m2 and a M7
    between two sounding notes, and those are the same pair of pitch
    classes an octave apart, so the pass decides in pitch class where no
    octave placement can undo it. Only non-chord tones are affected — a
    chord tone against a chord tone of the same bar is a voicing, which
    the check exempts.

    A tie holds one pitch across two noteheads, so a tied pair stands or
    falls together and is snapped as a unit — otherwise the two halves
    could snap opposite ways and the engraver would draw a tie between
    two different pitches.
    """
    degrees = [slot[2] for slot in slots]
    count = len(degrees)
    if count == 0:
        return slots
    tied = [bool(slot[3]) for slot in slots]
    remainders = chord_tone_degrees(tone_count)

    def pitch_class(index: int) -> int:
        """The slot's pitch class — what the octave placement cannot change."""
        return (chord_root + scale[degrees[index] % 7]) % 12

    def clashes(index: int) -> bool:
        """Whether a non-chord tone here would rub another voice."""
        return any(
            (pitch_class(index) - other) % 12 in DISSONANT_INTERVALS
            for other in avoid_pcs
        )

    def needs_snapping(index: int) -> bool:
        if degrees[index] % 7 in remainders:
            return False
        if clashes(index):
            return True
        # The bar's first slot falls on the downbeat, and its last has no
        # note inside the bar to step away to; either is unlicensable.
        if index == 0 or index == count - 1:
            return True
        if slots[index][1] > PPQ:
            return True
        if tied[index] or tied[index - 1]:
            return True
        # Both neighbours must be a single degree away: one degree is a
        # semitone or a whole tone, two is a third and no step.
        return not (
            abs(degrees[index] - degrees[index - 1]) == 1
            and abs(degrees[index + 1] - degrees[index]) == 1
        )

    changed = True
    while changed:
        changed = False
        for index in range(count):
            if not needs_snapping(index):
                continue
            start = index - 1 if index > 0 and tied[index - 1] else index
            stop = start + 1 if tied[start] else start
            prefer_up = start > 0 and degrees[start] >= degrees[start - 1]
            neighbours = tuple(
                degrees[position]
                for position in (start - 1, stop + 1)
                if 0 <= position < count
            )
            for member in range(start, stop + 1):
                degrees[member] = _snap_to_chord(
                    degrees[member],
                    tone_count=tone_count,
                    prefer_up=prefer_up,
                    neighbours=neighbours,
                    shape=shape,
                )
            changed = True
    return [
        (slot[0], slot[1], degrees[index], slot[3])
        for index, slot in enumerate(slots)
    ]


def _walk_shape(
    variant: MotifVariant,
    *,
    bar_ticks: int,
    tone_count: int,
    closing_degree: int | None = None,
    shape: MelodyShape = DEFAULT_MELODY_SHAPE,
) -> tuple[list[int], list[int]]:
    """Walk one bar's motif: scale degrees and durations, in slot order.

    The degrees are relative to the bar's anchor chord tone, and the
    shape does not depend on which tone that is: every candidate start
    sits a whole number of chord tones above the others, so the same
    shape serves all of them, shifted.

    `closing_degree` is written over the walk's last slot before the
    leap answering runs, so the answer can step into the note the bar
    has to land on rather than turn away from it. `_close_bar` writes
    the same degree again once the start's offset is applied — the walk
    here only has to know where the bar is going.

    The walk is folded to within an octave of where it starts, its leaps
    are answered by a turn back, and the line is bent so every note off
    the chord is a step from both its neighbours. Those three passes are
    what let the bar reach the licence already legal: the generator and
    the linter have to agree note for note, and the safest way to agree
    is for the walk to be written the way the linter reads it.
    """
    degrees: list[int] = []
    durations: list[int] = []
    offset = 0
    degree = variant.anchor_offset
    while offset < bar_ticks:
        for cell in variant.motif:
            if offset >= bar_ticks:
                break
            # The step is the move *into* the note, so it is taken before
            # the note is emitted: the motif's first step is always 0 —
            # "start on the bar's anchor tone" — and taking it afterwards
            # sounded that step as a repeat of the anchor and dropped the
            # last step the motif actually drew.
            degree += cell.step
            degrees.append(degree)
            durations.append(min(cell.length_ticks, bar_ticks - offset))
            offset += cell.length_ticks
        if not variant.repeat:
            break
        # The sequence advances one chord tone per replay, so each replay
        # starts on the next tone of the chord.
        degree += shape.chord_tone_degrees
    if not degrees:
        return degrees, durations
    if closing_degree is not None:
        degrees[-1] = closing_degree
    remainders = chord_tone_degrees(tone_count)
    return (
        _licit_line(
            _answer_leaps(
                _bound_walk(degrees, degrees[0]),
                fixed_tail=1 if closing_degree is not None else 0,
                remainders=remainders,
                shape=shape,
            ),
            remainders=remainders,
            shape=shape,
        ),
        durations,
    )


def _closing_tone(
    closing_degree: int,
    offset: int,
    *,
    half_cadence: bool,
    shape: MelodyShape = DEFAULT_MELODY_SHAPE,
) -> int | None:
    """The degree a bar closes on when its walk starts `offset` away.

    The gesture is written in the bar's own frame — the walk was shaped
    around it, so the last note steps into it — and a bar restated a
    chord tone higher therefore closes one chord tone higher, with the
    approach intact. What the gesture may not do is land somewhere the
    phrase has not asked for: the half cadence rests on the root of the
    V, the final bar on the tonic or its third (`closing_degree`, whose
    chord degrees are 0 and 2). A start whose closing tone falls outside
    that has no closing degree, and the caller drops it.
    """
    degree = closing_degree + offset
    allowed = (0,) if half_cadence else (0, shape.chord_tone_degrees)
    return degree if degree % 7 in allowed else None


def _close_bar(
    slots: list[BarSlot],
    *,
    degree: int | None,
    ticks: int | None,
) -> list[BarSlot]:
    """Give a bar's last slot the phrase's closing gesture.

    A half cadence lands on the chord's root, the final bar on the tonic
    or its third, a breathing bar simply shortens what it had. The
    gesture is applied to the *degrees*, before the licence pass runs,
    so the pass approves the notes that are actually sounded — writing
    the closing pitch in afterwards is what left a licensed passing tone
    a leap away from the note it was licensed to step into.

    `degree` arrives in the bar's own frame, the one the walk was shaped
    in (see `_closing_tone`), not in the chord's: the degrees the slots
    carry are already shifted by whatever tone of the chord the bar
    starts on, and a closing degree that ignored that shift would be
    written a chord tone or two below the line it has to step out of.
    """
    if not slots or (degree is None and ticks is None):
        return slots
    offset, duration, last_degree, tie = slots[-1]
    closing = last_degree if degree is None else degree
    out = list(slots)
    if closing != last_degree and len(out) >= 2 and out[-2][3]:
        # The gesture moved the note the bar ends on, so a tie into it is
        # off: a tie holds one pitch and this one now lands elsewhere. The
        # cadence is what the phrase asked for, so the tie yields.
        head = out[-2]
        out[-2] = (head[0], head[1], head[2], 0)
    out[-1] = (offset, duration if ticks is None else ticks, closing, tie)
    return out


def _land_on_chord(
    degrees: list[int],
    durations: list[int],
    *,
    tone_count: int,
    closing_degree: int | None,
) -> tuple[list[int], list[int]]:
    """Add the step that carries a bar's last note onto a chord tone.

    A bar ends on the harmony: its last note has nothing after it to be
    a passing tone *to*, so the licence cannot cover a non-chord tone
    there. The composer's move is not to rewrite that note but to keep
    walking — a line that has stepped down to A over a C chord takes one
    more step to G — so the repair is one extra note, and the note it
    was built to serve keeps the pitch the motif gave it.

    One step always suffices: a non-chord tone is one degree from a
    chord tone in a chord scale, so the landing is the note one degree
    further along the line. A bar that already ends on a chord tone —
    including every bar whose closing gesture put it there — is left
    alone.
    """
    if not degrees or closing_degree is not None:
        return degrees, durations
    remainders = chord_tone_degrees(tone_count)
    last = degrees[-1]
    if last % 7 in remainders:
        return degrees, durations
    forward = -1 if len(degrees) < 2 or last <= degrees[-2] else 1
    landing = last + forward
    if landing % 7 not in remainders:
        landing = last - forward
    half = durations[-1] // 2
    return [*degrees, landing], [*durations[:-1], durations[-1] - half, half]


def _start_offsets(
    anchor: int, tone_count: int, *, shape: MelodyShape = DEFAULT_MELODY_SHAPE
) -> tuple[int, ...]:
    """Every chord tone a bar could be restated on, the drawn one first.

    A bar is one line on one chord, and every tone of that chord is a
    place the line can begin: restating it from the third or the fifth is
    how a composer moves a phrase into another register without rewriting
    a note of it. The offsets are the chord's own degrees — the ones
    congruent to a chord tone modulo 7 — so the restatement keeps every
    note of the line a chord tone of the bar (`_legal_slots` then has
    nothing to snap and the line survives intact), and its reach is an
    octave either way, which is as far as the tessitura band can use.

    The drawn anchor leads the tuple because the caller keeps it when
    nothing else fits better; the rest of the lattice is what lets a bar
    come in by step when its own register would have made it leap.
    """
    drawn = shape.chord_tone_degrees * anchor
    return (drawn, *(o for o in _chord_lattice(anchor, tone_count) if o != drawn))


def _apex_starts(
    anchor: int, tone_count: int, *, shape: MelodyShape = DEFAULT_MELODY_SHAPE
) -> tuple[int, ...]:
    """The higher tones a section's peak bar may be restated on.

    The apex is the one bar whose register is chosen rather than fitted,
    and a *lift* is what makes it the peak: the line goes up a tone or
    two of its own chord — a third or a fourth, the interval a phrase
    rises by — while every other bar is placed where its entrance is
    plainest. That bound is the whole point. Restating the bar an octave
    up is the same statement made by a leap the listener has to recover
    from, and it is what used to make the section's peak arrive as a
    twelve-semitone jump rather than as the top of a climb.

    The set is therefore the lattice's offsets strictly above the drawn
    anchor and strictly inside the octave: never empty (any six
    consecutive degrees hold two tones of a triad), and never a jump.
    """
    drawn = shape.chord_tone_degrees * anchor
    return tuple(o for o in _chord_lattice(anchor, tone_count) if drawn < o < drawn + 7)


def _chord_lattice(anchor: int, tone_count: int) -> tuple[int, ...]:
    """The degrees congruent to a tone of the bar's chord, within reach."""
    remainders = chord_tone_degrees(tone_count)
    reach = _START_REACH_DEGREES
    return tuple(
        offset for offset in range(-reach, reach + 1) if offset % 7 in remainders
    )


def _entrance_cost(entrance: int | None, prev_leap: int | None) -> int:
    """What it costs a bar to be entered the way a shift makes it enter.

    A step into the bar is what a melody does and it is free. A skip — a
    third or a fourth — still moves and still comes back easily, so it
    is next. A repeat leaves the line where it was, which is a note the
    bar did not need and a repeat the score counts. A leap is the one
    entrance a bar should avoid, because a leap is what the *listener*
    has to recover from.

    When what ran into this bar was itself a leap, the entrance is the
    note that answers it, so a step back the other way is the one
    entrance that is free and every other way in is a fault: a step
    carrying on the same way never answers, and a repeat is the line
    refusing to move at all, which the score reads the same way. A skip
    is no better there — a leap wants a step, and a third is not one.

    The caller ranks equal costs by the size of the entrance, so the
    grades are deliberately few: what matters is that a skip outranks a
    repeat and both outrank a leap, not that the numbers are spaced.
    """
    if entrance is None:
        return 0
    distance = abs(entrance)
    if prev_leap is not None and abs(prev_leap) >= LEAP_MIN_SEMITONES:
        if 0 < distance <= STEP_MAX_SEMITONES and (entrance > 0) != (prev_leap > 0):
            return 0
        return 3
    if distance == 0:
        return 2
    if distance <= STEP_MAX_SEMITONES:
        return 0
    return 1 if distance < LEAP_MIN_SEMITONES else 3


def _opening_step(pitches: Sequence[int], slots: Sequence[BarSlot]) -> int | None:
    """The bar's first *sounding* move, or None if it has only one note.

    A tied continuation is not struck, so the interval the ear hears
    first is the one out of the tie, not the one between the tied
    noteheads. Both the seam's answer (`_entry_answer`) and the
    placement's ranking ask whether the bar's opening move answers the
    leap it came in on, and a tied opening read as a repeat answers
    nothing — which would have the bar turn a seam it has already
    answered and leave the answer unheard.
    """
    if len(pitches) < 2:
        return None
    index = 1
    while index < len(pitches) and slots[index - 1][3]:
        index += 1
    return None if index >= len(pitches) else pitches[index] - pitches[0]


def _answered(entrance: int | None, opening: int | None) -> bool:
    """Whether a bar's opening move answers the leap it was entered on.

    The rule the score measures is the same one a listener hears: a leap
    is recovered by the step after it, and the step has to go back the
    way the leap came. `opening` is the bar's own first interval, so a
    bar entered by a step — or entered by nothing, at the top of the
    piece — has nothing to answer and is trivially fine.
    """
    if entrance is None or abs(entrance) < LEAP_MIN_SEMITONES:
        return True
    if opening is None:
        return False
    return 0 < abs(opening) <= STEP_MAX_SEMITONES and (opening > 0) != (entrance > 0)


def _entry_answer(entrance: int | None, opening: int | None) -> int | None:
    """The degree step that answers a leap into the bar, or None if none is owed.

    A leap across a bar line is a leap: the line has to come back, and
    the note that comes back is the bar's second. One degree the other
    way is a semitone or a whole tone in the opposite direction, which is
    exactly what `_answered` asks for — the same repair `_answer_leaps`
    makes inside a bar, applied to the one seam that pass cannot see.
    """
    if _answered(entrance, opening):
        return None
    assert entrance is not None
    return -1 if entrance > 0 else 1


def _place_bar(
    pitches: list[int],
    *,
    band: MelodyBand,
    prev_pitch: int | None,
    apex: bool,
    bass_pitches: tuple[int, ...] = (),
    chord_pcs: frozenset[int] = frozenset(),
    prev_leap: int | None = None,
    opening: int | None = None,
) -> tuple[tuple[float, ...], int, int, int, int]:
    """Choose the octave a bar's line sits in, and how well the bar fits it.

    The shift is applied to the whole bar, so the bar's intervals — its
    motif — come through unchanged. The only interval the choice can
    damage is the one into the bar from the previous bar's last note,
    which is why the octaves are ranked by that interval once the
    register is right (`_entrance_cost`).

    `bass_pitches` is what the bar sounds against. A melody note a
    semitone or a major seventh from a sounding bass note is the one
    collision the linter refuses, and register is the honest way to
    settle it: the note keeps its pitch class and moves away from the
    bass an octave at a time, which no rewrite of the line can do. So a
    shift that clears those is preferred to one that does not, after the
    tessitura and before the approach — the band is not negotiable, the
    collision is.

    `chord_pcs` are the pitch classes of the bar's own chord, and they
    are what keeps that count honest: the linter exempts a note that is a
    chord tone of its bar, because two tones of the bar's chord are a
    voicing and not a clash — a seventh chord may sound its own seventh
    against its root. Only the notes off the chord are counted, and those
    are the ones the licence pass would snap away from the bass anyway.

    `opening` is the bar's first *sounding* move, which is the caller's
    to know: a tied continuation is not struck, so the move the ear hears
    is the one out of the tie. It is the interval the seam is judged on
    and the octave cannot change it, so it is passed in rather than read
    off the first two notes, which a tie leaves at one pitch.

    An `apex` bar is the section's peak, and its height is already in
    its start (`_apex_starts` lifts the line by a tone or two of its own
    chord), so what is left for the octave here is only the band. What
    it weighs, after the band and the collision: a bounded entrance,
    then the smallest displacement that fits, then whether the entrance
    is answered, then the height, then the entrance's grade. Weighing
    the entrance above the displacement is the one key that differs from
    every other bar's order, and it is a correction rather than a
    refinement: it used to sit below the displacement, which made the
    apex the one bar per section that would rather speak from the
    register it was written in than enter quietly from an octave away. A
    36-piece sweep of the corpus found 19 intervals wider than an octave
    before the key moved and 8 after, and the body's own wide placements
    did not fall — they rose — so every one the change removed was the
    apex's. The interval it may not be bought with is still the one
    wider than any answer can cover, and the band and a rub still
    outrank the entrance both: the tessitura is a fact about the
    instrument and a collision is a linter refusal.

    Whether the entrance *is* answered is weighed before the height,
    though, and that is not the same key as the entrance's grade: a
    whole bar's line is the choice here, so a variant that comes back
    from the leap into it at the price of a semitone or two of the
    bar's top has bought the one thing the apex owes the phrase — the
    peak is a peak because it is arrived at and left, not because it
    is the highest note in a line that stalled on it.

    Every other bar weighs its entrance first: a step is free, a repeat
    is a note the bar did not need, a leap is what the listener has to
    recover from — and a leap the bar's own opening step answers is
    better than one it does not.

    Returns the ranking the placement earned, in the order the keys were
    weighed, then the shift itself, the number of notes it leaves outside
    the band — zero for every bar the walk wrote inside it, and
    occasionally one or two for a line too wide to sit in the band at any
    octave — the number of notes it still leaves rubbing the bass, and
    what its entrance costs. The ranking is handed back so the caller
    that chooses between *starts* ranks them on the same scale the
    octaves were ranked on, rather than on a second one of its own.
    """
    best: tuple[tuple[float, ...], int, int, int, int] | None = None
    for octave in range(-3, 4):
        shift = 12 * octave
        shifted = [pitch + shift for pitch in pitches]
        outside = sum(
            1 for pitch in shifted if not band.contains(pitch)
        )
        rubbing = sum(
            1
            for pitch in shifted
            if pitch % 12 not in chord_pcs
            for bass in bass_pitches
            if abs(pitch - bass) in DISSONANT_INTERVALS
        )
        step_into_bar = None if prev_pitch is None else shifted[0] - prev_pitch
        entrance = _entrance_cost(step_into_bar, prev_leap)
        # Among entrances of the same grade the smaller one wins: the
        # octave grid can leave a bar with nothing but leaps to choose
        # between, and a bar entered a fifth away is a bar entered well
        # next to one entered a tenth away.
        gap = 0 if step_into_bar is None else abs(step_into_bar)
        # An entrance wider than an octave is a fault whatever else is on
        # offer: the line is lost before the step that recovers it can
        # arrive. So it is weighed before the entrance's grade — a repeat
        # the score counts is the smaller price — and, for the apex,
        # before the height, which may not be bought at that price.
        within = 0 if gap <= _MAX_ENTRANCE_SEMITONES else 1
        answered = 0 if _answered(step_into_bar, opening) else 1
        if apex:
            rank: tuple[float, ...] = (
                outside,
                rubbing,
                within,
                abs(octave),
                answered,
                -max(shifted),
                entrance,
                gap,
            )
        else:
            rank = (
                outside,
                rubbing,
                within,
                entrance,
                answered,
                gap,
                abs(shifted[0] - band.centre_midi),
            )
        if best is None or rank < best[0]:
            best = (rank, shift, outside, rubbing, entrance)
    assert best is not None
    return best[0], best[1], best[2], best[3], best[4]


def _pickup_pitch(
    *,
    band: MelodyBand,
    root: int,
    tones: tuple[int, ...],
    nearby: int | None,
    bass_pitches: tuple[int, ...],
) -> int | None:
    """The anacrusis pitch leading into the next bar, or None if none fits.

    A pickup is a chord tone of the bar it leads into, placed in the
    tessitura band near the note it follows — the register it must
    approach from, not an octave above it. Candidates that would sound a
    close m2/M7 against the sounding bass are skipped; when every
    candidate clashes the pickup is dropped rather than played against a
    clash.

    Among the playable ones the pickup is a *step* from the note it
    follows — never that note itself, and never a leap. A pickup that
    cannot step is not a pickup: the figure exists to leave the melody
    before the downbeat, and a chord tone a third or more from the note
    it follows would be a leap into the bar line with nothing after it
    to answer it, which is worse than no anacrusis at all.
    """
    target = nearby if nearby is not None else band.centre_midi
    candidates = sorted(
        (root + tone + 12 * octave for tone in tones for octave in range(-3, 4)),
        key=lambda pitch: abs(pitch - target),
    )
    steps = [
        candidate
        for candidate in candidates
        if band.contains(candidate)
        and 0 < abs(candidate - target) <= STEP_MAX_SEMITONES
        and all(abs(candidate - bass) not in DISSONANT_INTERVALS for bass in bass_pitches)
    ]
    return steps[0] if steps else None


def _final_closing_degree(
    rng: random.Random, *, shape: MelodyShape = DEFAULT_MELODY_SHAPE
) -> int:
    """The degree the piece's last bar lands on: the tonic, or its third.

    Home twice as often as its third, because a resolution onto the third
    is a colour and one onto the tonic is an ending. Which third is not a
    constant: the chord tone away from the tonic is the plan's, so a plan
    that states a different chord spelling states its own close.

    Named rather than written inline so the read is a thing a test can
    hold. It is the only reader of `chord_tone_degrees` that is not a
    helper taking a shape, and an inline expression sharing its field with
    six other sites cannot be shown to read the plan at all — reverting
    this one to the constant leaves every test green. The probability
    itself stays a literal: it is a musical decision, but not one the
    quality thresholds name, which is the rule the plan's scope follows.
    """
    return 0 if rng.random() < 0.6 else shape.chord_tone_degrees


def _melody_bar(
    *,
    band: MelodyBand,
    variant: MotifVariant,
    chord_root: int,
    chord_tones: tuple[int, ...],
    scale: tuple[int, ...],
    anchor: int,
    prev_pitch: int | None,
    start_tick: int,
    bar_ticks: int,
    rng: random.Random,
    position: float,
    ticks_per_bar: int,
    seed_for_variation: int,
    shape: MelodyShape = DEFAULT_MELODY_SHAPE,
    is_final_bar: bool = False,
    half_cadence: bool = False,
    apex: bool = False,
    breathe: bool = False,
    pickup: tuple[int, tuple[int, ...]] | None = None,
    bass_pitches: tuple[int, ...] = (),
    prev_leap: int | None = None,
) -> list[NoteEvent]:
    """Render one bar of melody from a motif variant.

    The motif is walked in *scale degrees* of `scale` — the bar's own
    chord scale, spelled from `chord_root` — so a step is a semitone or a
    whole tone and the same shape lands correctly on every chord of the
    template. When `repeat` is set (the sequence operation) the motif
    keeps replaying from the top, the walk advancing one chord tone per
    cycle, until the bar is full. The bar's slots are then re-voiced
    through the melody's rhythm library (`motif.apply_rhythm`): dotted
    figures, 16th subdivisions, ties, which move durations and never
    pitches.

    Which of the chord's tones the bar starts on is then chosen: the
    drawn `anchor` when it leaves the bar sounding, and another tone of
    the same chord when it would force a leap into the bar from the
    previous one. Each candidate is walked, snapped to the passing-tone
    licence (`_legal_slots`) and placed in the tessitura, so the choice
    is made on a bar that is already legal and in register.

    Phrase shape:
    - an `apex` bar is placed at the top of the tessitura, with a
      velocity lift — the section's melodic peak;
    - a `half_cadence` bar ends early on the chord's root, leaving a
      rest (the phrase breathes on the V);
    - a breathing bar shortens its last note into a rest;
    - the `is_final_bar` of the piece resolves onto the tonic or its
      third, held to the bar line;
    - when the bar leaves at least an eighth of space at its end and
      `pickup` is given (the next bar's root and tones), an anacrusis
      pickup note sounds on the last eighth, leading into the next bar.

    `prev_leap` is the interval the previous bar ended on, when it was a
    leap: this bar's entrance is the note that answers it, so the choice
    of octave is told which way the answer has to go.
    """
    # The closing gesture's degree is decided before the walk is built:
    # the walk has to know which note the bar lands on so its leap
    # answering can approach that note by step. Its durations come later,
    # because the rhythm library re-voices the bar's slots first.
    closing_degree: int | None = None
    if is_final_bar:
        closing_degree = _final_closing_degree(rng, shape=shape)
    elif half_cadence:
        closing_degree = 0

    # Degrees first: a slot's pitch depends on the bar's octave, which
    # depends on the whole bar. So the walk is settled in degree space
    # before any pitch is spelled — walked, answered, and landed on a
    # chord tone, which is what most of the passing-tone licence needs
    # and what the rhythm library then dresses.
    degrees, durations = _walk_shape(
        variant,
        bar_ticks=bar_ticks,
        tone_count=len(chord_tones),
        closing_degree=closing_degree,
        shape=shape,
    )
    degrees, durations = _land_on_chord(
        degrees,
        durations,
        tone_count=len(chord_tones),
        closing_degree=closing_degree,
    )
    slots = [
        (sum(durations[:index]), duration, degrees[index])
        for index, duration in enumerate(durations)
    ]
    rhythm_slots: list[BarSlot]
    if is_final_bar:
        # The closing bar keeps the motif's own rhythm: the resolution
        # is the one event that should not be dressed up.
        rhythm_slots = [(o, d, t, False) for o, d, t in slots]
    else:
        rhythm_slots = apply_rhythm(
            slots,
            rng=rng,
            weights=shape.rhythm_weights,
            remainders=chord_tone_degrees(len(chord_tones)),
        )

    # The gesture's durations are the last word on the bar, so they are
    # taken after the rhythm library has re-voiced it.
    closing_ticks: int | None = None
    if rhythm_slots:
        last_offset, last_duration, _last_degree, _last_tie = rhythm_slots[-1]
        if is_final_bar:
            # Held to the bar line.
            closing_ticks = bar_ticks - last_offset
        elif half_cadence or breathe:
            # Lifted early, so a rest follows.
            closing_ticks = last_duration // 2

    # An apex bar is placed by height, not by its approach: it is the
    # section's peak, and the top of the band is worth a wide interval
    # into it. Every other bar tries each of the chord's tones as its
    # start, cheapest approach winning, and keeps the drawn one when
    # none of them makes the approach cheaper.
    #
    # The bar is placed before it is repaired. Register is the cheap fix
    # for a rub against the sounding bass — a note a semitone from the
    # bass a tenth below is the same note a semitone from it an octave
    # up — and moving the whole bar never touches the line. Only a bar
    # that rubs at every octave the tessitura allows is handed to the
    # licence pass with the bass's pitch classes to steer around, and
    # that pass rewrites the line, so a bar reaches it only when nothing
    # else can be done.
    lattice = _start_offsets(anchor, len(chord_tones), shape=shape)
    # The drawn start is the lattice's first offset, and it is kept as the
    # fallback below: a bar whose every start is barred by the closing
    # gesture is restated where it was drawn. Read from the lattice rather
    # than recomputed, so the plan's chord tone is read in one place.
    drawn = lattice[0]
    starts = (
        _apex_starts(anchor, len(chord_tones), shape=shape) if apex else lattice
    )
    if closing_degree is not None:
        # The closing gesture is a chord tone of the bar, and the bar it
        # closes is one of the chord's tones tall. Which *one* is not
        # free: the half cadence rests on the root of the V and the
        # final bar on the tonic or its third, so a start whose closing
        # tone lands elsewhere is not a candidate. This is also what
        # keeps the landing a step away: the walk was shaped around the
        # closing degree, so shifting the bar by the start's own offset
        # moves the two together and the approach survives.
        starts = tuple(
            offset
            for offset in starts
            if _closing_tone(
                closing_degree, offset, half_cadence=half_cadence, shape=shape
            )
            is not None
        ) or (drawn,)
    bass_pcs = frozenset(pitch % 12 for pitch in bass_pitches)
    chord_pcs = frozenset((chord_root + tone) % 12 for tone in chord_tones)
    remainders = chord_tone_degrees(len(chord_tones))

    def build(
        start: int,
        entry: int | None,
        avoid_pcs: frozenset[int],
    ) -> tuple[tuple[float, ...], int, list[BarSlot], int | None, int | None]:
        """One start's bar: its line, its register, and how well it fits.

        `entry` forces the bar's opening two moves (see `_entry_answer`) —
        the turn that answers a leap into the bar — and is applied before
        the closing gesture and the licence pass, so what is placed and
        ranked is the line the bar will sound. A tie the rhythm library
        drew is held across those rewrites (`_hold_tied_pairs`), so a
        turn written onto a tied note comes out as the move *out of* the
        tie, which is the only move the bar has there. The last two
        elements are the interval the bar is entered on and the interval
        it opens with, which together say whether the entrance was
        answered.
        """
        closing = (
            None
            if closing_degree is None
            else _closing_tone(
                closing_degree, start, half_cadence=half_cadence, shape=shape
            )
        )
        degrees = [
            degree + start
            for _offset, _duration, degree, _tie in rhythm_slots
        ]
        if entry is not None and len(degrees) > 2:
            if rhythm_slots[0][3]:
                # The bar opens on a tie: its first notehead sounds
                # through the second, so the seam hears one move where two
                # are written, and the turn goes on the move out of the
                # tie. Writing it twice would put the turn's second step
                # on a note the ear never hears struck and leave a third
                # standing at the seam with nothing to answer it.
                degrees[2] = degrees[0] + entry
            else:
                degrees[1] = degrees[0] + entry
                degrees[2] = degrees[1] + entry
        closed = _close_bar(
            [
                (offset, duration, degrees[index], tie)
                for index, (offset, duration, _degree, tie) in enumerate(rhythm_slots)
            ],
            degree=closing,
            ticks=closing_ticks,
        )
        # The closing gesture is written onto the last slot after the
        # line has been shaped, which can leave the note before it a
        # third away rather than a step. Bending the line toward the
        # landing is the same repair the walk took, and it is the only
        # one the bar's last note is allowed to need.
        bent = _licit_line(
            [slot[2] for slot in closed],
            remainders=remainders,
            fixed_tail=1 if closing_degree is not None else 0,
            shape=shape,
        )
        candidate = _legal_slots(
            _hold_tied_pairs(
                [
                    (offset, duration, bent[index], tie)
                    for index, (offset, duration, _degree, tie) in enumerate(closed)
                ]
            ),
            tone_count=len(chord_tones),
            chord_root=chord_root,
            scale=scale,
            avoid_pcs=avoid_pcs,
            shape=shape,
        )
        pitches = [
            scale_walk(degree, chord_root, scale) for _o, _d, degree, _t in candidate
        ]
        opening = _opening_step(pitches, candidate)
        rank, shift, _outside, _rubbing, _entrance = _place_bar(
            pitches,
            band=band,
            prev_pitch=prev_pitch,
            apex=apex,
            bass_pitches=bass_pitches,
            chord_pcs=chord_pcs,
            prev_leap=prev_leap,
            opening=opening,
        )
        approach = None if prev_pitch is None else pitches[0] + shift - prev_pitch
        # Everything the seam is judged on is already in the placement's
        # ranking; the only key left is which tone of the chord the bar
        # was drawn on, and it is last because a restatement is a device,
        # not a preference — it is kept when nothing about the bar's fit
        # makes it worse.
        return (
            (*rank, 0 if start == drawn else 1),
            shift,
            candidate,
            approach,
            opening,
        )

    def choose(
        avoid_pcs: frozenset[int],
    ) -> tuple[tuple[float, ...], int, list[BarSlot], int | None, int | None]:
        best: tuple[tuple[float, ...], int, list[BarSlot], int | None, int | None] | None = None
        for start in starts:
            here = build(start, None, avoid_pcs)
            entry = _entry_answer(here[3], here[4])
            if entry is not None:
                # The bar came in on a leap it does not answer, and the
                # step that would answer it is a cheap, local repair —
                # so the bar is built twice and the better of the two
                # kept, which leaves the placement free to prefer the
                # natural line when that is the better one.
                turned = build(start, entry, avoid_pcs)
                if turned[0] < here[0]:
                    here = turned
            if best is None or here[0] < best[0]:
                best = here
        assert best is not None
        return best

    chosen = choose(frozenset())
    if chosen[0][_RANK_RUBBING] and bass_pcs:
        # A note rubbing the bass is the fault no voicing can undo: the
        # rank weighs the line's own shape ahead of it, and the line was
        # shaped without knowing what the bass plays under it. So when the
        # winner rubs, the same bar is built once more with the licence
        # pass steered around the bass's pitch classes, and the better of
        # the two is kept. A bar that needed no steering comes back as the
        # same line, so this pass reaches only the bars that had no other
        # way out.
        steered = choose(bass_pcs)
        if steered[0] < chosen[0]:
            chosen = steered
    shift, rhythm_slots = chosen[1], chosen[2]

    notes: list[NoteEvent] = []
    for bar_offset, duration, degree, tie in rhythm_slots:
        tick = start_tick + bar_offset
        notes.append(
            NoteEvent(
                voice_id=VOICE_MELODY,
                # The bar's register is the placement's shift, applied to
                # the whole bar and to nothing else. Folding an
                # out-of-band *note* an octave instead would move it
                # against the line it belongs to — a note at the band's
                # floor lifted an octave is a twelve-semitone tear in the
                # middle of a phrase the walk wrote as a step — and no
                # later pass repairs an interval that no longer matches
                # the line the licence was checked against. A bar the
                # placement could not fit therefore sits at the band's
                # edge, which is where `_place_bar` already ranks it: the
                # count of notes left outside is the first key of the
                # ranking, so the octave that leaves the fewest is the
                # octave that wins.
                pitch_midi=scale_walk(degree, chord_root, scale) + shift,
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
    # on the next chord leads into the next downbeat.
    if (
        pickup is not None
        and not is_final_bar
        and notes
        and notes[-1].tick + notes[-1].duration_ticks <= start_tick + bar_ticks - PPQ // 2
    ):
        pickup_pitch = _pickup_pitch(
            band=band,
            root=pickup[0],
            tones=pickup[1],
            nearby=notes[-1].pitch_midi,
            bass_pitches=bass_pitches,
        )
        if pickup_pitch is not None:
            pickup_tick = start_tick + bar_ticks - PPQ // 2
            notes.append(
                NoteEvent(
                    voice_id=VOICE_MELODY,
                    pitch_midi=pickup_pitch,
                    tick=pickup_tick,
                    duration_ticks=PPQ // 2,
                    velocity=max(
                        1,
                        _shaped_velocity(
                            base=DEFAULT_VELOCITY + 8,
                            position=position,
                            tick=pickup_tick,
                            ticks_per_bar=ticks_per_bar,
                            rng_seed=seed_for_variation + start_tick,
                        )
                        - 8,
                    ),
                )
            )
    return notes


def _generate_percussion(
    *,
    kit: DrumKit = DEFAULT_DRUM_KIT,
    time_signature: str,
    form_bars: int,
    repetition_count: int,
    total_bars: int,
    seed: int,
    rest_bars: frozenset[int] = frozenset(),
    arc: SectionArc = DEFAULT_SECTION_ARC,
) -> list[NoteEvent]:
    """Generate the percussion voice for a drum-set piece.

    The style is the plan's (`percussion.style_name_for` resolved it from
    the mood and the meter, and `CompositionPlan.drum_kit` looked it up);
    its variants rotate across sections on a longer cycle than plain A/B,
    with the coda treated as one more section. A section's last bar hands
    off to the next through the style's fill (never on the piece's final
    bar, which must resolve), and every section downbeat is marked with a
    crash cymbal — plus a kick when the pattern does not already open with
    one. A per-bar seeded jitter of a few velocity points keeps repeated
    bars from sounding machine-stamped. Bars in `rest_bars` (the intro and
    one mid-piece section on long pieces) are silent, and the sections
    that do play follow the terraced dynamic arc.
    """
    style = kit.style
    if style is None:
        return []
    ticks_per_bar = bar_ticks(time_signature)
    mood_scale = kit.velocity_scale
    notes: list[NoteEvent] = []
    for bar in range(total_bars):
        if bar in rest_bars:
            continue
        in_body = bar < repetition_count * form_bars
        section_idx = bar // form_bars if in_body else repetition_count
        section_start = bar % form_bars == 0
        is_final_bar = bar == total_bars - 1
        terrace = _section_velocity_scale(section_idx, repetition_count, arc)
        if bar % form_bars == form_bars - 1 and not is_final_bar:
            pattern = style.fill(time_signature, section_idx)
        else:
            pattern = None
        if pattern is None:
            pattern = style.pattern(
                time_signature,
                rotation_index(
                    section_idx,
                    len(style.variants.get(time_signature, ())),
                    cycle=kit.rotation_cycle,
                ),
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
                kit.crash_velocity * style.velocity_scale * mood_scale * terrace
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

# The realization is drawn from two streams, one per group of voices: the
# melodic group (the tune and the bed under it) draws from one and the kit
# from the other. One stream could not do, and the measurement is the
# reason. The draws are taken in event order, so a voice that writes one
# more note shifts every draw the other voice makes after it:
# `bass_root_motion` shortened this piece's tune by one note and moved the
# kit's realized timing in **all 207** of its hits — its fills with it. A
# coupling between two voices neither of which reads the other is one no
# plan knob can express and no critic can attribute, so each voice's
# realized timing is a function of its own notes and the piece's seed and
# of nothing that happens in the other stream.
#
# Only that arrow was observable — the melodic events are built before the
# kit's, so the kit's length could never shift the tune's draws — but which
# order `events` is built in is an accident of this function rather than a
# property of the design, and a shared stream is a coupling waiting to point
# the other way.
_MELODIC_REALIZATION_SALT: Final[int] = 11
_KIT_REALIZATION_SALT: Final[int] = 12


def _realization_stream(seed: int | None, salt: int) -> random.Random:
    """The stream the voices of one group draw their realization from.

    The salt separates the groups; `None` seeds the piece's own default,
    which is what a caller that names no seed gets.
    """
    return random.Random(((seed or 0) * 2654435761 + salt) % (2**31))


# CC11 (expression) rides the dynamic arch so phrases swell and relax
# even inside a held chord. 96 is near-full expression at the arch peak.
EXPRESSION_BASE: int = 96

# Arrangement arc (S8): long pieces lift their final repetition a whole
# step (the piece ends in the new key — the lift IS the ending), drop
# the drums for one mid-piece section to give the texture a hole, and
# step the dynamics per section instead of arching continuously. All
# three decisions now arrive through the plan: the offset and the terraces
# live in `duration.py` with the arrangement they belong to, and the rest
# section in `percussion.py`, which is the module that owns the kit.


def _section_velocity_scale(
    section_idx: int,
    repetition_count: int,
    arc: SectionArc = DEFAULT_SECTION_ARC,
) -> float:
    """Terraced dynamics: the arc is stepped per section, not continuous.

    The opening sits back, the penultimate section peaks, and the
    final one settles slightly for the cadence home. A single-section
    piece has nowhere to move and plays at full. The four terraces are
    the plan's, so a critic can move a section's weight without moving
    the piece's overall level.
    """
    if repetition_count < 2:
        return 1.0
    if section_idx == 0:
        return arc.energy_opening
    if section_idx == repetition_count - 1:
        return arc.energy_final
    if section_idx == repetition_count - 2:
        return arc.energy_peak
    return arc.energy_middle


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
    arc: SectionArc = DEFAULT_SECTION_ARC,
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
    harmony_instruments = {
        voice: instrument
        for voice, instrument in voice_instruments.items()
        if voice >= VOICE_HARMONY
    }

    # Legato: sustained instruments let each note of a line ring a
    # little past the next attack so the release tail blurs into the
    # next note. A same-pitch neighbour is capped at its start: a late
    # note_off on the same key would re-attack or cut the line.
    legato_extended: set[int] = set()
    legato_voices = [
        voice
        for voice, instrument in (
            (VOICE_MELODY, melody_instrument),
            *harmony_instruments.items(),
        )
        if instrument in SUSTAINED_INSTRUMENTS
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
        swell_voices.extend(
            voice
            for voice, instrument in harmony_instruments.items()
            if instrument in SUSTAINED_INSTRUMENTS
        )
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
                    * _section_velocity_scale(
                        section_idx, arrangement.repetition_count, arc
                    )
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
                *harmony_instruments.items(),
            )
            if instrument in PEDAL_INSTRUMENTS
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
        rng = _realization_stream(seed, _MELODIC_REALIZATION_SALT)
        kit_rng = _realization_stream(seed, _KIT_REALIZATION_SALT)
        timing_us = HUMANIZE_TIMING_US.get(humanization, HUMANIZE_TIMING_US["light"])
        velocity_span = HUMANIZE_VELOCITY_SPAN.get(
            humanization, HUMANIZE_VELOCITY_SPAN["light"]
        )
        perc_timing_us = PERCUSSION_TIMING_US.get(humanization, PERCUSSION_TIMING_US["light"])
        humanized: list[PerformanceNoteEvent] = []
        for event in events:
            if event.voice_id == VOICE_MELODY or event.voice_id in harmony_instruments:
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
                offset = round(kit_rng.uniform(-1.0, 1.0) * perc_timing_us)
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
        # notes, skipped when a real hit already occupies the slot. Both
        # of the kit's draws come from the kit's stream, so nothing the
        # tune or the bed writes can move them.
        percussion = [e for e in events if e.voice_id == VOICE_PERCUSSION]
        if percussion:
            sixteenth_us = round(60_000_000 / score.tempo.bpm / 4)
            ghosts: list[PerformanceNoteEvent] = []
            for event in percussion:
                if kit_rng.random() >= GHOST_NOTE_PROBABILITY:
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
                        velocity=kit_rng.randint(*GHOST_NOTE_VELOCITY_RANGE),
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
