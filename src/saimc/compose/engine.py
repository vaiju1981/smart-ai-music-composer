"""Phase 1 composition engine — the score, assembled from its voices.

Takes a CompositionSpec and returns (NotationScore, PerformancePlan).
Rule-based and deterministic: `(plan, seed) -> notes` is byte-identical,
which `tests/acceptance/test_golden_corpus.py` pins against recorded
hashes. Chord tones and key handling come from pure interval tables in
`saimc.compose.forms` — no music21 dependency at generation time.

Stages (per `docs/roadmap.md` §2 step 3):

1. Resolve key + time signature from the spec.
2. Pick a mood-appropriate chord template (form).
3. Apply the §10 #1 duration policy to find the form, repetition
   count, and tempo that fit the spec's `duration_seconds` within
   ±2% tolerance.
4. Write each voice over the chord progression. Repeated sections rotate
   through the mood's chord-template variants (A/B form) and the coda
   closes on the tonic.
5. Theory-lint the resulting NotationScore.
6. Build the PerformancePlan with integer-microsecond timestamps.

**What is here and what is not.** This module owns step 4's *arrangement* —
which voices exist, which bars they play, how the sections lay out — and
delegates each voice's own writing to the module that owns it:

- `melody` — the line: the walk, the leap and the step that answers it.
- `harmony` — the bed: how many voices sound, and where they sit under
  the tune.
- `bass` — the left hand's register and the figure it states.
- `percussion` — the kit's part, beside the style tables it plays from.
- `dynamics` — velocity as a reading of position, shared by all four.
- `performance` — step 6, the expression layer over the notated surface.

The dependency runs one way, downward, and `dynamics` is at the bottom:
nothing in that list imports this module back. Splitting it this way moved
no logic and changed no byte of output — the golden hashes are the proof,
and they did not move.

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
from typing import Any, Final

from saimc.compose.bass import BassRegister, bass_figure_pitches, bass_register
from saimc.compose.duration import (
    ArrangementKnobs,
    DurationArrangement,
    DurationUnfulfillableError,
    SectionArc,
    arrange_for_duration,
    bar_ticks,
    section_seed,
)
from saimc.compose.dynamics import section_velocity_scale, shaped_velocity
from saimc.compose.ensemble import Ensemble, resolve_ensemble
from saimc.compose.forms import (
    PHRASE_BARS,
    ChordSlot,
    ChordTemplate,
    apply_final_cadence,
    apply_harmonic_rhythm,
    apply_section_close,
    bar_scale_intervals,
    chord_intervals,
    chord_root_offset,
    chosen_key,
    get_template_for_form,
    key_root_midi,
    transposed_key,
)
from saimc.compose.harmony import (
    active_harmony_voices,
    generate_harmony_section,
    melody_band_for,
    settle_harmony_register,
)
from saimc.compose.linter import LintIssue, lint
from saimc.compose.melody import downbeat_anchor, melody_bar
from saimc.compose.motif import (
    DEFAULT_MELODY_SHAPE,
    PLAIN_BASS_FIGURE,
    BassFigure,
    MelodyShape,
    MotifVariant,
    draw_bass_figures,
    generate_motif,
    vary_motif,
)
from saimc.compose.percussion import bass_onsets_by_bar, generate_percussion
from saimc.compose.performance import build_performance_plan
from saimc.compose.plan import CompositionPlan, resolve_plan
from saimc.compose.score import (
    PPQ,
    VOICE_BASS,
    VOICE_HARMONY,
    VOICE_MELODY,
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
)
from saimc.compose.voices import (
    DEFAULT_HARMONY_VOICES,
    HarmonyVoices,
)
from saimc.instruments import (
    MelodyBand,
    bed_registers,
    bed_window,
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
            controllers=tuple(ControllerEvent(**c) for c in plan_payload.get("controllers", ())),
            pitch_bends=tuple(PitchBendEvent(**b) for b in plan_payload.get("pitch_bends", ())),
        )
        arrangement = DurationArrangement(
            form_bars=arrangement_payload["form_bars"],
            template=ChordTemplate(
                name=arrangement_payload["template"]["name"],
                bars=arrangement_payload["template"]["bars"],
                chords=tuple(
                    ChordSlot(*chord) for chord in arrangement_payload["template"]["chords"]
                ),
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


def compose(spec: CompositionSpec, *, plan: CompositionPlan | None = None) -> EngineOutput:
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
    performance = build_performance_plan(
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


def _bass_bar(
    figure: tuple[tuple[int, int, int], ...] | None,
    *,
    anchor: int,
    chord_root: int,
    chord_tones: tuple[int, ...],
    ceiling: int,
    bar_tick: int,
    bar_pos: float,
    ticks_per_bar: int,
    rng_seed: int,
) -> list[NoteEvent]:
    """One bar of the left hand: the slot's figure, sounded.

    Rung 0 states the walk's landing tone on the bar line, and the rungs above
    it decorate the chord — which is what makes the left hand an accompaniment
    rather than a new idea every bar.

    `figure` of `None` means the piece's last bar. A close is *stated*, not
    decorated, so that bar plays the plain figure whatever its slot drew, and
    the cadence's pinned degree stays where the final bar's harmony already is.
    """
    if figure is None:
        figure = bass_figure_pitches(
            PLAIN_BASS_FIGURE,
            anchor=anchor,
            chord_root=chord_root,
            chord_tones=chord_tones,
            ceiling=ceiling,
        )
    written = []
    for figure_index, (offset16, length16, pitch) in enumerate(figure):
        onset = offset16 * ticks_per_bar // 16
        written.append(
            NoteEvent(
                voice_id=VOICE_BASS,
                pitch_midi=pitch,
                tick=bar_tick + onset,
                duration_ticks=length16 * ticks_per_bar // 16,
                velocity=shaped_velocity(
                    # The bar's first note carries the weight; the ones after
                    # it are answered, not stated.
                    base=56 if figure_index == 0 else 50,
                    position=bar_pos,
                    tick=bar_tick + onset,
                    ticks_per_bar=ticks_per_bar,
                    rng_seed=rng_seed,
                ),
            )
        )
    return written


def _slot_offset(slot_index: int, slot_count: int, key_offset: int) -> int:
    """The transposition one slot takes, exempting the final cadence.

    `key_offset` lifts a long piece's final repetition into the new key. The
    last two slots are the cadence and stay where they are: the piece lifts,
    and then comes home for its close.
    """
    return 0 if slot_index >= slot_count - 2 else key_offset


def _resolve_chords(
    template: ChordTemplate,
    *,
    key: KeySignature,
    key_offset: int,
    tonic_midi: int,
) -> tuple[list[tuple[int, tuple[int, ...], int]], list[tuple[int, ...]], list[KeySignature]]:
    """Every slot's chord, up front, plus the per-bar tables the linter reads.

    Resolved ahead of the writing because a bar can pick up *into* the next
    chord's register — an anacrusis needs to know what it leads to — and that
    is not knowable from a loop that resolves as it goes.

    The two tables travel together on purpose: every bar of a slot sounds the
    same pitch classes, and the linter checks melody and bass against that set
    while reading its passing-tone licence against the key those classes belong
    to. A bar whose classes and key came from different places is a bar the
    linter can judge against a key it is not in.
    """
    chords: list[tuple[int, tuple[int, ...], int]] = []
    bar_pcs: list[tuple[int, ...]] = []
    bar_keys: list[KeySignature] = []
    for slot_index, slot in enumerate(template.chords):
        slot_offset = _slot_offset(slot_index, len(template.chords), key_offset)
        chord_root = (
            tonic_midi + slot_offset + chord_root_offset(slot.degree, key, borrowed=slot.borrowed)
        )
        chord_tones = _chord_intervals(
            slot.degree, key, seventh=slot.seventh, borrowed=slot.borrowed
        )
        chords.append((chord_root, chord_tones, slot.bars))
        bar_pcs.extend(
            tuple(sorted({(chord_root + tone) % 12 for tone in chord_tones}))
            for _ in range(slot.bars)
        )
        bar_keys.extend([transposed_key(key, slot_offset)] * slot.bars)
    return chords, bar_pcs, bar_keys


def _bass_landing(
    slot: ChordSlot,
    *,
    key: KeySignature,
    tonic_midi: int,
    key_offset: int,
    chord_root: int,
    chord_tones: tuple[int, ...],
    bass: BassRegister,
    root_motion: bool,
    prev_bass: int | None,
) -> int:
    """The tone the bar's walk lands on — the figure's rung 0 and the walk's step.

    The pinned degree wins where a cadence pins one; otherwise the walk takes
    the chord tone nearest where it already was, which is what makes the bass
    move stepwise through inversions instead of jumping root to root.

    **A pin is a degree, not an octave.** The pinned tone keeps its pitch class
    and takes the register nearest the line it joins, exactly as every other
    landing tone is drawn from the register nearest the walk. With the octave
    fixed as well, the cadence became the one place in the piece the bass
    leaps, because the walk had no say in where the pin landed. The register
    filter matters for the same reason: an octave above the walk's ceiling
    leaves the figure every rung clamped below its own anchor, and the bar
    loses its bass.

    Without a pin, `root_motion` decides what the landing may choose from. The
    root alone states a chord change where it happens; every chord tone lets
    the walk join the last bar by the smallest step and land on an inversion —
    or on a tone the two chords share, and then the bass does not move at all.
    Two octaves of candidates keep the walk inside the bass register even in
    sharp minor keys whose chord roots sit above the middle of the keyboard.
    """
    if slot.bass_degree is not None:
        # A pinned degree belongs to the chord's own table, so a borrowed
        # chord would pin the other mode's degree.
        # `test_a_pinned_bass_degree_is_never_a_borrowed_chord` sweeps the
        # tables and holds that no pin is ever borrowed — the argument is
        # inert today and correct if that changes.
        pinned = _octave_down(
            tonic_midi
            + key_offset
            + chord_root_offset(slot.bass_degree, key, borrowed=slot.borrowed),
            octaves=1,
        )
        if prev_bass is None:
            return pinned
        in_register = [
            p
            for p in (pinned - 12, pinned, pinned + 12)
            if bass.window.low_midi <= p <= bass.window.high_midi
        ]
        last_bass = prev_bass
        return min(in_register or [pinned], key=lambda p: (abs(p - last_bass), p))

    tones = (0,) if root_motion else chord_tones
    candidates = [
        _octave_down(chord_root + tone, octaves=octaves) for tone in tones for octaves in (1, 2)
    ]
    candidates = [c for c in candidates if bass.window.low_midi <= c <= bass.window.high_midi]
    if not candidates:
        return _octave_down(chord_root, octaves=2)
    if prev_bass is None:
        return candidates[0]
    last_bass = prev_bass
    return min(candidates, key=lambda c: abs(c - last_bass))


def _settle_bed(
    notes: list[NoteEvent],
    *,
    harmony_voices: tuple[tuple[int, str], ...],
    clearance: int,
) -> list[NoteEvent]:
    """Drop the pad clear of the tune, once the tune exists.

    The bed's register is a property of the *finished* piece, not of the
    section that wrote it: the melody's floor and ceiling are one number each
    for the whole piece, so the pad is settled against them once, here. It has
    to wait until every section exists — the tune's band is not known until the
    tune is — which is the same reason this is a post-pass at all rather than a
    bound inside `generate_harmony_section`.

    The sort is not incidental. The pass moves notes after the sections sorted
    them, and a moved note can land under another note of its own voice at the
    same tick (a pad's fifth folded below the root that stayed). Canonical
    order is what the engraving and the plan read the score back by, so it is
    restored here rather than left to the call order of the pass.
    """
    melody_pitches = [note.pitch_midi for note in notes if note.voice_id == VOICE_MELODY]
    if not melody_pitches:
        return notes
    settled = settle_harmony_register(
        notes,
        melody_floor=min(melody_pitches),
        melody_ceiling=max(melody_pitches),
        registers={voice_id: bed_registers(name) for voice_id, name in harmony_voices},
        clearance=clearance,
    )
    settled.sort(key=lambda note: (note.tick, note.voice_id, note.pitch_midi))
    return settled


def _kit(
    *,
    plan: CompositionPlan,
    arrangement: DurationArrangement,
    time_signature: str,
    seed: int,
    long_piece: bool,
    arc: SectionArc,
    written: list[NoteEvent],
) -> list[NoteEvent]:
    """The drum part, written last because it is written against the bass.

    When the piece is for the kit, the piano stays as the accompaniment and a
    percussion voice plays the mood's pattern (voice 2, GM channel-10 keys).
    Styles with A/B variants rotate across sections, matching the chord
    templates' rule. Meters the library does not cover (5/4, 7/8) get no
    percussion rather than a wrong pattern.

    **The kit does not open the piece.** The intro rests it for as long as the
    intro lasts, and `percussion_entry_bar` is the floor under that — the bed
    and the bass state the groove first and the kit arrives on a downbeat with
    a crash. A short piece has no intro bars at all, so without the floor it
    opened on the crash. `generate_percussion` rests everything below
    `entry_bar` itself, so the two cannot disagree; only bars silent for
    another reason travel separately.

    A long piece also rests the kit through one mid-piece section, so the
    texture has a hole in it before it refills.
    """
    entry_bar = max(arrangement.intro_bars if long_piece else 0, plan.percussion_entry_bar)
    rest_bars: set[int] = set()
    if long_piece:
        rest_bars.update(
            range(
                plan.percussion_rest_section * arrangement.form_bars,
                (plan.percussion_rest_section + 1) * arrangement.form_bars,
            )
        )
    return generate_percussion(
        kit=plan.drum_kit(),
        time_signature=time_signature,
        form_bars=arrangement.form_bars,
        repetition_count=arrangement.repetition_count,
        total_bars=arrangement.total_bars_with_coda,
        seed=seed,
        rest_bars=frozenset(rest_bars),
        entry_bar=entry_bar,
        # The bass is already written, so the kit is written knowing where it
        # attacks — see `generate_percussion`.
        bass_onsets=bass_onsets_by_bar(written, bar_ticks(time_signature)),
        arc=arc,
    )


def _section_template(
    spec: CompositionSpec,
    arrangement: DurationArrangement,
    plan: CompositionPlan,
    section_index: int,
    *,
    is_final: bool,
) -> ChordTemplate:
    """The chords one section plays, and the way it stops.

    Three rewrites, and the order between them is the only subtle part.

    The first section carries the arrangement's own template and later ones
    cycle the mood's remaining variants (an A/B form), so a repeat differs
    harmonically and not only in surface rhythm.

    The plan's harmonic rate is cut in *before* the ending is rewritten. A
    close is two one-bar chords, so re-cutting the bars afterwards would
    overwrite the cadence the plan just asked for.

    The last repetition lands at home; every earlier one closes the way the
    plan says, which by default is a half cadence whose V resolves into the
    next section's I. A phrase that stops mid-thought leaves the bar's harmony
    unstated and gives that resolution nothing to resolve.
    """
    template = (
        arrangement.template
        if section_index == 0
        else get_template_for_form(
            spec.mood.value, arrangement.form_bars, variant_index=section_index
        )
    )
    template = _with_harmonic_rhythm(template, plan)
    if is_final:
        return apply_final_cadence(
            template, cadence_degree=plan.cadence_degree, seventh=plan.cadence_seventh
        )
    return apply_section_close(
        template,
        close=plan.section_close,
        cadence_degree=plan.cadence_degree,
        seventh=plan.cadence_seventh,
    )


def _seam(
    section_notes: list[NoteEvent],
    time_signature: str,
    *,
    carried: tuple[int | None, int | None, int | None],
) -> tuple[int | None, int | None, int | None]:
    """What the next section joins onto: the bass's landing tone and the tune's last step.

    A section boundary is a bar line, and the bar after it has to join the line
    the way any other bar does — so these travel across it rather than the next
    section starting against nothing. `carried` is what came in, returned
    unchanged for a voice this section did not write, which is how a
    bass-alone intro hands the melody's seam through to the section that has
    one.

    The bass carries its **landing** tone, not its last note heard. A figure
    decorates above its landing tone, so a bar's last note is wherever the
    figure climbed to, and joining the next chord to *that* would move the walk
    by the figure's reach instead of by the harmony's step. The landing tone is
    the last bar's lowest bass note: every rung resolves at or above it, and
    rung 0 sounds it on the bar line.
    """
    prev_bass, prev_melody, prev_melody_leap = carried
    ticks = bar_ticks(time_signature)
    bass_notes = [n for n in section_notes if n.voice_id == VOICE_BASS]
    if bass_notes:
        last_bar = max(n.tick for n in bass_notes) // ticks
        prev_bass = min(n.pitch_midi for n in bass_notes if n.tick // ticks == last_bar)
    melody_notes = [n for n in section_notes if n.voice_id == VOICE_MELODY]
    if melody_notes:
        prev_melody = melody_notes[-1].pitch_midi
        prev_melody_leap = (
            melody_notes[-1].pitch_midi - melody_notes[-2].pitch_midi
            if len(melody_notes) >= 2
            else None
        )
    return prev_bass, prev_melody, prev_melody_leap


def _terraced(
    notes: list[NoteEvent],
    section_index: int,
    arrangement: DurationArrangement,
    arc: SectionArc,
) -> list[NoteEvent]:
    """A section's notes at its step of the arc.

    Terraced rather than continuous: the section's whole dynamic sits at one
    step, the way a repeat is louder than the statement before it, rather than
    drifting inside the section. The body and the coda both need this and a
    scale of 1.0 rebuilds nothing, which is the case the guard is for.
    """
    scale = section_velocity_scale(section_index, arrangement.repetition_count, arc)
    if scale == 1.0:
        return notes
    return [
        replace(note, velocity=max(1, min(127, round(note.velocity * scale)))) for note in notes
    ]


def _measures(arrangement: DurationArrangement, time_signature: str) -> list[Measure]:
    """One measure per bar, coda included, at a fixed width."""
    ticks = bar_ticks(time_signature)
    return [
        Measure(
            index=bar_idx,
            start_tick=bar_idx * ticks,
            end_tick=(bar_idx + 1) * ticks,
            time_signature=time_signature,
        )
        for bar_idx in range(arrangement.total_bars_with_coda)
    ]


def _ritardando(
    arrangement: DurationArrangement,
    knobs: ArrangementKnobs,
    time_signature: str,
) -> tuple[TempoPoint, ...]:
    """The outro slowdown, as a tempo change or as nothing.

    A coda'd piece slows across its whole coda; a long piece without one eases
    in over its final cadence bars. `arrange_for_duration` already counted the
    slowdown when it fitted the piece to its length, so the tempo map and the
    realised duration agree by construction rather than by luck.
    """
    if arrangement.ritardando_factor >= 1.0:
        return ()
    ticks = bar_ticks(time_signature)
    if arrangement.coda_bars > 0:
        change_tick = arrangement.repetition_count * arrangement.form_bars * ticks
    else:
        change_tick = (arrangement.total_bars - knobs.ritardando_bars) * ticks
    return (
        TempoPoint(
            tick=change_tick,
            bpm=round(arrangement.tempo_bpm * arrangement.ritardando_factor, 1),
        ),
    )


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
    band = melody_band_for(
        melody=ensemble.melody,
        bed=ensemble.harmony,
        band_semitones=shape.line_band_semitones,
        clearance=voices.melody_clearance,
    )
    bass = bass_register(ensemble.bass)
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
        is_final_section = section_idx == arrangement.repetition_count - 1
        section_template = _section_template(
            spec, arrangement, plan, section_idx, is_final=is_final_section
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
            bass=bass,
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
            harmony_voices=active_harmony_voices(
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
        section_notes = _terraced(section_notes, section_idx, arrangement, arc)
        notes.extend(section_notes)
        prev_bass, prev_melody, prev_melody_leap = _seam(
            section_notes,
            time_signature,
            carried=(prev_bass, prev_melody, prev_melody_leap),
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
            bass=bass,
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
        # The coda sits at the step *after* the last section's, which is what
        # makes it the piece's outro rather than one more repeat.
        coda_notes = _terraced(coda_notes, arrangement.repetition_count, arrangement, arc)
        notes.extend(coda_notes)
        cursor_tick += arrangement.coda_bars * bar_ticks(time_signature)

    notes = _settle_bed(notes, harmony_voices=harmony_voices, clearance=voices.melody_clearance)

    if ensemble.percussion == "drum_set":
        notes.extend(
            _kit(
                plan=plan,
                arrangement=arrangement,
                time_signature=time_signature,
                seed=rng_base_seed,
                long_piece=long_piece,
                arc=arc,
                written=notes,
            )
        )

    return (
        NotationScore.make(
            ppq=PPQ,
            key=key,
            time_signature=time_signature,
            tempo_bpm=arrangement.tempo_bpm,
            measures=_measures(arrangement, time_signature),
            notes=notes,
            tempo_changes=_ritardando(arrangement, knobs, time_signature),
        ),
        tuple(chord_bars),
        tuple(bar_keys),
    )


def _generate_section(
    *,
    key: KeySignature,
    time_signature: str,
    template: ChordTemplate,
    section_start_tick: int,
    rng: random.Random,
    seed_for_variation: int,
    band: MelodyBand,
    bass: BassRegister,
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
    keeps. Every landing tone is drawn from `bass.window` and every rung
    resolves under `bass.ceiling` — the walk's own register, narrowed to
    the compass of the instrument that plays it (`bass_register`), so
    neither the walk nor a figure can write a bar the player cannot
    sound. The melody carries too, through `prev_melody` and
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
    # The melodic apex sits at the fraction asked for, rounded *up*, and
    # never on the final bar — the closing gesture is the last bar's, so
    # the peak has to arrive with a bar left after it to be left by.
    #
    # Rounding up rather than truncating is the whole of the placement's
    # correctness at the section sizes the arrangement actually picks. A
    # fraction of 0.6 truncates to bar 4 of an eight-bar section, which is
    # its *middle* bar: the peak arrived at 50%, not 60%, and the phrase
    # was a rise and a fall rather than a climb. Measured over the 960
    # sections of the 3-mood x 7-duration x 10-seed grid, rounding up
    # lifts the share of sections whose highest note lands in the later
    # third — the section's bars from `2 * form // 3` on — from 62.2% to
    # 77.6%, and the share whose highest note is in the apex bar itself
    # from 23.7% to 45.4%.
    apex_bar = min(math.ceil(template.bars * shape.apex_position), template.bars - 2)
    motif = generate_motif(rng, bar_ticks=ticks_per_bar, shape=shape)

    chords, bar_pcs, bar_keys = _resolve_chords(
        template, key=key, key_offset=key_offset, tonic_midi=tonic_midi
    )

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

        bass_pitch = _bass_landing(
            slot,
            key=key,
            tonic_midi=tonic_midi,
            key_offset=_slot_offset(slot_index, len(template.chords), key_offset),
            chord_root=chord_root,
            chord_tones=chord_tones,
            bass=bass,
            root_motion=bass_root_motion,
            prev_bass=prev_bass,
        )
        # The slot's figure, its rungs resolved onto this chord and this
        # landing tone. Every bar of the slot plays it, which is what
        # makes the left hand an accompaniment rather than a new idea
        # every bar.
        bar_figure = bass_figure_pitches(
            slot_figures[slot_index],
            anchor=bass_pitch,
            chord_root=chord_root,
            chord_tones=chord_tones,
            ceiling=bass.ceiling,
        )

        chord_root_tick = section_start_tick + cursor
        for _bar in range(dur):
            bar_tick = chord_root_tick + _bar * ticks_per_bar
            bar_pos = (cursor + _bar * ticks_per_bar) / max(1, section_ticks)
            is_final_bar = is_final_section and bar_index == total_bars - 1

            # Kept, not just extended: the melody is placed against the bass
            # *sounding in its own bar*, and these are those notes.
            bass_bar = _bass_bar(
                None if is_final_bar else bar_figure,
                anchor=bass_pitch,
                chord_root=chord_root,
                chord_tones=chord_tones,
                ceiling=bass.ceiling,
                bar_tick=bar_tick,
                bar_pos=bar_pos,
                ticks_per_bar=ticks_per_bar,
                rng_seed=seed_for_variation,
            )
            notes.extend(bass_bar)

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
            breathe = not is_final_bar and not is_half_cadence and rng.random() < 0.18
            # The breath is a guarantee, not a coin toss: when the
            # melody has run `PHRASE_BARS` bars without a gap (counting
            # the previous section's trailing run), this bar must lift
            # off early. A half cadence already leaves the rest.
            #
            # The apex is not a bar the guarantee may land on, and the
            # reason is the span rather than the peak: deferring the
            # breath past the apex leaves the run one bar over the limit
            # and `PHRASE_GAP_MISSING` then refuses the whole piece. So
            # when the turn falls on the bar before the apex it is taken a
            # bar early, which is also the better phrase: the line closes,
            # then climbs to the peak. Measured over the 3-mood x 7-duration
            # x 10-seed grid, dropping this look-ahead refuses exactly one
            # piece of 210 — `sleep/600/seed 5`, whose run reaches 4.9 bars
            # against the 4-bar span — and leaves the peak readings on the
            # 240-section matrix a wash: the share of sections entering their
            # apex bar from a breath falls from 60.0% to 56.7%, and the share
            # whose apex bar holds the section's top is 53.3% against 52.9%.
            # So the span is what the look-ahead is for, and it is pinned as
            # such by
            # `test_the_breath_guarantee_takes_the_turn_before_the_apex`.
            #
            # The coin toss is *not* vetoed at the apex, deliberately. A bar
            # that reaches the top of its phrase and then lifts off is
            # idiomatic, and the veto was dropped on a measurement. Re-taken
            # over the 960-section grid: vetoing the toss at the apex raises
            # the share of sections whose apex bar stops short from 54.6% to
            # 57.9% and lowers the share whose apex bar holds the section's
            # top from 45.4% to 42.1%. It buys nothing — and costs a little
            # — so the toss keeps the meaning it always had. (The numbers
            # this comment first carried were taken before the apex moved to
            # `ceil` and before the look-ahead above; both flipped the
            # sign of the effect, which is why they are re-stated here
            # rather than dropped.)
            #
            # The condition below carries no `is_apex` clause, and it does
            # not need one: the apex bar can never be *due*. The bar before
            # it either takes the deferred turn — which sets this bar's run
            # to one — or had a run of at most two, so the run at the apex
            # is at most three against a span of four. Confirmed by removal:
            # dropping the `not is_apex` this condition used to carry leaves
            # all 1080 pieces of a 3-mood x 9-duration x 40-seed sweep
            # byte-identical. The guarantee is kept off the apex by the
            # look-ahead above, which is the mechanism the paragraph is
            # about, not by a clause repeated here.
            run = bar_index - last_gap_bar
            due = run >= PHRASE_BARS or (bar_index + 1 == apex_bar and run + 1 >= PHRASE_BARS)
            if not is_final_bar and not is_half_cadence and due:
                breathe = True
            if is_half_cadence or breathe:
                last_gap_bar = bar_index
            anchor = downbeat_anchor(rng, len(chord_tones))
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
                    next_slot = next((c for c in chords[slot_index + 1 :] if c[2] > 0), None)
                    if next_slot is not None:
                        pickup = (next_slot[0], next_slot[1])
            bar_melody = melody_bar(
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
                bass_pitches=tuple(note.pitch_midi for note in bass_bar),
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
            generate_harmony_section(
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
                # How many pads there are decides whether there is a chord to
                # share out at all, so the layer is told rather than guessing
                # from its own index.
                layer_count=len(harmony_voices),
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


# ---------------------------------------------------------------------------
# Performance plan: convert tick-level notation into microsecond timestamps,
# then lay the expression layer on top (the engraved score stays on-grid).
# ---------------------------------------------------------------------------


# Arrangement arc (S8): long pieces lift their final repetition a whole
# step (the piece ends in the new key — the lift IS the ending), drop
# the drums for one mid-piece section to give the texture a hole, and
# step the dynamics per section instead of arching continuously. All
# three decisions now arrive through the plan: the offset and the terraces
# live in `duration.py` with the arrangement they belong to, and the rest
# section in `percussion.py`, which is the module that owns the kit.


__all__ = [
    "CompositionEngineError",
    "EngineErrorCode",
    "EngineOutput",
    "VoiceInstrument",
    "compose",
]
