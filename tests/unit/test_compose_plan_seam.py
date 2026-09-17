"""The `compose(spec, *, plan=...)` seam, and proof that it is live.

`tests/acceptance/test_golden_corpus.py` proves the necessary half: under
`plan=None` the engine still produces the artifacts it produced before the
plan existed. That is what makes threading the plan through safe.

It is not sufficient, and this module is the other half. A `compose` that
accepted `plan` and then ignored it entirely would satisfy every frozen
hash in that corpus — the corpus composes with no plan at all, so the
parameter being dead is invisible to it. A seam that cannot be shown to be
live is a seam that will be found dead at Phase C, when the conductor
starts passing plans that do nothing.

So each knob is composed twice, once at its default and once mutated,
and the whole output — score, performance plan and the arrangement the
sidecar records — has to move. A knob the engine reads from a module
table instead of from the plan moves nothing and fails here.

The cases are grouped by the layer that reads them, and each group names
its fields explicitly, because there is no struct to enumerate them
against the way `ArrangementKnobs` enumerates the arrangement's. B9
closes that: once every layer is wired, one test asserts the union of
these groups names every field of the plan, so a knob added without a
seam case fails rather than quietly going uncovered.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping
from dataclasses import fields, replace
from typing import Any

import pytest

from saimc.canonical import canonical_dumps, canonical_sha256
from saimc.compose.duration import (
    DEFAULT_ARRANGEMENT_KNOBS,
    ArrangementKnobs,
    bar_ticks,
)
from saimc.compose.engine import (
    CompositionEngineError,
    EngineErrorCode,
    EngineOutput,
    _answer_leaps,
    _apex_starts,
    _bent_step,
    _closing_tone,
    _final_closing_degree,
    _generate_harmony_section,
    _generate_percussion,
    _melody_band_for,
    _settle_harmony_register,
    _snap_to_chord,
    _start_offsets,
    _walk_shape,
    compose,
)
from saimc.compose.ensemble import resolve_ensemble
from saimc.compose.motif import (
    BASS_FIGURES,
    DEFAULT_MELODY_SHAPE,
    RHYTHM_WEIGHTS,
    MelodyShape,
    MotifCell,
    MotifVariant,
    _apply_operation,
    _draw_step,
    recover_leaps,
    vary_motif,
)
from saimc.compose.percussion import (
    DEFAULT_DRUM_KIT,
    DRUM_CRASH,
    DRUM_STYLES,
    DrumKit,
)
from saimc.compose.plan import (
    PLAN_SCHEMA_VERSION,
    CompositionPlan,
    PlanError,
    default_plan,
)
from saimc.compose.score import (
    PPQ,
    VOICE_BASS,
    VOICE_HARMONY,
    VOICE_MELODY,
    VOICE_PERCUSSION,
    NoteEvent,
)
from saimc.compose.voices import (
    BROKEN_CHORD_MOODS,
    DEFAULT_HARMONY_VOICES,
    HARMONY_STAB_INSTRUMENTS,
    HarmonyVoices,
)
from saimc.instruments import BedRegisters, MelodyBand, bed_window
from saimc.spec import CompositionSpec

_SPEC = CompositionSpec(mood="calming", duration_seconds=180, seed=11)
"""Long enough that every arc knob is reachable: 180 s at this mood
arranges to eight bars repeated five times, so the piece counts as long
and gets its intro and its ritardando. A short piece would make
`arc_min_reps`, `intro_bars` and the ritardando knobs unobservable and
their half of this module vacuous."""


def _fingerprint(output: EngineOutput) -> tuple[str, str, str]:
    """Everything the plan can reach, in one comparable value.

    The score and the performance plan carry the notes and the tempo map;
    the arrangement is hashed separately because the sidecar rounds it
    through JSON — which is also where a knob that only moved the
    tempo-map's start bar, and no note, would show up.
    """
    return (
        output.notation_score.compute_hash(),
        output.performance_plan.compute_hash(),
        canonical_sha256(output.to_sidecar()["arrangement"]),
    )


_DEFAULT_FINGERPRINT = _fingerprint(compose(_SPEC))

_CODA_SPEC = CompositionSpec(mood="calming", duration_seconds=41, seed=3)
"""The shortest spec that reaches the coda arm of the duration search.

It needs the plan for the reason B2 found: no default arrangement has both
a coda and `repetition_count >= arc_min_reps`, because a coda is only
reached where few repetitions fit — so the long-piece branches inside the
coda arm are reachable only through a lowered `arc_min_reps`. Forty-one
seconds of `calming` arranges to eight bars once plus a four-bar coda, so
the coda is the piece's final four bars.
"""

_LONG_WITH_CODA = replace(default_plan(_CODA_SPEC), arc_min_reps=1)

_TWO_HARMONY_VOICES_SPEC = CompositionSpec(
    mood="calming",
    duration_seconds=180,
    seed=11,
    instrumentation=[
        {"role": "melody", "instrument": "piano"},
        {"role": "harmony", "instrument": "strings"},
        {"role": "harmony", "instrument": "choir"},
        {"role": "bass", "instrument": "cello"},
    ],
)
"""Two harmony voices, which is what the texture cycle needs to say anything.

`_SPEC`'s ensemble has one, and `_active_harmony_voices` returns the voices
untouched when there is only one to choose between — see
`test_a_single_harmony_voice_leaves_the_cycle_unread`.
"""

_DRUM_KIT_SPEC = CompositionSpec(
    mood="electrifying", duration_seconds=180, seed=11, instrumentation="drum_set"
)
"""A kit and a long piece, which is what the drum rest needs to say anything.

The rest section is only consulted `if long_piece`, and there is no
percussion voice at all without `drum_set` — so a spec missing either
would leave `percussion_rest_section` unobservable.
"""

_PINNED_TEMPO_CODA_SPEC = CompositionSpec(
    mood="calming", duration_seconds=59, seed=3, tempo_bpm=80
)
"""A coda over a repeated form, which a free tempo does not reach.

With the tempo free, the search only enters its coda arm in the one gap its
repetition ladder leaves before the second repetition — and the coda's own
ladder covers that gap at one repetition, so every such coda has exactly
one. A pinned tempo makes the fit a point instead of a window, and the coda
then lands over a form repeated twice — 59 s of `calming` at 80 BPM — and
the terrace reads the plan rather than returning 1.0.
"""


_ARRANGEMENT_KNOBS: dict[str, Any] = {
    # A form the default search would not have picked at this length.
    "form_sizes": (16, 32),
    # Below the five repetitions the default search chose, so the form
    # has to grow to still reach the target — a different arrangement,
    # not merely a truncated one.
    "max_repeats": 4,
    # Looser than §8's ±2%: the search accepts the first candidate it
    # would otherwise have passed over, which is the slower end of the
    # mood's tempo range.
    "duration_tolerance": 0.5,
    # No piece is ever long enough to earn the arc, so the intro and the
    # modulation both vanish from a piece that had them.
    "arc_min_reps": 99,
    # A four-bar intro instead of a two-bar one, carved from the same
    # section so the bar count and the duration math are untouched.
    "intro_bars": 4,
    # A deeper slowdown at the close — and, because the duration math
    # includes the slowdown, a different tempo chosen to pay for it.
    "ritardando_factor": 0.5,
    # A longer slowdown, which moves the bar the tempo map changes at
    # without moving the arrangement's own numbers.
    "ritardando_bars": 4,
}
"""One legal, observable value per arrangement field the plan carries.

The keys are exactly `ArrangementKnobs`' fields, checked by the first test
below, so a field added to the struct without a case here fails rather
than quietly going uncovered.
"""


class TestTheSeamIsLive:
    """A non-default plan moves the music. This is the point of the module."""

    def test_every_arrangement_knob_has_a_case(self) -> None:
        declared = {field.name for field in fields(ArrangementKnobs)}
        assert declared == set(_ARRANGEMENT_KNOBS), (
            "an arrangement knob has no case below, so nothing asserts the "
            "engine actually reads it from the plan"
        )

    @pytest.mark.parametrize("knob", sorted(_ARRANGEMENT_KNOBS))
    def test_a_non_default_plan_moves_the_output(self, knob: str) -> None:
        plan = replace(default_plan(_SPEC), **{knob: _ARRANGEMENT_KNOBS[knob]})
        moved = _fingerprint(compose(_SPEC, plan=plan)) != _DEFAULT_FINGERPRINT
        assert moved, (
            f"{knob} is carried by the plan but does not reach the engine: "
            "composing under it produced the same score, performance plan and "
            "arrangement as the default. The seam is dead for this knob."
        )

    def test_the_cases_are_not_all_one_knob_in_disguise(self) -> None:
        """Each case has to be a mutation of the field it names.

        Without this a copy-paste that set `intro_bars` in every case would
        pass, and the parametrised test above would be asserting the same
        thing seven times.
        """
        for knob, value in _ARRANGEMENT_KNOBS.items():
            assert getattr(DEFAULT_ARRANGEMENT_KNOBS, knob) != value, knob

    def test_a_honoured_plan_is_still_a_canonical_artifact(self) -> None:
        """The plan the engine composed under is the plan that was passed.

        Not re-derived, not re-stamped: the caller's number reaches the
        engine, and the engine's answer is `(plan, seed)`, not
        `(plan-overridden-by-a-table, seed)`.
        """
        plan = replace(default_plan(_SPEC), max_repeats=2)
        output = compose(_SPEC, plan=plan)
        assert output.arrangement.repetition_count <= 2
        assert plan.max_repeats == 2


class TestTheArcKnobReachesTheScoreBuilder:
    """`arc_min_reps` is read twice, and one read can hide the other.

    The duration search reads it to decide a piece's intro and its
    slowdown; `_build_score` reads it to decide whether the final
    repetition modulates. Both compare it against `repetition_count`, so a
    mutation that disables the arc disables both at once — the arrangement
    changes, the fingerprint moves, and the second read is never actually
    exercised. The parametrised case above passes even when the score
    builder ignores the plan entirely.

    So this isolates it. A base plan with no intro and a ritardando factor
    of 1.0 leaves the arrangement identical whichever way the knob falls —
    no intro to switch off, and a slowdown of 1.0 leaves the duration math
    untouched — so the modulation lift on the final repetition is the only
    thing left that can move.
    """

    _FLAT_ARC = replace(default_plan(_SPEC), intro_bars=0, ritardando_factor=1.0)

    def test_the_arrangement_does_not_move(self) -> None:
        """The premise. If the arrangement moved too, the assertion below
        would be satisfied by the duration search alone."""
        with_arc = compose(_SPEC, plan=self._FLAT_ARC)
        without_arc = compose(_SPEC, plan=replace(self._FLAT_ARC, arc_min_reps=99))
        assert with_arc.to_sidecar()["arrangement"] == without_arc.to_sidecar()["arrangement"]

    def test_the_modulation_lift_follows_the_plan(self) -> None:
        with_arc = compose(_SPEC, plan=self._FLAT_ARC)
        without_arc = compose(_SPEC, plan=replace(self._FLAT_ARC, arc_min_reps=99))
        assert with_arc.bar_keys != without_arc.bar_keys, (
            "the final repetition did not modulate under a plan that asked for "
            "the arc, or it modulated under one that did not: `_build_score` is "
            "not reading arc_min_reps from the plan"
        )
        assert with_arc.notation_score.compute_hash() != without_arc.notation_score.compute_hash()


_HARMONY_KNOBS: dict[str, Any] = {
    # Another mood's vocabulary. A bass figure decorates a chord slot for
    # as long as its harmony lasts, and the three mood tables state the
    # same slots differently, so swapping the vocabulary re-writes the
    # bass line without touching the chords it is built on.
    "bass_figures": BASS_FIGURES["electrifying"],
    # The walk landing on the chord tone nearest the line rather than on
    # the chord's root. It changes nothing but which tones the landing may
    # choose from, so it re-writes the bass line against the same chords —
    # and, through the tune's own avoid-the-bass rule, whatever the tune
    # writes against it.
    "bass_root_motion": False,
    # Calming closes plagally, approaching the tonic from degree 3; this
    # is the authentic V-I the cadence would otherwise be written with.
    "cadence_degree": 4,
    # And with it a seventh on the cadence chord, which calming's plain
    # triad does not carry.
    "cadence_seventh": True,
    # A lift of a whole tone rather than the engine's minor third. Both
    # are legal (the plan refuses only beyond an octave), and the piece
    # ends in a different new key.
    "modulation_offset": 3,
    # Another pool for the seed to walk. `_SPEC` names no key, which is the
    # only case this field is read in, and it is read once — `chosen_key`
    # picks `pool[seed % len(pool)]` — so a pool of a different length
    # moves the piece to a different root *and* mode and the change is
    # attributable to the field alone. The base is calming's six keys and
    # seed 11 lands on Dm; this two-key pool lands the same seed on Em.
    "key_pool": ("G", "Em"),
}

_HARMONY_FIELDS = frozenset(
    {
        "bass_figures",
        "bass_root_motion",
        "cadence_degree",
        "cadence_seventh",
        "key_pool",
        "modulation_offset",
    }
)
"""The plan's `--- Harmony ---` block, which reads these five.

Spelled out because, unlike the arrangement, this layer has no struct to
enumerate — the comparison below is what keeps this set honest against
the plan's own field list until B9's union check makes it mechanical.
"""


class TestTheHarmonyLayerIsLive:
    """The chords, the cadence, the bass vocabulary and the lift.

    `bass_figures` reaches the notes through `draw_bass_figures`;
    `bass_root_motion` through the candidate tones the walk may land on;
    the two cadence fields through `apply_final_cadence`;
    `modulation_offset` through the `key_offset` the section is transposed
    by. The first four were module-table or inline-mood reads that B3
    lifted into the plan; the fifth is a new musical choice Phase F added,
    and it is here because it belongs to the same block.
    """

    def test_every_harmony_knob_has_a_case(self) -> None:
        assert set(_HARMONY_KNOBS) == _HARMONY_FIELDS

    @pytest.mark.parametrize("knob", sorted(_HARMONY_KNOBS))
    def test_a_non_default_plan_moves_the_output(self, knob: str) -> None:
        plan = replace(default_plan(_SPEC), **{knob: _HARMONY_KNOBS[knob]})
        moved = _fingerprint(compose(_SPEC, plan=plan)) != _DEFAULT_FINGERPRINT
        assert moved, (
            f"{knob} is carried by the plan but does not reach the engine: "
            "composing under it produced the same score, performance plan and "
            "arrangement as the default. The seam is dead for this knob."
        )

    def test_the_cases_are_not_all_one_knob_in_disguise(self) -> None:
        base = default_plan(_SPEC)
        for knob, value in _HARMONY_KNOBS.items():
            assert getattr(base, knob) != value, knob


class TestTheHarmonyKnobsReachTheCodaToo:
    """`_SPEC` reaches the body's cadence and no coda; the coda is the hidden half.

    Every harmony knob is honoured at two call sites. `apply_final_cadence`
    runs on the last repetition of the body and again on the coda template,
    the modulation lift is passed at both of those `_generate_section`
    calls, and the bass vocabulary goes to the coda's call as well. At
    180 s `_SPEC` arranges to five repetitions and no coda at all, so a
    sabotage that ignored the plan at the coda's call sites would leave
    every test above green — the same shape as B2's `arc_min_reps`, where
    one read masked another.

    **The parametrised tests above cannot cover this**, and that is the
    point of the class: a fingerprint that moved because the body read the
    plan moved for reasons that have nothing to do with the coda. So each
    test here reads a quantity the coda alone writes, compares it against
    the same quantity under a plan mutated in exactly one knob, and fails
    if it did not move. What the body did is irrelevant to all three.

    The spec that reaches the coda needs the plan for the reason B2 found:
    no default arrangement has both a coda and
    `repetition_count >= arc_min_reps`, because a coda is only reached
    where few repetitions fit. So the long-piece branches inside the coda
    arm are reachable only through a lowered `arc_min_reps` — see
    `_CODA_SPEC`, whose 41 s arranges to eight bars once plus a four-bar
    coda, so the coda is the piece's final four bars.
    """

    def _compose(self, **changes: Any) -> EngineOutput:
        return compose(_CODA_SPEC, plan=replace(_LONG_WITH_CODA, **changes))

    def _coda_start(self, output: EngineOutput) -> int:
        """The first bar the coda writes.

        `total_bars` counts the body — `form_bars` times the repetitions —
        and `coda_bars` is appended to it rather than included in it, so
        the coda starts where the body ends.
        """
        return output.arrangement.total_bars

    def test_a_long_piece_with_a_coda_is_what_this_composes(self) -> None:
        """The premise, asserted rather than assumed: a spec that stopped
        reaching the coda, or stopped counting as long, would leave the
        assertions below reading the body and covering nothing."""
        output = self._compose()
        arrangement = output.arrangement
        assert arrangement.coda_bars > 0, "the coda arm was not reached"
        assert arrangement.repetition_count >= _LONG_WITH_CODA.arc_min_reps, (
            "the piece is not long, so the coda's long-piece branches are not taken"
        )
        assert len(output.bar_keys) == arrangement.total_bars + arrangement.coda_bars, (
            "the coda is not appended after the body, so `_coda_start` is wrong"
        )

    def test_the_coda_cadence_is_the_plans_cadence(self) -> None:
        """`chord_bars` is written from the template, so the coda's last two
        bars are the coda's cadence and nothing the body handed over can
        move them."""
        for knob in ("cadence_degree", "cadence_seventh"):
            assert (
                self._compose(**{knob: _HARMONY_KNOBS[knob]}).chord_bars[-2:]
                != self._compose().chord_bars[-2:]
            ), (
                f"{knob} reaches the body's cadence but not the coda's: "
                "`apply_final_cadence` is not being given the plan at the coda call"
            )

    def test_the_coda_bass_is_written_from_the_plans_vocabulary(self) -> None:
        """Read as the rhythm of the coda's bass onsets, measured from the coda's start.

        The pitches are not usable here: the coda's walk starts from the
        landing tone the body handed it, and a vocabulary change moves that
        landing tone too, so a pitch-only comparison could move without the
        coda reading anything. The onsets cannot — `draw_bass_figures`
        chooses a figure by index, and a figure states a rhythm before it
        states any pitch, so the coda's onset positions follow the
        vocabulary and nothing else.
        """
        ticks = bar_ticks(_CODA_SPEC.time_signature.value)

        def onsets(output: EngineOutput) -> tuple[int, ...]:
            start = self._coda_start(output) * ticks
            return tuple(
                sorted(
                    note.tick - start
                    for note in output.notation_score.notes
                    if note.tick >= start and note.voice_id == VOICE_BASS
                )
            )

        assert onsets(self._compose(bass_figures=_HARMONY_KNOBS["bass_figures"])) != onsets(
            self._compose()
        ), (
            "the coda's bass rhythm did not follow the plan's vocabulary: "
            "`_generate_section` is not being given it at the coda call"
        )

    def test_the_coda_is_lifted_by_the_plans_offset(self) -> None:
        """The coda's lifted bars carry the key the body's do.

        The assertion is agreement rather than movement, because movement
        is what the body already provides: under a plan that lifts by five
        semitones the body's final repetition is in F, and the coda has to
        be in F too. A coda reading a constant would sit a third away from
        the section it follows.
        """
        output = self._compose(modulation_offset=5)
        start = self._coda_start(output)
        assert output.bar_keys[start] == output.bar_keys[0], (
            "the coda is not in the key the final repetition modulated to: "
            "`_build_score` is not reading the plan's modulation_offset at the coda call"
        )
        assert output.bar_keys[start] != self._compose(modulation_offset=0).bar_keys[start]


_SECTION_KNOBS: dict[str, Any] = {
    # Every terrace pushed well clear of its default, because the value is
    # carried into a MIDI velocity and rounded: a tenth of a step can land
    # on the same integer twice and prove nothing.
    "section_energy_opening": 0.5,
    "section_energy_peak": 1.4,
    "section_energy_final": 0.6,
    "section_energy_middle": 1.3,
    # The arc's own shape with the two thinning phases swapped, so the
    # breakdown arrives second instead of third. Four phases, still: a
    # shorter cycle would be a differently-shaped arc rather than a
    # mutation of this one.
    "harmony_texture_cycle": ("all", "rest", "first", "all"),
    # The kit rests late instead of early, which moves the hole in the
    # texture rather than removing it.
    "percussion_rest_section": 2,
}

_SECTION_FIELDS = frozenset(
    {
        "section_energy_opening",
        "section_energy_peak",
        "section_energy_final",
        "section_energy_middle",
        "harmony_texture_cycle",
        "percussion_rest_section",
    }
)
"""The plan's `--- Sections ---` block, which reads these six."""

_SECTION_SPECS: dict[str, CompositionSpec] = {
    # The four energies are readable on any long piece: `_SPEC` arranges to
    # five repetitions, so opening, peak, final and middle are four
    # different sections of it.
    "section_energy_opening": _SPEC,
    "section_energy_peak": _SPEC,
    "section_energy_final": _SPEC,
    "section_energy_middle": _SPEC,
    # The other two need an ensemble. A texture cycle chooses between
    # harmony voices, so a piece with one has nothing for it to choose; a
    # drum rest is only consulted when there is a kit.
    "harmony_texture_cycle": _TWO_HARMONY_VOICES_SPEC,
    "percussion_rest_section": _DRUM_KIT_SPEC,
}
"""The spec each knob needs to be observable at all — see the note above."""

_ENERGY_KNOBS = tuple(sorted(name for name in _SECTION_KNOBS if name.startswith("section_energy")))
"""The four terraces, in the order `_section_velocity_scale` reads them."""


class TestTheSectionsLayerIsLive:
    """The arc, the texture and the kit's rest.

    Unlike B2's and B3's layers these six do not share one spec: the
    texture cycle is unobservable without a second harmony voice and the
    drum rest without a kit, so each knob is composed under the ensemble
    that can show it. A layer this thin would otherwise be "covered" by a
    case that could not have failed.
    """

    def test_every_section_knob_has_a_case(self) -> None:
        assert set(_SECTION_KNOBS) == _SECTION_FIELDS

    @pytest.mark.parametrize("knob", sorted(_SECTION_KNOBS))
    def test_a_non_default_plan_moves_the_output(self, knob: str) -> None:
        spec = _SECTION_SPECS[knob]
        base = default_plan(spec)
        before = _fingerprint(compose(spec, plan=base))
        after = _fingerprint(compose(spec, plan=replace(base, **{knob: _SECTION_KNOBS[knob]})))
        assert after != before, (
            f"{knob} is carried by the plan but does not reach the engine: "
            f"composing {spec.duration_seconds}s of {spec.mood.value} under it "
            "produced the same score, performance plan and arrangement as the "
            "default. The seam is dead for this knob."
        )

    def test_the_cases_are_not_all_one_knob_in_disguise(self) -> None:
        for knob, value in _SECTION_KNOBS.items():
            assert getattr(default_plan(_SECTION_SPECS[knob]), knob) != value, knob


class TestTheVelocityTerracesAreReadAtEveryCallSite:
    """`_section_velocity_scale` is read four times, and one read hides another.

    B3's lesson one level down. The parametrised case above passes
    whichever call site is reading the plan — a mutation of any terrace
    moves the whole fingerprint — and three of the four are further
    masked from each other, because a `_SPEC` that had no kit would leave
    the percussion site unexercised and a kit piece would move for the
    kit whether or not the body read anything. So each site is read here
    through a quantity only that site writes: the melodic notes'
    velocities for the body, the percussion voice's own velocities for the
    kit, the CC11 lane for the controller, and the coda's own notes for
    the coda. A site that stopped reading the plan leaves its quantity
    exactly where it was.
    """

    @staticmethod
    def _melodic_velocities(output: EngineOutput) -> tuple[tuple[int, int, int, int], ...]:
        return tuple(
            (note.tick, note.voice_id, note.pitch_midi, note.velocity)
            for note in output.notation_score.notes
            if note.voice_id != VOICE_PERCUSSION
            and note.voice_id in {VOICE_BASS, VOICE_MELODY, VOICE_HARMONY}
        )

    @staticmethod
    def _kit_velocities(output: EngineOutput) -> tuple[tuple[int, int, int], ...]:
        return tuple(
            (note.tick, note.pitch_midi, note.velocity)
            for note in output.notation_score.notes
            if note.voice_id == VOICE_PERCUSSION
        )

    @staticmethod
    def _expression_values(output: EngineOutput) -> tuple[int, ...]:
        """The CC11 lane, which only the controller site writes."""
        return tuple(c.value for c in output.performance_plan.controllers if c.control == 11)

    @pytest.mark.parametrize("knob", _ENERGY_KNOBS)
    def test_the_body_notes_follow_the_plan(self, knob: str) -> None:
        base = default_plan(_SPEC)
        changed = replace(base, **{knob: _SECTION_KNOBS[knob]})
        assert self._melodic_velocities(compose(_SPEC, plan=changed)) != self._melodic_velocities(
            compose(_SPEC, plan=base)
        ), (
            f"{knob} does not reach the note velocities: the body's terrace in "
            "`_build_score` is not reading the plan's arc"
        )

    @pytest.mark.parametrize("knob", _ENERGY_KNOBS)
    def test_the_kits_own_notes_follow_the_plan(self, knob: str) -> None:
        """The kit's terrace, read from the percussion voice alone.

        It has to be read from there: `_generate_percussion` writes voice 2
        and nothing else does, so these velocities cannot move because some
        other site read the plan.
        """
        plan = default_plan(_DRUM_KIT_SPEC)
        changed = replace(plan, **{knob: _SECTION_KNOBS[knob]})
        before, after = (
            self._kit_velocities(compose(_DRUM_KIT_SPEC, plan=p)) for p in (plan, changed)
        )
        assert before, "the premise: this spec composes with a kit at all"
        assert after != before, (
            f"{knob} does not reach the kit's velocities: the terrace in "
            "`_generate_percussion` is not reading the plan's arc"
        )

    @pytest.mark.parametrize("knob", _ENERGY_KNOBS)
    def test_the_expression_controller_follows_the_plan(self, knob: str) -> None:
        base = default_plan(_SPEC)
        changed = replace(base, **{knob: _SECTION_KNOBS[knob]})
        before = self._expression_values(compose(_SPEC, plan=base))
        assert before, "the premise: this spec composes with an expression lane"
        assert self._expression_values(compose(_SPEC, plan=changed)) != before, (
            f"{knob} does not reach the CC11 lane: the controller site in "
            "`_build_performance_plan` is not reading the plan's arc"
        )

    def test_the_coda_follows_the_plans_middle_terrace(self) -> None:
        """The fourth site, and the one whose index falls through.

        `_build_score` calls `_section_velocity_scale` a fourth time for the
        coda, passing `repetition_count` as both the section index and the
        count. That index is never 0, so the opening arm is out; it is never
        `repetition_count - 1`, so the final arm is out; and it is never
        `repetition_count - 2`, so the peak is out. The coda's whole dynamic
        is therefore the plan's *middle* terrace — which is why the test
        asserts both directions: the middle moves the coda and the other
        three cannot. A site ignored by `_build_score` would leave the
        middle's mutation without an effect.

        The spec has to be the pinned-tempo one. With the tempo free the
        coda always has a single repetition, and `_section_velocity_scale`
        returns 1.0 below two — so the loop is skipped and the site is inert.
        """
        plan = default_plan(_PINNED_TEMPO_CODA_SPEC)
        arrangement = compose(_PINNED_TEMPO_CODA_SPEC, plan=plan).arrangement
        assert arrangement.coda_bars > 0, "the premise: a coda was reached"
        assert arrangement.repetition_count >= 2, (
            "the premise: with one repetition the coda's terrace returns 1.0 "
            "and this site is inert, which is what a free tempo always gives"
        )
        ticks = bar_ticks(_PINNED_TEMPO_CODA_SPEC.time_signature.value)
        coda_start = arrangement.total_bars * ticks

        def coda_velocities(**changes: Any) -> tuple[int, ...]:
            return tuple(
                note.velocity
                for note in compose(
                    _PINNED_TEMPO_CODA_SPEC, plan=replace(plan, **changes)
                ).notation_score.notes
                if note.tick >= coda_start and note.voice_id != VOICE_PERCUSSION
            )

        baseline = coda_velocities()
        assert baseline, "the premise: the coda writes notes with velocities"
        assert coda_velocities(section_energy_middle=_SECTION_KNOBS["section_energy_middle"]) != (
            baseline
        ), (
            "section_energy_middle does not reach the coda's velocities: the terrace "
            "at the coda's call to `_generate_section` is not reading the plan"
        )
        for knob in ("section_energy_opening", "section_energy_peak", "section_energy_final"):
            assert coda_velocities(**{knob: _SECTION_KNOBS[knob]}) == baseline, (
                f"{knob} moved the coda's velocities, so the coda's terrace is not "
                "indexed by `repetition_count` the way this test assumes"
            )


class TestTheTextureCycleNeedsVoicesToChoose:
    """A texture cycle says which harmony voices play, so it needs two.

    `_active_harmony_voices` returns the voices untouched with fewer than
    two, or on a piece too short to have an arc. That is why `_SPEC` does
    not cover this knob above, and the second test here pins it: the cycle
    is not ignored, it has nothing to decide.
    """

    def test_the_two_voice_spec_is_what_the_cycle_needs(self) -> None:
        """The premise, asserted rather than assumed."""
        arrangement = compose(_TWO_HARMONY_VOICES_SPEC).arrangement
        assert arrangement.repetition_count >= default_plan(_TWO_HARMONY_VOICES_SPEC).arc_min_reps, (
            "the piece is not long, so `_active_harmony_voices` would return the "
            "voices untouched whatever the cycle said"
        )
        harmony = {
            note.voice_id
            for note in compose(_TWO_HARMONY_VOICES_SPEC).notation_score.notes
            if note.voice_id >= VOICE_HARMONY
        }
        assert len(harmony) >= 2, "the premise: this spec writes two harmony voices"

    def test_the_cycle_moves_the_output(self) -> None:
        base = default_plan(_TWO_HARMONY_VOICES_SPEC)
        changed = replace(base, harmony_texture_cycle=_SECTION_KNOBS["harmony_texture_cycle"])
        assert _fingerprint(compose(_TWO_HARMONY_VOICES_SPEC, plan=changed)) != _fingerprint(
            compose(_TWO_HARMONY_VOICES_SPEC, plan=base)
        ), (
            "harmony_texture_cycle does not reach the engine: `_active_harmony_voices` "
            "is not reading the plan's cycle"
        )

    def test_a_single_harmony_voice_leaves_the_cycle_unread(self) -> None:
        """Not a dead seam — an empty choice.

        With one harmony voice every group names the same set, so the
        engine's own guard returns the voices before the cycle is
        consulted. Byte-identical, which is the honest behaviour: a plan
        that names a cycle for a piece with nothing to cycle is describing
        an arc of one colour.
        """
        changed = replace(
            default_plan(_SPEC), harmony_texture_cycle=_SECTION_KNOBS["harmony_texture_cycle"]
        )
        assert _fingerprint(compose(_SPEC, plan=changed)) == _DEFAULT_FINGERPRINT


class TestTheDrumRestFollowsThePlan:
    """The hole in the texture is a section of the plan's choosing.

    `_generate_percussion` takes a set of silent bars, and on a long piece
    that set is the intro plus section `percussion_rest_section`. The
    second test reads the silence back out of the score rather than the
    plan, so a rest window computed from a constant would be caught.
    """

    def test_the_kit_spec_is_what_the_rest_needs(self) -> None:
        """The premise: a kit, and a piece long enough for the rest arm."""
        output = compose(_DRUM_KIT_SPEC)
        arrangement = output.arrangement
        assert arrangement.repetition_count >= default_plan(_DRUM_KIT_SPEC).arc_min_reps, (
            "the piece is not long, so no rest is scheduled at all"
        )
        ticks = bar_ticks(_DRUM_KIT_SPEC.time_signature.value)
        assert {
            note.tick // ticks
            for note in output.notation_score.notes
            if note.voice_id == VOICE_PERCUSSION
        }, "the premise: this spec composes with a kit at all"

    def _silent_bars(self, **changes: Any) -> tuple[int, ...]:
        plan = replace(default_plan(_DRUM_KIT_SPEC), **changes)
        output = compose(_DRUM_KIT_SPEC, plan=plan)
        ticks = bar_ticks(_DRUM_KIT_SPEC.time_signature.value)
        played = {
            note.tick // ticks
            for note in output.notation_score.notes
            if note.voice_id == VOICE_PERCUSSION
        }
        return tuple(
            bar for bar in range(output.arrangement.total_bars_with_coda) if bar not in played
        )

    def test_the_rest_window_is_the_plans_section(self) -> None:
        form_bars = compose(_DRUM_KIT_SPEC).arrangement.form_bars
        rest = _SECTION_KNOBS["percussion_rest_section"]

        def window(section: int) -> range:
            return range(section * form_bars, (section + 1) * form_bars)

        silent = self._silent_bars()
        assert set(window(1)) <= set(silent), "the default's rest section does not rest"
        assert not set(window(rest)) & set(silent), "the plan's section was rested anyway"

        moved = self._silent_bars(percussion_rest_section=rest)
        assert set(window(rest)) <= set(moved), (
            "percussion_rest_section does not reach `_generate_percussion`: the "
            "rest window is not the section the plan names"
        )
        assert not set(window(1)) & set(moved), "the default's section was rested anyway"


class TestTheWidestLiftIsOneTheEngineCanHonour:
    """`abs(modulation_offset) <= 12` is a musical bound, not a guess.

    A twelve-semitone lift leaves the final repetition's *key signature*
    alone and moves its notes an octave — which is exactly what makes it
    the edge of a lift rather than a modulation to somewhere else. The
    plan refuses thirteen (`test_it_raises_with_the_reason` in
    `test_compose_plan.py`), so twelve is the far end of the range an
    editor is allowed to reach, and a bound the engine could not honour at
    its own limit would be a bound that refuses legal plans.
    """

    def test_the_extreme_compiles_and_moves_the_notes(self) -> None:
        """`compose` lints and raises on an illegal piece, so composing at
        all is half the assertion. The other half needs a bar that is
        actually lifted: the last two bars of a section are its cadence and
        carry no offset (`slot_offset` in `_generate_section`), so the
        lifted key has to be read from the top of the final repetition —
        and the notes have to move while the key does not."""
        unlifted = compose(_SPEC, plan=replace(default_plan(_SPEC), modulation_offset=0))
        lifted_bar = unlifted.arrangement.total_bars - unlifted.arrangement.form_bars
        for offset in (12, -12):
            lifted = compose(_SPEC, plan=replace(default_plan(_SPEC), modulation_offset=offset))
            assert lifted.notation_score.compute_hash() != unlifted.notation_score.compute_hash()
            assert lifted.bar_keys[lifted_bar] == unlifted.bar_keys[lifted_bar], (
                "an octave away is the same key, which is why an octave is the "
                "bound the plan states"
            )


_MELODY_KNOBS: dict[str, Any] = {
    # 0 and 1 swapped, so the two most-weighted steps trade places. A
    # permutation rather than a shortened table: the plan requires one
    # weight per choice, and a table of another length would be refused
    # before the hash was ever taken.
    "step_choices": (-4, -3, -2, -1, 1, 0, 2, 3, 4),
    # The same shape of table with the two commonest steps flattened: the
    # draw still walks by step most of the time, and lands elsewhere.
    "step_weights": (5, 3, 10, 28, 6, 28, 10, 5, 3),
    # Narrower than any phrase the renderer writes, so the walk folds
    # where the default let it climb. Four rather than five: the span only
    # binds once a walk has already left the middle of the vocabulary, and
    # at five this piece's own draws never leave it — a lower span is the
    # same knob, past the threshold where it is observable.
    "max_motif_span_degrees": 4,
    # A fourth upwards counts as a leap, so the answering pass has more
    # to answer.
    "leap_degrees": 4,
    # Transposition moves the anchor a fourth instead of a third, and
    # every chord-tone snap looks one degree further.
    "chord_tone_degrees": 3,
    # Another mood's figures, which dress the same pitches in a
    # different rhythm.
    "rhythm_weights": tuple(RHYTHM_WEIGHTS["electrifying"].items()),
    # The operation draw all but forced onto inversion, which mirrors the
    # contour rather than restating it.
    "motif_operation_weights": (
        ("repeat", 0.05),
        ("transpose", 0.05),
        ("sequence", 0.05),
        ("invert", 0.80),
        ("truncate", 0.05),
    ),
    # Ties nearly always, where calming holds them a little over a quarter
    # of the time.
    "tie_probability": 0.9,
    # The apex early in the section rather than late.
    "apex_position": 0.3,
    # A narrower line band, which the instrument's tessitura can hold.
    "line_band_semitones": 17,
}

_MELODY_FIELDS = frozenset(
    {
        "step_choices",
        "step_weights",
        "max_motif_span_degrees",
        "leap_degrees",
        "chord_tone_degrees",
        "motif_operation_weights",
        "rhythm_weights",
        "tie_probability",
        "apex_position",
        "line_band_semitones",
    }
)
"""The plan's melody group, which reads these ten — the vocabulary, the
phrase shape, and the register window the line is written in."""


class TestTheMelodyLayerIsLive:
    """The vocabulary, the phrase shape and the line's register window.

    Every one of these was a module constant or an inline literal before
    B5: the step table and the motif's span in `motif.py`, the tie
    probability and the apex fraction inline in `_generate_section`, the
    rhythm figures looked up by mood inside `_melody_bar`, and the line
    band from `instruments.LINE_BAND_SEMITONES`.
    """

    def test_every_melody_knob_has_a_case(self) -> None:
        assert set(_MELODY_KNOBS) == _MELODY_FIELDS

    @pytest.mark.parametrize("knob", sorted(_MELODY_KNOBS))
    def test_a_non_default_plan_moves_the_output(self, knob: str) -> None:
        plan = replace(default_plan(_SPEC), **{knob: _MELODY_KNOBS[knob]})
        moved = _fingerprint(compose(_SPEC, plan=plan)) != _DEFAULT_FINGERPRINT
        assert moved, (
            f"{knob} is carried by the plan but does not reach the engine: "
            "composing under it produced the same score, performance plan and "
            "arrangement as the default. The seam is dead for this knob."
        )

    def test_the_cases_are_not_all_one_knob_in_disguise(self) -> None:
        base = default_plan(_SPEC)
        for knob, value in _MELODY_KNOBS.items():
            assert getattr(base, knob) != value, knob


_TWO_CELL_MOTIF = (
    MotifCell(step=0, length_ticks=PPQ),
    MotifCell(step=2, length_ticks=PPQ),
)
"""The motif the operation cases are applied to."""

_REPEATING_VARIANT = MotifVariant(
    motif=(MotifCell(step=0, length_ticks=PPQ), MotifCell(step=1, length_ticks=PPQ)),
    repeat=True,
)
"""A sequence — the one walk that reads `chord_tone_degrees` between replays."""

_FLAT_OPERATION_WEIGHTS = (
    ("repeat", 0.05),
    ("transpose", 0.05),
    ("sequence", 0.05),
    ("invert", 0.80),
    ("truncate", 0.05),
)


def _drawn_steps(shape: MelodyShape, *, degree: int = 0) -> tuple[int, ...]:
    """Thirty draws of `_draw_step`, which is where the step table is read."""
    return tuple(_draw_step(random.Random(seed), degree, shape=shape) for seed in range(30))


_MELODY_SITES: dict[str, tuple[str, Any, Any]] = {
    # One case per *site*, not per field. The parametrised class above
    # cannot separate these: six of the sites below read the same
    # `chord_tone_degrees` and three the same `leap_degrees`, so a plan
    # that mutates either moves the whole fingerprint whichever site is
    # reading, and a helper that had gone back to the module constant
    # would leave the layer test green. Each case therefore calls the
    # reader directly, with a shape mutated in exactly the field that
    # reader reads, and asserts its own answer moves.
    "motif._draw_step draws from the plan's step table": (
        "step_choices",
        (-4, -3, -2, -1, 1, 0, 2, 3, 4),
        _drawn_steps,
    ),
    "motif._draw_step weights the draw by the plan's weights": (
        "step_weights",
        (1,) * 9,
        _drawn_steps,
    ),
    "motif._draw_step stops at the plan's span": (
        # Walked from degree 2, where a span of 2 and a span of 8 stop
        # admitting the same steps — the knob only binds away from the
        # middle of the vocabulary.
        "max_motif_span_degrees",
        2,
        lambda shape: _drawn_steps(shape, degree=2),
    ),
    "motif.recover_leaps answers at the plan's leap size": (
        "leap_degrees",
        4,
        lambda shape: recover_leaps([0, 3, 0], shape=shape),
    ),
    "motif.vary_motif draws operations by the plan's weights": (
        "motif_operation_weights",
        _FLAT_OPERATION_WEIGHTS,
        lambda shape: vary_motif(_TWO_CELL_MOTIF, random.Random(0), shape=shape),
    ),
    "motif._apply_operation transposes by the plan's chord tone": (
        "chord_tone_degrees",
        4,
        lambda shape: _apply_operation(
            _TWO_CELL_MOTIF, "transpose", random.Random(0), shape=shape
        ),
    ),
    "engine._answer_leaps answers at the plan's leap size": (
        "leap_degrees",
        4,
        lambda shape: _answer_leaps([0, 3, 0], shape=shape),
    ),
    "engine._bent_step bends within the plan's chord tone": (
        # A wide chord tone, because with the default's two both of this
        # bend's candidates fail the test and the fallback happens to pick
        # the same direction — so a near-default value would be a case
        # that could not have failed.
        "chord_tone_degrees",
        5,
        lambda shape: _bent_step(0, True, -4, shape=shape),
    ),
    "engine._snap_to_chord weighs leaps at the plan's size": (
        # The read is a tie-break among equally costly candidates, so the
        # case needs a neighbour a leap of three away from one candidate
        # and of five from the other — which is where the two leap sizes
        # part company.
        "leap_degrees",
        4,
        lambda shape: _snap_to_chord(
            1, tone_count=3, prefer_up=True, neighbours=(-3,), shape=shape
        ),
    ),
    "engine._walk_shape advances by the plan's chord tone": (
        "chord_tone_degrees",
        3,
        lambda shape: _walk_shape(
            _REPEATING_VARIANT, bar_ticks=4 * PPQ, tone_count=4, shape=shape
        )[0],
    ),
    "engine._closing_tone closes on the plan's chord tone": (
        "chord_tone_degrees",
        3,
        lambda shape: _closing_tone(2, 0, half_cadence=False, shape=shape),
    ),
    "engine._start_offsets restates by the plan's chord tone": (
        "chord_tone_degrees",
        3,
        lambda shape: _start_offsets(1, 4, shape=shape),
    ),
    "engine._apex_starts lifts by the plan's chord tone": (
        # Anchored at 2, where a third and a fourth put the drawn tone on
        # opposite sides of the lattice and the lift is a different set.
        "chord_tone_degrees",
        3,
        lambda shape: _apex_starts(2, 4, shape=shape),
    ),
    "engine._final_closing_degree closes on the plan's chord tone": (
        # Seed 0 draws past the two-in-three the tonic gets, so this is the
        # branch that reads the plan at all.
        "chord_tone_degrees",
        3,
        lambda shape: _final_closing_degree(random.Random(0), shape=shape),
    ),
}
"""One case per place the melody layer reads its shape, and why that case
is the one that can see it.

Every value here is a legal one — a caller may build a `MelodyShape` by
hand and the plan checks its own — and every case was checked to move its
reader's answer before it was written down: a case whose two shapes agree
would be a guard that cannot fail.

`_melody_bar`'s own inline reads are absent because they are gone. The
drawn start now comes from the lattice `_start_offsets` builds, and the
final bar's closing degree from `_final_closing_degree` — both because an
inline expression reading a field that six helpers also read cannot be
shown to read the plan: reverting it to the constant leaves every other
case green, which was verified by firing exactly that sabotage.
"""


class TestTheMelodyVocabularyIsReadAtEachSite:
    """Proof that the melody layer reads its vocabulary where it reads it.

    The fingerprint class above proves the plan reaches the music. This one
    proves *which read* does it, which a fingerprint cannot: the melody
    layer reads its shape in thirteen places across two modules, six of
    them reading the same field, so a single plan mutation moves everything
    and attributes nothing.
    """

    @pytest.mark.parametrize("site", sorted(_MELODY_SITES))
    def test_the_reader_reads_the_shape_it_is_handed(self, site: str) -> None:
        field, value, call = _MELODY_SITES[site]
        default = call(DEFAULT_MELODY_SHAPE)
        changed = call(replace(DEFAULT_MELODY_SHAPE, **{field: value}))
        assert changed != default, (
            f"{site}: the reader's answer did not move when `{field}` did, so it "
            "is reading the module constant rather than the shape it was handed"
        )

    def test_the_cases_are_not_all_one_reader_in_disguise(self) -> None:
        """Each case has to be a mutation of the field it names, of the
        reader it names, and nothing else.

        Two ways this could quietly stop covering what it says. A value
        that is the field's own default would leave the case asserting on
        two identical shapes; and two cases reaching the same reader
        through the same field would be the same guard written twice. A
        reader may appear more than once — `_draw_step` reads three of the
        plan's fields — but not for the same one.

        The six chord-tone cases deliberately take different values — 3,
        4 and 5 — because each site needs the value that makes *it* move:
        `_bent_step`'s read only surfaces past a wide chord tone, and the
        lattice helpers need the drawn tone on the other side of the
        octave. One value for all six would be a tidier table and a worse
        set of guards.
        """
        covered: set[tuple[str, str]] = set()
        for site, (field, value, _call) in _MELODY_SITES.items():
            assert getattr(DEFAULT_MELODY_SHAPE, field) != value, site
            pair = (site.split(maxsplit=1)[0], field)
            assert pair not in covered, f"{pair} is covered twice"
            covered.add(pair)

    def test_every_reader_of_the_two_masked_fields_has_a_case(self) -> None:
        """`chord_tone_degrees` and `leap_degrees` are the fields a
        fingerprint cannot attribute, so their readers are enumerated here
        rather than left to the parametrised class: seven sites for the
        first and three for the second. A reader added to either field
        without a case above fails here."""
        by_field: dict[str, list[str]] = {}
        for site, (field, _value, _call) in _MELODY_SITES.items():
            by_field.setdefault(field, []).append(site)
        assert len(by_field["chord_tone_degrees"]) == 7
        assert len(by_field["leap_degrees"]) == 3


_BAR = bar_ticks("4/4")

_ELECTRIFYING_SPEC = CompositionSpec(mood="electrifying", duration_seconds=180, seed=11)
"""A mood whose own default is the broken chord.

`BROKEN_CHORD_MOODS` holds one mood of three, and the arpeggio and the
stab figures only exist under one — so the two cases that read those
figures compose at this spec rather than at `_SPEC`, whose texture is
the sustained pad.
"""

_BRASS_HARMONY_SPEC = CompositionSpec(
    mood="electrifying",
    duration_seconds=180,
    seed=11,
    instrumentation=[
        {"role": "melody", "instrument": "piano"},
        {"role": "harmony", "instrument": "brass_section"},
        {"role": "bass", "instrument": "contrabass"},
    ],
)
"""A broken chord written for an instrument the stab pattern names.

`_ELECTRIFYING_SPEC`'s ensemble states its harmony on strings, which is
not in `HARMONY_STAB_INSTRUMENTS` — so its broken chord is an arpeggio
and the stab's own velocity is unreachable there.
"""

_VOICES_KNOBS: dict[str, tuple[CompositionSpec, Any]] = {
    # The texture itself, against the mood's own sustained default. The
    # other five cases below are read *under* a texture, which is why
    # this one has to be the flip.
    "harmony_broken_chord": (_SPEC, True),
    # Three levels, one per figure, each at the spec whose default
    # texture reaches it.
    "harmony_pad_velocity": (_SPEC, 60),
    "harmony_arpeggio_velocity": (_ELECTRIFYING_SPEC, 80),
    "harmony_stab_velocity": (_BRASS_HARMONY_SPEC, 90),
    # How fine the arpeggio's grid is — an eighth by default, a quarter
    # here, which halves the onsets in a bar.
    "harmony_arpeggio_step_ticks": (_ELECTRIFYING_SPEC, 4 * PPQ),
    # How far the bed keeps off the tune, on both sides of it.
    "harmony_melody_clearance": (_SPEC, 6),
}

_VOICES_FIELDS = frozenset(
    {
        "harmony_broken_chord",
        "harmony_arpeggio_step_ticks",
        "harmony_pad_velocity",
        "harmony_arpeggio_velocity",
        "harmony_stab_velocity",
        "harmony_melody_clearance",
    }
)
"""The plan's `--- Voices ---` block, which is these six.

Enumerated against `HarmonyVoices`' own fields as well, so the struct and
the plan cannot drift apart here — and spelled out because B9's union
check needs the layer's field set by name.
"""

_ARPEGGIO_KNOBS = frozenset(
    {"harmony_arpeggio_step_ticks", "harmony_arpeggio_velocity", "harmony_stab_velocity"}
)
"""The cases whose reader only exists under a broken chord."""


class TestTheVoicesLayerIsLive:
    """How the accompaniment under the tune is written.

    The plan's `--- Voices ---` group is the texture (`broken_chord`), the
    figure's grid, the three levels, and the clearance the bed keeps from
    the tune. All six were module constants or inline expressions in
    `engine.py` before B6 — `HARMONY_*_VELOCITY`, the `PPQ // 2` step, the
    `mood != "electrifying"` texture rule, `HARMONY_MELODY_CLEARANCE` —
    and now live in `saimc.compose.voices`, which is where a plan and the
    engine can both read them.
    """

    def test_every_voices_knob_has_a_case(self) -> None:
        declared = {"harmony_" + field.name for field in fields(HarmonyVoices)}
        assert declared == set(_VOICES_KNOBS) == set(_VOICES_FIELDS), (
            "a voices knob has no case below, so nothing asserts the engine "
            "actually reads it from the plan"
        )

    @pytest.mark.parametrize("knob", sorted(_VOICES_KNOBS))
    def test_a_non_default_plan_moves_the_output(self, knob: str) -> None:
        spec, value = _VOICES_KNOBS[knob]
        default = compose(spec)
        plan = replace(default_plan(spec), **{knob: value})
        moved = _fingerprint(compose(spec, plan=plan)) != _fingerprint(default)
        assert moved, (
            f"{knob} is carried by the plan but does not reach the engine: "
            "composing under it produced the same score, performance plan and "
            "arrangement as the default. The seam is dead for this knob."
        )

    def test_each_case_is_composed_at_the_texture_its_reader_needs(self) -> None:
        """The reason this layer needs three specs rather than one.

        A figure only reads the plan where it is written at all: the
        arpeggio's grid and level are read by a broken chord on an
        instrument the stab pattern is not written for, and the stab's
        level only by a broken chord on one that it is. A case run at
        another texture would be naming a field its figure never consults
        — a guard that cannot fail — so the texture each case needs is
        asserted rather than left in a comment for the next spec edit.
        """
        for knob in _ARPEGGIO_KNOBS:
            spec, _value = _VOICES_KNOBS[knob]
            assert default_plan(spec).harmony_broken_chord, (
                f"{knob}: nothing but a broken chord writes this figure"
            )
        assert resolve_ensemble(_BRASS_HARMONY_SPEC).harmonies[0] in HARMONY_STAB_INSTRUMENTS, (
            "the stab case's ensemble states its harmony on an instrument "
            "the pattern is not written for, so it arpeggiates instead"
        )
        pad_spec, _pad_value = _VOICES_KNOBS["harmony_pad_velocity"]
        assert pad_spec.mood.value not in BROKEN_CHORD_MOODS, (
            "the pad case's spec is no longer a mood outside `BROKEN_CHORD_MOODS`"
        )

    def test_the_cases_are_not_all_one_knob_in_disguise(self) -> None:
        for knob, (spec, value) in _VOICES_KNOBS.items():
            assert getattr(default_plan(spec), knob) != value, knob


def _harmony_notes(
    voices: HarmonyVoices, *, instrument: str = "strings", layer_index: int = 0
) -> list[NoteEvent]:
    """Two bars of one chord, written by the harmony pass at this shape.

    The section is called directly rather than composed, for the reason B5
    found: a plan mutation moves the whole fingerprint whichever of these
    sites reads it, so only a direct call can attribute a read to a site.
    The window is the instrument's own, as the pass is handed it.
    """
    return _generate_harmony_section(
        chords=[(60, (0, 4, 7), 2)],
        section_start_tick=0,
        ticks_per_bar=_BAR,
        rng=random.Random(7),
        melody_from_bar=0,
        seed_for_variation=0,
        instrument=instrument,
        window=bed_window(instrument),
        layer_index=layer_index,
        voices=voices,
    )


def _shape_of(notes: list[NoteEvent]) -> list[tuple[int, int, int]]:
    """Where every note is — its tick, pitch and length — but not its level."""
    return [(note.tick, note.pitch_midi, note.duration_ticks) for note in notes]


_PAD = replace(DEFAULT_HARMONY_VOICES, broken_chord=False)
_BROKEN = replace(DEFAULT_HARMONY_VOICES, broken_chord=True)


class TestTheVoicesWritersReadTheirShape:
    """One case per read inside the harmony writer.

    `harmony_broken_chord` is read twice in one expression pair — once for
    the pad, once for the stabs — and three levels and a grid are read
    below it. The fingerprint class above proves the plan reaches the
    music; this one proves *which read* writes it.
    """

    def test_the_leading_voice_sustains_or_arpeggiates_by_the_shape(self) -> None:
        """The pad's read: the same instrument holds a bar under a
        sustained shape and steps through the chord under a broken one."""
        pad = _harmony_notes(_PAD)
        arpeggio = _harmony_notes(_BROKEN)
        assert {note.duration_ticks for note in pad} == {_BAR}
        assert {note.tick for note in pad} == {0, _BAR}
        assert {note.duration_ticks for note in arpeggio} == {
            DEFAULT_HARMONY_VOICES.arpeggio_step_ticks
        }
        assert len({note.tick for note in arpeggio}) == 2 * _BAR // (
            DEFAULT_HARMONY_VOICES.arpeggio_step_ticks
        )

    def test_a_stab_instrument_states_accents_only_under_a_broken_chord(self) -> None:
        """The stabs' read of the same field, and the instrument is not what
        decides it: brass under a sustained shape holds its bar like any
        other bed, while under a broken chord it states two pulses to the
        bar instead of eight steps."""
        sustained = _harmony_notes(_PAD, instrument="brass_section")
        stabs = _harmony_notes(_BROKEN, instrument="brass_section")
        assert {note.duration_ticks for note in sustained} == {_BAR}
        assert {note.duration_ticks for note in stabs} == {PPQ}
        assert {note.tick for note in stabs} == {0, _BAR // 2, _BAR, _BAR + _BAR // 2}

    def test_the_voices_under_the_leading_one_sustain_whatever_the_texture(self) -> None:
        """Only the leading layer breaks its chord.

        `pad` is `not broken_chord or layer_index > 0`, so the second
        harmony voice under a broken chord must write exactly what it
        writes under a sustained one — and that is not the leading
        layer's notes, which are a level quieter one layer down.
        """
        assert _harmony_notes(_BROKEN, layer_index=1) == _harmony_notes(_PAD, layer_index=1)
        assert _harmony_notes(_BROKEN, layer_index=1) != _harmony_notes(_PAD)

    def test_the_broken_chord_steps_on_the_interval_the_shape_names(self) -> None:
        """The grid, which the arpeggio's onsets and lengths both read."""
        coarse = _harmony_notes(replace(_BROKEN, arpeggio_step_ticks=4 * PPQ))
        assert {note.tick for note in coarse} == set(range(0, 2 * _BAR, 4 * PPQ))
        assert {note.duration_ticks for note in coarse} == {4 * PPQ}

    @pytest.mark.parametrize(
        ("field", "instrument", "texture"),
        [
            ("pad_velocity", "strings", _PAD),
            ("arpeggio_velocity", "strings", _BROKEN),
            ("stab_velocity", "brass_section", _BROKEN),
        ],
    )
    def test_each_figure_sounds_at_the_level_the_shape_names(
        self, field: str, instrument: str, texture: HarmonyVoices
    ) -> None:
        """A level, not a note.

        Every one of the three velocities moves the figure's loudness and
        leaves its ticks, pitches and lengths exactly where they were —
        which is both what a level knob should do and what says the read
        being tested is the level and not the draw above it.
        """
        default = _harmony_notes(texture, instrument=instrument)
        changed = _harmony_notes(replace(texture, **{field: 100}), instrument=instrument)
        assert _shape_of(changed) == _shape_of(default)
        assert sorted(note.velocity for note in changed) != sorted(
            note.velocity for note in default
        )


def _settled_pitches(
    *, clearance: int, window: MelodyBand, melody_pitch: int
) -> tuple[int, ...]:
    """Where the settle pass puts a bed written at pitch 40 against this tune.

    The bed note is a chord tone two octaves under the tune's middle, so
    every candidate the pass can fold it to is an octave of it — which is
    what makes the two cases below land on different octaves exactly when
    the clearance says they should.
    """
    melody = NoteEvent(
        voice_id=VOICE_MELODY,
        pitch_midi=melody_pitch,
        tick=0,
        duration_ticks=_BAR,
        velocity=70,
    )
    settled = _settle_harmony_register(
        [
            NoteEvent(
                voice_id=VOICE_HARMONY,
                pitch_midi=40,
                tick=0,
                duration_ticks=_BAR,
                velocity=46,
            ),
            melody,
        ],
        melody_floor=melody_pitch,
        melody_ceiling=melody_pitch,
        registers={VOICE_HARMONY: BedRegisters(comfortable=window, compass=window)},
        clearance=clearance,
    )
    return tuple(note.pitch_midi for note in settled if note.voice_id == VOICE_HARMONY)


class TestTheClearanceIsReadOnBothSidesOfTheTune:
    """One case per side, because the parameter is read as two expressions.

    `harmony_melody_clearance` is `melody_floor - clearance` below the tune
    and `melody_ceiling + clearance` above it, and the two are chosen by the
    window rather than both applied: a window with an octave of room under
    the tune goes under it and reads only the first, a window whose range
    starts above the tune — a celesta's — has no room underneath and reads
    only the second. Neither expression can be seen by the other's case, so
    a case apiece is the only way to attribute either.

    Each case is tuned to the octave grid of the bed note, which is the
    difference from the melody layer's threshold knobs: the candidate set is
    every octave of the written pitch, so a clearance that moves the bound
    without crossing an octave moves nothing, and a case that took two such
    values would prove nothing about the read.
    """

    def test_the_melody_band_is_raised_by_the_clearance_the_shape_names(self) -> None:
        """The first of the three reads: the room the bed needs under the
        tune is what raises the tune's own band."""
        narrow = _melody_band_for(melody="piano", bed="brass_section", clearance=3)
        wide = _melody_band_for(melody="piano", bed="brass_section", clearance=6)
        assert wide.low_midi - narrow.low_midi == 3
        assert wide.high_midi - wide.low_midi == narrow.high_midi - narrow.low_midi, (
            "the raise moved the band's width, and the walk needs the twelfth it holds"
        )

    def test_a_bed_under_the_tune_settles_by_the_clearance(self) -> None:
        """The under read, taken as close under the tune as the window allows."""
        window = MelodyBand(low_midi=48, high_midi=84)
        assert _settled_pitches(clearance=3, window=window, melody_pitch=80) == (76,)
        assert _settled_pitches(clearance=6, window=window, melody_pitch=80) == (64,)

    def test_a_bed_over_the_tune_settles_by_the_clearance(self) -> None:
        """The over read, in a window with no octave underneath the tune."""
        window = MelodyBand(low_midi=72, high_midi=96)
        assert _settled_pitches(clearance=3, window=window, melody_pitch=72) == (76,)
        assert _settled_pitches(clearance=6, window=window, melody_pitch=72) == (88,)


_DRUM_KNOBS: dict[str, Any] = {
    # Another kit entirely: a different pattern and a different level, so
    # the note set cannot survive the swap.
    "drum_style_name": "funk",
    # The rock style has two 4/4 variants, so a cycle that names the
    # second one for the second section is a section whose pace changes.
    "rotation_cycle": (0, 1, 0, 0),
    # Half the level, well clear of the rounding: the values are carried
    # into MIDI velocities and rounded, and a tenth of a step can land on
    # the same integer twice.
    "percussion_velocity_scale": 0.5,
    # A quieter crash under the section downbeats.
    "section_crash_velocity": 40,
}

_DRUM_KIT_FIELDS = frozenset(
    {
        "drum_style_name",
        "rotation_cycle",
        "percussion_velocity_scale",
        "section_crash_velocity",
    }
)
"""The plan's `--- Percussion ---` block, which reads these four.

`percussion_rest_section` stays in the sections group and is covered
there, although both blocks are read by the same pass.
"""


class TestTheDrumKitIsLive:
    """The style, the variant rotation, the mood's level and the crash.

    Four module-table reads before B7 — `style_for(mood, …)`,
    `ROTATION_CYCLE`, `MOOD_VELOCITY_SCALE.get(mood)` and
    `SECTION_CRASH_VELOCITY` — and the mood no longer reaches the
    percussion pass at all. Unlike the voices layer, every one of these
    four is read at exactly one site, so the fingerprint class is enough
    to prove the plan reaches the music; what the *site* does with it is
    covered by the direct calls below.
    """

    def test_every_drum_knob_has_a_case(self) -> None:
        assert set(_DRUM_KNOBS) == _DRUM_KIT_FIELDS

    @pytest.mark.parametrize("knob", sorted(_DRUM_KNOBS))
    def test_a_non_default_plan_moves_the_output(self, knob: str) -> None:
        plan = replace(default_plan(_DRUM_KIT_SPEC), **{knob: _DRUM_KNOBS[knob]})
        moved = _fingerprint(compose(_DRUM_KIT_SPEC, plan=plan)) != _fingerprint(
            compose(_DRUM_KIT_SPEC)
        )
        assert moved, (
            f"{knob} is carried by the plan but does not reach the engine: "
            "composing under it produced the same score, performance plan and "
            "arrangement as the default. The seam is dead for this knob."
        )

    def test_the_cases_are_not_all_one_knob_in_disguise(self) -> None:
        base = default_plan(_DRUM_KIT_SPEC)
        for knob, value in _DRUM_KNOBS.items():
            assert getattr(base, knob) != value, knob

    def test_the_spec_reaches_the_kit_this_class_needs(self) -> None:
        """The premise, asserted: a spec that stopped writing drums, or a
        style with one variant, would leave `rotation_cycle`'s case
        asserting nothing — the cycle names the same pattern either way."""
        base = default_plan(_DRUM_KIT_SPEC)
        assert base.drum_style_name == "rock"
        assert len(DRUM_STYLES[base.drum_style_name].variants["4/4"]) > 1, (
            "the kit's rotation is unobservable with one variant"
        )
        notes = [
            note
            for note in compose(_DRUM_KIT_SPEC).notation_score.notes
            if note.voice_id == VOICE_PERCUSSION
        ]
        assert notes, "the spec no longer writes a percussion voice"

    def test_a_style_with_one_variant_leaves_the_cycle_unread(self) -> None:
        """And the honest unobservability of the knob is pinned: the waltz
        and the shuffle carry a single variant per meter, so
        `rotation_index` returns 0 for every section and no cycle can move
        them. A future style with a second 3/4 variant makes this fail,
        which is the right time to notice."""
        for style_name in ("waltz", "shuffle"):
            style = DRUM_STYLES[style_name]
            for meter, variants in style.variants.items():
                assert len(variants) == 1, (style_name, meter)


def _kit_notes(
    kit: DrumKit, *, total_bars: int = 8, form_bars: int = 4, repetition_count: int = 2
) -> list[NoteEvent]:
    """Two sections of a 4/4 bar, written by the percussion pass at this kit.

    Two, because the rotation only says anything from the second section
    on, and the section before the last is where the style's fill is
    played. The pass is called directly for the reason the voices layer's
    cases are: a plan mutation moves the whole fingerprint whichever read
    it reaches.
    """
    return _generate_percussion(
        kit=kit,
        time_signature="4/4",
        form_bars=form_bars,
        repetition_count=repetition_count,
        total_bars=total_bars,
        seed=7,
    )


def _hits(notes: list[NoteEvent]) -> list[tuple[int, int]]:
    """Every hit but the crash, as where and which drum it is."""
    return [
        (note.tick, note.pitch_midi) for note in notes if note.pitch_midi != DRUM_CRASH
    ]


def _crashes(notes: list[NoteEvent]) -> list[tuple[int, int]]:
    """The section downbeats, as where the crash is and how loud."""
    return [
        (note.tick, note.velocity) for note in notes if note.pitch_midi == DRUM_CRASH
    ]


class TestTheDrumWritersReadTheirShape:
    """One case per read, direct, so the value is attributed to its site."""

    def test_the_pass_plays_the_style_it_is_handed(self) -> None:
        """Two styles, two different bar templates — and the difference is
        in the pattern, not only the level, which is what says the style
        object is what writes."""
        rock = _kit_notes(DrumKit(style=DRUM_STYLES["rock"]))
        march = _kit_notes(DrumKit(style=DRUM_STYLES["march"]))
        assert rock
        assert march
        assert _hits(rock) != _hits(march)

    def test_an_unstyled_kit_writes_no_drums(self) -> None:
        """The honest skip, which is why the default kit carries no style:
        a piece with no kit has no percussion voice, and inventing a
        default pattern is the wrong pattern."""
        assert _kit_notes(DEFAULT_DRUM_KIT) == []

    def test_a_style_with_no_template_for_the_meter_writes_nothing(self) -> None:
        """A waltz asked to play 4/4 has no bar to play. The plan can only
        check the name is a style; whether that style covers the meter is
        a fact about the style, so the pass skips the whole piece rather
        than writing a wrong groove."""
        assert _kit_notes(DrumKit(style=DRUM_STYLES["waltz"])) == []
        assert _hits(_kit_notes(DrumKit(style=DRUM_STYLES["rock"])))

    def test_the_rotation_cycle_decides_the_pattern_per_section(self) -> None:
        """The first section reads the cycle's first entry, the second its
        second — so a cycle that changes only the second entry moves the
        second section's hits and leaves the first's exactly where they
        were."""
        default = _kit_notes(DrumKit(style=DRUM_STYLES["rock"]))
        rotated = _kit_notes(
            DrumKit(style=DRUM_STYLES["rock"], rotation_cycle=(0, 1, 0, 0))
        )
        second = 4 * PPQ  # the fifth bar of two four-bar sections
        assert [hit for hit in _hits(default) if hit[0] < second] == [
            hit for hit in _hits(rotated) if hit[0] < second
        ]
        assert [hit for hit in _hits(default) if hit[0] >= second] != [
            hit for hit in _hits(rotated) if hit[0] >= second
        ]

    def test_the_level_scales_every_hit(self) -> None:
        """The mood's scaling, which multiplies the groove and the crash
        alike: the pattern is untouched and the whole kit is quieter."""
        loud = _kit_notes(DrumKit(style=DRUM_STYLES["rock"]))
        soft = _kit_notes(DrumKit(style=DRUM_STYLES["rock"], velocity_scale=0.5))
        assert _hits(soft) == _hits(loud)
        assert _crashes(soft) != _crashes(loud)
        assert max(note.velocity for note in soft) < max(note.velocity for note in loud)

    def test_the_crash_velocity_moves_the_crash_alone(self) -> None:
        """The section marker's own level: every groove hit stays exactly
        where it was, at the velocity it was, and only the downbeat
        changes."""
        default = _kit_notes(DrumKit(style=DRUM_STYLES["rock"]))
        quiet = _kit_notes(DrumKit(style=DRUM_STYLES["rock"], crash_velocity=40))
        assert [(n.tick, n.pitch_midi, n.velocity) for n in quiet] != [
            (n.tick, n.pitch_midi, n.velocity) for n in default
        ]
        assert _hits(quiet) == _hits(default)
        assert _crashes(quiet) != _crashes(default)
        assert min(velocity for _tick, velocity in _crashes(quiet)) < min(
            velocity for _tick, velocity in _crashes(default)
        )


class TestTheCodaBranchReadsThePlan:
    """The fallback arm of the duration search, which nothing covered.

    When no clean (form, repetition, tempo) combination fits the target,
    `arrange_for_duration` tries a coda — a shorter tail of complete
    measures. Coverage showed the branch was never entered by any test, so
    the three knobs it reads (`ritardando_factor`, `intro_bars`,
    `arc_min_reps`) were unexercised, and B2 rewrote all three. Forty
    seconds of `calming` in 4/4 is the shortest spec that reaches it: eight
    bars once, plus a four-bar coda.

    The branch's intro read is reachable only through a plan. No default
    arrangement has both a coda and `repetition_count >= arc_min_reps`,
    because a coda is only needed where few repetitions fit — so the arm
    that hands the intro over is exercised by a lowered `arc_min_reps` and
    by nothing else.
    """

    _CODA_SPEC = CompositionSpec(mood="calming", duration_seconds=40, seed=3)

    def test_a_spec_that_reaches_the_coda_branch_really_does(self) -> None:
        """The premise, asserted rather than assumed: if the default search
        stopped reaching for a coda, every test below would still pass while
        covering nothing."""
        arrangement = compose(self._CODA_SPEC).arrangement
        assert arrangement.coda_bars > 0
        assert arrangement.ritardando_factor < 1.0

    def test_the_coda_slows_at_the_plans_factor(self) -> None:
        output = compose(self._CODA_SPEC)
        changes = output.notation_score.tempo.changes
        assert len(changes) == 1
        assert changes[0].bpm == round(
            output.arrangement.tempo_bpm * output.arrangement.ritardando_factor, 1
        )

    def test_the_coda_hands_over_an_intro_only_when_the_plan_says_so(self) -> None:
        """`intro_bars` inside the coda arm — the read no default reaches."""
        spec = CompositionSpec(
            mood=self._CODA_SPEC.mood.value,
            duration_seconds=self._CODA_SPEC.duration_seconds + 1,
            seed=self._CODA_SPEC.seed,
        )
        relaxed = replace(default_plan(spec), arc_min_reps=1)
        output = compose(spec, plan=relaxed)
        assert output.arrangement.coda_bars > 0, "the coda arm was not reached"
        assert output.arrangement.intro_bars == relaxed.intro_bars

    def test_a_slowdown_the_coda_cannot_pay_for_is_refused(self) -> None:
        """The coda arm's `ritardando_factor` read reaches the duration math.

        A deeper slowdown costs the coda more seconds than the ±2% promise
        can absorb, and no tempo in the mood's range can pay for it — so
        the plan is refused rather than honoured with a piece that misses
        the duration it was asked for.
        """
        plan = replace(default_plan(self._CODA_SPEC), ritardando_factor=0.6)
        with pytest.raises(CompositionEngineError) as caught:
            compose(self._CODA_SPEC, plan=plan)
        assert caught.value.code is EngineErrorCode.DURATION_UNFULFILLABLE


class TestTheImplicitDefaultIsTheExplicitDefault:
    """`plan=None` and `plan=default_plan(spec)` are the same piece.

    The corpus pins the first against a value recorded earlier; this pins
    the two against each other, which is what a caller who wants to
    mention the plan explicitly is entitled to assume.
    """

    @pytest.mark.parametrize(
        "spec",
        [
            CompositionSpec(mood=mood, duration_seconds=duration, seed=seed, key=key)
            for mood in ("calming", "electrifying", "sleep")
            for duration in (30, 120, 600)
            for seed in (0, 7)
            for key in (None, "C", "Am")
        ],
    )
    def test_the_two_compositions_agree_exactly(self, spec: CompositionSpec) -> None:
        implicit = compose(spec)
        explicit = compose(spec, plan=default_plan(spec))
        assert _fingerprint(implicit) == _fingerprint(explicit)

    def test_the_default_plan_is_a_plan_the_engine_can_honour(self) -> None:
        """A default that named an unhonourable value would make every
        composition raise, which the tests above would report as a moved
        fingerprint rather than as the defect it is."""
        plan = default_plan(_SPEC)
        assert isinstance(plan, CompositionPlan)
        assert plan.format == f"CompositionPlan:{PLAN_SCHEMA_VERSION}"
        assert plan.arrangement_knobs() == DEFAULT_ARRANGEMENT_KNOBS


class TestAPlanTheEngineCannotHonourRefuses:
    """The product rule: refused with a reason, never silently downgraded.

    A plan arrives from outside the engine, so the failure has to be one
    the calling layer can catch and act on.
    """

    def test_a_tolerance_nothing_can_meet_raises_a_typed_error(self) -> None:
        """The knob really does reach the search's tolerance read: a
        tolerance no arrangement can satisfy makes the piece
        unfulfillable rather than quietly loosened back to the default."""
        plan = replace(default_plan(_SPEC), duration_tolerance=1e-9)
        with pytest.raises(CompositionEngineError) as caught:
            compose(_SPEC, plan=plan)
        assert caught.value.code is EngineErrorCode.DURATION_UNFULFILLABLE

    def test_narrowing_the_forms_to_one_too_small_raises(self) -> None:
        """Refused for the reason §10 #1 gives, not by returning a piece
        short of the target."""
        plan = replace(default_plan(_SPEC), form_sizes=(8,), max_repeats=1)
        with pytest.raises(CompositionEngineError) as caught:
            compose(_SPEC, plan=plan)
        assert caught.value.code is EngineErrorCode.DURATION_UNFULFILLABLE
        assert "no Phase 1 form can reach" in str(caught.value) or "no (form" in str(
            caught.value
        )

    def test_an_intro_no_section_could_carve_never_reaches_the_engine(self) -> None:
        """The plan refuses it at construction, where both numbers are
        known — rather than letting it surface as a bare `ValueError`
        from inside the duration search, which is what it did before the
        invariant was added."""
        with pytest.raises(PlanError, match="shorter than the shortest form"):
            replace(default_plan(_SPEC), form_sizes=(8,), intro_bars=16)


class TestAStoredPlanReplaysAfterTheTablesMove:
    """The claim the whole materialized-plan rule exists to make true.

    A plan that carried overrides would make `(plan, seed) -> notes` a
    claim about `motif.py` rather than about the artifact: edit a table and
    every stored plan replays differently while the manifest still promised
    reproducibility. So the plan is stored complete, and this is the test
    that says so — the table really moves, the default path really moves
    with it, and the stored plan does not.
    """

    def test_a_module_table_moving_does_not_touch_a_stored_plan(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        spec = _SPEC
        stored = default_plan(spec)
        # What a caller persists: the canonical document, not the object.
        persisted = canonical_dumps(stored.to_canonical_dict())
        before = _fingerprint(compose(spec, plan=stored))

        # The mood's bass vocabulary, replaced by a pedal. Mutating the
        # dict rather than rebinding the name because `plan.py` imports
        # `BASS_FIGURES` by value once — the dict is the thing both sides
        # read, which is exactly the coupling this test is about.
        monkeypatch.setitem(BASS_FIGURES, "calming", (((0, 16, 0),),))

        # The premise, asserted rather than assumed: if the move changed
        # nothing, the rest of this test would pass for the wrong reason.
        assert default_plan(spec) != stored, "the table move did not reach the default plan"

        reloaded = CompositionPlan.from_canonical_dict(json.loads(persisted))
        after = _fingerprint(compose(spec, plan=reloaded))
        assert after == before, "a stored plan replays the module tables it was meant to replace"

    def test_the_default_path_does_move_with_the_table(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The other side of the same coin, and the plan's stated limit.

        `default_plan` is a *description* of the current engine — it reads
        the tables — so a caller who passes no plan is asking for whatever
        the build does today. That is what `plan=None` is for, and it is
        why a persisted plan has to be materialized rather than resolved.
        """
        spec = _SPEC
        before = _fingerprint(compose(spec))
        monkeypatch.setitem(BASS_FIGURES, "calming", (((0, 16, 0),),))
        assert _fingerprint(compose(spec)) != before


_LAYER_INVENTORIES: dict[str, frozenset[str]] = {
    "arrangement": frozenset(_ARRANGEMENT_KNOBS),
    "harmony": _HARMONY_FIELDS,
    "sections": _SECTION_FIELDS,
    "melody": _MELODY_FIELDS,
    "voices": _VOICES_FIELDS,
    "percussion": _DRUM_KIT_FIELDS,
}
"""Every layer's field set, by name, because a class cannot be asked.

B5-B7 each spell their layer's fields out rather than deriving them, and
B4's finding is why this mapping exists at all: a class like
`TestTheVelocityTerracesAreReadAtEveryCallSite` deliberately owns *no*
field set, so walking the module's classes and guessing which attribute is
an inventory would either misread one or need a special case per exception.
Naming them here makes the inventory a thing the check can count.
"""


def _uncovered_plan_fields(inventories: Mapping[str, frozenset[str]]) -> list[str]:
    """The plan's own knobs that no listed layer claims. Empty is healthy.

    A function rather than three lines inside the test, so the check can be
    fired on a deliberately incomplete inventory: an assertion about a
    partition that has never been seen to fail is a comment.
    """
    declared = {field.name for field in fields(CompositionPlan)} - {"format"}
    return sorted(declared - set().union(*inventories.values()))


class TestEveryPlanFieldIsCoveredByASeamCase:
    """The union check the seam's own header promised.

    Each layer knows its own fields, and nothing else does — there is no
    struct for the melody, the sections or the drums to enumerate against
    the way `ArrangementKnobs` enumerates the arrangement's. So a knob added
    to `CompositionPlan` without a seam case would go unnoticed: the layer
    tests all pass, the field exists, and no composition ever reads it.
    That is a dead seam found at Phase C rather than here, which is exactly
    what this module exists to prevent.
    """

    def test_the_layer_field_sets_name_every_field_of_the_plan(self) -> None:
        assert _uncovered_plan_fields(_LAYER_INVENTORIES) == []

    def test_the_field_sets_name_nothing_the_plan_does_not_have(self) -> None:
        """The other direction: a renamed knob leaves the old name behind
        in some layer's set, where it would keep the union looking complete
        while the new name went uncovered."""
        declared = {field.name for field in fields(CompositionPlan)} - {"format"}
        extra = sorted(set().union(*_LAYER_INVENTORIES.values()) - declared)
        assert not extra, (
            f"the layer field sets name {extra}, which CompositionPlan does not have"
        )

    def test_no_field_is_claimed_by_two_layers(self) -> None:
        """Disjoint, not merely covering.

        A field in two sets is a field two layers both believe they own —
        and when one of them is rewritten the other's case keeps passing,
        which is the attribution failure B2-B6 kept finding in other
        clothing. `format` is excluded from all of this: it is the
        document's tag, not a knob any layer reads.
        """
        seen: dict[str, str] = {}
        for layer, names in _LAYER_INVENTORIES.items():
            for name in names:
                assert name not in seen, (
                    f"{name} is claimed by both {seen[name]} and {layer}; a layer "
                    "field set is a partition of the plan's knobs"
                )
                seen[name] = layer

    def test_the_check_notices_an_uncovered_field(self) -> None:
        """Fired, with one layer's set withheld.

        The melody's ten knobs go uncovered, so the check has to name all
        ten — and the count is asserted as well as the membership, because
        a check that reported *some* uncovered field would be no use to
        whoever has to add the missing seam case.
        """
        without_melody = {k: v for k, v in _LAYER_INVENTORIES.items() if k != "melody"}
        assert _uncovered_plan_fields(without_melody) == sorted(_MELODY_FIELDS)
