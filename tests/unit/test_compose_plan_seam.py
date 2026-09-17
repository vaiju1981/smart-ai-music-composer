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

from dataclasses import fields, replace
from typing import Any

import pytest

from saimc.canonical import canonical_sha256
from saimc.compose.duration import (
    DEFAULT_ARRANGEMENT_KNOBS,
    ArrangementKnobs,
    bar_ticks,
)
from saimc.compose.engine import CompositionEngineError, EngineErrorCode, EngineOutput, compose
from saimc.compose.motif import BASS_FIGURES
from saimc.compose.plan import (
    PLAN_SCHEMA_VERSION,
    CompositionPlan,
    PlanError,
    default_plan,
)
from saimc.compose.score import VOICE_BASS
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
}

_HARMONY_FIELDS = frozenset(
    {"bass_figures", "cadence_degree", "cadence_seventh", "modulation_offset"}
)
"""The plan's `--- Harmony ---` block, which reads these four.

Spelled out because, unlike the arrangement, this layer has no struct to
enumerate — the comparison below is what keeps this set honest against
the plan's own field list until B9's union check makes it mechanical.
"""


class TestTheHarmonyLayerIsLive:
    """The chords, the cadence, the bass vocabulary and the lift.

    `bass_figures` reaches the notes through `draw_bass_figures`; the two
    cadence fields through `apply_final_cadence`; `modulation_offset`
    through the `key_offset` the section is transposed by. All four were
    module-table or inline-mood reads before B3.
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
    arm are reachable only through a lowered `arc_min_reps` — here 41 s of
    `calming`, which arranges to eight bars once plus a four-bar coda, so
    the coda is the piece's final four bars.
    """

    _CODA_SPEC = CompositionSpec(mood="calming", duration_seconds=41, seed=3)
    _LONG_WITH_CODA = replace(default_plan(_CODA_SPEC), arc_min_reps=1)

    def _compose(self, **changes: Any) -> EngineOutput:
        return compose(self._CODA_SPEC, plan=replace(self._LONG_WITH_CODA, **changes))

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
        assert arrangement.repetition_count >= self._LONG_WITH_CODA.arc_min_reps, (
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
        ticks = bar_ticks(self._CODA_SPEC.time_signature.value)

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
