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

So each arrangement knob is composed twice, once at its default and once
mutated, and the whole output — score, performance plan and the
arrangement the sidecar records — has to move. A knob the engine reads
from a module table instead of from the plan moves nothing and fails here.
"""

from __future__ import annotations

from dataclasses import fields, replace
from typing import Any

import pytest

from saimc.canonical import canonical_sha256
from saimc.compose.duration import DEFAULT_ARRANGEMENT_KNOBS, ArrangementKnobs
from saimc.compose.engine import CompositionEngineError, EngineErrorCode, EngineOutput, compose
from saimc.compose.plan import (
    PLAN_SCHEMA_VERSION,
    CompositionPlan,
    PlanError,
    default_plan,
)
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
