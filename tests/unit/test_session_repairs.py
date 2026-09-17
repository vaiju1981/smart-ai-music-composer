"""The repair loop: what it moves, what it refuses, and what it must never do.

The table it works from is measured rather than reasoned (see the module
docstring), so the cases here are pieces of music rather than illustrations of a
rule, and every constant in them came from a run. Four properties carry the
weight.

- **Smaller is better, and the fixture where that is decidable is the point of
  the file.** `musical_order` counts what is *wrong* with a piece, so a repair
  that makes a piece worse moves the tuple up. `_try` reads `order < floor`, and
  a piece exists in which the table's two candidates pull opposite ways —
  `electrifying/30s/16`, where narrowing the band fixes the piece and halving
  the motif variation makes it worse. A loop with the comparison inverted keeps
  the second one, which `revise_draft`'s ratchet would then refuse as a
  regression: the candidate that helps discarded and the one that hurts kept.
  That is not a near miss, and it is why the direction gets a case rather than a
  comment.
- **It never makes a piece worse, on the arbiter's own key.** Every kept chain
  is asserted to have moved the order down — measured through the engine, not
  read off the loop's own numbers.
- **Empty is an answer with three different reasons, and they are told apart.**
  Nothing to repair, a bar with no request in the table, and a bar whose
  requests were tried and kept none are three states a caller has to explain
  differently, so each is reached by a piece.
- **The bound is a backstop rather than a wall.** At `maximum=1` the loop stops
  with bars still missing and `stopped_by` is `None`, because stopping for
  budget is not stopping because nothing works — the first is the caller's
  choice, the second is a fact about the music.

The pieces are 30 to 120 seconds long and composed for real; the loop is
arithmetic over the engine, so nothing here is faked.
"""

from __future__ import annotations

import pytest

from saimc.compose.engine import CompositionEngineError, compose
from saimc.quality import PieceQuality, score_piece
from saimc.session.arbiter import METRIC_TIERS, musical_order, worst_finding
from saimc.session.deltas import (
    Delta,
    SetAccompanimentDensity,
    SetHarmonyClearance,
    SetMelodyBand,
    SetMotifVariation,
    apply_deltas,
)
from saimc.session.repairs import _REPAIRS, Attempt, Repair, _Trial, _try, repair_chain
from saimc.spec import CompositionSpec, Mood

_CALM = CompositionSpec(mood=Mood.CALMING, duration_seconds=30, seed=0)
"""Clean before anything runs — found by scanning a corpus rather than by luck."""
_ELECTRIC_30_0 = CompositionSpec(mood=Mood.ELECTRIFYING, duration_seconds=30, seed=0)
"""The bed's two candidates tie here, both reaching a clean piece."""
_ELECTRIC_120_1 = CompositionSpec(mood=Mood.ELECTRIFYING, duration_seconds=120, seed=1)
"""Three bars in report order, and clearing the tune's exposes the bed's."""
_ELECTRIC_90_17 = CompositionSpec(mood=Mood.ELECTRIFYING, duration_seconds=90, seed=17)
"""The engine refuses the leap bar's first request here.

Found by sweeping the vocabulary's magnitudes over a grid rather than by
reusing the piece this role had before `bass_root_motion`: the old
`electrifying/120s/1` no longer produces an unplayable score at band 9,
and a fixture that stopped raising would leave the trial above passing as
`kept`. Band 5 still refuses that piece, which is how the sweep says the
outcome is alive rather than that the arm is dead.
"""
_ELECTRIC_30_16 = CompositionSpec(mood=Mood.ELECTRIFYING, duration_seconds=30, seed=16)
"""The two candidates pull strictly opposite ways, which the old one no longer did.

Narrowing the band moves the order from `(0, 3, 1.385, 9)` to
`(0, 2, 1.169, 4)`; halving the motif variation moves it to
`(0, 3, 1.643, 9)` — past the piece it started from. A sweep of 1200
specs found eight such pieces and this is the shortest, so the case costs
a 30-second compose rather than a 300-second one.
"""
_ELECTRIC_30_2 = CompositionSpec(mood=Mood.ELECTRIFYING, duration_seconds=30, seed=2)
"""The piece on which the report's order and the arbiter's disagree."""
_SLEEP_30_3 = CompositionSpec(mood=Mood.SLEEP, duration_seconds=30, seed=3)
"""The refusal: both of the leap-recovery bar's requests measure no better."""
_CLEARANCE = SetHarmonyClearance(semitones=24)
"""A prefix, because production folds the draft's own chain as one.

Wide enough to push the bed under the tune, which is the one way a sweep of
5182 pieces found to reach a bar the table holds no request for through the
delta vocabulary alone.
"""
_TINY_BAND = SetMelodyBand(semitones=1)
"""A prefix, and the reason the aim is a *choice* rather than a reading.

`findings()` lists a piece's misses in `QUALITY_THRESHOLDS`' order and the
arbiter ranks them in `METRIC_TIERS`': the two tables are not the same order, and
they differ on exactly the pair this prefix breaches together — the report says
`range_semitones` first, the arbiter says `max_leap_semitones`. Every other
breaching set reachable from a bare spec happens to sort the same way in both, so
without this piece a loop aiming at `findings[0]` would pass.
"""


def _quality(spec: CompositionSpec, chain: tuple[Delta, ...] = ()) -> PieceQuality:
    """Score the piece `chain` composes to — the draft's own scorecard."""
    application = apply_deltas(spec, list(chain))
    assert not application.refused, application.refused
    return score_piece(compose(application.spec, plan=application.plan).notation_score)


def _repair(spec: CompositionSpec, chain: tuple[Delta, ...] = (), **kw) -> Repair:
    return repair_chain(spec, chain, _quality(spec, chain), **kw)


def _after(spec: CompositionSpec, chain: tuple[Delta, ...], outcome: Repair) -> PieceQuality:
    """Score what the kept chain composes to, on top of the draft's own chain."""
    return _quality(spec, (*chain, *outcome.added))


class TestTheTable:
    def test_every_entry_is_a_request_that_writes_the_plan_and_not_the_spec(self) -> None:
        """A repair is the product's arithmetic over the piece, never the user's
        brief: a request that rewrote the spec would change what was asked for
        while claiming to fix how it came out."""
        for metric, candidates in _REPAIRS.items():
            for candidate in candidates:
                assert candidate.spec_changes(_CALM) == {}, (metric, candidate)

    def test_every_entry_is_a_request_the_applier_honours(self) -> None:
        """A candidate the applier refuses measures nothing, so an entry it
        refused could only ever come back `refused`."""
        for metric, candidates in _REPAIRS.items():
            application = apply_deltas(_CALM, list(candidates))
            assert not application.refused, (metric, application.refused)

    def test_the_table_names_bars_the_arbiter_ranks(self) -> None:
        """A ratchet against the typo this module cannot otherwise catch: a
        metric spelled wrong would be an entry nothing ever looks up."""
        assert set(_REPAIRS) <= set(METRIC_TIERS), set(_REPAIRS) - set(METRIC_TIERS)

    def test_every_entry_is_a_bar_a_piece_in_this_file_breaches(self) -> None:
        """A ratchet on the table's own premise: each key was chosen because a
        real piece breached it, and each must still have one here — otherwise a
        key's own case has no music behind it."""
        breaching: set[str] = set()
        for spec in (_ELECTRIC_30_0, _ELECTRIC_120_1, _SLEEP_30_3):
            breaching |= {finding.metric for finding in _quality(spec).findings()}
        assert set(_REPAIRS) <= breaching, set(_REPAIRS) - breaching

    def test_the_table_is_two_wide(self) -> None:
        """The width is a measurement — a third candidate never changed a
        verdict — so a later phase widening it has to say so."""
        assert {metric: len(candidates) for metric, candidates in _REPAIRS.items()} == {
            "harmony_pad_coverage": 2,
            "texture_hierarchy": 2,
            "max_leap_semitones": 2,
            "leap_recovery_ratio": 2,
        }


class TestTheCleanDraft:
    def test_the_fixture_misses_nothing_before_anything_runs(self) -> None:
        assert _quality(_CALM).findings() == ()

    def test_a_piece_that_misses_nothing_is_left_alone(self) -> None:
        outcome = _repair(_CALM, maximum=4)
        assert outcome.added == ()
        assert outcome.remaining == ()
        assert outcome.stopped_by is None

    def test_it_does_not_even_try(self) -> None:
        """No findings means no candidate composed, which is what makes the
        clean case free rather than a loop that runs and keeps nothing."""
        assert _repair(_CALM, maximum=4).attempts == ()


class TestOneKeptRequest:
    def test_the_kept_chain_clears_the_piece(self) -> None:
        outcome = _repair(_ELECTRIC_30_0, maximum=4)
        assert [delta.describe() for delta in outcome.added] == [
            "SetAccompanimentDensity(step_ticks=1440)"
        ]
        assert _after(_ELECTRIC_30_0, (), outcome).findings() == ()

    def test_a_repair_never_makes_the_piece_worse(self) -> None:
        outcome = _repair(_ELECTRIC_30_0, maximum=4)
        assert musical_order(_after(_ELECTRIC_30_0, (), outcome), lint_passed=True) < (
            musical_order(_quality(_ELECTRIC_30_0), lint_passed=True)
        )

    def test_the_kept_request_is_the_better_of_two_that_tie(self) -> None:
        """Both of the bed's requests reach a clean piece here — measured, not
        assumed — so which one is kept is decided by the table's order, which is
        what makes a tie deterministic rather than incidental."""
        assert {finding.metric for finding in _quality(_ELECTRIC_30_0).findings()} == {
            "texture_hierarchy",
            "harmony_pad_coverage",
        }
        reached = {
            candidate.describe(): musical_order(
                _quality(_ELECTRIC_30_0, (candidate,)), lint_passed=True
            )
            for candidate in _REPAIRS["texture_hierarchy"]
        }
        assert reached == {
            "SetAccompanimentDensity(step_ticks=1440)": (0, 0, 0, 0),
            "SetHarmonyTexture(broken_chord=False)": (0, 0, 0, 0),
        }, reached
        assert _repair(_ELECTRIC_30_0, maximum=4).added == (
            SetAccompanimentDensity(step_ticks=1440),
        )

    def test_both_candidates_are_recorded_and_not_only_the_kept_one(self) -> None:
        """The attempts are the search record: a reader has to be able to see
        what was tried and rejected, or "it repaired" is a property of the loop
        rather than of the piece."""
        outcome = _repair(_ELECTRIC_30_0, maximum=4)
        assert [(a.metric, a.outcome) for a in outcome.attempts] == [
            ("texture_hierarchy", "kept"),
            ("texture_hierarchy", "kept"),
        ]
        assert [a.delta.describe() for a in outcome.attempts] == [
            "SetAccompanimentDensity(step_ticks=1440)",
            "SetHarmonyTexture(broken_chord=False)",
        ]


class TestTheDirection:
    """The case the file exists for: one candidate helps and one hurts.

    Inverted, `_try` keeps what makes the piece worse and refuses what makes it
    better. `electrifying/30s/16` is the piece where both are in the same round,
    so the chain the loop keeps names which reading it took.
    """

    def test_the_two_candidates_pull_opposite_ways_premise(self) -> None:
        floor = musical_order(_quality(_ELECTRIC_30_16), lint_passed=True)
        assert worst_finding(_quality(_ELECTRIC_30_16).findings()).metric == "max_leap_semitones"
        better, worse = _REPAIRS["max_leap_semitones"]
        assert musical_order(_quality(_ELECTRIC_30_16, (better,)), lint_passed=True) < floor
        assert musical_order(_quality(_ELECTRIC_30_16, (worse,)), lint_passed=True) >= floor

    def test_the_loop_keeps_the_one_that_moves_the_order_down(self) -> None:
        outcome = _repair(_ELECTRIC_30_16, maximum=1)
        assert outcome.added == (SetMelodyBand(semitones=9),)
        assert [attempt.outcome for attempt in outcome.attempts] == ["kept", "no_gain"]

    def test_a_trial_that_ties_is_not_kept(self) -> None:
        """Strictly better, not better-or-equal: this candidate produces a
        piece whose order is *identical* to the one in hand, and keeping it
        would spin the loop until the bound stopped it."""
        quality = _quality(_SLEEP_30_3)
        trial = _try(
            _SLEEP_30_3,
            (),
            _REPAIRS["leap_recovery_ratio"][0],
            aimed_at="leap_recovery_ratio",
            floor=musical_order(quality, lint_passed=True),
            remaining=("leap_recovery_ratio",),
        )
        assert isinstance(trial, _Trial)
        assert trial.order == musical_order(quality, lint_passed=True)
        assert trial.attempt.outcome == "no_gain"


class TestTheUnplayableTrial:
    def test_the_engine_refuses_this_candidate_premise(self) -> None:
        """Without this the case below is evidence of nothing: if the engine
        stopped raising, `unplayable` would go on passing as `kept`."""
        candidate = _REPAIRS["max_leap_semitones"][0]
        application = apply_deltas(_ELECTRIC_90_17, [candidate])
        assert application.applied == (candidate,)
        with pytest.raises(CompositionEngineError):
            compose(application.spec, plan=application.plan)

    def test_a_candidate_the_engine_refuses_is_recorded_and_skipped(self) -> None:
        """A loop that let the engine's refusal propagate would turn a candidate
        it cannot use into a failed turn."""
        outcome = _repair(_ELECTRIC_90_17, maximum=4)
        assert [(a.metric, a.outcome) for a in outcome.attempts][:2] == [
            ("max_leap_semitones", "unplayable"),
            ("max_leap_semitones", "kept"),
        ]
        assert outcome.added[0] == SetMotifVariation(factor=0.5)
        assert _after(_ELECTRIC_90_17, (), outcome).findings() == ()


class TestTheTwoRoundChain:
    def test_a_second_round_aims_at_what_the_first_one_exposed(self) -> None:
        """Clearing the tune's leap bar leaves the bed's bars missed: the engine
        wrote a different tune under a different band. The loop re-aims because
        of this, and the second round is a second bar rather than a second try."""
        outcome = _repair(_ELECTRIC_120_1, maximum=4)
        assert [attempt.metric for attempt in outcome.attempts] == [
            "max_leap_semitones",
            "max_leap_semitones",
            "texture_hierarchy",
            "texture_hierarchy",
        ]
        assert [delta.describe() for delta in outcome.added] == [
            "SetMelodyBand(semitones=9)",
            "SetAccompanimentDensity(step_ticks=1440)",
        ]
        assert _after(_ELECTRIC_120_1, (), outcome).findings() == ()

    def test_the_fixture_has_three_bars_and_clears_them_all(self) -> None:
        assert [finding.metric for finding in _quality(_ELECTRIC_120_1).findings()] == [
            "max_leap_semitones",
            "texture_hierarchy",
            "harmony_pad_coverage",
        ]


class TestTheRefusals:
    def test_a_bar_with_no_request_in_the_table_is_the_one_nothing_was_tried_on(self) -> None:
        """Reached the way production reaches it — a draft whose chain already
        carries a request — and the outcome says "nothing was tried" rather than
        "everything was tried and nothing worked"."""
        outcome = _repair(_CALM, (_CLEARANCE,), maximum=4)
        assert outcome.attempts == (
            Attempt(
                metric="register_separation_semitones",
                delta=None,
                outcome="unmapped",
                remaining=("register_separation_semitones",),
            ),
        )
        assert outcome.added == ()
        assert outcome.remaining == ("register_separation_semitones",)
        assert outcome.stopped_by is not None
        assert outcome.stopped_by.metric == "register_separation_semitones"

    def test_that_bar_has_no_entry_and_that_is_the_whole_of_the_refusal(self) -> None:
        """The premise the case above rests on, asserted rather than believed:
        the refusal is about a gap in the vocabulary, so filling the gap has to
        fail a test rather than quietly change what the refusal means."""
        assert "register_separation_semitones" not in _REPAIRS
        assert len(_REPAIRS) < len(METRIC_TIERS)

    def test_a_bar_whose_requests_all_measured_no_better_is_a_measurement(self) -> None:
        outcome = _repair(_SLEEP_30_3, maximum=4)
        assert [(a.metric, a.outcome) for a in outcome.attempts] == [
            ("leap_recovery_ratio", "no_gain"),
            ("leap_recovery_ratio", "no_gain"),
        ]
        assert outcome.added == ()
        assert outcome.stopped_by is not None
        assert outcome.stopped_by.metric == "leap_recovery_ratio"

    def test_a_no_gain_trial_left_the_bar_it_aimed_at_missed(self) -> None:
        """The measurement the refusal's wording rests on: every `no_gain` in
        the corpus left its own bar missed, so nothing tells a user that a bar
        was cleared and the piece lost anyway."""
        for attempt in _repair(_SLEEP_30_3, maximum=4).attempts:
            assert attempt.metric in attempt.remaining

    def test_a_trial_records_what_its_own_piece_left_not_the_one_in_hand(self) -> None:
        """A trial's `remaining` is read off the piece the candidate *produced*,
        so it can be shorter than the misses it started from — and that is the
        whole value of the record: a reader given only the piece in hand's misses
        cannot tell a trial that cleared the bar it aimed at from one that did
        not. Both of this round's candidates clear the leap bar and still miss
        the bed's, which no summary of the parent could say."""
        quality = _quality(_ELECTRIC_120_1)
        assert "max_leap_semitones" in (in_hand := tuple(f.metric for f in quality.findings()))
        outcome = _repair(_ELECTRIC_120_1, maximum=4)
        assert {attempt.remaining for attempt in outcome.attempts[:2]} == {
            ("texture_hierarchy", "harmony_pad_coverage")
        }, in_hand
        assert in_hand not in {attempt.remaining for attempt in outcome.attempts}


class TestTheBound:
    def test_the_bound_stops_the_loop_without_calling_it_a_wall(self) -> None:
        """`stopped_by` is `None` at the bound on purpose: stopping because the
        caller allowed one move is not stopping because nothing works, and the
        two carry different advice."""
        outcome = _repair(_ELECTRIC_120_1, maximum=1)
        assert outcome.added == (SetMelodyBand(semitones=9),)
        assert outcome.remaining == ("texture_hierarchy", "harmony_pad_coverage")
        assert outcome.stopped_by is None

    def test_the_second_move_is_the_one_that_finishes_it(self) -> None:
        one = _repair(_ELECTRIC_120_1, maximum=1)
        two = _repair(_ELECTRIC_120_1, maximum=2)
        assert two.added[:1] == one.added
        assert two.remaining == ()
        assert musical_order(_after(_ELECTRIC_120_1, (), two), lint_passed=True) < (
            musical_order(_after(_ELECTRIC_120_1, (), one), lint_passed=True)
        )

    def test_the_bound_is_a_backstop_and_the_corpus_never_needs_it(self) -> None:
        """Measured: the widest chain any corpus piece needed was two kept
        requests, which is why the budget's default is four rather than two."""
        for spec in (_ELECTRIC_120_1, _ELECTRIC_90_17, _ELECTRIC_30_16, _SLEEP_30_3):
            assert len(_repair(spec, maximum=4).added) <= 2, spec


class TestDeterminism:
    def test_the_same_piece_repairs_the_same_way_twice(self) -> None:
        """No model, no audio, and no dependence on iteration order: the two
        `Repair`s are equal, which is what makes a recorded turn replayable."""
        first = _repair(_ELECTRIC_120_1, maximum=4)
        second = _repair(_ELECTRIC_120_1, maximum=4)
        assert first == second
        assert list(first.added) == list(second.added)

    def test_the_candidates_are_tried_in_the_tables_order(self) -> None:
        """A tie goes to the table rather than to whoever happens to sort first,
        so the order the `dict` states is contract."""
        assert [a.delta for a in _repair(_ELECTRIC_120_1, maximum=1).attempts] == list(
            _REPAIRS["max_leap_semitones"]
        )


class TestTheAim:
    def test_the_report_order_and_the_arbiters_are_not_the_same_order(self) -> None:
        """The premise the case below rests on, asserted rather than believed:
        the two tables really do disagree, and on a pair a piece reaches. If
        either table is re-ordered so they agree, this fails and the case that
        follows stops claiming anything."""
        findings = _quality(_ELECTRIC_30_2, (_TINY_BAND,)).findings()
        assert [finding.metric for finding in findings][:2] == [
            "range_semitones",
            "max_leap_semitones",
        ]
        assert worst_finding(findings).metric == "max_leap_semitones"

    def test_every_round_aims_at_the_arbiters_worst_bar(self) -> None:
        """The loop has no taste of its own: which bar it repairs is the
        arbiter's tier order's decision, so the first attempt of every case here
        names `worst_finding` of the findings it was given.

        `_ELECTRIC_30_2` under a one-semitone band is the case that makes this a
        test rather than a restatement: it is the only piece here whose report
        order differs from the arbiter's, and a loop reading the report's first
        miss would aim at `range_semitones` — a bar the table holds no request
        for — instead of at the leap bar above it.
        """
        for spec, chain in (
            (_ELECTRIC_30_0, ()),
            (_ELECTRIC_120_1, ()),
            (_ELECTRIC_30_16, ()),
            (_SLEEP_30_3, ()),
            (_CALM, (_CLEARANCE,)),
            (_ELECTRIC_30_2, (_TINY_BAND,)),
        ):
            quality = _quality(spec, chain)
            expected = worst_finding(quality.findings())
            assert expected is not None, spec
            assert _repair(spec, chain, maximum=4).attempts[0].metric == expected.metric, spec

    def test_a_kept_chain_leaves_the_ratchet_nothing_to_refuse(self) -> None:
        """Every kept chain has already moved the arbiter's order down — the
        same comparison the ratchet makes — so no repair this loop returns can
        be refused as a regression by the tool that applies it."""
        for spec in (_ELECTRIC_30_0, _ELECTRIC_120_1, _ELECTRIC_90_17, _ELECTRIC_30_16):
            outcome = _repair(spec, maximum=4)
            assert musical_order(_after(spec, (), outcome), lint_passed=True) <= (
                musical_order(_quality(spec), lint_passed=True)
            ), spec
