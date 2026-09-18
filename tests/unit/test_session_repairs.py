"""The repair loop: what it moves, what it refuses, and what it must never do.

The table it works from is measured rather than reasoned (see the module
docstring), so the cases here are pieces of music rather than illustrations of a
rule, and every constant in them came from a run. Four properties carry the
weight.

- **Smaller is better, and the fixture where that is decidable is the point of
  the file.** `musical_order` counts what is *wrong* with a piece, so a repair
  that makes a piece worse moves the tuple up. `_try` reads `order < floor`, and
  a piece exists in which the table's two candidates pull opposite ways —
  `calming/60s/13`, where narrowing the band fixes the piece and halving the
  motif variation makes it worse. A loop with the comparison inverted keeps
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
  differently, so each is reached by a piece. The first of the three is itself
  two cases since Phase F4, and they are not interchangeable: a bar whose
  request was never measured (`register_separation_semitones`), and a bar for
  which none *exists* because the delta vocabulary cannot write what it reads
  (`step_ratio` and `leap_ratio`, the melody's rates of stepping and leaping).
  Both take the same outcome, and the sentences behind them are different facts.
- **The bound is a backstop rather than a wall.** At `maximum=1` the loop stops
  with bars still missing and `stopped_by` is `None`, because stopping for
  budget is not stopping because nothing works — the first is the caller's
  choice, the second is a fact about the music.

The pieces are 30 to 120 seconds long and composed for real; the loop is
arithmetic over the engine, so nothing here is faked.

Every spec below names its key, and that is what keeps the constants in its
docstring the measurement it claims to be. Each piece was found by sweeping a
corpus, so each is a fact about one run; a spec that left `key` unset used to
compose in C major unconditionally and now walks the mood's own pool, which
would quietly re-point every fixture here at different music while the prose
went on describing the old. The key is pinned for the reason the seed already
is: this file measures the repair loop, so the pieces it measures have to stay
the pieces it measured.
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
from saimc.session.repairs import (
    _REPAIRS,
    Attempt,
    Repair,
    _Trial,
    _try,
    repair_chain,
    repair_table,
)
from saimc.session.tools import MAX_REPAIRS_PER_TURN
from saimc.spec import CompositionSpec, Mood

_CALM = CompositionSpec(mood=Mood.CALMING, duration_seconds=30, seed=0, key="C")
"""Clean before anything runs — found by scanning a corpus rather than by luck."""
_ELECTRIC_30_0 = CompositionSpec(mood=Mood.ELECTRIFYING, duration_seconds=30, seed=0, key="C")
"""The bed's two candidates tie here, both reaching a clean piece."""
_ELECTRIC_120_22 = CompositionSpec(
    mood=Mood.ELECTRIFYING, duration_seconds=120, seed=22, key="C"
)
"""Three bars in report order, and clearing the tune's exposes the bed's.

Re-found by sweeping the wider grid again once Phase F4 gave the melody's leap
bar a floor. Its predecessor was `electrifying/240s/31`, and F4 moved it because
this is the narrowest role in the file: of the 840 pieces the grid offers,
twenty-four breach the three bars in this order, eight of those have both of the
leap bar's candidates leave the bed's two behind, and only four come out clean
in exactly the two moves the cases read. This is the shortest of the four and
the only one under three minutes, which is why the fixture shrank.

What the role needs is exact rather than approximate, so it is worth naming.
The piece breaches the leap bar and the bed's two; *both* of the leap bar's
candidates clear it and leave the same two behind; and the band is the better
of the two on the arbiter's order, so the round keeps it and the next round
re-aims at a different bar. The "both candidates leave the same remaining"
half is what `test_a_trial_records_what_its_own_piece_left_not_the_one_in_hand`
reads, so a fixture where only one candidate cleared the bar would satisfy the
two-round case and quietly stop testing that.

The sixteen pieces the grid offers that breach the three bars and miss this
half are the ones where the leap bar is not cleared by both candidates — the
variation is kept for the smaller leap it leaves while the bar itself stays
missed — which is the shape Phase F4's floor made common by asking the line for
a leap in the first place.
"""
_ELECTRIC_90_32 = CompositionSpec(
    mood=Mood.ELECTRIFYING, duration_seconds=90, seed=32, key="C"
)
"""The engine refuses the leap bar's first request here.

Re-found by sweeping the wider grid again once Phase F4 gave the melody its
shape, and the two counts have to be told apart to say what this piece is.
Band 9 is still *refused* on 24 of the grid's 840 pieces — the narrow band writes
a dissonant collision the linter raises on — but the loop only ever asks for it
on a piece whose worst bar is the leap bar, and there is exactly one of those in
the grid. This is it. Its predecessor `electrifying/300s/47` — a piece a slower
sweep reached, outside this grid's seeds — still refuses the band as it always
did, and it no longer *reaches* it, because the melody work lifted the leap bar
out of that piece's report — its remaining misses are the bed's two, so the band
is never asked for and the arm goes unwitnessed there. A refusal is not a
breach, and a piece that refuses a request it never makes is not this fixture's
role.

The piece reproduces the role exactly and cheaper: band 9 is refused on the bare
piece, the chain is `[SetMotifVariation(0.5), SetAccompanimentDensity(1440)]`, and
the piece comes out clean in those two moves. It is a third the length of the
piece it replaces, which is the one thing about the re-finding that is a
preference rather than a measurement.

The refusal is worked around rather than waited out. Band 9 and the halving are
the leap bar's *two* candidates, so the loop does not spend a move discovering
that the band cannot be had: the halving clears the leap bar in the round the
band is refused in, and the second move is the bed's. Two is what the case
reads — the refusal is recorded rather than fatal, and the piece still ends
clean.

The thinness is the honest reading of the melody work, and it is worth stating
plainly rather than as a note about a sweep: a refused candidate needs the narrow
band to write a collision, and the floor asks the line for a leap, so the
collision got rarer rather than commoner. Twenty-four pieces still refuse the
band and one piece still reaches it; a later phase that narrows the second count
to none fails here rather than quietly dropping the case.

It is also still the piece the corpus `repairs.py` names cannot witness: that
corpus runs at ten seeds of 30, 60 and 120 seconds, and nothing the loop tries
over its 90 pieces is refused at all, so the arm is reachable and the corpus is
not where it shows. That is why this fixture is named by its spec rather than
counted there.
"""
_CALM_60_13 = CompositionSpec(mood=Mood.CALMING, duration_seconds=60, seed=13, key="C")
"""The two candidates pull strictly opposite ways, which the old one no longer did.

Narrowing the band moves the order down; halving the motif variation moves it
past the piece it started from. This is the shortest piece a sweep of the
corpus found doing that after the apex-entrance fix — `electrifying/30s/16`,
which held this role before, now has both candidates improving the piece, so
the round it was here for no longer contains a candidate to reject. The mood
changed with the piece, which is why the name did: a fixture called
`_ELECTRIC_*` that composes a calming piece is a name that lies about what the
file is measuring.

The case is worth its cost because the direction is not observable anywhere
else: an inverted comparison keeps the candidate that hurts and refuses the one
that helps, and `revise_draft`'s ratchet would then refuse the repair as a
regression — the wrong answer arriving as a refused turn rather than as bad
music.
"""
_ELECTRIC_90_15 = CompositionSpec(
    mood=Mood.ELECTRIFYING, duration_seconds=90, seed=15, key="C"
)
"""The deepest chain the music needs, and therefore the reason the bound is four.

A corpus of 90 pieces cannot bound a search over 90 pieces, so the depth was
measured on a wider grid — three moods, seven durations and forty seeds, every
piece pinned to C, 840 pieces — where the deepest chain is three kept requests
and exactly three pieces need it: this one and `electrifying/120s/15` and
`electrifying/180s/15`. This is the shortest of the three, so it carries the
ratchet.

It is also the piece that shows what the deepest chains *are*, and that is the
half Phase F4's leap floor rewrote. The chain is three *different* requests —
`[SetMelodyBand(9), SetMotifVariation(0.5), SetAccompanimentDensity(1440)]` —
and each is a move the one before it exposed: the two the table holds for the
tune's leap bar in the table's own order, then the bed's once the tune is clear.

The `no_gain` in the middle is the part worth reading, because it is the loop
declining a request it has *already carried* rather than one that never bound.
The band is round one's best and is added; the second round aims at the same bar
and tries both candidates again, so the band is folded onto a chain that already
states it, the plan is unchanged, the piece measures identical, and a tie is not
strictly better — while the halving, folded onto a chain that has the band and
not the halving, moves the piece and is kept. A loop that dropped a candidate it
had already added would not record that round at all; a loop that kept a tie
would add the band twice and never reach the halving.

`electrifying/120s/15` keeps the same three with the melody's first two decided
the other way round — the halving first and the band second, so its middle round
is `kept` rather than `no_gain` — which makes this pair the place where a piece's
own arithmetic outvotes the table's order rather than restating it.
`electrifying/180s/15` reads identically to this one, `no_gain` included.

This role used to be `electrifying/180s/17`, whose chain was
`SetMotifVariation(0.5)` three deep and then two more — one request applied three
times, which is the grinding `repairs.py` recorded against Phase F4's goal of
measurable motivic development. That piece now needs one request where it needed
five, and the deepest chains that remain are three different bars rather than one
bar thrice.
"""
_ELECTRIC_90_2 = CompositionSpec(mood=Mood.ELECTRIFYING, duration_seconds=90, seed=2, key="C")
"""The piece on which the report's order and the arbiter's disagree."""
_ELECTRIC_30_28 = CompositionSpec(mood=Mood.ELECTRIFYING, duration_seconds=30, seed=28, key="C")
"""The one piece here where the better of two kept requests is the *second*.

Both of the leap-recovery bar's requests clear it and leave the same two bars
behind them, so the bar's own order cannot decide between them and the arbiter's
total order does — band 9 misses by 1.5167 where band 14 misses by 1.5512, and the
loop takes the narrower band. That makes this the piece where "keep the best of
the kept" and "keep the first of the kept" produce different chains.

It is also the *only* piece of the 360 the sweep covers where both requests are
kept and the later one is the better: three moods, four durations and thirty
seeds puts twelve pieces in a leap-recovery round that keeps both, and this is
the one of the twelve that adds the second. So the fixture is not one example of
a common shape — it is the whole population of the shape, which is why the case
that reads it asserts the direction rather than merely the winner.
"""
_SLEEP_30_20 = CompositionSpec(mood=Mood.SLEEP, duration_seconds=30, seed=20, key="C")
"""The refusal: both of the leap-recovery bar's requests measure no better.

The one miss is the leap-recovery bar, so the round has a bar to aim at and two
requests to try, and both come back `no_gain` — which is what makes the refusal a
measurement rather than a gap. A piece that missed a bar the table has *no*
entry for would take the other refusal, which is `_CALM` under `_CLEARANCE`'s
case; this one has to miss a bar the table *does* hold requests for, and have
every one of them tie.

Found by sweeping the mood's own thirty-second pieces for the property rather
than by keeping the seed this fixture used to name. That was `sleep/30s/3`,
which Phase F4's melody work lifted onto the leap-recovery bar and then lifted
out of the report entirely — it comes out clean before the loop runs, so it can
no longer witness a refusal at all.

Twenty is the first seed of the forty where the sweep finds the property whole,
and thirty-eight is the only other one; the three seeds that miss the
leap-recovery bar as their sole bar and *not* this way — 4, 10 and 23 — each have
a request that clears it, which is a repair rather than a refusal. Thirty-eight is
also the narrower miss: its second request leaves `leap_ratio` behind instead, so
the round is a no-gain *and* a re-aim rather than the two ties this case reads.
"""
_SLEEP_30_7 = CompositionSpec(mood=Mood.SLEEP, duration_seconds=30, seed=7, key="C")
"""The refusal Phase F4 invented: the leap floor is missed and nothing can move it.

`leap_ratio` is tier one of the arbiter's order and `_REPAIRS` holds no request
for it, because no delta in the vocabulary writes the melody's *interval*
vocabulary — the two bars the floor and the cap measure are properties of the
line's shape rather than of a knob. So a piece whose worst bar is the floor takes
the `unmapped` refusal, which is the same outcome `register_separation_semitones`
produces and a different reason for it: that bar has no request because none was
measured, this one because none exists. Both are in the 90-piece corpus the table
was measured on — this one measures 0.00 against a floor of 0.01, a line of
nothing but steps and thirds.
"""
_CALM_30_6 = CompositionSpec(mood=Mood.CALMING, duration_seconds=30, seed=6, key="C")
"""The other new bar, and the cap's side of it: `step_ratio` 0.93 against 0.90.

F4 made `step_ratio` two-sided — the floor it always had, plus a cap — and the
cap has no request for the same reason the floor has none. A pair rather than one
fixture because the two bars are read from the same walk and measure opposite
faults of it: a line that never leaps, and a line that never does anything else.
"""
_SLEEP_30_28 = CompositionSpec(mood=Mood.SLEEP, duration_seconds=30, seed=28, key="C")
"""Misses the step floor *and* the leap-recovery bar, and the refusal costs a move.

The piece that shows the new bar's refusal is not merely the absence of one. The
step floor is aimed at first and has no request, so the loop stops before any
round reaches the leap-recovery bar the piece also misses — and that bar *is*
movable: `SetMelodyBand(9)` leaves it as the piece's only miss, two bars down to
one, and the order moves with it. So the refusal is a real cost on this piece
rather than a report about a bar nothing would have fixed.

Two pieces of the 840 have the shape — a new bar is the worst bar *and* some
request the table holds for another missed bar would have moved the order down —
and this is the lower seed of the two. The other is `sleep/30s/35`, which carries
a third bar and is the reason this one was preferred.
"""
_CLEARANCE = SetHarmonyClearance(semitones=24)
"""A prefix, because production folds the draft's own chain as one.

Wide enough to push the bed under the tune, which is how the wider sweep reaches
`register_separation_semitones` — a bar the table holds no request for — through
the delta vocabulary alone.
"""
_TINY_BAND = SetMelodyBand(semitones=1)
"""A prefix, and the reason the aim is a *choice* rather than a reading.

`findings()` lists a piece's misses in `QUALITY_THRESHOLDS`' order and the
arbiter ranks them in `METRIC_TIERS`': the two tables are not the same order,
and the pair they transpose is `range_semitones` and `max_leap_semitones` —
under this band the report lists the range bar first and the arbiter ranks the
leap bar above it. It is a common reading rather than a constructed one: of the
840 pieces the wider grid covers, 438 have findings under this band and 55 of
those transpose the pair, so a loop aiming at `findings[0]` would pass on most
pieces and fail on one in eight.
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
        for spec in (_ELECTRIC_30_0, _ELECTRIC_120_22, _SLEEP_30_20):
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

    def test_the_table_is_the_order_it_is_written_as(self) -> None:
        """The order itself, pinned, because it is the one property a set cannot
        carry and the three cases above hold for every permutation of it.

        A tuple's order is what settles a tie between two candidates that
        measure the same: `test_the_kept_request_is_the_better_of_two_that_tie`
        keeps the first of its two, and both reach a clean piece. So reordering
        a bar changes *which repair a piece gets* rather than merely which one
        is proposed first, and the bars' own order is what
        `retune.proposal` iterates and what `saimc-preferences` prints beside
        the table. Not evidence that the order is right — nothing can be,
        because it is a judgement measured on a corpus — but evidence that
        changing it was deliberate. The arbiter's `METRIC_TIERS` gets the same
        pin for the same reason.
        """
        assert list(_REPAIRS) == [
            "harmony_pad_coverage",
            "texture_hierarchy",
            "max_leap_semitones",
            "leap_recovery_ratio",
        ]
        assert {
            metric: tuple(candidate.describe() for candidate in candidates)
            for metric, candidates in _REPAIRS.items()
        } == {
            "harmony_pad_coverage": (
                "SetHarmonyTexture(broken_chord=False)",
                "SetAccompanimentDensity(step_ticks=1440)",
            ),
            "texture_hierarchy": (
                "SetAccompanimentDensity(step_ticks=1440)",
                "SetHarmonyTexture(broken_chord=False)",
            ),
            "max_leap_semitones": (
                "SetMelodyBand(semitones=9)",
                "SetMotifVariation(factor=0.5)",
            ),
            "leap_recovery_ratio": (
                "SetMelodyBand(semitones=14)",
                "SetMelodyBand(semitones=9)",
            ),
        }

    def test_the_table_is_handed_out_as_a_copy(self) -> None:
        """A caller gets the table to reason about, not the loop's own mapping.

        `repair_chain` reads `_REPAIRS` directly, so an accessor that returned it
        would let a reader retune the loop by assignment — from a caller that
        thought it was only looking, which is exactly what `saimc-preferences`
        is. The values are tuples and immutable; the mapping is not.
        """
        handed = repair_table()
        assert handed == _REPAIRS
        assert handed is not _REPAIRS
        handed.clear()
        assert len(_REPAIRS) == 4, "clearing the copy cleared the loop's table"


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

    def test_the_kept_request_is_the_better_of_two_that_were_both_kept(self) -> None:
        """Strictly better decides *whether* a request is kept; the arbiter's
        total order decides *which* of two kept ones is added. Here both of the
        leap-recovery bar's requests clear it and leave the same two bars
        missing, so the bar itself cannot separate them and the magnitudes can:
        band 9 misses by 1.5167 against band 14's 1.5512, and the loop takes the
        smaller. A loop that kept the first of the kept would take band 14,
        which is a different chain out of the same search.

        The fixture is the only place in the sweep where this can happen at all:
        of the 360 pieces three moods, four durations and thirty seeds cover,
        twelve reach a leap-recovery round that keeps both requests, and this is
        the one of the twelve whose *second* is the better. So the direction this
        case asserts is not one example of a common reading — it is the whole of
        the population, which is why it is asserted rather than assumed.
        """
        floor = musical_order(_quality(_ELECTRIC_30_28), lint_passed=True)
        assert worst_finding(_quality(_ELECTRIC_30_28).findings()).metric == "leap_recovery_ratio"
        wide, narrow = _REPAIRS["leap_recovery_ratio"]
        assert [wide.describe(), narrow.describe()] == [
            "SetMelodyBand(semitones=14)",
            "SetMelodyBand(semitones=9)",
        ]
        assert {f.metric for f in _quality(_ELECTRIC_30_28, (wide,)).findings()} == {
            f.metric for f in _quality(_ELECTRIC_30_28, (narrow,)).findings()
        }, "the other half of the premise: the two leave the same bars behind"
        assert (
            musical_order(_quality(_ELECTRIC_30_28, (narrow,)), lint_passed=True)
            < musical_order(_quality(_ELECTRIC_30_28, (wide,)), lint_passed=True)
            < floor
        ), "and both are improvements, the narrower one the better of them"

        outcome = _repair(_ELECTRIC_30_28, maximum=4)

        assert [attempt.outcome for attempt in outcome.attempts[:2]] == ["kept", "kept"]
        assert outcome.added[0] == SetMelodyBand(semitones=9)

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
    better. `calming/60s/13` is the piece where both are in the same round,
    so the chain the loop keeps names which reading it took.
    """

    def test_the_two_candidates_pull_opposite_ways_premise(self) -> None:
        floor = musical_order(_quality(_CALM_60_13), lint_passed=True)
        assert worst_finding(_quality(_CALM_60_13).findings()).metric == "max_leap_semitones"
        better, worse = _REPAIRS["max_leap_semitones"]
        assert musical_order(_quality(_CALM_60_13, (better,)), lint_passed=True) < floor
        assert musical_order(_quality(_CALM_60_13, (worse,)), lint_passed=True) >= floor

    def test_the_loop_keeps_the_one_that_moves_the_order_down(self) -> None:
        outcome = _repair(_CALM_60_13, maximum=1)
        assert outcome.added == (SetMelodyBand(semitones=9),)
        assert [attempt.outcome for attempt in outcome.attempts] == ["kept", "no_gain"]

    def test_a_trial_that_ties_is_not_kept(self) -> None:
        """Strictly better, not better-or-equal: this candidate produces a
        piece whose order is *identical* to the one in hand, and keeping it
        would spin the loop until the bound stopped it."""
        quality = _quality(_SLEEP_30_20)
        trial = _try(
            _SLEEP_30_20,
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
        application = apply_deltas(_ELECTRIC_90_32, [candidate])
        assert application.applied == (candidate,)
        with pytest.raises(CompositionEngineError):
            compose(application.spec, plan=application.plan)

    def test_a_candidate_the_engine_refuses_is_recorded_and_skipped(self) -> None:
        """A loop that let the engine's refusal propagate would turn a candidate
        it cannot use into a failed turn."""
        outcome = _repair(_ELECTRIC_90_32, maximum=4)
        assert [(a.metric, a.outcome) for a in outcome.attempts][:2] == [
            ("max_leap_semitones", "unplayable"),
            ("max_leap_semitones", "kept"),
        ]
        assert outcome.added[0] == SetMotifVariation(factor=0.5)
        assert _after(_ELECTRIC_90_32, (), outcome).findings() == ()
        assert len(outcome.added) == 2, "one move to work around the refusal, one to clear"


class TestTheTwoRoundChain:
    def test_a_second_round_aims_at_what_the_first_one_exposed(self) -> None:
        """Clearing the tune's leap bar leaves the bed's bars missed: the engine
        wrote a different tune under a different band. The loop re-aims because
        of this, and the second round is a second bar rather than a second try."""
        outcome = _repair(_ELECTRIC_120_22, maximum=4)
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
        assert _after(_ELECTRIC_120_22, (), outcome).findings() == ()

    def test_the_fixture_has_three_bars_and_clears_them_all(self) -> None:
        assert [finding.metric for finding in _quality(_ELECTRIC_120_22).findings()] == [
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
        outcome = _repair(_SLEEP_30_20, maximum=4)
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
        for attempt in _repair(_SLEEP_30_20, maximum=4).attempts:
            assert attempt.metric in attempt.remaining

    def test_a_trial_records_what_its_own_piece_left_not_the_one_in_hand(self) -> None:
        """A trial's `remaining` is read off the piece the candidate *produced*,
        so it can be shorter than the misses it started from — and that is the
        whole value of the record: a reader given only the piece in hand's misses
        cannot tell a trial that cleared the bar it aimed at from one that did
        not. Both of this round's candidates clear the leap bar and still miss
        the bed's, which no summary of the parent could say."""
        quality = _quality(_ELECTRIC_120_22)
        assert "max_leap_semitones" in (in_hand := tuple(f.metric for f in quality.findings()))
        outcome = _repair(_ELECTRIC_120_22, maximum=4)
        assert {attempt.remaining for attempt in outcome.attempts[:2]} == {
            ("texture_hierarchy", "harmony_pad_coverage")
        }, in_hand
        assert in_hand not in {attempt.remaining for attempt in outcome.attempts}


class TestTheTwoNewBars:
    """Phase F4's measurements, and the refusal they arrive with.

    The two bars F4 added read the melody's interval vocabulary — the share of
    steps and the share of leaps — and `_REPAIRS` holds no request for either,
    because no delta in the vocabulary writes that vocabulary: a band changes how
    high a line goes and a variation changes what it is made of, and neither
    changes the size of the intervals it uses. So the honest outcome is the
    `unmapped` refusal, and F4 put `leap_ratio` at tier one where it can be aimed
    at first.

    That is a change in behaviour rather than only in measurement, so it gets
    cases: a piece that misses only the new bar, and a piece that misses the new
    bar *and* one the table can clear — which is refused anyway, because the
    arbiter ranks the tune above the bed. Every fixture here is a piece of the
    90-piece corpus or the wider grid, found by sweeping rather than built.
    """

    def test_neither_bar_has_a_request_in_the_table(self) -> None:
        """The premise of every case below, and the reason the outcome is
        `unmapped` rather than a move. A later phase that adds a request for
        either bar — a delta that writes the interval vocabulary — fails here
        rather than quietly turning these cases into repairs."""
        assert "leap_ratio" not in _REPAIRS
        assert "step_ratio" not in _REPAIRS

    def test_the_arbiter_ranks_both_above_a_bar_the_table_can_move(self) -> None:
        """The other half of the premise: a bar with no request only reaches the
        refusal if it can be the *worst* bar. Tier-first placement is what makes
        `leap_ratio` able to outrank the bed, and the table's own keys would have
        sorted it last."""
        assert METRIC_TIERS.index("leap_ratio") < METRIC_TIERS.index("texture_hierarchy")
        assert METRIC_TIERS.index("step_ratio") < METRIC_TIERS.index("texture_hierarchy")

    @pytest.mark.parametrize(
        ("spec", "metric"),
        [(_SLEEP_30_7, "leap_ratio"), (_CALM_30_6, "step_ratio")],
        ids=["leap-floor", "step-cap"],
    )
    def test_a_piece_that_misses_only_that_bar_is_refused_by_name(
        self, spec: CompositionSpec, metric: str
    ) -> None:
        outcome = _repair(spec, maximum=MAX_REPAIRS_PER_TURN)
        assert outcome.attempts == (
            Attempt(metric=metric, delta=None, outcome="unmapped", remaining=(metric,)),
        )
        assert outcome.added == ()
        assert outcome.remaining == (metric,)
        assert outcome.stopped_by is not None
        assert outcome.stopped_by.metric == metric

    def test_a_piece_that_misses_the_new_bar_and_a_movable_one_stops_on_the_tune(
        self,
    ) -> None:
        """The cost of the tier-first placement, on a real piece.

        The loop aims at the arbiter's worst bar; the step floor is tier one and
        has no request, so the round that would have reached the leap-recovery bar
        never happens. The move it costs is measured here rather than argued: the
        band the table holds for that bar leaves it as the piece's only miss, and
        the order moves down with it. Without this, "the refusal is honest" would
        be a claim that the unaimed bar was unmovable anyway, which is exactly
        what this piece shows is false.
        """
        assert [finding.metric for finding in _quality(_SLEEP_30_28).findings()] == [
            "step_ratio",
            "leap_recovery_ratio",
        ]
        _wide, narrow = _REPAIRS["leap_recovery_ratio"]
        assert narrow.describe() == "SetMelodyBand(semitones=9)"  # the move in question
        rescued = _quality(_SLEEP_30_28, (narrow,))
        assert [finding.metric for finding in rescued.findings()] == ["leap_recovery_ratio"]
        assert musical_order(rescued, lint_passed=True) < (
            musical_order(_quality(_SLEEP_30_28), lint_passed=True)
        ), "and the move would really have helped, which is what makes the cost real"

        outcome = _repair(_SLEEP_30_28, maximum=MAX_REPAIRS_PER_TURN)
        assert outcome.attempts == (
            Attempt(
                metric="step_ratio",
                delta=None,
                outcome="unmapped",
                remaining=("step_ratio", "leap_recovery_ratio"),
            ),
        )
        assert outcome.added == ()
        assert outcome.remaining == ("step_ratio", "leap_recovery_ratio")
        assert outcome.stopped_by is not None
        assert outcome.stopped_by.metric == "step_ratio"



        """`stopped_by` is `None` at the bound on purpose: stopping because the
        caller allowed one move is not stopping because nothing works, and the
        two carry different advice."""
        outcome = _repair(_ELECTRIC_120_22, maximum=1)
        assert outcome.added == (SetMelodyBand(semitones=9),)
        assert outcome.remaining == ("texture_hierarchy", "harmony_pad_coverage")
        assert outcome.stopped_by is None

    def test_the_second_move_is_the_one_that_finishes_it(self) -> None:
        one = _repair(_ELECTRIC_120_22, maximum=1)
        two = _repair(_ELECTRIC_120_22, maximum=2)
        assert two.added[:1] == one.added
        assert two.remaining == ()
        assert musical_order(_after(_ELECTRIC_120_22, (), two), lint_passed=True) < (
            musical_order(_after(_ELECTRIC_120_22, (), one), lint_passed=True)
        )

    def test_the_bound_sits_above_the_deepest_chain_the_music_needs(self) -> None:
        """The one property that makes `maximum` a backstop rather than a wall.

        A corpus of 90 pieces cannot bound a search over 90 pieces, so the depth
        is measured on a wider grid — three moods, seven durations and forty
        seeds, every piece pinned to C, 840 pieces. The deepest chain there is
        three kept requests, at `_ELECTRIC_90_15` and at
        `electrifying/120s/15` and `electrifying/180s/15`, and the bound is one
        above it. Nothing on the grid is stopped by the bound — measured, not
        assumed: not one of the 840 comes back with a bar still missed because
        the budget ran out — so the fourth move is headroom rather than slack.

        The headroom is deliberate, because a sample cannot prove a negative and
        this grid has already moved once. Before Phase F4 gave the melody its
        shape, the same sweep needed five kept requests; at a bound of four the
        longest chains there came back with a bar still missed that one more move
        cleared, which is a backstop being *reached* rather than a search ending
        on its own. So the depth is a fact about today's music and not a ceiling,
        and this assertion is what makes a later deepening fail here rather than
        quietly truncate.

        The assertion is `==` on the measurement and `<` on the bound, so a
        later phase that deepens the music past the bound fails here rather than
        quietly shipping a piece with a bar left missed — and lowering the
        budget below the measurement fails here too, which is the half a test on
        the fixture alone cannot see.
        """
        deepest = _repair(_ELECTRIC_90_15, maximum=MAX_REPAIRS_PER_TURN)
        assert len(deepest.added) == 3, [delta.describe() for delta in deepest.added]
        assert deepest.remaining == ()
        assert len(deepest.added) < MAX_REPAIRS_PER_TURN

    def test_the_corpus_fixtures_clear_well_inside_the_bound(self) -> None:
        """The corpus's own reading, for contrast with the grid's: every piece
        the table was measured on needs at most two kept requests, so for the
        90 pieces the table was chosen from, the bound is nowhere near the usual
        end of a repair."""
        for spec in (_ELECTRIC_120_22, _CALM_60_13, _SLEEP_30_20):
            assert len(_repair(spec, maximum=MAX_REPAIRS_PER_TURN).added) <= 2, spec


class TestDeterminism:
    def test_the_same_piece_repairs_the_same_way_twice(self) -> None:
        """No model, no audio, and no dependence on iteration order: the two
        `Repair`s are equal, which is what makes a recorded turn replayable."""
        first = _repair(_ELECTRIC_120_22, maximum=4)
        second = _repair(_ELECTRIC_120_22, maximum=4)
        assert first == second
        assert list(first.added) == list(second.added)

    def test_the_candidates_are_tried_in_the_tables_order(self) -> None:
        """A tie goes to the table rather than to whoever happens to sort first,
        so the order the `dict` states is contract."""
        assert [a.delta for a in _repair(_ELECTRIC_120_22, maximum=1).attempts] == list(
            _REPAIRS["max_leap_semitones"]
        )


class TestTheAim:
    def test_the_report_order_and_the_arbiters_are_not_the_same_order(self) -> None:
        """The premise the case below rests on, asserted rather than believed:
        the two tables really do disagree, over the whole breaching set rather
        than on one swapped pair. If either table is re-ordered so they agree,
        this fails and the case that follows stops claiming anything."""
        findings = _quality(_ELECTRIC_90_2, (_TINY_BAND,)).findings()
        reported = [finding.metric for finding in findings]
        arbitrated = [metric for metric in METRIC_TIERS if metric in reported]
        assert reported != arbitrated, (reported, arbitrated)
        assert reported[:2] == ["range_semitones", "max_leap_semitones"]
        assert worst_finding(findings).metric == "max_leap_semitones"

    def test_every_round_aims_at_the_arbiters_worst_bar(self) -> None:
        """The loop has no taste of its own: which bar it repairs is the
        arbiter's tier order's decision, so the first attempt of every case here
        names `worst_finding` of the findings it was given.

        `_ELECTRIC_90_2` under a one-semitone band is the case that makes this a
        test rather than a restatement: it is the only piece here whose report
        order differs from the arbiter's, and a loop reading the report's first
        miss would aim at `range_semitones` — a bar the table holds no request
        for — instead of at the leap bar above it.
        """
        for spec, chain in (
            (_ELECTRIC_30_0, ()),
            (_ELECTRIC_120_22, ()),
            (_CALM_60_13, ()),
            (_SLEEP_30_20, ()),
            (_CALM, (_CLEARANCE,)),
            (_ELECTRIC_90_2, (_TINY_BAND,)),
        ):
            quality = _quality(spec, chain)
            expected = worst_finding(quality.findings())
            assert expected is not None, spec
            assert _repair(spec, chain, maximum=4).attempts[0].metric == expected.metric, spec

    def test_a_kept_chain_leaves_the_ratchet_nothing_to_refuse(self) -> None:
        """Every kept chain has already moved the arbiter's order down — the
        same comparison the ratchet makes — so no repair this loop returns can
        be refused as a regression by the tool that applies it."""
        for spec in (_ELECTRIC_30_0, _ELECTRIC_120_22, _ELECTRIC_90_32, _CALM_60_13):
            outcome = _repair(spec, maximum=4)
            assert musical_order(_after(spec, (), outcome), lint_passed=True) <= (
                musical_order(_quality(spec), lint_passed=True)
            ), spec
