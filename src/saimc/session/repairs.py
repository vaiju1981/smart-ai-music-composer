"""The repair loop: what moves a bar, and what that buys.

Phase E's critics say what a draft misses and where; each finding's `hint` says
in a sentence what would move it. The hints are written for a maintainer —
almost all of them name a source file and a constant — so a repair cannot read
the knob off the hint. It reads it off `_REPAIRS`, and the table is *measured*
rather than reasoned from the hints: for every piece of a 90-piece corpus
(three moods x 30, 60 and 120 seconds x ten seeds, every piece pinned to C so
that a re-measurement composes the same music), every bar it breaches, and
every Tier-2 delta at a few magnitudes, does the reading clear and does the
piece come out clean.

What the measurement says, and it is the whole design:

- **Five metrics breach at all, and 45 of the 90 pieces breach something.**
  `texture_hierarchy` (30 pieces), `harmony_pad_coverage` (30),
  `leap_recovery_ratio` (8), `leap_ratio` (7) and `step_ratio` (1). So a repair
  loop is a loop over five bars in practice — three the table holds a request
  for and two it does not — and the other eight of the thirteen metrics the
  scorecard measures have no entry because nothing asked for them by *that*
  corpus, which is not the same as their being unreachable. Three of the five
  predate Phase F4 and two are its own, and the count rose from 33 to 45 because
  of those two: a floor under the line's leaps and a ceiling over its steps hold
  it to a shape it did not have before, so pieces the old set of bars passed now
  miss a bar that did not exist. `max_leap_semitones` runs the other way — the
  one bar whose entry predates the phase, it no longer breaches the corpus at
  all.
  A sweep of four requests over the same 90 pieces — a 24- and a 60-semitone
  harmony clearance and a 500- and a 1-semitone melody band, reached through
  the delta vocabulary rather than through a mood and a seed — reaches ten
  bars, six of which no entry in this table holds. Three of those six are the
  ones the loop aims at *first*, and they are the reading this paragraph exists
  for: `register_separation_semitones` (58 pieces under the 24-semitone
  clearance, 89 under 60 — the bed pushed down under the tune),
  `tessitura_overlap_semitones` (52 under 24, 53 under 60 — the bed's window
  taking the tune inside it) and `range_semitones` (65 under the 500-semitone
  band), which the loop aims at first and stops on — `unmapped` — on 31, 53 and
  57 pieces respectively. That is the outcome the next bullet's design has to be
  able to say out loud.
  `range_semitones` earns its place twice over: under a *one*-semitone band it
  arrives at `electrifying/90s/2`, each time alongside
  `max_leap_semitones`, and that pair is one on which the quality report's order
  and the arbiter's tier order **disagree** — the report lists
  `range_semitones` first and the arbiter ranks `max_leap_semitones` higher — so
  it is what makes "the loop aims at the arbiter's worst bar" a claim a test can
  fail rather than a restatement of `findings()[0]`. (The one-semitone arm is
  the one reading above the 90-piece corpus no longer carries: it holds no
  90-second piece, and of the pieces a one-semitone band does reach, the linter
  refuses four before a bar can be read and none of the rest breaches the
  range. So the witness the test uses is named by its own spec rather than
  counted here — a refusal is not a breach, and counting the two together would
  be counting pieces nothing measured.)
- **`harmonic_rhythm_variety` is reachable, and it is the one bar this table must
  not answer.** Measured across the golden corpus under each of the plan's three
  section closes: under `hold` — sections that run on, so every chord lasts the
  template's uniform two bars — 36 of 81 pieces breach it, and under the shipped
  default `half` and under `full`, none of 81 do. So it is not a bar nothing
  reaches; it is a bar the plan reaches *by asking for it*. Every request in the
  vocabulary was then measured against those 36 pieces and not one moved any of
  them, while `SetSectionClose("half")` clears 36 of 36. The only delta that
  clears the bar is the one that undoes the request that made the piece breach —
  and a repair is the product's own arithmetic over the piece, which may change
  how the material is placed and never what the user asked for. So this absence
  is a judgement rather than a gap, and it is the only one here that is.
- **The bed's two bars are one repair seen from two sides.**
  `SetHarmonyTexture(broken_chord=False)` — the knob `harmony_pad_coverage`'s
  own hint names, and the mood's texture rule the other way round — clears the
  pad bar in 30 of 30 pieces and the texture bar in 30 of 30, and
  `SetAccompanimentDensity(step_ticks=1440)`, an arpeggio that steps once per
  three beats instead of once per beat, measures *identically* on all 30: the
  two readings are the same 29 pieces out clean and one whose other bar was
  already missing — the tune's recovery bar. They are both "let the bed hold
  rather than
  flurry", and the density move is the one that works when the figure is not the
  broken chord. Those are single-candidate readings, taken to *choose* the
  table; what the loop does with them is the next bullet.
- **The tune has two bars and neither is dominant, which is what makes the
  chains long.** `max_leap_semitones` is breached by 44 of the grid's 840
  pieces — it was 78 before the seam fix that kept the leaps a bar's last slot
  cannot answer, and 125 before Phase F4's leap floor — and the table's two
  requests for it split the work: `SetMelodyBand(semitones=9)`, the narrowest
  band the range bar allows, clears it in 39 of the 44 and leaves the whole
  piece clean in 19, and the linter *refuses* it on one of them;
  `SetMotifVariation(factor=0.5)` clears it in 22 and leaves the piece clean in
  12, and is never refused. So 41 of the 44 clear the bar on one move and **the
  three that clear it on neither are exactly the grid's three deepest chains** —
  `electrifying/90s/15`, `120s/15` and `180s/15`. On those, the pair is the
  repair and neither half is: narrowing the band brings the leap under the bar
  and the halving is what keeps it there, which is why the loop needs two moves
  a musician would follow rather than one it could take twice.

  `leap_recovery_ratio` is the commoner of the two and the harder — 51 of the
  840, its candidates are `SetMelodyBand(14)` and `SetMelodyBand(9)`, and 39 of
  the 51 clear it on one move, 27 come out clean and 12 clear it on neither,
  with nothing refused. Those 12 are what the hint is about without meaning to:
  it asks for a change to `motif.py`'s walk, and no request in the vocabulary is
  that change. So each bar is repairable where a knob can reach it and reported
  where one cannot, and the loop keeps the best of what it tried rather than
  claiming a move it does not have. The bound in `tools.py` is set from the
  chains this leaves.

  **The two bars pull against each other, and the table's candidates are where
  that shows.** A floor under the leaps and a bar on how well they are recovered
  want opposite things of the line — the first asks it for a leap, the second
  asks it to answer the leap well — so a band narrow enough to satisfy the
  recovery bar can drop the same line under the floor. That is measured in the
  next paragraph's sweep rather than reasoned here, and it is why the corpus
  ends with one piece improved and still missing a bar.
- **A trial the engine refuses is skipped, not fatal.** `compose` raises on a
  lint failure (C4's finding 1), and a repair loop that let that propagate would
  turn a candidate it cannot use into a failed turn, so `unplayable` is a
  recorded outcome. The 360-request sweep above is refused 4 times, all of them
  a one-semitone band writing dissonant collisions. What the *corpus* cannot
  reach is the bare reading of the arm — over its 90 pieces and 86 trials,
  nothing the loop tries is refused at all — and that is a fact about its
  durations rather than about the arm: the collision a refusal needs belongs to
  the narrow band, and the corpus's own bars are the bed's and the two the
  melody's floor and cap added, whose candidates never write one.
  `SetMelodyBand(semitones=9)` is refused on the bare piece at
  `electrifying/90s/32`, which is outside the corpus and is where the test that
  reads the arm names its fixture. Over the wider grid of three moods, seven
  durations and forty seeds the band is refused on 24 of its 840 pieces, and the
  loop *asks* for the band on exactly one of them — this one. The two counts have
  to be told apart rather than added: a refusal on a piece whose worst bar is
  something else is a request the loop never makes, so the arm would be
  unwitnessed there and the 24 would be a number about nothing. So the corpus is
  not evidence the arm is dead, and `test_session_repairs.py` is where the two
  are told apart.
- **The deepest chains are three different requests, and each is a bar of its
  own.** `electrifying/90s/15` — one of the grid's three deepest chains — keeps
  `SetMelodyBand(semitones=9)`, then `SetMotifVariation(factor=0.5)`, then
  `SetAccompanimentDensity(step_ticks=1440)`; `electrifying/180s/15` takes the
  same order, and `electrifying/120s/15` the same three with the first two the
  other way round. Each is a move the one before it exposed — narrowing the band
  brings the leap bar inside it, halving the motif is what keeps the leap under
  the bar, and clearing that leaves the bed's bars, which is the third — and
  each measures strictly better than the piece in hand.

  **These three are exactly the pieces the previous bullet's pair cannot finish
  alone**, which is the whole reason they are three deep: on every other piece
  that breaches the leap bar, one of its two candidates clears it by itself. So
  the depth is not the loop grinding — it is the one place where two bars have to
  be satisfied by two moves instead of one.

  Two of the three also record a `no_gain` in their middle round, and it is the
  opposite of a wasted move: the band is *re-tried* in a round after the chain
  already carries it, so it is folded onto a chain that already states it, the
  plan does not change, the piece measures identical, and a tie is not strictly
  better. `electrifying/120s/15` records none, because its first two requests are
  decided the other way round and its middle round is a `kept`.

  This used to be one request applied three times (`SetMotifVariation(0.5)`
  three deep at `electrifying/180s/17`), and two changes removed that shape.
  Phase F4's leap floor took the grid's leap-bar breaches from 125 to 78, and
  the seam fix took them to 51 — so the leap bar needs working around in half as
  many pieces as it did, `electrifying/180s/17` now needs one request where it
  needed five and no longer breaches the leap bar at all (its remaining misses
  are the bed's two), and the chains that
  survive are three different bars rather than one bar thrice. So the specific
  grinding the item Phase F4 left open was about — the
  loop paying a piece's melodic variation for a smaller worst leap, once per
  round — is gone from the deepest chains, and what is left is three requests
  a musician would follow.
- **`refused` is the one outcome no request in this table produces**, and it is
  kept for a reason a test cannot supply. `_try` checks that the applier honoured
  the whole chain and records `refused` when it did not; every entry in `_REPAIRS`
  is asserted to be honoured by the applier, so that check never fires today, and
  a sabotage that turned its outcome into `kept` was missed — the honest reading
  of the branch. What it buys is the *shape* of the failure: without it, a chain
  the applier trimmed would be composed and measured as though the candidate had
  been applied, which is a wrong answer arriving silently rather than a named
  one. A test can be deleted or mis-edited; this cannot, and it is the reason the
  branch stays without a case to reach it.

**What the loop does to that corpus**, which is the number the table above was
chosen for rather than a claim about it — 53 of the 90 pieces are clean before
it runs, 36 of the 45 breaching ones come out clean, and 9 are left as they
were. Thirty-six of the repaired pieces need one kept request and one needs two,
so `maximum` never binds *here*; across all 86 trials, 72 outcomes are `kept`, 6
are `no_gain` and 8 are `unmapped`.

The 8 `unmapped` are worth naming as a set, because they are what Phase F4 did
to this loop: they are attempts aimed at `leap_ratio` and `step_ratio`, the two
bars the phase added and the two no request in this table moves. So two of the
five bars this corpus breaches cannot be repaired by anything the vocabulary
contains, and the loop's refusals are no longer all of the bed-and-tune kind the
table was built for.

The corpus is not the whole picture, and the gap between the two is worth the
sentence. A corpus of 90 pieces says nothing about a piece it does not contain,
so the depth was re-measured over a wider grid — three moods, seven durations
and forty seeds, every piece pinned to C, 840 pieces — where the deepest chain
needed **three** kept requests and three pieces needed it. That is what
`MAX_REPAIRS_PER_TURN` is set from, and it is one above rather than equal to it
for the reason this module states about `maximum`: a backstop that is reached is
not a backstop. Nothing on the grid reaches it — no piece ends with a bar still
missed because it ran out of rounds, and the three deepest chains all finish
clean at a bound of three.

The grid also shows what the corpus is too short to: **33 of its 840 pieces end
stopped**, and the split is exact. Eighteen stop `unmapped` — `leap_ratio` on
12, `step_ratio` on 5 and `distinct_durations` on 1 — and fifteen stop `no_gain`
on `leap_recovery_ratio`, the only bar where both of the table's requests were
tried and declined. So every stop is one of two things: a bar no request here
holds, or the recovery bar's trade. One of the corpus's pieces is the smaller
form of that second thing — `sleep/60s/8`, where the band is kept for the ground
it gains and the recovery bar stays missed — and it is the corpus's one piece
improved and still missing a bar.

**A `no_gain` is not one failure but two**, and Phase F4 is what made the second
one reachable. In the 360-request sweep, 26 trials are `no_gain`; 22 of them
left the bar they aimed at still missed, and the other **4 cleared it and left a
different bar in its place**. Those four are the trade the tune's two-bar bullet
predicts: the candidate narrowed the band to satisfy the recovery bar and
dropped the same line under the leap floor, so the worst bar *changed* rather
than the piece improving. The reading the old version of this paragraph said a
wide enough sweep had never found — "a candidate that clears its bar and still
loses on the piece as a whole" — is now measured rather than hypothetical, and
the two-sided pair is why: before the floor and the cap there was no bar whose
satisfaction could break another. What the refusal says is what the attempts
recorded: which requests were tried and that none was kept. It does not narrate
the trade, because the trade is a property of the measurements rather than of
the request, and the loop keeps the bar it started with rather than the one it
was handed.

The loop itself: aim at the worst miss by the arbiter's own tier order, try
every request the table holds for that bar, keep the one that measures best on
the arbiter's total order, and stop when nothing measures strictly better. It
is deterministic end to end — the candidates are literals, `compose` is a
function, and a tie between two candidates goes to the table's order — and it
spends no model calls and no audio. Its whole cost is bounded by `maximum`
times the table's width in composes of arithmetic — eight composes, and four
seconds, at the 600-second cap where a compose measures 0.53 s, and under a
second at the durations the deep chains actually occur, since the deepest ones
found are a 90-, a 120- and a 180-second piece — which is the reason nothing here
touches the turn's ledger: the ledger counts what is expensive (sketches and
model calls), not what is cheap.

`revise_draft` composes the winning chain a second time, and that is deliberate
rather than wasteful: the loop composes a trial to *measure* it, and the draft
is built by the function that owns the chain, the ratchet and the lineage. The
second compose is the same deterministic function on the same inputs — three to
four milliseconds of arithmetic against a second source of truth about what a
chain produces.

Two things are deliberately absent. There is no `metric` argument: which bar to
repair is the arbiter's tier order's decision, and a caller free to override it
would be choosing by taste a thing the arbiter owns — the conductor that has an
opinion can say it with `revise`. And there is no `SetTempo`-shaped candidate:
Phase D measured that the arrangement trades a pinned tempo away, so a repair
that asked for one could be honoured and still not happen.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal

from saimc.compose.engine import CompositionEngineError, compose
from saimc.quality import PieceQuality, QualityFinding, score_piece
from saimc.session.arbiter import musical_order, worst_finding
from saimc.session.deltas import (
    Delta,
    SetAccompanimentDensity,
    SetHarmonyTexture,
    SetMelodyBand,
    SetMotifVariation,
    apply_deltas,
)
from saimc.spec import CompositionSpec

_REPAIRS: Final[dict[str, tuple[Delta, ...]]] = {
    "harmony_pad_coverage": (
        SetHarmonyTexture(broken_chord=False),
        SetAccompanimentDensity(step_ticks=1440),
    ),
    "texture_hierarchy": (
        SetAccompanimentDensity(step_ticks=1440),
        SetHarmonyTexture(broken_chord=False),
    ),
    "max_leap_semitones": (
        SetMelodyBand(semitones=9),
        SetMotifVariation(factor=0.5),
    ),
    "leap_recovery_ratio": (
        SetMelodyBand(semitones=14),
        SetMelodyBand(semitones=9),
    ),
}
"""The requests that move each bar, in the order a tie between them is settled.

Plan-written rather than spec-written, and that is the point of a repair: a
*request* is the user's, and a repair is the product's own arithmetic over the
piece — it may change how the material is placed and never what the user asked
for. Every entry was measured to clear its bar somewhere in the corpus, and the
widths are two because a third candidate never changed a verdict.
"""

AttemptOutcome = Literal["kept", "no_gain", "refused", "unplayable", "unmapped"]
"""How one trial ended.

`kept` and `no_gain` are the two the loop decides on — the trial's own order
against the piece in hand's — and they are the two that produced a piece to
measure. `refused` is the applier declining the request inside the chain,
`unplayable` is the engine declining the notes it wrote, and `unmapped` is a bar
with no entry in the table at all, which is the outcome a reader most needs told
apart from the others: nothing was tried, rather than everything was tried and
nothing worked.
"""


@dataclass(frozen=True)
class Attempt:
    """One candidate request, and what the piece looked like after it.

    A trial that produced no piece still carries `remaining`, because the piece
    in hand is the piece it left behind: a refused or unplayable candidate
    changes nothing, and writing "nothing is left to repair" there would be the
    opposite of the truth.
    """

    metric: str
    delta: Delta | None
    outcome: AttemptOutcome
    remaining: tuple[str, ...]


@dataclass(frozen=True)
class Repair:
    """What a pass of the loop bought, and everything it tried.

    `added` is what a caller applies, in the order it was kept, and `remaining`
    is what the piece still misses afterwards. `stopped_by` is the bar the loop
    could not move — `None` when it ran out of misses or of budget. It is the
    *finding* rather than its name, because a caller that has to explain the
    failure needs the bar's own hint and sentence, and looking them up by name
    would be a second read of the report the finding came from.
    """

    added: tuple[Delta, ...]
    remaining: tuple[str, ...]
    attempts: tuple[Attempt, ...]
    stopped_by: QualityFinding | None


@dataclass(frozen=True)
class _Trial:
    """A candidate that produced a piece, measured against the one in hand."""

    order: tuple[int, int, float, int]
    quality: PieceQuality
    attempt: Attempt


def _try(
    root_spec: CompositionSpec,
    prefix: tuple[Delta, ...],
    candidate: Delta,
    *,
    aimed_at: str,
    floor: tuple[int, int, float, int],
    remaining: tuple[str, ...],
) -> _Trial | Attempt:
    """Fold one candidate onto the chain, compose it, and score what came out.

    The two outcomes are different types rather than one type with a nullable
    score. A candidate either produces a piece or it does not, and a trial that
    did not has no order to compare and no scorecard to read — a field standing
    in for the missing one would have to be guard-read at every site.

    **A smaller order is a better piece.** `musical_order` counts what is wrong
    with one — legality, how many bars it misses, how far past them it went,
    which bar was the highest — so clearing a bar moves the tuple *down*, and
    the comparisons here read the same way the arbiter's own ratchet reads them.
    Getting that backwards is not a near miss: it keeps every candidate that
    makes the piece worse and refuses every one that makes it better, which is
    what the first draft of this function did.

    The chain is folded from the *root* spec, which is the same rule
    `revise_draft` follows and for the same reason: the plan is derived from the
    spec, so a plan request folded onto a later spec would drop whatever the
    earlier requests asked for.
    """
    chain = (*prefix, candidate)
    application = apply_deltas(root_spec, chain)
    if len(application.applied) != len(chain):
        return Attempt(aimed_at, candidate, "refused", remaining)
    try:
        output = compose(application.spec, plan=application.plan)
    except CompositionEngineError:
        return Attempt(aimed_at, candidate, "unplayable", remaining)
    quality = score_piece(output.notation_score)
    order = musical_order(quality, lint_passed=True)
    return _Trial(
        order=order,
        quality=quality,
        attempt=Attempt(
            metric=aimed_at,
            delta=candidate,
            outcome="kept" if order < floor else "no_gain",
            remaining=tuple(finding.metric for finding in quality.findings()),
        ),
    )


def repair_table() -> dict[str, tuple[Delta, ...]]:
    """The table `repair_chain` reads, for a caller that has to reason about it.

    A copy, so a reader cannot assign into the loop's own mapping — the values
    are tuples and immutable, the mapping is not — and a function rather than a
    public constant because the table stays private on purpose: a repair reads
    its knob off *this* table and never off the finding's hint, and a module
    attribute that looked like a suggestion would invite the second read the
    docstring above exists to prevent.

    **Each tuple's order is significant.** It is what settles a tie between two
    candidates that measure the same, so a caller ranking these requests is
    proposing to *reorder the table* — which changes which repair a piece gets —
    and not merely to sort a set of requests.
    """
    return dict(_REPAIRS)


def repair_chain(
    root_spec: CompositionSpec,
    prefix: Sequence[Delta],
    quality: PieceQuality,
    *,
    maximum: int,
) -> Repair:
    """Repair the worst bar `quality` misses, one kept request at a time.

    `prefix` is the chain the draft already carries, and `quality` is *its*
    scorecard — the misses are read from the record rather than recomposed,
    because a draft that stores no notes still stores what its notes measured.

    Each round aims at the piece's own worst bar, tries everything the table
    holds for it, and keeps the best of what measured strictly better than the
    piece in hand — strictly, because a candidate that ties buys nothing and
    would spin the loop. Then it aims again, because clearing one bar can expose
    another.

    The loop ends on one of four things, and empty `added` means the first
    three: nothing to repair, a bar with no request in the table, or a bar whose
    requests all measured no better. `maximum` is the fourth and it is a
    backstop rather than the usual end — measured over the corpus, no piece
    needs more than two kept requests, and over the wider grid the deepest is
    three.
    """
    findings: tuple[QualityFinding, ...] = quality.findings()
    order = musical_order(quality, lint_passed=True)
    added: list[Delta] = []
    attempts: list[Attempt] = []
    stopped_by: QualityFinding | None = None

    while findings and len(added) < maximum:
        worst = worst_finding(findings)
        assert worst is not None  # the loop's own condition is that there is one
        candidates = _REPAIRS.get(worst.metric)
        misses = tuple(finding.metric for finding in findings)
        if not candidates:
            attempts.append(Attempt(worst.metric, None, "unmapped", misses))
            stopped_by = worst
            break
        results = [
            _try(
                root_spec,
                (*prefix, *added),
                candidate,
                aimed_at=worst.metric,
                floor=order,
                remaining=misses,
            )
            for candidate in candidates
        ]
        attempts.extend(
            result.attempt if isinstance(result, _Trial) else result for result in results
        )
        kept = [
            result
            for result in results
            if isinstance(result, _Trial) and result.attempt.outcome == "kept"
        ]
        if not kept:
            stopped_by = worst
            break
        best = min(kept, key=lambda trial: trial.order)
        assert best.attempt.delta is not None  # a measured trial carries its request
        added.append(best.attempt.delta)
        order = best.order
        findings = best.quality.findings()

    return Repair(
        added=tuple(added),
        remaining=tuple(finding.metric for finding in findings),
        attempts=tuple(attempts),
        stopped_by=stopped_by,
    )
