"""The repair loop: what moves a bar, and what that buys.

Phase E's critics say what a draft misses and where; each finding's `hint` says
in a sentence what would move it. The hints are written for a maintainer —
almost all of them name a source file and a constant — so a repair cannot read
the knob off the hint. It reads it off `_REPAIRS`, and the table is *measured*
rather than reasoned from the hints: for every piece of a 90-piece corpus
(three moods x three durations x ten seeds, every piece pinned to C so that a
re-measurement composes the same music), every bar it breaches, and every
Tier-2 delta at a few magnitudes, does the reading clear and does the piece
come out clean.

What the measurement says, and it is the whole design:

- **Four metrics breach at all, and 36 of the 90 pieces breach something.**
  `texture_hierarchy` (30 pieces), `harmony_pad_coverage` (30),
  `max_leap_semitones` (12) and `leap_recovery_ratio` (1). So a repair loop is
  a loop over four bars in practice, and the other eight have no entry because
  nothing asked for them by *that* corpus — not because they cannot be reached.
  A later sweep of 2592 applied requests over 72 pieces, reached through the
  delta vocabulary rather than through a mood and a seed, finds three of them:
  `register_separation_semitones` (59 pieces under a 24-semitone harmony
  clearance, 88 under 60 — the bed pushed down under the tune),
  `tessitura_overlap_semitones` (50 under a 24-semitone clearance, 60 under 60 —
  the bed's window taking the tune inside it) and `range_semitones` (76 under a
  500-semitone band). All three land on `unmapped`, which is the outcome the
  next bullet's design has to be able to say out loud. `range_semitones` earns
  its place twice
  over: under a *one*-semitone band it arrives in 2 pieces, each time alongside
  `max_leap_semitones`, and that pair is one on which the quality report's order
  and the arbiter's tier order **disagree** — the report lists
  `range_semitones` first and the arbiter ranks `max_leap_semitones` higher — so
  it is what makes "the loop aims at the arbiter's worst bar" a claim a test can
  fail rather than a restatement of `findings()[0]`.
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
  two readings are the same 23 pieces out clean and 7 pieces whose other bar —
  the tune's — was already missing. They are both "let the bed hold rather than
  flurry", and the density move is the one that works when the figure is not the
  broken chord. Those are single-candidate readings, taken to *choose* the
  table; what the loop does with them is the next bullet.
- **The tune's leap bar is only partly repairable, and the honest reading of
  that is a bound rather than a knob.** `SetMelodyBand(semitones=9)` — the
  narrowest band the range bar allows — clears `max_leap_semitones` in 7 of the
  12 pieces that breach it, and 4 of those still miss another bar afterwards;
  `SetMotifVariation(factor=0.5)` clears 4 of the 12. No single move clears the
  bar *and* the piece, which is what the hint says without meaning to: it asks
  for a change to `motif.py`'s walk, and no request in the vocabulary is that
  change. The loop therefore keeps the best of the two and reports what is left.
- **A trial the engine refuses is skipped, not fatal.** `compose` raises on a
  lint failure (C4's finding 1), and the wider sweep's 2592 requests are refused
  24 times — a five-semitone band and a four-bar arpeggio step both write
  dissonant collisions. A repair loop that let that propagate would turn a
  candidate it cannot use into a failed turn, so `unplayable` is a recorded
  outcome. No candidate the table itself holds is refused *on a bare piece* —
  all eight measure on a single move — but two of the corpus's 88 trials are
  refused, and both are a candidate tried on a round *after* the first, where it
  is folded onto a two-deep chain rather than onto the piece alone. So what the
  corpus cannot reach is the bare reading of the arm, not the arm itself, and
  `test_session_repairs.py` is where the two are told apart.
- **The deepest chains are one request applied three times, and that is the
  loop's real shape rather than a defect in it.** `electrifying/180s/17` — the
  deepest chain the wider grid needs — keeps `SetMotifVariation(factor=0.5)`
  three times, then `SetMelodyBand(semitones=9)`, then
  `SetAccompanimentDensity(step_ticks=1440)`. Each halving is a *different*
  request from the one before it (the delta multiplies the operation weights,
  so three of them are x0.125 where one is x0.5), each measures strictly better
  than the piece in hand, and the band only binds once the motif is stable
  enough for it to — which is why the band is skipped as `no_gain` in the first
  three rounds and kept in the fourth. So the loop is not spinning, but it is
  trading melodic variation for a smaller worst leap, and three halvings is past
  the point a musician would stop. That is worth recording against Phase F4's
  own goal of measurable motivic development rather than against this module:
  the loop is doing what the table says, and the table has one knob for this bar
  that a repair can turn more than once.
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
chosen for rather than a claim about it — 54 of the 90 pieces are clean before
it runs, 33 of the 36 breaching ones come out clean, two are improved and still
missing a bar, and one is left as it was. Thirty of the repaired pieces need one
kept request, four need two and one needs three, so `maximum` never binds *here*;
across all 88 trials, 76 outcomes are `kept`, 10 are `no_gain` and 2 are
`unplayable`.

The corpus is not the whole picture, and the gap between the two is worth the
sentence. A corpus of 90 pieces says nothing about a piece it does not contain,
so the depth was re-measured over a wider grid — three moods, seven durations
and forty seeds, every piece pinned to C, 840 pieces — where the deepest chain
needed **five** kept requests and three pieces needed it. That is what
`MAX_REPAIRS_PER_TURN` is set from, and it is one above rather than equal to it
for the reason this module states about `maximum`: a backstop that is reached is
not a backstop.

**All ten `no_gain` trials left the bar they aimed at still missed**, in all
2592 requests swept, which is worth stating because the other reading is the one
that sounds more likely: a candidate that clears its bar and still loses on the
piece as a whole. Nothing writes a sentence about that case — not because it is
impossible, but because two sweeps wide enough to find it did not, and a message
branch describing a state nothing has reached is a claim the tests cannot hold
up. What the refusal says instead is what the attempts recorded: which requests
were tried and that none was kept.

The loop itself: aim at the worst miss by the arbiter's own tier order, try
every request the table holds for that bar, keep the one that measures best on
the arbiter's total order, and stop when nothing measures strictly better. It
is deterministic end to end — the candidates are literals, `compose` is a
function, and a tie between two candidates goes to the table's order — and it
spends no model calls and no audio. Its whole cost is bounded by `maximum`
times the table's width in composes of arithmetic — twelve composes, and six
seconds, at the 600-second cap where a compose measures 0.53 s, and under a
second at the durations the deep chains actually occur, since the deepest one
found is a 180-second piece — which is the reason nothing here touches the
turn's ledger: the ledger counts what is expensive (sketches and model calls),
not what is cheap.

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
    needs more than three kept requests, and over the wider grid the deepest is
    five.
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
