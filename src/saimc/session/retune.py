"""The retuning proposal: what the preference log says about the repair table.

`repairs._REPAIRS` is four bars x two requests, in an order that settles a tie
between them, and it was chosen by measuring a corpus. A corpus is a snapshot of
the pieces someone thought to compose; the preference log is the only record of
the pieces a *user* asked for and then had an opinion about. This module reads
the log and says what it holds about the table.

**It says it, and a person edits the table.** Retuning the defaults from
aggregate likes is a change every user feels, and the plan keeps it
human-gated (Open Question 5), for the reason `repairs` gives every request in
the table: a repair is the product's own arithmetic over a piece, and which
arithmetic it reaches for first is a judgement a list of percentages cannot
make. So there is no `apply` here and nothing writes `_REPAIRS`; the proposal is
printed, and the same division holds as between `saimc-benchmark` and
`MODELS.md`.

Three things it reports, and the unit each one counts:

1. **Every row of the table, ranked by the acceptance of the requests it
   holds** — plus the order that ranking implies, printed beside the table's
   own, because a proposal to reorder is a proposal to change which repair a
   piece gets and not merely which one is printed first.
2. **Requests a repair served that the table does not hold**, which are
   candidates to add. Restricted to `requests_source == "repair"` on purpose: a
   table entry is drawn from the vocabulary the *product* authors, so a request
   the user typed and liked is evidence about the music but not a candidate
   here — adding it would be proposing that a repair second-guess a request the
   user made, which is the one thing a repair may never do.
3. **Table entries no judgement carries**, which is the ordinary case for most
   of the table rather than a finding about it, and is reported so that a
   request nothing has ever needed is visible as such instead of looking as
   well-evidenced as the ones beside it.

**There is a larger gap than the third item can show, and it is not this
module's to report.** The table holds four of the ten bars the arbiter ranks, so
six bars have no request at all — and over `repairs`' own 840-piece grid 18
pieces end stopped `unmapped`: `leap_ratio` on twelve, `step_ratio` on five and
`distinct_durations` on one. That is a vocabulary addition rather than a
reordering, and it is invisible from here for the reason below — a row carries a
request and a verdict and no bar — so it is recorded in `repairs`' notes, beside
the table a person editing it is looking at, rather than inferred from a log
that cannot see it.

**A judgement is `(draft_id, at)`, and it votes once per request it carries.**
One verdict writes one row per request in the judged draft's chain, all of them
carrying that verdict and that `at` (`models.preference_rows`). Counting rows
would weight one opinion by the length of the chain it happened to be given, so
a fan-out would outvote a single revision and a piece with three requests would
be three times as loud as a piece with one. Counting distinct requests rather
than distinct rows also stops a chain that carries the same request twice — the
band re-tried onto a chain that already states it — from voting twice, which is
why each judgement contributes its *set* of requests. And the key is both
fields and not `draft_id` alone, because a draft can be revised and judged
again, and the second opinion would otherwise be folded into the first.

**What this cannot say, and each is a limit rather than a defect.**
The row records what was asked for and what the listener thought of the piece,
so a verdict is evidence for every request in its chain and the proposal cannot
tell which of them earned it. No bar is attached to a row at all — nothing in
the log says which threshold a piece missed — so this ranks requests by the
verdict on the pieces they appear in and never by the metric they were aimed at,
which is why it reports *the table* rather than "the best repair for
`leap_ratio`". And a request with no evidence is silent rather than
discredited: the log holds what users asked for, and nothing in it can say a
request the table holds does not work.

The residual collision is the log's own known limit seen from here: `at` is
generated server-side, so two verdicts on one draft in the same microsecond
share a key and are read as one opinion.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime

from saimc.canonical import canonical_dumps
from saimc.session.deltas import Delta, delta_to_dict
from saimc.session.models import VerdictValue
from saimc.session.preferences import Preference
from saimc.session.repairs import repair_table

REPAIR_SOURCE = "repair"
"""The `requests_source` a request the repair loop authored carries.

Named here rather than spelled inline, because the restriction it makes is the
difference between the second section being candidates to add and being every
request any user ever made.
"""


@dataclass(frozen=True)
class Candidate:
    """One request, and the evidence the log holds for it.

    Frozen, and its rate is a property rather than a field: a stored rate is a
    second value that can disagree with the two counts it was derived from,
    which is the rule every record here follows.
    """

    delta: Delta
    likes: int
    dislikes: int

    @property
    def judgements(self) -> int:
        """How many distinct opinions are behind these counts."""
        return self.likes + self.dislikes

    @property
    def acceptance(self) -> float | None:
        """Likes as a share of the judgements that carried this request.

        `None` for "nothing measured it", the convention the scorecard already
        uses — a request no user has ever been served has no rate, and `0.0`
        would say every user who saw it disliked it.
        """
        if self.judgements == 0:
            return None
        return self.likes / self.judgements

    @property
    def rank(self) -> tuple[int, float, int]:
        """The sort key, smallest first: measured before unmeasured, then best
        rate, then the most evidence.

        The first element is a flag rather than the rate itself because there is
        no rate to put there, and a `-1.0` invented for it would outrank a
        request every user disliked. Ties fall through to whatever order the
        caller passed, which is how the table's own order settles them.
        """
        if self.judgements == 0:
            return (1, 0.0, 0)
        return (0, -self.likes / self.judgements, -self.judgements)


@dataclass(frozen=True)
class MetricProposal:
    """One bar of the table: what it holds, and the order the log implies.

    `tabled` and `proposed` are both printed by the CLI, and `proposed` is a
    property of `ranked` rather than a third field, so the two spellings of the
    order cannot come apart.
    """

    metric: str
    tabled: tuple[Delta, ...]
    ranked: tuple[Candidate, ...]

    @property
    def proposed(self) -> tuple[Delta, ...]:
        """The table's requests in the order their evidence puts them."""
        return tuple(candidate.delta for candidate in self.ranked)

    @property
    def moved(self) -> bool:
        """Whether the evidence would reorder this bar."""
        return self.proposed != self.tabled


@dataclass(frozen=True)
class Proposal:
    """Everything the log says about the table, and nothing it does not.

    `requests` is carried beside `judgements` deliberately: the two are the
    counts a reader would otherwise have to take on trust, and printing both is
    what makes "one opinion per judgement" visible rather than a claim in a
    docstring.
    """

    judgements: int
    requests: int
    metrics: tuple[MetricProposal, ...]
    unheld: tuple[Candidate, ...]

    @property
    def unproven(self) -> tuple[tuple[str, Delta], ...]:
        """`(bar, request)` for every table entry no judgement carries.

        A property rather than a field, so it cannot disagree with the `ranked`
        entries it is a reading of, and a pair rather than a request because the
        table is keyed by bar: the same request can be held for two bars, and
        naming only the request would report it twice with nothing to say why.
        """
        return tuple(
            (proposal.metric, candidate.delta)
            for proposal in self.metrics
            for candidate in proposal.ranked
            if candidate.judgements == 0
        )

    @property
    def reordered(self) -> tuple[MetricProposal, ...]:
        """The bars the evidence would reorder, which are the ones to read."""
        return tuple(proposal for proposal in self.metrics if proposal.moved)


@dataclass
class _Tally:
    """Two counters for one request, built up as the judgements are walked."""

    likes: int = 0
    dislikes: int = 0

    def vote(self, verdict: VerdictValue) -> None:
        """Record one judgement's opinion of the request."""
        if verdict == "like":
            self.likes += 1
        else:
            self.dislikes += 1


def _key(delta: Delta) -> str:
    """A request's identity, as the log itself writes it.

    The canonical document rather than the dataclass: it is hashable whatever a
    delta's fields hold, it is the bytes the row was stored as, and two
    instances that differ in nothing are one request here by the same rule that
    makes them one line in the file.
    """
    return canonical_dumps(delta_to_dict(delta))


def _verdict_of(draft_id: str, at: datetime, rows: Sequence[Preference]) -> VerdictValue:
    """The one verdict a judgement's rows agree on.

    One verdict writes all of them, so disagreeing rows are a hand-edited log
    rather than a state this build can produce — and reading them as "the last
    one wins" would count an opinion nobody gave. Refused by name, the way a
    malformed line is refused one layer down.
    """
    verdicts = {row.verdict for row in rows}
    if len(verdicts) > 1:
        raise ValueError(
            f"draft {draft_id} was judged {' and '.join(sorted(verdicts))} at one moment "
            f"({at.isoformat()}); a judgement has one verdict"
        )
    return verdicts.pop()


def _evidence(measured: Mapping[str, Candidate], delta: Delta) -> Candidate:
    """What the log holds about `delta`, or a candidate with no evidence."""
    known = measured.get(_key(delta))
    return known if known is not None else Candidate(delta=delta, likes=0, dislikes=0)


def _rank(candidates: Iterable[Candidate]) -> tuple[Candidate, ...]:
    """Best evidence first, the order the caller passed settling ties.

    `sorted` is stable and `Candidate.rank` puts every unmeasured request in one
    equal group, so a request the log says nothing about keeps the place the
    table gives it. That is what makes this a proposal to reorder the table
    *where the log has something to say* rather than a reshuffle on no evidence.
    Which is also why the caller's order has to be one: the table's is, and the
    candidates drawn out of a set are sorted before they arrive here — by their
    canonical spelling, which is what makes the order a tie falls through to a
    property of the log rather than of the interpreter's per-run hash salt.
    """
    return tuple(sorted(candidates, key=lambda candidate: candidate.rank))


def proposal(rows: Iterable[Preference]) -> Proposal:
    """Read `rows` as evidence about the table the repair loop reads.

    The table is read here rather than passed in, because the proposal is about
    the table in force: a caller free to hand it a different one would be asking
    what the log says about a table no piece is repaired with.
    """
    table = repair_table()
    tabled_keys = {_key(delta) for candidates in table.values() for delta in candidates}

    grouped: dict[tuple[str, datetime], list[Preference]] = {}
    for row in rows:
        grouped.setdefault((row.draft_id, row.at), []).append(row)

    counts: dict[str, _Tally] = {}
    seen: dict[str, Delta] = {}
    served_by_a_repair: set[str] = set()
    for (draft_id, at), judged in grouped.items():
        verdict = _verdict_of(draft_id, at, judged)
        for row in judged:
            key = _key(row.delta)
            seen.setdefault(key, row.delta)
            if row.requests_source == REPAIR_SOURCE:
                served_by_a_repair.add(key)
        # One vote per request, from the judgement and not from the rows: a
        # chain that carries the same request twice is one opinion about it.
        for key in {_key(row.delta) for row in judged}:
            counts.setdefault(key, _Tally()).vote(verdict)

    measured = {
        key: Candidate(delta=delta, likes=counts[key].likes, dislikes=counts[key].dislikes)
        for key, delta in seen.items()
    }

    return Proposal(
        judgements=len(grouped),
        requests=sum(len(judged) for judged in grouped.values()),
        metrics=tuple(
            MetricProposal(
                metric=metric,
                tabled=tuple(candidates),
                ranked=_rank(_evidence(measured, delta) for delta in candidates),
            )
            for metric, candidates in table.items()
        ),
        # Sorted by the request's own spelling before it is ranked, because the
        # set is not an order and the ranking only settles ties by the order it
        # is given. Without this the same log prints differently in two runs.
        unheld=_rank(
            measured[key] for key in sorted(served_by_a_repair) if key not in tabled_keys
        ),
    )


__all__ = [
    "REPAIR_SOURCE",
    "Candidate",
    "MetricProposal",
    "Proposal",
    "proposal",
]
