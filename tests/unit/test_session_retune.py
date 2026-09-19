"""Unit tests for the retuning proposal.

Two things are being pinned, and they are different kinds of claim. The first
is arithmetic: what a judgement is, how many votes it casts, and which order the
evidence puts the table in. The second is the *reason* the arithmetic is the one
it is — that a chain is one opinion however long it is, and that a request the
user typed is not a candidate for the repair table — because those are the
decisions a later reader would otherwise re-litigate from the numbers alone.

The cases are built from the real table rather than from a fixture table: the
proposal is about the table `repair_chain` reads, so a case that ranked a
stand-in would prove something about a table no piece is repaired with.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import get_args

import pytest

from saimc.session import retune
from saimc.session.deltas import (
    Delta,
    RequestSource,
    SetAccompanimentDensity,
    SetHarmonyTexture,
    SetMelodyBand,
    SetMotifVariation,
)
from saimc.session.models import VerdictValue
from saimc.session.preferences import Preference
from saimc.session.repairs import repair_table
from saimc.session.retune import Candidate, Proposal, proposal

_NOW = datetime(2026, 9, 1, 12, 0, tzinfo=UTC)
_PLAN_HASH = "0" * 64

# The two requests `harmony_pad_coverage` holds, in the order the table holds
# them. Named because several cases are about the pair being reordered.
_PAD = SetHarmonyTexture(broken_chord=False)
_DENSITY = SetAccompanimentDensity(step_ticks=1440)


def _rows(
    draft_id: str,
    requests: Sequence[Delta],
    *,
    verdict: VerdictValue = "like",
    at: datetime = _NOW,
    source: RequestSource | None = None,
) -> tuple[Preference, ...]:
    """The rows one verdict writes for a chain, as the writer writes them.

    One row per request, every row carrying that verdict and that moment — so a
    case that means "one judgement" has to build a chain and cannot build a row.
    """
    return tuple(
        Preference(
            draft_id=draft_id,
            at=at,
            plan_hash=_PLAN_HASH,
            delta=request,
            verdict=verdict,
            requests_source=source,
        )
        for request in requests
    )


def _evidence(report: Proposal, metric: str, delta: Delta) -> Candidate:
    """What one bar of the table holds about one request."""
    bar = next(held for held in report.metrics if held.metric == metric)
    return next(candidate for candidate in bar.ranked if candidate.delta == delta)


def _bar(report: Proposal, metric: str) -> retune.MetricProposal:
    return next(held for held in report.metrics if held.metric == metric)


class TestTheCountingRule:
    def test_one_judgement_votes_once_per_request_it_carries(self) -> None:
        """A three-request chain is one opinion about each of the three."""
        report = proposal(_rows("draft-one", (_PAD, _DENSITY, SetMelodyBand(semitones=12))))
        assert report.judgements == 1
        assert report.requests == 3
        assert _evidence(report, "harmony_pad_coverage", _PAD) == Candidate(_PAD, 1, 0)

    def test_a_chain_three_long_is_not_three_times_as_loud(self) -> None:
        """The reading row-counting would give, beside the one this makes.

        One judgement liked a chain of one; one judgement disliked a chain of
        three whose first request is the same. Counting rows would report a
        quarter — one like against three dislikes — where counting judgements
        reports a half.
        """
        report = proposal(
            (*_rows("draft-one", (_PAD,)), *_rows("draft-two", (_PAD, _DENSITY, _DENSITY), verdict="dislike"))
        )
        assert _evidence(report, "harmony_pad_coverage", _PAD) == Candidate(_PAD, 1, 1)

    def test_the_same_request_twice_in_one_chain_is_one_opinion(self) -> None:
        """A chain can carry a request it already carries, and that is one vote.

        The repair loop re-tries a request onto a chain that already states it —
        recorded in `repairs`' own notes as the middle round of the deepest
        chains — so this is a shape the loop really produces.
        """
        twice = SetMotifVariation(factor=0.5)
        report = proposal(_rows("draft-one", (twice, twice)))
        assert report.requests == 2
        assert _evidence(report, "max_leap_semitones", twice) == Candidate(twice, 1, 0)

    def test_two_verdicts_on_one_draft_are_two_judgements(self) -> None:
        """Why the key is `(draft_id, at)` and not `draft_id` alone.

        A draft can be revised and judged again, and folding the second opinion
        into the first would let one draft be liked and disliked at once.
        """
        report = proposal(
            (
                *_rows("draft-one", (_PAD,)),
                *_rows("draft-one", (_PAD,), verdict="dislike", at=_NOW + timedelta(minutes=5)),
            )
        )
        assert report.judgements == 2
        assert _evidence(report, "harmony_pad_coverage", _PAD) == Candidate(_PAD, 1, 1)

    def test_a_verdict_reaches_every_request_in_its_chain(self) -> None:
        """The verdict is about the piece, so it is evidence for the whole chain."""
        report = proposal(
            _rows("draft-one", (_PAD, _DENSITY, SetMelodyBand(semitones=12)), verdict="dislike")
        )
        assert _evidence(report, "harmony_pad_coverage", _PAD) == Candidate(_PAD, 0, 1)
        assert _evidence(report, "harmony_pad_coverage", _DENSITY) == Candidate(_DENSITY, 0, 1)

    def test_an_empty_log_proposes_nothing_about_anything(self) -> None:
        report = proposal(())
        assert report.judgements == 0
        assert report.requests == 0
        assert report.unheld == ()
        assert len(report.unproven) == sum(len(held) for held in repair_table().values())
        assert report.reordered == ()


class TestTheRanking:
    def test_the_better_acceptance_comes_first(self) -> None:
        """The reorder the whole report exists to make visible."""
        report = proposal(
            (
                *_rows("draft-one", (_PAD, _PAD, _PAD), verdict="dislike"),
                *_rows("draft-two", (_DENSITY,)),
            )
        )
        bar = _bar(report, "harmony_pad_coverage")
        assert bar.tabled == (_PAD, _DENSITY)
        assert bar.proposed == (_DENSITY, _PAD)
        assert bar.moved is True

    def test_more_evidence_settles_an_equal_rate(self) -> None:
        """Two requests at a half each, and the better-attested one leads."""
        report = proposal(
            (
                *_rows("draft-one", (_PAD,)),
                *_rows("draft-two", (_PAD,), verdict="dislike", at=_NOW + timedelta(minutes=1)),
                *_rows("draft-three", (_DENSITY,)),
                *_rows("draft-four", (_DENSITY,), verdict="dislike", at=_NOW + timedelta(minutes=2)),
                *_rows("draft-five", (_DENSITY,)),
                *_rows("draft-six", (_DENSITY,), verdict="dislike", at=_NOW + timedelta(minutes=3)),
            )
        )
        first = _evidence(report, "harmony_pad_coverage", _PAD)
        second = _evidence(report, "harmony_pad_coverage", _DENSITY)
        assert first.acceptance == second.acceptance == 0.5
        assert first.judgements == 2
        assert second.judgements == 4
        assert _bar(report, "harmony_pad_coverage").proposed == (_DENSITY, _PAD)

    def test_the_tables_own_order_settles_a_tie(self) -> None:
        """Equal rate and equal evidence is not a reason to reorder."""
        report = proposal((*_rows("draft-one", (_PAD,)), *_rows("draft-two", (_DENSITY,))))
        bar = _bar(report, "harmony_pad_coverage")
        assert bar.proposed == bar.tabled
        assert bar.moved is False
        assert report.reordered == ()

    def test_an_unmeasured_request_does_not_displace_a_measured_one(self) -> None:
        """No evidence is not a rate, and it cannot outrank one.

        The table leads with `SetMelodyBand(semitones=9)` under
        `max_leap_semitones`; the log has only ever seen the other request, so
        the measured one leads.
        """
        narrow = SetMelodyBand(semitones=9)
        variation = SetMotifVariation(factor=0.5)
        report = proposal(_rows("draft-one", (variation,)))
        bar = _bar(report, "max_leap_semitones")
        assert bar.tabled == (narrow, variation)
        assert bar.proposed == (variation, narrow)
        assert _evidence(report, "max_leap_semitones", narrow).acceptance is None

    def test_requests_with_no_evidence_keep_the_tables_order_among_themselves(self) -> None:
        report = proposal(())
        bar = _bar(report, "max_leap_semitones")
        assert bar.proposed == bar.tabled

    def test_every_bar_is_read_from_the_table_the_loop_reads(self) -> None:
        report = proposal(())
        table = repair_table()
        assert [bar.metric for bar in report.metrics] == list(table)
        assert {bar.metric: bar.tabled for bar in report.metrics} == table

    def test_the_same_log_proposes_the_same_thing_in_any_row_order(self) -> None:
        """Determinism, and it is not free: `unheld` is drawn out of a set.

        Two logs holding the same judgements written in opposite orders have to
        propose the same report, which is why the candidates are sorted before
        they are ranked — a set's iteration order is not an order.
        """
        rows = (
            *_rows("draft-one", (SetMelodyBand(semitones=12),), source="repair"),
            *_rows("draft-two", (SetMelodyBand(semitones=13),), source="repair", at=_NOW + timedelta(minutes=1)),
            *_rows("draft-three", (SetMelodyBand(semitones=14),), source="repair", at=_NOW + timedelta(minutes=2)),
        )
        assert proposal(rows) == proposal(tuple(reversed(rows)))

    def test_candidates_with_nothing_between_them_are_ordered_by_their_spelling(self) -> None:
        """The tie is settled by the request, and not left to the interpreter.

        A set's iteration order is not an order and is not stable *between
        processes* either — string hashing is salted per run — so three
        equal-evidence candidates would be listed in whatever order the salt
        happened to give them and the same log would print two different
        reports. `saimc-preferences` is read by a person editing a table and
        diffed between runs, so the order has to be a property of the log rather
        than of the run, and the request's own canonical spelling is what the
        rows are keyed by and therefore what settles it.

        The rows arrive descending, which is the whole of the case: the two
        orders the code could produce differ, so a report that skipped the sort
        is visibly a different report. The three requests are off the table, so
        all three are candidates to add rather than one of them being a bar's own
        entry.
        """
        offered = (
            SetMelodyBand(semitones=12),
            SetMelodyBand(semitones=11),
            SetMelodyBand(semitones=10),
        )
        report = proposal(
            tuple(
                row
                for index, delta in enumerate(offered)
                for row in _rows(
                    f"draft-{index}", (delta,), source="repair", at=_NOW + timedelta(minutes=index)
                )
            )
        )
        assert [candidate.delta for candidate in report.unheld] == [
            SetMelodyBand(semitones=10),
            SetMelodyBand(semitones=11),
            SetMelodyBand(semitones=12),
        ]


class TestTheCandidatesToAdd:
    def test_a_request_a_repair_served_and_the_table_lacks_is_a_candidate(self) -> None:
        offered = SetMelodyBand(semitones=12)
        report = proposal(_rows("draft-one", (offered,), source="repair"))
        assert [candidate.delta for candidate in report.unheld] == [offered]
        assert report.unheld[0].acceptance == 1.0

    def test_a_request_the_user_asked_for_is_not_a_candidate(self) -> None:
        """The restriction that keeps this from proposing the product overrule a user.

        A table entry is drawn from the vocabulary the *product* authors. A
        request the user typed and liked is evidence about the music, and
        offering it as a repair would be proposing that a repair ask for what
        the user already asked for.
        """
        typed = SetMelodyBand(semitones=12)
        report = proposal(_rows("draft-one", (typed,), source="typed"))
        assert report.unheld == ()

    def test_a_request_the_table_already_holds_is_not_a_candidate(self) -> None:
        report = proposal(_rows("draft-one", (_PAD,), source="repair"))
        assert report.unheld == ()

    def test_candidates_are_ranked_with_the_accepted_one_first(self) -> None:
        rejected = SetMelodyBand(semitones=12)
        accepted = SetMelodyBand(semitones=13)
        report = proposal(
            (
                *_rows("draft-one", (rejected,), verdict="dislike", source="repair"),
                *_rows("draft-two", (accepted,), source="repair", at=_NOW + timedelta(minutes=1)),
            )
        )
        assert [candidate.delta for candidate in report.unheld] == [accepted, rejected]

    def test_a_repair_served_request_is_one_request_however_it_was_reached(self) -> None:
        """The source is the step's, and a step is one row of the chain."""
        offered = SetMelodyBand(semitones=12)
        report = proposal(
            (
                *_rows("draft-one", (_PAD,), source="repair"),
                *_rows("draft-one", (offered,), source="repair", at=_NOW + timedelta(minutes=1)),
            )
        )
        assert [candidate.delta for candidate in report.unheld] == [offered]


class TestWhatTheTableSaysNothingAbout:
    def test_a_request_in_no_judgement_is_named_as_unproven(self) -> None:
        report = proposal(_rows("draft-one", (_PAD,)))
        assert ("harmony_pad_coverage", _DENSITY) in report.unproven
        assert ("harmony_pad_coverage", _PAD) not in report.unproven

    def test_the_unproven_entries_name_their_bar(self) -> None:
        """A request two bars both hold is two entries, and the bar says which.

        `SetMelodyBand(semitones=9)` is held for two bars, so naming only the
        request would report the same line twice with nothing to explain it.
        """
        report = proposal(())
        bars = [metric for metric, _ in report.unproven]
        assert bars.count("max_leap_semitones") == 2
        assert bars.count("leap_recovery_ratio") == 2


class TestTheRefusals:
    def test_a_judgement_whose_rows_disagree_is_refused_by_name(self) -> None:
        """A hand-edited log, which is the only way this state is reached.

        One verdict writes all of a judgement's rows, so a disagreement is not a
        state this build produces — and reading it as "the last row wins" would
        count an opinion nobody gave.
        """
        rows = (
            *_rows("draft-one", (_PAD,)),
            *_rows("draft-one", (_DENSITY,), verdict="dislike"),
        )
        with pytest.raises(ValueError, match="a judgement has one verdict"):
            proposal(rows)


class TestTheReportItself:
    def test_the_module_exports_what_it_claims(self) -> None:
        assert retune.__all__ == [
            "REPAIR_SOURCE",
            "Candidate",
            "MetricProposal",
            "Proposal",
            "proposal",
        ]

    def test_the_source_a_repair_writes_is_the_vocabularys_own_spelling(self) -> None:
        assert retune.REPAIR_SOURCE in get_args(RequestSource)
