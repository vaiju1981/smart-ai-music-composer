"""The session gate: what the harness shipped, judged as a release.

Three properties carry the weight here.

- **The gate reads the search, not only the winner.** The release gate over
  the engine cannot see how many candidates a session tried, so "the shipped
  piece is clean" would be a claim about a search rather than about the
  generator. The detail carries both numbers, and the test for that asserts
  the *rate*, which is the half a winner-only gate cannot produce.
- **A failure names the miss and the alternative.** "Two thresholds missed"
  is not actionable; the finding's own line is, and naming a sibling that
  would have cleared it is the sentence the fan-out exists to make possible.
- **The gate does not gate the publish.** Open Question 4 of the plan settles
  that a user may ship a piece the scorecard dislikes, so the test that
  matters is behavioural: a draft that misses bars still publishes.

The fixtures compose for real and publish through the real tool, so the
scorecards and the publication are the ones the product writes.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import saimc.jobs.worker
from saimc.jobs.storage import JobStorage
from saimc.llm.base import ToolCall
from saimc.quality import QUALITY_THRESHOLDS
from saimc.session.gates import gate_session_publication
from saimc.session.models import Publication, Session
from saimc.session.store import SessionStorage
from saimc.session.tools import ToolContext, dispatch
from saimc.spec import CompositionSpec, Mood

_SLEEP = CompositionSpec(mood=Mood.SLEEP, duration_seconds=60, seed=0)
"""A spec whose four-wide fan-out is mixed: one seed misses, three are clean.

Measured rather than assumed — a sweep of twelve seeds at each mood and each
of 30/60/120 s puts `sleep`'s sixty-second pieces at `X....X.X.X..`, and the
fan-out composes `spec.seed + offset`, so the candidates here are seeds 0
through 3 and it is **the first** of them that misses. The gate's two branches
need a session that holds both a clean candidate and an unclean one, and this
is the smallest fixture that really does.

The breach profile is a fact about the keys these seeds land on, which is the
mood's own pool rather than a constant — `sleep`'s sixty-second pieces were
`.X....X..X..` under the old C-major resolution and the missing seed moved with
the key. So the ids below are the measured ones and the sweep above is
re-derived whenever they are re-read, rather than the fixture pinning a key to
keep an older reading standing.
"""

_ELECTRIFYING = CompositionSpec(mood=Mood.ELECTRIFYING, duration_seconds=30, seed=0)
"""A spec whose candidates *all* miss `texture_hierarchy`, at every seed.

The other half of the alternative's branch: when nothing in the session would
have cleared the bars, the failure must not invent a sibling to name."""


def _call(ctx: ToolContext, name: str, **args: Any) -> Any:
    return asyncio.run(dispatch(ToolCall(name=name, arguments=args), ctx))


@pytest.fixture
def ctx(tmp_path) -> ToolContext:
    sessions = SessionStorage(tmp_path / "sessions")
    jobs = JobStorage(tmp_path / "jobs")
    session: Session = sessions.create("something for a rainy day")
    return ToolContext(session=session, sessions=sessions, jobs=jobs)


@pytest.fixture(autouse=True)
def _queued(monkeypatch: pytest.MonkeyPatch) -> None:
    """The broker is not part of what this module judges."""
    monkeypatch.setattr(saimc.jobs.worker, "enqueue_job", lambda *_a, **_k: "rq:test")


def _draft_and_publish(ctx: ToolContext, spec: CompositionSpec, *, chosen: str) -> None:
    """Fan out four candidates at `spec`, then publish exactly one of them."""
    ctx.session.spec = spec
    assert _call(ctx, "draft", n=4).ok
    assert _call(ctx, "finalize", draft_id=chosen).ok


class TestTheGateJudgesAPublication:
    def test_an_unpublished_session_fails_rather_than_passing_vacuously(
        self, ctx: ToolContext
    ) -> None:
        """`gate_musical_quality`'s rule for an empty matrix, one level up: a
        bar with nothing to judge is not a bar that was cleared."""
        result = gate_session_publication(ctx.session)
        assert result.name == "session_publication"
        assert result.passed is False
        assert "has not published" in result.detail

    def test_a_published_clean_piece_passes(self, ctx: ToolContext) -> None:
        _draft_and_publish(ctx, _SLEEP, chosen="draft-1")
        result = gate_session_publication(ctx.session)
        assert result.passed is True
        assert f"clears all {len(QUALITY_THRESHOLDS)} thresholds" in result.detail

    def test_the_detail_reports_the_fan_out_and_not_only_the_winner(self, ctx: ToolContext) -> None:
        """The number the engine-side gate cannot produce.

        Three of these four candidates are clean — seed 0 misses — so a gate
        that reported the published draft alone would say the same thing about
        a session that searched four times and one that searched once. The
        premise is asserted rather than trusted: if the fixture's fan-out
        stopped being mixed, this test would be asserting a rate that a
        winner-only gate could also have written.
        """
        _draft_and_publish(ctx, _SLEEP, chosen="draft-1")
        assert [len(draft.quality.findings()) > 0 for draft in ctx.session.drafts] == [
            True,
            False,
            False,
            False,
        ]
        assert "3/4 of this session's candidates cleared every threshold" in (
            gate_session_publication(ctx.session).detail
        )


class TestAFailingPublication:
    def test_the_failure_names_the_miss_with_the_finding_s_own_line(self, ctx: ToolContext) -> None:
        """`QualityFinding.message()` carries the measured value, the bar and
        the hint — so the gate's detail is something a critic could act on,
        where a count of missed thresholds is not."""
        _draft_and_publish(ctx, _SLEEP, chosen="draft-0")
        result = gate_session_publication(ctx.session)
        assert result.passed is False
        published = ctx.session.draft("draft-0")
        for finding in published.quality.findings():
            assert finding.message() in result.detail

    def test_the_failure_names_a_candidate_that_would_have_cleared_it(
        self, ctx: ToolContext
    ) -> None:
        """The search made visible: the session holds a draft that clears
        every bar and published one that does not, and the report says so."""
        _draft_and_publish(ctx, _SLEEP, chosen="draft-0")
        detail = gate_session_publication(ctx.session).detail
        assert "including 'draft-1', which was not published" in detail

    def test_no_alternative_is_named_when_no_candidate_cleared_it(self, ctx: ToolContext) -> None:
        """Every candidate here misses, so there is no sibling to name — and
        inventing one would be the report claiming the session had a choice it
        never had."""
        _draft_and_publish(ctx, _ELECTRIFYING, chosen="draft-0")
        result = gate_session_publication(ctx.session)
        assert result.passed is False
        assert [
            draft.draft_id for draft in ctx.session.drafts if not draft.quality.findings()
        ] == []
        assert "which was not published" not in result.detail

    def test_the_alternative_named_is_stable_across_reads(self, ctx: ToolContext) -> None:
        """First in the session's own order — the order the candidates were
        drafted in — so two reports of one session cannot disagree about which
        draft they are pointing at."""
        _draft_and_publish(ctx, _SLEEP, chosen="draft-0")
        assert gate_session_publication(ctx.session).detail == (
            gate_session_publication(ctx.session).detail
        )


class TestTheGateDoesNotGateThePublish:
    def test_a_draft_the_scorecard_dislikes_still_publishes(self, ctx: ToolContext) -> None:
        """Open Question 4, pinned as behaviour rather than as a docstring.

        Correctness (the linter) is non-negotiable; taste is the user's, and
        the scorecard is a proxy for taste. So a session may publish a piece
        this gate would fail — and the day someone wires the gate into the
        publish path, this fails rather than the decision silently reversing.
        """
        _draft_and_publish(ctx, _ELECTRIFYING, chosen="draft-0")
        assert gate_session_publication(ctx.session).passed is False
        assert ctx.session.publication is not None
        assert ctx.jobs.get(ctx.session.publication.job_id).input_spec is not None

    def test_the_publication_names_the_draft_it_published(self, ctx: ToolContext) -> None:
        """The pair the gate reads, written by the publish and nothing else."""
        _draft_and_publish(ctx, _SLEEP, chosen="draft-2")
        assert ctx.session.publication == Publication(
            job_id=ctx.session.finalized_job_id or "", draft_id="draft-2"
        )
