"""The tools, the dispatcher, and the budgets they cannot talk their way past.

Four properties carry the weight here.

- **A tool refuses rather than no-ops, and the log tells that apart from a
  crash.** Both are answers, both are recorded, and the reason code is the
  thing a conductor branches on — so each is tested through `dispatch` rather
  than by calling a handler, because "the call answered" is the property.
- **A budget is enforced, not advertised.** The test that matters is not that
  the schema says 4; it is that the schema says whatever the budget says *and*
  the code accepts exactly that. A non-default budget is run end to end.
- **The same arguments produce the same music.** Drafting twice at one seed in
  one session yields two drafts whose performance hashes are equal, which is
  the whole of what makes a recorded turn replayable.
- **A finalized session is durable at the moment it publishes.** The record
  that links the session to the job is written by the tool, not left to the
  end of a turn that might not arrive.

The fixtures compose for real — 30 seconds of music, so a fan-out of four is
still fast — and only the outer edges are faked: the model (which no test here
has), FluidSynth and the font resolver (which are `render/`'s own tests), and
the broker.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from typing import Any, ClassVar

import pytest

import saimc.jobs.worker
from saimc.compose.engine import CompositionEngineError, EngineErrorCode, compose
from saimc.compose.motif import BassMotion
from saimc.jobs.storage import JobStorage
from saimc.llm.base import ParseRequest, ParseResult, ToolCall
from saimc.quality import AXES
from saimc.render.audio import AudioArtifact
from saimc.session import tools
from saimc.session.arbiter import ELEMENTS, musical_key, rank
from saimc.session.deltas import SetBassMotion, SetMotifVariation, SetTempo
from saimc.session.models import Draft, Session, ToolInvocation
from saimc.session.store import SessionStorage
from saimc.session.tools import (
    MAX_CANDIDATES_PER_DRAFT,
    TOOLS,
    ToolBudget,
    ToolContext,
    ToolRefusal,
    TurnLedger,
    dispatch,
    parse_requests,
    revise_draft,
    tool_specs,
)
from saimc.spec import CompositionSpec, Mood, SpecError, VoiceRole

_SPEC = CompositionSpec(mood=Mood.CALMING, duration_seconds=30, seed=5)
_ELECTRIFYING = CompositionSpec(mood=Mood.ELECTRIFYING, duration_seconds=30, seed=16)
"""The piece whose bed breaches while the tune and the bass do not.

The critique cases need an axis that is clean beside one that is not, so the
findings have to land on the accompaniment alone — and at seed 5 this mood's
30-second piece misses the leap bar too, which puts a finding on the melody and
leaves `test_every_axis_is_reported_and_a_clean_one_says_so` asserting a state
the piece is not in. Re-found by sweeping three moods, five durations and forty
seeds for a piece whose findings are all on one axis rather than re-pinned by
key, which is this file's precedent for a fixture F3a's music moved."""
_SIXTY = CompositionSpec(mood=Mood.CALMING, duration_seconds=60, seed=5)
"""The one length a tempo test needs, and it is a length rather than a mood.

The duration search is allowed to re-derive a tempo, so whether a tempo request
survives depends on the length: a minute-long calming piece holds 80 BPM
exactly and holds nothing else in its range, while the 30-second fixture plays
at 64 BPM whatever it is asked for. Both halves of the check need a spec, and
this is the one the honoured half needs."""
_OUTPUT = compose(_SPEC)


def _call(ctx: ToolContext, name: str, **args: Any) -> ToolInvocation:
    """Dispatch one call and hand back the record of it.

    Sync, and driven with `asyncio.run`, which is this repo's precedent for
    reaching an async call from a sync test (`test_llm_config.py`): an
    `@asynccontextmanager` here leaks a loop's self-pipe past pytest-asyncio's
    teardown and, under `filterwarnings = ["error"]`, reports the failure in
    some other module's test.
    """
    return asyncio.run(dispatch(ToolCall(name=name, arguments=args), ctx))


def _payload(invocation: ToolInvocation) -> Any:
    return json.loads(invocation.result)


@pytest.fixture
def ctx(tmp_path) -> ToolContext:
    sessions = SessionStorage(tmp_path / "sessions")
    jobs = JobStorage(tmp_path / "jobs")
    session: Session = sessions.create("something for a rainy day")
    return ToolContext(session=session, sessions=sessions, jobs=jobs)


def _ready(ctx: ToolContext, spec: CompositionSpec = _SPEC) -> ToolContext:
    """A context whose brief has already been parsed."""
    ctx.session.spec = spec
    return ctx


def _drafts(ctx: ToolContext, **args: Any) -> list[dict[str, Any]]:
    invocation = _call(ctx, "draft", **args)
    assert invocation.ok, invocation.result
    return _payload(invocation)["drafts"]


class _FakeLLM:
    """Stands in for the adapter: it records what it was asked, and answers.

    It satisfies `LLMClient` structurally rather than by inheriting, which is
    the property the protocol is there for.
    """

    def __init__(self, *, spec: CompositionSpec | None = None) -> None:
        self.spec = spec
        self.prompts: list[str] = []

    async def parse(self, request: ParseRequest) -> ParseResult:
        self.prompts.append(request.prompt)
        if self.spec is None:
            return ParseResult(
                parser_source="llm",
                error=SpecError(error_code="llm_unreachable", message="no host", stage="parsing"),
            )
        return ParseResult(parser_source="llm", spec=self.spec)

    async def aclose(self) -> None:
        return None


class TestTheCatalogue:
    def test_the_table_and_the_names_agree(self) -> None:
        """One table, so a tool that is offered is a tool that runs."""
        for name, tool in TOOLS.items():
            assert tool.name == name

    def test_every_tool_is_offered_with_a_schema_and_a_description(self) -> None:
        specs = tool_specs()
        assert [spec.name for spec in specs] == list(TOOLS)
        for spec in specs:
            assert spec.description
            assert spec.parameters["type"] == "object"

    def test_an_argument_the_model_does_not_know_is_refused(self) -> None:
        """`additionalProperties: false` on every schema, so a model that
        invents a knob is told rather than silently obeyed."""
        for spec in tool_specs():
            assert spec.parameters["additionalProperties"] is False

    def test_the_catalogue_names_exactly_the_tools_this_phase_wrote(self) -> None:
        """A ratchet. D3 adds `revise` and `compare`, E3 adds `repair`; a later
        phase adding a tool has to say so here rather than letting it appear.

        `apply_delta` is still absent, and deliberately: C4's finding 8 left it
        out because a delta is only meaningful applied to a lineage, and `revise`
        is that whole act. The module docstring carries the same reason, and the
        same one is why `repair` is not `apply_delta` either.
        """
        assert sorted(TOOLS) == [
            "compare",
            "critique",
            "draft",
            "finalize",
            "parse_brief",
            "repair",
            "revise",
            "sketch",
        ]

    def test_the_fan_out_ceiling_is_the_budget_it_is_checked_against(self) -> None:
        """The drift this design exists to prevent: a `maximum` the schema
        advertises being a different number from the one the code enforces."""
        schema = next(s for s in tool_specs(ToolBudget(max_candidates=7)) if s.name == "draft")
        assert schema.parameters["properties"]["n"]["maximum"] == 7

    def test_the_default_ceiling_is_a_named_decision(self) -> None:
        schema = next(s for s in tool_specs() if s.name == "draft")
        assert schema.parameters["properties"]["n"]["maximum"] == MAX_CANDIDATES_PER_DRAFT


class TestTheBudget:
    def test_a_limit_that_allows_nothing_is_refused(self) -> None:
        with pytest.raises(ValueError, match="at least one call"):
            ToolBudget(max_candidates=0)

    def test_a_deadline_that_has_already_passed_is_refused(self) -> None:
        with pytest.raises(ValueError, match="deadline_seconds"):
            ToolBudget(deadline_seconds=0)

    def test_a_ledger_that_has_run_too_long_is_expired(self) -> None:
        assert TurnLedger(started=time.monotonic() - 300).expired(ToolBudget()) is True

    def test_a_fresh_ledger_is_not_expired(self) -> None:
        assert TurnLedger().expired(ToolBudget()) is False

    def test_beginning_a_turn_forgets_the_last_ones_spend(self, ctx: ToolContext) -> None:
        ctx.ledger.sketches = 3
        ctx.ledger.llm_calls = 1
        ctx.begin_turn()
        assert (ctx.ledger.sketches, ctx.ledger.llm_calls) == (0, 0)


class TestTheDispatcher:
    def test_a_tool_that_does_not_exist_is_an_answer_not_an_exception(
        self, ctx: ToolContext
    ) -> None:
        invocation = _call(ctx, "compose_the_whole_thing")
        assert invocation.outcome == "refused"
        assert invocation.error_code == "unknown_tool"
        assert "draft" in invocation.result

    def test_the_call_is_logged_with_the_arguments_it_was_made_with(self, ctx: ToolContext) -> None:
        """What makes a session replayable: the log holds the call."""
        _ready(ctx)
        invocation = _call(ctx, "draft", n=2, seed=11)
        assert invocation.name == "draft"
        assert invocation.arguments == {"n": 2, "seed": 11}
        assert invocation.duration_ms >= 0

    def test_a_refusal_carries_its_reason_code(self, ctx: ToolContext) -> None:
        invocation = _call(ctx, "critique", draft_id="draft-0")
        assert invocation.outcome == "refused"
        assert invocation.error_code == "unknown_draft"

    def test_a_refusal_is_not_a_crash(self, ctx: ToolContext) -> None:
        """The two outcomes are the product's "no" and the harness's fall —
        different sentences, and the log keeps them apart."""
        assert _call(ctx, "critique", draft_id="draft-0").outcome != "error"

    def test_an_unexpected_exception_becomes_a_recorded_crash(
        self, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A conductor that receives no reply cannot even say it failed."""
        _ready(ctx)

        def _explode(spec: Any, *, plan: Any = None) -> Any:
            raise RuntimeError("the engine fell over")

        monkeypatch.setattr(tools, "compose", _explode)
        invocation = _call(ctx, "draft")
        assert invocation.outcome == "error"
        assert invocation.error_code == "tool_crashed"
        assert "the engine fell over" in invocation.result

    def test_no_new_work_starts_past_the_deadline(self, ctx: ToolContext) -> None:
        _ready(ctx)
        ctx.ledger.started = time.monotonic() - 300
        invocation = _call(ctx, "draft")
        assert invocation.outcome == "refused"
        assert invocation.error_code == "turn_deadline"

    def test_the_deadline_is_checked_before_the_tool_runs(self, ctx: ToolContext) -> None:
        """Not merely reported afterwards: nothing was drafted."""
        _ready(ctx)
        ctx.ledger.started = time.monotonic() - 300
        _call(ctx, "draft", n=2)
        assert ctx.session.drafts == []


class TestArguments:
    def test_a_bool_is_not_an_integer(self, ctx: ToolContext) -> None:
        """`isinstance(True, int)` is true, and a `true` where a width was
        expected is a model that has misunderstood."""
        _ready(ctx)
        invocation = _call(ctx, "draft", n=True)
        assert invocation.outcome == "refused"
        assert "must be an integer" in invocation.result

    @pytest.mark.parametrize("bad", ["2", 2.5, None, [2]])
    def test_a_width_that_is_not_an_integer_is_refused_by_name(
        self, ctx: ToolContext, bad: Any
    ) -> None:
        _ready(ctx)
        invocation = _call(ctx, "draft", n=bad)
        assert invocation.outcome == "refused"
        assert "n" in invocation.result

    def test_a_width_past_the_ceiling_names_the_ceiling(self, ctx: ToolContext) -> None:
        _ready(ctx)
        invocation = _call(ctx, "draft", n=MAX_CANDIDATES_PER_DRAFT + 1)
        assert invocation.outcome == "refused"
        assert str(MAX_CANDIDATES_PER_DRAFT) in invocation.result

    def test_a_negative_seed_is_refused(self, ctx: ToolContext) -> None:
        _ready(ctx)
        assert _call(ctx, "draft", seed=-1).outcome == "refused"

    def test_a_missing_required_argument_is_refused(self, ctx: ToolContext) -> None:
        invocation = _call(ctx, "sketch")
        assert invocation.outcome == "refused"
        assert "draft_id is required" in invocation.result

    @pytest.mark.parametrize("bad", [5, "", "   ", ["draft-0"]])
    def test_an_id_that_is_not_a_usable_string_is_refused_by_name(
        self, ctx: ToolContext, bad: Any
    ) -> None:
        """An id arrives from a model that may have sent a number, or a blank.
        Either way the answer names the argument rather than raising past the
        turn — and a whitespace id is refused here rather than becoming a
        directory named `   `."""
        invocation = _call(ctx, "critique", draft_id=bad)
        assert invocation.outcome == "refused"
        assert "draft_id" in invocation.result
        assert invocation.error_code == "invalid_arguments"


class TestParseBrief:
    def test_the_parser_is_given_the_sessions_own_brief(self, ctx: ToolContext) -> None:
        client = _FakeLLM(spec=_SPEC)
        ctx.llm = client
        invocation = _call(ctx, "parse_brief")
        assert invocation.ok, invocation.result
        assert client.prompts == [ctx.session.brief]
        assert ctx.session.spec == _SPEC

    def test_a_supplied_text_is_parsed_without_rewriting_the_brief(self, ctx: ToolContext) -> None:
        """The session keeps the words the user typed; the spec tracks the
        meaning, which may have moved on."""
        client = _FakeLLM(spec=_SPEC)
        ctx.llm = client
        _call(ctx, "parse_brief", text="something calmer")
        assert client.prompts == ["something calmer"]
        assert ctx.session.brief == "something for a rainy day"

    def test_the_spec_the_model_may_read_is_the_spec_that_was_stored(
        self, ctx: ToolContext
    ) -> None:
        ctx.llm = _FakeLLM(spec=_SPEC)
        payload = _payload(_call(ctx, "parse_brief"))
        assert payload["spec"] == _SPEC.model_dump(mode="json")
        assert payload["spec"] == ctx.session.spec.model_dump(mode="json")

    def test_no_model_configured_is_a_refusal_not_a_crash(self, ctx: ToolContext) -> None:
        """The deterministic tools have to work without one, so this is the
        only tool that says so."""
        invocation = _call(ctx, "parse_brief")
        assert invocation.outcome == "refused"
        assert invocation.error_code == "llm_not_configured"

    def test_a_parser_failure_refuses_with_the_parsers_own_code(
        self, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The parser has its own retry and fallback policy and its own tests;
        what is tested here is that the tool does not invent a second one."""
        ctx.llm = _FakeLLM()

        async def _fail(*_args: Any, **_kwargs: Any) -> ParseResult:
            return ParseResult(
                parser_source="llm",
                error=SpecError(
                    error_code="out_of_vocabulary", message="no mood in there", stage="fallback"
                ),
            )

        monkeypatch.setattr(tools, "parse_prompt", _fail)
        invocation = _call(ctx, "parse_brief")
        assert invocation.outcome == "refused"
        assert invocation.error_code == "out_of_vocabulary"
        assert ctx.session.spec is None

    def test_a_turn_may_only_spend_its_model_calls(self, ctx: ToolContext) -> None:
        ctx.llm = _FakeLLM(spec=_SPEC)
        ctx.ledger.llm_calls = ctx.budget.max_llm_calls
        invocation = _call(ctx, "parse_brief")
        assert invocation.outcome == "refused"
        assert invocation.error_code == "budget_exhausted"


class TestDraft:
    def test_drafting_without_a_spec_says_what_to_do_about_it(self, ctx: ToolContext) -> None:
        invocation = _call(ctx, "draft")
        assert invocation.outcome == "refused"
        assert invocation.error_code == "no_spec"
        assert "parse_brief" in invocation.result

    def test_one_call_is_one_draft_by_default(self, ctx: ToolContext) -> None:
        assert len(_drafts(_ready(ctx))) == 1

    def test_a_fan_out_drafts_every_candidate(self, ctx: ToolContext) -> None:
        assert [d["draft_id"] for d in _drafts(_ready(ctx), n=3)] == [
            "draft-0",
            "draft-1",
            "draft-2",
        ]
        assert len(ctx.session.drafts) == 3

    def test_the_candidates_are_seeded_apart(self, ctx: ToolContext) -> None:
        seeds = [d["seed"] for d in _drafts(_ready(ctx), n=3, seed=10)]
        assert seeds == [10, 11, 12]

    def test_a_draft_id_continues_where_the_session_left_off(self, ctx: ToolContext) -> None:
        """Ids are positions in the session, so a second turn's candidates
        cannot collide with the first turn's."""
        _ready(ctx)
        _drafts(ctx, n=2)
        assert [d["draft_id"] for d in _drafts(ctx)] == ["draft-2"]

    def test_the_same_arguments_compose_the_same_music(self, ctx: ToolContext) -> None:
        """The property a recorded turn replays by: the arguments are the
        whole input, so nothing about the run has to be remembered."""
        _ready(ctx)
        _drafts(ctx, seed=5)
        _drafts(ctx, seed=5)
        first, second = ctx.session.drafts
        assert first.performance_plan_hash == second.performance_plan_hash
        assert first.draft_id != second.draft_id

    def test_different_seeds_compose_different_music(self, ctx: ToolContext) -> None:
        """The other half: a fan-out that produced identical candidates would
        be a fan-out of one."""
        _ready(ctx)
        _drafts(ctx, n=2, seed=5)
        first, second = ctx.session.drafts
        assert first.performance_plan_hash != second.performance_plan_hash

    def test_a_draft_records_the_plan_the_engine_composed_under(
        self, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Against a *non-default* plan, because the obvious version is a tautology.

        Comparing against `compose(spec).plan` proves nothing: that is the
        same value `default_plan(spec)` produces, so a `_draft_from` that
        re-derived the default instead of reading the output's plan would pass
        it. Fired as a sabotage and it did pass. So the engine here composes
        under a plan it was handed and would never have chosen, and the draft
        has to carry *that* — which is what makes the record replayable after
        the module tables move (B's whole verification bullet).
        """
        _ready(ctx)
        shaped = replace(_OUTPUT.plan, modulation_offset=9)
        drifted = compose(_SPEC, plan=shaped)
        monkeypatch.setattr(tools, "compose", lambda spec, plan=None: drifted)
        _drafts(ctx)
        assert ctx.session.drafts[0].plan == shaped
        assert ctx.session.drafts[0].plan != _OUTPUT.plan

    def test_a_draft_is_scored_and_linted(self, ctx: ToolContext) -> None:
        _ready(ctx, _ELECTRIFYING)
        payload = _drafts(ctx)[0]
        assert payload["lint_passed"] is True
        assert payload["quality"]["melody_notes"] > 0

    def test_a_draft_has_no_audio_until_it_is_sketched(self, ctx: ToolContext) -> None:
        """A candidate can be drafted, scored and compared before anyone
        listens to it, which is what makes a wide fan-out affordable."""
        _ready(ctx)
        _drafts(ctx)
        assert ctx.session.drafts[0].sketch is None
        assert _drafts(ctx)[0]["sketched"] is False

    def test_a_candidate_the_engine_cannot_honour_does_not_sink_the_others(
        self, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One bad seed out of four is worth reporting, not worth losing the
        other three over."""
        _ready(ctx)
        real = compose

        def _sometimes(spec: CompositionSpec, *, plan: Any = None) -> Any:
            if spec.seed == 7:
                raise CompositionEngineError(
                    code=EngineErrorCode.LINT_FAILED, message="score failed lint"
                )
            return real(spec, plan=plan)

        monkeypatch.setattr(tools, "compose", _sometimes)
        invocation = _call(ctx, "draft", n=3, seed=6)
        assert invocation.ok, invocation.result
        payload = _payload(invocation)
        assert [d["seed"] for d in payload["drafts"]] == [6, 8]
        assert payload["refused"] == [
            {"seed": 7, "error_code": "lint_failed", "reason": "score failed lint"}
        ]

    def test_a_fan_out_that_produced_nothing_refuses_with_the_engines_reason(
        self, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _ready(ctx)

        def _never(spec: CompositionSpec, *, plan: Any = None) -> Any:
            raise CompositionEngineError(
                code=EngineErrorCode.DURATION_UNFULFILLABLE, message="no arrangement fits"
            )

        monkeypatch.setattr(tools, "compose", _never)
        invocation = _call(ctx, "draft", n=2)
        assert invocation.outcome == "refused"
        assert invocation.error_code == "duration_unfulfillable"
        assert ctx.session.drafts == []

    def test_the_ceiling_the_schema_advertises_is_the_one_that_is_enforced(
        self, ctx: ToolContext
    ) -> None:
        """Run at a non-default budget, end to end: the point is not what the
        schema says but that the code agrees with it."""
        _ready(ctx)
        ctx.budget = ToolBudget(max_candidates=7)
        invocation = _call(ctx, "draft", n=7)
        assert invocation.ok, invocation.result
        assert len(_payload(invocation)["drafts"]) == 7


class TestCritique:
    def test_critique_reads_the_scorecard_rather_than_a_second_opinion(
        self, ctx: ToolContext
    ) -> None:
        _ready(ctx, _ELECTRIFYING)
        _drafts(ctx)
        draft = ctx.session.drafts[0]
        payload = _payload(_call(ctx, "critique", draft_id=draft.draft_id))
        assert payload["measured"] == draft.quality.as_dict()
        reported = {
            finding["metric"] for report in payload["axes"] for finding in report["findings"]
        }
        assert reported == {finding.metric for finding in draft.quality.findings()}

    def test_a_breached_threshold_is_reported_with_what_would_move_it(
        self, ctx: ToolContext
    ) -> None:
        """This spec breaches two accompaniment metrics, so the finding path is
        fired rather than asserted about in the abstract."""
        _ready(ctx, _ELECTRIFYING)
        _drafts(ctx)
        payload = _payload(_call(ctx, "critique", draft_id="draft-0"))
        accompaniment = next(
            report for report in payload["axes"] if report["axis"] == "accompaniment"
        )
        assert accompaniment["findings"], "this fixture is meant to breach a threshold"
        finding = accompaniment["findings"][0]
        assert set(finding) == {
            "metric",
            "measured",
            "target",
            "direction",
            "rationale",
            "hint",
            "axis",
            "bars",
        }
        assert finding["hint"]

    def test_every_axis_is_reported_and_a_clean_one_says_so(self, ctx: ToolContext) -> None:
        """The tune, the bass and the harmony are clean on this piece, and the
        report says that rather than leaving them out: an axis missing from a
        report cannot be told from one whose critic never ran."""
        _ready(ctx, _ELECTRIFYING)
        _drafts(ctx)
        payload = _payload(_call(ctx, "critique", draft_id="draft-0"))
        assert [report["axis"] for report in payload["axes"]] == list(AXES)
        assert {report["axis"]: report["clean"] for report in payload["axes"]} == {
            "melody": True,
            "accompaniment": False,
            "bass": True,
            "harmony": True,
        }

    @pytest.mark.parametrize("axis", ["melody", "accompaniment", "bass", "harmony"])
    def test_one_part_can_be_asked_for_on_its_own(self, ctx: ToolContext, axis: str) -> None:
        _ready(ctx, _ELECTRIFYING)
        _drafts(ctx)
        payload = _payload(_call(ctx, "critique", draft_id="draft-0", axis=axis))
        assert [report["axis"] for report in payload["axes"]] == [axis]

    @pytest.mark.parametrize("asked", ["percussion", "orchestration", "Melody"])
    def test_a_part_nothing_measures_is_refused_rather_than_reported_clean(
        self, ctx: ToolContext, asked: str
    ) -> None:
        """An empty report for a part with no critic reads exactly like a clean
        piece, which is the wrong answer dressed as the right one."""
        _ready(ctx, _ELECTRIFYING)
        _drafts(ctx)
        invocation = _call(ctx, "critique", draft_id="draft-0", axis=asked)
        assert invocation.outcome == "refused"
        assert invocation.error_code == "invalid_arguments"
        assert "melody" in invocation.result

    def test_a_miss_is_placed_in_the_bars_its_own_events_sit_in(self, ctx: ToolContext) -> None:
        _ready(ctx, _ELECTRIFYING)
        _drafts(ctx)
        payload = _payload(_call(ctx, "critique", draft_id="draft-0"))
        findings = [f for report in payload["axes"] for f in report["findings"]]
        assert findings
        assert all(finding["bars"] for finding in findings), "both metrics localise"
        for finding in findings:
            for span in finding["bars"]:
                assert set(span) == {"first_bar", "last_bar"}
                # Numbered the way a reader counts them, which is a bar 1 and
                # not a bar 0 — the score indexes its measures from zero.
                assert 1 <= span["first_bar"] <= span["last_bar"]

    def test_a_clean_draft_is_reported_without_composing_it_again(
        self, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The bars are read off the notes, so a draft with nothing to place
        needs no score — and the way to say so is to make composing it fatal."""
        _ready(ctx)
        _drafts(ctx)

        def _explode(draft: Draft) -> Any:
            raise AssertionError(f"{draft.draft_id} was recomposed with nothing to place")

        monkeypatch.setattr(tools, "_recompose", _explode)
        payload = _payload(_call(ctx, "critique", draft_id="draft-0"))
        assert all(report["clean"] for report in payload["axes"])

    def test_a_draft_that_no_longer_recomposes_to_itself_is_a_failure(
        self, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The same claim `sketch` cashes, cashed before any bar is named: a
        draft that cannot be placed cannot be reported on either."""
        _ready(ctx, _ELECTRIFYING)
        _drafts(ctx)
        # Not `_OUTPUT`: the draft is at the electrifying spec, so returning
        # the calming piece is returning music the draft never described.
        monkeypatch.setattr(tools, "compose", lambda spec, plan=None: _OUTPUT)
        invocation = _call(ctx, "critique", draft_id="draft-0")
        assert invocation.outcome == "error"
        assert invocation.error_code == "performance_mismatch"

    def test_a_revised_draft_is_placed_against_the_plan_it_was_revised_under(
        self, ctx: ToolContext
    ) -> None:
        """A revision's plan is not the one its own spec would derive, so a
        localisation that re-derived the default would recompose a different
        piece — the failure `_recompose` exists to catch, and a revised draft is
        the only draft that can reach it.

        The clearance is inside the run this piece's ratchet honours — three
        through fifteen, measured rather than guessed — and that is why it is
        ten rather than the bound itself: a revision that measures worse than
        its parent is refused with a reason (D1's ratchet), so a fixture that
        asked for more would be asserting about a draft the tool declined to
        make. What the case is for is that the *honoured* revision is placed
        against its own plan, which the plan-hash premise below already asserts.
        """
        _ready(ctx, _ELECTRIFYING)
        _drafts(ctx)
        revised = _payload(
            _call(
                ctx,
                "revise",
                draft_id="draft-0",
                deltas=[{"knob": "SetHarmonyClearance", "semitones": 10}],
            )
        )
        draft = ctx.session.draft(revised["draft_id"])
        assert draft.plan_hash != ctx.session.drafts[0].plan_hash, "the premise: the plan moved"
        payload = _payload(_call(ctx, "critique", draft_id=draft.draft_id))
        findings = [f for report in payload["axes"] for f in report["findings"]]
        assert findings
        assert all(finding["bars"] for finding in findings)

    def test_an_unknown_draft_lists_the_ones_the_session_has(self, ctx: ToolContext) -> None:
        _ready(ctx)
        _drafts(ctx)
        invocation = _call(ctx, "critique", draft_id="draft-9")
        assert invocation.error_code == "unknown_draft"
        assert "draft-0" in invocation.result


class TestRevise:
    """A revision is a chain of requests folded onto a lineage, and the ratchet gates it.

    Three properties carry the weight. The first is the one `Draft`'s record
    shape was changed for: a second revision must not discard the first one's
    plan requests. The others are the tool's own contract — a request it cannot
    honour is refused rather than obeyed or crashed on, and a revision that
    measures worse is refused with the arbiter's own sentence.
    """

    _BASS: ClassVar[dict[str, object]] = {"knob": "SetBassMotion", "motion": "sparse"}
    _MOTIF: ClassVar[dict[str, object]] = {"knob": "SetMotifVariation", "factor": 2.0}
    _TEMPO: ClassVar[dict[str, object]] = {"knob": "SetTempo", "tempo_bpm": 96}
    """Three requests, and they are three because they do three different things.

    `_BASS` and `_MOTIF` both move the piece and move nothing the scorecard
    measures, so a revision carrying either is a revision whose only deciding
    element is the plan hash. `_TEMPO` is the request this length cannot hold —
    a 30-second calming piece plays at 64 BPM whatever it is asked for — which
    is why the swallow test is written about it rather than about a mover."""

    def test_a_revision_is_a_new_draft_that_names_its_parent(self, ctx: ToolContext) -> None:
        _ready(ctx)
        _drafts(ctx)

        payload = _payload(_call(ctx, "revise", draft_id="draft-0", deltas=[self._BASS]))

        assert payload["draft_id"] == "draft-1"
        assert payload["parent_id"] == "draft-0"
        assert [draft.draft_id for draft in ctx.session.drafts] == ["draft-0", "draft-1"]
        root, child = ctx.session.drafts
        assert child.parent_id == "draft-0"
        assert child.plan.bass_figures != root.plan.bass_figures, "and the request was honoured"

    def test_the_seed_is_the_parents_so_the_material_survives_the_edit(
        self, ctx: ToolContext
    ) -> None:
        """ "Hold the bass" must not become a different tune.

        The seed decides the draws behind the arrangement, the harmony and the
        melody, so a revision that re-rolled it would be a new piece wearing the
        old one's name — and the user's edit would be indistinguishable from a
        reroll.
        """
        _ready(ctx)
        _drafts(ctx, seed=11)

        payload = _payload(_call(ctx, "revise", draft_id="draft-0", deltas=[self._BASS]))

        assert payload["seed"] == 11
        assert ctx.session.drafts[1].spec.seed == ctx.session.drafts[0].spec.seed

    def test_a_second_revision_keeps_the_firsts_plan_requests(self, ctx: ToolContext) -> None:
        """The bug the chain exists to prevent, and the reason `Draft.deltas` is one.

        A plan is derived from the spec and `default_plan` reads only a mood and a
        meter, so the second revision cannot patch the first one's plan; folding
        only its own request from its own spec would silently revert the bass.
        Both halves are asserted — that the first revision moved the vocabulary,
        and that the second did not move it back.
        """
        _ready(ctx)
        _drafts(ctx)
        _call(ctx, "revise", draft_id="draft-0", deltas=[self._BASS])
        _call(ctx, "revise", draft_id="draft-1", deltas=[self._MOTIF])

        root, first, second = ctx.session.drafts
        assert first.plan.bass_figures != root.plan.bass_figures, "the premise: the bass"
        assert second.plan.bass_figures == first.plan.bass_figures, "and the second did not undo it"
        assert second.plan.motif_operation_weights != first.plan.motif_operation_weights
        assert [delta.knob for delta in second.deltas] == ["SetBassMotion", "SetMotifVariation"]

    def test_the_answer_carries_what_was_applied_and_what_was_refused(
        self, ctx: ToolContext
    ) -> None:
        """One request honoured and one not, so both halves of the answer are read.

        A chain that refused everything whole would be refused before reaching
        the payload, so the partial case is the only one where the `refused` list
        is visible at all.
        """
        _ready(ctx)
        _drafts(ctx)

        payload = _payload(
            _call(
                ctx,
                "revise",
                draft_id="draft-0",
                deltas=[self._BASS, {"knob": "SetTempo", "tempo_bpm": 1000}],
            )
        )

        assert payload["applied"] == [self._BASS]
        (refused,) = payload["refused"]
        assert refused["reason"] == "violates_the_spec"
        assert refused["nearest"] == "SetTempo(240)"
        root, child = ctx.session.drafts
        assert child.plan.bass_figures != root.plan.bass_figures, "the honoured half stands"

    def test_the_answer_names_the_element_of_the_order_that_moved(self, ctx: ToolContext) -> None:
        """The revision's own explanation, and the one that does not depend on taste.

        Holding the bass is a plan hash and nothing else: the notes are re-barred
        under a different figure but the piece measures the same bar for bar,
        which is asserted as the premise rather than trusted — if this request
        ever moved a measurement, this test would be measuring the wrong element
        and must say so.
        """
        _ready(ctx)
        _drafts(ctx)

        payload = _payload(_call(ctx, "revise", draft_id="draft-0", deltas=[self._BASS]))

        parent, child = ctx.session.drafts
        assert musical_key(child) == musical_key(parent), "the premise: no measurement moved"
        assert payload["moved_on"] == "plan_hash"

    def test_a_tempo_the_length_cannot_hold_is_refused_rather_than_quietly_replaced(
        self, ctx: ToolContext
    ) -> None:
        """A request the engine traded away, which is not the same as one it cannot hold.

        `arrange_for_duration` catches its own refusal when a pinned tempo does
        not fit the requested length and derives the tempo from the mood instead
        — the length is a release gate and outranks the tempo. That is a
        deliberate engine policy, and it leaves the one thing the vocabulary may
        not produce: a request answered with a draft that does not do what it
        asks. This piece plays at 64 BPM whatever the tempo is set to, so without
        the check a user asking for 96 would be told the piece was theirs.
        """
        _ready(ctx)
        _drafts(ctx)

        invocation = _call(ctx, "revise", draft_id="draft-0", deltas=[self._TEMPO])

        assert invocation.outcome == "refused"
        assert invocation.error_code == "tempo_not_honoured"
        assert "SetTempo(96)" in invocation.result, "the request, named"
        assert "64 BPM" in invocation.result, "the tempo it plays instead"
        assert "50-80 BPM" in invocation.result, "and the range that could have reached it"
        assert len(ctx.session.drafts) == 1, "and nothing was recorded"

    def test_a_tempo_the_length_can_hold_is_applied_and_the_draft_says_so(
        self, ctx: ToolContext
    ) -> None:
        """The other half of the check, without which it would be a refusal of everything.

        80 BPM fits a minute-long calming piece exactly, so the arrangement keeps
        it and the revision lands — with the performance plan moved, which is the
        observable that says the tempo reached the notes' timing rather than only
        the spec.
        """
        _ready(ctx, _SIXTY)
        _drafts(ctx)

        payload = _payload(
            _call(ctx, "revise", draft_id="draft-0", deltas=[{"knob": "SetTempo", "tempo_bpm": 80}])
        )

        assert payload["seed"] == _SIXTY.seed
        parent, child = ctx.session.drafts
        assert child.spec.tempo_bpm == 80
        assert child.performance_plan_hash != parent.performance_plan_hash, "and it is played"

    def test_a_revision_that_measures_worse_is_refused_with_the_arbiters_sentence(
        self, ctx: ToolContext
    ) -> None:
        """The ratchet, fired through the tool rather than asserted about.

        Closing the clearance puts the bed against the tune, and this seed then
        misses `register_separation_semitones`. The refusal names the bar that
        moved and the draft to keep, because that is what the conductor reads and
        the user is told.
        """
        _ready(ctx)
        _drafts(ctx)

        invocation = _call(
            ctx,
            "revise",
            draft_id="draft-0",
            deltas=[{"knob": "SetHarmonyClearance", "semitones": 1}],
        )

        assert invocation.outcome == "refused"
        assert invocation.error_code == "revision_regressed"
        assert "misses 1 quality bar" in invocation.result, "the bar that moved, counted"
        assert "may not be worse than the draft it came from" in invocation.result, "the rule"
        assert "draft-0" in invocation.result, "and which draft to keep"

    def test_a_refused_revision_leaves_the_session_as_it_was(self, ctx: ToolContext) -> None:
        """A refusal is an answer that changed nothing, so nothing is recorded."""
        _ready(ctx)
        _drafts(ctx)
        before = list(ctx.session.drafts)

        _call(
            ctx,
            "revise",
            draft_id="draft-0",
            deltas=[{"knob": "SetHarmonyClearance", "semitones": 1}],
        )

        assert ctx.session.drafts == before

    def test_a_request_the_engine_has_no_knob_for_is_answered_as_unbuilt(
        self, ctx: ToolContext
    ) -> None:
        """ "Understood and unbuilt" is a different answer from "not understood".

        A swing ratio is a real musical request the engine cannot yet honour, so
        it refuses by name with the alternative that does exist rather than as a
        misspelling — which is what `refuse_uncarried` is for and what a user's
        own words turn into.
        """
        _ready(ctx)
        _drafts(ctx)

        invocation = _call(ctx, "revise", draft_id="draft-0", deltas=[{"knob": "SetSwing"}])

        assert invocation.outcome == "refused"
        assert invocation.error_code == "unknown_knob"
        assert "SetSwing" in invocation.result, "the request, named"
        assert "SetDrumStyle" in invocation.result, "and what it can do instead"

    def test_a_value_the_request_cannot_hold_is_refused_rather_than_crashed(
        self, ctx: ToolContext
    ) -> None:
        """A `Literal`-typed field, which is the case that used to crash the tool.

        A terrace names a plan field, so an unknown one raised an `AttributeError`
        from inside the fold and the user was told the tool was broken. Both
        halves are asserted — a refusal, and *not* the `error` outcome a crash
        records — because a crash is also a recorded answer and only one of the
        two is the contract.
        """
        _ready(ctx)
        _drafts(ctx)

        invocation = _call(
            ctx,
            "revise",
            draft_id="draft-0",
            deltas=[{"knob": "SetSectionEnergy", "role": "verse", "factor": 2.0}],
        )

        assert invocation.outcome == "refused"
        assert invocation.error_code == "invalid_arguments"
        assert "verse" in invocation.result
        assert "'opening', 'peak', 'final', 'middle'" in invocation.result, "the vocabulary, named"

    @pytest.mark.parametrize(
        "requests",
        [[], [{"knob": "SetTempo"}], [{"knob": "SetTempo", "tempo_bpm": 96}, "faster"], ["x"]],
    )
    def test_a_list_that_is_not_a_list_of_requests_is_refused(
        self, ctx: ToolContext, requests: Any
    ) -> None:
        _ready(ctx)
        _drafts(ctx)

        invocation = _call(ctx, "revise", draft_id="draft-0", deltas=requests)

        assert invocation.error_code == "invalid_arguments"

    def test_more_requests_than_the_budget_allows_is_refused(self, ctx: ToolContext) -> None:
        """The bound is read off the budget, so a smaller one refuses sooner."""
        small = ToolContext(
            session=ctx.session,
            sessions=ctx.sessions,
            jobs=ctx.jobs,
            budget=ToolBudget(max_deltas=2),
        )
        _ready(small)
        _drafts(small)

        invocation = _call(small, "revise", draft_id="draft-0", deltas=[self._TEMPO] * 3)

        assert invocation.error_code == "invalid_arguments"
        assert "at most 2" in invocation.result
        assert len(small.session.drafts) == 1, "and nothing was recorded"

    def test_revising_a_draft_the_session_does_not_have_refuses(self, ctx: ToolContext) -> None:
        _ready(ctx)
        _drafts(ctx)

        invocation = _call(ctx, "revise", draft_id="draft-9", deltas=[self._BASS])

        assert invocation.error_code == "unknown_draft"
        assert "draft-0" in invocation.result

    def test_a_line_may_be_revised_only_as_far_as_the_budget(self, ctx: ToolContext) -> None:
        """A small budget, so the bound is reached in two revisions rather than eight."""
        small = ToolContext(
            session=ctx.session,
            sessions=ctx.sessions,
            jobs=ctx.jobs,
            budget=ToolBudget(max_revisions=2),
        )
        _ready(small)
        _drafts(small)
        _call(small, "revise", draft_id="draft-0", deltas=[self._BASS])
        _call(small, "revise", draft_id="draft-1", deltas=[self._BASS])

        invocation = _call(small, "revise", draft_id="draft-2", deltas=[self._BASS])

        assert invocation.outcome == "refused"
        assert invocation.error_code == "revision_limit"
        assert "draft-0" in invocation.result, "the piece the line started as"
        assert "2 time(s)" in invocation.result
        assert len(small.session.drafts) == 3

    def test_a_line_longer_than_the_budget_is_read_against_the_budget(
        self, ctx: ToolContext
    ) -> None:
        """The walk's own bound, reached by reading a line built under a larger one.

        A line's depth is a property of the documents in the session rather than
        of the budget in hand, so a smaller budget reads a line it did not build
        — which is the same "a stored document is not the writer" case the
        `parent_id` cycle guard is for, and the two bounds are separate refusals
        because they fire one revision apart.
        """
        _ready(ctx)
        _drafts(ctx)
        for _ in range(3):
            _call(ctx, "revise", draft_id=ctx.session.drafts[-1].draft_id, deltas=[self._BASS])
        assert ctx.session.drafts[-1].parent_id is not None, "the premise: the line is deep"

        small = ToolContext(
            session=ctx.session,
            sessions=ctx.sessions,
            jobs=ctx.jobs,
            budget=ToolBudget(max_revisions=2),
        )
        invocation = _call(small, "revise", draft_id="draft-3", deltas=[self._BASS])

        assert invocation.outcome == "refused"
        assert invocation.error_code == "revision_limit"
        assert "more than 2 revision(s)" in invocation.result
        assert len(ctx.session.drafts) == 4

    def test_a_recorded_chain_that_does_not_replay_is_a_failure_rather_than_a_revision(
        self, ctx: ToolContext
    ) -> None:
        """The guard on a document that is not the writer, fired by editing one.

        The chain is folded from the line's root spec and the prefix has to
        re-fold to exactly what was recorded — otherwise the draft in hand is a
        piece nothing else can reach, and applying new requests to it would record
        a lineage that never existed. `error` rather than `refused`, because this
        is not a request the engine declined: it is a record that cannot be
        honoured at all.
        """
        _ready(ctx)
        _drafts(ctx)
        _call(ctx, "revise", draft_id="draft-0", deltas=[self._BASS])

        document = ctx.session.drafts[1].to_document()
        document["deltas"].append({"knob": "SetCadence", "degree": 9, "seventh": False})
        ctx.session.replace_draft(Draft.from_document(document))

        invocation = _call(ctx, "revise", draft_id="draft-1", deltas=[self._BASS])

        assert invocation.outcome == "error"
        assert invocation.error_code == "lineage_mismatch"
        assert len(ctx.session.drafts) == 2, "and nothing was recorded"

    def test_a_revision_whose_every_request_was_refused_names_the_first_reason(
        self, ctx: ToolContext
    ) -> None:
        """A call with nothing left in it, which is a real answer rather than an empty turn.

        `SetTempo(1000)` is refused by the spec's own bound and it is the only
        request in the call, so there is nothing to compose and nothing to
        record. What comes back is the *applier's* sentence with the nearest
        legal request, which is what a user who asked for 1000 BPM needs to hear
        — and it is the one path where the refusal is not the engine's.
        """
        _ready(ctx)
        _drafts(ctx)

        invocation = _call(
            ctx, "revise", draft_id="draft-0", deltas=[{"knob": "SetTempo", "tempo_bpm": 1000}]
        )

        assert invocation.outcome == "refused"
        assert invocation.error_code == "violates_the_spec"
        assert "SetTempo(240)" in invocation.result, "and the nearest legal request"
        assert len(ctx.session.drafts) == 1, "and nothing was recorded"

    def test_a_sole_refusal_with_no_nearest_carries_the_appliers_sentence_alone(
        self, ctx: ToolContext
    ) -> None:
        """Some refusals name no alternative, and the answer must not invent one.

        A cadence degree is refused for being outside the scale rather than for
        exceeding a bound, so there is no "nearest legal value" to offer and the
        applier says so by leaving `nearest` unset. The sentence is then the
        whole of the answer, which is what keeps the tool from promising a
        request the vocabulary would refuse in turn.
        """
        _ready(ctx)
        _drafts(ctx)

        invocation = _call(
            ctx,
            "revise",
            draft_id="draft-0",
            deltas=[{"knob": "SetCadence", "degree": 9, "seventh": False}],
        )

        assert invocation.outcome == "refused"
        assert invocation.error_code == "violates_the_plan"
        assert "cadence_degree" in invocation.result
        assert "nearest request" not in invocation.result, "and no alternative is invented"
        assert len(ctx.session.drafts) == 1, "and nothing was recorded"

    def test_a_lineage_naming_a_draft_this_session_lacks_is_refused(self, ctx: ToolContext) -> None:
        """A `parent_id` is a string read off a document, and a document is not the writer.

        Deleting the draft a lineage names leaves a record that points at
        nothing, which no tool can build and a hand-edited file can — so the
        walk refuses it by name rather than folding a chain whose root does not
        exist.
        """
        _ready(ctx)
        _drafts(ctx)
        _call(ctx, "revise", draft_id="draft-0", deltas=[self._BASS])

        document = ctx.session.drafts[1].to_document()
        document["parent_id"] = "draft-9"
        ctx.session.replace_draft(Draft.from_document(document))

        invocation = _call(ctx, "revise", draft_id="draft-1", deltas=[self._BASS])

        assert invocation.outcome == "refused"
        assert invocation.error_code == "broken_lineage"
        assert "draft-9" in invocation.result, "and it names the draft that is missing"
        assert len(ctx.session.drafts) == 2, "and nothing was recorded"

    def test_a_lineage_that_names_itself_back_is_refused(self, ctx: ToolContext) -> None:
        """The other half of the same guard, and the reason it is a `seen` set.

        A draft that is its own parent is a walk with no end. The two refusals
        are one code because they are one situation — a line with no root — but
        they are two sentences, because "this draft is missing" and "these drafts
        name each other" are different things to tell a user.
        """
        _ready(ctx)
        _drafts(ctx)
        _call(ctx, "revise", draft_id="draft-0", deltas=[self._BASS])

        document = ctx.session.drafts[1].to_document()
        document["parent_id"] = "draft-1"
        ctx.session.replace_draft(Draft.from_document(document))

        invocation = _call(ctx, "revise", draft_id="draft-1", deltas=[self._BASS])

        assert invocation.outcome == "refused"
        assert invocation.error_code == "broken_lineage"
        assert "cycle" in invocation.result
        assert len(ctx.session.drafts) == 2, "and nothing was recorded"

    def test_a_revision_the_engine_will_not_compose_is_refused_with_its_own_reason(
        self, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The engine's refusal reaches the conductor as a refusal, not as a crash.

        Every request in the chain was honoured and the piece the fold describes
        is still one `compose` declines to write — so the answer has to be the
        engine's own code, because that is the only thing that says what is
        wrong with the music rather than with the request.
        """
        _ready(ctx)
        _drafts(ctx)

        def _never(spec: CompositionSpec, *, plan: Any = None) -> Any:
            raise CompositionEngineError(
                code=EngineErrorCode.DURATION_UNFULFILLABLE, message="no arrangement fits"
            )

        monkeypatch.setattr(tools, "compose", _never)
        invocation = _call(ctx, "revise", draft_id="draft-0", deltas=[self._BASS])

        assert invocation.outcome == "refused"
        assert invocation.error_code == "duration_unfulfillable"
        assert "no arrangement fits" in invocation.result
        assert len(ctx.session.drafts) == 1, "and nothing was recorded"


class TestReviseDraftAndParseRequests:
    """The two pieces of `revise` that a second caller now shares.

    `POST /sessions/{id}/deltas` reads requests out of an HTTP body and keeps a
    revision without taking a turn, which is the same rule read from a different
    document and reported by a different caller — `publish_draft`'s shape at the
    other end of a session. These are the properties the two callers depend on:
    the ceiling is the caller's, the answer is this step's rather than the
    chain's, and the record is durable before the answer is.
    """

    _BASS: ClassVar[dict[str, object]] = {"knob": "SetBassMotion", "motion": "sparse"}
    _MOTIF: ClassVar[dict[str, object]] = {"knob": "SetMotifVariation", "factor": 2.0}

    def test_requests_are_read_in_order_and_with_their_values(self) -> None:
        parsed = parse_requests([self._BASS, {"knob": "SetTempo", "tempo_bpm": 96}], maximum=8)

        assert parsed == (SetBassMotion(motion=BassMotion.SPARSE), SetTempo(tempo_bpm=96))

    def test_the_ceiling_is_the_callers_and_not_the_budgets(self) -> None:
        """A policy rather than a shape: the tool's ceiling is its own budget.

        Asserted against a number that is *not* the default, because a reading
        that quietly used the budget itself would pass every case written at it.
        """
        with pytest.raises(ToolRefusal) as caught:
            parse_requests([self._BASS, self._MOTIF], maximum=1)

        assert caught.value.error_code == "invalid_arguments"
        assert "at most 1 request(s)" in caught.value.message

    def test_the_tool_reads_its_argument_through_the_shared_reader(self, ctx: ToolContext) -> None:
        """One rule, one owner — so the refusals are identical, not merely similar.

        Fired against a document both callers must refuse; a second reading at
        either call site would answer with its own words rather than these.
        """
        _ready(ctx)
        _drafts(ctx)

        invocation = _call(ctx, "revise", draft_id="draft-0", deltas=[{"knob": "SetSaxophone"}])
        with pytest.raises(ToolRefusal) as direct:
            parse_requests([{"knob": "SetSaxophone"}], maximum=8)

        assert invocation.outcome == "refused"
        assert invocation.error_code == direct.value.error_code
        assert invocation.result == direct.value.message

    def test_the_source_is_the_callers_and_is_recorded_on_the_child(self, ctx: ToolContext) -> None:
        """Who chose these requests is the one thing this function cannot know.

        A conductor's tool call and a studio control both arrive as typed
        requests, and the difference between them is who picked them — which is
        what the record keeps, and what a verdict will later be attached to.
        """
        _ready(ctx)
        _drafts(ctx)

        revision = revise_draft(
            ctx,
            ctx.session.draft("draft-0"),
            (SetBassMotion(motion=BassMotion.SPARSE),),
            source="typed",
        )

        assert revision.draft.requests_source == "typed"

    def test_the_tool_records_its_own_source_without_being_asked(self, ctx: ToolContext) -> None:
        """The tool's caller is the model, and that is a fact only the tool has."""
        _ready(ctx)
        _drafts(ctx)

        _call(ctx, "revise", draft_id="draft-0", deltas=[self._BASS])

        assert ctx.session.draft("draft-1").requests_source == "conductor"

    def test_the_answer_carries_this_steps_requests_and_not_the_chain(
        self, ctx: ToolContext
    ) -> None:
        """The property `Revision` exists for: the child's `deltas` is the whole line.

        A caller rendering the answer says what *this* edit did, and it already
        has the chain through the draft's own record — so an `applied` that grew
        with the line would be the same list twice, one of them not this step's.
        """
        _ready(ctx)
        _drafts(ctx)

        first = revise_draft(
            ctx,
            ctx.session.draft("draft-0"),
            (SetBassMotion(motion=BassMotion.SPARSE),),
            source="typed",
        )
        second = revise_draft(
            ctx, ctx.session.draft("draft-1"), (SetMotifVariation(factor=2.0),), source="typed"
        )

        assert [delta.knob for delta in first.applied] == ["SetBassMotion"]
        assert [delta.knob for delta in second.applied] == ["SetMotifVariation"]
        assert [delta.knob for delta in second.draft.deltas] == [
            "SetBassMotion",
            "SetMotifVariation",
        ]

    def test_a_revision_with_nothing_to_apply_is_refused(self, ctx: ToolContext) -> None:
        """A step that asked for nothing is not a step, and it is refused before the walk."""
        _ready(ctx)
        _drafts(ctx)

        with pytest.raises(ToolRefusal) as caught:
            revise_draft(ctx, ctx.session.draft("draft-0"), (), source="typed")

        assert caught.value.error_code == "invalid_arguments"
        assert "at least one request" in caught.value.message

    def test_the_child_is_durable_before_the_answer_is(self, ctx: ToolContext) -> None:
        """Both callers answer with a draft id, and a caller with the id will fetch it.

        So the record has to be on the disk at the moment it becomes true rather
        than at the end of a turn — and on `/deltas` there is no turn to end.
        """
        _ready(ctx)
        _drafts(ctx)

        revision = revise_draft(
            ctx,
            ctx.session.draft("draft-0"),
            (SetBassMotion(motion=BassMotion.SPARSE),),
            source="typed",
        )

        reloaded = SessionStorage(ctx.sessions.root).get(ctx.session.session_id)
        assert reloaded.draft(revision.draft.draft_id).requests_source == "typed"


class TestCompare:
    """Ranking several drafts, and naming what put each one where it sits."""

    def test_a_fan_out_is_ranked_and_every_step_says_what_decided_it(
        self, ctx: ToolContext
    ) -> None:
        _ready(ctx)
        _drafts(ctx, n=3)

        ranking = _payload(_call(ctx, "compare"))["ranking"]

        assert [entry["rank"] for entry in ranking] == [1, 2, 3]
        assert len({entry["draft_id"] for entry in ranking}) == 3
        assert "behind" not in ranking[0], "nothing is ahead of the leader"
        assert "on" not in ranking[0]
        for previous, entry in pairwise(ranking):
            assert entry["behind"] == previous["draft_id"]
            assert entry["on"] in ELEMENTS

    def test_the_ranking_is_the_arbiters_own(self, ctx: ToolContext) -> None:
        """The tool has no order of its own, and this is what says so."""
        _ready(ctx)
        _drafts(ctx, n=3)

        ranking = _payload(_call(ctx, "compare"))["ranking"]

        assert [entry["draft_id"] for entry in ranking] == [
            draft.draft_id for draft in rank(ctx.session.drafts)
        ]

    def test_two_candidates_of_one_fan_out_are_decided_by_their_seeds(
        self, ctx: ToolContext
    ) -> None:
        """The tie the plan hash cannot settle, seen from the tool.

        Both candidates are one plan at two seeds, and this spec breaches nothing
        at either, so the seed is the only element left — which is the case the
        sixth element of the order exists for. The premise is asserted rather
        than trusted, because a spec that breached something would be decided by
        the breach and say nothing about the seed.
        """
        _ready(ctx)
        _drafts(ctx, n=2, seed=5)

        first, second = ctx.session.drafts
        assert musical_key(first) == musical_key(second), "the premise: neither breaches a bar"
        assert first.plan_hash == second.plan_hash, "the premise: one plan, two seeds"

        ranking = _payload(_call(ctx, "compare"))["ranking"]

        assert ranking[1]["behind"] == ranking[0]["draft_id"]
        assert ranking[1]["on"] == "seed"

    def test_only_the_drafts_asked_for_are_ranked(self, ctx: ToolContext) -> None:
        _ready(ctx)
        _drafts(ctx, n=3)

        ranking = _payload(_call(ctx, "compare", draft_ids=["draft-0", "draft-2"]))["ranking"]

        assert [entry["rank"] for entry in ranking] == [1, 2]
        assert {entry["draft_id"] for entry in ranking} == {"draft-0", "draft-2"}

    def test_a_revision_is_ranked_beside_its_parent_and_says_which_it_came_from(
        self, ctx: ToolContext
    ) -> None:
        """The lineage is on the card, so a model reading it does not treat one
        piece as two rivals."""
        _ready(ctx)
        _drafts(ctx)
        _call(
            ctx,
            "revise",
            draft_id="draft-0",
            deltas=[{"knob": "SetBassMotion", "motion": "sparse"}],
        )

        entries = {entry["draft_id"]: entry for entry in _payload(_call(ctx, "compare"))["ranking"]}

        assert entries["draft-0"]["parent_id"] is None
        assert entries["draft-1"]["parent_id"] == "draft-0"

    def test_a_session_with_no_drafts_says_what_to_do_about_it(self, ctx: ToolContext) -> None:
        invocation = _call(ctx, "compare")

        assert invocation.outcome == "refused"
        assert invocation.error_code == "no_drafts"
        assert "draft" in invocation.result

    def test_a_draft_asked_for_twice_is_refused(self, ctx: ToolContext) -> None:
        """A draft ranked against itself says nothing, and the id list is the model's."""
        _ready(ctx)
        _drafts(ctx)

        invocation = _call(ctx, "compare", draft_ids=["draft-0", "draft-0"])

        assert invocation.error_code == "invalid_arguments"
        assert "twice" in invocation.result

    @pytest.mark.parametrize("bad", [[7], [], "draft-0", {}])
    def test_a_list_of_anything_but_ids_is_refused(self, ctx: ToolContext, bad: Any) -> None:
        _ready(ctx)
        _drafts(ctx)

        invocation = _call(ctx, "compare", draft_ids=bad)

        assert invocation.error_code == "invalid_arguments"

    def test_an_unknown_draft_lists_the_ones_the_session_has(self, ctx: ToolContext) -> None:
        _ready(ctx)
        _drafts(ctx)

        invocation = _call(ctx, "compare", draft_ids=["draft-9"])

        assert invocation.error_code == "unknown_draft"
        assert "draft-0" in invocation.result


class TestRepair:
    """`repair` chooses its own requests, and that is what its tests are about.

    Every other tool's arguments are the conductor's; this one's are the
    product's, measured off the piece. So the cases here are about the three
    things a caller cannot see from the outside: that the request it chose is
    the one that moved the bar, that the search is recorded rather than implied,
    and that the three ways of coming back empty are told apart. The loop's own
    arithmetic lives in `test_session_repairs.py`; what is tested here is the
    tool around it — the record it writes, the budget it honours, and the fact
    that it costs the turn nothing.
    """

    _BREACHING = CompositionSpec(mood=Mood.ELECTRIFYING, duration_seconds=30, seed=11)
    """Two bars missed, one kept request, and it is the same one every run.

    Both of the bed's bars, which is why one request leaves the piece clean and
    why the search is one round of two: the table's two candidates for
    `texture_hierarchy` are the same repair seen from two sides, so both are
    `kept` and the order settles which. Re-found by the same sweep the class's
    critique fixture was — seed 5's piece now misses the leap bar as well, so a
    repair aimed at the arbiter's worst bar has three to work through and the
    "one kept request" this case reads is not the shape it takes.
    """
    _STUBBORN = CompositionSpec(mood=Mood.ELECTRIFYING, duration_seconds=180, seed=4)
    """A piece one move cannot finish, which is what the bound needs.

    Measured over three moods, four durations and forty seeds at
    `max_repairs=1`: this one and its counterpart at seed 27 are the two pieces
    a single move leaves with `max_leap_semitones`, `texture_hierarchy` and
    `harmony_pad_coverage` all still missed. The three-bar remainder is what the
    case reads — one move applied, three bars named in `remaining` — and it is
    three rather than one because the leap bar's own requests are partly
    repairable at best (see `repairs.py`'s table for the reading), which is the
    reason a bound has anything to report here at all.

    The older fixture was a 30-second piece at seed 3, and the key pool moved it
    onto a key whose leap bar one move does clear — so this one is re-measured
    rather than re-keyed, because the *property* the case asserts is what the
    sweep was for, and pinning a key to preserve a seed's old behaviour would
    leave the file measuring a path the product no longer takes.
    """
    _OUT_OF_TABLE = CompositionSpec.model_validate(
        {
            "mood": Mood.SLEEP,
            "duration_seconds": 30,
            "seed": 2,
            "time_signature": "3/4",
            "instrumentation": [
                {"role": VoiceRole.MELODY.value, "instrument": "glockenspiel"},
                {"role": VoiceRole.HARMONY.value, "instrument": "harp"},
                {"role": VoiceRole.BASS.value, "instrument": "tuba"},
            ],
        }
    )
    """A bar the table holds no request for, reached by the spec alone.

    Seven of the arbiter's eleven bars have no entry, because the 90-piece corpus
    that chose the table never breached them — the table is measured rather than
    reasoned, so a bar nothing breached is a bar with nothing to measure. One of
    them is reachable from a spec a brief can carry: a slow waltz for a
    glockenspiel, a harp and a tuba misses `step_ratio`, how much of the tune
    moves by step, which no 4/4 corpus piece does.

    The clearance route this case used to take does not work. Widening the
    harmony clearance far enough to push the bed under the tune does create the
    bar — but it makes a *clean* piece worse, so the ratchet refuses the revision
    that would have created the draft the case repairs.
    """

    def test_it_repairs_the_worst_bar_and_names_what_moved(self, ctx: ToolContext) -> None:
        _ready(ctx, self._BREACHING)
        _drafts(ctx)

        payload = _payload(_call(ctx, "repair", draft_id="draft-0"))

        assert payload["applied"] == [
            {"knob": "SetAccompanimentDensity", "step_ticks": 1440},
        ]
        assert payload["remaining"] == []
        assert payload["draft_id"] == "draft-1"
        assert payload["parent_id"] == "draft-0"
        assert payload["moved_on"] in ELEMENTS

    def test_the_child_records_that_the_product_chose_the_request(self, ctx: ToolContext) -> None:
        """The vocabulary's fifth source, and the one nobody else can write: a
        request the user did not make and the conductor did not author."""
        _ready(ctx, self._BREACHING)
        _drafts(ctx)

        _call(ctx, "repair", draft_id="draft-0")

        child = ctx.session.drafts[1]
        assert child.requests_source == "repair"
        assert child.parent_id == "draft-0"
        assert child.spec.seed == self._BREACHING.seed, "the material survives the edit"

    def test_the_search_is_recorded_rather_than_summarised(self, ctx: ToolContext) -> None:
        """Every candidate, including the one not kept: a repair that reported
        only its answer would make the piece's quality a property of the loop."""
        _ready(ctx, self._BREACHING)
        _drafts(ctx)

        payload = _payload(_call(ctx, "repair", draft_id="draft-0"))

        assert [(a["metric"], a["outcome"]) for a in payload["attempts"]] == [
            ("texture_hierarchy", "kept"),
            ("texture_hierarchy", "kept"),
        ]
        assert [a["request"]["knob"] for a in payload["attempts"]] == [
            "SetAccompanimentDensity",
            "SetHarmonyTexture",
        ]

    def test_it_spends_nothing_the_turn_has_to_budget_for(self, ctx: ToolContext) -> None:
        """No model call and no audio — the loop's whole cost is arithmetic, and
        the ledger counts what is expensive. A repair that charged a sketch
        would make the cheap tool the one that runs out of turn."""
        _ready(ctx, self._BREACHING)
        _drafts(ctx)
        before = (ctx.ledger.sketches, ctx.ledger.llm_calls)

        _call(ctx, "repair", draft_id="draft-0")

        assert (ctx.ledger.sketches, ctx.ledger.llm_calls) == before

    def test_a_draft_that_misses_nothing_is_refused_in_words(self, ctx: ToolContext) -> None:
        """Not a crash and not a silent success: the answer a user pressing
        "fix it" on a piece with nothing wrong should get."""
        _ready(ctx)
        _drafts(ctx)

        invocation = _call(ctx, "repair", draft_id="draft-0")

        assert invocation.error_code == "no_repair"
        assert "nothing to repair" in invocation.result
        assert len(ctx.session.drafts) == 1, "and nothing was recorded"

    def test_a_bar_the_table_has_no_request_for_is_refused_by_name(self, ctx: ToolContext) -> None:
        """The refusal says *nothing was tried*, rather than that everything was
        tried and nothing worked — the distinction a reader most needs, and the
        one this branch exists for. It names the bar and hands on the bar's own
        hint, which is written for a maintainer and is the whole reason the table
        is measured rather than read off the hints."""
        _ready(ctx, self._OUT_OF_TABLE)
        _drafts(ctx)

        invocation = _call(ctx, "repair", draft_id="draft-0")

        assert invocation.error_code == "no_repair"
        assert "step_ratio" in invocation.result
        assert "nothing was tried" in invocation.result
        assert "saimc/compose/motif.py" in invocation.result, "the hint, verbatim"

    def test_a_bar_whose_requests_measured_no_better_is_refused_differently(
        self, ctx: ToolContext
    ) -> None:
        """The other empty: requests were tried and none was kept. Same code,
        opposite sentence, and the difference is what the attempts recorded.

        The piece is the one a sweep of three moods, five durations and forty
        seeds finds with this property at sixty seconds, and there is exactly
        one of them — the key pool moved every mood's pieces onto their own keys
        and took the older fixture's leap bar from unrepairable to repairable
        with it, so the case is re-measured rather than re-pinned. What it
        asserts is the sentence, and the sentence names the bar the attempts
        aimed at.
        """
        _ready(ctx, CompositionSpec(mood=Mood.CALMING, duration_seconds=60, seed=18))
        _drafts(ctx)

        invocation = _call(ctx, "repair", draft_id="draft-0")

        assert invocation.error_code == "no_repair"
        assert "leap_recovery_ratio" in invocation.result
        assert "were all tried and none was kept" in invocation.result

    def test_the_budget_bounds_the_chain_and_says_what_is_left(self, ctx: ToolContext) -> None:
        """A repair that cannot finish is a partial repair, not a refusal: the
        moves it made are real and the bars it left are named. The budget is
        what stops it, so `remaining` is the whole of the difference.

        `_STUBBORN`'s docstring is where the piece is accounted for; this case
        reads its `remaining` because it is the one place the two halves of a
        bounded repair — what was applied and what was not — are visible
        together.
        """
        _ready(ctx, self._STUBBORN)
        ctx.budget = replace(ctx.budget, max_repairs=1)
        _drafts(ctx)

        payload = _payload(_call(ctx, "repair", draft_id="draft-0"))

        assert len(payload["applied"]) == 1
        assert "max_leap_semitones" in payload["remaining"]
        assert payload["draft_id"] == "draft-1", "and the move that was made stands"

    def test_an_unknown_draft_is_refused_the_way_every_other_tool_refuses_it(
        self, ctx: ToolContext
    ) -> None:
        _ready(ctx)
        _drafts(ctx)

        assert _call(ctx, "repair", draft_id="draft-9").error_code == "unknown_draft"


@pytest.fixture
def rendered(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """A stand-in for the FluidSynth pass, writing the files it claims to.

    Faked at the renderer's boundary rather than at `render_sketch`'s, because
    the paths the tool records have to be the paths a real render writes: the
    fake writes them, so a change to the naming scheme shows up here.
    """
    calls: list[dict[str, Any]] = []

    def _fake(plan: Any, **kwargs: Any) -> AudioArtifact:
        out_dir: Path = kwargs["out_dir"]
        job_id: str = kwargs["job_id"]
        calls.append({"plan": plan, "job_id": job_id, "bpm": kwargs["bpm"], **kwargs})
        (out_dir / f"{job_id}.mid").write_bytes(b"MThd")
        wav = out_dir / "audio.wav"
        wav.write_bytes(b"RIFF")
        ogg = out_dir / "audio.ogg"
        ogg.write_bytes(b"OggS")
        return AudioArtifact(
            primary_path=wav,
            primary_container="wav",
            primary_codec="pcm_s16le",
            primary_sha256="a" * 64,
            primary_size_bytes=4,
            ogg_path=ogg,
            ogg_codec="opus",
            ogg_sha256="b" * 64,
            ogg_size_bytes=4,
        )

    monkeypatch.setattr(tools, "render_sketch", _fake)
    monkeypatch.setattr(tools, "resolve_job_soundfont", lambda _voices: Path("/tmp/one.sf2"))
    return calls


class TestSketch:
    def test_a_sketch_attaches_audio_to_the_draft_it_rendered(
        self, ctx: ToolContext, rendered: list[dict[str, Any]]
    ) -> None:
        _ready(ctx)
        _drafts(ctx)
        invocation = _call(ctx, "sketch", draft_id="draft-0")
        assert invocation.ok, invocation.result
        record = ctx.session.drafts[0].sketch
        assert record is not None
        assert record.ogg_path == "sketches/draft-0/audio.ogg"

    def test_the_sketch_is_written_under_the_session_and_the_draft(
        self, ctx: ToolContext, rendered: list[dict[str, Any]]
    ) -> None:
        """A fan-out writes several in one turn, and they must not overwrite
        each other's `audio.wav`."""
        _ready(ctx)
        _drafts(ctx, n=2)
        _call(ctx, "sketch", draft_id="draft-0")
        _call(ctx, "sketch", draft_id="draft-1")
        assert [c["job_id"] for c in rendered] == ["draft-0", "draft-1"]
        first = ctx.session.drafts[0].sketch
        second = ctx.session.drafts[1].sketch
        assert first is not None
        assert second is not None
        assert first.ogg_path != second.ogg_path
        assert ctx.sessions.sketch_path(ctx.session.session_id, first.ogg_path).exists()

    def test_the_draft_is_replaced_and_not_appended(
        self, ctx: ToolContext, rendered: list[dict[str, Any]]
    ) -> None:
        """A draft gains a sketch in place; appending would leave two drafts
        of one id and a session its own store refuses."""
        _ready(ctx)
        _drafts(ctx)
        _call(ctx, "sketch", draft_id="draft-0")
        assert [draft.draft_id for draft in ctx.session.drafts] == ["draft-0"]

    def test_the_render_is_given_the_music_the_draft_recomposed_to(
        self, ctx: ToolContext, rendered: list[dict[str, Any]]
    ) -> None:
        _ready(ctx)
        _drafts(ctx)
        _call(ctx, "sketch", draft_id="draft-0")
        assert rendered[0]["plan"].compute_hash() == _OUTPUT.performance_plan.compute_hash()
        assert rendered[0]["bpm"] == _OUTPUT.arrangement.tempo_bpm

    def test_a_draft_that_no_longer_recomposes_to_itself_is_a_failure(
        self, ctx: ToolContext, rendered: list[dict[str, Any]], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The draft's whole claim is that it describes the music the user
        heard; this is the moment the claim is cashed."""
        _ready(ctx)
        _drafts(ctx)
        # Not `_OUTPUT`: the draft is at the spec's own seed, so returning the
        # same music would be returning the music it *did* describe.
        drifted = compose(_ELECTRIFYING)
        monkeypatch.setattr(tools, "compose", lambda spec, plan=None: drifted)
        invocation = _call(ctx, "sketch", draft_id="draft-0")
        assert invocation.outcome == "error"
        assert invocation.error_code == "performance_mismatch"
        assert ctx.session.drafts[0].sketch is None

    def test_a_turn_may_only_spend_its_sketches(
        self, ctx: ToolContext, rendered: list[dict[str, Any]]
    ) -> None:
        _ready(ctx)
        _drafts(ctx, n=2)
        ctx.ledger.sketches = ctx.budget.max_sketches
        invocation = _call(ctx, "sketch", draft_id="draft-0")
        assert invocation.outcome == "refused"
        assert invocation.error_code == "budget_exhausted"
        assert rendered == []

    def test_a_render_that_produced_no_audio_is_a_failure(
        self, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _ready(ctx)
        _drafts(ctx)

        def _no_ogg(plan: Any, **kwargs: Any) -> AudioArtifact:
            return AudioArtifact(
                primary_path=Path(kwargs["out_dir"]) / "audio.wav",
                primary_container="wav",
                primary_codec="pcm_s16le",
                primary_sha256="a" * 64,
                primary_size_bytes=4,
            )

        monkeypatch.setattr(tools, "render_sketch", _no_ogg)
        monkeypatch.setattr(tools, "resolve_job_soundfont", lambda _voices: Path("/tmp/one.sf2"))
        invocation = _call(ctx, "sketch", draft_id="draft-0")
        assert invocation.outcome == "error"
        assert invocation.error_code == "sketch_incomplete"

    def test_a_failed_render_still_spent_the_turn_s_sketch(
        self, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FluidSynth ran whatever happened next, so the seconds are gone."""
        _ready(ctx)
        _drafts(ctx)

        def _boom(plan: Any, **kwargs: Any) -> AudioArtifact:
            raise OSError("fluidsynth not found")

        monkeypatch.setattr(tools, "render_sketch", _boom)
        monkeypatch.setattr(tools, "resolve_job_soundfont", lambda _voices: Path("/tmp/one.sf2"))
        assert _call(ctx, "sketch", draft_id="draft-0").outcome == "error"
        assert ctx.ledger.sketches == 1


class TestFinalize:
    @pytest.fixture(autouse=True)
    def _queued(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(saimc.jobs.worker, "enqueue_job", lambda *_a, **_k: "rq:test")

    def test_the_job_is_the_draft_it_publishes(self, ctx: ToolContext) -> None:
        """Same spec, same plan, same seed: the render is of the music the user
        heard, not of a default re-derived at publish time."""
        _ready(ctx)
        _drafts(ctx)
        draft = ctx.session.drafts[0]
        payload = _payload(_call(ctx, "finalize", draft_id=draft.draft_id))
        job = ctx.jobs.get(payload["job_id"])
        assert job.input_spec == draft.spec
        assert job.input_plan == draft.plan
        assert job.seed == draft.spec.seed
        assert job.parser_source == "from-spec"

    def test_a_finalized_draft_is_an_already_validated_spec(self, ctx: ToolContext) -> None:
        _ready(ctx)
        _drafts(ctx)
        _call(ctx, "finalize", draft_id="draft-0")
        assert ctx.jobs.get(ctx.session.finalized_job_id).input_spec is not None

    def test_the_link_is_durable_when_it_becomes_true(self, ctx: ToolContext) -> None:
        """Written by the tool rather than left to the end of a turn that
        might not arrive: a session that forgot it published would let the
        user publish a second render of the same piece."""
        _ready(ctx)
        _drafts(ctx)
        _call(ctx, "finalize", draft_id="draft-0")
        assert ctx.sessions.get(ctx.session.session_id).finalized_job_id is not None

    def test_a_session_publishes_once(self, ctx: ToolContext) -> None:
        _ready(ctx)
        _drafts(ctx, n=2)
        first = _call(ctx, "finalize", draft_id="draft-0")
        assert first.ok
        invocation = _call(ctx, "finalize", draft_id="draft-1")
        assert invocation.outcome == "refused"
        assert invocation.error_code == "already_finalized"
        assert first.result in invocation.result or "already published" in invocation.result

    def test_a_broker_outage_refuses_and_leaves_the_session_unpublished(
        self, ctx: ToolContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The real `enqueue_or_fail` runs, so what is tested is the shared
        policy — the record is marked failed *and* the call refuses."""
        _ready(ctx)
        _drafts(ctx)

        def _down(*_args: Any, **_kwargs: Any) -> str:
            raise ConnectionError("broker down")

        monkeypatch.setattr(saimc.jobs.worker, "enqueue_job", _down)
        invocation = _call(ctx, "finalize", draft_id="draft-0")
        assert invocation.outcome == "refused"
        assert invocation.error_code == "queue_unavailable"
        assert ctx.session.finalized_job_id is None
        published = list(ctx.jobs.list_all())
        assert len(published) == 1
        assert published[0].state.value == "failed"

    def test_publishing_an_unknown_draft_refuses(self, ctx: ToolContext) -> None:
        _ready(ctx)
        assert _call(ctx, "finalize", draft_id="draft-0").error_code == "unknown_draft"


class TestTheHandlersAreAllReachable:
    def test_every_tool_in_the_table_has_a_handler_that_is_a_coroutine_function(self) -> None:
        for tool in TOOLS.values():
            assert asyncio.iscoroutinefunction(tool.handler)

    def test_the_module_exports_what_it_claims(self) -> None:
        assert set(tools.__all__) == {
            "MAX_CANDIDATES_PER_DRAFT",
            "MAX_DELTAS_PER_REVISION",
            "MAX_LLM_CALLS_PER_TURN",
            "MAX_REPAIRS_PER_TURN",
            "MAX_REVISIONS_PER_LINE",
            "MAX_SKETCHES_PER_TURN",
            "TOOLS",
            "TURN_DEADLINE_SECONDS",
            "Tool",
            "ToolBudget",
            "ToolContext",
            "ToolError",
            "ToolFailure",
            "ToolRefusal",
            "Revision",
            "TurnLedger",
            "dispatch",
            "parse_requests",
            "publish_draft",
            "request_schema",
            "revise_draft",
            "tool_specs",
        }


class TestWithoutAModel:
    """The product has to work with no model in the loop, and the tools are
    where that is either true or aspirational: `ctx.llm` is `None` throughout
    this module, and everything except `parse_brief` runs anyway."""

    def test_only_the_parser_needs_a_model(self, ctx: ToolContext) -> None:
        _ready(ctx)
        assert _call(ctx, "draft").ok
        assert _call(ctx, "critique", draft_id="draft-0").ok

    def test_the_whole_feedback_loop_runs_with_no_model_in_the_loop(self, ctx: ToolContext) -> None:
        """Drafting, revising and ranking are arithmetic and the arbiter's order.

        This is the plan's "feedback degrades" bullet as a test: with the model
        absent the conductor cannot speak, but every Tier-1 edit and every
        deterministic tool still answers, so the product is usable by a user who
        only wants to steer with controls.
        """
        _ready(ctx)
        assert _call(ctx, "draft").ok
        assert _call(
            ctx,
            "revise",
            draft_id="draft-0",
            deltas=[{"knob": "SetBassMotion", "motion": "sparse"}],
        ).ok
        assert _call(ctx, "compare").ok
        assert ctx.llm is None, "the premise: no model is wired"

    def test_the_one_tool_that_needs_a_model_says_so_by_name(self, ctx: ToolContext) -> None:
        assert _call(ctx, "parse_brief").error_code == "llm_not_configured"
