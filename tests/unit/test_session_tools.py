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
from pathlib import Path
from typing import Any

import pytest

import saimc.jobs.worker
from saimc.compose.engine import CompositionEngineError, EngineErrorCode, compose
from saimc.jobs.storage import JobStorage
from saimc.llm.base import ParseRequest, ParseResult, ToolCall
from saimc.render.audio import AudioArtifact
from saimc.session import tools
from saimc.session.models import Session, ToolInvocation
from saimc.session.store import SessionStorage
from saimc.session.tools import (
    MAX_CANDIDATES_PER_DRAFT,
    TOOLS,
    ToolBudget,
    ToolContext,
    TurnLedger,
    dispatch,
    tool_specs,
)
from saimc.spec import CompositionSpec, Mood, SpecError

_SPEC = CompositionSpec(mood=Mood.CALMING, duration_seconds=30, seed=5)
_ELECTRIFYING = CompositionSpec(mood=Mood.ELECTRIFYING, duration_seconds=30, seed=5)
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
        """A ratchet. Phase D adds `revise`, `apply_delta` and `compare`, and
        will have to say so here rather than letting them appear."""
        assert sorted(TOOLS) == ["critique", "draft", "finalize", "parse_brief", "sketch"]

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
        assert len(payload["findings"]) == len(draft.quality.findings())

    def test_a_breached_threshold_is_reported_with_what_would_move_it(
        self, ctx: ToolContext
    ) -> None:
        """This spec breaches `texture_hierarchy`, so the finding path is
        fired rather than asserted about in the abstract."""
        _ready(ctx, _ELECTRIFYING)
        _drafts(ctx)
        payload = _payload(_call(ctx, "critique", draft_id="draft-0"))
        assert payload["findings"], "this fixture is meant to breach a threshold"
        finding = payload["findings"][0]
        assert set(finding) == {"metric", "measured", "target", "direction", "rationale", "hint"}
        assert finding["hint"]

    def test_an_unknown_draft_lists_the_ones_the_session_has(self, ctx: ToolContext) -> None:
        _ready(ctx)
        _drafts(ctx)
        invocation = _call(ctx, "critique", draft_id="draft-9")
        assert invocation.error_code == "unknown_draft"
        assert "draft-0" in invocation.result


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
            "MAX_LLM_CALLS_PER_TURN",
            "MAX_SKETCHES_PER_TURN",
            "TOOLS",
            "TURN_DEADLINE_SECONDS",
            "Tool",
            "ToolBudget",
            "ToolContext",
            "ToolError",
            "ToolFailure",
            "ToolRefusal",
            "TurnLedger",
            "dispatch",
            "publish_draft",
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

    def test_the_one_tool_that_needs_a_model_says_so_by_name(self, ctx: ToolContext) -> None:
        assert _call(ctx, "parse_brief").error_code == "llm_not_configured"
