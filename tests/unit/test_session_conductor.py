"""The conductor loop: one model call per turn, and the tools it asks for.

Four load-bearing properties, and the tests are grouped to witness them:

- **A turn is bounded by the harness, not by the model.** One model call, then
  the calls it asked for, and the turn ends. A model that answers in prose, or
  that asks for nothing, or that asks for something the tools refuse, all end up
  in the same place: a recorded `Turn` with what actually happened on it.
- **Nothing about a request is remembered — it is derived.** Two sessions in the
  same state are asked the same question, byte for byte, which is what makes a
  recorded log the whole of what a session did.
- **A refusal is recorded, not raised.** The turn log is the record of what
  happened, so a tool that refused and a tool that broke both leave a turn
  behind; the caller still gets control back.
- **The log replays.** `replay` re-runs a recorded turn's calls against a fresh
  context and reproduces the drafts and their hashes. That is the claim a
  non-deterministic conductor has to earn, and the only test that earns it.

The fixtures compose for real and the model is scripted; the renderer and the
broker are faked at the outer edge only, as in `test_session_tools.py`, because
FluidSynth and RQ are not what any of this is about.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from saimc.jobs.storage import JobStorage
from saimc.llm.base import (
    ChatRequest,
    ChatResult,
    LLMError,
    ParseRequest,
    ParseResult,
    ToolCall,
)
from saimc.session import conductor, tools
from saimc.session.conductor import MAX_DIGEST_TURNS, SYSTEM_PROMPT, digest, replay, take_turn
from saimc.session.models import (
    Publication,
    Session,
    ToolInvocation,
    Turn,
    TurnTrigger,
    Verdict,
)
from saimc.session.store import SessionStorage
from saimc.session.tools import ToolBudget, ToolContext, dispatch, tool_specs
from saimc.spec import CompositionSpec, Mood

_SPEC = CompositionSpec(mood=Mood.CALMING, duration_seconds=30, seed=5)

_MISSING_SPEC = CompositionSpec(mood=Mood.ELECTRIFYING, duration_seconds=30, seed=5)
"""A spec whose candidates all miss `texture_hierarchy`, at every seed.

Named because the digest's findings line is only exercised by a draft that has
one. At `_SPEC` every candidate is clean, so the test that asserts the digest
names what a draft missed iterated an empty set and asserted nothing — and the
missing `or 'nothing'` branch was invisible to coverage, because a branch inside
an f-string is not traced. Checked across a three-wide fan-out before it was
written down, rather than assumed from one seed.
"""


def _call(ctx: ToolContext, name: str, **args: Any) -> Any:
    """Dispatch one call on its own, for the tests that are not about a turn."""
    return asyncio.run(dispatch(ToolCall(name=name, arguments=args), ctx))


def _payload(invocation: Any) -> Any:
    return json.loads(invocation.result)


def _untimed(invocations: Iterable[ToolInvocation]) -> list[ToolInvocation]:
    """The calls as a function of the state, with the clock taken out.

    Every field of a `ToolInvocation` but one is determined by what the model
    asked and what the tool did with it. `duration_ms` is a wall-clock reading
    taken around that work, so the same call twice differs on it and only on
    it — which is the one place a determinism test has to say what it means by
    "the same".
    """
    return [replace(invocation, duration_ms=0) for invocation in invocations]


class _Model:
    """A scripted model that satisfies both protocols and records what it read.

    Both, because the conductor calls `chat` and `parse_brief` calls `parse`,
    and a turn that parses a brief exercises both in one go.
    """

    def __init__(self, *replies: ChatResult, spec: CompositionSpec = _SPEC) -> None:
        self.replies = list(replies)
        self.spec = spec
        self.requests: list[ChatRequest] = []
        self.prompts: list[str] = []

    async def chat(self, request: ChatRequest) -> ChatResult:
        self.requests.append(request)
        if self.replies:
            return self.replies.pop(0)
        return ChatResult(content="Nothing more to ask for.")

    async def parse(self, request: ParseRequest) -> ParseResult:
        self.prompts.append(request.prompt)
        return ParseResult(parser_source="llm", spec=self.spec)

    async def aclose(self) -> None:
        return None


def _reply(*calls: tuple[str, dict[str, Any]], content: str = "", **extra: str) -> ChatResult:
    return ChatResult(
        content=content,
        tool_calls=tuple(ToolCall(name=name, arguments=args) for name, args in calls),
        extra=extra,
    )


def _failed(code: str = "llm_unreachable") -> ChatResult:
    return ChatResult(error=LLMError(code, "the host refused the connection"))


@pytest.fixture
def ctx(tmp_path: Path) -> ToolContext:
    sessions = SessionStorage(tmp_path / "sessions")
    jobs = JobStorage(tmp_path / "jobs")
    session: Session = sessions.create("something for a rainy day")
    return ToolContext(session=session, sessions=sessions, jobs=jobs)


def _ready(ctx: ToolContext, spec: CompositionSpec = _SPEC) -> ToolContext:
    ctx.session.spec = spec
    return ctx


def _turn(ctx: ToolContext, model: _Model, **kwargs: Any) -> Turn:
    trigger: TurnTrigger = kwargs.pop("trigger", "brief")
    return asyncio.run(take_turn(ctx, model, trigger=trigger, **kwargs))


def _model(ctx: ToolContext, *replies: ChatResult, **kwargs: Any) -> _Model:
    """A model wired into the context *and* handed to the conductor.

    Two handles, one object: `take_turn` takes the client it talks through,
    and `parse_brief` reads `ctx.llm`. `_build_default_llm_client` builds one
    adapter that satisfies both protocols over one connection, so the default
    build passes the same object to each — but the conductor's model and the
    parser's are separately replaceable, which is why the seam is two names.
    """
    model = _Model(*replies, **kwargs)
    ctx.llm = model
    return model


def _drafts(ctx: ToolContext, spec: CompositionSpec = _SPEC, **args: Any) -> list[Any]:
    """Draft for real, so the digest is rendered against real measurements."""
    _ready(ctx, spec)
    invocation = _call(ctx, "draft", **args)
    assert invocation.ok, invocation.result
    return ctx.session.drafts


@pytest.fixture
def rendered(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, Any]]:
    """A stand-in for the FluidSynth pass, writing the files it claims to.

    A sketch that wrote nothing would let a replay compare records of audio
    that was never produced, which is the failure this fake exists to make
    impossible.
    """
    calls: list[dict[str, Any]] = []

    def _fake(plan: Any, **kwargs: Any) -> Any:
        out_dir: Path = kwargs["out_dir"]
        job_id: str = kwargs["job_id"]
        calls.append({"plan": plan, "job_id": job_id, "bpm": kwargs["bpm"]})
        (out_dir / f"{job_id}.mid").write_bytes(b"MThd")
        wav = out_dir / "audio.wav"
        wav.write_bytes(b"RIFF")
        ogg = out_dir / "audio.ogg"
        ogg.write_bytes(b"OggS")
        from saimc.render.audio import AudioArtifact

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


class TestTheDigest:
    """What the model is shown about the session, and what it is not."""

    def test_a_session_with_no_spec_says_which_tool_makes_one(self, ctx: ToolContext) -> None:
        text = digest(ctx.session)
        assert "not parsed yet" in text
        assert "parse_brief" in text

    def test_the_spec_renders_only_the_fields_it_actually_set(self, ctx: ToolContext) -> None:
        """An unset field is the engine's choice, which is not a value.

        Showing `key=None` would invite the model to read a choice into a
        field nobody chose — and `key` unset is the common case, because the
        parser leaves it to the engine.
        """
        _ready(ctx, CompositionSpec(mood=Mood.CALMING, duration_seconds=30, seed=5, tempo_bpm=76))
        text = digest(ctx.session)
        assert "mood=calming" in text
        assert "duration_seconds=30s" in text
        assert "tempo_bpm=76" in text
        assert "key=" not in text

    def test_the_instrumentation_reads_as_roles_and_instruments(self, ctx: ToolContext) -> None:
        """Not as a Pydantic repr the model has to parse past.

        The entry's own `repr()` carries the enum's *type* and its value —
        `role=<VoiceRole.MELODY: 'melody'>` — and a digest is read by a model
        deciding what to draft next, not by a debugger.
        """
        _ready(ctx)
        line = digest(ctx.session).splitlines()[1]
        assert "InstrumentationEntry" not in line
        assert "VoiceRole" not in line
        for entry in ctx.session.spec.instrumentation:
            assert f"{entry.role.value}={entry.instrument.value}" in line

    def test_an_empty_session_says_so_rather_than_omitting_the_line(self, ctx: ToolContext) -> None:
        """A missing line reads as "not considered"; the count is the fact."""
        assert "drafts (0): none yet" in digest(ctx.session)

    def test_a_draft_renders_its_id_its_seed_and_what_it_missed(self, ctx: ToolContext) -> None:
        """The findings, not the raw metrics: what is wrong is what a next call is about.

        The premise is asserted rather than trusted. A spec whose candidates are
        all clean would leave the loop below iterating an empty set, and a test
        that asserts nothing about the line it is named for is the failure this
        whole module keeps rediscovering.
        """
        drafts = _drafts(ctx, _MISSING_SPEC, n=2)
        text = digest(ctx.session)
        for index, draft in enumerate(drafts):
            assert f"draft-{index} seed={draft.spec.seed}" in text
        metrics = {finding.metric for draft in drafts for finding in draft.quality.findings()}
        assert metrics, "this spec's candidates are all clean, so this test asserts nothing"
        for metric in metrics:
            assert metric in text, f"{metric} misses but the digest does not say so"

    def test_a_draft_with_nothing_to_report_says_nothing_rather_than_a_blank(
        self, ctx: ToolContext
    ) -> None:
        """A blank would read as a tool that failed; the word is the measurement."""
        drafts = _drafts(ctx)
        clean = [draft for draft in drafts if not draft.quality.findings()]
        assert clean, "this spec now misses something, so it cannot show the clean line"
        assert "misses: nothing" in digest(ctx.session)

    def test_a_sketched_draft_says_so(
        self, ctx: ToolContext, rendered: list[dict[str, Any]]
    ) -> None:
        """The model has to be able to tell audio that exists from a plan."""
        _drafts(ctx)
        assert "sketched" not in digest(ctx.session)
        assert _call(ctx, "sketch", draft_id="draft-0").ok
        assert "sketched" in digest(ctx.session)

    def test_a_verdict_renders_its_value_and_quotes_its_feedback(self, ctx: ToolContext) -> None:
        """A verdict is any of three shapes, and all three reach the model.

        A like with no words, words with no like, and both: the digest shows
        what the user actually gave, because "they said nothing about this one"
        and "they liked it" are different instructions for the next turn.
        """
        _drafts(ctx, n=3)
        at = ctx.session.created_at
        ctx.session.verdicts.extend(
            [
                Verdict(draft_id="draft-0", at=at, value="like", feedback="keep it"),
                Verdict(draft_id="draft-1", at=at, value="dislike"),
                Verdict(draft_id="draft-2", at=at, feedback="warmer, please"),
            ]
        )
        assert "verdicts: draft-0 like 'keep it'; draft-1 dislike; draft-2 'warmer, please'" in (
            digest(ctx.session)
        )

    def test_the_published_line_flips_once_a_session_has_published(self, ctx: ToolContext) -> None:
        assert "published: not yet" in digest(ctx.session)
        _drafts(ctx, n=1)
        ctx.session.publication = Publication(job_id="0f8a4c2b", draft_id="draft-0")
        assert "published: job 0f8a4c2b" in digest(ctx.session)

    def test_a_turn_renders_its_trigger_and_the_calls_in_order(self, ctx: ToolContext) -> None:
        _ready(ctx)
        asyncio.run(
            take_turn(
                ctx,
                _Model(_reply(("draft", {"n": 2}), ("critique", {"draft_id": "draft-0"}))),
                trigger="brief",
            )
        )
        text = digest(ctx.session)
        assert "brief → draft(n=2), critique(draft_id='draft-0')" in text

    def test_a_refused_call_is_marked_as_refused_in_the_digest(self, ctx: ToolContext) -> None:
        """Otherwise the model reads its own failed call as a success and
        repeats it, which is the failure the turn log exists to prevent."""
        asyncio.run(take_turn(ctx, _Model(_reply(("draft", {"n": 1}))), trigger="brief"))
        text = digest(ctx.session)
        assert "draft(n=1) → refused: no_spec" in text

    def test_a_failed_model_call_renders_as_its_error_code(self, ctx: ToolContext) -> None:
        _turn(ctx, _Model(_failed()))
        assert "the model call failed: llm_unreachable" in digest(ctx.session)

    def test_a_silent_turn_is_distinguishable_from_no_turn(self, ctx: ToolContext) -> None:
        _turn(ctx, _Model(ChatResult(content="I have nothing to add.")))
        assert "said nothing and called nothing" in digest(ctx.session)

    def test_only_the_last_few_turns_are_shown_and_the_rest_are_counted(
        self, ctx: ToolContext
    ) -> None:
        """Bounded, but not silently: a model shown six turns of twenty should
        know it is reading the end of a longer session."""
        for _ in range(MAX_DIGEST_TURNS + 3):
            _turn(ctx, _Model(ChatResult(content="pass")))
        text = digest(ctx.session)
        assert "turns (9 in all)" in text
        assert "3 earlier turn(s) not shown" in text
        assert text.count("said nothing and called nothing") == MAX_DIGEST_TURNS

    def test_the_digest_is_a_function_of_the_session(self, ctx: ToolContext) -> None:
        _drafts(ctx, n=2)
        assert digest(ctx.session) == digest(ctx.session)


class TestTheRequest:
    """What leaves the process, and what it is allowed to contain."""

    def test_the_prompt_does_not_restate_the_tool_catalogue(self, ctx: ToolContext) -> None:
        """Policy in the prompt, descriptions in the `ToolSpec`s.

        Two copies of one fact drift, and the copy inside a string literal is
        the one nothing checks. This cannot catch a paraphrase — only a second
        copy — which is exactly the drift that happens.
        """
        catalogue = tool_specs(ctx.budget)
        assert catalogue, "an empty catalogue would make this test vacuous"
        for spec in catalogue:
            assert spec.description not in SYSTEM_PROMPT

    def test_the_tools_offered_are_the_catalogue_at_this_budget(self, ctx: ToolContext) -> None:
        """The model is told the ceiling it is actually held to."""
        ctx.budget = ToolBudget(
            max_candidates=7, max_sketches=2, max_llm_calls=1, deadline_seconds=30
        )
        model = _Model(_reply(("draft", {"n": 1})))
        _turn(ctx, model)
        offered = {spec.name: spec for spec in model.requests[0].tools}
        assert offered["draft"].parameters["properties"]["n"]["maximum"] == 7
        assert set(offered) == set(tools.TOOLS)

    def test_the_first_message_is_the_policy_prompt_and_the_second_is_the_state(
        self, ctx: ToolContext
    ) -> None:
        model = _Model(_reply(("draft", {"n": 1})))
        _ready(ctx)
        _turn(ctx, model)
        system, user = model.requests[0].messages
        assert system.role == "system"
        assert system.content == SYSTEM_PROMPT
        assert user.role == "user"
        assert "something for a rainy day" in user.content

    def test_the_user_message_carries_what_the_user_said(self, ctx: ToolContext) -> None:
        model = _Model(ChatResult(content="noted"))
        _turn(ctx, model, trigger="message", message="make the bass walk")
        assert "The user says: make the bass walk" in model.requests[0].messages[1].content

    @pytest.mark.parametrize(
        ("trigger", "said"),
        [
            ("brief", "This is the session's first turn"),
            ("message", "The user says: make the bass walk"),
            ("auto", "No one has spoken"),
        ],
    )
    def test_each_trigger_names_itself_and_only_itself(
        self, ctx: ToolContext, trigger: str, said: str
    ) -> None:
        """Three triggers, three sentences, and no turn says another's.

        The digest is the same in all three, so what distinguishes a first turn
        from a continuation is this line alone — and the failure it guards
        against is silent, because a turn labelled `brief` that reads like a
        continuation still drafts something plausible.
        """
        model = _Model(ChatResult(content="noted"))
        _turn(ctx, model, trigger=trigger, message="make the bass walk")  # type: ignore[arg-type]
        content = model.requests[0].messages[1].content
        assert said in content
        for other in {"brief", "message", "auto"} - {trigger}:
            assert {
                "brief": "This is the session's first turn",
                "message": "The user says:",
                "auto": "No one has spoken",
            }[other] not in content

    def test_the_request_id_names_the_session_and_which_turn_this_is(
        self, ctx: ToolContext
    ) -> None:
        """One request id per turn is what makes a run log line up with a
        recorded turn, which is the whole of its job."""
        model = _Model(ChatResult(content="a"), ChatResult(content="b"))
        _turn(ctx, model)
        _turn(ctx, model, trigger="auto")
        assert [request.request_id for request in model.requests] == [
            f"{ctx.session.session_id}:0",
            f"{ctx.session.session_id}:1",
        ]

    def test_two_sessions_in_the_same_state_are_asked_the_same_question(
        self, ctx: ToolContext, tmp_path: Path
    ) -> None:
        """The request is derived from the log, not accumulated alongside it.

        This is what makes replay possible at all: nothing has to be
        remembered separately about what the model was once told, because the
        same state produces the same text. Compared in full — messages, tools
        and request id — because a digest that agreed on the state and drifted
        on the catalogue would make the test pass and the claim false. Twice,
        at two turn counts, so the recorded turns are inside the compared state
        and not merely outside it.
        """
        _ready(ctx)
        mirror = SessionStorage(tmp_path / "mirror")

        def twin() -> ToolContext:
            """A second context holding a copy of `ctx`'s session as it stands."""
            mirror.save(Session.from_document(ctx.session.to_document()))
            return ToolContext(
                session=mirror.get(ctx.session.session_id),
                sessions=mirror,
                jobs=JobStorage(tmp_path / "mirror-jobs"),
            )

        asks = (_reply(("draft", {"n": 2})), _reply(("critique", {"draft_id": "draft-0"})))
        mine, theirs = _model(ctx, *asks), _model(ctx, *asks)
        for trigger in ("brief", "auto"):
            elsewhere = twin()
            elsewhere.llm = theirs
            one = _turn(ctx, mine, trigger=trigger)
            two = _turn(elsewhere, theirs, trigger=trigger)
            assert theirs.requests[-1] == mine.requests[-1]
            assert _untimed(two.calls) == _untimed(one.calls)

    def test_the_digest_shown_to_the_model_is_the_digest_function(self, ctx: ToolContext) -> None:
        """One renderer, not two: a second inline rendering would be the copy
        that drifts, and this is the only place that would show it."""
        ctx.session.spec = _SPEC
        before = digest(ctx.session)
        model = _model(ctx, ChatResult(content="pass"))
        _turn(ctx, model)
        assert model.requests[0].messages[1].content.startswith(before)


class TestTakeTurn:
    def test_a_turn_runs_the_calls_it_asked_for_in_the_order_it_asked(
        self, ctx: ToolContext
    ) -> None:
        """Order is observable: the second call reads what the first one wrote."""
        model = _model(ctx, _reply(("parse_brief", {}), ("draft", {"n": 2})))
        turn = _turn(ctx, model)
        assert [call.name for call in turn.calls] == ["parse_brief", "draft"]
        assert all(call.ok for call in turn.calls), [call.result for call in turn.calls]
        assert [draft.draft_id for draft in ctx.session.drafts] == ["draft-0", "draft-1"]

    def test_parse_brief_reads_the_context_and_not_the_conductor(self, ctx: ToolContext) -> None:
        """Two handles, one object — and the tool's is the context's.

        If `take_turn`'s client were what `parse_brief` used, replacing the
        parser's model would silently do nothing, which is the kind of wiring
        mistake that only shows up as a worse spec.
        """
        conductor_model = _model(ctx, _reply(("parse_brief", {})))
        parser_model = _Model(spec=CompositionSpec(mood=Mood.SLEEP, duration_seconds=30, seed=1))
        ctx.llm = parser_model
        turn = _turn(ctx, conductor_model)
        assert turn.calls[0].ok, turn.calls[0].result
        assert parser_model.prompts == ["something for a rainy day"]
        assert ctx.session.spec is not None
        assert ctx.session.spec.mood == Mood.SLEEP

    def test_a_later_call_in_the_same_turn_sees_what_an_earlier_one_did(
        self, ctx: ToolContext, rendered: list[dict[str, Any]]
    ) -> None:
        """The calls are a sequence, not a set: `sketch` names `draft`'s id."""
        _ready(ctx)
        model = _Model(_reply(("draft", {"n": 2}), ("sketch", {"draft_id": "draft-1"})))
        turn = _turn(ctx, model)
        assert all(call.ok for call in turn.calls), [call.result for call in turn.calls]
        assert ctx.session.draft("draft-0").sketch is None
        assert ctx.session.draft("draft-1").sketch is not None

    def test_the_turn_is_appended_and_durable_before_the_caller_sees_it(
        self, ctx: ToolContext
    ) -> None:
        """A turn is a decision whose result the user has already been shown."""
        turn = _turn(ctx, _Model(_reply(("draft", {"n": 1}))))
        assert ctx.session.turns == [turn]
        assert ctx.sessions.get(ctx.session.session_id).turns == [turn]

    def test_the_turn_records_the_model_and_the_latency_it_took(self, ctx: ToolContext) -> None:
        model = _Model(_reply(("draft", {"n": 1}), model_identifier="some-model"))
        turn = _turn(ctx, model)
        assert turn.model == "some-model"
        assert turn.latency_ms >= 0

    def test_narration_is_stripped_before_it_is_recorded(self, ctx: ToolContext) -> None:
        turn = _turn(ctx, _Model(ChatResult(content="  Two candidates.\n\n")))
        assert turn.narration == "Two candidates."

    def test_a_failed_model_call_records_a_turn_with_no_calls_at_all(
        self, ctx: ToolContext
    ) -> None:
        """Nothing ran, so the turn may not claim anything did."""
        turn = _turn(ctx, _Model(_failed()))
        assert turn.failed is True
        assert turn.calls == ()
        assert turn.llm_error is not None
        assert turn.llm_error.error_code == "llm_unreachable"
        assert ctx.sessions.get(ctx.session.session_id).turns[0].failed is True

    def test_a_refused_call_is_recorded_rather_than_raised(self, ctx: ToolContext) -> None:
        """The caller gets control back; the turn log is where the refusal is.

        A turn is not sunk by a tool that refused, because the model that asked
        for it is the one that should hear about it — on the next turn's digest.
        """
        _ready(ctx)
        turn = _turn(ctx, _Model(_reply(("sketch", {"draft_id": "draft-9"}))))
        assert turn.calls[0].outcome == "refused"
        assert turn.calls[0].error_code == "unknown_draft"
        assert turn.failed is False

    def test_a_turn_that_refuses_the_same_call_twice_records_both_attempts(
        self, ctx: ToolContext
    ) -> None:
        """The log is a record, so it does not deduplicate: the model asking
        twice is itself the thing worth seeing."""
        _ready(ctx)
        turn = _turn(
            ctx,
            _Model(
                _reply(("critique", {"draft_id": "draft-9"}), ("critique", {"draft_id": "draft-9"}))
            ),
        )
        assert [call.error_code for call in turn.calls] == ["unknown_draft", "unknown_draft"]

    def test_each_turn_starts_with_a_fresh_budget(self, ctx: ToolContext) -> None:
        """Otherwise a long session would starve on its own history.

        Budgets are spent per turn, and `begin_turn` is what the conductor
        calls before each one. So the *same* call that was refused for budget
        on one turn succeeds on the next, which is the whole reason a session
        has more than one turn.
        """
        ctx.budget = ToolBudget(
            max_candidates=4, max_sketches=4, max_llm_calls=1, deadline_seconds=60
        )
        _ready(ctx)
        twice = _model(
            ctx, _reply(("parse_brief", {}), ("parse_brief", {})), _reply(("parse_brief", {}))
        )
        first = _turn(ctx, twice)
        assert (first.calls[0].outcome, first.calls[0].error_code) == ("ok", None)
        assert (first.calls[1].outcome, first.calls[1].error_code) == (
            "refused",
            "budget_exhausted",
        )
        second = _turn(ctx, twice, trigger="auto")
        assert second.calls[0].ok, second.calls[0].result

    def test_a_turn_with_no_spec_still_records_what_the_model_asked_for(
        self, ctx: ToolContext
    ) -> None:
        turn = _turn(ctx, _Model(_reply(("draft", {"n": 1}))))
        assert turn.calls[0].arguments == {"n": 1}
        assert turn.calls[0].result.startswith("this session has no spec yet")


class TestReplay:
    """The log re-runs. This is what a non-deterministic conductor costs."""

    def _recorded(self, ctx: ToolContext, model: _Model, *, sketched: bool = True) -> Session:
        """A session with a real log: a brief, two drafts, and a sketch."""
        _ready(ctx)
        ctx.llm = model
        asyncio.run(take_turn(ctx, model, trigger="brief"))
        asyncio.run(take_turn(ctx, model, trigger="auto"))
        assert [draft.draft_id for draft in ctx.session.drafts] == ["draft-0", "draft-1"]
        assert (ctx.session.draft("draft-0").sketch is not None) is sketched
        return ctx.session

    def test_a_recorded_log_replays_to_the_drafts_it_recorded(
        self, ctx: ToolContext, tmp_path: Path, rendered: list[dict[str, Any]]
    ) -> None:
        """The claim: the calls reproduce the state, so the log is faithful.

        The model is skipped because it is recorded; the state it left behind
        is what is compared, by the hashes of the notes and the performance
        each draft describes rather than by anything about how it was made.
        """
        model = _model(
            ctx,
            _reply(("draft", {"n": 2}), content="Here is where we are."),
            _reply(("sketch", {"draft_id": "draft-0"})),
        )
        recorded = self._recorded(ctx, model)

        fresh = SessionStorage(tmp_path / "replay")
        replayed_session = Session.from_document(recorded.to_document())
        replayed_session.turns = []
        replayed_session.drafts = []
        fresh.save(replayed_session)
        into = ToolContext(
            session=fresh.get(recorded.session_id),
            sessions=fresh,
            jobs=JobStorage(tmp_path / "replay-jobs"),
        )

        invocations = asyncio.run(replay(recorded.turns, into))

        assert [invocation.name for invocation in invocations] == ["draft", "sketch"]
        assert all(invocation.ok for invocation in invocations), [
            invocation.result for invocation in invocations
        ]
        for field in ("performance_plan_hash", "plan_hash", "score_hash"):
            assert [getattr(draft, field) for draft in into.session.drafts] == [
                getattr(draft, field) for draft in recorded.drafts
            ], field

    def test_a_replay_does_not_read_the_session_it_replays(
        self, ctx: ToolContext, rendered: list[dict[str, Any]]
    ) -> None:
        """It is given a context, not a session: replaying into the session
        that already holds the drafts would append a second copy of each."""
        recorded = self._recorded(ctx, _model(ctx, _reply(("draft", {"n": 2}))), sketched=False)
        before = len(recorded.drafts)
        asyncio.run(replay(recorded.turns[:1], ctx))
        assert len(ctx.session.drafts) == before + 2

    def test_a_replay_leaves_the_log_it_reads_untouched(
        self, ctx: ToolContext, rendered: list[dict[str, Any]]
    ) -> None:
        """A replay is a read. `dispatch` is handed a rebuilt call with copied
        arguments, so a handler that mutated what it was given would be
        editing the record of a turn that already happened."""
        recorded = self._recorded(
            ctx,
            _model(
                ctx,
                _reply(("draft", {"n": 2})),
                _reply(("sketch", {"draft_id": "draft-0"})),
            ),
        )
        before = [turn.to_document() for turn in recorded.turns]
        asyncio.run(replay(recorded.turns, ctx))
        assert [turn.to_document() for turn in recorded.turns] == before

    def test_a_failed_turn_contributes_nothing_to_a_replay(self, ctx: ToolContext) -> None:
        """It asked for no work, so there is no work to re-run."""
        _ready(ctx)
        asyncio.run(take_turn(ctx, _Model(_failed()), trigger="brief"))
        asyncio.run(take_turn(ctx, _Model(_reply(("draft", {"n": 1}))), trigger="auto"))
        invocations = asyncio.run(replay(ctx.session.turns, ctx))
        assert [invocation.name for invocation in invocations] == ["draft"]

    def test_a_replay_without_a_model_refuses_visibly_rather_than_silently(
        self, ctx: ToolContext
    ) -> None:
        """A log that begins with `parse_brief` needs a model or a seeded spec.

        Both refusals are read off the returned invocations rather than raised,
        and the second is downstream of the first: the brief could not be
        parsed, so the draft it would have composed from has no spec. That is
        why the reason matters — `no_spec` alone would send the caller looking
        for a missing spec instead of an unconfigured model.
        """
        model = _model(ctx, _reply(("parse_brief", {})), _reply(("draft", {"n": 1})))
        asyncio.run(take_turn(ctx, model, trigger="brief"))
        asyncio.run(take_turn(ctx, model, trigger="auto"))
        assert ctx.session.drafts, "the recording itself must have drafted"

        seedless = SessionStorage(ctx.sessions.root.parent / "seedless")
        blank = seedless.create("something for a rainy day")
        into = ToolContext(
            session=blank,
            sessions=seedless,
            jobs=JobStorage(ctx.jobs.root.parent / "seedless-jobs"),
        )
        invocations = asyncio.run(replay(ctx.session.turns, into))

        assert [invocation.error_code for invocation in invocations] == [
            "llm_not_configured",
            "no_spec",
        ]

    def test_the_replayed_turns_come_back_in_the_order_they_were_recorded(
        self, ctx: ToolContext
    ) -> None:
        _ready(ctx)
        asyncio.run(take_turn(ctx, _Model(_reply(("draft", {"n": 2}))), trigger="brief"))
        asyncio.run(
            take_turn(ctx, _Model(_reply(("critique", {"draft_id": "draft-1"}))), trigger="auto")
        )
        blank = Session.from_document(ctx.session.to_document())
        blank.drafts = []
        blank.turns = ctx.session.turns
        into = ToolContext(
            session=blank,
            sessions=ctx.sessions,
            jobs=ctx.jobs,
            budget=ctx.budget,
        )
        invocations = asyncio.run(replay(blank.turns, into))
        assert [invocation.name for invocation in invocations] == ["draft", "critique"]
        assert _payload(invocations[1])["draft_id"] == "draft-1"


class TestTheLoopIsBoundedByOneModelCall:
    def test_one_turn_is_exactly_one_chat_call(self, ctx: ToolContext) -> None:
        """The type's own invariant is why: a turn cannot record both a failed
        model call and the tools that ran, so a second call mid-turn would make
        a mid-turn failure unrepresentable."""
        model = _Model(_reply(("draft", {"n": 1})), ChatResult(content="and also"))
        _ready(ctx)
        _turn(ctx, model)
        assert len(model.requests) == 1

    def test_a_model_that_answers_in_prose_makes_a_turn_with_no_calls(
        self, ctx: ToolContext
    ) -> None:
        """Prose is a normal outcome, not an error — the conductor decides
        whether words are an acceptable answer, and stopping is one."""
        turn = _turn(ctx, _Model(ChatResult(content="Nothing is worth doing yet.")))
        assert turn.calls == ()
        assert turn.failed is False
        assert turn.narration == "Nothing is worth doing yet."

    def test_the_conductor_module_names_the_one_model_call_it_makes(self) -> None:
        """A ratchet: a second call site added here would have to argue with
        this test, and the argument is the thing worth having."""
        source = Path(conductor.__file__).read_text()
        assert source.count("await client.chat(") == 1
