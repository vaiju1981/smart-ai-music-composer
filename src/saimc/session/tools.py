"""What the conductor may call, and what one turn is allowed to spend.

A tool is three things: a name the model asks for, a JSON Schema telling it
what the arguments mean, and an async handler that returns the text the model
reads back. The handler is the only place a session is mutated — the conductor
decides, the tools act, and neither of them writes a note. Every note in a
draft came out of `compose`.

Three rules hold this module together.

**Every call and result is logged into the turn.** The text a handler returns
is the same text fed back to the model as the `tool` message, so the turn log
records exactly what the next turn saw and a replay re-derives nothing it
could have read. That is what makes a session replayable, and it is why a
handler returns `str` rather than a typed object the dispatcher would have to
render a second time and could render differently.

**A tool refuses; it never no-ops.** A refusal is an answer: a named reason
code and a sentence the conductor can act on, recorded with
`outcome="refused"` rather than raised past the loop. Returning an empty
success for something the engine could not do would teach the model that the
call worked and teach the user that the product is deaf. `ToolFailure` is the
other half — the tool *broke* — and the log keeps the two apart because they
are different sentences to read.

**Budgets are in code, not in a prompt.** `ToolBudget` is what a turn may
spend and `TurnLedger` is what it has spent; neither moves because the model
asked nicely. `tool_specs` builds each schema *from* the budget, so the
`maximum` the model is told cannot disagree with the limit it is held to.

One table (`TOOLS`) serves both the catalogue and the dispatcher. Two tables
keyed by the same names would be two chances for a name to exist in one and
not the other; one table means a tool that is offered is a tool that runs.

What is deliberately *not* here: `revise`, `apply_delta` and `compare`. Each
takes a `Delta`, and the delta vocabulary is Phase D's deliverable — writing
them now would mean inventing a placeholder vocabulary here and rewriting the
appliers when the real one lands. The fan-out they will rank already exists.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Final

from saimc.compose.engine import CompositionEngineError, EngineOutput, compose
from saimc.compose.linter import lint
from saimc.jobs.storage import Job, JobStorage
from saimc.jobs.worker import QueueUnavailable, enqueue_or_fail
from saimc.llm.base import LLMClient, ToolCall, ToolSpec
from saimc.parser import parse_prompt
from saimc.quality import score_piece
from saimc.render.audio import render_sketch
from saimc.render.instruments import resolve_job_soundfont
from saimc.session.models import (
    Draft,
    Session,
    SketchRecord,
    ToolInvocation,
    ToolOutcome,
)
from saimc.session.store import SKETCHES_DIRNAME, SessionStorage
from saimc.spec import CompositionSpec

logger = logging.getLogger(__name__)

MAX_CANDIDATES_PER_DRAFT: Final[int] = 4
"""How wide one `draft` call may fan out.

The plan's own default, and the answer to the question it left open: the
conductor decides per brief *whether* to fan out, and this decides how far it
can. Four is the width at which a fan-out costs the measured worst case of a
turn — four sketches is the whole of it, ~20 s at the 600 s cap.
"""

MAX_SKETCHES_PER_TURN: Final[int] = 4
"""How many sketches one turn may render. One per candidate it can draft."""

MAX_LLM_CALLS_PER_TURN: Final[int] = 2
"""How many model calls the *tools* may make in one turn.

The conductor's own call is not counted: this bounds what a tool spends once
it is running, and only `parse_brief` spends any — a turn that re-parses a
brief the user has just rephrased, and then again once, is the most that is
ever useful.
"""

TURN_DEADLINE_SECONDS: Final[float] = 120.0
"""Wall clock for one turn's tools. Six times the measured worst case."""


class ToolError(Exception):
    """A tool's answer, when the answer is not a result.

    Both subclasses carry the code the conductor branches on and the sentence
    the model reads. `outcome` is what the turn log records, and the two
    values stay distinct because "declined, with a reason" and "broke" are
    different things to be told.
    """

    outcome: ToolOutcome

    def __init__(self, error_code: str, message: str) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.message = message


class ToolRefusal(ToolError):
    """The tool understood the call and declined it, naming why.

    A refusal is not a failure of the harness: an unfillable duration, a
    budget already spent, a draft this session does not have. The model is
    expected to read the reason and choose differently.
    """

    outcome = "refused"


class ToolFailure(ToolError):
    """The tool broke: the engine raised, the render died, a draft drifted.

    Told apart from a refusal so that a user looking at the log can see the
    difference between "the product said no" and "the product fell over".
    """

    outcome = "error"


@dataclass(frozen=True)
class ToolBudget:
    """What one turn may spend. Frozen, because a limit that can move is not one.

    `max_candidates` bounds a single `draft` call's *width* — it is the
    fan-out ceiling, not a per-turn total, which is why it is read by the tool
    and not counted in the ledger: composing a candidate is about 10 ms of
    arithmetic and the turn's real cost is the sketching that follows.

    `max_sketches`, `max_llm_calls` and the deadline bound the *turn*, and are
    the ledger's business. Each sketch is a FluidSynth pass — 0.65 s at 30 s
    of music, 4.81 s at the 600 s cap — so the sketch count is the number that
    decides how long a turn takes.

    The deadline is checked between calls rather than during one, so the
    guarantee is "no *new* work starts after this". That is the honest form
    when the work is a subprocess nobody can interrupt cleanly.
    """

    max_candidates: int = MAX_CANDIDATES_PER_DRAFT
    max_sketches: int = MAX_SKETCHES_PER_TURN
    max_llm_calls: int = MAX_LLM_CALLS_PER_TURN
    deadline_seconds: float = TURN_DEADLINE_SECONDS

    def __post_init__(self) -> None:
        for label, value in (
            ("max_candidates", self.max_candidates),
            ("max_sketches", self.max_sketches),
            ("max_llm_calls", self.max_llm_calls),
        ):
            if value < 1:
                raise ValueError(f"{label} must allow at least one call, got {value}")
        if self.deadline_seconds <= 0:
            raise ValueError(f"deadline_seconds must be positive, got {self.deadline_seconds}")


@dataclass
class TurnLedger:
    """What a turn has spent so far. One per turn, made by `begin_turn`.

    Counters move *after* a spend is decided and *before* it is made, so a
    call that is refused consumes nothing and a call that is made and then
    fails still consumes its share. That ordering is the whole reason this is
    a separate object from `ToolBudget`: the budget is policy and the ledger
    is history, and only history can be corrupted by getting the order wrong.
    """

    started: float = field(default_factory=time.monotonic)
    sketches: int = 0
    llm_calls: int = 0

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started

    def expired(self, budget: ToolBudget) -> bool:
        return self.elapsed_seconds() >= budget.deadline_seconds


@dataclass
class ToolContext:
    """Everything a handler is allowed to touch.

    Passing this rather than a bare `Session` is what keeps a tool honest
    about its reach: a handler can append a draft, write audio into the
    session's own sketch directory, and create the one job the session
    finalizes into — and it holds no other handle on the world.

    `llm` is optional on purpose. The deterministic tools have to run without
    one, which is what makes "feedback degrades" a property of the code rather
    than a promise about it.
    """

    session: Session
    sessions: SessionStorage
    jobs: JobStorage
    budget: ToolBudget = field(default_factory=ToolBudget)
    ledger: TurnLedger = field(default_factory=TurnLedger)
    llm: LLMClient | None = None

    def begin_turn(self) -> None:
        """Start a fresh turn's spend. The conductor calls this once per turn."""
        self.ledger = TurnLedger()


ToolHandler = Callable[[ToolContext, Mapping[str, Any]], Awaitable[str]]
ToolParameters = Callable[[ToolBudget], dict[str, Any]]


def _int_arg(
    args: Mapping[str, Any],
    name: str,
    *,
    default: int | None = None,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    """Read an integer argument, refusing anything else by name.

    `default=None` means the argument is required, and there is no other way
    to say so: every tool that has an optional integer also has a computable
    one to fall back on.

    A model that sends `"n": "2"` or `"n": 2.5` gets a refusal naming the
    argument rather than a Python error the turn log would have to explain.
    Bools are refused explicitly — `isinstance(True, int)` is true, and a
    `true` where a width was expected is a model that has misunderstood.
    """
    raw = args.get(name, default)
    if raw is None:
        raise ToolRefusal("invalid_arguments", f"{name} is required")
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ToolRefusal("invalid_arguments", f"{name} must be an integer, got {raw!r}")
    if minimum is not None and raw < minimum:
        raise ToolRefusal("invalid_arguments", f"{name} must be at least {minimum}, got {raw}")
    if maximum is not None and raw > maximum:
        raise ToolRefusal("invalid_arguments", f"{name} must be at most {maximum}, got {raw}")
    return int(raw)


def _str_arg(args: Mapping[str, Any], name: str, *, default: str | None = None) -> str | None:
    """Read a string argument, or `None` when it is absent."""
    raw = args.get(name, default)
    if raw is None:
        return None
    if not isinstance(raw, str) or not raw.strip():
        raise ToolRefusal("invalid_arguments", f"{name} must be a non-empty string, got {raw!r}")
    return raw


def _named_draft(ctx: ToolContext, args: Mapping[str, Any]) -> Draft:
    """Resolve the `draft_id` argument against this session's drafts.

    The list of ids it *does* have goes into the refusal: a model that guessed
    an id can correct itself in the next turn instead of asking again.
    """
    draft_id = _str_arg(args, "draft_id")
    if draft_id is None:
        raise ToolRefusal("invalid_arguments", "draft_id is required")
    try:
        return ctx.session.draft(draft_id)
    except KeyError as exc:
        known = [draft.draft_id for draft in ctx.session.drafts]
        raise ToolRefusal(
            "unknown_draft",
            f"this session has no draft {draft_id!r}; it has {known or 'none'}. "
            "Call draft first, or use one of those ids.",
        ) from exc


def _render(payload: Mapping[str, Any]) -> str:
    """A tool's answer, as the text the model reads and the log keeps."""
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def _with_seed(spec: CompositionSpec, seed: int) -> CompositionSpec:
    """The same spec at a different seed — validated, not copied blindly.

    `model_copy(update=...)` does not re-run a field's own `ge=0`, which is
    how a candidate could be composed at a seed the spec type would refuse.
    Round-tripping through `model_validate` puts the value back through the
    model that defines what the value may be, so a bad seed refuses by name.
    """
    return CompositionSpec.model_validate({**spec.model_dump(mode="json"), "seed": seed})


def _draft_from(draft_id: str, spec: CompositionSpec, output: EngineOutput) -> Draft:
    """One candidate, recorded from what the engine actually produced.

    The plan is read from the output rather than re-derived with
    `default_plan(spec)`. They are the same value today, and the one the
    engine published is the one it composed under — the distinction the oracle
    test makes for the same reason.

    The lint report is recomputed: `EngineOutput` does not carry it and
    nothing hashes it. That is a second lint of the same score, not a second
    measurement of a different one.
    """
    plan = output.plan
    assert plan is not None  # compose always publishes the plan it composed under
    return Draft(
        draft_id=draft_id,
        created_at=datetime.now(UTC),
        spec=spec,
        plan=plan,
        performance_plan_hash=output.performance_plan.compute_hash(),
        quality=score_piece(output.notation_score, piece=draft_id),
        lint=lint(
            output.notation_score,
            chord_bars=output.chord_bars or None,
            bar_keys=output.bar_keys or None,
            voice_instruments={v.voice_id: v.instrument for v in output.voice_instruments},
        ),
    )


async def _parse_brief(ctx: ToolContext, args: Mapping[str, Any]) -> str:
    """Turn the brief — or a replacement text — into the session's spec."""
    if ctx.llm is None:
        raise ToolRefusal(
            "llm_not_configured",
            "no language model is configured, so a brief cannot be parsed into a spec. "
            "The session needs a spec before anything can be drafted from it.",
        )
    if ctx.ledger.llm_calls >= ctx.budget.max_llm_calls:
        raise ToolRefusal(
            "budget_exhausted",
            f"this turn has spent its {ctx.budget.max_llm_calls} model call(s). Start a new "
            "turn, or work with the spec the session already has.",
        )
    text = _str_arg(args, "text") or ctx.session.brief
    ctx.ledger.llm_calls += 1
    result = await parse_prompt(ctx.llm, text, request_id=ctx.session.session_id)
    if result.error is not None:
        raise ToolRefusal(result.error.error_code, result.error.message)
    spec = result.spec
    assert spec is not None  # ParseResult carries exactly one of spec or error
    ctx.session.spec = spec
    return _render(
        {
            "spec": spec.model_dump(mode="json"),
            "parser_source": result.parser_source,
            "attempts": result.attempts,
        }
    )


async def _draft(ctx: ToolContext, args: Mapping[str, Any]) -> str:
    """Compose, score and lint candidates from the session's spec."""
    spec = ctx.session.spec
    if spec is None:
        raise ToolRefusal(
            "no_spec",
            "this session has no spec yet, so there is nothing to draft from. Call "
            "parse_brief first.",
        )
    width = _int_arg(args, "n", default=1, minimum=1, maximum=ctx.budget.max_candidates)
    base = _int_arg(args, "seed", default=spec.seed if spec.seed is not None else 0, minimum=0)

    first_new_index = len(ctx.session.drafts)
    made: list[Draft] = []
    refused: list[dict[str, Any]] = []
    for offset in range(width):
        seed = base + offset
        candidate = spec if spec.seed == seed else _with_seed(spec, seed)
        try:
            output = compose(candidate)
        except CompositionEngineError as exc:
            # One candidate the engine cannot honour does not sink the
            # fan-out: the other seeds may well compose, and the reason is
            # reported beside them rather than replacing them.
            refused.append({"seed": seed, "error_code": str(exc.code), "reason": exc.message})
            continue
        made.append(_draft_from(f"draft-{first_new_index + len(made)}", candidate, output))

    if not made:
        first = refused[0]
        raise ToolRefusal(str(first["error_code"]), str(first["reason"]))

    ctx.session.drafts.extend(made)
    return _render(
        {
            "drafts": [
                {
                    "draft_id": draft.draft_id,
                    "seed": draft.spec.seed,
                    "lint_passed": draft.lint.passed,
                    "sketched": draft.sketch is not None,
                    "quality": draft.quality.entry(),
                }
                for draft in made
            ],
            "refused": refused,
        }
    )


async def _critique(ctx: ToolContext, args: Mapping[str, Any]) -> str:
    """Report every quality threshold one draft misses, and what moves it."""
    draft = _named_draft(ctx, args)
    return _render(
        {
            "draft_id": draft.draft_id,
            "lint_passed": draft.lint.passed,
            "lint_issues": [
                {"code": str(issue.code), "message": issue.message} for issue in draft.lint.issues
            ],
            "measured": draft.quality.as_dict(),
            "findings": [asdict(finding) for finding in draft.quality.findings()],
        }
    )


async def _sketch(ctx: ToolContext, args: Mapping[str, Any]) -> str:
    """Render one draft's audio, fast enough to sit inside a turn."""
    draft = _named_draft(ctx, args)
    if ctx.ledger.sketches >= ctx.budget.max_sketches:
        raise ToolRefusal(
            "budget_exhausted",
            f"this turn has already spent its {ctx.budget.max_sketches} sketch(es), and each "
            "one renders audio. A new turn is what buys another.",
        )

    output = compose(draft.spec, plan=draft.plan)
    if output.performance_plan.compute_hash() != draft.performance_plan_hash:
        raise ToolFailure(
            "performance_mismatch",
            f"draft {draft.draft_id} recomposed to a different performance than the one it "
            "recorded, so the draft no longer describes the music it claims to.",
        )

    voice_instruments = {v.voice_id: v.instrument for v in output.voice_instruments}
    out_dir = ctx.sessions.sketch_dir(ctx.session.session_id, draft.draft_id)
    # Spent before it is made: FluidSynth is about to run whatever happens
    # next, and a render that dies still cost the turn its seconds.
    ctx.ledger.sketches += 1
    started = time.monotonic()
    # A sketch is seconds of subprocesses; the conductor runs on the API's
    # event loop, and blocking it is how one user's turn freezes another's.
    artifact = await asyncio.to_thread(
        render_sketch,
        output.performance_plan,
        bpm=output.arrangement.tempo_bpm,
        soundfont_path=resolve_job_soundfont(voice_instruments),
        out_dir=out_dir,
        job_id=draft.draft_id,
        tempo_changes=output.notation_score.tempo.changes,
        voice_instruments=voice_instruments,
    )
    if artifact.ogg_path is None or artifact.ogg_sha256 is None or artifact.ogg_size_bytes is None:
        raise ToolFailure(
            "sketch_incomplete",
            f"the sketch render for {draft.draft_id} wrote no Ogg, so there is nothing to play.",
        )

    record = SketchRecord(
        wav_path=f"{SKETCHES_DIRNAME}/{draft.draft_id}/{artifact.primary_path.name}",
        ogg_path=f"{SKETCHES_DIRNAME}/{draft.draft_id}/{artifact.ogg_path.name}",
        ogg_sha256=artifact.ogg_sha256,
        ogg_size_bytes=artifact.ogg_size_bytes,
    )
    ctx.session.replace_draft(replace(draft, sketch=record))
    return _render(
        {
            "draft_id": draft.draft_id,
            "ogg_path": record.ogg_path,
            "seconds": round(time.monotonic() - started, 2),
            "size_bytes": record.ogg_size_bytes,
        }
    )


def publish_draft(ctx: ToolContext, draft: Draft) -> Job:
    """Publish `draft`: create the job that renders it, queue it, and record it.

    The policy, once, for two callers that report it differently: the
    conductor's `finalize` tool answers with text the model reads, and the
    session API answers with a status code. Both get the same record and the
    same refusal codes, which is why this is a function and not two.

    Both refusals are `ToolRefusal`s rather than exceptions past the caller.
    `already_finalized`, because a session publishes once and a second publish
    is a decision rather than a side effect. `queue_unavailable`, because the
    broker would not take the job: the record is marked failed, the session has
    **not** published, and the caller may try again — which the message says,
    since a caller that assumed otherwise would go looking for a render that
    does not exist.
    """
    if ctx.session.finalized_job_id is not None:
        raise ToolRefusal(
            "already_finalized",
            f"this session has already published a piece as job {ctx.session.finalized_job_id}, "
            "and publishing again is a decision rather than a side effect. A published piece "
            "cannot be unpublished — its render is already queued — so a different piece needs "
            "a new session.",
        )
    job = ctx.jobs.create(ctx.session.brief)
    job.input_spec = draft.spec
    job.input_plan = draft.plan
    # Never None: `_draft` composes every candidate at a concrete seed, so a
    # draft's spec always names one. The two other places a job's seed is set
    # (the from-spec endpoint and the compose stage) assign it the same way.
    job.seed = draft.spec.seed
    # A finalized draft *is* an already-validated spec, which is what this
    # parser_source value has always meant on the from-spec endpoint.
    job.parser_source = "from-spec"
    ctx.jobs.save(job)
    try:
        enqueue_or_fail(job, ctx.jobs)
    except QueueUnavailable as exc:
        raise ToolRefusal("queue_unavailable", str(exc)) from exc

    ctx.session.finalized_job_id = job.job_id
    # Saved here, not left to the end of the turn: the job is already
    # durably queued and is about to render, so the record that this session
    # published *it* has to be durable at the moment it becomes true.
    ctx.sessions.save(ctx.session)
    return job


async def _finalize(ctx: ToolContext, args: Mapping[str, Any]) -> str:
    """Publish one draft, on the user's behalf or the conductor's."""
    draft = _named_draft(ctx, args)
    job = publish_draft(ctx, draft)
    return _render({"job_id": job.job_id, "draft_id": draft.draft_id, "state": job.state.value})


_PARSE_BRIEF_PARAMETERS: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "text": {
            "type": "string",
            "description": "The brief to parse. Omit to parse the session's own brief.",
        },
    },
    "additionalProperties": False,
}

_CRITIQUE_PARAMETERS: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "draft_id": {
            "type": "string",
            "description": "The id of the draft to read, exactly as `draft` returned it.",
        },
    },
    "required": ["draft_id"],
    "additionalProperties": False,
}

_SKETCH_PARAMETERS: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "draft_id": {
            "type": "string",
            "description": "The id of the draft to render audio for, exactly as `draft` returned it.",
        },
    },
    "required": ["draft_id"],
    "additionalProperties": False,
}

_FINALIZE_PARAMETERS: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "draft_id": {
            "type": "string",
            "description": "The id of the draft to publish, exactly as `draft` returned it.",
        },
    },
    "required": ["draft_id"],
    "additionalProperties": False,
}


def _draft_parameters(budget: ToolBudget) -> dict[str, Any]:
    """The `draft` schema, sized by the budget the dispatcher enforces.

    This is the one schema that depends on a limit, and it is built from the
    budget rather than written beside it so the `maximum` the model is told
    and the `maximum` `_int_arg` holds it to cannot be two numbers.
    """
    return {
        "type": "object",
        "properties": {
            "n": {
                "type": "integer",
                "minimum": 1,
                "maximum": budget.max_candidates,
                "description": (
                    "How many candidates to compose. Each is the session's spec at a "
                    "different seed. A fan-out costs no model calls and about ten "
                    "milliseconds a candidate, and is worth it on an open brief where "
                    "several readings are possible — not on a specific one, where every "
                    "candidate would be a way of obeying the same instruction."
                ),
            },
            "seed": {
                "type": "integer",
                "minimum": 0,
                "description": (
                    "The seed of the first candidate; the rest follow it. Omit to continue "
                    "from the session's own seed. The same seed always produces the same "
                    "music, which is what makes a draft replays."
                ),
            },
        },
        "additionalProperties": False,
    }


def _static(schema: dict[str, Any]) -> ToolParameters:
    """A schema that does not depend on the budget.

    The indirection exists so every entry in `TOOLS` has the same
    `parameters` type — the one that reads the budget and the four that do
    not — and `tool_specs` can build the catalogue without knowing which is
    which, or which tool will need the budget next.
    """

    def parameters(_budget: ToolBudget) -> dict[str, Any]:
        return schema

    return parameters


@dataclass(frozen=True)
class Tool:
    """One entry in the catalogue and in the dispatcher, at once."""

    name: str
    description: str
    parameters: ToolParameters
    handler: ToolHandler


TOOLS: Final[Mapping[str, Tool]] = {
    "parse_brief": Tool(
        name="parse_brief",
        description=(
            "Read the session's brief and record the musical spec it asks for: mood, "
            "duration, tempo, key, time signature and ensemble. Every draft is composed "
            "from that spec. Call this before drafting, and call it again if what the "
            "user means has changed. Uses the model."
        ),
        parameters=_static(_PARSE_BRIEF_PARAMETERS),
        handler=_parse_brief,
    ),
    "draft": Tool(
        name="draft",
        description=(
            "Compose candidate pieces from the session's spec and score and lint them. "
            "No model call, and no audio: a draft is measurements until you sketch it. "
            "The candidates come back with their ids and their measured quality, and "
            "those ids are what critique, sketch and finalize take."
        ),
        parameters=_draft_parameters,
        handler=_draft,
    ),
    "critique": Tool(
        name="critique",
        description=(
            "Read one draft's measurements against the quality thresholds and report "
            "every bar it misses: the metric, what it measured, the target, why the "
            "target exists, and the hint naming what would move it. These are arithmetic "
            "over the score rather than an opinion, so they cost nothing and never vary."
        ),
        parameters=_static(_CRITIQUE_PARAMETERS),
        handler=_critique,
    ),
    "sketch": Tool(
        name="sketch",
        description=(
            "Render a draft's audio so it can be listened to. Takes a second or two at "
            "ordinary lengths and longer on very long pieces, and a turn may only render "
            "a few, so sketch the drafts worth hearing rather than all of them."
        ),
        parameters=_static(_SKETCH_PARAMETERS),
        handler=_sketch,
    ),
    "finalize": Tool(
        name="finalize",
        description=(
            "Publish a draft: start the full, mastered render of it as a job the user "
            "can download. This is the end of the session's work on that piece, not a "
            "step in it, and a session publishes once."
        ),
        parameters=_static(_FINALIZE_PARAMETERS),
        handler=_finalize,
    ),
}


def tool_specs(budget: ToolBudget | None = None) -> tuple[ToolSpec, ...]:
    """The catalogue offered to the model, built from the budget."""
    limits = budget or ToolBudget()
    return tuple(
        ToolSpec(name=tool.name, description=tool.description, parameters=tool.parameters(limits))
        for tool in TOOLS.values()
    )


def _ended(
    call: ToolCall,
    started: float,
    *,
    result: str,
    outcome: ToolOutcome = "ok",
    error_code: str | None = None,
) -> ToolInvocation:
    return ToolInvocation(
        name=call.name,
        arguments=dict(call.arguments),
        result=result,
        outcome=outcome,
        error_code=error_code,
        duration_ms=int((time.monotonic() - started) * 1000),
    )


async def dispatch(call: ToolCall, ctx: ToolContext) -> ToolInvocation:
    """Run one tool call and record how it ended — never by raising.

    The turn log is the record of what happened, so a call that refused, a
    call that broke and a call that succeeded are all *answers* here. The one
    thing this cannot return is nothing: an unexpected exception is caught,
    logged with its traceback, and reported as `tool_crashed` carrying the
    exception's own text, because a conductor that receives no reply cannot
    even tell the user that it failed.
    """
    started = time.monotonic()
    tool = TOOLS.get(call.name)
    if tool is None:
        return _ended(
            call,
            started,
            result=(
                f"there is no tool named {call.name!r}; this session offers "
                f"{sorted(TOOLS)}. Call one of those, or answer in words."
            ),
            outcome="refused",
            error_code="unknown_tool",
        )
    if ctx.ledger.expired(ctx.budget):
        return _ended(
            call,
            started,
            result=(
                f"this turn has been running for {ctx.ledger.elapsed_seconds():.0f}s, past "
                f"its {ctx.budget.deadline_seconds:.0f}s budget, so no new work starts. "
                "Report what was done and let the next turn continue."
            ),
            outcome="refused",
            error_code="turn_deadline",
        )
    try:
        result = await tool.handler(ctx, call.arguments)
    except ToolError as exc:
        return _ended(
            call, started, result=exc.message, outcome=exc.outcome, error_code=exc.error_code
        )
    except Exception as exc:
        logger.exception("tool %s crashed", call.name)
        return _ended(
            call,
            started,
            result=f"the tool broke: {type(exc).__name__}: {exc}",
            outcome="error",
            error_code="tool_crashed",
        )
    return _ended(call, started, result=result)


__all__ = [
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
]
