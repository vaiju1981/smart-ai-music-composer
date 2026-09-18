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

What is deliberately *not* here: `apply_delta`, as a tool on its own. The
plan's tool table lists it beside `revise`, and they are the same act seen
from two sides — `revise` is the whole of it, from the draft the user is
looking at to the draft that answers them — so a second tool that folded
deltas without composing would let a model move the plan of a piece it never
heard. `revise` returns the refusal an `apply_delta` would have, in the same
words, which is what the design asks for.

`repair` is the other side of that line. It is not `apply_delta` with fewer
arguments: it chooses the requests itself, by measuring the worst bar a draft
misses against a table of what has been measured to move it, and keeps only
what improves the piece. A caller has no knob to name the bar or the request,
because a caller free to name them would be choosing by taste a thing the
measurement owns — and a conductor with an opinion has `revise` for saying so
in the user's own terms. Both tools end in the same applier, which is what
makes a repaired draft a draft like any other: the chain it records is the
chain the piece was made from.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Final

from saimc.compose.engine import CompositionEngineError, EngineOutput, compose
from saimc.compose.linter import lint
from saimc.jobs.storage import Job, JobStorage
from saimc.jobs.worker import QueueUnavailable, enqueue_or_fail
from saimc.llm.base import LLMClient, ToolCall, ToolSpec
from saimc.parser import parse_prompt
from saimc.quality import AXES, Axis, QualityFinding, localize, score_piece
from saimc.render.audio import render_sketch
from saimc.render.instruments import resolve_job_soundfont
from saimc.session.arbiter import deciding_element, rank, regression
from saimc.session.deltas import (
    DELTA_TYPES,
    Delta,
    DeltaRefusal,
    RequestSource,
    apply_deltas,
    delta_from_dict,
    delta_to_dict,
    refusal_line,
    refuse_uncarried,
    swallowed_tempo,
)
from saimc.session.models import (
    Draft,
    Publication,
    Session,
    SketchRecord,
    ToolInvocation,
    ToolOutcome,
)
from saimc.session.repairs import Attempt, Repair, repair_chain
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

MAX_DELTAS_PER_REVISION: Final[int] = 8
"""How many requests one `revise` call may carry.

The width of a revision, in the same sense `MAX_CANDIDATES_PER_DRAFT` is the
width of a fan-out, and it is read by the tool rather than counted in the
ledger for the same reason: folding deltas is arithmetic over a frozen
dataclass, and the turn's cost is the one composition that follows.
"""

MAX_REVISIONS_PER_LINE: Final[int] = 8
"""How many revisions one draft's line may accumulate.

The answer to the question the plan left open — how long a session stays
revisable — and it is also the bound on the walk to the line's root, because
that walk is how a revision finds the spec its chain folds onto. A bound is
needed for the second reason as much as the first: a hand-edited document can
name a parent that names a parent that names the first one, and an unbounded
walk would not return.
"""

MAX_REPAIRS_PER_TURN: Final[int] = 4
"""How many requests one `repair` call may keep.

A repair composes each candidate it tries to measure it, so this is the loop's
backstop in the same sense `MAX_CANDIDATES_PER_DRAFT` is the fan-out's: it
bounds a width, and it is read by the tool rather than counted in the ledger
because a compose is arithmetic. The worst case stays around a second.

Four is one round above the deepest chain the music needs, and the two numbers
come from different sweeps on purpose. Over the 90-piece corpus `repairs.py`
names, the widest chain a piece needed was two kept requests; over a wider grid
— three moods, seven durations, forty seeds, every piece pinned to C — the
deepest was **three**, at `electrifying/90s/31` and `electrifying/300s/25`. The
corpus is 90 pieces and the grid is 840, so the grid is the one that bounds the
default.

Both depth numbers were re-measured once Phase F4's leap floor moved the
melody, and the deepest chain fell from five to three: the grid's
`max_leap_semitones` breaches fell from 125 to 78 in the same pass, so the
leap bar needs working around in far fewer pieces and the longest surviving
chains are the bed's.

It is one *above* the deepest rather than equal to it because a backstop that
is reached is not a backstop: `repair_chain` states the same principle from
the other side ("`maximum` is the fourth and it is a backstop rather than the
usual end"), and at the depth the grid used to need, the pieces with the
longest chains came back with a bar still missed that one more move cleared.
"""


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
    max_deltas: int = MAX_DELTAS_PER_REVISION
    max_revisions: int = MAX_REVISIONS_PER_LINE
    max_repairs: int = MAX_REPAIRS_PER_TURN
    max_sketches: int = MAX_SKETCHES_PER_TURN
    max_llm_calls: int = MAX_LLM_CALLS_PER_TURN
    deadline_seconds: float = TURN_DEADLINE_SECONDS

    def __post_init__(self) -> None:
        for label, value in (
            ("max_candidates", self.max_candidates),
            ("max_deltas", self.max_deltas),
            ("max_revisions", self.max_revisions),
            ("max_repairs", self.max_repairs),
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


def _axis_arg(args: Mapping[str, Any]) -> Axis | None:
    """Read the `axis` argument, or `None` when the caller wants all of them.

    Checked against the closed vocabulary rather than accepted as a free
    string, because a critic asked for a part of the piece no metric
    measures has to be told so: an empty report for the percussion reads
    exactly like a clean piece, which is the wrong answer dressed as the
    right one. The membership check and the narrowing are one statement, so
    what a caller may ask for and what the loop returns cannot come apart.
    """
    asked = _str_arg(args, "axis")
    if asked is None:
        return None
    for known in AXES:
        if asked == known:
            return known
    raise ToolRefusal(
        "invalid_arguments",
        f"{asked!r} is not a part of a piece the scorecard measures; it measures "
        f"{', '.join(AXES)}.",
    )


def _named_draft(ctx: ToolContext, args: Mapping[str, Any]) -> Draft:
    """Resolve the `draft_id` argument against this session's drafts."""
    draft_id = _str_arg(args, "draft_id")
    if draft_id is None:
        raise ToolRefusal("invalid_arguments", "draft_id is required")
    return _draft_named(ctx, draft_id)


def _draft_named(ctx: ToolContext, draft_id: str) -> Draft:
    """The draft with this id, or a refusal naming the ids this session has.

    The list goes into the refusal: a model that guessed an id can correct
    itself in the next turn instead of asking again.
    """
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


def _recompose(draft: Draft) -> EngineOutput:
    """The draft's music, recomposed from what the draft recorded.

    A draft stores no notes — it stores the pair `compose` is a function of
    and reads the music back by composing again — so a tool that needs the
    notes has to do the same. Shared by `sketch` and `critique`, because the
    check below is what makes "this draft describes the music the user heard"
    a claim rather than a hope, and a second copy of it would be a second
    place for that claim to be made differently.
    """
    output = compose(draft.spec, plan=draft.plan)
    if output.performance_plan.compute_hash() != draft.performance_plan_hash:
        raise ToolFailure(
            "performance_mismatch",
            f"draft {draft.draft_id} recomposed to a different performance than the one it "
            "recorded, so the draft no longer describes the music it claims to.",
        )
    return output


def _with_seed(spec: CompositionSpec, seed: int) -> CompositionSpec:
    """The same spec at a different seed — validated, not copied blindly.

    `model_copy(update=...)` does not re-run a field's own `ge=0`, which is
    how a candidate could be composed at a seed the spec type would refuse.
    Round-tripping through `model_validate` puts the value back through the
    model that defines what the value may be, so a bad seed refuses by name.
    """
    return CompositionSpec.model_validate({**spec.model_dump(mode="json"), "seed": seed})


def _draft_from(
    draft_id: str,
    spec: CompositionSpec,
    output: EngineOutput,
    *,
    parent_id: str | None = None,
    deltas: tuple[Delta, ...] = (),
    requests_source: RequestSource | None = None,
) -> Draft:
    """One candidate, recorded from what the engine actually produced.

    The plan is read from the output rather than re-derived with
    `default_plan(spec)`. They are the same value for a draft composed from
    the brief, and for a revision they are not: the plan a revision folds is
    the one the engine composed under, and re-deriving would silently drop
    every plan request in the chain. Reading it off the output is the same
    distinction the oracle test makes.

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
        parent_id=parent_id,
        deltas=deltas,
        requests_source=requests_source,
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


def _axis_reports(
    grouped: Mapping[Axis, tuple[QualityFinding, ...]], *, only: Axis | None
) -> list[dict[str, Any]]:
    """One report per axis, in `AXES` order, the clean ones included.

    Every axis and not only the offending ones: a report that listed the
    axes which found something cannot be told from one whose other critics
    never ran, and "the melody is clean" is the thing a user asking about the
    melody wants to be told.
    """
    return [
        {
            "axis": name,
            "clean": not grouped[name],
            "findings": [asdict(finding) for finding in grouped[name]],
        }
        for name in AXES
        if only is None or name == only
    ]


async def _critique(ctx: ToolContext, args: Mapping[str, Any]) -> str:
    """Report every quality threshold one draft misses, and what moves it.

    Grouped by the part of the piece a remedy would move, because that is how
    a finding is read: the melody's critic answers about the tune, the
    accompaniment's about the bed under it, and the bars each finding names
    are the bars its own counted events sit in.
    """
    draft = _named_draft(ctx, args)
    axis = _axis_arg(args)
    grouped = draft.quality.findings_by_axis()
    if any(grouped.values()):
        # The bars are a reading of the notes, and a draft stores none — so
        # the score comes back the way `sketch` gets it, and only when there
        # is a miss to place. A clean draft costs no composition at all.
        score = _recompose(draft).notation_score
        grouped = {name: localize(score, findings) for name, findings in grouped.items()}
    return _render(
        {
            "draft_id": draft.draft_id,
            "lint_passed": draft.lint.passed,
            "lint_issues": [
                {"code": str(issue.code), "message": issue.message} for issue in draft.lint.issues
            ],
            "measured": draft.quality.as_dict(),
            "axes": _axis_reports(grouped, only=axis),
        }
    )


def _lineage(session: Session, draft: Draft, *, limit: int) -> tuple[Draft, int]:
    """The draft this one's line started from, and how many revisions deep it is.

    A revision folds its chain onto the *root's* spec rather than onto the spec
    of the draft being revised. It has to: the plan is derived from the spec, so
    the plan-writing half of a chain has to be folded from the base the chain
    was built on, and folding it onto a later spec would discard whatever the
    earlier revisions asked for. The root is where the chain starts, and a
    root's spec is the root spec because a root draft has no deltas.

    Bounded, and the bound is not decoration: `parent_id` is a string read off a
    document, so a hand-edited one can name a draft that names it back. `seen`
    catches that by name and `limit` catches a line longer than the budget, and
    neither is reachable through the tools.
    """
    seen = {draft.draft_id}
    walk, depth = draft, 0
    while walk.parent_id is not None:
        if depth >= limit:
            raise ToolRefusal(
                "revision_limit",
                f"{draft.draft_id} is more than {limit} revision(s) from the piece its line "
                "started as, which is as far as a line may be revised. Start a new draft "
                "from the brief, or publish what you have.",
            )
        parent_id = walk.parent_id
        if parent_id in seen:
            raise ToolRefusal(
                "broken_lineage",
                f"draft {walk.draft_id} and its ancestors name each other in a cycle, so "
                "there is no piece this line started from.",
            )
        seen.add(parent_id)
        try:
            walk = session.draft(parent_id)
        except KeyError as exc:
            raise ToolRefusal(
                "broken_lineage",
                f"draft {parent_id} is named as the parent of {walk.draft_id} and this "
                "session does not have it, so the chain cannot be folded.",
            ) from exc
        depth += 1
    return walk, depth


def parse_requests(raw: Any, *, maximum: int) -> tuple[Delta, ...]:
    """Read a list of requests out of a document nobody here wrote.

    Public because two callers read requests out of a body they did not author
    — the conductor's `revise` call, whose document is a tool-call argument, and
    `POST /sessions/{id}/deltas`, whose document is an HTTP body — and they have
    to refuse the same documents for the same reasons. The shape is the same
    `publish_draft` story at the other end of the session: one rule, one owner,
    two callers that report it differently.

    The distinction that matters is in here rather than at the call sites: a
    request the engine carries no knob for is answered from `UNCARRIED` rather
    than as a misspelling — which is what a user's words turn into, since "add a
    saxophone" is understood and unbuilt, and telling them it was not understood
    would be false. Everything else — a bad knob name, a value the knob's own
    type cannot hold, an argument the knob does not have — is refused by name
    with the vocabulary listed.

    The ceiling is the caller's because it is a policy rather than a shape: the
    tool's is the budget it advertises, and a second caller with a different one
    should not have to agree with it.
    """
    if not isinstance(raw, list) or not raw:
        raise ToolRefusal("invalid_arguments", "deltas must be a non-empty list of requests")
    if len(raw) > maximum:
        raise ToolRefusal(
            "invalid_arguments",
            f"at most {maximum} request(s) per revision, got {len(raw)}. Split them across "
            "two revisions if they are all worth making.",
        )
    parsed: list[Delta] = []
    for entry in raw:
        if not isinstance(entry, Mapping):
            raise ToolRefusal("invalid_arguments", f"each request must be an object, got {entry!r}")
        knob = entry.get("knob")
        unbuilt = refuse_uncarried(knob) if isinstance(knob, str) else None
        if unbuilt is not None:
            raise ToolRefusal(unbuilt.reason, unbuilt.message)
        try:
            parsed.append(delta_from_dict(entry))
        except (TypeError, ValueError) as exc:
            raise ToolRefusal("invalid_arguments", str(exc)) from exc
    return tuple(parsed)


def _deltas_arg(args: Mapping[str, Any], *, maximum: int) -> tuple[Delta, ...]:
    """The `revise` tool's own reading of its `deltas` argument."""
    return parse_requests(args.get("deltas"), maximum=maximum)


@dataclass(frozen=True)
class Revision:
    """A revision that was kept: the draft it made, and what it was made of.

    `applied` and `refused` are *this step's* requests and not the chain's. The
    child's own `deltas` is the chain, because that is what makes the piece
    reproducible; these two are the report of what just happened, which is what
    a caller renders. A bare `Draft` would lose them, and a caller that walked
    the chain itself to recover them would be re-deciding which requests were
    accepted — the answer this type exists to carry.
    """

    draft: Draft
    applied: tuple[Delta, ...]
    refused: tuple[DeltaRefusal, ...]


def revise_draft(
    ctx: ToolContext,
    parent: Draft,
    requests: Sequence[Delta],
    *,
    source: RequestSource,
) -> Revision:
    """Apply `requests` to `parent`'s line, and keep the result as a draft.

    The rule, once, for two callers that report it differently: the conductor's
    `revise` tool answers with text the model reads, and `POST /deltas` answers
    with a draft card. Both get the same child and the same refusals, which is
    why this is a function rather than two — `publish_draft`'s shape, at the
    other end of a session.

    `source` is how the requests were arrived at, and it is the caller's because
    only the caller knows: a conductor's tool call and a studio control both
    arrive as typed requests, and the difference between them is who chose them.
    It is recorded on the child, which is where a verdict will later be attached
    to it.

    The ratchet runs before anything is recorded: a revision that measures worse
    than the draft it came from is refused with the arbiter's own sentence, and
    the session keeps the draft it had. Equal is allowed — moving the music
    without moving a measurement is what "the same piece, differently" means.

    The child is composed at the parent's seed unless a request re-rolls it,
    which is what makes a revision recognisably the same piece: the material the
    seed decides survives the edit.

    The session is saved here rather than left to the caller, for the reason
    `publish_draft` saves: the answer to both callers is a draft id, and a caller
    that has the id will fetch the draft by it, so the record has to be durable
    at the moment it becomes true.
    """
    if not requests:
        raise ToolRefusal(
            "invalid_arguments",
            "a revision must carry at least one request; there is nothing to apply.",
        )
    root, depth = _lineage(ctx.session, parent, limit=ctx.budget.max_revisions)
    if depth >= ctx.budget.max_revisions:
        raise ToolRefusal(
            "revision_limit",
            f"{parent.draft_id} is {depth} revision(s) from {root.draft_id}, and a line may "
            f"be revised {ctx.budget.max_revisions} time(s). Start a new draft from the brief, "
            "or publish what you have.",
        )

    application = apply_deltas(root.spec, (*parent.deltas, *requests))
    # The prefix re-folds to exactly the state it was folded in — same spec,
    # same plan, same order — so a request the chain already accepted cannot
    # refuse now, and everything the applier refused is from *this* call.
    # Checked rather than assumed, because a draft whose own requests do not
    # replay is a draft describing a piece nothing else can reach.
    if application.applied[: len(parent.deltas)] != parent.deltas:
        raise ToolFailure(
            "lineage_mismatch",
            f"{parent.draft_id}'s recorded requests do not replay against the spec its line "
            "started from, so this revision cannot be applied to the piece it names.",
        )
    added = application.applied[len(parent.deltas) :]
    if not added:
        first = application.refused[0]
        raise ToolRefusal(first.reason, refusal_line(first))

    try:
        output = compose(application.spec, plan=application.plan)
    except CompositionEngineError as exc:
        raise ToolRefusal(str(exc.code), exc.message) from exc

    # A request the engine traded away is refused before anything else, because
    # the piece in hand is not the piece that was asked for: a tempo the length
    # outranked composes a perfectly good draft, and keeping it would answer
    # "make it faster" with a draft that is not. This is the only place the
    # question can be asked, since it needs the arrangement a composition made.
    swallowed = swallowed_tempo(added, spec=application.spec, arrangement=output.arrangement)
    if swallowed is not None:
        raise ToolRefusal(swallowed.reason, swallowed.message)

    child = _draft_from(
        f"draft-{len(ctx.session.drafts)}",
        application.spec,
        output,
        parent_id=parent.draft_id,
        deltas=application.applied,
        requests_source=source,
    )
    regressed = regression(parent, child)
    if regressed is not None:
        raise ToolRefusal("revision_regressed", regressed)

    ctx.session.drafts.append(child)
    ctx.sessions.save(ctx.session)
    return Revision(draft=child, applied=added, refused=application.refused)


async def _revise(ctx: ToolContext, args: Mapping[str, Any]) -> str:
    """The `revise` tool: apply the model's requests, and report what happened.

    The decision is `revise_draft`'s; what is left here is the reading of the
    argument and the rendering of the answer the model reads.
    """
    parent = _named_draft(ctx, args)
    requests = _deltas_arg(args, maximum=ctx.budget.max_deltas)
    revision = revise_draft(ctx, parent, requests, source="conductor")
    return _render(
        {
            "draft_id": revision.draft.draft_id,
            "parent_id": parent.draft_id,
            "seed": revision.draft.spec.seed,
            "lint_passed": revision.draft.lint.passed,
            "quality": revision.draft.quality.entry(),
            "applied": [delta_to_dict(delta) for delta in revision.applied],
            "refused": [
                {"reason": refusal.reason, "message": refusal.message, "nearest": refusal.nearest}
                for refusal in revision.refused
            ],
            "moved_on": deciding_element(revision.draft, parent),
        }
    )


def _no_repair(parent: Draft, outcome: Repair) -> str:
    """Why a repair kept nothing, in the words of the bar it aimed at.

    Three sentences, one per way the loop can come back empty, and they differ
    because the advice does. A clean draft has nothing to fix. A bar with no
    entry in the repair table is a bar *nothing was tried* on, which is a
    different answer from a bar whose requests were all tried and none kept —
    the first is a gap in the vocabulary, the second is a measurement — and
    conflating them would have the product claim a request does not exist while
    holding the attempts that prove it does.

    Neither sentence says "no request this build carries moves it". A request
    can clear the bar it was aimed at and still leave the piece no better
    overall, and a sentence denying it would be false about exactly the case the
    recorded attempts describe. What both end with instead is the finding's own
    hint, quoted as the finding's — which is where the maintainer's advice lives
    — rather than as this function's conclusion.
    """
    if outcome.stopped_by is None:
        return f"{parent.draft_id} misses no bar on the scorecard, so there is nothing to repair."
    worst = outcome.stopped_by
    requests = [
        attempt.delta
        for attempt in outcome.attempts
        if attempt.metric == worst.metric and attempt.delta is not None
    ]
    if not requests:
        return (
            f"{worst.metric} is the bar {parent.draft_id} misses most, and nothing was tried for "
            f"it: the repair table holds no request for that bar. The bar's own hint names what "
            f"would move it: {worst.hint}"
        )
    tried = ", ".join(request.describe() for request in requests)
    return (
        f"{worst.metric} is the bar {parent.draft_id} misses most, and the requests this build "
        f"carries for it ({tried}) were all tried and none was kept, so the draft stands as it "
        f"is. The bar's own hint names what would move it: {worst.hint}"
    )


def _attempt_entry(attempt: Attempt) -> dict[str, Any]:
    """One trial, as the model and the disclosure panel read it."""
    return {
        "metric": attempt.metric,
        "request": delta_to_dict(attempt.delta) if attempt.delta is not None else None,
        "outcome": attempt.outcome,
        "remaining": list(attempt.remaining),
    }


async def _repair(ctx: ToolContext, args: Mapping[str, Any]) -> str:
    """The `repair` tool: measure the worst bar, and keep what actually moves it.

    No model call and no audio. The loop composes candidates to measure them,
    which is arithmetic — about ten milliseconds each — so this costs a turn
    nothing it has to budget for and can be called whenever a draft is close.
    """
    parent = _named_draft(ctx, args)
    root, _ = _lineage(ctx.session, parent, limit=ctx.budget.max_revisions)
    outcome = repair_chain(
        root.spec,
        parent.deltas,
        parent.quality,
        maximum=ctx.budget.max_repairs,
    )
    if not outcome.added:
        raise ToolRefusal("no_repair", _no_repair(parent, outcome))
    revision = revise_draft(ctx, parent, outcome.added, source="repair")
    return _render(
        {
            "draft_id": revision.draft.draft_id,
            "parent_id": parent.draft_id,
            "seed": revision.draft.spec.seed,
            "lint_passed": revision.draft.lint.passed,
            "quality": revision.draft.quality.entry(),
            "applied": [delta_to_dict(delta) for delta in revision.applied],
            "moved_on": deciding_element(revision.draft, parent),
            "attempts": [_attempt_entry(attempt) for attempt in outcome.attempts],
            "remaining": list(outcome.remaining),
        }
    )


async def _compare(ctx: ToolContext, args: Mapping[str, Any]) -> str:
    """Rank drafts against one another, and name what decided each step.

    The order is the arbiter's, and the reason is the first element of it on
    which a draft lost — the same element `sorted` read, so the sentence and the
    ranking cannot disagree. `critique` says what one draft measures; this says
    which of several is ahead and why, which is the question a fan-out leaves
    the conductor holding.
    """
    raw = args.get("draft_ids")
    if raw is None:
        drafts = list(ctx.session.drafts)
    elif not isinstance(raw, list) or not raw:
        raise ToolRefusal(
            "invalid_arguments", "draft_ids must be a non-empty list of ids, or omitted"
        )
    else:
        ids: list[str] = []
        for item in raw:
            if not isinstance(item, str) or not item.strip():
                raise ToolRefusal(
                    "invalid_arguments", f"each draft_id must be a string, got {item!r}"
                )
            if item in ids:
                raise ToolRefusal(
                    "invalid_arguments",
                    f"{item!r} is asked for twice, and a draft ranked against itself says "
                    "nothing. Ask for distinct drafts.",
                )
            ids.append(item)
        drafts = [_draft_named(ctx, draft_id) for draft_id in ids]

    if not drafts:
        raise ToolRefusal(
            "no_drafts",
            "this session has no drafts yet, so there is nothing to rank. Call draft first.",
        )

    ranked = rank(drafts)
    entries: list[dict[str, Any]] = []
    for index, draft in enumerate(ranked):
        entry: dict[str, Any] = {
            "rank": index + 1,
            "draft_id": draft.draft_id,
            "parent_id": draft.parent_id,
            "seed": draft.spec.seed,
            "lint_passed": draft.lint.passed,
            "quality": draft.quality.entry(),
        }
        if index:
            leader = ranked[index - 1]
            entry["behind"] = leader.draft_id
            entry["on"] = deciding_element(leader, draft)
        entries.append(entry)
    return _render({"ranking": entries})


async def _sketch(ctx: ToolContext, args: Mapping[str, Any]) -> str:
    """Render one draft's audio, fast enough to sit inside a turn."""
    draft = _named_draft(ctx, args)
    if ctx.ledger.sketches >= ctx.budget.max_sketches:
        raise ToolRefusal(
            "budget_exhausted",
            f"this turn has already spent its {ctx.budget.max_sketches} sketch(es), and each "
            "one renders audio. A new turn is what buys another.",
        )

    output = _recompose(draft)

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

    ctx.session.publication = Publication(job_id=job.job_id, draft_id=draft.draft_id)
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
        "axis": {
            "type": "string",
            "enum": list(AXES),
            "description": (
                "Which part of the piece to report on: `melody` is the tune's own line, "
                "`accompaniment` the bed and the bass's relation to the tune's register, "
                "and `bass` the figure the bass line plays. Omit to hear from all three."
            ),
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


_COMPARE_PARAMETERS: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "draft_ids": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 1,
            "description": (
                "The drafts to rank against each other, by id. Omit to rank every draft the "
                "session has, which is what a fan-out wants."
            ),
        },
    },
    "additionalProperties": False,
}


_REPAIR_PARAMETERS: Final[dict[str, Any]] = {
    "type": "object",
    "properties": {
        "draft_id": {
            "type": "string",
            "description": (
                "The id of the draft to repair, exactly as `draft`, `revise` or a previous "
                "`repair` returned it."
            ),
        },
    },
    "required": ["draft_id"],
    "additionalProperties": False,
}


def request_schema() -> dict[str, Any]:
    """One typed request, as the JSON Schema a model authors against.

    The `enum` is `DELTA_TYPES`, so a knob a model is offered is a knob this
    build carries — a hand-written list here would be a second copy of the
    vocabulary, and the copy in a schema is the one that drifts. Only the knob
    is described: a request's own arguments are checked by its own constructor
    (`_deltas_arg`), which is the engine's business and not the schema's, so
    restating a bound here would be a third place it is written down.

    Public because two callers offer this vocabulary to a model and they must
    offer the same one: `revise` as a list of requests, and the feedback
    translator as one call per request.
    """
    return {
        "type": "object",
        "properties": {
            "knob": {
                "type": "string",
                "enum": sorted(DELTA_TYPES),
                "description": "Which request this is. Its other keys are that "
                "request's own arguments.",
            },
        },
        "required": ["knob"],
    }


def _revise_parameters(budget: ToolBudget) -> dict[str, Any]:
    """The `revise` schema, with the vocabulary and the width read off the code.

    Two indirections for the same reason `_draft_parameters` has one: the
    vocabulary comes from `request_schema`, and the ceiling is the one
    `_deltas_arg` reads off the budget it is handed, so the number the model is
    told cannot be a different number from the one it is held to.
    """
    return {
        "type": "object",
        "properties": {
            "draft_id": {
                "type": "string",
                "description": (
                    "The id of the draft to revise, exactly as `draft` or `revise` returned it."
                ),
            },
            "deltas": {
                "type": "array",
                "minItems": 1,
                "maxItems": budget.max_deltas,
                "items": request_schema(),
                "description": (
                    "The requests to apply, in order. Each is one knob and the value to set "
                    "it to — `{knob: SetTempo, tempo_bpm: 76}` takes the piece to 76 BPM. The "
                    "engine owns every value's legal range, so a request it cannot honour "
                    "comes back refused with the range and the nearest legal value, and the "
                    "rest of the requests still apply."
                ),
            },
        },
        "required": ["draft_id", "deltas"],
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
            "every bar it misses, one report per part of the piece: the metric, what it "
            "measured, the target, why the target exists, the hint naming what would move "
            "it, and the bars the offending events sit in. Ask for one part by name or "
            "take all three; a part with nothing to say is reported clean rather than "
            "left out. These are arithmetic over the score rather than an opinion, so "
            "they cost nothing and never vary."
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
    "revise": Tool(
        name="revise",
        description=(
            "Change a draft instead of redrafting it: apply typed requests to the piece it "
            "already is and keep the result as a new draft, composed at the same seed so it "
            "is recognisably the same piece. This is how feedback becomes music — the user's "
            "sentence turned into requests, never into a new prompt. A request the engine "
            "cannot honour is refused by name with the nearest legal value, and the rest "
            "still apply. A revision that would measure worse than the draft it came from is "
            "refused; keep the draft and ask for something else."
        ),
        parameters=_revise_parameters,
        handler=_revise,
    ),
    "repair": Tool(
        name="repair",
        description=(
            "Repair a draft's worst measured bar: read the highest bar it misses, measure "
            "the requests this build has for that bar, keep only the one that measurably "
            "improves the piece, and go again until nothing measures better. No model call "
            "and no audio — a few compositions of arithmetic. Use it when a draft is close "
            "and the misses are mechanical rather than a matter of taste; a repair changes "
            "how the material is placed and never what the user asked for, and it refuses "
            "with the bar's own hint when no request this build carries moves it."
        ),
        parameters=_static(_REPAIR_PARAMETERS),
        handler=_repair,
    ),
    "compare": Tool(
        name="compare",
        description=(
            "Rank drafts against each other by one written-down order — legality first, then "
            "how many quality bars a draft misses, how far past them it went, which bar was "
            "the highest, and finally its plan's hash and its seed — and say which of those "
            "decided each step. Use it to tell the user which candidate leads and why, "
            "rather than reading the numbers out."
        ),
        parameters=_static(_COMPARE_PARAMETERS),
        handler=_compare,
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
    "MAX_DELTAS_PER_REVISION",
    "MAX_LLM_CALLS_PER_TURN",
    "MAX_REPAIRS_PER_TURN",
    "MAX_REVISIONS_PER_LINE",
    "MAX_SKETCHES_PER_TURN",
    "TOOLS",
    "TURN_DEADLINE_SECONDS",
    "Revision",
    "Tool",
    "ToolBudget",
    "ToolContext",
    "ToolError",
    "ToolFailure",
    "ToolRefusal",
    "TurnLedger",
    "dispatch",
    "parse_requests",
    "publish_draft",
    "request_schema",
    "revise_draft",
    "tool_specs",
]
