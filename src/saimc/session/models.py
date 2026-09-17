"""The session's records: a brief, its turns, its drafts, the user's verdicts.

A session is the interactive unit: a user's brief, the conductor's turns
over it, the candidate drafts those turns produced, what the user thought
of them, and — once the user decides — the job that renders the piece.

Only `Draft` carries music, and it does so by holding the pair `compose`
is a function of. It stores no notes. `compose(spec, plan=plan)` is
deterministic, so a draft's score is recovered by composing again, and
`performance_plan_hash` is what proves the recovery wrote the bytes the
user was shown. That is why a draft is a *record* and not a copy of the
music, and it is the mechanism behind the plan's "revision determinism"
claim.

`Turn` and `ToolInvocation` are the replay log. A `CompositionPlan` is
what makes a *piece* replayable; the turn log is what makes a *session*
one — a recorded turn names the tools it called and the arguments it
called them with, and the deterministic tools answer the same way twice.

`Verdict` is the user's, and nothing else writes it. `Preference` is that
verdict read against the draft it judged — one row per request the piece was
built from — and it is the one record here that is stored *because* the thing
it describes may not survive: a dataset wants it after the drafts it came from
have been pruned.

Every record knows its own document form (`to_document` / `from_document`),
the way `CompositionPlan` knows its canonical one. These documents are
**not** canonical in §6's sense — nothing hashes a session, and it is not
one of the artifacts the determinism contract is made about — so they are
named for what they are rather than borrowing a word with a stricter
meaning. The encoding is still `canonical_dumps`, which buys deterministic
bytes worth diffing and refuses a NaN that wandered into a measurement.

Every stored value has exactly one owner here. `Draft` does not store the
score hash beside the lint report that already carries it, and it does not
store the seed beside the spec that already carries it: two copies of one
value are two chances to disagree, and a record read back from disk is
exactly where that disagreement would go unnoticed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Final, Literal, TypeVar, cast

from saimc.compose.linter import LintCode, LintIssue, LintReport
from saimc.compose.plan import CompositionPlan, PlanError, UnsupportedPlanVersionError
from saimc.llm.base import LLMError
from saimc.quality import PieceQuality
from saimc.session.deltas import Delta, RequestSource, delta_from_dict, delta_to_dict
from saimc.spec import (
    SPEC_SCHEMA_VERSION,
    CompositionSpec,
    UnsupportedSpecVersionError,
)

SESSION_SCHEMA_VERSION: Final[int] = 6
"""Bump when a record in this module gains, loses or reshapes a field.

Moved to 2 when `Session` gained `spec`, to 3 when `Draft` gained `deltas`,
to 4 when it gained `requests_source`, to 5 when the session gained
`preferences`, and to 6 when a draft's scorecard gained
`harmony_pad_coverage`. Bumps 2 to 5 added fields with defaults, so an older
document would have loaded with the field silently missing; the sixth is
different in kind — `PieceQuality` takes no defaults and is read strictly —
and either way the load has to *refuse*, which is the reason the guard
compares the tag rather than tolerating what it recognises. A draft's lineage
read as empty is a draft that claims to have been drafted from the brief when
it was revised from another, its source read as absent is a request with no
record of who asked for it, a preference log read as absent is every
judgement the user has made thrown away, and a scorecard read as absent is a
piece whose harmony was never measured where null means it had none.
"""

SESSION_FORMAT_PREFIX: Final[str] = "Session"

SESSION_FORMAT: Final[str] = f"{SESSION_FORMAT_PREFIX}:{SESSION_SCHEMA_VERSION}"
"""The tag the stored session document carries.

Not a field of `Session`, unlike `NotationScore.format`: the tag describes
the shape of the *container* the records are written into, and a mutable
record that is never hashed gains nothing from carrying a constant every
instance sets identically. Each draft inside already carries its plan's own
tag, and each spec its schema version, so nothing is left unnamed.
"""

TurnTrigger = Literal["brief", "message", "auto"]
"""What started a turn: the brief, a user message, or the harness itself."""

ToolOutcome = Literal["ok", "refused", "error"]
"""How a tool call ended.

Three values rather than a boolean, because a refusal is not a failure. A
tool that declines — a delta the engine cannot honour — has *answered*, with
a reason the conductor can act on and the user can read; an error is a tool
that broke. Collapsing them would make the product's "refuse with a reason,
never no-op" rule indistinguishable from a crash in the one place a user
sees it.
"""

VerdictValue = Literal["like", "dislike"]

_TURN_TRIGGERS: Final[tuple[TurnTrigger, ...]] = ("brief", "message", "auto")
_REQUEST_SOURCES: Final[tuple[RequestSource, ...]] = ("typed", "conductor", "model", "keywords")
_TOOL_OUTCOMES: Final[tuple[ToolOutcome, ...]] = ("ok", "refused", "error")
_VERDICT_VALUES: Final[tuple[VerdictValue, ...]] = ("like", "dislike")

_V = TypeVar("_V", bound=str)


def _one_of(value: Any, allowed: tuple[_V, ...], *, field_name: str) -> _V:
    """Read a closed vocabulary out of a document, or refuse it.

    A `Literal`-typed field is a promise made by the *writer*; a document on
    disk is not the writer. Loading an unrecognised value into one would
    leave a record whose type claims something it does not honour, so an
    unknown value is refused where it is read — the same rule the plan and
    the spec version guards apply, one level down.
    """
    if value not in allowed:
        raise ValueError(f"{field_name} must be one of {allowed}, got {value!r}")
    return cast(_V, value)


def require_id_segment(value: str, *, label: str) -> None:
    """Refuse an id that is not a single relative path segment.

    Session and draft ids become directory names under the sessions root,
    and ids arrive from URL paths and from documents on disk. `root / value`
    with `value` of `".."` — or of `"a/b"` — names a path outside the root,
    which is a file the store has no business reading or deleting. Every id
    this store issues is one path segment, so anything else is refused
    rather than normalised: guessing what a malformed id meant is how a
    storage bug becomes a security bug.
    """
    if not value:
        raise ValueError(f"{label} must not be empty")
    if value in {".", ".."} or "/" in value or "\\" in value or "\0" in value:
        raise ValueError(f"{label} must be a single path segment, got {value!r}")


@dataclass(frozen=True)
class SketchRecord:
    """Where a draft's sketch was written, and how to verify it.

    Paths are relative to the session directory, for `jobs.storage`'s
    reason: the pair moves with the session and the recorded path still
    resolves. They are validated here, at construction and therefore also
    on load, so a document naming `../../etc/passwd` is refused when it is
    read rather than when it is served.

    A draft with no sketch is a real state, not a failure. A sketch costs a
    FluidSynth pass, a turn is budgeted in how many it may spend, and a
    candidate can be drafted, scored and compared before anyone listens to
    it.
    """

    wav_path: str
    ogg_path: str
    ogg_sha256: str
    ogg_size_bytes: int

    def __post_init__(self) -> None:
        for label, path in (("wav_path", self.wav_path), ("ogg_path", self.ogg_path)):
            if not path:
                raise ValueError(f"{label} must not be empty")
            if path.startswith("/") or path.startswith("\\"):
                raise ValueError(f"{label} must be relative to the session directory, got {path!r}")
            if ".." in path.split("/"):
                raise ValueError(
                    f"{label} must not climb out of the session directory, got {path!r}"
                )
        if not self.ogg_sha256:
            raise ValueError("a sketch records the digest of the audio it wrote")
        if self.ogg_size_bytes < 0:
            raise ValueError(f"ogg_size_bytes must not be negative, got {self.ogg_size_bytes}")

    def to_document(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_document(cls, payload: dict[str, Any]) -> SketchRecord:
        return cls(
            wav_path=payload["wav_path"],
            ogg_path=payload["ogg_path"],
            ogg_sha256=payload["ogg_sha256"],
            ogg_size_bytes=payload["ogg_size_bytes"],
        )


@dataclass(frozen=True)
class ToolInvocation:
    """One tool the conductor called, and what came back.

    `result` is the tool's answer as text, and it is also the text the model
    is fed back as the `tool` message — so the log records exactly what the
    next turn saw, and a replay re-derives nothing it could have read. When
    the result is structured, it is the JSON the tool rendered; when it is a
    refusal, it is the reason.

    `outcome` separates the three ways a call can end. A call that is not
    `ok` must name an `error_code`: the machine-readable reason code the
    conductor branches on, with the readable reason in `result`.
    """

    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    result: str = ""
    outcome: ToolOutcome = "ok"
    error_code: str | None = None
    duration_ms: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("a tool invocation must name the tool it called")
        if self.outcome == "ok" and self.error_code is not None:
            raise ValueError(
                f"{self.name} succeeded, so it cannot also carry error_code {self.error_code!r}"
            )
        if self.outcome != "ok" and not self.error_code:
            raise ValueError(
                f"a {self.outcome} call must name its reason: {self.name} has no error_code"
            )
        if self.duration_ms < 0:
            raise ValueError(f"duration_ms must not be negative, got {self.duration_ms}")

    @property
    def ok(self) -> bool:
        return self.outcome == "ok"

    def to_document(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "arguments": dict(self.arguments),
            "result": self.result,
            "outcome": self.outcome,
            "error_code": self.error_code,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_document(cls, payload: dict[str, Any]) -> ToolInvocation:
        return cls(
            name=payload["name"],
            arguments=dict(payload.get("arguments", {})),
            result=payload.get("result", ""),
            outcome=_one_of(payload.get("outcome", "ok"), _TOOL_OUTCOMES, field_name="outcome"),
            error_code=payload.get("error_code"),
            duration_ms=payload.get("duration_ms", 0),
        )


@dataclass(frozen=True)
class Turn:
    """One conductor step: what it said, and what it called.

    A turn is the unit of replay, so `calls` holds the tool calls in the
    order the model asked for them. A turn whose own model call failed has
    no calls — nothing ran — and carries `llm_error` instead; the two are
    mutually exclusive by construction, because a turn that claims both
    would describe a conversation that cannot have happened.

    A successful turn with neither narration nor calls is allowed and is not
    an error: a model that answers nothing when asked to act is a real
    outcome, and the honest thing is to record it and let the budget bound
    the loop, rather than to refuse the record of what happened.
    """

    created_at: datetime
    trigger: TurnTrigger
    narration: str = ""
    calls: tuple[ToolInvocation, ...] = ()
    model: str | None = None
    latency_ms: int = 0
    llm_error: LLMError | None = None

    def __post_init__(self) -> None:
        if self.latency_ms < 0:
            raise ValueError(f"latency_ms must not be negative, got {self.latency_ms}")
        if self.llm_error is not None and self.calls:
            raise ValueError(
                "a turn whose model call failed ran no tools, so it cannot also carry "
                f"{len(self.calls)} call(s)"
            )

    @property
    def failed(self) -> bool:
        """True when the conductor's own call failed and the turn did nothing."""
        return self.llm_error is not None

    def to_document(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at.isoformat(),
            "trigger": self.trigger,
            "narration": self.narration,
            "calls": [call.to_document() for call in self.calls],
            "model": self.model,
            "latency_ms": self.latency_ms,
            "llm_error": (
                None
                if self.llm_error is None
                else {"error_code": self.llm_error.error_code, "message": self.llm_error.message}
            ),
        }

    @classmethod
    def from_document(cls, payload: dict[str, Any]) -> Turn:
        raw_error = payload.get("llm_error")
        return cls(
            created_at=datetime.fromisoformat(payload["created_at"]),
            trigger=_one_of(payload["trigger"], _TURN_TRIGGERS, field_name="trigger"),
            narration=payload.get("narration", ""),
            calls=tuple(ToolInvocation.from_document(call) for call in payload.get("calls", ())),
            model=payload.get("model"),
            latency_ms=payload.get("latency_ms", 0),
            llm_error=(
                None
                if raw_error is None
                else LLMError(error_code=raw_error["error_code"], message=raw_error["message"])
            ),
        )


@dataclass(frozen=True)
class Draft:
    """One candidate piece: what produced it, and what it measured.

    `spec` and `plan` are the whole input to `compose`, and `compose` is
    deterministic in them — so the score is recovered by composing again
    rather than stored. `performance_plan_hash` is the check on that
    recovery: recomposing and hashing the result must reproduce it, or the
    draft does not describe the music the user heard.

    `plan` is materialized, for the reason `plan.py` gives: a plan that came
    back from a module table at read time would make this draft's
    reproducibility a claim about this build rather than about the draft.

    `parent_id` is the lineage — the draft a revision started from, `None`
    for one drafted from the brief alone. `deltas` is the rest of it: every
    request that has been applied to this line, in the order it was applied,
    with the ones an earlier revision carried and this one added both present.
    It is the *chain* rather than the step, and that is what makes a revision
    reproducible: `apply_deltas` folds these from the line's root spec, and the
    root's own spec is the spec of the draft at the end of the `parent_id` walk.
    A step would not do, because the plan is derived from the spec and the
    plan-writing half of a chain has to be folded from the root or a second
    revision quietly discards the first.

    `requests_source` is how *this* draft's own step was asked for — a studio
    control, the conductor's `revise` call, or feedback read with or without a
    model — and it is `None` on a draft drafted from the brief, which has no
    requests to place. One value for the chain rather than one per request: the
    requests of a single step are all read by the same reader, and a draft that
    recorded a source per delta would be storing the same string N times. It is
    what makes the preference log a log of preferences, since a like attached to
    a change a model chose and a like attached to one a user named are not the
    same datum.

    A chain with no source is `None` and is read tolerantly, like every other
    default-bearing field here: it means the step's reader was not recorded,
    which is what a document written before this field existed says. What the
    record refuses is the shape nothing can produce — a source for a step that
    has no requests — because a source is a claim *about* requests and there are
    none to be about. The complete-pair discipline belongs to the writer rather
    than to the constructor: `revise_draft` is the only thing that builds a
    revision, and it takes `source` as a required keyword.
    """

    draft_id: str
    created_at: datetime
    spec: CompositionSpec
    plan: CompositionPlan
    performance_plan_hash: str
    quality: PieceQuality
    lint: LintReport
    parent_id: str | None = None
    deltas: tuple[Delta, ...] = ()
    requests_source: RequestSource | None = None
    sketch: SketchRecord | None = None

    def __post_init__(self) -> None:
        require_id_segment(self.draft_id, label="draft_id")
        if self.parent_id is not None:
            require_id_segment(self.parent_id, label="parent_id")
        if not self.performance_plan_hash:
            raise ValueError("a draft records the digest of the performance it composed")
        if self.deltas and self.parent_id is None:
            raise ValueError(
                "a draft that carries deltas must name the draft they were applied to: "
                "deltas with no parent are a revision of nothing"
            )
        if self.requests_source is not None and not self.deltas:
            raise ValueError(
                "a draft records how its own requests were asked for, so a recorded source "
                "needs requests to be about: a draft with no chain has no step to place, and "
                "a source on one would be provenance for nothing"
            )

    @property
    def score_hash(self) -> str:
        """The composed score's canonical hash, read from the lint report.

        Not a stored field: the report already carries
        `NotationScore.compute_hash()`, and a second copy is a second value
        that can disagree with the notes it describes.
        """
        return self.lint.score_hash

    @property
    def plan_hash(self) -> str:
        """The composition plan's canonical hash — the arbiter's final tiebreak."""
        return self.plan.compute_hash()

    @property
    def lint_passed(self) -> bool:
        return self.lint.passed

    def to_document(self) -> dict[str, Any]:
        return {
            "draft_id": self.draft_id,
            "created_at": self.created_at.isoformat(),
            "parent_id": self.parent_id,
            "deltas": [delta_to_dict(delta) for delta in self.deltas],
            "requests_source": self.requests_source,
            "spec": self.spec.model_dump(mode="json"),
            "plan": self.plan.to_canonical_dict(),
            "performance_plan_hash": self.performance_plan_hash,
            "quality": asdict(self.quality),
            "lint": _lint_to_document(self.lint),
            "sketch": None if self.sketch is None else self.sketch.to_document(),
        }

    @classmethod
    def from_document(cls, payload: dict[str, Any]) -> Draft:
        draft_id = payload.get("draft_id", "?")
        raw_sketch = payload.get("sketch")
        raw_source = payload.get("requests_source")
        return cls(
            draft_id=draft_id,
            created_at=datetime.fromisoformat(payload["created_at"]),
            parent_id=payload.get("parent_id"),
            deltas=tuple(delta_from_dict(entry) for entry in payload.get("deltas", [])),
            requests_source=(
                None
                if raw_source is None
                else _one_of(raw_source, _REQUEST_SOURCES, field_name="requests_source")
            ),
            spec=_read_spec(payload["spec"], owner=f"draft {draft_id}"),
            plan=_read_plan(payload["plan"], owner=f"draft {draft_id}"),
            performance_plan_hash=payload["performance_plan_hash"],
            quality=PieceQuality(**payload["quality"]),
            lint=_lint_from_document(payload["lint"]),
            sketch=None if raw_sketch is None else SketchRecord.from_document(raw_sketch),
        )


@dataclass(frozen=True)
class Verdict:
    """The user's judgement of one draft.

    A like, a dislike, or words — and any combination. Requiring a like or a
    dislike would throw away the most useful feedback the product gets (the
    user explaining what they meant), and requiring feedback would throw
    away the fastest signal it gets. What is refused is a verdict with
    nothing in it, which is not a verdict.

    The user is the final judge of taste and this is where that is written
    down: no agent and no threshold writes one of these.
    """

    draft_id: str
    at: datetime
    value: VerdictValue | None = None
    feedback: str = ""

    def __post_init__(self) -> None:
        require_id_segment(self.draft_id, label="draft_id")
        if self.value is None and not self.feedback.strip():
            raise ValueError("a verdict must carry a like, a dislike, or feedback")

    def to_document(self) -> dict[str, Any]:
        return {
            "draft_id": self.draft_id,
            "at": self.at.isoformat(),
            "value": self.value,
            "feedback": self.feedback,
        }

    @classmethod
    def from_document(cls, payload: dict[str, Any]) -> Verdict:
        raw_value = payload.get("value")
        return cls(
            draft_id=payload["draft_id"],
            at=datetime.fromisoformat(payload["at"]),
            value=None
            if raw_value is None
            else _one_of(raw_value, _VERDICT_VALUES, field_name="verdict value"),
            feedback=payload.get("feedback", ""),
        )


@dataclass(frozen=True)
class Preference:
    """One accepted request, and what the listener thought of the piece it made.

    The row of a preference dataset: `(plan_hash, delta, verdict)` says what was
    asked for, in which piece, and how the result was received — and the last of
    those is the only part a threshold cannot supply. Nothing consumes it yet,
    and it is written now because the alternative is losing it: the chain it
    describes lives on drafts, and drafts are what a session prunes.

    `requests_source` is the *step's* reader rather than the judged draft's,
    because one chain can be built by two of them — a slider dragged and then a
    sentence typed — and a row labelling both with the last one would record a
    preference the user never expressed. `None` means the step's reader was not
    recorded, which is what it means on a draft and covers both a document
    written before that field existed and a chain this build cannot place.

    No `check()` requires the draft to exist, unlike a verdict: a row outliving
    the piece it judges is the point of writing it down.
    """

    draft_id: str
    at: datetime
    plan_hash: str
    delta: Delta
    verdict: VerdictValue
    requests_source: RequestSource | None = None

    def __post_init__(self) -> None:
        require_id_segment(self.draft_id, label="draft_id")

    def to_document(self) -> dict[str, Any]:
        return {
            "draft_id": self.draft_id,
            "at": self.at.isoformat(),
            "plan_hash": self.plan_hash,
            "delta": delta_to_dict(self.delta),
            "verdict": self.verdict,
            "requests_source": self.requests_source,
        }

    @classmethod
    def from_document(cls, payload: dict[str, Any]) -> Preference:
        raw_source = payload.get("requests_source")
        return cls(
            draft_id=payload["draft_id"],
            at=datetime.fromisoformat(payload["at"]),
            plan_hash=payload["plan_hash"],
            delta=delta_from_dict(payload["delta"]),
            verdict=_one_of(payload["verdict"], _VERDICT_VALUES, field_name="preference verdict"),
            requests_source=(
                None
                if raw_source is None
                else _one_of(raw_source, _REQUEST_SOURCES, field_name="requests_source")
            ),
        )


def _request_origins(
    session: Session, draft: Draft
) -> tuple[tuple[Delta, RequestSource | None], ...]:
    """Every request in `draft`'s chain, paired with the step that asked for it.

    A chain is its steps folded in order, so a request's step is the draft in
    the `parent_id` walk whose own `deltas` first held it. That is what makes
    the pairing a walk rather than a copy of something already stored — a draft
    stores its chain and its own step's reader, and nothing else — and it is the
    same walk `_lineage` makes in `tools.py`, in the one direction that module
    cannot provide: it imports this one.

    The walk is bounded by the drafts the session has and cycle-checked, because
    a lineage is read off a document rather than built in memory, and a document
    is not the writer. A record that does not line up — a parent whose chain is
    not a prefix of the child's, which only a hand-edited file produces — has
    the requests it cannot place paired with no source rather than with a guess:
    inventing one would write a preference the record does not have.

    A draft with no parent carries no deltas, so a piece drafted from the brief
    has nothing to log: liking it is a judgement about the spec, and there is no
    request to weigh. That is why the caller writes no row for one.
    """
    by_id = {known.draft_id: known for known in session.drafts}
    steps: list[tuple[tuple[Delta, ...], RequestSource | None]] = []
    current = draft
    seen: set[str] = set()
    while True:
        seen.add(current.draft_id)
        parent = None if current.parent_id is None else by_id.get(current.parent_id)
        if (
            parent is None
            or parent.draft_id in seen
            or current.deltas[: len(parent.deltas)] != parent.deltas
        ):
            steps.append((current.deltas, None))
            break
        steps.append((current.deltas[len(parent.deltas) :], current.requests_source))
        current = parent
    return tuple(
        (delta, source) for step_deltas, source in reversed(steps) for delta in step_deltas
    )


def _read_spec(payload: dict[str, Any], *, owner: str) -> CompositionSpec:
    """Read a stored spec, refusing one written by a newer build.

    Both the session and each of its drafts carry a spec, and both are read
    by a constructor on load, so the guard lives in one place rather than
    being written twice with two chances to drift. Unlike a plan, a spec is
    widened additively and completed by Pydantic, so only a *future*
    document is refused.
    """
    written = payload.get("schema_version")
    if written is not None and int(written) > SPEC_SCHEMA_VERSION:
        raise UnsupportedSpecVersionError(
            f"{owner} carries a spec written with schema version {written}, but this build "
            f"understands up to version {SPEC_SCHEMA_VERSION}; upgrade saimc or delete the "
            "session directory."
        )
    return CompositionSpec.model_validate(payload)


def _read_plan(payload: dict[str, Any], *, owner: str) -> CompositionPlan:
    """Read a stored plan, refusing one this build cannot read *exactly*.

    A plan has no additive completion — an older document is missing fields
    and a newer one may carry a knob the engine would silently ignore — so
    `from_canonical_dict` compares the format tag with `!=` and the refusal
    is re-raised here naming which draft or session held it.
    """
    try:
        return CompositionPlan.from_canonical_dict(payload)
    except PlanError as exc:
        raise UnsupportedPlanVersionError(
            f"{owner} carries a composition plan this build cannot read ({exc}); upgrade "
            "saimc or delete the session directory."
        ) from exc


@dataclass
class Session:
    """The persisted record for one interactive session.

    Mutable, like `Job`, and for the same reason: a session is appended to
    as the loop runs, and rebuilding it through a frozen `replace` on every
    turn would allocate the whole turn log to add one entry.

    `spec` is the parsed brief — the answer `parse_brief` gave, kept because
    every later turn is a proposal *about it* rather than about the words the
    user typed. Without it a session could not say what it is making, and a
    `draft` call would have to restate the whole spec in its arguments: a
    copy that can be mangled and that the recorded turn would then be a
    record of. It is `None` until the brief has been parsed.

    `finalized_job_id` is the only *state* the session has. Everything else
    it knows is what happened, in order; whether the piece has been
    published is the one thing a reader cannot recover from the log, and it
    is explicit so that finalizing twice is a decision rather than an
    accident.

    `preferences` is the one collection that is not simply a log of what
    happened: each row is a verdict read against the chain of the draft it
    judged, and it is stored rather than derived so that it survives the drafts
    it came from.
    """

    session_id: str
    created_at: datetime
    updated_at: datetime
    brief: str
    spec: CompositionSpec | None = None
    turns: list[Turn] = field(default_factory=list)
    drafts: list[Draft] = field(default_factory=list)
    verdicts: list[Verdict] = field(default_factory=list)
    preferences: list[Preference] = field(default_factory=list)
    finalized_job_id: str | None = None

    def __post_init__(self) -> None:
        require_id_segment(self.session_id, label="session_id")
        if not self.brief.strip():
            raise ValueError("a session needs a brief; there is nothing to conduct without one")
        if self.finalized_job_id is not None:
            require_id_segment(self.finalized_job_id, label="finalized_job_id")
        self.check()

    def check(self) -> None:
        """Re-check the consistency of the collections, on demand.

        Separate from `__post_init__` because this record is *mutable*: the
        lists a constructor validated can be appended to afterwards, and a
        session that gained a duplicate draft id or a verdict naming a draft
        that does not exist would be written to disk happily and then
        refused on load, because every read goes through this constructor.
        A store that never writes a document it could not read is worth one
        method.

        `preferences` is deliberately not checked against the drafts the way
        `verdicts` are. A verdict naming a draft this session does not have is a
        broken record, because the judgement is *about* something that should be
        here; a preference row is the same judgement kept after the pruning that
        would remove that something, so refusing the row for outliving its draft
        would refuse exactly what it is for.
        """
        draft_ids = [draft.draft_id for draft in self.drafts]
        duplicates = sorted({seen for seen in draft_ids if draft_ids.count(seen) > 1})
        if duplicates:
            raise ValueError(f"two drafts in one session share an id: {duplicates}")
        known = set(draft_ids)
        for verdict in self.verdicts:
            if verdict.draft_id not in known:
                raise ValueError(
                    f"verdict names draft {verdict.draft_id!r}, which this session does not have"
                )

    @property
    def is_finalized(self) -> bool:
        return self.finalized_job_id is not None

    def draft(self, draft_id: str) -> Draft:
        """The draft with `draft_id`. Raises `KeyError` if there is none."""
        for draft in self.drafts:
            if draft.draft_id == draft_id:
                return draft
        raise KeyError(draft_id)

    def record_verdict(self, verdict: Verdict) -> None:
        """Record the user's judgement, and the preference rows it writes.

        One owner for the pair, because the rows are the verdict read against
        the judged draft's chain and the two have to agree: a caller that
        appended one without the other would leave a log that cannot be rebuilt
        from what the session holds.

        A verdict carrying no like and no dislike writes no row. Words are the
        most useful thing the product gets and they are not by themselves a
        preference — nothing here knows whether they are praise — so they are
        recorded on the verdict for the reader that can tell, and the log keeps
        the judgements that are already unambiguous.

        The draft's own `plan_hash` is the piece every row names: a chain's
        requests are all changes to the one piece the user heard, and the
        requests are what tell one row from another.

        Raises `KeyError` for a draft this session does not have, like every
        other reader of a draft id — the verdict names something that has to be
        here.
        """
        draft = self.draft(verdict.draft_id)
        self.verdicts.append(verdict)
        if verdict.value is None:
            return
        self.preferences.extend(
            Preference(
                draft_id=draft.draft_id,
                at=verdict.at,
                plan_hash=draft.plan_hash,
                delta=delta,
                verdict=verdict.value,
                requests_source=source,
            )
            for delta, source in _request_origins(self, draft)
        )

    def replace_draft(self, draft: Draft) -> None:
        """Put `draft` back where the draft it replaces already sits.

        A draft is only ever replaced by a later version of itself — a
        sketch is attached, a plan is revised — so the position does not
        move. `Draft` is frozen, so this is how a draft gains a sketch, and
        appending instead would leave the session with two drafts of one id
        and a `check()` that refuses to save it. Raises `KeyError` for a
        draft this session does not have, which is the honest alternative to
        silently appending one.
        """
        for index, existing in enumerate(self.drafts):
            if existing.draft_id == draft.draft_id:
                self.drafts[index] = draft
                return
        raise KeyError(draft.draft_id)

    def to_document(self) -> dict[str, Any]:
        """Every record, in order, and nothing derived.

        The turn log, the drafts and the verdicts are arrays rather than
        objects keyed by id: order is meaning here — the turns are a
        sequence of decisions and the drafts were drafted in this order —
        and a document that sorts them by id would lose it.
        """
        return {
            "format": SESSION_FORMAT,
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "brief": self.brief,
            "spec": None if self.spec is None else self.spec.model_dump(mode="json"),
            "finalized_job_id": self.finalized_job_id,
            "turns": [turn.to_document() for turn in self.turns],
            "drafts": [draft.to_document() for draft in self.drafts],
            "verdicts": [verdict.to_document() for verdict in self.verdicts],
            "preferences": [row.to_document() for row in self.preferences],
        }

    @classmethod
    def from_document(cls, payload: dict[str, Any]) -> Session:
        written = str(payload.get("format", ""))
        if written != SESSION_FORMAT:
            raise UnsupportedSessionVersionError(
                f"session document names format {written!r}, but this build writes and "
                f"understands {SESSION_FORMAT!r}; a session's records are built by their "
                "constructors on load, so a document from another shape cannot be read "
                "faithfully. Upgrade saimc or delete the session directory."
            )
        raw_spec = payload.get("spec")
        return cls(
            session_id=payload["session_id"],
            created_at=datetime.fromisoformat(payload["created_at"]),
            updated_at=datetime.fromisoformat(payload["updated_at"]),
            brief=payload["brief"],
            spec=None if raw_spec is None else _read_spec(raw_spec, owner="this session"),
            turns=[Turn.from_document(turn) for turn in payload.get("turns", ())],
            drafts=[Draft.from_document(draft) for draft in payload.get("drafts", ())],
            verdicts=[Verdict.from_document(v) for v in payload.get("verdicts", ())],
            preferences=[Preference.from_document(p) for p in payload.get("preferences", ())],
            finalized_job_id=payload.get("finalized_job_id"),
        )


class UnsupportedSessionVersionError(Exception):
    """A stored session document this build cannot read.

    The container's own refusal, distinct from the ones a *record* raises
    (`UnsupportedSpecVersionError`, `UnsupportedPlanVersionError`): those
    name a value this build cannot complete, this one names the document
    itself. A session is read by rebuilding every record through its
    constructor, so a document whose shape moved would not be read
    faithfully by taking the keys this build recognises and dropping the
    rest — and a session that silently lost a turn is worse than one that
    refuses to load.
    """


def _lint_to_document(report: LintReport) -> dict[str, Any]:
    return {
        "score_hash": report.score_hash,
        "passed": report.passed,
        "issues": [
            {
                "code": str(issue.code),
                "message": issue.message,
                "measure_index": issue.measure_index,
                "tick": issue.tick,
                "voice_id": issue.voice_id,
                "pitch_midi": issue.pitch_midi,
            }
            for issue in report.issues
        ],
    }


def _lint_from_document(payload: dict[str, Any]) -> LintReport:
    """Rebuild a `LintReport`, codes included.

    The issues are kept whole rather than reduced to a pass/fail flag: the
    linter's findings are what a user is shown, and "the piece is illegal"
    with no reason is exactly the un-actionable answer the product avoids.
    """
    return LintReport(
        score_hash=payload["score_hash"],
        passed=payload["passed"],
        issues=tuple(
            LintIssue(
                code=LintCode(issue["code"]),
                message=issue["message"],
                measure_index=issue.get("measure_index"),
                tick=issue.get("tick"),
                voice_id=issue.get("voice_id"),
                pitch_midi=issue.get("pitch_midi"),
            )
            for issue in payload.get("issues", ())
        ),
    )


__all__ = [
    "SESSION_FORMAT",
    "SESSION_FORMAT_PREFIX",
    "SESSION_SCHEMA_VERSION",
    "Draft",
    "Preference",
    "Session",
    "SketchRecord",
    "ToolInvocation",
    "ToolOutcome",
    "Turn",
    "TurnTrigger",
    "UnsupportedSessionVersionError",
    "Verdict",
    "VerdictValue",
    "require_id_segment",
]
