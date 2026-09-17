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

`Verdict` is the user's, and nothing else writes it.

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
from saimc.spec import (
    SPEC_SCHEMA_VERSION,
    CompositionSpec,
    UnsupportedSpecVersionError,
)

SESSION_SCHEMA_VERSION: Final[int] = 1
"""Bump when a record in this module gains, loses or reshapes a field."""

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
    for one drafted from the brief alone. It is the one piece of the
    revision's provenance that does not depend on the delta vocabulary, so
    it is recorded now; the deltas that produced the draft land beside it in
    Phase D, with the types that express them.
    """

    draft_id: str
    created_at: datetime
    spec: CompositionSpec
    plan: CompositionPlan
    performance_plan_hash: str
    quality: PieceQuality
    lint: LintReport
    parent_id: str | None = None
    sketch: SketchRecord | None = None

    def __post_init__(self) -> None:
        require_id_segment(self.draft_id, label="draft_id")
        if self.parent_id is not None:
            require_id_segment(self.parent_id, label="parent_id")
        if not self.performance_plan_hash:
            raise ValueError("a draft records the digest of the performance it composed")

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
        spec_payload = payload["spec"]
        written = spec_payload.get("schema_version")
        if written is not None and int(written) > SPEC_SCHEMA_VERSION:
            raise UnsupportedSpecVersionError(
                f"draft {draft_id} carries a spec written with schema version {written}, "
                f"but this build understands up to version {SPEC_SCHEMA_VERSION}; upgrade "
                "saimc or delete the session directory."
            )
        try:
            plan = CompositionPlan.from_canonical_dict(payload["plan"])
        except PlanError as exc:
            raise UnsupportedPlanVersionError(
                f"draft {draft_id} carries a composition plan this build cannot read ({exc}); "
                "upgrade saimc or delete the session directory."
            ) from exc
        raw_sketch = payload.get("sketch")
        return cls(
            draft_id=draft_id,
            created_at=datetime.fromisoformat(payload["created_at"]),
            parent_id=payload.get("parent_id"),
            spec=CompositionSpec.model_validate(spec_payload),
            plan=plan,
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


@dataclass
class Session:
    """The persisted record for one interactive session.

    Mutable, like `Job`, and for the same reason: a session is appended to
    as the loop runs, and rebuilding it through a frozen `replace` on every
    turn would allocate the whole turn log to add one entry.

    `finalized_job_id` is the only state the session has. Everything else it
    knows is what happened, in order; whether the piece has been published
    is the one thing a reader cannot recover from the log, and it is
    explicit so that finalizing twice is a decision rather than an accident.
    """

    session_id: str
    created_at: datetime
    updated_at: datetime
    brief: str
    turns: list[Turn] = field(default_factory=list)
    drafts: list[Draft] = field(default_factory=list)
    verdicts: list[Verdict] = field(default_factory=list)
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
            "finalized_job_id": self.finalized_job_id,
            "turns": [turn.to_document() for turn in self.turns],
            "drafts": [draft.to_document() for draft in self.drafts],
            "verdicts": [verdict.to_document() for verdict in self.verdicts],
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
        return cls(
            session_id=payload["session_id"],
            created_at=datetime.fromisoformat(payload["created_at"]),
            updated_at=datetime.fromisoformat(payload["updated_at"]),
            brief=payload["brief"],
            turns=[Turn.from_document(turn) for turn in payload.get("turns", ())],
            drafts=[Draft.from_document(draft) for draft in payload.get("drafts", ())],
            verdicts=[Verdict.from_document(v) for v in payload.get("verdicts", ())],
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
