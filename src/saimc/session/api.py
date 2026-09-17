"""The session's HTTP surface: open, look, speak, judge, publish, rewind.

The studio is one page over a session, and this module is what it talks to. It
carries no rules of its own — every rule it enforces is a rule something else
already enforces, translated into a status code. What a verdict may be is
`Verdict.__post_init__`'s; what may be published, and how often, is
`publish_draft`'s; what may be undone is `SessionStorage.undo`'s. A refusal that
originates in one of those arrives here as a code to look up, not as a reason to
re-decide.

Two things *are* this module's own.

**The turn runs here**, because the API is what knows a request arrived.
`take_turn` is the conductor's; `conduct` is the harness's — one turn, then up
to `MAX_AUTO_TURNS` continuations, then a last decision about publishing. It is
synchronous inside the request rather than queued, and that is deliberate: the
alternative has to answer before it knows whether a piece was published, and
"was one published" is the first thing the caller wants. The price is that a
turn's model calls are the request's latency, and the return is that a caller
who has the answer already has the whole result — there is no second thing to
go and fetch.

**The refusal codes are translated here**, because a status code is a fact about
the product rather than about the engine: a session that has already published
is a 409 whether the publish came from a tool call or from a button.

Two endpoints the plan named are deliberately absent, and both for the same
reason — a turn runs inline, so there is nothing in flight for either to be
*about*:

- **`/deltas`.** A delta's vocabulary is Phase D's deliverable. An endpoint whose
  body is a placeholder is worse than one that does not exist yet.
- **`/events`.** A change stream is worth its keep when something happens
  between requests. Nothing does: a turn's state changes at the moment its own
  request returns, so `GET /sessions/{id}` carries exactly the information a
  stream would push, and a second window can poll it. It would also be the first
  endpoint here that no test can exercise — it has no terminal state, so its body
  is buffered forever rather than read (`TestClient` and `httpx.ASGITransport`
  both wait for a response to complete), and a route whose behaviour nothing can
  witness is the liability this phase has already refused four times. It arrives
  with the thing that would give it an ending: a turn that runs somewhere other
  than inside the request that asked for it.

The job-storage lookup below is written out again rather than imported from
`saimc.jobs.api`, which has the same one: that module imports *this* one for its
`POST /jobs` alias, so the two cannot share an import in the other direction.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any, Final

from fastapi import APIRouter, HTTPException, Request, status
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from saimc.jobs.storage import JobStorage
from saimc.llm.base import ChatClient, LLMClient
from saimc.session.conductor import digest, take_turn
from saimc.session.models import (
    Draft,
    Session,
    Turn,
    TurnTrigger,
    Verdict,
    VerdictValue,
)
from saimc.session.store import SessionStorage, UndoUnavailable
from saimc.session.tools import ToolContext, ToolRefusal, publish_draft

router = APIRouter()

DEFAULT_AUTO_FINALIZE: Final[bool] = True
"""Whether a drafting pass ends by starting the full render.

On by default, per the decision that a passive user still gets a finished
piece: with it on, one prompt is still one mastered piece, and the deliberate
mode is one request field away rather than absent.

A request field rather than a field of `Session`, and that is a decision with a
reason. A session records what *happened* — its turns, its drafts, the job it
published. This is what the caller wants to happen next. Stored on the session
it would be a preference sitting inside the log, restored by `undo` along with
the rest of an earlier document, and eventually disagreeing with whatever the
workspace is showing the user right now.
"""

MAX_AUTO_TURNS: Final[int] = 2
"""How many continuations follow a turn that left a decision to make.

A pass that ends with several candidates and no publish is the ordinary case
for an open brief, and the answer is to ask the conductor again —
`trigger="auto"`, which is what that value has always meant — rather than for
the harness to choose on its behalf. Two is one to read the digest of what it
just drafted and one to act on it. The bound is in code because a loop the
model can extend is a turn whose length is the model's.
"""

_REFUSAL_STATUS: Final[Mapping[str, int]] = {
    "already_finalized": status.HTTP_409_CONFLICT,
    "queue_unavailable": status.HTTP_503_SERVICE_UNAVAILABLE,
}
"""The refusals `publish_draft` raises, as the status codes they become.

Only the two the harness itself can raise are here. A refusal a *tool* raised
is already recorded in the turn log and read from `turns`, so it never reaches
this table.
"""

_SKETCH_FILES: Final[Mapping[str, tuple[str, str]]] = {
    "ogg": ("ogg_path", "audio/ogg"),
    "wav": ("wav_path", "audio/wav"),
}
"""The two files a sketch writes, by the name the route uses for each.

The attribute name is spelled out rather than composed as `f"{kind}_path"`, so
a renamed field on `SketchRecord` is a loud failure at the one place it is read
instead of a silent miss on a name that happens to exist.
"""


class ToolCallResponse(BaseModel):
    """One recorded tool call, as the log holds it."""

    name: str
    arguments: dict[str, Any]
    result: str
    outcome: str
    error_code: str | None
    duration_ms: int


class TurnResponse(BaseModel):
    """One conductor step: what it said, and what it called."""

    created_at: str
    trigger: str
    narration: str
    calls: list[ToolCallResponse]
    model: str | None
    latency_ms: int
    llm_error: dict[str, str] | None


class VerdictResponse(BaseModel):
    """What the user thought of one draft."""

    draft_id: str
    at: str
    value: str | None
    feedback: str


class SketchResponse(BaseModel):
    """How to hear one draft, and how to check it is the audio that was rendered."""

    url: str
    ogg_sha256: str
    ogg_size_bytes: int


class DraftResponse(BaseModel):
    """One draft card: what it is, what it measured, how to play it.

    The findings are sent beside the raw measurements because they are what a
    next decision is about — the critic's read, which is arithmetic over the
    score rather than an opinion — and the measurements are what the numbers
    behind that read are.
    """

    draft_id: str
    created_at: str
    parent_id: str | None
    seed: int | None
    score_hash: str
    plan_hash: str
    performance_plan_hash: str
    lint_passed: bool
    quality: dict[str, Any]
    findings: list[dict[str, Any]]
    sketch: SketchResponse | None


class SessionResponse(BaseModel):
    """A whole session, plus the digest that stands behind its next turn.

    The digest is in the payload on purpose. The orchestration is meant to be
    visible, and the honest way to show a user what the conductor is told is to
    send the text it is told rather than a second rendering of the session that
    can disagree with it. This is the digest as it stands *now* — the one the
    next turn will be given — which is not the one the last turn was given.
    """

    session_id: str
    created_at: str
    updated_at: str
    brief: str
    spec: dict[str, Any] | None
    finalized_job_id: str | None
    digest: str
    turns: list[TurnResponse]
    drafts: list[DraftResponse]
    verdicts: list[VerdictResponse]


class CreateSessionRequest(BaseModel):
    """Body for `POST /sessions`."""

    model_config = ConfigDict(extra="forbid")

    brief: str = Field(min_length=1, max_length=4096)
    auto_finalize: bool = DEFAULT_AUTO_FINALIZE


class MessageRequest(BaseModel):
    """Body for `POST /sessions/{id}/message`."""

    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=4096)
    auto_finalize: bool = DEFAULT_AUTO_FINALIZE


class VerdictRequest(BaseModel):
    """Body for `POST /sessions/{id}/verdict`.

    Any combination of a like, a dislike and words is a verdict; none of them is
    not, and the record refuses that shape rather than this request model, so
    the rule has one owner.
    """

    model_config = ConfigDict(extra="forbid")

    draft_id: str
    value: VerdictValue | None = None
    feedback: str = ""


class FinalizeRequest(BaseModel):
    """Body for `POST /sessions/{id}/finalize`."""

    model_config = ConfigDict(extra="forbid")

    draft_id: str


class FinalizeResponse(BaseModel):
    """The job the session published. Read it at `GET /jobs/{job_id}`."""

    job_id: str
    draft_id: str
    state: str


def _storage(request: Request) -> SessionStorage:
    storage: SessionStorage | None = getattr(request.app.state, "session_storage", None)
    if storage is None:
        raise HTTPException(status_code=500, detail="Session storage not configured.")
    return storage


def _jobs(request: Request) -> JobStorage:
    storage: JobStorage | None = getattr(request.app.state, "job_storage", None)
    if storage is None:
        raise HTTPException(status_code=500, detail="Job storage not configured.")
    return storage


def _session(request: Request, session_id: str) -> Session:
    """Load a session, telling "no such session" apart from "not an id".

    The store raises `KeyError` for the first and `ValueError` for the second,
    and they are different answers: one is a 404 the caller reaches by following
    a stale link, the other a 400 they reach by sending something that was never
    a session id. Collapsing them would tell a user with a typo to stop looking
    for a session that exists.
    """
    try:
        return _storage(request).get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Session not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"not a session id: {exc}") from exc


def _model(request: Request) -> ChatClient:
    """The model this app was built with, or a refusal naming what is missing.

    A session *needs* one: the conductor is what decides what happens next, so a
    session with no model would be a session with nothing in it. So the refusal
    comes before anything is written — a 503 on the way in rather than an empty
    session the caller has to notice and clean up.
    """
    client: ChatClient | None = getattr(request.app.state, "session_llm", None)
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "no language model is configured, so the conductor cannot take a turn. "
                "Set OLLAMA_BASE_URL and OLLAMA_MODEL, or add an [llm] section to "
                "saimc.toml."
            ),
        )
    return client


def _parser(client: ChatClient) -> LLMClient | None:
    """The model as a parser, when it is one.

    `parse_brief` reads an `LLMClient` and the conductor a `ChatClient`; in every
    build that has a model at all they are one object (C1: two protocols, one
    connection), and `isinstance` is how that is asked rather than assumed. A
    client that is only a `ChatClient` gets a working loop whose `parse_brief`
    refuses `llm_not_configured` — a named answer recorded in the turn, which is
    the honest degradation and not a crash.
    """
    return client if isinstance(client, LLMClient) else None


def http_from_refusal(exc: ToolRefusal, session: Session) -> HTTPException:
    """The status code for a refusal the *harness* raised, and which session it was about.

    Named publicly because `saimc.jobs.api`'s `POST /jobs` alias runs the same
    pass and has to translate the same refusals.

    Two codes are reachable this way — a session that has already published, and
    a broker that would not take the job — and both are the caller's situation
    rather than a server fault. A code with no mapping is a wiring mistake, so it
    becomes a 500 carrying the refusal's own words rather than being guessed into
    a 4xx that would tell the caller to change something they cannot. The session
    id is in every message because in both cases the work is not lost: the
    caller's next move is another request against that session.
    """
    code = _REFUSAL_STATUS.get(exc.error_code)
    if code is None:
        return HTTPException(
            status_code=500,
            detail=f"{exc.error_code}: {exc.message} (session {session.session_id})",
        )
    return HTTPException(status_code=code, detail=f"{exc.message} (session {session.session_id})")


def _lone_draft(session: Session) -> Draft | None:
    """The session's only candidate, or `None` — never the best of several."""
    return session.drafts[0] if len(session.drafts) == 1 else None


def _already_published(session: Session) -> HTTPException:
    """409, with the tool's own advice, for a session that has published.

    The same rule `publish_draft` enforces, refused earlier: a turn on a session
    that has published could only ever produce candidates this session cannot
    master, so accepting it would hand the user work with no way to finish it.
    """
    return HTTPException(
        status_code=status.HTTP_409_CONFLICT,
        detail=(
            f"this session published job {session.finalized_job_id}, and publishing is final; "
            "start a new session for a different piece"
        ),
    )


async def conduct(
    sessions: SessionStorage,
    jobs: JobStorage,
    client: ChatClient,
    session: Session,
    *,
    trigger: TurnTrigger,
    message: str = "",
    auto_finalize: bool,
) -> None:
    """Take one turn, and take the piece to a render if the setting asks.

    The turn is the conductor's (`take_turn`). Everything after it is the
    harness's, and it is three decisions:

    1. **Continue while there is something to decide.** A pass that published
       nothing gets up to `MAX_AUTO_TURNS` more turns with `trigger="auto"`,
       which carries the nudge to publish. The loop stops early in the two cases
       where a continuation has nothing to answer: a turn whose own model call
       failed (there is nothing to ask when the ask cannot be delivered), and a
       session left holding exactly one draft — the harness will publish it, so
       there is no choice to make and no reason to spend a model call saying so.
    2. **A lone candidate is published.** One draft is not a choice, it is the
       only thing the session made, and mastering it is what keeps "one prompt,
       one finished piece" true for a brief specific enough to get a single
       draft. *Several* drafts are a choice, and choosing between candidates is
       the arbiter's job — Phase D's — so the harness does not guess: it leaves
       the user with the candidates and a Finalize button.
    3. **Nothing is invented.** With no drafts, or with a model that never
       spoke, the session is left exactly as it is. The caller can read the
       failed turn and ask again.
    """
    ctx = ToolContext(session=session, sessions=sessions, jobs=jobs, llm=_parser(client))
    await take_turn(ctx, client, trigger=trigger, message=message)
    if not auto_finalize or session.is_finalized:
        return

    for _ in range(MAX_AUTO_TURNS):
        if session.is_finalized or session.turns[-1].failed or _lone_draft(session) is not None:
            break
        await take_turn(ctx, client, trigger="auto")

    if session.is_finalized:
        return
    draft = _lone_draft(session)
    if draft is not None:
        # The only refusal reachable here is a broker outage, and it propagates:
        # the session has *not* published, and the caller's 503 is what says so.
        publish_draft(ctx, draft)


def _serialize_session(session: Session) -> SessionResponse:
    return SessionResponse(
        session_id=session.session_id,
        created_at=session.created_at.isoformat(),
        updated_at=session.updated_at.isoformat(),
        brief=session.brief,
        spec=None if session.spec is None else session.spec.model_dump(mode="json"),
        finalized_job_id=session.finalized_job_id,
        digest=digest(session),
        turns=[_serialize_turn(turn) for turn in session.turns],
        drafts=[_serialize_draft(session.session_id, draft) for draft in session.drafts],
        verdicts=[VerdictResponse(**verdict.to_document()) for verdict in session.verdicts],
    )


def _serialize_turn(turn: Turn) -> TurnResponse:
    """A turn, rendered from the record's own document form.

    `ToolCallResponse(**call.to_document())` rather than a field-by-field copy:
    the log's document form already exists, and a second rendering is a second
    place a field can be forgotten.
    """
    return TurnResponse(
        created_at=turn.created_at.isoformat(),
        trigger=turn.trigger,
        narration=turn.narration,
        calls=[ToolCallResponse(**call.to_document()) for call in turn.calls],
        model=turn.model,
        latency_ms=turn.latency_ms,
        llm_error=(
            None
            if turn.llm_error is None
            else {"error_code": turn.llm_error.error_code, "message": turn.llm_error.message}
        ),
    )


def _serialize_draft(session_id: str, draft: Draft) -> DraftResponse:
    return DraftResponse(
        draft_id=draft.draft_id,
        created_at=draft.created_at.isoformat(),
        parent_id=draft.parent_id,
        seed=draft.spec.seed,
        score_hash=draft.score_hash,
        plan_hash=draft.plan_hash,
        performance_plan_hash=draft.performance_plan_hash,
        lint_passed=draft.lint.passed,
        quality=draft.quality.as_dict(),
        findings=[asdict(finding) for finding in draft.quality.findings()],
        sketch=(
            None
            if draft.sketch is None
            else SketchResponse(
                url=f"/sessions/{session_id}/sketch/{draft.draft_id}/ogg",
                ogg_sha256=draft.sketch.ogg_sha256,
                ogg_size_bytes=draft.sketch.ogg_size_bytes,
            )
        ),
    )


@router.post("/sessions", status_code=status.HTTP_201_CREATED)
async def create_session(body: CreateSessionRequest, request: Request) -> SessionResponse:
    """Open a session on `brief`, and run its first turn.

    The turn runs inside this request rather than in a queue: the conductor's
    first decision *is* the session's content, so a 201 carrying an empty
    session would be a promise rather than an answer.
    """
    sessions = _storage(request)
    jobs = _jobs(request)
    client = _model(request)
    session = sessions.create(body.brief)
    try:
        await conduct(
            sessions,
            jobs,
            client,
            session,
            trigger="brief",
            auto_finalize=body.auto_finalize,
        )
    except ToolRefusal as exc:
        raise http_from_refusal(exc, session) from exc
    return _serialize_session(session)


@router.get("/sessions/{session_id}")
def get_session(session_id: str, request: Request) -> SessionResponse:
    return _serialize_session(_session(request, session_id))


@router.post("/sessions/{session_id}/message")
async def send_message(session_id: str, body: MessageRequest, request: Request) -> SessionResponse:
    """Say something to the conductor, and take the turn it decides on.

    Refused once the session has published, for `_already_published`'s reason:
    the conductor could draft all afternoon and none of it could be mastered.
    """
    sessions = _storage(request)
    jobs = _jobs(request)
    client = _model(request)
    session = _session(request, session_id)
    if session.is_finalized:
        raise _already_published(session)
    try:
        await conduct(
            sessions,
            jobs,
            client,
            session,
            trigger="message",
            message=body.message,
            auto_finalize=body.auto_finalize,
        )
    except ToolRefusal as exc:
        raise http_from_refusal(exc, session) from exc
    return _serialize_session(session)


@router.post("/sessions/{session_id}/verdict")
def record_verdict(session_id: str, body: VerdictRequest, request: Request) -> SessionResponse:
    """Record what the user thought of one draft.

    This writes a `Verdict` and takes no turn, because a like or a dislike is
    not by itself an instruction. It is not inert either: the digest carries the
    verdicts, so the conductor's next turn reads them, and each one is a
    `(plan_hash, verdict)` pair — the raw material of the slow loop, recorded
    now because nothing else can recover it later. Feedback that *is* an
    instruction goes to `/message`, where the conductor can act on it.

    Verdicts accumulate rather than replace: a user changing their mind about a
    draft is the most informative thing they can do, and a record that kept only
    the latest would throw that away. Judging a *published* draft is allowed —
    the piece is rendered and the opinion still counts.
    """
    sessions = _storage(request)
    session = _session(request, session_id)
    known = sorted(draft.draft_id for draft in session.drafts)
    if body.draft_id not in known:
        raise HTTPException(
            status_code=404,
            detail=f"this session has no draft {body.draft_id!r}; it has {known or 'none'}",
        )
    try:
        verdict = Verdict(
            draft_id=body.draft_id,
            at=datetime.now(UTC),
            value=body.value,
            feedback=body.feedback,
        )
    except ValueError as exc:
        # The record owns the rule — a verdict carries a like, a dislike or
        # words, and nothing is not a verdict — so this translates rather than
        # restates it.
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    session.verdicts.append(verdict)
    sessions.save(session)
    return _serialize_session(session)


@router.post("/sessions/{session_id}/finalize", status_code=status.HTTP_202_ACCEPTED)
def finalize_session(session_id: str, body: FinalizeRequest, request: Request) -> FinalizeResponse:
    """Publish one draft by hand — the button the auto setting would have pressed.

    `publish_draft` is the conductor's `finalize` tool, so this endpoint has no
    rules of its own: a session publishes once, the refusal for a second publish
    is the sentence the model reads, and the code it carries is what says so.

    No model is needed. Nothing here consults one — the draft was already
    composed and measured when it was drafted — which is what makes publishing
    work in a process with no configuration at all.
    """
    sessions = _storage(request)
    jobs = _jobs(request)
    session = _session(request, session_id)
    try:
        draft = session.draft(body.draft_id)
    except KeyError as exc:
        known = sorted(draft.draft_id for draft in session.drafts)
        raise HTTPException(
            status_code=404,
            detail=f"this session has no draft {body.draft_id!r}; it has {known or 'none'}",
        ) from exc
    ctx = ToolContext(session=session, sessions=sessions, jobs=jobs)
    try:
        job = publish_draft(ctx, draft)
    except ToolRefusal as exc:
        raise http_from_refusal(exc, session) from exc
    return FinalizeResponse(job_id=job.job_id, draft_id=draft.draft_id, state=job.state.value)


@router.post("/sessions/{session_id}/undo")
def undo_session(session_id: str, request: Request) -> SessionResponse:
    """Step the session back one save.

    Both refusals are the store's and both are a 409: nothing is behind the
    session, or the session has published and `undo` will not cross a publish
    because the render is already queued. The store raises with the reason, so
    this is a translation and not a second rule.
    """
    sessions = _storage(request)
    _session(request, session_id)
    try:
        session = sessions.undo(session_id)
    except UndoUnavailable as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return _serialize_session(session)


@router.get("/sessions/{session_id}/sketch/{draft_id}/{kind}")
def get_sketch(session_id: str, draft_id: str, kind: str, request: Request) -> FileResponse:
    """Serve a draft's sketch audio, by the name of the file wanted.

    Two kinds, because the render writes two files and both are useful: the Ogg
    is what a browser plays, and the Wav is the same performance before the
    single-pass encode. An unplayable *record* is a 404 — this draft has no
    sketch — while a record whose file is gone is a 410, on the job artifact
    route's reasoning: the sketch was rendered and the disk lost it, which is a
    different thing to be told than "there is no such sketch".
    """
    sessions = _storage(request)
    session = _session(request, session_id)
    try:
        draft = session.draft(draft_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"no draft {draft_id!r}") from exc
    if draft.sketch is None:
        raise HTTPException(
            status_code=404,
            detail=f"draft {draft_id!r} has no sketch yet; call sketch for it first",
        )
    entry = _SKETCH_FILES.get(kind)
    if entry is None:
        raise HTTPException(
            status_code=404,
            detail=f"a sketch has no {kind!r}; it has {sorted(_SKETCH_FILES)}",
        )
    attribute, media_type = entry
    path = sessions.sketch_path(session_id, getattr(draft.sketch, attribute))
    if not path.exists():
        raise HTTPException(status_code=410, detail="the sketch file is missing on disk")
    return FileResponse(path, media_type=media_type, filename=path.name)


__all__ = [
    "DEFAULT_AUTO_FINALIZE",
    "MAX_AUTO_TURNS",
    "CreateSessionRequest",
    "DraftResponse",
    "FinalizeRequest",
    "FinalizeResponse",
    "MessageRequest",
    "SessionResponse",
    "SketchResponse",
    "ToolCallResponse",
    "TurnResponse",
    "VerdictRequest",
    "VerdictResponse",
    "conduct",
    "http_from_refusal",
    "router",
]
