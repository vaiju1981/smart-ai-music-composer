"""FastAPI routes for the Phase 1 web app.

Per `docs/roadmap.md` §7:

- `POST /jobs` — accepts `{prompt: string}`, creates the job, returns `job_id`.
  Prompt parsing happens in the worker's `parsing` stage.

  **It is now a thin alias over a session.** When this app has a model and a
  session store — which is what the served app has and what a test app does not
  — the prompt opens a session, the conductor takes the first turn, and the job
  it publishes is what comes back. The caller cannot tell the two paths apart
  and does not have to: one prompt in, one `job_id` out, which is the contract
  this endpoint has always had. There are exactly two reasons to take the old
  path instead, plus one deliberate bypass, and all three are named in
  `create_job` below.
- `POST /jobs/from-spec` — internal/test-only Phase 1 endpoint that
  accepts an already validated `CompositionSpec`; it is not exposed by
  the user-facing UI.
- `GET /jobs/{id}` — returns state + progress + artifact URLs once complete.
- `GET /jobs/{id}/artifact/{kind}` — streams the artifact (opaque,
  unguessable token tied to the job ID; no auth needed in Phase 1 since
  the app is single-user/local).
- `POST /jobs/{id}/cancel` — allowed only in `queued | parsing | composing`;
  returns `409 Conflict` once rendering has started.

Phase 1 is single-user / local-only — there is no per-user auth and
artifact URLs are opaque tokens scoped to the job ID itself.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from saimc.jobs.state import JobState, JobStateMachine
from saimc.jobs.storage import (
    DEFAULT_JOBS_DIR,
    ArtifactRecord,
    Job,
    JobStorage,
)
from saimc.jobs.worker import DEFAULT_QUEUE, QueueUnavailable, enqueue_or_fail
from saimc.llm.base import ChatClient
from saimc.session.api import conduct, http_from_refusal
from saimc.session.api import router as session_router
from saimc.session.preferences import PreferenceLog
from saimc.session.store import SessionStorage
from saimc.session.tools import ToolRefusal
from saimc.spec import CompositionSpec

router = APIRouter()
_state_machine = JobStateMachine()

# The web UI is a single static page (roadmap §2 "prompt box, job status,
# preview, downloads") that talks to the JSON routes below; no templating
# or asset pipeline for Phase 1.
STATIC_DIR = Path(__file__).resolve().parent / "static"


class CreateJobRequest(BaseModel):
    """Body for `POST /jobs`."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=4096)
    seed: int | None = Field(default=None, ge=0)
    """Optional reproducibility seed; the parsed spec can still override it."""


class CreateJobFromSpecRequest(BaseModel):
    """Body for `POST /jobs/from-spec` (internal/test endpoint)."""

    model_config = ConfigDict(extra="forbid")

    spec: CompositionSpec


class JobResponse(BaseModel):
    """Public job view returned by the API."""

    job_id: str
    state: str
    progress: float
    current_stage: str
    created_at: str
    updated_at: str
    input_prompt: str
    parser_source: str | None
    attempts: int
    seed: int | None
    model: str | None = None
    """The LLM that served this job's parse, when one did (§9); None for a
    fallback-only parse."""
    input_spec: dict[str, Any] | None = None
    """The parsed CompositionSpec, once available (None until parsing succeeds)."""
    artifacts: dict[str, dict[str, Any]]
    error: dict[str, str] | None


def _storage(request: Request) -> JobStorage:
    """Resolve the JobStorage from app state.

    Tests inject a custom storage via `app.state.job_storage`; the
    production wiring in `saimc.jobs.worker` populates it from
    `SAIMC_JOBS_DIR`.
    """
    storage: JobStorage | None = getattr(request.app.state, "job_storage", None)
    if storage is None:
        raise HTTPException(status_code=500, detail="Job storage not configured.")
    return storage


def _session_llm(request: Request) -> ChatClient | None:
    """The conductor's model, or `None` when this app was built without one.

    `None` is not an error here, unlike on the session routes: this endpoint has
    a whole second path that needs no model at all, so a model-less app answers
    `/jobs` exactly as it always did. That is what keeps the existing tests —
    and a deployment with no Ollama — working unchanged.
    """
    return getattr(request.app.state, "session_llm", None)


def _serialize_job(job: Job) -> JobResponse:
    return JobResponse(
        job_id=job.job_id,
        state=job.state.value,
        progress=job.progress,
        current_stage=job.current_stage,
        created_at=job.created_at.isoformat(),
        updated_at=job.updated_at.isoformat(),
        input_prompt=job.input_prompt,
        parser_source=job.parser_source,
        attempts=job.attempts,
        seed=job.seed,
        model=job.model,
        input_spec=(job.input_spec.model_dump(mode="json") if job.input_spec is not None else None),
        artifacts={
            kind: {
                "kind": a.kind,
                "container": a.container,
                "codec": a.codec,
                "path": a.path,
                "sha256": a.sha256,
                "size_bytes": a.size_bytes,
            }
            for kind, a in job.artifacts.items()
        },
        error=(
            {
                "error_code": job.error.error_code,
                "message": job.error.message,
                "stage": job.error.stage,
            }
            if job.error is not None
            else None
        ),
    )


def _service_health() -> dict[str, Any]:
    """Return a small operational snapshot for the local studio UI."""
    from rq import Queue, Worker

    from saimc.jobs.worker import _redis_from_url

    url = os.environ.get("SAIMC_VALKEY_URL") or "valkey://127.0.0.1:6379/0"
    try:
        connection = _redis_from_url(url)
        connection.ping()
        workers = [
            worker
            for worker in Worker.all(connection=connection)
            if connection.ttl(worker.key) > 0 and _local_worker_process_alive(worker.pid)
        ]
        return {
            "status": "ready" if workers else "waiting_for_worker",
            "broker": "ready",
            "workers": len(workers),
            "queue_depth": Queue(DEFAULT_QUEUE, connection=connection).count,
        }
    except Exception:
        return {
            "status": "unavailable",
            "broker": "unavailable",
            "workers": 0,
            "queue_depth": None,
        }


def _local_worker_process_alive(pid: int | None) -> bool:
    """Reject RQ registrations left behind by an abruptly killed worker.

    RQ deliberately keeps an idle worker key alive for several minutes.
    This application runs API and worker on the same machine, so checking
    the registered PID prevents the UI from calling a stale registration
    "ready" while newly submitted jobs wait forever.
    """
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        return True
    return True


def _enqueue(job: Job, storage: JobStorage) -> None:
    """Put the job on the RQ queue, surfacing broker outages as 503.

    The job is persisted before enqueueing, and the outage policy — mark the
    record `failed` rather than leave it `queued` — lives in
    `enqueue_or_fail`, because the session's `finalize` tool has to apply the
    same one. This is only the HTTP translation of it.
    """
    try:
        enqueue_or_fail(job, storage)
    except QueueUnavailable as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc


@router.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the single-page web UI."""
    return FileResponse(
        STATIC_DIR / "index.html",
        media_type="text/html",
        headers={"Cache-Control": "no-store, max-age=0"},
    )


def _queue_one_shot(body: CreateJobRequest, request: Request) -> JobResponse:
    """The Phase 1 path, unchanged: persist the prompt and queue it.

    Kept whole rather than folded into the alias so that the two ways of
    answering this endpoint stay readable side by side — one of them is the
    product's front door and the other is the harness's.
    """
    storage = _storage(request)
    job = storage.create(body.prompt)
    if body.seed is not None:
        job.seed = body.seed
        storage.save(job)
    _enqueue(job, storage)
    return _serialize_job(job)


@router.post("/jobs", status_code=status.HTTP_202_ACCEPTED)
async def create_job(body: CreateJobRequest, request: Request) -> JobResponse:
    """Create a job from a user prompt — through a session, or the old way.

    The session pass is the product: the conductor reads the brief, drafts, and
    publishes, and the job that comes back is the piece. The Phase 1 path is
    taken in three situations, and every one of them is a case where the session
    pass cannot answer the request as asked:

    - **No model.** A conductor with nothing to think with would draft nothing;
      queueing the prompt lets the worker's own parse stage handle it, which is
      what "the product keeps working with the model absent" means here.
    - **Nothing published.** An open brief can end a pass with several
      candidates and no piece, because choosing between them is the user's. The
      alias still promises a job, so the prompt is queued by the old path and
      the session is left where it is, holding its candidates, for the user to
      come back to. Exactly one job exists per request on every path.
    - **An explicit seed.** A seed is a request for one exact piece, which is
      the one-shot contract itself: it says "compose this again", not "choose
      something for me". The seeds the conductor fans out over are its own, so
      there is nothing to hand it — and silently ignoring a request parameter is
      the deaf-product failure this codebase refuses elsewhere.

    A refusal the *harness* raised — a broker that would not take the job —
    propagates as its own status, because in that case the session has not
    published and the caller's recovery is the session's own `/finalize`.
    """
    sessions: SessionStorage | None = getattr(request.app.state, "session_storage", None)
    client = _session_llm(request)
    if sessions is None or client is None or body.seed is not None:
        return _queue_one_shot(body, request)

    session = sessions.create(body.prompt)
    try:
        await conduct(
            sessions,
            _storage(request),
            client,
            session,
            trigger="brief",
            auto_finalize=True,
        )
    except ToolRefusal as exc:
        raise http_from_refusal(exc, session) from exc
    if session.finalized_job_id is None:
        return _queue_one_shot(body, request)
    return _serialize_job(_storage(request).get(session.finalized_job_id))


@router.post("/jobs/from-spec", status_code=status.HTTP_202_ACCEPTED)
def create_job_from_spec(body: CreateJobFromSpecRequest, request: Request) -> JobResponse:
    """Internal/test endpoint: create a job whose spec is already parsed.

    Not exposed in the Phase 1 user-facing UI. Used by the parser
    benchmark CLI and by acceptance tests to skip the LLM round-trip.
    """
    storage = _storage(request)
    job = storage.create(prompt="(from-spec)")
    job.input_spec = body.spec
    if body.spec.seed is not None:
        job.seed = body.spec.seed
    job.parser_source = "from-spec"
    storage.save(job)
    _enqueue(job, storage)
    return _serialize_job(job)


@router.get("/jobs")
def list_jobs(request: Request, limit: int = 50) -> list[JobResponse]:
    """List jobs, newest first — the UI's history panel.

    `limit` caps the number returned (default 50); jobs beyond the cap
    remain accessible by their `/jobs/{id}` permalink.
    """
    storage = _storage(request)
    jobs = sorted(storage.list_all(), key=lambda j: j.created_at, reverse=True)
    return [_serialize_job(job) for job in jobs[: max(0, limit)]]


@router.get("/meta")
def get_meta() -> dict[str, Any]:
    """Authoring vocabulary for the UI — what Phase 1 accepts.

    One source of truth: the values come straight from the spec module,
    the ensemble tables, and the instrument registry, so schema and UI
    can never drift. `roles` is the ensemble vocabulary a spec's
    `instrumentation` list accepts; `default_ensembles` is what a bare
    instrument string expands to per mood; `dedicated_font_only` names
    the instruments that render only through their dedicated soundfont.
    """
    from saimc.compose.ensemble import SCALAR_BASS, SCALAR_HARMONY
    from saimc.render.instruments import FONT_ONLY_INSTRUMENTS, SUPPORTED_INSTRUMENTS
    from saimc.spec import (
        DURATION_SECONDS_DEFAULT,
        DURATION_SECONDS_MAX,
        DURATION_SECONDS_MIN,
        ROLE_ORDER,
        Mood,
        TimeSignature,
    )

    return {
        "moods": [m.value for m in Mood],
        "time_signatures": [t.value for t in TimeSignature],
        "instruments": sorted(SUPPORTED_INSTRUMENTS),
        "roles": [r.value for r in ROLE_ORDER],
        "default_ensembles": {
            mood: {"harmony": SCALAR_HARMONY[mood], "bass": SCALAR_BASS[mood]}
            for mood in SCALAR_HARMONY
        },
        "dedicated_font_only": sorted(FONT_ONLY_INSTRUMENTS),
        "duration_seconds": {
            "min": DURATION_SECONDS_MIN,
            "max": DURATION_SECONDS_MAX,
            "default": DURATION_SECONDS_DEFAULT,
        },
    }


@router.get("/health")
def get_health() -> dict[str, Any]:
    """Expose broker/worker readiness to the local UI."""
    return _service_health()


@router.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request) -> JobResponse:
    storage = _storage(request)
    try:
        job = storage.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    return _serialize_job(job)


@router.get("/jobs/{job_id}/events")
async def stream_job(job_id: str, request: Request) -> StreamingResponse:
    """Stream job changes so the UI does not depend on browser timers.

    Some embedded browsers heavily throttle or suspend `setInterval`, which
    made progress appear frozen until a refresh. This local SSE stream stays
    open and emits each persisted state change as soon as it is visible.
    """
    storage = _storage(request)
    try:
        storage.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc

    async def events() -> AsyncIterator[str]:
        last_updated_at: str | None = None
        idle_cycles = 0
        while not await request.is_disconnected():
            try:
                job = storage.get(job_id)
            except KeyError:
                yield 'event: error\ndata: {"message":"job not found"}\n\n'
                return
            response = _serialize_job(job)
            updated_at = response.updated_at
            if updated_at != last_updated_at:
                yield f"data: {response.model_dump_json()}\n\n"
                last_updated_at = updated_at
                idle_cycles = 0
            else:
                idle_cycles += 1
                if idle_cycles >= 20:
                    yield ": keep-alive\n\n"
                    idle_cycles = 0
            if job.state in {JobState.COMPLETE, JobState.FAILED, JobState.CANCELLED}:
                return
            await asyncio.sleep(0.5)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/jobs/{job_id}/artifact/{kind}")
def get_artifact(job_id: str, kind: str, request: Request) -> FileResponse:
    storage = _storage(request)
    try:
        job = storage.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    if kind not in job.artifacts:
        raise HTTPException(status_code=404, detail=f"Artifact {kind!r} not found")
    artifact: ArtifactRecord = job.artifacts[kind]
    full_path = storage.artifact_path(job_id, artifact.path)
    if not full_path.exists():
        raise HTTPException(status_code=410, detail="Artifact file missing on disk")
    return FileResponse(
        full_path,
        media_type=_media_type(artifact),
        filename=artifact.path,
    )


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str, request: Request) -> JSONResponse:
    storage = _storage(request)
    try:
        job = storage.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    if not _state_machine.can_cancel(job.state):
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={
                "error_code": "cancel_not_allowed",
                "message": f"Job in state {job.state.value} cannot be cancelled.",
                "state": job.state.value,
            },
        )
    transition = _state_machine.transition(job.state, JobState.CANCELLED)
    job.state = transition.state
    job.progress = transition.progress
    job.current_stage = transition.current_stage
    storage.save(job)
    return JSONResponse(
        status_code=status.HTTP_200_OK,
        content=_serialize_job(job).model_dump(),
    )


def _media_type(artifact: ArtifactRecord) -> str:
    table = {
        "wav": "audio/wav",
        "ogg": "audio/ogg",
        "svg": "image/svg+xml",
        "png": "image/png",
        "pdf": "application/pdf",
        "webm": "video/webm",
        "mid": "audio/midi",
    }
    return table.get(artifact.container, "application/octet-stream")


def create_app(
    jobs_root: Path | str | None = None,
    sessions_root: Path | str | None = None,
    preferences_root: Path | str | None = None,
    *,
    session_llm: ChatClient | None = None,
) -> FastAPI:
    """Build the FastAPI app with its three storages and, optionally, a model.

    **No model by default, and that is the decision worth reading.** Every
    existing test and every local run builds the app through here, and a factory
    that read `saimc.toml` would hand each of them a real adapter — so
    `create_app(jobs_root=tmp_path)` would reach the configured host on every
    `POST /jobs`, in tests that mean to touch no network at all. A factory
    whose defaults are inert is one whose callers opt into a model.

    Three roots rather than one because the three records have three lifetimes:
    a job is pruned at 7 days, a session at 30, and the preference log never.
    All three are optional and each store defaults its own root from the
    environment, so a caller that names none gets the deployment's own layout.

    `create_served_app` is the one that reads the configuration, and it is what
    the server runs.
    """
    app = FastAPI(
        title="Smart AI Music Composer",
        version="0.1.0",
        description=(
            "Phase 1 local API. See `docs/roadmap.md` for the full "
            "architecture and `docs/parser-benchmark.md` for the parser "
            "benchmark corpus."
        ),
    )
    app.state.job_storage = JobStorage(jobs_root or DEFAULT_JOBS_DIR)
    app.state.session_storage = SessionStorage(sessions_root)
    app.state.preference_log = PreferenceLog(preferences_root)
    app.state.session_llm = session_llm
    app.include_router(router)
    app.include_router(session_router)
    return app


def create_served_app() -> FastAPI:
    """Build the app a server runs: the same one, with the configured model.

    The only factory that reads `saimc.toml` and the environment, and therefore
    the only one whose behaviour depends on the machine it is started on. A
    configuration that cannot be read leaves the model empty rather than failing
    to start: the sessions surface says so with a 503 naming what to set, and
    `POST /jobs` falls back to the Phase 1 path — a server that serves something
    is more useful than one that refuses to boot over a missing file.

    The import is deferred so that building an app with no model never pays for
    the provider's transport.
    """
    from saimc.llm.config import build_ollama_adapter

    return create_app(session_llm=build_ollama_adapter())


__all__ = [
    "CreateJobFromSpecRequest",
    "CreateJobRequest",
    "JobResponse",
    "create_app",
    "create_served_app",
    "router",
]
