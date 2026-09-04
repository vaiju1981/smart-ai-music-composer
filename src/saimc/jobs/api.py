"""FastAPI routes for the Phase 1 web app.

Per `docs/roadmap.md` §7:

- `POST /jobs` — accepts `{prompt: string}`, creates the job, returns `job_id`.
  Prompt parsing happens in the worker's `parsing` stage.
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

from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, HTTPException, Request, status
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field

from saimc.jobs.state import JobState, JobStateMachine
from saimc.jobs.storage import (
    DEFAULT_JOBS_DIR,
    ArtifactRecord,
    Job,
    JobStorage,
)
from saimc.spec import CompositionSpec

router = APIRouter()
_state_machine = JobStateMachine()


class CreateJobRequest(BaseModel):
    """Body for `POST /jobs`."""

    model_config = ConfigDict(extra="forbid")

    prompt: str = Field(min_length=1, max_length=4096)


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


@router.post("/jobs", status_code=status.HTTP_202_ACCEPTED)
def create_job(body: CreateJobRequest, request: Request) -> JobResponse:
    """Create a queued job from a user prompt.

    Parsing happens later, in the worker's `parsing` stage — the API
    only persists the prompt and returns the job_id.
    """
    storage = _storage(request)
    job = storage.create(body.prompt)
    return _serialize_job(job)


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
    return _serialize_job(job)


@router.get("/jobs/{job_id}")
def get_job(job_id: str, request: Request) -> JobResponse:
    storage = _storage(request)
    try:
        job = storage.get(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found") from exc
    return _serialize_job(job)


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


def create_app(jobs_root: Path | str | None = None) -> FastAPI:
    """Build the FastAPI app with a JobStorage wired into state."""
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
    app.include_router(router)
    return app


__all__ = [
    "CreateJobFromSpecRequest",
    "CreateJobRequest",
    "JobResponse",
    "create_app",
    "router",
]
