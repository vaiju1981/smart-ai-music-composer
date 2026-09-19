"""Job storage — on-disk persistence for single-user Phase 1.

Per `docs/roadmap.md` §7 (Authentication / authorization):

> Phase 1 is **single-user / local-only.** The FastAPI app runs on the
> user's own machine, jobs are scoped to that machine, and there is no
> concept of a job owner or per-user auth.
> Artifact URLs are opaque, unguessable tokens scoped to the job ID.

This module implements that contract: every Job is a JSON file under
`SAIMC_JOBS_DIR/{job_id}.json`, and the artifact token is just the job
ID itself. There is no multi-tenant addressing because there are no
multi-tenant users in Phase 1.

This layer is intentionally minimal and synchronous. The RQ worker
(slice 3.3) reads/writes via this same API; the FastAPI app does too.
We are not optimising for throughput — Phase 1 is a local single-user
app, and a JSON file is faster than the round-trip to Valkey for the
fields the worker needs.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from saimc.canonical import canonical_dumps
from saimc.compose.plan import (
    CompositionPlan,
    PlanError,
    UnsupportedPlanVersionError,
)
from saimc.jobs.state import JobState
from saimc.spec import SPEC_SCHEMA_VERSION, CompositionSpec, UnsupportedSpecVersionError

logger = logging.getLogger(__name__)

DEFAULT_JOBS_DIR = Path("./var/jobs")
"""Default location for job state files. Override via `SAIMC_JOBS_DIR`."""

COMPLETED_RETENTION_DAYS = 7
FAILED_RETENTION_HOURS = 24


@dataclass(frozen=True)
class ArtifactRecord:
    """Tracks one renderer's output (audio / sheet / animation).

    `path` is relative to the jobs root (i.e. `{job_id}/audio.wav`), so
    the artifact can be moved with the directory and the token still
    resolves.

    `toolchain` carries the producing toolchain's provenance (engine
    names/versions/build hashes and asset identifiers) so the §9
    manifest can be emitted from the persisted job alone.
    """

    kind: str  # "audio" | "audio_ogg" | "sheet" | "animation"
    container: str  # "wav" | "ogg" | "svg" | "png" | "pdf" | "webm"
    codec: str
    path: str
    sha256: str
    size_bytes: int
    toolchain: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class JobError:
    """Structured error attached to a job in `failed` (or cancellable) states."""

    error_code: str
    message: str
    stage: str


@dataclass
class Job:
    """The persisted record for one job.

    `engine_version` and `seed` are written here (alongside `spec`) so a
    re-render against the same persisted values reproduces the canonical
    artifacts hash-for-hash per §8.
    """

    job_id: str
    created_at: datetime
    updated_at: datetime
    state: JobState
    progress: float
    current_stage: str
    input_prompt: str
    input_spec: CompositionSpec | None
    input_plan: CompositionPlan | None = None
    """The plan this job composes under, or `None` for the engine's defaults.

    Stored materialized, for the reason `plan.py` gives: a plan that came
    back from a module table at read time would make this job's
    reproducibility a claim about this build rather than about the job.
    `None` is the caller saying nothing about the music beyond the spec,
    which is what every job written before this field existed also says.
    """
    artifacts: dict[str, ArtifactRecord] = field(default_factory=dict)
    error: JobError | None = None
    engine_version: str = "0.1.0"
    seed: int | None = None
    parser_source: str | None = None
    model: str | None = None
    """The LLM that served this job's parse, when one did (§9: the manifest
    records the model identifier alongside `parser_source`). None for a
    fallback-only parse, and for jobs written before this field existed."""
    attempts: int = 0

    def is_terminal(self) -> bool:
        return self.state in {JobState.COMPLETE, JobState.FAILED, JobState.CANCELLED}


class JobStorage:
    """Synchronous JSON-file backend for Job records.

    Layout::

        {root}/
            {job_id}/
                job.json
                artifacts/
                    audio.wav
                    sheet.svg
                    animation.webm
                manifest.json   # written by saimc.jobs.manifest
    """

    def __init__(self, root: Path | str | None = None) -> None:
        env_root = os.environ.get("SAIMC_JOBS_DIR")
        if root is None:
            root = env_root or DEFAULT_JOBS_DIR
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def new_job_id(self) -> str:
        return uuid.uuid4().hex

    def create(self, prompt: str) -> Job:
        """Create a fresh `queued` job for `prompt`. Persists immediately."""
        now = datetime.now(UTC)
        job = Job(
            job_id=self.new_job_id(),
            created_at=now,
            updated_at=now,
            state=JobState.QUEUED,
            progress=0.0,
            current_stage=JobState.QUEUED.value,
            input_prompt=prompt,
            input_spec=None,
        )
        self._write(job)
        return job

    def get(self, job_id: str) -> Job:
        """Load a job by ID. Raises `KeyError` if missing."""
        path = self._job_path(job_id)
        if not path.exists():
            raise KeyError(job_id)
        return self._read(path)

    def save(self, job: Job, *, at: datetime | None = None) -> None:
        """Persist the job, bumping `updated_at` unless `at` is given."""
        job.updated_at = at or datetime.now(UTC)
        self._write(job)

    def list_all(self) -> Iterator[Job]:
        """Iterate over every persisted job, oldest first.

        Jobs that fail to deserialize (e.g. a future spec schema
        version) are skipped rather than breaking the whole listing.
        """
        paths = sorted(self._root.glob("*/job.json"))
        for path in paths:
            try:
                yield self._read(path)
            except Exception:
                logger.warning("skipping unreadable job file: %s", path, exc_info=True)

    def prune(self, now: datetime | None = None) -> int:
        """Delete jobs past their retention horizon. Returns count deleted.

        Jobs that cannot be deserialized (e.g. written by a newer spec
        schema version) are left alone — pruning never deletes what it
        cannot read.
        """
        now = now or datetime.now(UTC)
        deleted = 0
        for job in list(self.list_all()):
            if not job.is_terminal():
                continue
            if job.state == JobState.COMPLETE:
                age_limit = COMPLETED_RETENTION_DAYS * 24 * 3600
            else:
                age_limit = FAILED_RETENTION_HOURS * 3600
            age = (now - job.updated_at).total_seconds()
            if age > age_limit:
                self._delete(job.job_id)
                deleted += 1
        return deleted

    def artifact_path(self, job_id: str, kind: str) -> Path:
        """Resolve the on-disk path for a job's artifact of the given kind."""
        return self._root / job_id / "artifacts" / kind

    def ensure_artifact_dir(self, job_id: str) -> Path:
        d = self._root / job_id / "artifacts"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def job_dir(self, job_id: str) -> Path:
        return self._root / job_id

    def attach_artifact(self, job: Job, artifact: ArtifactRecord) -> Job:
        job.artifacts[artifact.kind] = artifact
        return job

    def _job_path(self, job_id: str) -> Path:
        return self._root / job_id / "job.json"

    def _write(self, job: Job) -> None:
        path = self._job_path(job.job_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = self._serialize(job)
        # Atomic write: tmp + rename, so a crash mid-write doesn't leave a half-file.
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(path.parent),
            delete=False,
            prefix=".job-",
            suffix=".tmp",
        ) as fh:
            tmp_name = fh.name
            fh.write(canonical_dumps(payload))
            fh.write("\n")
        os.replace(tmp_name, path)

    def _read(self, path: Path) -> Job:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.loads(fh.read())
        return self._deserialize(payload)

    def _delete(self, job_id: str) -> None:
        import shutil

        shutil.rmtree(self._root / job_id, ignore_errors=True)

    @staticmethod
    def _serialize(job: Job) -> dict[str, Any]:
        spec_payload: dict[str, Any] | None = (
            job.input_spec.model_dump(mode="json") if job.input_spec is not None else None
        )
        plan_payload: dict[str, Any] | None = (
            job.input_plan.to_canonical_dict() if job.input_plan is not None else None
        )
        return {
            "job_id": job.job_id,
            "created_at": job.created_at.isoformat(),
            "updated_at": job.updated_at.isoformat(),
            "state": job.state.value,
            "progress": job.progress,
            "current_stage": job.current_stage,
            "input_prompt": job.input_prompt,
            "input_spec": spec_payload,
            "input_plan": plan_payload,
            "artifacts": {
                kind: {
                    "kind": a.kind,
                    "container": a.container,
                    "codec": a.codec,
                    "path": a.path,
                    "sha256": a.sha256,
                    "size_bytes": a.size_bytes,
                    "toolchain": dict(a.toolchain),
                }
                for kind, a in job.artifacts.items()
            },
            "error": (
                {
                    "error_code": job.error.error_code,
                    "message": job.error.message,
                    "stage": job.error.stage,
                }
                if job.error is not None
                else None
            ),
            "engine_version": job.engine_version,
            "seed": job.seed,
            "parser_source": job.parser_source,
            "model": job.model,
            "attempts": job.attempts,
        }

    @staticmethod
    def _deserialize(payload: dict[str, Any]) -> Job:
        spec_payload = payload.get("input_spec")
        if spec_payload is not None and spec_payload.get("schema_version") is not None:
            written = int(spec_payload["schema_version"])
            if written > SPEC_SCHEMA_VERSION:
                raise UnsupportedSpecVersionError(
                    f"job {payload.get('job_id', '?')} was written with spec schema "
                    f"version {written}, but this build understands up to version "
                    f"{SPEC_SCHEMA_VERSION}; upgrade saimc or delete the job directory."
                )
        spec = CompositionSpec.model_validate(spec_payload) if spec_payload is not None else None
        plan_payload = payload.get("input_plan")
        # A plan document of any version but this build's is refused rather
        # than coerced, unlike a spec: the spec is widened additively and
        # Pydantic fills what it is missing, while a plan is stored
        # *materialized*, so a version it cannot read is a plan whose
        # fields it cannot know. Reading one anyway would make the job's
        # reproducibility a claim this build cannot honour while still
        # loading it successfully.
        plan: CompositionPlan | None = None
        if plan_payload is not None:
            try:
                plan = CompositionPlan.from_canonical_dict(plan_payload)
            except PlanError as exc:
                raise UnsupportedPlanVersionError(
                    f"job {payload.get('job_id', '?')} carries a composition plan this "
                    f"build cannot read ({exc}); upgrade saimc or delete the job directory."
                ) from exc
        error = (
            JobError(
                error_code=payload["error"]["error_code"],
                message=payload["error"]["message"],
                stage=payload["error"]["stage"],
            )
            if payload.get("error") is not None
            else None
        )
        artifacts = {
            kind: ArtifactRecord(
                kind=a["kind"],
                container=a["container"],
                codec=a["codec"],
                path=a["path"],
                sha256=a["sha256"],
                size_bytes=a["size_bytes"],
                toolchain=dict(a.get("toolchain", {})),
            )
            for kind, a in payload.get("artifacts", {}).items()
        }
        return Job(
            job_id=payload["job_id"],
            created_at=datetime.fromisoformat(payload["created_at"]),
            updated_at=datetime.fromisoformat(payload["updated_at"]),
            state=JobState(payload["state"]),
            progress=payload["progress"],
            current_stage=payload["current_stage"],
            input_prompt=payload["input_prompt"],
            input_spec=spec,
            input_plan=plan,
            artifacts=artifacts,
            error=error,
            engine_version=payload["engine_version"],
            seed=payload.get("seed"),
            parser_source=payload.get("parser_source"),
            model=payload.get("model"),
            attempts=payload.get("attempts", 0),
        )


__all__ = [
    "COMPLETED_RETENTION_DAYS",
    "DEFAULT_JOBS_DIR",
    "FAILED_RETENTION_HOURS",
    "ArtifactRecord",
    "Job",
    "JobError",
    "JobStorage",
]
