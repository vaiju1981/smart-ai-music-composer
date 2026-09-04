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
import os
import secrets
import tempfile
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from saimc.canonical import canonical_dumps
from saimc.jobs.state import JobState
from saimc.spec import CompositionSpec

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
    """

    kind: str  # "audio" | "sheet" | "animation"
    container: str  # "wav" | "ogg" | "svg" | "png" | "pdf" | "webm"
    codec: str
    path: str
    sha256: str
    size_bytes: int


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
    artifacts: dict[str, ArtifactRecord] = field(default_factory=dict)
    error: JobError | None = None
    engine_version: str = "0.1.0"
    seed: int | None = None
    parser_source: str | None = None
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
        """Iterate over every persisted job, oldest first."""
        paths = sorted(self._root.glob("*/job.json"))
        for path in paths:
            yield self._read(path)

    def prune(self, now: datetime | None = None) -> int:
        """Delete jobs past their retention horizon. Returns count deleted."""
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

    def attach_spec(
        self, job: Job, spec: CompositionSpec, parser_source: str, attempts: int
    ) -> Job:
        """Attach a parsed spec to the job (called after the `parsing` stage)."""
        job.input_spec = spec
        job.parser_source = parser_source
        job.attempts = attempts
        if spec.seed is not None:
            job.seed = spec.seed
        return job

    def attach_error(self, job: Job, error: JobError) -> Job:
        job.error = error
        return job

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
        return {
            "job_id": job.job_id,
            "created_at": job.created_at.isoformat(),
            "updated_at": job.updated_at.isoformat(),
            "state": job.state.value,
            "progress": job.progress,
            "current_stage": job.current_stage,
            "input_prompt": job.input_prompt,
            "input_spec": spec_payload,
            "artifacts": {
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
            "attempts": job.attempts,
        }

    @staticmethod
    def _deserialize(payload: dict[str, Any]) -> Job:
        spec = (
            CompositionSpec.model_validate(payload["input_spec"])
            if payload.get("input_spec") is not None
            else None
        )
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
            artifacts=artifacts,
            error=error,
            engine_version=payload["engine_version"],
            seed=payload.get("seed"),
            parser_source=payload.get("parser_source"),
            attempts=payload.get("attempts", 0),
        )


def make_opaque_token(nbytes: int = 16) -> str:
    """Return a URL-safe random token for artifact URLs.

    Currently just the hex job_id, but exposed as its own function so we
    can rotate the generation scheme without rewriting call sites.
    """
    return secrets.token_urlsafe(nbytes)


__all__ = [
    "COMPLETED_RETENTION_DAYS",
    "DEFAULT_JOBS_DIR",
    "FAILED_RETENTION_HOURS",
    "ArtifactRecord",
    "Job",
    "JobError",
    "JobStorage",
    "make_opaque_token",
]
