"""RQ worker entry point.

Per `docs/roadmap.md` §10 #8:

> Jobs: RQ + Valkey, bound locally, with RQ `JSONSerializer` and
> primitive job arguments. Exact versions are pinned in lockfiles; an
> integration test against the selected Valkey release is a release gate.

This module provides:

- `worker_entry(jobs_root, valkey_url)`: spawn an RQ worker on the
  `default` queue, configured to use `JSONSerializer` (no pickle).
- `enqueue_job(job_id, jobs_root)`: submit a job for processing from
  the FastAPI process.
- `run_job(job_id, jobs_root, llm_client)`: the actual job function
  RQ invokes. Loads the canonical job from storage, walks it through
  every stage (currently `parsing` → `composing`; renderers are
  stubbed until slices 5-7 land), and persists state after every
  transition.

The worker is intentionally synchronous per stage (RQ has no native
async workers) and uses only `job_id` as the RQ argument so payloads
in the broker contain no Python objects that would need pickling.
"""

from __future__ import annotations

import os
from typing import Any

from saimc.jobs.stages import (
    compose_stage,
    parse_stage,
    transition_to,
)
from saimc.jobs.state import IllegalTransitionError, JobState, JobStateMachine
from saimc.jobs.storage import Job, JobError, JobStorage

DEFAULT_QUEUE = "default"


def worker_entry(
    *,
    jobs_root: str | None = None,
    valkey_url: str | None = None,
    queue: str = DEFAULT_QUEUE,
) -> None:
    """Start an RQ worker process.

    This is a console-script entry point (`saimc-jobs`). It blocks on
    the worker until SIGINT/SIGTERM.

    Requires `redis` (the `redis` Python client that RQ depends on)
    and `rq` to be importable; both are pinned in pyproject.toml.
    """
    from rq import Queue, Worker
    from rq.serializers import JSONSerializer

    url = valkey_url or os.environ.get("SAIMC_VALKEY_URL") or "valkey://127.0.0.1:6379/0"
    queue_obj = Queue(
        name=queue,
        connection=_redis_from_url(url),
        serializer=JSONSerializer(),
    )
    # The worker reads from `queue_obj` and calls functions registered
    # with `@job` (or looked up by name). Phase 1 keeps things simple:
    # the worker always runs `run_job` for any message.
    worker = Worker(
        [queue_obj],
        connection=queue_obj.connection,
        serializer=JSONSerializer(),
    )
    worker.work()


def enqueue_job(job_id: str, *, valkey_url: str | None = None) -> str:
    """Submit `job_id` for processing by the RQ worker.

    Returns the RQ job id. Fails if Valkey is unreachable — the API
    caller should treat this as a transient error and let the user retry.
    """
    from rq import Queue
    from rq.serializers import JSONSerializer

    url = valkey_url or os.environ.get("SAIMC_VALKEY_URL") or "valkey://127.0.0.1:6379/0"
    queue = Queue(
        name=DEFAULT_QUEUE,
        connection=_redis_from_url(url),
        serializer=JSONSerializer(),
    )
    rq_job = queue.enqueue(
        "saimc.jobs.worker.run_job",
        job_id,
        job_timeout="30m",
        serializer=JSONSerializer(),
    )
    return rq_job.get_id()


def run_job(job_id: str, jobs_root: str | None = None) -> dict[str, Any]:
    """RQ job function — walks a job through every stage and persists state.

    This is the only function the worker invokes. It loads the
    canonical Job from storage, runs `parse_stage` then `compose_stage`,
    and persists the job after each transition. Render stages (audio /
    sheet / animation) are stubbed until slices 5-7 land; the worker
    transitions through `validating → rendering_audio → ... → complete`
    as no-ops for now so the state-machine wiring is exercised.

    Args:
        job_id: the opaque job id. Only primitive arguments may be
            serialized into the broker — never put the Job itself in here.
        jobs_root: jobs directory. Optional; falls back to
            `SAIMC_JOBS_DIR` or `./var/jobs`.

    Returns:
        A small status dict for the broker log.
    """
    storage = JobStorage(jobs_root or os.environ.get("SAIMC_JOBS_DIR"))
    sm = JobStateMachine()
    job = storage.get(job_id)

    try:
        _walk_stages(job, storage, sm)
    except IllegalTransitionError as exc:
        _mark_failed(
            job,
            storage,
            JobError(
                error_code="illegal_transition",
                message=str(exc),
                stage=exc.from_state.value,
            ),
        )
        return {"job_id": job_id, "state": job.state.value, "error": "illegal_transition"}

    return {"job_id": job_id, "state": job.state.value}


def _walk_stages(job: Job, storage: JobStorage, sm: JobStateMachine) -> None:
    """Stage-by-stage state walk. Each transition is persisted before the next stage."""
    llm_client = _build_default_llm_client()

    # queued -> parsing
    if job.state == JobState.QUEUED:
        transition_to(job, JobState.PARSING)
        storage.save(job)

    # parsing -> composing (or -> failed).
    # Skip the LLM parse if the spec is already attached (the /jobs/from-spec
    # internal endpoint sets it directly).
    if job.state == JobState.PARSING:
        if job.input_spec is not None:
            transition_to(job, JobState.COMPOSING, sm)
            storage.save(job)
        else:
            result = parse_stage(job, llm_client, request_id=job.job_id)
            job.input_spec = result.job.input_spec
            job.parser_source = result.job.parser_source
            job.attempts = result.job.attempts
            if result.job.seed is not None:
                job.seed = result.job.seed
            if result.error is not None:
                job.error = result.error
            if result.next_state is not None:
                transition_to(job, result.next_state, sm)
            storage.save(job)

    # composing -> validating (or -> failed)
    if job.state == JobState.COMPOSING:
        result = compose_stage(job, storage)
        if result.error is not None:
            job.error = result.error
        if result.next_state is not None:
            transition_to(job, result.next_state, sm)
        storage.save(job)

    # validating -> rendering_audio -> rendering_sheet -> rendering_animation -> complete
    for next_state in (
        JobState.RENDERING_AUDIO,
        JobState.RENDERING_SHEET,
        JobState.RENDERING_ANIMATION,
        JobState.COMPLETE,
    ):
        if job.state == JobState.VALIDATING and next_state == JobState.RENDERING_AUDIO:
            transition_to(job, next_state, sm)
            storage.save(job)
        elif job.state in (
            JobState.RENDERING_AUDIO,
            JobState.RENDERING_SHEET,
            JobState.RENDERING_ANIMATION,
        ):
            # Renderers are stubs in this slice — move on to the next stage.
            transition_to(job, next_state, sm)
            storage.save(job)


def _mark_failed(job: Job, storage: JobStorage, error: JobError) -> None:
    from contextlib import suppress

    job.error = error
    with suppress(IllegalTransitionError):
        transition_to(job, JobState.FAILED)
    storage.save(job)


def _build_default_llm_client() -> Any:
    """Construct the default LLM client. Phase 1 uses OllamaAdapter.

    The import is deferred so that running `worker_entry` without an
    Ollama endpoint doesn't crash on import. If the env is missing,
    we fall back to a no-op client whose `parse` always fails cleanly.
    """
    from saimc.llm.base import ParseRequest, ParseResult

    class _NoopLLM:
        async def parse(self, request: ParseRequest) -> ParseResult:
            return ParseResult(
                parser_source="llm",
                error=__import__("saimc.spec", fromlist=["SpecError"]).SpecError(
                    error_code="llm_not_configured",
                    message="No OLLAMA_* environment variables set.",
                    stage="parsing",
                ),
            )

        async def aclose(self) -> None:
            return None

    try:
        from saimc.llm.ollama import OllamaAdapter

        return OllamaAdapter()
    except ValueError:
        return _NoopLLM()


def _redis_from_url(url: str) -> Any:
    """Build a `redis.Redis` client from a Valkey-compatible URL."""
    import redis

    return redis.Redis.from_url(url)


__all__ = [
    "DEFAULT_QUEUE",
    "enqueue_job",
    "run_job",
    "worker_entry",
]


def main() -> None:  # pragma: no cover — entry point
    import argparse

    parser = argparse.ArgumentParser(description="saimc job worker")
    parser.add_argument("tojobs-root", default=os.environ.get("SAIMC_JOBS_DIR"))
    parser.add_argument("tovalkey-url", default=os.environ.get("SAIMC_VALKEY_URL"))
    args = parser.parse_args()
    worker_entry(jobs_root=args.jobs_root, valkey_url=args.valkey_url)


if __name__ == "__main__":
    main()
