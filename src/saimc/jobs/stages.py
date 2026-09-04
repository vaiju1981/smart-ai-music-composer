"""Per-stage job runners — pure logic the RQ worker dispatches into.

The worker (saimc/jobs/worker.py) is the thin RQ wrapper. This module
contains the actual stage transitions and the bookkeeping that goes
with them, kept separate so it can be unit-tested without a real RQ
queue or a running Valkey.

The stages wired up here are:

- `parse_stage(job, llm_client, request_id)`: parse the prompt via the
  LLM client (with the §6 2-repair policy already handled by
  `saimc.parser.parse_prompt`). On success the spec is attached to the
  job; on failure the job transitions to `failed`.
- `compose_stage(job, engine)` runs the music21 composition engine.
  Stubbed in Phase 1 — the real engine lands in slice 4.
- `validate_stage(job, linter)` runs the theory linter.
- `render_*_stage` runs each renderer (audio / sheet / animation).

Each stage is a pure function: it mutates and returns the `Job`, and
the worker is responsible for persisting the result and enqueueing the
next stage. Stages must NEVER do unbounded work — every subprocess is
launched through `safe_run()` which enforces a timeout.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from saimc.jobs.state import JobState, JobStateMachine
from saimc.jobs.storage import (
    Job,
    JobError,
    JobStorage,
)

if TYPE_CHECKING:
    from saimc.llm.base import LLMClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StageResult:
    """Bookkeeping returned by every stage."""

    job: Job
    next_state: JobState | None
    """`None` means no automatic transition; the stage was a no-op or
    returned control to the worker. Otherwise the worker must transition
    the job to this state via `JobStateMachine.transition`."""

    error: JobError | None = None


class SubprocessTimeoutError(Exception):
    """Raised by `safe_run` when a subprocess exceeds its timeout."""


def safe_run(
    cmd: Iterable[str],
    *,
    timeout_s: float,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    stdin_bytes: bytes | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run `cmd` with an absolute timeout and a bounded working directory.

    The timeout is wall-clock, not CPU. The subprocess is killed and
    reaped on timeout. Output is captured to a string; `text=True`
    forces UTF-8 decoding.
    """
    cmd_list = [str(c) for c in cmd]
    if not cmd_list:
        raise ValueError("cmd must be non-empty")
    logger.debug(
        "subprocess start: %s (timeout=%.1fs)",
        " ".join(shlex.quote(c) for c in cmd_list),
        timeout_s,
    )
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(
            cmd_list,
            cwd=cwd,
            env=env,
            input=stdin_bytes,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        elapsed = time.perf_counter() - t0
        logger.error("subprocess timeout after %.1fs: %s", elapsed, cmd_list)
        raise SubprocessTimeoutError(f"{cmd_list[0]} exceeded {timeout_s}s") from exc
    logger.debug("subprocess done in %.2fs: rc=%d", time.perf_counter() - t0, proc.returncode)
    return proc


def parse_stage(
    job: Job,
    llm_client: LLMClient,
    request_id: str,
) -> StageResult:
    """Run the `parsing` stage.

    Synchronous wrapper around `parse_prompt` — the actual parse call is
    awaited via `asyncio.run` because RQ worker functions are
    synchronous (RQ has no native async-worker support).
    """
    import asyncio

    from saimc.parser import parse_prompt

    try:
        result = asyncio.run(parse_prompt(llm_client, job.input_prompt, request_id=request_id))
    except Exception as exc:
        return StageResult(
            job=job,
            next_state=JobState.FAILED,
            error=JobError(
                error_code="parser_exception",
                message=f"parse_prompt raised: {exc}",
                stage="parsing",
            ),
        )

    if result.spec is not None:
        job.input_spec = result.spec
        if result.spec.seed is not None:
            job.seed = result.spec.seed
        job.parser_source = result.parser_source
        job.attempts = result.attempts
        return StageResult(job=job, next_state=JobState.COMPOSING)

    spec_err = result.error
    if spec_err is not None:
        return StageResult(
            job=job,
            next_state=JobState.FAILED,
            error=JobError(
                error_code=spec_err.error_code,
                message=spec_err.message,
                stage="parsing",
            ),
        )
    return StageResult(
        job=job,
        next_state=JobState.FAILED,
        error=JobError(
            error_code="parser_no_spec_no_error",
            message="Parser returned neither spec nor error.",
            stage="parsing",
        ),
    )


def compose_stage(job: Job, storage: JobStorage, engine: Any | None = None) -> StageResult:
    """Run the `composing` stage.

    `engine` is a callable `(spec) -> (NotationScore, PerformancePlan)`.
    It is `None` for the stub implementation, in which case this stage
    fails with `engine_not_implemented` — Phase 1 acceptance tests will
    surface this until slice 4 lands.
    """
    if engine is None:
        return StageResult(
            job=job,
            next_state=JobState.FAILED,
            error=JobError(
                error_code="engine_not_implemented",
                message="Composition engine is not wired in this build.",
                stage="composing",
            ),
        )
    if job.input_spec is None:
        return StageResult(
            job=job,
            next_state=JobState.FAILED,
            error=JobError(
                error_code="no_spec",
                message="Cannot compose without a parsed spec.",
                stage="composing",
            ),
        )
    try:
        notation_score, performance_plan = engine(job.input_spec)
    except Exception as exc:
        return StageResult(
            job=job,
            next_state=JobState.FAILED,
            error=JobError(
                error_code="compose_failed",
                message=str(exc),
                stage="composing",
            ),
        )
    # The real slice-3 + slice-4 work will compute hashes here and attach
    # them; for now we leave the spec attached and move on.
    del notation_score, performance_plan
    return StageResult(job=job, next_state=JobState.VALIDATING)


def transition_to(job: Job, target: JobState, sm: JobStateMachine | None = None) -> Job:
    """Apply `target` via the state machine, raising `IllegalTransitionError` on bad moves."""
    sm = sm or JobStateMachine()
    result = sm.transition(job.state, target)
    job.state = result.state
    job.progress = result.progress
    job.current_stage = result.current_stage
    return job


__all__ = [
    "StageResult",
    "SubprocessTimeoutError",
    "compose_stage",
    "parse_stage",
    "safe_run",
    "transition_to",
]
