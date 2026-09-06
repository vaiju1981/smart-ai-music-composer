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
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from saimc.compose.engine import CompositionEngineError
from saimc.jobs.state import JobState, JobStateMachine
from saimc.jobs.storage import (
    ArtifactRecord,
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
                stage=spec_err.stage,
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


def _composer_for_instrumentation(instrumentation: str) -> Callable[[Any], Any]:
    """Resolve the composer for a spec's instrumentation.

    The registry is the Phase 2 seam for dedicated engines (e.g. a
    raga/tala engine for Indian classical instrumentations). Every
    registered instrument without a dedicated engine composes with the
    Phase 1 engine — composition is notation and the engine is
    instrument-agnostic; rendering maps the voices onto the instrument's
    soundfont patch. Non-western instruments sound western-styled until
    their dedicated engines land.

    Names outside the instrument registry surface as a clean structured
    error naming what *is* supported.
    """
    from saimc.compose.engine import compose as _phase1_compose
    from saimc.render.instruments import SUPPORTED_INSTRUMENTS

    registry: dict[str, Callable[[Any], Any]] = {
        # e.g. "sitar": _raga_compose — dedicated engines go here.
    }
    composer = registry.get(instrumentation)
    if composer is not None:
        return composer
    if instrumentation in SUPPORTED_INSTRUMENTS:
        return _phase1_compose
    raise LookupError(
        f"no composer registered for instrumentation {instrumentation!r}; "
        f"known instrumentations: {', '.join(sorted(SUPPORTED_INSTRUMENTS))}"
    ) from None


def compose_stage(
    job: Job,
    storage: JobStorage,
    engine: Callable[[Any], Any] | None = None,
) -> StageResult:
    """Run the `composing` stage.

    `engine` is a callable `(spec) -> EngineOutput`. When `None`, the
    composer registered for the spec's instrumentation is used
    (Phase 1: piano -> `saimc.compose.engine.compose`).
    """
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

    if engine is None:
        try:
            engine = _composer_for_instrumentation(job.input_spec.instrumentation)
        except LookupError as exc:
            return StageResult(
                job=job,
                next_state=JobState.FAILED,
                error=JobError(
                    error_code="instrumentation_unsupported",
                    message=str(exc),
                    stage="composing",
                ),
            )

    try:
        output = engine(job.input_spec)
    except CompositionEngineError as exc:
        return StageResult(
            job=job,
            next_state=JobState.FAILED,
            error=JobError(
                error_code=exc.code.value,
                message=exc.message,
                stage="composing",
            ),
        )
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

    # Successful compose: attach hashes for the manifest. Rebuilt from
    # the base version each time — a retried compose overwrites the
    # previous hashes instead of growing the string on every re-run.
    score_hash = output.notation_score.compute_hash()
    plan_hash = output.performance_plan.compute_hash()
    base_version = job.engine_version.split(";score_hash=", 1)[0]
    job.engine_version = f"{base_version};score_hash={score_hash[:12]};plan_hash={plan_hash[:12]}"
    # Persist the full engine output as a sidecar so later render
    # stages (audio, sheet, animation) can read it without re-running
    # the composer.
    from saimc.compose.serialization import write_engine_output

    sidecar_path = storage.job_dir(job.job_id) / "engine_output.json"
    write_engine_output(sidecar_path, output)
    # The next-state decision (continue to validating or fail with
    # lint_unpassable) is made here. Lint is also re-checked inside the
    # engine, but we re-run here defensively in case a future engine
    # raises before lint.
    from saimc.compose.linter import lint

    lint_report = lint(output.notation_score)
    if not lint_report.passed:
        return StageResult(
            job=job,
            next_state=JobState.FAILED,
            error=JobError(
                error_code="lint_unpassable",
                message=f"engine output failed lint: {[i.code.value for i in lint_report.issues]}",
                stage="validating",
            ),
        )
    return StageResult(job=job, next_state=JobState.VALIDATING)


def render_audio_stage(
    job: Job,
    storage: JobStorage,
    *,
    soundfont_path: Path | None = None,
    fluidsynth_bin: str | None = None,
    ffmpeg_bin: str | None = None,
) -> StageResult:
    """Run the `rendering_audio` stage.

    Loads the engine output sidecar, calls `render_audio`, and
    attaches one or two `ArtifactRecord` entries to the job (WAV
    primary, OGG Opus secondary when encoding succeeded).

    `soundfont_path` overrides the per-instrument resolution (handy in
    tests); otherwise `soundfont_for_instrument()` resolves the SF2 —
    `$SAIMC_SOUNDFONT_<INSTRUMENT>` or the Phase 1 Salamander default.
    `fluidsynth_bin` and `ffmpeg_bin`
    default to the env vars `$SAIMC_FLUIDSYNTH_BIN` /
    `$SAIMC_FFMPEG_BIN`, falling back to PATH lookup inside
    `render_audio`.

    Module-level so tests can monkeypatch it.
    """
    from saimc.compose.score import VOICE_BASS, VOICE_MELODY, VOICE_PERCUSSION
    from saimc.compose.serialization import read_engine_output
    from saimc.render.audio import AudioRenderError, render_audio
    from saimc.render.instruments import accompaniment_for, soundfont_for_instrument

    if job.input_spec is None:
        return StageResult(
            job=job,
            next_state=JobState.FAILED,
            error=JobError(
                error_code="no_spec",
                message="Cannot render audio without a parsed spec.",
                stage="rendering_audio",
            ),
        )

    sidecar_path = storage.job_dir(job.job_id) / "engine_output.json"
    if not sidecar_path.exists():
        return StageResult(
            job=job,
            next_state=JobState.FAILED,
            error=JobError(
                error_code="engine_output_missing",
                message=f"Engine output sidecar not found at {sidecar_path}",
                stage="rendering_audio",
            ),
        )

    output = read_engine_output(sidecar_path)

    # Phase 1's spec carries one instrumentation for the whole piece;
    # the melody voice plays it and the accompaniment voice gets its
    # complementary patch (per-voice orchestration arrives in Phase 2).
    # Dedicated-font instruments keep their own preset on both voices —
    # only one font loads per job — and separate via channel gain/pan.
    # A drum-set piece is the other exception: the piano stays as the
    # accompaniment and the engine's percussion voice (voice 2, GM
    # channel-10 keys) carries the kit — its notes sound as drums
    # precisely because they go to channel 10.
    instrumentation = job.input_spec.instrumentation
    if instrumentation == "drum_set":
        voice_instruments = {
            VOICE_BASS: "piano",
            VOICE_MELODY: "piano",
            VOICE_PERCUSSION: instrumentation,
        }
    else:
        voice_instruments = {
            VOICE_BASS: accompaniment_for(instrumentation),
            VOICE_MELODY: instrumentation,
        }
    sf = soundfont_path or soundfont_for_instrument(instrumentation)
    artifacts_dir = storage.ensure_artifact_dir(job.job_id)

    try:
        artifact = render_audio(
            output.performance_plan,
            bpm=output.arrangement.tempo_bpm,
            soundfont_path=sf,
            out_dir=artifacts_dir,
            job_id=job.job_id,
            voice_instruments=voice_instruments,
            fluidsynth_bin=fluidsynth_bin,
            ffmpeg_bin=ffmpeg_bin,
        )
    except AudioRenderError as exc:
        return StageResult(
            job=job,
            next_state=JobState.FAILED,
            error=JobError(
                error_code=exc.code,
                message=exc.message,
                stage="rendering_audio",
            ),
        )

    audio_toolchain = {
        "engine": "fluidsynth",
        "version": artifact.fluidsynth_version,
        "build_sha": artifact.fluidsynth_build_sha,
        "soundfont": artifact.soundfont_name,
        "soundfont_sha256": artifact.soundfont_sha256,
        "ffmpeg_version": artifact.ffmpeg_version,
        "ffmpeg_build_sha": artifact.ffmpeg_build_sha,
        "ffmpeg_configuration": artifact.ffmpeg_configuration,
    }
    storage.attach_artifact(
        job,
        ArtifactRecord(
            kind="audio",
            container=artifact.primary_container,
            codec=artifact.primary_codec,
            path=artifact.primary_path.name,
            sha256=artifact.primary_sha256,
            size_bytes=artifact.primary_size_bytes,
            toolchain=audio_toolchain,
        ),
    )
    if artifact.ogg_path is not None and artifact.ogg_path.exists():
        storage.attach_artifact(
            job,
            ArtifactRecord(
                # A second kind, not a second "audio" record: the
                # kind-keyed dict holds one record per key, and the
                # streamable primary must keep `kind="audio"` for
                # `GET /jobs/{id}/artifact/audio`.
                kind="audio_ogg",
                container="ogg",
                codec=artifact.ogg_codec or "opus",
                path=artifact.ogg_path.name,
                sha256=artifact.ogg_sha256 or "",
                size_bytes=artifact.ogg_size_bytes or 0,
                toolchain=audio_toolchain,
            ),
        )
    return StageResult(job=job, next_state=JobState.RENDERING_SHEET)


def render_sheet_stage(job: Job, storage: JobStorage) -> StageResult:
    """Run the `rendering_sheet` stage.

    Loads the engine output sidecar, exports the NotationScore to
    MusicXML, calls the render-service CLI for the SVG, and attaches
    the artifact (kind="sheet") to the job.
    """
    from saimc.compose.serialization import read_engine_output
    from saimc.render.sheet import SheetRenderError, notation_score_to_musicxml, render_sheet

    if job.input_spec is None:
        return StageResult(
            job=job,
            next_state=JobState.FAILED,
            error=JobError(
                error_code="no_spec",
                message="Cannot render sheet without a parsed spec.",
                stage="rendering_sheet",
            ),
        )

    sidecar_path = storage.job_dir(job.job_id) / "engine_output.json"
    if not sidecar_path.exists():
        return StageResult(
            job=job,
            next_state=JobState.FAILED,
            error=JobError(
                error_code="engine_output_missing",
                message=f"Engine output sidecar not found at {sidecar_path}",
                stage="rendering_sheet",
            ),
        )

    output = read_engine_output(sidecar_path)
    artifacts_dir = storage.ensure_artifact_dir(job.job_id)

    musicxml_path = artifacts_dir / "score.musicxml"
    try:
        musicxml_path.write_text(
            notation_score_to_musicxml(output.notation_score), encoding="utf-8"
        )
        artifact = render_sheet(musicxml_path, artifacts_dir / "sheet.svg")
    except (SheetRenderError, ValueError, OSError) as exc:
        code = exc.code if isinstance(exc, SheetRenderError) else "sheet_export_failed"
        return StageResult(
            job=job,
            next_state=JobState.FAILED,
            error=JobError(
                error_code=code,
                message=str(exc),
                stage="rendering_sheet",
            ),
        )

    storage.attach_artifact(
        job,
        ArtifactRecord(
            kind="sheet",
            container=artifact.container,
            codec=artifact.codec,
            path=artifact.sheet_path.name,
            sha256=artifact.sha256,
            size_bytes=artifact.size_bytes,
            toolchain={
                "engine": "opensheetmusicdisplay",
                "version": artifact.osmd_version,
                "renderer": "render-service (headless Chromium)",
            },
        ),
    )
    return StageResult(job=job, next_state=JobState.RENDERING_ANIMATION)


def render_animation_stage(
    job: Job,
    storage: JobStorage,
    *,
    ffmpeg_bin: str | None = None,
) -> StageResult:
    """Run the `rendering_animation` stage.

    Loads the engine output sidecar, locates the audio stage's WAV
    (the animation's audio track), renders the piano-roll WebM, and
    attaches the artifact (kind="animation") to the job.
    """
    from saimc.compose.serialization import read_engine_output
    from saimc.render.animation import AnimationRenderError, render_animation

    if job.input_spec is None:
        return StageResult(
            job=job,
            next_state=JobState.FAILED,
            error=JobError(
                error_code="no_spec",
                message="Cannot render animation without a parsed spec.",
                stage="rendering_animation",
            ),
        )

    sidecar_path = storage.job_dir(job.job_id) / "engine_output.json"
    if not sidecar_path.exists():
        return StageResult(
            job=job,
            next_state=JobState.FAILED,
            error=JobError(
                error_code="engine_output_missing",
                message=f"Engine output sidecar not found at {sidecar_path}",
                stage="rendering_animation",
            ),
        )

    output = read_engine_output(sidecar_path)
    artifacts_dir = storage.ensure_artifact_dir(job.job_id)

    audio_record = job.artifacts.get("audio")
    if audio_record is None:
        return StageResult(
            job=job,
            next_state=JobState.FAILED,
            error=JobError(
                error_code="audio_artifact_missing",
                message="Animation needs the audio stage's WAV, but no audio artifact is attached.",
                stage="rendering_animation",
            ),
        )
    audio_wav_path = artifacts_dir / audio_record.path

    try:
        artifact = render_animation(
            output.performance_plan,
            audio_wav_path=audio_wav_path,
            out_dir=artifacts_dir,
            ffmpeg_bin=ffmpeg_bin,
        )
    except AnimationRenderError as exc:
        return StageResult(
            job=job,
            next_state=JobState.FAILED,
            error=JobError(
                error_code=exc.code,
                message=exc.message,
                stage="rendering_animation",
            ),
        )

    storage.attach_artifact(
        job,
        ArtifactRecord(
            kind="animation",
            container=artifact.container,
            codec=artifact.codec,
            path=artifact.webm_path.name,
            sha256=artifact.sha256,
            size_bytes=artifact.size_bytes,
            toolchain={
                "engine": "ffmpeg",
                "version": artifact.ffmpeg_version,
                "build_sha": artifact.ffmpeg_build_sha,
                "configuration": artifact.ffmpeg_configuration,
                "video_codec": artifact.codec,
                "audio_codec": artifact.audio_codec,
                "dimensions": f"{artifact.width}x{artifact.height}",
                "fps": str(artifact.fps),
            },
        ),
    )
    return StageResult(job=job, next_state=JobState.COMPLETE)


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
    "render_animation_stage",
    "render_audio_stage",
    "render_sheet_stage",
    "safe_run",
    "transition_to",
]
