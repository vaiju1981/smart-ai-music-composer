"""Tests for the `render_animation_stage` worker helper."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from saimc.jobs.stages import render_animation_stage, transition_to
from saimc.jobs.state import JobState
from saimc.jobs.storage import ArtifactRecord, JobStorage
from saimc.render.animation import AnimationRenderError, AnimationRenderErrorCode
from saimc.spec import CompositionSpec, Mood


def _spec() -> CompositionSpec:
    return CompositionSpec(mood=Mood.CALMING, seed=42, duration_seconds=180)


def _prepared_job(tmp_path: Path, *, with_audio: bool = True) -> tuple[JobStorage, Any]:
    """A job with a spec, engine-output sidecar, and audio artifact, in RENDERING_ANIMATION."""
    from saimc.compose.engine import compose
    from saimc.compose.serialization import write_engine_output

    store = JobStorage(tmp_path)
    job = store.create("job1")
    job.input_spec = _spec()
    output = compose(job.input_spec)
    write_engine_output(store.job_dir(job.job_id) / "engine_output.json", output)

    for state in (JobState.PARSING, JobState.COMPOSING, JobState.VALIDATING):
        transition_to(job, state)
    if with_audio:
        artifacts_dir = store.ensure_artifact_dir(job.job_id)
        (artifacts_dir / "audio.wav").write_bytes(b"RIFF")
        store.attach_artifact(
            job,
            ArtifactRecord(
                kind="audio",
                container="wav",
                codec="pcm_s24le",
                path="audio.wav",
                sha256=hashlib.sha256(b"RIFF").hexdigest(),
                size_bytes=4,
            ),
        )
    transition_to(job, JobState.RENDERING_AUDIO)
    transition_to(job, JobState.RENDERING_SHEET)
    transition_to(job, JobState.RENDERING_ANIMATION)
    store.save(job)
    return store, store.get(job.job_id)


def test_no_spec_returns_failed(tmp_path: Path) -> None:
    store = JobStorage(tmp_path)
    job = store.create("job1")
    result = render_animation_stage(job, store)
    assert result.next_state == JobState.FAILED
    assert result.error is not None
    assert result.error.error_code == "no_spec"
    assert result.error.stage == "rendering_animation"


def test_missing_sidecar_returns_failed(tmp_path: Path) -> None:
    store = JobStorage(tmp_path)
    job = store.create("job1")
    job.input_spec = _spec()
    result = render_animation_stage(job, store)
    assert result.next_state == JobState.FAILED
    assert result.error is not None
    assert result.error.error_code == "engine_output_missing"


def test_missing_audio_artifact_returns_failed(tmp_path: Path) -> None:
    store, job = _prepared_job(tmp_path, with_audio=False)
    result = render_animation_stage(job, store)
    assert result.next_state == JobState.FAILED
    assert result.error is not None
    assert result.error.error_code == "audio_artifact_missing"
    assert result.error.stage == "rendering_animation"


def test_advances_and_attaches_animation(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Happy path: sidecar + audio WAV on disk, mocked render, artifact attached."""
    store, job = _prepared_job(tmp_path)

    def _fake_render_animation(
        plan: Any, *, audio_wav_path: Path, out_dir: Path, **_kw: Any
    ) -> Any:
        from saimc.render.animation import AnimationArtifact

        webm_path = out_dir / "animation.webm"
        webm_path.write_bytes(b"\x1aE\xdf\xa3webm")
        return AnimationArtifact(
            webm_path=webm_path,
            container="webm",
            codec="vp9",
            audio_codec="opus",
            sha256=hashlib.sha256(b"\x1aE\xdf\xa3webm").hexdigest(),
            size_bytes=webm_path.stat().st_size,
            width=1280,
            height=720,
            fps=24,
            ffmpeg_version="stub",
            ffmpeg_build_sha="c" * 64,
        )

    monkeypatch.setattr("saimc.render.animation.render_animation", _fake_render_animation)

    result = render_animation_stage(job, store)
    assert result.error is None
    assert result.next_state == JobState.COMPLETE

    # The worker persists the job after the stage returns; mirror that
    # here so the reload below sees the attached artifact.
    store.save(job)
    job = store.get(job.job_id)
    animation = job.artifacts["animation"]
    assert animation.container == "webm"
    assert animation.codec == "vp9"
    assert animation.sha256 == hashlib.sha256(b"\x1aE\xdf\xa3webm").hexdigest()


def test_render_error_returns_failed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store, job = _prepared_job(tmp_path)

    def _fake_fail(plan: Any, **_kwargs: Any) -> Any:
        raise AnimationRenderError(
            code=AnimationRenderErrorCode.FFMPEG_AUDIT_FAILED,
            message="ffmpeg failed audit",
        )

    monkeypatch.setattr("saimc.render.animation.render_animation", _fake_fail)

    result = render_animation_stage(job, store)
    assert result.next_state == JobState.FAILED
    assert result.error is not None
    assert result.error.error_code == AnimationRenderErrorCode.FFMPEG_AUDIT_FAILED
    assert result.error.stage == "rendering_animation"
