"""Tests for the `render_audio_stage` worker helper."""

from __future__ import annotations

import hashlib
import wave
from pathlib import Path

import pytest

from saimc.jobs.stages import render_audio_stage
from saimc.jobs.state import JobState
from saimc.jobs.storage import JobStorage
from saimc.spec import CompositionSpec, Mood


def _write_soundfont(path: Path) -> None:
    """A non-empty placeholder; the renderer's existence check only."""
    path.write_bytes(b"RIFF")


def _write_wav(path: Path) -> None:
    """A minimal valid 1-sample 16-bit mono WAV at 44.1 kHz."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(b"\x00\x00")


def _spec() -> CompositionSpec:
    return CompositionSpec(mood=Mood.CALMING, seed=42, duration_seconds=180)


def test_no_spec_returns_failed(tmp_path: Path) -> None:
    store = JobStorage(tmp_path)
    job = store.create("job1")
    # No input_spec attached.
    result = render_audio_stage(job, store)
    assert result.next_state == JobState.FAILED
    assert result.error is not None
    assert result.error.error_code == "no_spec"
    assert result.error.stage == "rendering_audio"


def test_missing_sidecar_returns_failed(tmp_path: Path) -> None:
    store = JobStorage(tmp_path)
    job = store.create("job1")
    job.input_spec = _spec()
    result = render_audio_stage(job, store)
    assert result.next_state == JobState.FAILED
    assert result.error is not None
    assert result.error.error_code == "engine_output_missing"


def test_advances_and_attaches_wav(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Happy path: sidecar + spec on disk, mocked render_audio, artifact attached."""
    from saimc.compose.engine import compose
    from saimc.compose.serialization import write_engine_output

    store = JobStorage(tmp_path)
    job = store.create("job1")
    job.input_spec = _spec()
    # Compose and write the sidecar.
    out = compose(job.input_spec)
    write_engine_output(store.job_dir(job.job_id) / "engine_output.json", out)
    # Move the job to VALIDATING and persist (the worker does this
    # before calling render_audio_stage).
    from saimc.jobs.stages import transition_to as _transition_to

    _transition_to(job, JobState.PARSING)
    _transition_to(job, JobState.COMPOSING)
    _transition_to(job, JobState.VALIDATING)
    store.save(job)
    job = store.get(job.job_id)
    assert (store.job_dir(job.job_id) / "engine_output.json").exists()

    # Now patch render_audio to a deterministic stub.
    expected_wav = tmp_path / "audio.wav"
    expected_ogg = tmp_path / "audio.ogg"
    _write_wav(expected_wav)
    expected_ogg.write_bytes(b"OggS")

    def _fake_render_audio(plan, bpm, soundfont_path, out_dir, job_id, **_kwargs):  # type: ignore[no-untyped-def]
        # The renderer must write into the supplied out_dir.
        wav_path = out_dir / "audio.wav"
        ogg_path = out_dir / "audio.ogg"
        _write_wav(wav_path)
        ogg_path.write_bytes(b"OggS")

        from saimc.render.audio import AudioArtifact

        return AudioArtifact(
            primary_path=wav_path,
            primary_container="wav",
            primary_codec="pcm_s16le",
            primary_sha256=hashlib.sha256(wav_path.read_bytes()).hexdigest(),
            primary_size_bytes=wav_path.stat().st_size,
            ogg_path=ogg_path,
            ogg_codec="opus",
            ogg_sha256=hashlib.sha256(ogg_path.read_bytes()).hexdigest(),
            ogg_size_bytes=ogg_path.stat().st_size,
            fluidsynth_version="stub",
            ffmpeg_version="stub",
        )

    monkeypatch.setattr("saimc.render.audio.render_audio", _fake_render_audio)

    result = render_audio_stage(job, store)
    assert result.error is None
    assert result.next_state == JobState.RENDERING_SHEET

    # The worker persists the job after the stage returns; mirror that
    # here so the reload below sees the attached artifacts.
    store.save(job)
    job = store.get(job.job_id)

    # Two artifacts: WAV primary (`audio`) + OGG Opus secondary
    # (`audio_ogg`) — separate kinds, since the artifact map is keyed by
    # kind and the streamable primary must stay `audio`.
    audio = job.artifacts["audio"]
    assert audio.container == "wav"
    assert audio.codec == "pcm_s16le"
    assert audio.sha256 == hashlib.sha256(expected_wav.read_bytes()).hexdigest()

    ogg = job.artifacts["audio_ogg"]
    assert ogg.container == "ogg"
    assert ogg.codec == "opus"
    assert ogg.sha256 == hashlib.sha256(expected_ogg.read_bytes()).hexdigest()


def test_audio_error_returns_failed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """An AudioRenderError surfaces with the right error code."""
    from saimc.render.audio import AudioRenderError, AudioRenderErrorCode

    store = JobStorage(tmp_path)
    job = store.create("job1")
    job.input_spec = _spec()

    # Manually write a sidecar so the stage can read it.
    from saimc.compose.engine import compose
    from saimc.compose.serialization import write_engine_output

    out = compose(job.input_spec)
    write_engine_output(store.job_dir(job.job_id) / "engine_output.json", out)

    def _fake_fail(plan, bpm, soundfont_path, out_dir, job_id, **_kwargs):  # type: ignore[no-untyped-def]
        raise AudioRenderError(
            code=AudioRenderErrorCode.SOUNDFONT_MISSING,
            message="soundfont not found",
        )

    monkeypatch.setattr("saimc.render.audio.render_audio", _fake_fail)

    result = render_audio_stage(job, store)
    assert result.next_state == JobState.FAILED
    assert result.error is not None
    assert result.error.error_code == AudioRenderErrorCode.SOUNDFONT_MISSING
    assert result.error.stage == "rendering_audio"
