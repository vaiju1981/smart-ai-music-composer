"""Tests for the animation renderer (piano-roll frames + WebM encode)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from saimc.compose.score import PerformanceNoteEvent, PerformancePlan
from saimc.render.animation import (
    AnimationRenderError,
    AnimationRenderErrorCode,
    encode_webm,
    render_animation,
    render_frames,
    total_frames,
)


def _plan() -> PerformancePlan:
    """Two notes over ~1.5s: one at t=0, one at t=1s."""
    return PerformancePlan.make(
        sample_rate=44100,
        notes=[
            PerformanceNoteEvent(
                voice_id=0, pitch_midi=60, start_us=0, duration_us=500_000, velocity=64
            ),
            PerformanceNoteEvent(
                voice_id=1, pitch_midi=72, start_us=1_000_000, duration_us=500_000, velocity=80
            ),
        ],
    )


def test_total_frames_counts_one_extra_settled_frame() -> None:
    assert total_frames(1.0, 24) == 25
    assert total_frames(0.0, 24) == 1


class TestRenderFrames:
    def test_writes_one_png_per_frame(self, tmp_path: Path) -> None:
        frames_dir = tmp_path / "frames"
        count = render_frames(_plan(), frames_dir, fps=24)
        assert count == 37  # 1.5s of notes at 24fps -> 36 frames + the settled one
        assert len(list(frames_dir.glob("*.png"))) == count
        assert (frames_dir / "000000.png").is_file()

    def test_empty_plan_raises(self, tmp_path: Path) -> None:
        empty = PerformancePlan.make(sample_rate=44100, notes=[])
        with pytest.raises(AnimationRenderError) as exc_info:
            render_frames(empty, tmp_path / "frames")
        assert exc_info.value.code == AnimationRenderErrorCode.FRAME_RENDER_FAILED

    def test_note_appears_in_first_frame_at_playhead(self, tmp_path: Path) -> None:
        """The note starting at t=0 sits under the playhead in frame 0."""
        from PIL import Image

        frames_dir = tmp_path / "frames"
        render_frames(_plan(), frames_dir, fps=24)
        img = Image.open(frames_dir / "000000.png")
        # Playhead x for the defaults: 1280 * 0.25 = 320. The pitch-60
        # note's lane starts at y=700 (720 minus the 20px bottom margin)
        # and the note spans x 320..344. Sample inside that rectangle.
        pixel = img.getpixel((324, 705))
        assert pixel == (86, 156, 214)


class TestEncodeWebm:
    def test_ffmpeg_missing_raises(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF")
        # The audio module's find_ffmpeg raises AudioRenderError; the
        # animation renderer re-raises it under its own error type with
        # the code intact (the machine may have a PATH ffmpeg, so patch
        # the lookup rather than pointing it at a bogus path).
        from saimc.render.audio import AudioRenderError, AudioRenderErrorCode

        def _raise_missing(_explicit: str | None = None) -> str:
            raise AudioRenderError(
                AudioRenderErrorCode.FFMPEG_MISSING,
                "ffmpeg not found",
            )

        monkeypatch.setattr("saimc.render.animation.find_ffmpeg", _raise_missing)
        with pytest.raises(AnimationRenderError) as exc_info:
            encode_webm(frames_dir, audio, tmp_path / "out.webm")
        assert exc_info.value.code == AnimationRenderErrorCode.FFMPEG_MISSING

    def test_audit_failure_raises(self, tmp_path: Path) -> None:
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF")
        with (
            patch("saimc.render.animation.find_ffmpeg", return_value="ffmpeg"),
            patch("saimc.render.animation.audit_ffmpeg") as mock_audit,
        ):
            mock_audit.return_value.ok = False
            mock_audit.return_value.reasons = ("missing --enable-libvpx",)
            with pytest.raises(AnimationRenderError) as exc_info:
                encode_webm(frames_dir, audio, tmp_path / "out.webm")
        assert exc_info.value.code == AnimationRenderErrorCode.FFMPEG_AUDIT_FAILED

    def test_missing_audio_raises(self, tmp_path: Path) -> None:
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        with (
            patch("saimc.render.animation.find_ffmpeg", return_value="ffmpeg"),
            patch("saimc.render.animation.audit_ffmpeg") as mock_audit,
        ):
            mock_audit.return_value.ok = True
            mock_audit.return_value.version = "7.0"
            mock_audit.return_value.binary_sha256 = "a" * 64
            with pytest.raises(AnimationRenderError) as exc_info:
                encode_webm(frames_dir, tmp_path / "missing.wav", tmp_path / "out.webm")
        assert exc_info.value.code == AnimationRenderErrorCode.AUDIO_MISSING

    def test_encode_failure_surfaces_stderr(self, tmp_path: Path) -> None:
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF")

        class _Proc:
            returncode = 1
            stdout = ""
            stderr = "Encoder init failed\n"

        with (
            patch("saimc.render.animation.find_ffmpeg", return_value="ffmpeg"),
            patch("saimc.render.animation.audit_ffmpeg") as mock_audit,
            patch("saimc.render.animation.safe_run", return_value=_Proc()),
        ):
            mock_audit.return_value.ok = True
            mock_audit.return_value.version = "7.0"
            mock_audit.return_value.binary_sha256 = "a" * 64
            with pytest.raises(AnimationRenderError) as exc_info:
                encode_webm(frames_dir, audio, tmp_path / "out.webm")
        assert exc_info.value.code == AnimationRenderErrorCode.FFMPEG_FAILED
        assert "Encoder init failed" in exc_info.value.message

    def test_success_invokes_ffmpeg_with_expected_args(self, tmp_path: Path) -> None:
        frames_dir = tmp_path / "frames"
        frames_dir.mkdir()
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF")
        out = tmp_path / "out.webm"
        seen_cmds: list[list[str]] = []

        def _fake_safe_run(cmd: list[str], *, timeout_s: float, **_kw: Any) -> Any:
            seen_cmds.append(cmd)
            out.write_bytes(b"\x1aE\xdf\xa3webm")
            return type("Proc", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        with (
            patch("saimc.render.animation.find_ffmpeg", return_value="ffmpeg"),
            patch("saimc.render.animation.audit_ffmpeg") as mock_audit,
            patch("saimc.render.animation.safe_run", side_effect=_fake_safe_run),
        ):
            mock_audit.return_value.ok = True
            mock_audit.return_value.version = "7.0-stub"
            mock_audit.return_value.binary_sha256 = "b" * 64
            version, _sha = encode_webm(frames_dir, audio, out)

        assert version == "7.0-stub"
        assert out.is_file()
        cmd = seen_cmds[0]
        assert cmd[0] == "ffmpeg"
        assert cmd[cmd.index("-c:v") + 1] == "libvpx-vp9"
        assert cmd[cmd.index("-c:a") + 1] == "libopus"
        assert cmd[-1] == str(out)


class TestRenderAnimation:
    def test_end_to_end_writes_webm_and_cleans_frames(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr("saimc.render.animation.find_ffmpeg", lambda _=None: "ffmpeg")

        class _Audit:
            ok = True
            version = "7.0-stub"
            binary_sha256 = "b" * 64
            reasons: tuple[str, ...] = ()

        monkeypatch.setattr("saimc.render.animation.audit_ffmpeg", lambda *_a, **_k: _Audit())

        def _fake_safe_run(cmd: list[str], *, timeout_s: float, **_kw: Any) -> Any:
            Path(cmd[-1]).write_bytes(b"\x1aE\xdf\xa3webm")
            return type("Proc", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        monkeypatch.setattr("saimc.render.animation.safe_run", _fake_safe_run)

        out_dir = tmp_path / "artifacts"
        audio_wav = tmp_path / "a.wav"
        audio_wav.write_bytes(b"RIFF")
        artifact = render_animation(_plan(), audio_wav_path=audio_wav, out_dir=out_dir)

        assert artifact.container == "webm"
        assert artifact.codec == "vp9"
        assert artifact.audio_codec == "opus"
        assert artifact.width == 1280
        assert artifact.height == 720
        assert artifact.fps == 24
        assert artifact.ffmpeg_version == "7.0-stub"
        assert artifact.ffmpeg_build_sha == "b" * 64

        webm = out_dir / "animation.webm"
        assert webm.is_file()
        assert artifact.sha256 == hashlib.sha256(webm.read_bytes()).hexdigest()
        assert artifact.size_bytes == webm.stat().st_size
        # Frames are intermediate: cleaned up after the encode.
        assert not (out_dir / "frames").exists()

    def test_nonzero_rc_raises(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr("saimc.render.animation.find_ffmpeg", lambda _=None: "ffmpeg")

        class _Audit:
            ok = True
            version = "7.0-stub"
            binary_sha256 = "b" * 64
            reasons: tuple[str, ...] = ()

        monkeypatch.setattr("saimc.render.animation.audit_ffmpeg", lambda *_a, **_k: _Audit())

        def _fake_safe_run(cmd: list[str], *, timeout_s: float, **_kw: Any) -> Any:
            return type("Proc", (), {"returncode": 1, "stdout": "", "stderr": "boom\n"})()

        monkeypatch.setattr("saimc.render.animation.safe_run", _fake_safe_run)

        audio_wav = tmp_path / "a.wav"
        audio_wav.write_bytes(b"RIFF")
        with pytest.raises(AnimationRenderError) as exc_info:
            render_animation(_plan(), audio_wav_path=audio_wav, out_dir=tmp_path)
        assert exc_info.value.code == AnimationRenderErrorCode.FFMPEG_FAILED
