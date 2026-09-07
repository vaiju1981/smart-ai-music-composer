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
    VOICE_COLORS,
    build_roll,
    encode_webm,
    iter_roll_frames,
    render_animation,
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


class TestBuildRoll:
    def test_empty_plan_raises(self) -> None:
        empty = PerformancePlan.make(sample_rate=44100, notes=[])
        with pytest.raises(AnimationRenderError) as exc_info:
            build_roll(empty)
        assert exc_info.value.code == AnimationRenderErrorCode.FRAME_RENDER_FAILED

    def test_note_appears_in_roll_at_playhead_offset(self) -> None:
        """A note at t=0 is drawn at x = playhead (1280 * 0.25 = 320)."""
        from PIL import Image

        roll = build_roll(_plan())
        assert isinstance(roll, Image.Image)
        # The pitch-60 note's lane starts at y=700 (720 minus the 20px
        # bottom margin) and the note spans x 320..344 at 48 px/s.
        pixel = roll.getpixel((324, 705))
        assert pixel == (86, 156, 214)

    def test_each_voice_gets_its_own_colour(self) -> None:
        """Bass/melody/kit/harmony lanes read as four instruments."""
        plan = PerformancePlan.make(
            sample_rate=44100,
            notes=[
                PerformanceNoteEvent(
                    voice_id=voice_id,
                    pitch_midi=60 + 12 * voice_id,
                    start_us=0,
                    duration_us=500_000,
                    velocity=64,
                )
                for voice_id in range(len(VOICE_COLORS))
            ],
        )
        roll = build_roll(plan)
        lane_height = max(8, (720 - 40) // 37)
        for voice_id, color in enumerate(VOICE_COLORS):
            lane_y = 700 - 12 * voice_id * lane_height
            assert roll.getpixel((324, lane_y + 2)) == color

    def test_roll_is_wide_enough_for_the_last_frame(self) -> None:
        roll = build_roll(_plan())
        # 1.5s * 48 px/s + one full frame width of lead-in/out.
        assert roll.size[0] >= 1280 + 72
        assert roll.size[1] == 720


class TestIterRollFrames:
    def test_yields_one_frame_per_tick(self) -> None:
        roll = build_roll(_plan())
        frames = list(iter_roll_frames(roll, count=total_frames(1.5, 24)))
        assert len(frames) == 37  # 36 frames + the settled one
        # Each frame is one raw 1280x720 RGB buffer.
        assert len(frames[0]) == 1280 * 720 * 3

    def test_playhead_is_fixed_in_every_frame(self) -> None:
        from PIL import Image

        roll = build_roll(_plan())
        frames = list(iter_roll_frames(roll, count=2))
        for raw in frames:
            img = Image.frombytes("RGB", (1280, 720), raw)
            assert img.getpixel((320, 100)) == (232, 232, 232)
            # The playhead column equals the crop origin: frame 1 has
            # advanced exactly 2 px (48 px/s / 24 fps).
        first = Image.frombytes("RGB", (1280, 720), frames[0])
        second = Image.frombytes("RGB", (1280, 720), frames[1])
        assert first.getpixel((324, 705)) == (86, 156, 214)  # t=0 note
        assert second.getpixel((322, 705)) == (86, 156, 214)  # scrolled 2px


class TestEncodeWebm:
    def test_ffmpeg_missing_raises(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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
            encode_webm(iter(()), audio, tmp_path / "out.webm")
        assert exc_info.value.code == AnimationRenderErrorCode.FFMPEG_MISSING

    def test_audit_failure_raises(self, tmp_path: Path) -> None:
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF")
        with (
            patch("saimc.render.animation.find_ffmpeg", return_value="ffmpeg"),
            patch("saimc.render.animation.audit_ffmpeg") as mock_audit,
        ):
            mock_audit.return_value.ok = False
            mock_audit.return_value.reasons = ("missing --enable-libvpx",)
            with pytest.raises(AnimationRenderError) as exc_info:
                encode_webm(iter(()), audio, tmp_path / "out.webm")
        assert exc_info.value.code == AnimationRenderErrorCode.FFMPEG_AUDIT_FAILED

    def test_missing_audio_raises(self, tmp_path: Path) -> None:
        with (
            patch("saimc.render.animation.find_ffmpeg", return_value="ffmpeg"),
            patch("saimc.render.animation.audit_ffmpeg") as mock_audit,
        ):
            mock_audit.return_value.ok = True
            mock_audit.return_value.version = "7.0"
            mock_audit.return_value.binary_sha256 = "a" * 64
            with pytest.raises(AnimationRenderError) as exc_info:
                encode_webm(iter(()), tmp_path / "missing.wav", tmp_path / "out.webm")
        assert exc_info.value.code == AnimationRenderErrorCode.AUDIO_MISSING

    def test_encode_failure_surfaces_stderr(self, tmp_path: Path) -> None:
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF")

        class _Proc:
            returncode = 1

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                import io

                self.stdin = io.BytesIO()
                self.stderr = io.BytesIO(b"Encoder init failed\n")

            def wait(self, timeout: float | None = None) -> int:
                return self.returncode

        with (
            patch("saimc.render.animation.find_ffmpeg", return_value="ffmpeg"),
            patch("saimc.render.animation.audit_ffmpeg") as mock_audit,
            patch("saimc.render.animation.subprocess.Popen", return_value=_Proc()),
        ):
            mock_audit.return_value.ok = True
            mock_audit.return_value.version = "7.0"
            mock_audit.return_value.binary_sha256 = "a" * 64
            with pytest.raises(AnimationRenderError) as exc_info:
                encode_webm(iter(()), audio, tmp_path / "out.webm")
        assert exc_info.value.code == AnimationRenderErrorCode.FFMPEG_FAILED
        assert "Encoder init failed" in exc_info.value.message

    def test_success_streams_rawvideo_and_sets_vp9_flags(self, tmp_path: Path) -> None:
        audio = tmp_path / "audio.wav"
        audio.write_bytes(b"RIFF")
        out = tmp_path / "out.webm"
        seen_cmds: list[list[str]] = []
        written: list[bytes] = []

        class _Proc:
            returncode = 0

            def __init__(self, cmd: list[str], **_kw: Any) -> None:
                import io

                seen_cmds.append(cmd)
                self.stdin = _Writer(written)
                self.stderr = io.BytesIO(b"")

            def wait(self, timeout: float | None = None) -> int:
                return self.returncode

        class _Writer:
            def __init__(self, sink: list[bytes]) -> None:
                self._sink = sink
                self.closed = False

            def write(self, data: bytes) -> int:
                self._sink.append(data)
                return len(data)

            def close(self) -> None:
                self.closed = True

        def _popen(cmd: list[str], **_kw: Any) -> Any:
            out.write_bytes(b"\x1aE\xdf\xa3webm")
            return _Proc(cmd)

        with (
            patch("saimc.render.animation.find_ffmpeg", return_value="ffmpeg"),
            patch("saimc.render.animation.audit_ffmpeg") as mock_audit,
            patch("saimc.render.animation.subprocess.Popen", side_effect=_popen),
        ):
            mock_audit.return_value.ok = True
            mock_audit.return_value.version = "7.0-stub"
            mock_audit.return_value.binary_sha256 = "b" * 64
            mock_audit.return_value.configuration_line = "--enable-libvpx"
            version, _sha, _config = encode_webm(
                iter([b"\x00" * 16]), audio, out, width=1280, height=720
            )

        assert version == "7.0-stub"
        assert out.is_file()
        cmd = seen_cmds[0]
        assert cmd[0] == "ffmpeg"
        assert cmd[cmd.index("-f") + 1] == "rawvideo"
        assert cmd[cmd.index("-s") + 1] == "1280x720"
        assert cmd[cmd.index("-c:v") + 1] == "libvpx-vp9"
        assert cmd[cmd.index("-deadline") + 1] == "good"
        assert cmd[cmd.index("-cpu-used") + 1] == "4"
        assert cmd[cmd.index("-c:a") + 1] == "libopus"
        assert cmd[-1] == str(out)
        # The frame bytes reached ffmpeg's stdin.
        assert written == [b"\x00" * 16]


class TestRenderAnimation:
    def test_end_to_end_writes_webm(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr("saimc.render.animation.find_ffmpeg", lambda _=None: "ffmpeg")

        class _Audit:
            ok = True
            version = "7.0-stub"
            binary_sha256 = "b" * 64
            configuration_line = "--enable-libvpx --enable-libopus"
            reasons: tuple[str, ...] = ()

        monkeypatch.setattr("saimc.render.animation.audit_ffmpeg", lambda *_a, **_k: _Audit())

        def _fake_popen(cmd: list[str], **_kw: Any) -> Any:
            Path(cmd[-1]).write_bytes(b"\x1aE\xdf\xa3webm")

            class _P:
                returncode = 0

                def __init__(self) -> None:
                    import io

                    self.stdin = _DevNull()
                    self.stderr = io.BytesIO(b"")

                def wait(self, timeout: float | None = None) -> int:
                    return self.returncode

            class _DevNull:
                closed = False

                def write(self, data: bytes) -> int:
                    return len(data)

                def close(self) -> None:
                    self.closed = True

            return _P()

        monkeypatch.setattr("saimc.render.animation.subprocess.Popen", _fake_popen)

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
        # No intermediate frame directory exists any more: frames go
        # straight through the pipe.
        assert not (out_dir / "frames").exists()

    def test_nonzero_rc_raises(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr("saimc.render.animation.find_ffmpeg", lambda _=None: "ffmpeg")

        class _Audit:
            ok = True
            version = "7.0-stub"
            binary_sha256 = "b" * 64
            configuration_line = "--enable-libvpx --enable-libopus"
            reasons: tuple[str, ...] = ()

        monkeypatch.setattr("saimc.render.animation.audit_ffmpeg", lambda *_a, **_k: _Audit())

        def _fake_popen(cmd: list[str], **_kw: Any) -> Any:
            import io

            class _P:
                returncode = 1

                def __init__(self) -> None:
                    self.stdin = _DevNull()
                    self.stderr = io.BytesIO(b"boom\n")

                def wait(self, timeout: float | None = None) -> int:
                    return self.returncode

            class _DevNull:
                closed = False

                def write(self, data: bytes) -> int:
                    return len(data)

                def close(self) -> None:
                    self.closed = True

            return _P()

        monkeypatch.setattr("saimc.render.animation.subprocess.Popen", _fake_popen)

        audio_wav = tmp_path / "a.wav"
        audio_wav.write_bytes(b"RIFF")
        with pytest.raises(AnimationRenderError) as exc_info:
            render_animation(_plan(), audio_wav_path=audio_wav, out_dir=tmp_path / "o")
        assert exc_info.value.code == AnimationRenderErrorCode.FFMPEG_FAILED
        assert "boom" in exc_info.value.message
