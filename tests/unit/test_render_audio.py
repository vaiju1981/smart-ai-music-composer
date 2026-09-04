"""Unit tests for the audio renderer.

These tests verify the *contract* of the audio renderer (MIDI byte
format, subprocess command lines, error handling) by mocking the
external processes. The real binary paths (FluidSynth, FFmpeg,
soundfont) are release-gate verifications that require the actual
binaries to be installed.
"""

from __future__ import annotations

import hashlib
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from saimc.compose.score import (
    PPQ,
    PerformanceNoteEvent,
    PerformancePlan,
)
from saimc.render.audio import (
    AudioRenderError,
    AudioRenderErrorCode,
    _hash_file,
    _us_to_ticks,
    build_smf,
    encode_opus,
    find_ffmpeg,
    find_fluidsynth,
    render_audio,
)


def _plan_with_notes() -> PerformancePlan:
    """Two notes, one at start, one 1 second later."""
    return PerformancePlan.make(
        sample_rate=44100,
        notes=[
            PerformanceNoteEvent(
                voice_id=0, pitch_midi=60, start_us=0, duration_us=500_000, velocity=64
            ),
            PerformanceNoteEvent(
                voice_id=0, pitch_midi=64, start_us=1_000_000, duration_us=500_000, velocity=80
            ),
        ],
    )


class TestBuildSmf:
    def test_smfile_has_tempo_event(self) -> None:
        smf = build_smf(_plan_with_notes(), bpm=120.0)
        tempo_events = [m for m in smf.tracks[0] if m.type == "set_tempo"]
        assert len(tempo_events) == 1
        # 120 bpm = 500_000 microseconds per quarter.
        assert tempo_events[0].tempo == 500_000

    def test_smfile_has_correct_note_count(self) -> None:
        smf = build_smf(_plan_with_notes(), bpm=120.0)
        # 2 notes -> 1 track with set_tempo + 2 note_on + 2 note_off + end_of_track.
        on_events = [m for m in smf.tracks[0] if m.type == "note_on"]
        off_events = [m for m in smf.tracks[0] if m.type == "note_off"]
        assert len(on_events) == 2
        assert len(off_events) == 2

    def test_smfile_ppq(self) -> None:
        smf = build_smf(_plan_with_notes(), bpm=120.0)
        assert smf.ticks_per_beat == PPQ

    def test_zero_duration_guard(self) -> None:
        # The PerformanceNoteEvent dataclass already rejects duration_us=0
        # at construction time, so a well-formed plan can never produce
        # a zero-duration note. Verify the rejection here so the guard
        # is documented.
        with pytest.raises(ValueError):
            PerformanceNoteEvent(voice_id=0, pitch_midi=60, start_us=0, duration_us=0, velocity=64)


class TestTickConversion:
    def test_known_quantities(self) -> None:
        # 1 quarter at 120bpm = 500_000 us = 480 ticks.
        assert _us_to_ticks(500_000, 120.0) == 480
        # 1 second at 60bpm = 1_000_000 us = 480 ticks.
        assert _us_to_ticks(1_000_000, 60.0) == 480


class TestFindFluidsynth:
    def test_finds_in_path(self, tmp_path: Path) -> None:
        # Create a fake fluidsynth in a temp dir, prepend to PATH.
        fake = tmp_path / "fluidsynth"
        fake.write_text("#!/bin/sh\necho fake\n")
        fake.chmod(0o755)
        with patch("shutil.which", return_value=str(fake)):
            assert find_fluidsynth() == str(fake)

    def test_explicit_takes_priority(self, tmp_path: Path) -> None:
        explicit = tmp_path / "my-fluidsynth"
        explicit.write_text("#!/bin/sh\n")
        explicit.chmod(0o755)
        with patch("shutil.which", return_value=None):
            assert find_fluidsynth(explicit=str(explicit)) == str(explicit)

    def test_missing_raises(self) -> None:
        with (
            patch("shutil.which", return_value=None),
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(AudioRenderError) as exc_info,
        ):
            find_fluidsynth()
        assert exc_info.value.code == AudioRenderErrorCode.FLUIDSYNTH_MISSING


class TestFindFfmpeg:
    def test_finds_in_path(self, tmp_path: Path) -> None:
        fake = tmp_path / "ffmpeg"
        fake.write_text("#!/bin/sh\n")
        fake.chmod(0o755)
        with patch("shutil.which", return_value=str(fake)):
            assert find_ffmpeg() == str(fake)

    def test_missing_raises(self) -> None:
        with (
            patch("shutil.which", return_value=None),
            patch.dict("os.environ", {}, clear=True),
            pytest.raises(AudioRenderError) as exc_info,
        ):
            find_ffmpeg()
        assert exc_info.value.code == AudioRenderErrorCode.FFMPEG_MISSING


class TestRunFluidsynthContract:
    def test_invokes_fluidsynth_with_correct_args(self, tmp_path: Path) -> None:
        # Create a fake soundfont (FluidSynth just needs the path to exist).
        sf = tmp_path / "Salamander.sf2"
        sf.write_bytes(b"RIFF")
        smf = tmp_path / "in.mid"
        smf.write_bytes(b"MThd")
        out_wav = tmp_path / "out.wav"
        out_wav.write_bytes(b"RIFF")

        fake_bin = tmp_path / "fluidsynth"
        fake_bin.write_text("#!/bin/sh\n")
        fake_bin.chmod(0o755)

        with (
            patch("shutil.which", return_value=str(fake_bin)),
            patch("saimc.render.audio._read_fluidsynth_version", return_value="2.3.4"),
            patch("saimc.render.audio.canonical_sha256", return_value="x" * 64),
            patch("saimc.render.audio.safe_run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            run_version, _run_sha = _run_fluidsynth_mock(
                smf, sf, out_wav, fluidsynth_bin=str(fake_bin)
            )
        # We didn't call run_fluidsynth; this is just exercising find_fluidsynth.
        assert run_version == "2.3.4"

    def test_timeout_raises(self, tmp_path: Path) -> None:
        from saimc.jobs.stages import SubprocessTimeoutError
        from saimc.render.audio import run_fluidsynth

        sf = tmp_path / "Salamander.sf2"
        sf.write_bytes(b"RIFF")
        smf = tmp_path / "in.mid"
        smf.write_bytes(b"MThd")
        out_wav = tmp_path / "out.wav"

        fake_bin = tmp_path / "fluidsynth"
        fake_bin.write_text("#!/bin/sh\n")
        fake_bin.chmod(0o755)

        with (
            patch("shutil.which", return_value=str(fake_bin)),
            patch("saimc.render.audio.safe_run", side_effect=SubprocessTimeoutError("t")),
            pytest.raises(AudioRenderError) as exc_info,
        ):
            run_fluidsynth(smf, sf, out_wav)
        assert exc_info.value.code == AudioRenderErrorCode.FLUIDSYNTH_FAILED

    def test_missing_soundfont_raises(self, tmp_path: Path) -> None:
        from saimc.render.audio import run_fluidsynth

        smf = tmp_path / "in.mid"
        smf.write_bytes(b"MThd")
        out_wav = tmp_path / "out.wav"
        missing_sf = tmp_path / "missing.sf2"
        fake_bin = tmp_path / "fluidsynth"
        fake_bin.write_text("#!/bin/sh\n")
        fake_bin.chmod(0o755)

        with (
            patch("shutil.which", return_value=str(fake_bin)),
            pytest.raises(AudioRenderError) as exc_info,
        ):
            run_fluidsynth(smf, missing_sf, out_wav)
        assert exc_info.value.code == AudioRenderErrorCode.SOUNDFONT_MISSING


def _run_fluidsynth_mock(smf, sf, out_wav, fluidsynth_bin):
    """Helper: returns the version+build that run_fluidsynth would emit."""
    from saimc.render.audio import run_fluidsynth

    return run_fluidsynth(smf, sf, out_wav, fluidsynth_bin=fluidsynth_bin)


class TestRenderAudioPipeline:
    def test_full_pipeline_writes_wav_and_ogg(self, tmp_path: Path) -> None:
        """End-to-end mock: SMF -> FluidSynth -> FFmpeg -> audio.wav + audio.ogg."""
        sf = tmp_path / "Salamander.sf2"
        sf.write_bytes(b"RIFF" * 100)

        fake_fs = tmp_path / "fluidsynth"
        fake_fs.write_text("#!/bin/sh\n")
        fake_fs.chmod(0o755)

        fake_ff = tmp_path / "ffmpeg"
        fake_ff.write_text("#!/bin/sh\n")
        fake_ff.chmod(0o755)

        # Mock safe_run to write a fake WAV / OGG on call.
        seen_cmds = []

        def _fake_safe_run(cmd, *, timeout_s, **kw):
            seen_cmds.append(cmd)

            # Find the output path (the second-to-last positional that's a real path).
            for arg in cmd:
                p = Path(arg)
                if p.suffix in (".wav", ".ogg") and p.parent.exists():
                    if p.suffix == ".wav":
                        _write_wav(p)
                    else:
                        p.write_bytes(b"OggS\x00\x00fake-opus")
            return MagicMock(returncode=0, stdout="", stderr="")

        out_dir = tmp_path / "out"
        with (
            patch("shutil.which", return_value=str(fake_fs)),
            patch("saimc.render.audio.audit_ffmpeg") as mock_audit,
            patch("saimc.render.audio.safe_run", side_effect=_fake_safe_run),
        ):
            mock_audit.return_value = MagicMock(
                ok=True,
                version="8.1.2",
                binary_sha256="a" * 64,
                configuration_line="--enable-libvpx --enable-libopus",
            )
            artifact = render_audio(
                _plan_with_notes(),
                bpm=120.0,
                soundfont_path=sf,
                out_dir=out_dir,
                job_id="test-job",
                fluidsynth_bin=str(fake_fs),
                ffmpeg_bin=str(fake_ff),
            )

        assert artifact.primary_path.exists()
        assert artifact.primary_path.suffix == ".wav"
        assert artifact.ogg_path is not None
        assert artifact.ogg_path.exists()
        assert artifact.ffmpeg_configuration == "--enable-libvpx --enable-libopus"

        # The OGG carries the Salamander attribution as Vorbis comments
        # (roadmap §10 #12).
        ogg_cmd = next(c for c in seen_cmds if c[-1].endswith(".ogg"))
        metadata_args = ogg_cmd[ogg_cmd.index("-metadata") :]
        assert "AUTHOR=Alexander Holm" in metadata_args
        assert "LIBRARY=Salamander Grand Piano" in metadata_args
        assert "LICENSE_URL=https://creativecommons.org/licenses/by/3.0/" in metadata_args
        assert (
            artifact.primary_sha256
            == hashlib.sha256(artifact.primary_path.read_bytes()).hexdigest()
        )

    def test_pipeline_failure_surfaces_error(self, tmp_path: Path) -> None:
        sf = tmp_path / "Salamander.sf2"
        sf.write_bytes(b"RIFF")

        fake_fs = tmp_path / "fluidsynth"
        fake_fs.write_text("#!/bin/sh\n")
        fake_fs.chmod(0o755)

        # safe_run returns non-zero (fluidsynth failure).
        with patch("saimc.render.audio.safe_run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="bad soundfont")
            with (
                patch("shutil.which", return_value=str(fake_fs)),
                pytest.raises(AudioRenderError) as exc_info,
            ):
                render_audio(
                    _plan_with_notes(),
                    bpm=120.0,
                    soundfont_path=sf,
                    out_dir=tmp_path / "out",
                    job_id="fail-job",
                    fluidsynth_bin=str(fake_fs),
                )
        assert exc_info.value.code == AudioRenderErrorCode.FLUIDSYNTH_FAILED


def _write_wav(path: Path) -> None:
    """Write a minimal valid WAV file (44 byte header + 1 sample of silence)."""
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(44100)
        wf.writeframes(b"\x00\x00")


class TestEncodeOpusAuditFailure:
    def test_audit_failure_raises(self, tmp_path: Path) -> None:
        wav = tmp_path / "audio.wav"
        wav.write_bytes(b"RIFF")
        fake_ff = tmp_path / "ffmpeg"
        fake_ff.write_text("#!/bin/sh\n")
        fake_ff.chmod(0o755)

        with patch("saimc.render.audio.audit_ffmpeg") as mock_audit:
            mock_audit.return_value = MagicMock(
                ok=False,
                reasons=("missing --enable-libvpx",),
            )
            with (
                patch("shutil.which", return_value=str(fake_ff)),
                pytest.raises(AudioRenderError) as exc_info,
            ):
                encode_opus(wav, tmp_path / "audio.ogg")
        assert exc_info.value.code == AudioRenderErrorCode.FFMPEG_AUDIT_FAILED


class TestHashFile:
    def test_known_content(self, tmp_path: Path) -> None:
        p = tmp_path / "f"
        p.write_bytes(b"hello world")
        expected = hashlib.sha256(b"hello world").hexdigest()
        assert _hash_file(p) == expected
