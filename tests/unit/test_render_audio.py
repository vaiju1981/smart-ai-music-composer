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
    TempoMap,
    TempoPoint,
    microseconds_at_tick,
    ticks_at_microsecond,
)
from saimc.render.audio import (
    ACCOMPANIMENT_CC7,
    ACCOMPANIMENT_CC10_PAN,
    CENTER_CC10_PAN,
    MASTER_LOUDNESS_LUFS,
    MASTER_TRUE_PEAK_DBTP,
    MELODY_CC7,
    MELODY_CC10_PAN,
    AudioRenderError,
    AudioRenderErrorCode,
    _hash_file,
    _loudnorm_filter,
    build_smf,
    encode_opus,
    find_ffmpeg,
    find_fluidsynth,
    render_audio,
)
from saimc.render.instruments import accompaniment_for


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


class TestSmfChannelMix:
    """Per-voice CC7 gain and CC10 pan: the mix, not font defaults."""

    def _plan_both_voices(self) -> PerformancePlan:
        return PerformancePlan.make(
            sample_rate=44100,
            notes=[
                PerformanceNoteEvent(
                    voice_id=0, pitch_midi=48, start_us=0, duration_us=500_000, velocity=64
                ),
                PerformanceNoteEvent(
                    voice_id=1, pitch_midi=72, start_us=0, duration_us=500_000, velocity=72
                ),
            ],
        )

    def _cc(self, smf, control: int) -> dict[int, int]:
        return {
            m.channel: m.value
            for m in smf.tracks[0]
            if m.type == "control_change" and m.control == control
        }

    def test_melody_and_accompaniment_get_gain_and_pan(self) -> None:
        smf = build_smf(
            self._plan_both_voices(), bpm=120.0, voice_instruments={0: "strings", 1: "piano"}
        )
        cc7 = self._cc(smf, 7)
        cc10 = self._cc(smf, 10)
        # Voice 0 (bass/accompaniment) on channel 0, voice 1 (melody) on channel 1.
        assert cc7[0] == ACCOMPANIMENT_CC7
        assert cc7[1] == MELODY_CC7
        assert cc10[0] == ACCOMPANIMENT_CC10_PAN
        assert cc10[1] == MELODY_CC10_PAN
        # The two voices sit on opposite sides of centre.
        assert cc10[1] > 64 > cc10[0]

    def test_percussion_stays_centred(self) -> None:
        plan = PerformancePlan.make(
            sample_rate=44100,
            notes=[
                PerformanceNoteEvent(
                    voice_id=2, pitch_midi=36, start_us=0, duration_us=100_000, velocity=80
                )
            ],
        )
        smf = build_smf(plan, bpm=120.0, voice_instruments={2: "drum_set"})
        assert self._cc(smf, 10)[9] == CENTER_CC10_PAN


class TestLoudnormFilter:
    def test_linear_filter_carries_measured_values(self) -> None:
        measured = {
            "input_i": "-23.62",
            "input_tp": "-7.02",
            "input_lra": "0.00",
            "input_thresh": "-34.00",
            "target_offset": "0.00",
        }
        f = _loudnorm_filter(measured)
        assert f.startswith(f"loudnorm=I={MASTER_LOUDNESS_LUFS}:TP={MASTER_TRUE_PEAK_DBTP}")
        assert "measured_I=-23.62" in f
        assert "linear=true" in f

    def test_no_measurement_falls_back_to_single_pass(self) -> None:
        f = _loudnorm_filter(None)
        assert "measured_I" not in f
        assert "linear" not in f


class TestTickConversion:
    def test_known_quantities(self) -> None:
        # 1 quarter at 120bpm = 500_000 us = 480 ticks.
        assert ticks_at_microsecond(500_000, TempoMap(bpm=120.0)) == 480
        # 1 second at 60bpm = 1_000_000 us = 480 ticks.
        assert ticks_at_microsecond(1_000_000, TempoMap(bpm=60.0)) == 480

    def test_tempo_change_splits_the_map(self) -> None:
        # 1 bar (480 ticks) at 120bpm = 0.5s, then 60bpm halves the
        # tick rate; 2 seconds of wall time therefore reaches tick
        # 1200, and the map converts back exactly.
        tempo = TempoMap(bpm=120.0, changes=(TempoPoint(tick=480, bpm=60.0),))
        assert ticks_at_microsecond(2_000_000, tempo) == 1200
        assert microseconds_at_tick(1200, tempo) == 2_000_000


class TestSmfPercussion:
    """Drum-set voices: pinned to GM channel 10, no program change."""

    @staticmethod
    def _plan(voice_id: int) -> PerformancePlan:
        return PerformancePlan.make(
            sample_rate=44100,
            notes=[
                PerformanceNoteEvent(
                    voice_id=voice_id,
                    pitch_midi=38,  # GM snare
                    start_us=0,
                    duration_us=100_000,
                    velocity=80,
                )
            ],
        )

    def test_percussion_voice_goes_to_channel_10(self) -> None:
        smf = build_smf(self._plan(2), bpm=120.0, voice_instruments={2: "drum_set"})
        channels = {m.channel for m in smf.tracks[0] if m.type == "note_on"}
        assert channels == {9}

    def test_percussion_voice_gets_no_program_change(self) -> None:
        smf = build_smf(self._plan(2), bpm=120.0, voice_instruments={2: "drum_set"})
        changes = [m for m in smf.tracks[0] if m.type == "program_change"]
        assert {c.channel for c in changes} == set()  # only the drum voice exists

    def test_mixed_voices_keep_percussion_on_10_and_melody_off_it(self) -> None:
        plan = PerformancePlan.make(
            sample_rate=44100,
            notes=[
                PerformanceNoteEvent(
                    voice_id=1,
                    pitch_midi=60,
                    start_us=0,
                    duration_us=500_000,
                    velocity=64,
                ),
                PerformanceNoteEvent(
                    voice_id=2,
                    pitch_midi=36,  # kick
                    start_us=0,
                    duration_us=100_000,
                    velocity=90,
                ),
            ],
        )
        smf = build_smf(
            plan,
            bpm=120.0,
            voice_instruments={1: "piano", 2: "drum_set"},
        )
        note_channels = {m.channel for m in smf.tracks[0] if m.type == "note_on"}
        change_channels = {m.channel for m in smf.tracks[0] if m.type == "program_change"}
        assert 9 in note_channels
        assert 9 not in change_channels  # drums need no patch
        assert change_channels == {1}  # melody voice only

    def test_drum_set_resolves_to_the_general_font(self) -> None:
        from saimc.render.instruments import soundfont_for_instrument

        # With no local font files the fallback path is returned; either
        # way it must resolve without error.
        assert soundfont_for_instrument("drum_set") is not None


class TestSmfChannelMapping:
    def test_voice_ids_skip_gm_percussion_channel(self) -> None:
        """Channel 10 (index 9) is the GM percussion kit; voices must skip it."""

        def _plan(voice_id: int) -> PerformancePlan:
            return PerformancePlan.make(
                sample_rate=44100,
                notes=[
                    PerformanceNoteEvent(
                        voice_id=voice_id,
                        pitch_midi=60,
                        start_us=0,
                        duration_us=500_000,
                        velocity=64,
                    )
                ],
            )

        for voice_id, expected_channel in [(0, 0), (8, 8), (9, 10), (10, 11), (16, 0)]:
            smf = build_smf(_plan(voice_id), bpm=120.0)
            channels = {m.channel for m in smf.tracks[0] if m.type == "note_on"}
            assert channels == {expected_channel}


class TestSmfProgramChange:
    @staticmethod
    def _plan(*voice_ids: int) -> PerformancePlan:
        return PerformancePlan.make(
            sample_rate=44100,
            notes=[
                PerformanceNoteEvent(
                    voice_id=voice_id,
                    pitch_midi=60,
                    start_us=0,
                    duration_us=500_000,
                    velocity=64,
                )
                for voice_id in voice_ids
            ],
        )

    def test_default_program_is_piano(self) -> None:
        smf = build_smf(self._plan(0), bpm=120.0)
        changes = [m for m in smf.tracks[0] if m.type == "program_change"]
        assert len(changes) == 1
        assert changes[0].program == 0  # GM acoustic grand piano
        assert changes[0].channel == 0

    def test_voice_instruments_select_the_program(self) -> None:
        from saimc.render.instruments import INSTRUMENT_PROGRAMS

        smf = build_smf(
            self._plan(0, 1),
            bpm=120.0,
            voice_instruments={0: "piano", 1: "piano"},
        )
        changes = [m for m in smf.tracks[0] if m.type == "program_change"]
        assert {c.channel for c in changes} == {0, 1}
        assert all(c.program == INSTRUMENT_PROGRAMS["piano"] for c in changes)

    def test_unknown_instrument_raises(self) -> None:
        with pytest.raises(AudioRenderError) as exc_info:
            build_smf(self._plan(0), bpm=120.0, voice_instruments={0: "theremin"})
        assert exc_info.value.code == AudioRenderErrorCode.MIDI_BUILD_FAILED

    def test_program_changes_precede_notes(self) -> None:
        smf = build_smf(self._plan(0), bpm=120.0)
        types = [m.type for m in smf.tracks[0]]
        assert types.index("program_change") < types.index("note_on")


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
            patch("saimc.render.audio.sha256_file", return_value="x" * 64),
            patch("saimc.render.audio.safe_run") as mock_run,
        ):
            mock_run.return_value = MagicMock(returncode=0)
            run_version, _run_sha = _run_fluidsynth_mock(
                smf, sf, out_wav, fluidsynth_bin=str(fake_bin)
            )
        # We didn't call run_fluidsynth; this is just exercising find_fluidsynth.
        assert run_version == "2.3.4"
        # The file sample format is pinned explicitly: the manifest
        # reports pcm_s16le and must not depend on the binary's default.
        cmd = mock_run.call_args[0][0]
        assert "audio.file.format=s16" in cmd
        # The mix settings are pinned too: chorus off (its stereo LFO can
        # put a font's channels out of phase), reverb and gain explicit.
        assert "synth.chorus.active=0" in cmd
        assert any(setting.startswith("synth.reverb.") for setting in cmd)
        assert cmd[cmd.index("-g") + 1] == "0.6"

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


class TestAccompanimentMapping:
    """The accompaniment voice gets a complementary patch, not the lead."""

    def test_gm_backed_instruments_get_companions(self) -> None:
        assert accompaniment_for("piano") == "strings"
        assert accompaniment_for("violin") == "pizzicato_strings"
        assert accompaniment_for("accordion") == "choir"
        assert accompaniment_for("nylon_guitar") == "pizzicato_strings"

    def test_dedicated_font_instruments_return_themselves(self) -> None:
        # Only one font loads per job, so a dedicated-font lead keeps its
        # own preset on both voices; separation comes from CC7/CC10.
        assert accompaniment_for("harmonium") == "harmonium"
        assert accompaniment_for("sitar") == "sitar"
        assert accompaniment_for("bansuri") == "bansuri"

    def test_unknown_instrument_falls_back_to_strings(self) -> None:
        assert accompaniment_for("theremin") == "strings"


class TestSoundfontResolution:
    """Hermetic: run in an empty cwd and build the chain with real files."""

    @pytest.fixture(autouse=True)
    def _empty_cwd(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.delenv("SAIMC_SOUNDFONT_PIANO", raising=False)

    def _touch(self, rel: str) -> None:
        path = Path(rel)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"SF2")

    def test_bare_repo_falls_back_to_salamander_path(self) -> None:
        from saimc.render.instruments import soundfont_for_instrument

        assert soundfont_for_instrument("piano") == Path("./assets/Salamander.sf2")

    def test_general_font_covers_non_piano_instruments(self) -> None:
        from saimc.render.instruments import soundfont_for_instrument

        self._touch("assets/soundfonts/FluidR3_GM.sf2")
        assert soundfont_for_instrument("sitar") == Path("./assets/soundfonts/FluidR3_GM.sf2")
        assert soundfont_for_instrument("violin") == Path("./assets/soundfonts/FluidR3_GM.sf2")

    def test_piano_prefers_salamander_when_both_exist(self) -> None:
        from saimc.render.instruments import soundfont_for_instrument

        self._touch("assets/soundfonts/FluidR3_GM.sf2")
        self._touch("assets/Salamander.sf2")
        assert soundfont_for_instrument("piano") == Path("./assets/Salamander.sf2")
        assert soundfont_for_instrument("sitar") == Path("./assets/soundfonts/FluidR3_GM.sf2")

    def test_per_instrument_font_wins_over_the_general_font(self) -> None:
        from saimc.render.instruments import soundfont_for_instrument

        self._touch("assets/soundfonts/FluidR3_GM.sf2")
        self._touch("assets/soundfonts/sitar.sf2")
        assert soundfont_for_instrument("sitar") == Path("./assets/soundfonts/sitar.sf2")

    def test_env_override_wins(self) -> None:
        from saimc.render.instruments import soundfont_for_instrument

        with patch.dict("os.environ", {"SAIMC_SOUNDFONT_PIANO": "/tmp/grand.sf2"}):
            assert soundfont_for_instrument("piano") == Path("/tmp/grand.sf2")

    def test_env_key_normalizes_instrument_name(self) -> None:
        from saimc.render.instruments import soundfont_for_instrument

        with patch.dict("os.environ", {"SAIMC_SOUNDFONT_VIOLIN_II": "/tmp/violins.sf2"}):
            assert soundfont_for_instrument("violin-ii") == Path("/tmp/violins.sf2")
            assert soundfont_for_instrument("violin ii") == Path("/tmp/violins.sf2")

    def test_dedicated_font_wins_when_present(self) -> None:
        from saimc.render.instruments import soundfont_for_instrument

        self._touch("assets/soundfonts/FluidR3_GM.sf2")
        self._touch("assets/soundfonts/105-Sitar.sf2")
        self._touch("assets/soundfonts/Wetthasinghe_Harmonium.sf2")
        assert soundfont_for_instrument("sitar") == Path("./assets/soundfonts/105-Sitar.sf2")
        assert soundfont_for_instrument("harmonium") == (
            Path("./assets/soundfonts/Wetthasinghe_Harmonium.sf2")
        )

    def test_dedicated_font_missing_falls_back_to_general(self) -> None:
        from saimc.render.instruments import soundfont_for_instrument

        self._touch("assets/soundfonts/FluidR3_GM.sf2")
        # sitar falls back to the GM patch; harmonium has no GM voice.
        assert soundfont_for_instrument("sitar") == Path("./assets/soundfonts/FluidR3_GM.sf2")
        assert soundfont_for_instrument("harmonium") == Path("./assets/soundfonts/FluidR3_GM.sf2")


class TestDedicatedFontPresets:
    """FONT_PRESETS selection is valid only inside the font being loaded."""

    @staticmethod
    def _plan(voice_id: int = 0) -> PerformancePlan:
        return PerformancePlan.make(
            sample_rate=44100,
            notes=[
                PerformanceNoteEvent(
                    voice_id=voice_id,
                    pitch_midi=60,
                    start_us=0,
                    duration_us=500_000,
                    velocity=64,
                )
            ],
        )

    def test_preset_selected_when_dedicated_font_is_loaded(self) -> None:
        from saimc.render.instruments import preset_for_instrument

        dedicated = Path("./assets/soundfonts/105-Sitar.sf2")
        # The preset pair is tied to the exact font path being loaded.
        assert preset_for_instrument("sitar", dedicated) == (0, 0)
        assert preset_for_instrument("sitar", Path("./assets/FluidR3_GM.sf2")) is None

        smf = build_smf(
            self._plan(),
            bpm=120.0,
            voice_instruments={0: "sitar"},
            soundfont_path=dedicated,
        )
        programs = [m.program for m in smf.tracks[0] if m.type == "program_change"]
        assert programs == [0]  # bank 0, preset 0 of 105-Sitar

    def test_gm_fallback_program_with_the_general_font(self, tmp_path: Path) -> None:
        from saimc.render.instruments import INSTRUMENT_PROGRAMS

        gm = tmp_path / "FluidR3_GM.sf2"
        gm.write_bytes(b"SF2")
        smf = build_smf(
            self._plan(),
            bpm=120.0,
            voice_instruments={0: "sitar"},
            soundfont_path=gm,
        )
        programs = [m.program for m in smf.tracks[0] if m.type == "program_change"]
        assert programs == [INSTRUMENT_PROGRAMS["sitar"]]  # GM 104

    def test_font_only_instrument_without_font_raises_cleanly(self, tmp_path: Path) -> None:
        gm = tmp_path / "FluidR3_GM.sf2"
        gm.write_bytes(b"SF2")
        with pytest.raises(AudioRenderError) as exc_info:
            build_smf(
                self._plan(),
                bpm=120.0,
                voice_instruments={0: "harmonium"},
                soundfont_path=gm,
            )
        assert exc_info.value.code == AudioRenderErrorCode.MIDI_BUILD_FAILED
        assert "Wetthasinghe_Harmonium.sf2" in str(exc_info.value)


class TestAttributionTags:
    """OGG metadata tags carry the rendered font's license (§10 #12)."""

    def test_default_is_salamander(self) -> None:
        from saimc.render.attribution import audio_metadata_tags

        tags = audio_metadata_tags()
        assert tags["LIBRARY"] == "Salamander Grand Piano"
        assert tags["LICENSE"] == "CC BY 3.0"

    def test_dedicated_fonts_carry_their_own_license(self) -> None:
        from saimc.render.attribution import audio_metadata_tags

        boston = audio_metadata_tags("MFA_Boston_1.sf2")
        assert boston["LIBRARY"] == "MFA Boston 1"
        assert boston["LICENSE"] == "CC BY 3.0"
        sitar = audio_metadata_tags("105-Sitar.sf2")
        assert sitar["LICENSE"] == "Public domain"
        harmonium = audio_metadata_tags("Wetthasinghe_Harmonium.sf2")
        assert harmonium["LICENSE"] == "CC BY 4.0"

    def test_unknown_font_is_unverified_not_borrowed(self) -> None:
        from saimc.render.attribution import audio_metadata_tags

        tags = audio_metadata_tags("Mystery_Kit.sf2")
        assert tags["LIBRARY"] == "Mystery_Kit.sf2"
        assert tags["LICENSE"] == "unverified"


class TestBostonFontPresets:
    """MFA Boston 1's preset layout is its own — (bank, preset) pairs."""

    @staticmethod
    def _boston_programs(instrument: str) -> list[tuple[bool, int]]:
        """(is_bank_select, value) events emitted for `instrument`."""
        boston = Path("./assets/soundfonts/MFA_Boston_1.sf2")
        smf = build_smf(
            TestDedicatedFontPresets._plan(),
            bpm=120.0,
            voice_instruments={0: instrument},
            soundfont_path=boston,
        )
        events: list[tuple[bool, int]] = []
        for m in smf.tracks[0]:
            if m.type == "control_change" and m.control == 0 and m.channel == 0:
                events.append((True, m.value))  # bank select CC0
            elif m.type == "program_change" and m.channel == 0:
                events.append((False, m.program))
        return events

    def test_font_only_instruments_select_their_boston_preset(self) -> None:
        from saimc.render.instruments import preset_for_instrument

        boston = Path("./assets/soundfonts/MFA_Boston_1.sf2")
        expected = {
            "bansuri": (0, 77),
            "sarangi": (1, 110),
            "rudra_veena": (1, 104),
            "sarasvati_veena": (2, 104),
            "qanoon": (1, 107),
            "ud": (3, 105),
            "kora": (8, 105),
        }
        for instrument, pair in expected.items():
            assert preset_for_instrument(instrument, boston) == pair
            assert preset_for_instrument(instrument, Path("./assets/FluidR3_GM.sf2")) is None

    def test_bank_select_precedes_program_change_for_nonzero_banks(self) -> None:
        # sarangi lives in bank 1: a CC0 bank select must precede the
        # program change, and ud (bank 3, preset 105) selects program 105.
        sarangi = self._boston_programs("sarangi")
        assert sarangi == [(True, 1), (False, 110)]
        ud = self._boston_programs("ud")
        assert ud == [(True, 3), (False, 105)]

    def test_bank_zero_instruments_get_program_change_only(self) -> None:
        bansuri = self._boston_programs("bansuri")
        assert bansuri == [(False, 77)]
        koto = self._boston_programs("koto")
        assert koto == [(False, 107)]  # Boston's koto, not GM 107

    def test_gm_backed_upgrade_falls_back_to_gm_without_boston(self, tmp_path: Path) -> None:
        gm = tmp_path / "FluidR3_GM.sf2"
        gm.write_bytes(b"SF2")
        smf = build_smf(
            TestDedicatedFontPresets._plan(),
            bpm=120.0,
            voice_instruments={0: "shamisen"},
            soundfont_path=gm,
        )
        programs = [m.program for m in smf.tracks[0] if m.type == "program_change"]
        assert programs == [106]  # GM shamisen


class TestInstrumentRegistry:
    def test_all_programs_are_valid_gm_numbers(self) -> None:
        from saimc.render.instruments import INSTRUMENT_PROGRAMS

        assert INSTRUMENT_PROGRAMS["piano"] == 0
        for program in INSTRUMENT_PROGRAMS.values():
            assert 0 <= program <= 127

    def test_every_instrument_has_a_family(self) -> None:
        from saimc.render.instruments import INSTRUMENT_FAMILIES, SUPPORTED_INSTRUMENTS

        assert set(INSTRUMENT_FAMILIES) == set(SUPPORTED_INSTRUMENTS)
        assert set(INSTRUMENT_FAMILIES.values()) <= {"western", "world"}

    def test_western_orchestra_palette(self) -> None:
        from saimc.render.instruments import INSTRUMENT_PROGRAMS

        for name in (
            "violin",
            "viola",
            "cello",
            "contrabass",
            "harp",
            "flute",
            "oboe",
            "clarinet",
            "bassoon",
            "french_horn",
            "trumpet",
            "trombone",
            "tuba",
            "timpani",
        ):
            assert name in INSTRUMENT_PROGRAMS, name

    def test_world_palette(self) -> None:
        from saimc.render.instruments import INSTRUMENT_PROGRAMS

        # Non-western voices GM provides; dedicated fonts upgrade these.
        assert INSTRUMENT_PROGRAMS["sitar"] == 104
        assert INSTRUMENT_PROGRAMS["koto"] == 107
        assert INSTRUMENT_PROGRAMS["shanai"] == 111
        assert INSTRUMENT_PROGRAMS["taiko"] == 116
        assert INSTRUMENT_PROGRAMS["kalimba"] == 108


class TestSmfExpressionEvents:
    """The plan's controllers and pitch bends ride the MIDI timeline."""

    def _cc_events(self, smf, control: int) -> list[tuple[int, int, int]]:
        """(tick, channel, value) for one controller number."""
        result = []
        tick = 0
        for m in smf.tracks[0]:
            tick += m.time
            if m.type == "control_change" and m.control == control:
                result.append((tick, m.channel, m.value))
        return result

    def test_plan_controllers_reach_the_midi(self) -> None:
        from saimc.compose.score import ControllerEvent

        plan = PerformancePlan.make(
            sample_rate=44100,
            notes=_plan_with_notes().notes,
            controllers=[
                ControllerEvent(voice_id=1, control=11, value=96, start_us=0),
                ControllerEvent(voice_id=1, control=64, value=127, start_us=250_000),
            ],
        )
        smf = build_smf(plan, bpm=120.0, voice_instruments={1: "piano"})
        cc11 = self._cc_events(smf, 11)
        cc64 = self._cc_events(smf, 64)
        # 120 bpm: 250 ms = tick 240. Voice 1 rides channel 1.
        assert (0, 1, 96) in cc11
        assert (240, 1, 127) in cc64

    def test_pitch_bends_reach_the_midi(self) -> None:
        from saimc.compose.score import PitchBendEvent

        plan = PerformancePlan.make(
            sample_rate=44100,
            notes=_plan_with_notes().notes,
            pitch_bends=[PitchBendEvent(voice_id=1, start_us=500_000, bend=2048)],
        )
        smf = build_smf(plan, bpm=120.0, voice_instruments={1: "piano"})
        bends = []
        tick = 0
        for m in smf.tracks[0]:
            tick += m.time
            if m.type == "pitchwheel":
                bends.append((tick, m.channel, m.pitch))
        # 120 bpm: 500 ms = tick 480, voice 1 = channel 1, bend passed through.
        assert (480, 1, 2048) in bends

    def test_pedal_press_precedes_note_at_same_tick(self) -> None:
        from saimc.compose.score import ControllerEvent

        plan = PerformancePlan.make(
            sample_rate=44100,
            notes=_plan_with_notes().notes,
            controllers=[ControllerEvent(voice_id=1, control=64, value=127, start_us=0)],
        )
        smf = build_smf(plan, bpm=120.0, voice_instruments={1: "piano"})
        # The pedal CC64 must come before the first note_on of its voice.
        order = [m.type for m in smf.tracks[0]]
        first_note_on = order.index("note_on")
        pedal_index = next(
            i
            for i, m in enumerate(smf.tracks[0])
            if m.type == "control_change" and m.control == 64
        )
        assert pedal_index < first_note_on


class TestSmfTempoMap:
    """S8: the SMF reproduces a piecewise tempo map (the outro rit)."""

    def test_set_tempo_at_the_change_tick(self) -> None:
        plan = _plan_with_notes()
        smf = build_smf(plan, bpm=120.0, tempo_changes=(TempoPoint(tick=480, bpm=60.0),))
        absolute = 0
        tempos: dict[int, int] = {}
        for msg in smf.tracks[0]:
            absolute += msg.time
            if msg.type == "set_tempo":
                tempos[absolute] = msg.tempo
        # Base tempo at tick 0, then the ritardando from tick 480.
        assert tempos == {0: 500_000, 480: 1_000_000}

    def test_note_onsets_follow_the_piecewise_map(self) -> None:
        # Two notes: one before the change (120bpm), one an equal tick
        # span after it (60bpm). In wall time the second onset lands
        # twice as far out.
        tempo = TempoMap(bpm=120.0, changes=(TempoPoint(tick=480, bpm=60.0),))
        plan = PerformancePlan.make(
            sample_rate=44100,
            notes=[
                PerformanceNoteEvent(
                    voice_id=0,
                    pitch_midi=60,
                    start_us=microseconds_at_tick(tick, tempo),
                    duration_us=100_000,
                    velocity=64,
                )
                for tick in (0, 480, 960)
            ],
        )
        smf = build_smf(plan, bpm=120.0, tempo_changes=tempo.changes)
        elapsed_us = 0.0
        us_per_tick = 500_000 / PPQ
        onsets: list[float] = []
        for msg in smf.tracks[0]:
            elapsed_us += msg.time * us_per_tick
            if msg.type == "set_tempo":
                us_per_tick = msg.tempo / PPQ
            elif msg.type == "note_on" and msg.velocity > 0:
                onsets.append(elapsed_us)
        assert onsets == [0.0, 500_000.0, 1_500_000.0]
