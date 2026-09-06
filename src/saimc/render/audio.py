"""Audio renderer: PerformancePlan -> MIDI -> FluidSynth -> WAV (and OGG Opus).

Per `docs/roadmap.md` §2 step 4a and §4, the audio pipeline is:

    PerformancePlan (integer-microsecond timestamps)
      -> SMF Type-0 MIDI bytes (PPQ=480)
      -> FluidSynth (external LGPL process) + soundfont
      -> uncompressed WAV
      -> (optional) FFmpeg encode to OGG Opus (audio-only artifact)

Phase 1 ships WAV and OGG Opus (§4). MP3 is deliberately excluded
(§4 "MP3 is deliberately excluded from Phase 1") to keep the
codec surface minimal.

The renderer is split into three pure-ish functions plus a thin
`render_audio()` orchestrator. Each subprocess is invoked through
`safe_run` (saimc.jobs.stages) with an absolute timeout; failures
surface as `AudioRenderError` with a stable code.
"""

from __future__ import annotations

import logging
import shutil
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from saimc.compose.score import (
    PPQ,
    VOICE_BASS,
    VOICE_MELODY,
    PerformancePlan,
    TempoMap,
    TempoPoint,
    ticks_at_microsecond,
)
from saimc.jobs.stages import SubprocessTimeoutError, safe_run
from saimc.render.ffmpeg_audit import audit_ffmpeg
from saimc.render.instruments import (
    FONT_PRESETS,
    INSTRUMENT_PROGRAMS,
    PERCUSSION_INSTRUMENTS,
    SOUNDFONT_DIR,
    SUPPORTED_INSTRUMENTS,
    preset_for_instrument,
)
from saimc.render.util import sha256_file

if TYPE_CHECKING:
    from mido import MidiFile

logger = logging.getLogger(__name__)


DEFAULT_FLUIDSYNTH_TIMEOUT_S: float = 300.0
"""Five minutes for a typical 5-minute piece; the subprocess wrapper
kills it if it runs longer."""

DEFAULT_FFMPEG_TIMEOUT_S: float = 60.0

# Mix constants (§ "produced-track quality"). FluidSynth's own default
# gain (0.2) leaves WAV peaks around 11-19% of full scale; 0.6 lands the
# raw render around -6..-9 dBFS peaks with clipping headroom to spare —
# the loudnorm pass in the Opus encode sets the final delivery level.
FLUIDSYNTH_GAIN: float = 0.6
# Reverb pinned to FluidSynth's documented defaults (a version bump must
# not silently change the room). Chorus is disabled in run_fluidsynth:
# its stereo LFO can put a font's channels out of phase. Setting names
# follow FluidSynth 2.x: synth.reverb.room-size / .damp / .width / .level.
FLUIDSYNTH_REVERB_ROOM_SIZE: float = 0.61
FLUIDSYNTH_REVERB_DAMP: float = 0.23
FLUIDSYNTH_REVERB_WIDTH: float = 0.76
FLUIDSYNTH_REVERB_LEVEL: float = 0.87

# Master loudness target for the encoded deliverable (streaming-standard
# loudness with true-peak headroom). Applied via a two-pass ffmpeg
# loudnorm in encode_opus.
MASTER_LOUDNESS_LUFS: float = -16.0
MASTER_TRUE_PEAK_DBTP: float = -1.5
MASTER_LRA: float = 11.0

# CC7 (channel volume) per voice role, and CC10 (pan) to give the mix a
# stereo image: melody right of centre, accompaniment left of centre.
# 64 is centre; ±22 ≈ ±17% of full scale.
MELODY_CC7: int = 100
ACCOMPANIMENT_CC7: int = 84
MELODY_CC10_PAN: int = 86
ACCOMPANIMENT_CC10_PAN: int = 42
CENTER_CC10_PAN: int = 64


class AudioRenderErrorCode:
    """Stable error codes for the audio renderer."""

    FLUIDSYNTH_MISSING = "fluidsynth_missing"
    FLUIDSYNTH_FAILED = "fluidsynth_failed"
    SOUNDFONT_MISSING = "soundfont_missing"
    FFMPEG_MISSING = "ffmpeg_missing"
    FFMPEG_AUDIT_FAILED = "ffmpeg_audit_failed"
    FFMPEG_FAILED = "ffmpeg_failed"
    MIDI_BUILD_FAILED = "midi_build_failed"


class AudioRenderError(Exception):
    """Raised when audio rendering fails. Carries a stable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class AudioArtifact:
    """Result of one render: one or more audio files with provenance."""

    primary_path: Path
    primary_container: str  # "wav"
    primary_codec: str  # "pcm_s16le"
    primary_sha256: str
    primary_size_bytes: int

    ogg_path: Path | None = None
    ogg_codec: str | None = None  # "opus"
    ogg_sha256: str | None = None
    ogg_size_bytes: int | None = None

    fluidsynth_version: str = ""
    fluidsynth_build_sha: str = ""

    ffmpeg_version: str = ""
    ffmpeg_build_sha: str = ""

    soundfont_name: str = ""
    soundfont_sha256: str = ""

    ffmpeg_configuration: str = ""


def build_smf(
    plan: PerformancePlan,
    *,
    bpm: float,
    tempo_changes: tuple[TempoPoint, ...] = (),
    voice_instruments: Mapping[int, str] | None = None,
    soundfont_path: Path | None = None,
) -> MidiFile:
    """Convert a PerformancePlan into a SMF Type-0 MIDI file.

    `bpm` is the realised tempo of the plan (taken from the
    NotationScore.tempo that produced the plan); `tempo_changes`
    carries any later tempo points (the outro ritardando) so the SMF
    reproduces the plan's piecewise timing — the note ticks below are
    converted on that same map. The SMF uses PPQ=480, matching the
    NotationScore's PPQ. Each set_tempo meta event carries the integer
    microseconds-per-quarter so that downstream tools (FluidSynth
    included) get an exact tempo.

    `voice_instruments` maps voice_id -> instrument name; each mapped
    voice gets a General MIDI program_change on its channel so
    FluidSynth picks the right patch (the Phase 2 orchestra needs
    this; Phase 1 defaults every voice to piano anyway). Percussion
    voices are pinned to channel 10 with no program change, and a
    dedicated-font instrument (`FONT_PRESETS`) selects its own
    bank:preset — but only when `soundfont_path` (the font the caller
    will actually load) is that font.

    Returns the populated `mido.MidiFile` (caller persists to disk).
    """
    import mido

    tempo = TempoMap(bpm=bpm, ppq=PPQ, changes=tempo_changes)

    midi = mido.MidiFile(type=0)
    midi.ticks_per_beat = PPQ
    track = mido.MidiTrack()
    midi.tracks.append(track)

    # Tempo: microseconds per quarter note. bpm = 60_000_000 / us_per_quarter.
    # The tempo map's points land as set_tempo events at their ticks;
    # they sort ahead of anything else at the same tick (order -2).
    us_per_quarter = round(60_000_000 / bpm)
    track.append(mido.MetaMessage("set_tempo", tempo=us_per_quarter, time=0))
    events: list[tuple[int, int, mido.Message]] = [
        (
            point.tick,
            -2,
            mido.MetaMessage("set_tempo", tempo=round(60_000_000 / point.bpm)),
        )
        for point in tempo.changes
    ]

    # Program changes up front, one per (voice, channel) that appears in
    # the plan or in voice_instruments. Channels are assigned the same
    # way as the note events below. Percussion voices (drum set) go to
    # GM channel 10 and get NO program change — channel 10 already
    # selects the kit, and a melodic program there would sound wrong.
    instruments = dict(voice_instruments or {})
    percussion_voices = {
        voice_id for voice_id, inst in instruments.items() if inst in PERCUSSION_INSTRUMENTS
    }
    voice_channels: dict[int, int] = {}
    for note in plan.notes:
        if note.voice_id not in voice_channels:
            voice_channels[note.voice_id] = _channel_for_voice(
                note.voice_id, percussion=note.voice_id in percussion_voices
            )
    for voice_id in instruments:
        voice_channels.setdefault(
            voice_id,
            _channel_for_voice(voice_id, percussion=voice_id in percussion_voices),
        )
    for voice_id in sorted(voice_channels):
        instrument = instruments.get(voice_id, "piano")
        if voice_id in percussion_voices:
            continue
        # A dedicated-font instrument gets a bank-select + program pair
        # valid inside the font that is actually being loaded; with any
        # other font loaded the GM program is the only meaningful choice.
        preset = (
            preset_for_instrument(instrument, soundfont_path)
            if soundfont_path is not None
            else None
        )
        program: int
        if preset is not None:
            bank, program = preset
            if bank:
                track.append(
                    mido.Message(
                        "control_change", channel=voice_channels[voice_id], control=0, value=bank
                    )
                )
        else:
            gm_program = INSTRUMENT_PROGRAMS.get(instrument)
            if gm_program is None:
                font_note = ""
                mapping = FONT_PRESETS.get(instrument)
                if mapping is not None:
                    font_note = (
                        f"; {instrument!r} needs its dedicated soundfont "
                        f"{mapping[0]!r} in {SOUNDFONT_DIR}, which was not found"
                    )
                raise AudioRenderError(
                    AudioRenderErrorCode.MIDI_BUILD_FAILED,
                    f"unknown instrument {instrument!r} for voice {voice_id}{font_note}; "
                    f"known instruments: {', '.join(sorted(SUPPORTED_INSTRUMENTS))}",
                )
            program = gm_program
        track.append(
            mido.Message("program_change", channel=voice_channels[voice_id], program=program)
        )

    # Channel mix: static CC7 (channel volume) and CC10 (pan) per voice.
    # Without these the balance is whatever each SF2 preset's own
    # attenuation happens to be (measured ~10 dB between fonts), and the
    # stereo image is a centre mono-ish blob or the (now disabled)
    # chorus LFO. Melody sits right of centre and a touch louder; the
    # accompaniment sits left of centre and softer; percussion stays
    # centred.
    for voice_id in sorted(voice_channels):
        channel = voice_channels[voice_id]
        if voice_id == VOICE_BASS and voice_id not in percussion_voices:
            cc7, pan = ACCOMPANIMENT_CC7, ACCOMPANIMENT_CC10_PAN
        elif voice_id == VOICE_MELODY and voice_id not in percussion_voices:
            cc7, pan = MELODY_CC7, MELODY_CC10_PAN
        else:
            cc7, pan = MELODY_CC7, CENTER_CC10_PAN
        track.append(mido.Message("control_change", channel=channel, control=7, value=cc7))
        track.append(mido.Message("control_change", channel=channel, control=10, value=pan))

    track.append(mido.MetaMessage("end_of_track", time=0))

    # Convert each PerformanceNoteEvent to a note_on / note_off pair.
    # PPQ=480 ticks per beat. The plan's start_us is in microseconds;
    # we need integer ticks at PPQ=480, converted on the piecewise
    # tempo map so a ritardando lands where the plan says it does.
    # SMF event times are DELTAS from the previous event on the track,
    # and the plan's voices overlap, so collect all events on an
    # absolute-tick timeline and delta-encode in order (note_on before
    # note_off at the same tick). Tempo changes sort at order -2 and
    # controller changes / pitch bends at -1 so a tempo switch (or a
    # pedal press) precedes the note sounding at the same tick.
    for controller in plan.controllers:
        controller_tick = ticks_at_microsecond(controller.start_us, tempo)
        events.append(
            (
                controller_tick,
                -1,
                mido.Message(
                    "control_change",
                    channel=_channel_for_voice(
                        controller.voice_id,
                        percussion=controller.voice_id in percussion_voices,
                    ),
                    control=controller.control,
                    value=controller.value,
                ),
            )
        )
    for bend in plan.pitch_bends:
        bend_tick = ticks_at_microsecond(bend.start_us, tempo)
        events.append(
            (
                bend_tick,
                -1,
                mido.Message(
                    "pitchwheel",
                    channel=_channel_for_voice(
                        bend.voice_id, percussion=bend.voice_id in percussion_voices
                    ),
                    pitch=bend.bend,
                ),
            )
        )
    for note in plan.notes:
        on_tick = ticks_at_microsecond(note.start_us, tempo)
        off_tick = ticks_at_microsecond(note.start_us + note.duration_us, tempo)
        if off_tick <= on_tick:
            # Guard against zero-or-negative durations from the round
            # trip; the engine shouldn't produce these but we don't
            # want a malformed MIDI file to crash the renderer.
            off_tick = on_tick + 1
        channel = _channel_for_voice(note.voice_id, percussion=note.voice_id in percussion_voices)
        events.append(
            (
                on_tick,
                0,
                mido.Message(
                    "note_on",
                    channel=channel,
                    note=note.pitch_midi,
                    velocity=note.velocity,
                ),
            )
        )
        events.append(
            (
                off_tick,
                1,
                mido.Message(
                    "note_off",
                    channel=channel,
                    note=note.pitch_midi,
                    velocity=0,
                ),
            )
        )

    prev_tick = 0
    for tick, _order, message in sorted(events, key=lambda e: (e[0], e[1])):
        message.time = tick - prev_tick
        track.append(message)
        prev_tick = tick
    return midi


def _channel_for_voice(voice_id: int, *, percussion: bool = False) -> int:
    """Map a voice ID to a MIDI channel.

    Channel 10 (index 9) is the GM percussion kit — melodic voices are
    mapped around it, but a percussion voice (drum set) is pinned
    exactly there: on channel 10 the note pitch IS the drum piece.
    """
    if percussion:
        return 9
    channel = voice_id % 16
    if channel >= 9:
        channel += 1
    return channel


def _hash_file(path: Path, *, cached: bool = False) -> str:
    """SHA-256 over the file's bytes (streaming)."""
    from saimc.render.util import sha256_file

    return sha256_file(path, cached=cached)


def _file_size(path: Path) -> int:
    return path.stat().st_size


# ---------------------------------------------------------------------------
# FluidSynth subprocess
# ---------------------------------------------------------------------------


def find_fluidsynth(explicit: str | None = None) -> str:
    """Locate the fluidsynth binary.

    Resolution order: explicit > `SAIMC_RENDER_FLUIDSYNTH` env > PATH.
    Raises `AudioRenderError(FLUIDSYNTH_MISSING)` if not found.
    """
    import os

    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    env_path = os.environ.get("SAIMC_RENDER_FLUIDSYNTH")
    if env_path:
        candidates.append(env_path)
    which = shutil.which("fluidsynth")
    if which:
        candidates.append(which)

    for candidate in candidates:
        if Path(candidate).exists() and Path(candidate).is_file():
            return candidate
    raise AudioRenderError(
        AudioRenderErrorCode.FLUIDSYNTH_MISSING,
        "fluidsynth not found on PATH; set SAIMC_RENDER_FLUIDSYNTH",
    )


def run_fluidsynth(
    smf_path: Path,
    soundfont_path: Path,
    out_wav_path: Path,
    *,
    fluidsynth_bin: str | None = None,
    sample_rate: int = 44100,
    timeout_s: float = DEFAULT_FLUIDSYNTH_TIMEOUT_S,
) -> tuple[str, str]:
    """Run `fluidsynth` to render `smf_path` + `soundfont_path` -> `out_wav_path`.

    Returns `(version_line, build_sha)` for the manifest toolchain
    block. The `build_sha` is the sha256 of the fluidsynth binary
    (the FFmpeg audit pattern).
    """
    bin_path = find_fluidsynth(fluidsynth_bin)
    if not soundfont_path.exists():
        raise AudioRenderError(
            AudioRenderErrorCode.SOUNDFONT_MISSING,
            f"soundfont not found at {soundfont_path}",
        )
    out_wav_path.parent.mkdir(parents=True, exist_ok=True)

    # Pin the file sample format: FluidSynth's default is s16 and the
    # manifest reports pcm_s16le, so the setting is made explicit
    # rather than relying on the default never changing.
    #
    # Mix settings are explicit too: the default chorus is an LFO on the
    # stereo bus that can push a font's channels out of phase (measured
    # corr=-0.50 on the harmonium font, where mono fold-down loses 6 dB),
    # so it is disabled — stereo image comes from panning, not an LFO.
    # Reverb is pinned to FluidSynth's documented defaults so a version
    # bump cannot silently change the room. The gain raises the raw WAV
    # to a healthy 16-bit level (~-6..-9 dBFS peaks) with headroom left
    # for the loudnorm pass in the Opus encode.
    cmd = [
        bin_path,
        "-F",  # render to a file
        str(out_wav_path),
        "-q",  # quiet
        "-r",
        str(sample_rate),
        "-g",
        str(FLUIDSYNTH_GAIN),
        "-o",
        "synth.chorus.active=0",
        "-o",
        "synth.reverb.active=1",
        "-o",
        f"synth.reverb.room-size={FLUIDSYNTH_REVERB_ROOM_SIZE}",
        "-o",
        f"synth.reverb.damp={FLUIDSYNTH_REVERB_DAMP}",
        "-o",
        f"synth.reverb.width={FLUIDSYNTH_REVERB_WIDTH}",
        "-o",
        f"synth.reverb.level={FLUIDSYNTH_REVERB_LEVEL}",
        "-o",
        "audio.file.format=s16",
        "-o",
        "audio.file.endian=little",
        str(soundfont_path),
        str(smf_path),
    ]
    t0 = time.perf_counter()
    try:
        proc = safe_run(cmd, timeout_s=timeout_s)
    except SubprocessTimeoutError as exc:
        raise AudioRenderError(
            AudioRenderErrorCode.FLUIDSYNTH_FAILED,
            f"fluidsynth exceeded {timeout_s}s timeout",
        ) from exc
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise AudioRenderError(
            AudioRenderErrorCode.FLUIDSYNTH_FAILED,
            f"fluidsynth rc={proc.returncode} stderr={proc.stderr.strip()[:200]}",
        )
    if not out_wav_path.exists():
        raise AudioRenderError(
            AudioRenderErrorCode.FLUIDSYNTH_FAILED,
            "fluidsynth returned 0 but no WAV file was written",
        )
    version = _read_fluidsynth_version(bin_path)
    # Hash the binary itself (the FFmpeg audit pattern), not its path
    # string. The binary is effectively immutable for the process
    # lifetime, so cache the digest.
    build_sha = sha256_file(Path(bin_path), cached=True)
    logger.info("fluidsynth rendered %s -> %s in %.1fs", smf_path, out_wav_path, elapsed)
    return version, build_sha


def _read_fluidsynth_version(bin_path: str) -> str:
    """Read fluidsynth version line (or empty string on failure)."""
    try:
        proc = safe_run([bin_path, "--version"], timeout_s=5.0)
        return (
            (proc.stdout + proc.stderr).splitlines()[0].strip()
            if (proc.stdout or proc.stderr)
            else ""
        )
    except Exception:
        return ""


# ---------------------------------------------------------------------------
# FFmpeg encode (WAV -> OGG Opus)
# ---------------------------------------------------------------------------


def find_ffmpeg(explicit: str | None = None) -> str:
    """Locate the ffmpeg binary. Audits it before returning (per §10 #7).

    Raises `AudioRenderError(FFMPEG_MISSING)` if not found, or
    `FFMPEG_AUDIT_FAILED` if the located binary fails the audit.
    """
    import os

    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    env_path = os.environ.get("SAIMC_RENDER_FFMPEG")
    if env_path:
        candidates.append(env_path)
    which = shutil.which("ffmpeg")
    if which:
        candidates.append(which)

    for candidate in candidates:
        if Path(candidate).exists() and Path(candidate).is_file():
            return candidate
    raise AudioRenderError(
        AudioRenderErrorCode.FFMPEG_MISSING,
        "ffmpeg not found on PATH; set SAIMC_RENDER_FFMPEG or run scripts/build_ffmpeg.sh",
    )


def _loudnorm_measure(
    bin_path: str, wav_path: Path, *, timeout_s: float
) -> Mapping[str, str] | None:
    """First pass of the two-pass loudnorm: measure the input's loudness.

    Returns the loudnorm measurement JSON (input_i, input_tp, input_lra,
    input_thresh, target_offset) or None if the measurement pass could
    not run or the input is effectively silent (an -inf measurement
    cannot drive the linear second pass).
    """
    import json

    cmd = [
        bin_path,
        "-hide_banner",
        "-i",
        str(wav_path),
        "-af",
        (
            f"loudnorm=I={MASTER_LOUDNESS_LUFS}:TP={MASTER_TRUE_PEAK_DBTP}"
            f":LRA={MASTER_LRA}:print_format=json"
        ),
        "-f",
        "null",
        "-",
    ]
    try:
        proc = safe_run(cmd, timeout_s=timeout_s)
    except SubprocessTimeoutError:
        logger.warning("loudnorm measurement timed out; falling back to single-pass")
        return None
    if proc.returncode != 0:
        logger.warning("loudnorm measurement rc=%s; falling back to single-pass", proc.returncode)
        return None
    text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        logger.warning("loudnorm measurement JSON unreadable; falling back to single-pass")
        return None
    if not isinstance(data, dict):
        return None
    required = ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")
    if not all(key in data for key in required):
        return None
    measured = {key: str(data[key]) for key in required}
    if any("-inf" in value for value in measured.values()):
        return None
    return measured


def _loudnorm_filter(measured: Mapping[str, str] | None) -> str:
    """Build the loudnorm filter string for the encode pass.

    With measurements, `linear=true` applies a single gain so the mix's
    dynamics are untouched (true mastering); without them the filter
    degrades to single-pass dynamic loudnorm, which still targets the
    same loudness.
    """
    base = (
        f"loudnorm=I={MASTER_LOUDNESS_LUFS}"
        f":TP={MASTER_TRUE_PEAK_DBTP}:LRA={MASTER_LRA}"
    )
    if measured is None:
        return base
    return (
        base
        + f":measured_I={measured['input_i']}"
        + f":measured_TP={measured['input_tp']}"
        + f":measured_LRA={measured['input_lra']}"
        + f":measured_thresh={measured['input_thresh']}"
        + f":offset={measured['target_offset']}"
        + ":linear=true"
    )


def encode_opus(
    wav_path: Path,
    out_ogg_path: Path,
    *,
    ffmpeg_bin: str | None = None,
    bitrate_kbps: int = 128,
    timeout_s: float = DEFAULT_FFMPEG_TIMEOUT_S,
    soundfont_name: str | None = None,
) -> tuple[str, str, str]:
    """Encode WAV -> OGG Opus via the audited ffmpeg binary.

    The encode runs the two-pass loudnorm master first (measure, then
    encode with the measured values applied linearly) so every
    deliverable lands at the same loudness regardless of which font
    rendered it. Returns (version, build_sha, configuration_line) for
    the manifest toolchain block. The OGG carries the rendered font's
    attribution as Vorbis comments (§10 #12); `soundfont_name` picks
    which attribution record to embed (None keeps the Salamander
    default).
    """
    bin_path = find_ffmpeg(ffmpeg_bin)
    audit = audit_ffmpeg(bin_path)
    if not audit.ok:
        raise AudioRenderError(
            AudioRenderErrorCode.FFMPEG_AUDIT_FAILED,
            f"ffmpeg at {bin_path} failed audit: {audit.reasons}",
        )
    out_ogg_path.parent.mkdir(parents=True, exist_ok=True)
    from saimc.render.attribution import audio_metadata_tags

    measured = _loudnorm_measure(bin_path, wav_path, timeout_s=timeout_s)
    cmd = [
        bin_path,
        "-y",  # overwrite output if it exists
        "-i",
        str(wav_path),
        "-af",
        _loudnorm_filter(measured),
        "-c:a",
        "libopus",
        "-b:a",
        f"{bitrate_kbps}k",
        "-vbr",
        "on",
    ]
    for tag, value in audio_metadata_tags(soundfont_name).items():
        cmd += ["-metadata", f"{tag}={value}"]
    cmd.append(str(out_ogg_path))
    t0 = time.perf_counter()
    try:
        proc = safe_run(cmd, timeout_s=timeout_s)
    except SubprocessTimeoutError as exc:
        raise AudioRenderError(
            AudioRenderErrorCode.FFMPEG_FAILED,
            f"ffmpeg exceeded {timeout_s}s timeout",
        ) from exc
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise AudioRenderError(
            AudioRenderErrorCode.FFMPEG_FAILED,
            f"ffmpeg rc={proc.returncode} stderr={proc.stderr.strip()[:200]}",
        )
    if not out_ogg_path.exists():
        raise AudioRenderError(
            AudioRenderErrorCode.FFMPEG_FAILED,
            "ffmpeg returned 0 but no OGG file was written",
        )
    logger.info("ffmpeg encoded %s -> %s in %.1fs", wav_path, out_ogg_path, elapsed)
    return audit.version, audit.binary_sha256, audit.configuration_line


# ---------------------------------------------------------------------------
# Top-level orchestration
# ---------------------------------------------------------------------------


def render_audio(
    plan: PerformancePlan,
    *,
    bpm: float,
    soundfont_path: Path,
    out_dir: Path,
    job_id: str,
    tempo_changes: tuple[TempoPoint, ...] = (),
    voice_instruments: Mapping[int, str] | None = None,
    fluidsynth_bin: str | None = None,
    ffmpeg_bin: str | None = None,
    fluidsynth_timeout_s: float = DEFAULT_FLUIDSYNTH_TIMEOUT_S,
    ffmpeg_timeout_s: float = DEFAULT_FFMPEG_TIMEOUT_S,
) -> AudioArtifact:
    """Render a PerformancePlan to WAV + OGG Opus artifacts.

    `out_dir` is the per-job artifacts directory (the storage layer
    has already created it). `job_id` is used to construct artifact
    file names. `tempo_changes` is the NotationScore's tempo map
    (empty for constant-tempo pieces).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    smf_path = out_dir / f"{job_id}.mid"
    wav_path = out_dir / "audio.wav"
    ogg_path = out_dir / "audio.ogg"

    # 1. PerformancePlan -> SMF.
    try:
        smf = build_smf(
            plan,
            bpm=bpm,
            tempo_changes=tempo_changes,
            voice_instruments=voice_instruments,
            soundfont_path=soundfont_path,
        )
        smf.save(str(smf_path))
    except Exception as exc:
        raise AudioRenderError(
            AudioRenderErrorCode.MIDI_BUILD_FAILED,
            f"SMF build failed: {exc}",
        ) from exc

    # 2. SMF + soundfont -> WAV via FluidSynth.
    fs_version, fs_build = run_fluidsynth(
        smf_path,
        soundfont_path,
        wav_path,
        fluidsynth_bin=fluidsynth_bin,
        timeout_s=fluidsynth_timeout_s,
    )

    # 3. WAV -> OGG Opus via audited FFmpeg.
    ffmpeg_version, ffmpeg_sha, ffmpeg_config = encode_opus(
        wav_path,
        ogg_path,
        ffmpeg_bin=ffmpeg_bin,
        timeout_s=ffmpeg_timeout_s,
        soundfont_name=soundfont_path.name,
    )

    primary_sha = _hash_file(wav_path)
    primary_size = _file_size(wav_path)
    ogg_sha = _hash_file(ogg_path)
    ogg_size = _file_size(ogg_path)
    # The soundfont is an immutable asset: cache its digest per process
    # instead of re-hashing ~1 GB on every job.
    soundfont_sha = _hash_file(soundfont_path, cached=True) if soundfont_path.exists() else ""

    return AudioArtifact(
        primary_path=wav_path,
        primary_container="wav",
        primary_codec="pcm_s16le",
        primary_sha256=primary_sha,
        primary_size_bytes=primary_size,
        ogg_path=ogg_path,
        ogg_codec="opus",
        ogg_sha256=ogg_sha,
        ogg_size_bytes=ogg_size,
        fluidsynth_version=fs_version,
        fluidsynth_build_sha=fs_build,
        ffmpeg_version=ffmpeg_version,
        ffmpeg_build_sha=ffmpeg_sha,
        soundfont_name=soundfont_path.name,
        soundfont_sha256=soundfont_sha,
        ffmpeg_configuration=ffmpeg_config,
    )


__all__ = [
    "DEFAULT_FFMPEG_TIMEOUT_S",
    "DEFAULT_FLUIDSYNTH_TIMEOUT_S",
    "AudioArtifact",
    "AudioRenderError",
    "AudioRenderErrorCode",
    "build_smf",
    "encode_opus",
    "find_ffmpeg",
    "find_fluidsynth",
    "render_audio",
    "run_fluidsynth",
]
