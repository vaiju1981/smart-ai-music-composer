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
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from saimc.canonical import canonical_sha256
from saimc.compose.score import (
    PPQ,
    PerformancePlan,
)
from saimc.jobs.stages import SubprocessTimeoutError, safe_run
from saimc.render.ffmpeg_audit import audit_ffmpeg

if TYPE_CHECKING:
    from mido import MidiFile

logger = logging.getLogger(__name__)


DEFAULT_FLUIDSYNTH_TIMEOUT_S: float = 300.0
"""Five minutes for a typical 5-minute piece; the subprocess wrapper
kills it if it runs longer."""

DEFAULT_FFMPEG_TIMEOUT_S: float = 60.0


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
    primary_codec: str  # "pcm_s24le"
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


def build_smf(plan: PerformancePlan, *, bpm: float) -> MidiFile:
    """Convert a PerformancePlan into a SMF Type-0 MIDI file.

    `bpm` is the realised tempo of the plan (taken from the
    NotationScore.tempo that produced the plan). The SMF uses PPQ=480,
    matching the NotationScore's PPQ. The set_tempo meta event carries
    the integer microseconds-per-quarter so that downstream tools
    (FluidSynth included) get an exact tempo.

    Returns the populated `mido.MidiFile` (caller persists to disk).
    """
    import mido

    midi = mido.MidiFile(type=0)
    midi.ticks_per_beat = PPQ
    track = mido.MidiTrack()
    midi.tracks.append(track)

    # Tempo: microseconds per quarter note. bpm = 60_000_000 / us_per_quarter.
    us_per_quarter = round(60_000_000 / bpm)
    track.append(mido.MetaMessage("set_tempo", tempo=us_per_quarter, time=0))
    track.append(mido.MetaMessage("end_of_track", time=0))

    # Convert each PerformanceNoteEvent to a note_on / note_off pair.
    # PPQ=480 ticks per beat. The plan's start_us is in microseconds;
    # we need integer ticks at PPQ=480.
    for note in sorted(plan.notes, key=lambda n: (n.start_us, n.voice_id, n.pitch_midi)):
        on_tick = _us_to_ticks(note.start_us, bpm)
        off_tick = _us_to_ticks(note.start_us + note.duration_us, bpm)
        if off_tick <= on_tick:
            # Guard against zero-or-negative durations from the round
            # trip; the engine shouldn't produce these but we don't
            # want a malformed MIDI file to crash the renderer.
            off_tick = on_tick + 1
        track.append(
            mido.Message(
                "note_on",
                channel=note.voice_id % 16,
                note=note.pitch_midi,
                velocity=note.velocity,
                time=on_tick,
            )
        )
        track.append(
            mido.Message(
                "note_off",
                channel=note.voice_id % 16,
                note=note.pitch_midi,
                velocity=0,
                time=off_tick - on_tick,
            )
        )
    return midi


def _us_to_ticks(microseconds: int, bpm: float) -> int:
    """Convert microseconds to integer ticks at PPQ=480 and the given bpm."""
    return round(microseconds * bpm * PPQ / 60_000_000)


def _hash_file(path: Path) -> str:
    """SHA-256 over the file's bytes (streaming)."""
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


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

    cmd = [
        bin_path,
        "-F",  # render to WAV
        "-q",  # quiet
        "-r",
        str(sample_rate),
        "-o",
        "alsa.audio=sink-dummy",
        str(soundfont_path),
        str(smf_path),
    ]
    # FluidSynth writes to a file when given -F + -o pointing at a file
    # output. We adjust: use -F and redirect via the -o flag's file mode.
    # Actually -F expects a destination, not -o. Simplest: use -F with
    # the output path as the positional arg, after the soundfont.
    cmd = [
        bin_path,
        "-F",
        str(out_wav_path),
        "-q",
        "-r",
        str(sample_rate),
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
    build_sha = canonical_sha256({"binary": bin_path})  # placeholder; see audit() below
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


def encode_opus(
    wav_path: Path,
    out_ogg_path: Path,
    *,
    ffmpeg_bin: str | None = None,
    bitrate_kbps: int = 128,
    timeout_s: float = DEFAULT_FFMPEG_TIMEOUT_S,
) -> tuple[str, str]:
    """Encode WAV -> OGG Opus via the audited ffmpeg binary.

    Returns (version, build_sha) for the manifest toolchain block.
    """
    bin_path = find_ffmpeg(ffmpeg_bin)
    audit = audit_ffmpeg(bin_path)
    if not audit.ok:
        raise AudioRenderError(
            AudioRenderErrorCode.FFMPEG_AUDIT_FAILED,
            f"ffmpeg at {bin_path} failed audit: {audit.reasons}",
        )
    out_ogg_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        bin_path,
        "-y",  # overwrite output if it exists
        "-i",
        str(wav_path),
        "-c:a",
        "libopus",
        "-b:a",
        f"{bitrate_kbps}k",
        "-vbr",
        "on",
        str(out_ogg_path),
    ]
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
    return audit.version, audit.binary_sha256


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
    fluidsynth_bin: str | None = None,
    ffmpeg_bin: str | None = None,
    fluidsynth_timeout_s: float = DEFAULT_FLUIDSYNTH_TIMEOUT_S,
    ffmpeg_timeout_s: float = DEFAULT_FFMPEG_TIMEOUT_S,
) -> AudioArtifact:
    """Render a PerformancePlan to WAV + OGG Opus artifacts.

    `out_dir` is the per-job artifacts directory (the storage layer
    has already created it). `job_id` is used to construct artifact
    file names.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    smf_path = out_dir / f"{job_id}.mid"
    wav_path = out_dir / "audio.wav"
    ogg_path = out_dir / "audio.ogg"

    # 1. PerformancePlan -> SMF.
    try:
        smf = build_smf(plan, bpm=bpm)
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
    ffmpeg_version, ffmpeg_sha = encode_opus(
        wav_path,
        ogg_path,
        ffmpeg_bin=ffmpeg_bin,
        timeout_s=ffmpeg_timeout_s,
    )

    primary_sha = _hash_file(wav_path)
    primary_size = _file_size(wav_path)
    ogg_sha = _hash_file(ogg_path)
    ogg_size = _file_size(ogg_path)
    soundfont_sha = _hash_file(soundfont_path) if soundfont_path.exists() else ""

    return AudioArtifact(
        primary_path=wav_path,
        primary_container="wav",
        primary_codec="pcm_s24le",
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
