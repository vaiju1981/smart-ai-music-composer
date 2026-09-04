"""Animation renderer: PerformancePlan -> piano-roll frames -> WebM (VP9 + Opus).

Per `docs/roadmap.md` §2 step 4c and §4, the animation pipeline is:

    PerformancePlan (integer-microsecond timestamps)
      -> piano-roll frames (Pillow, drawn from the same event schedule
         the audio renderer plays)
      -> audited FFmpeg -> WebM (VP9 video + Opus audio from the WAV)

The piano-roll and the audio share the PerformancePlan schedule, which
is what makes the §8 audio/animation synchronization tolerance (±20 ms
after declared offsets) checkable.

`moviepy` is deliberately not used for the actual pipeline: it would
render through its own bundled ffmpeg, which cannot pass the project's
license audit gate. Frames go straight to the audited FFmpeg binary.
"""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from saimc.compose.score import PerformancePlan
from saimc.jobs.stages import SubprocessTimeoutError, safe_run
from saimc.render.audio import (
    AudioRenderError,
    _file_size,
    _hash_file,
    find_ffmpeg,
)
from saimc.render.ffmpeg_audit import audit_ffmpeg

logger = logging.getLogger(__name__)

DEFAULT_ANIM_FPS: int = 24
DEFAULT_ANIM_TIMEOUT_S: float = 600.0
"""VP9 is a slow encoder; five minutes of video can take minutes to
encode. Six hundred seconds keeps a stuck encode from blocking the
pipeline while leaving room for the §8 performance gate."""

FRAME_WIDTH: int = 1280
FRAME_HEIGHT: int = 720
PIXELS_PER_SECOND: int = 48
# The playhead sits a quarter of the way in from the left edge, so the
# viewer sees what is coming as well as what has passed.
PLAYHEAD_FRACTION: float = 0.25

BACKGROUND_RGB = (16, 18, 24)
NOTE_RGB = (86, 156, 214)
PLAYHEAD_RGB = (232, 232, 232)


class AnimationRenderErrorCode:
    """Stable error codes for the animation renderer."""

    FFMPEG_MISSING = "ffmpeg_missing"
    FFMPEG_AUDIT_FAILED = "ffmpeg_audit_failed"
    FFMPEG_FAILED = "ffmpeg_failed"
    AUDIO_MISSING = "animation_audio_missing"
    FRAME_RENDER_FAILED = "animation_frames_failed"


class AnimationRenderError(Exception):
    """Raised when animation rendering fails. Carries a stable code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


@dataclass(frozen=True)
class AnimationArtifact:
    """The rendered WebM plus the provenance the manifest needs."""

    webm_path: Path
    container: str  # "webm"
    codec: str  # "vp9"
    audio_codec: str  # "opus"
    sha256: str
    size_bytes: int
    width: int
    height: int
    fps: int
    ffmpeg_version: str = ""
    ffmpeg_build_sha: str = ""
    ffmpeg_configuration: str = ""


def total_frames(duration_s: float, fps: int) -> int:
    """One extra frame so the final instant is shown settled."""
    return max(1, int(duration_s * fps) + 1)


def render_frames(
    plan: PerformancePlan,
    frames_dir: Path,
    *,
    fps: int = DEFAULT_ANIM_FPS,
    width: int = FRAME_WIDTH,
    height: int = FRAME_HEIGHT,
    pixels_per_second: int = PIXELS_PER_SECOND,
) -> int:
    """Draw one PNG per frame of a scrolling piano-roll.

    The viewport slides along the piece at `pixels_per_second`; the
    playhead is fixed at `PLAYHEAD_FRACTION` of the frame width. Note
    rectangles are placed from the plan's integer-microsecond schedule,
    shared with the audio renderer. Returns the number of frames drawn.
    """
    from PIL import Image, ImageDraw

    notes = sorted(plan.notes, key=lambda n: n.start_us)
    if not notes:
        raise AnimationRenderError(
            AnimationRenderErrorCode.FRAME_RENDER_FAILED,
            "PerformancePlan has no notes; nothing to animate",
        )
    duration_s = max(n.start_us + n.duration_us for n in notes) / 1_000_000
    count = total_frames(duration_s, fps)

    pitches = [n.pitch_midi for n in notes]
    pitch_lo, pitch_hi = min(pitches), max(pitches)
    # A lane height of >=8px keeps sparse pieces readable; denser spans
    # simply compress vertically.
    lane_height = max(8, (height - 40) // max(1, pitch_hi - pitch_lo + 1))

    frames_dir.mkdir(parents=True, exist_ok=True)
    playhead_x = int(width * PLAYHEAD_FRACTION)
    window_s = (width - playhead_x) / pixels_per_second

    for frame_idx in range(count):
        t = frame_idx / fps
        img = Image.new("RGB", (width, height), BACKGROUND_RGB)
        draw = ImageDraw.Draw(img)
        for note in notes:
            start_s = note.start_us / 1_000_000
            end_s = (note.start_us + note.duration_us) / 1_000_000
            x0 = playhead_x + (start_s - t) * pixels_per_second
            x1 = playhead_x + (end_s - t) * pixels_per_second
            if x1 < 0 or x0 > width or start_s > t + window_s:
                continue
            lane = height - 20 - (note.pitch_midi - pitch_lo) * lane_height
            draw.rectangle(
                (max(0, int(x0)), lane, min(width - 1, int(x1)), lane + lane_height - 2),
                fill=NOTE_RGB,
            )
        draw.line((playhead_x, 0, playhead_x, height), fill=PLAYHEAD_RGB, width=2)
        img.save(frames_dir / f"{frame_idx:06d}.png")
    return count


def encode_webm(
    frames_dir: Path,
    audio_path: Path,
    out_webm_path: Path,
    *,
    fps: int = DEFAULT_ANIM_FPS,
    ffmpeg_bin: str | None = None,
    timeout_s: float = DEFAULT_ANIM_TIMEOUT_S,
) -> tuple[str, str, str]:
    """Encode frames + audio -> WebM (VP9 + Opus) via the audited ffmpeg.

    Returns (version, build_sha, configuration_line) for the manifest
    toolchain block.
    """
    try:
        bin_path = find_ffmpeg(ffmpeg_bin)
    except AudioRenderError as exc:
        raise AnimationRenderError(exc.code, exc.message) from exc
    audit = audit_ffmpeg(bin_path)
    if not audit.ok:
        raise AnimationRenderError(
            AnimationRenderErrorCode.FFMPEG_AUDIT_FAILED,
            f"ffmpeg at {bin_path} failed audit: {audit.reasons}",
        )
    if not audio_path.exists():
        raise AnimationRenderError(
            AnimationRenderErrorCode.AUDIO_MISSING,
            f"animation audio track not found at {audio_path}",
        )
    out_webm_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        bin_path,
        "-y",
        "-framerate",
        str(fps),
        "-i",
        str(frames_dir / "%06d.png"),
        "-i",
        str(audio_path),
        "-c:v",
        "libvpx-vp9",
        "-crf",
        "34",
        "-b:v",
        "0",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "libopus",
        "-b:a",
        "128k",
        "-shortest",
        str(out_webm_path),
    ]
    t0 = time.perf_counter()
    try:
        proc = safe_run(cmd, timeout_s=timeout_s)
    except SubprocessTimeoutError as exc:
        raise AnimationRenderError(
            AnimationRenderErrorCode.FFMPEG_FAILED,
            f"ffmpeg exceeded {timeout_s}s timeout",
        ) from exc
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise AnimationRenderError(
            AnimationRenderErrorCode.FFMPEG_FAILED,
            f"ffmpeg rc={proc.returncode} stderr={(proc.stderr or '').strip()[:200]}",
        )
    if not out_webm_path.exists():
        raise AnimationRenderError(
            AnimationRenderErrorCode.FFMPEG_FAILED,
            "ffmpeg returned 0 but no WebM file was written",
        )
    logger.info(
        "ffmpeg encoded %s frames -> %s in %.1fs",
        len(list(frames_dir.glob("*.png"))),
        out_webm_path,
        elapsed,
    )
    return audit.version, audit.binary_sha256, audit.configuration_line


def render_animation(
    plan: PerformancePlan,
    *,
    audio_wav_path: Path,
    out_dir: Path,
    ffmpeg_bin: str | None = None,
    fps: int = DEFAULT_ANIM_FPS,
    timeout_s: float = DEFAULT_ANIM_TIMEOUT_S,
) -> AnimationArtifact:
    """Render a PerformancePlan to a WebM piano-roll animation.

    `audio_wav_path` is the audio stage's WAV — the animation's audio
    track — so video and audio come from the same render. Frame PNGs
    are written under `out_dir/frames/` and removed after the encode.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = out_dir / "frames"
    webm_path = out_dir / "animation.webm"

    try:
        render_frames(plan, frames_dir, fps=fps)
    except AnimationRenderError:
        raise
    except Exception as exc:  # Pillow itself never raises our error type
        raise AnimationRenderError(
            AnimationRenderErrorCode.FRAME_RENDER_FAILED,
            f"frame rendering failed: {exc}",
        ) from exc

    ffmpeg_version, ffmpeg_sha, ffmpeg_config = encode_webm(
        frames_dir,
        audio_wav_path,
        webm_path,
        fps=fps,
        ffmpeg_bin=ffmpeg_bin,
        timeout_s=timeout_s,
    )

    shutil.rmtree(frames_dir, ignore_errors=True)

    return AnimationArtifact(
        webm_path=webm_path,
        container="webm",
        codec="vp9",
        audio_codec="opus",
        sha256=_hash_file(webm_path),
        size_bytes=_file_size(webm_path),
        width=FRAME_WIDTH,
        height=FRAME_HEIGHT,
        fps=fps,
        ffmpeg_version=ffmpeg_version,
        ffmpeg_build_sha=ffmpeg_sha,
        ffmpeg_configuration=ffmpeg_config,
    )


__all__ = [
    "DEFAULT_ANIM_FPS",
    "DEFAULT_ANIM_TIMEOUT_S",
    "AnimationArtifact",
    "AnimationRenderError",
    "AnimationRenderErrorCode",
    "encode_webm",
    "render_animation",
    "render_frames",
    "total_frames",
]
