"""Animation renderer: PerformancePlan -> piano-roll WebM (VP9 + Opus).

Per `docs/roadmap.md` §2 step 4c and §4, the animation pipeline is:

    PerformancePlan (integer-microsecond timestamps)
      -> one wide piano-roll image (Pillow, drawn from the same event
         schedule the audio renderer plays)
      -> per-frame crops streamed as raw RGB into the audited FFmpeg
      -> WebM (VP9 video + Opus audio from the WAV)

The piano-roll and the audio share the PerformancePlan schedule, which
is what makes the §8 audio/animation synchronization tolerance (±20 ms
after declared offsets) checkable.

Two performance decisions keep long pieces off the critical path:

- The roll is drawn ONCE at full width and each frame is a crop plus a
  playhead line. The viewport is a constant-speed scroll, so per-frame
  redrawing of every note rectangle (O(frames x notes)) is pure waste.
- Frames are streamed to FFmpeg's stdin as raw RGB instead of being
  PNG-encoded to disk and re-read via an image2 glob — the PNG round
  trip cost seconds per frame-minute and ~1 GB of disk for a 5-minute
  piece, for zero visual difference.

`moviepy` is deliberately not used for the actual pipeline: it would
render through its own bundled ffmpeg, which cannot pass the project's
license audit gate. Frames go straight to the audited FFmpeg binary.
"""

from __future__ import annotations

import logging
import subprocess
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from saimc.compose.score import PerformancePlan
from saimc.render.audio import AudioRenderError, _file_size, find_ffmpeg
from saimc.render.ffmpeg_audit import audit_ffmpeg
from saimc.render.util import sha256_file

if TYPE_CHECKING:
    from PIL import Image

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


def build_roll(
    plan: PerformancePlan,
    *,
    fps: int = DEFAULT_ANIM_FPS,
    width: int = FRAME_WIDTH,
    height: int = FRAME_HEIGHT,
    pixels_per_second: int = PIXELS_PER_SECOND,
) -> Image.Image:
    """Draw the full piano-roll once into a wide image.

    The roll's x-axis is `pixels_per_second` starting at the playhead
    offset, so the frame at time t is simply the crop at `t * pps`. A
    lane height of >=8px keeps sparse pieces readable; denser spans
    simply compress vertically.
    """
    from PIL import Image, ImageDraw

    notes = sorted(plan.notes, key=lambda n: n.start_us)
    if not notes:
        raise AnimationRenderError(
            AnimationRenderErrorCode.FRAME_RENDER_FAILED,
            "PerformancePlan has no notes; nothing to animate",
        )
    duration_s = max(n.start_us + n.duration_us for n in notes) / 1_000_000
    playhead_x = int(width * PLAYHEAD_FRACTION)
    roll_width = width + int(duration_s * pixels_per_second)

    pitches = [n.pitch_midi for n in notes]
    pitch_lo, pitch_hi = min(pitches), max(pitches)
    lane_height = max(8, (height - 40) // max(1, pitch_hi - pitch_lo + 1))

    img = Image.new("RGB", (roll_width, height), BACKGROUND_RGB)
    draw = ImageDraw.Draw(img)
    for note in notes:
        start_s = note.start_us / 1_000_000
        end_s = (note.start_us + note.duration_us) / 1_000_000
        # Clamp before culling: a degenerate rectangle (right <= left)
        # is rejected by Pillow.
        left = max(0, int(playhead_x + start_s * pixels_per_second))
        right = min(roll_width - 1, int(playhead_x + end_s * pixels_per_second))
        if right <= left:
            continue
        lane = height - 20 - (note.pitch_midi - pitch_lo) * lane_height
        draw.rectangle((left, lane, right, lane + lane_height - 2), fill=NOTE_RGB)
    return img


def iter_roll_frames(
    roll: Image.Image,
    *,
    count: int,
    fps: int = DEFAULT_ANIM_FPS,
    width: int = FRAME_WIDTH,
    height: int = FRAME_HEIGHT,
    pixels_per_second: int = PIXELS_PER_SECOND,
) -> Iterator[bytes]:
    """Yield one raw RGB frame per tick of the scrolling viewport."""
    from PIL import ImageDraw

    playhead_x = int(width * PLAYHEAD_FRACTION)
    for frame_idx in range(count):
        t = frame_idx / fps
        frame = roll.crop(
            (int(t * pixels_per_second), 0, int(t * pixels_per_second) + width, height)
        )
        draw = ImageDraw.Draw(frame)
        draw.line((playhead_x, 0, playhead_x, height), fill=PLAYHEAD_RGB, width=2)
        yield frame.tobytes()


def _plan_duration_s(plan: PerformancePlan) -> float:
    if not plan.notes:
        raise AnimationRenderError(
            AnimationRenderErrorCode.FRAME_RENDER_FAILED,
            "PerformancePlan has no notes; nothing to animate",
        )
    return max(n.start_us + n.duration_us for n in plan.notes) / 1_000_000


def encode_webm(
    frames: Iterator[bytes],
    audio_path: Path,
    out_webm_path: Path,
    *,
    width: int = FRAME_WIDTH,
    height: int = FRAME_HEIGHT,
    fps: int = DEFAULT_ANIM_FPS,
    ffmpeg_bin: str | None = None,
    timeout_s: float = DEFAULT_ANIM_TIMEOUT_S,
) -> tuple[str, str, str]:
    """Stream raw RGB frames + audio -> WebM (VP9 + Opus) via the audited ffmpeg.

    Frames are written to ffmpeg's stdin (`-f rawvideo`), so nothing
    touches the filesystem on the video side. Returns (version,
    build_sha, configuration_line) for the manifest toolchain block.
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
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "-",
        "-i",
        str(audio_path),
        "-c:v",
        "libvpx-vp9",
        # Defaults are "good" deadline at speed 0 — the slowest possible
        # setting. Speed 4 is the sweet spot for this content: typically
        # several times faster with no visible quality change at crf 34.
        "-deadline",
        "good",
        "-cpu-used",
        "4",
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
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise AnimationRenderError(
            AnimationRenderErrorCode.FFMPEG_MISSING,
            f"ffmpeg executable not found: {exc}",
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise AnimationRenderError(
            AnimationRenderErrorCode.FFMPEG_FAILED,
            f"ffmpeg did not start within {timeout_s}s",
        ) from exc

    assert proc.stdin is not None
    assert proc.stderr is not None
    stderr_text = ""
    try:
        try:
            for chunk in frames:
                proc.stdin.write(chunk)
            proc.stdin.close()
        except BrokenPipeError:
            # ffmpeg exited early (bad flag, unreadable input...); the
            # wait() below collects its rc and stderr for the error.
            pass
        stderr_text = proc.stderr.read().decode(errors="replace").strip()
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired as exc:
        proc.kill()
        raise AnimationRenderError(
            AnimationRenderErrorCode.FFMPEG_FAILED,
            f"ffmpeg exceeded {timeout_s}s timeout",
        ) from exc
    finally:
        if proc.stdin is not None and not proc.stdin.closed:
            proc.stdin.close()

    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        raise AnimationRenderError(
            AnimationRenderErrorCode.FFMPEG_FAILED,
            f"ffmpeg rc={proc.returncode} stderr={stderr_text[:200]}",
        )
    if not out_webm_path.exists():
        raise AnimationRenderError(
            AnimationRenderErrorCode.FFMPEG_FAILED,
            "ffmpeg returned 0 but no WebM file was written",
        )
    logger.info("ffmpeg encoded WebM in %.1fs (%dx%d @ %dfps)", elapsed, width, height, fps)
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
    track — so video and audio come from the same render.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    webm_path = out_dir / "animation.webm"
    width, height = FRAME_WIDTH, FRAME_HEIGHT

    duration_s = _plan_duration_s(plan)
    count = total_frames(duration_s, fps)
    roll = build_roll(plan, fps=fps, width=width, height=height)
    frames = iter_roll_frames(roll, count=count, fps=fps, width=width, height=height)

    ffmpeg_version, ffmpeg_sha, ffmpeg_config = encode_webm(
        frames,
        audio_wav_path,
        webm_path,
        width=width,
        height=height,
        fps=fps,
        ffmpeg_bin=ffmpeg_bin,
        timeout_s=timeout_s,
    )

    return AnimationArtifact(
        webm_path=webm_path,
        container="webm",
        codec="vp9",
        audio_codec="opus",
        sha256=sha256_file(webm_path),
        size_bytes=_file_size(webm_path),
        width=width,
        height=height,
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
    "build_roll",
    "encode_webm",
    "iter_roll_frames",
    "render_animation",
    "total_frames",
]
