"""Console-script wrapper around the pinned FFmpeg build script.

The build itself lives in `scripts/build_ffmpeg.sh` (POSIX shell): it
downloads the pinned FFmpeg 8.1.2 source, verifies its SHA-256, and
compiles an LGPL build with libvpx + libopus into
`vendor/ffmpeg-build/`, writing release artifacts to
`dist/ffmpeg/8.1.2/`. This wrapper exists so the declared
`saimc-build-ffmpeg` entry point (pyproject.toml) works from the venv.

Run from the repo root; the shell script resolves its own paths.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_ffmpeg.sh"


def main() -> int:
    if not BUILD_SCRIPT.is_file():
        print(f"build script not found: {BUILD_SCRIPT}", file=sys.stderr)
        return 1
    completed = subprocess.run([str(BUILD_SCRIPT)])
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
