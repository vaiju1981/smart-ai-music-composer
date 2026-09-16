#!/usr/bin/env python3
"""CLI: audit an FFmpeg binary against the §4 / §10 #7 policy.

Usage:
    scripts/audit_ffmpeg.py [path-to-ffmpeg]

Defaults to `ffmpeg` on $PATH. Exits non-zero if the binary fails
the audit; prints a structured JSON report on success or failure.
"""

from __future__ import annotations

import json
import shutil
import sys

from saimc.render.ffmpeg_audit import audit_ffmpeg


def main() -> int:
    binary = sys.argv[1] if len(sys.argv) > 1 else shutil.which("ffmpeg") or "ffmpeg"
    result = audit_ffmpeg(binary)
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
