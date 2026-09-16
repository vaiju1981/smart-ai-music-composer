"""FFmpeg license/feature audit.

Per `docs/roadmap.md` §4 and §10 #7, every FFmpeg binary that saimc
uses to render media must be auditable as:

- LGPL (or better): NOT configured with `--enable-gpl`.
- Distributable:    NOT configured with `--enable-nonfree`.
- Codec-complete for Phase 1 video: `--enable-libvpx` AND
  `--enable-libopus` (since Phase 1 ships WebM/VP9+Opus per §4).

This module is the single source of truth for that audit. The
release-gate binary (built by `scripts/build_ffmpeg.sh`) passes by
construction. A user-pointed-at binary (e.g. Homebrew `ffmpeg`)
passes only if it was configured the same way; Homebrew's default
formula does NOT pass (it enables libx264 / libx265 which require
GPL, so the binary is GPL).

The audit is intentionally cheap (one subprocess call), runs in well
under a second, and is safe to invoke at every renderer startup.
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

REQUIRED_CODEC_FLAGS: tuple[str, ...] = ("--enable-libvpx", "--enable-libopus")
FORBIDDEN_LICENSE_FLAGS: tuple[str, ...] = ("--enable-gpl", "--enable-nonfree")


class FfmpegLicense(StrEnum):
    """The license posture of an FFmpeg build, derived from its `configuration:` line."""

    LGPL = "lgpl"
    GPL = "gpl"
    NON_FREE = "non-free"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FfmpegAuditResult:
    """The structured outcome of auditing an FFmpeg binary."""

    binary_path: str
    version: str
    binary_sha256: str
    configuration_line: str
    license: FfmpegLicense
    enabled_codecs: tuple[str, ...]
    missing_required_codecs: tuple[str, ...]
    forbidden_flags_present: tuple[str, ...]
    ok: bool
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "binary_path": self.binary_path,
            "version": self.version,
            "binary_sha256": self.binary_sha256,
            "configuration_line": self.configuration_line,
            "license": self.license.value,
            "enabled_codecs": list(self.enabled_codecs),
            "missing_required_codecs": list(self.missing_required_codecs),
            "forbidden_flags_present": list(self.forbidden_flags_present),
            "ok": self.ok,
            "reasons": list(self.reasons),
        }


_VERSION_RE = re.compile(r"ffmpeg version (\S+)")
_CONFIG_LINE_RE = re.compile(r"^\s*configuration:\s*(.+)$", re.MULTILINE)

# (path, size, mtime_ns) -> result. The audit runs twice per job (audio
# encode + animation encode) against the same unchanging binary; the
# result is memoized so the second call and the binary SHA-256 are free.
_AUDIT_CACHE: dict[tuple[str, int, int], FfmpegAuditResult] = {}


def audit_ffmpeg(binary_path: str, *, timeout_s: float = 5.0) -> FfmpegAuditResult:
    """Run `binary_path -version` and audit the configuration line.

    Returns a structured result. `result.ok` is True iff the binary is
    LGPL, has every required codec, and has no non-free flags. Results
    are memoized per binary (path, size, mtime) — safe because the
    audited configuration cannot change without the file changing.
    """
    import os

    try:
        stat = os.stat(binary_path)
        cache_key = (str(binary_path), stat.st_size, stat.st_mtime_ns)
    except OSError:
        cache_key = None
    if cache_key is not None and cache_key in _AUDIT_CACHE:
        return _AUDIT_CACHE[cache_key]

    result = _audit_ffmpeg_uncached(binary_path, timeout_s=timeout_s)
    if cache_key is not None:
        _AUDIT_CACHE[cache_key] = result
    return result


def _audit_ffmpeg_uncached(binary_path: str, *, timeout_s: float) -> FfmpegAuditResult:
    """Run `binary_path -version` and audit the configuration line."""
    version_proc = subprocess.run(
        [binary_path, "-version"],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if version_proc.returncode != 0:
        return FfmpegAuditResult(
            binary_path=binary_path,
            version="",
            binary_sha256=_sha256_file(binary_path),
            configuration_line="",
            license=FfmpegLicense.UNKNOWN,
            enabled_codecs=(),
            missing_required_codecs=REQUIRED_CODEC_FLAGS,
            forbidden_flags_present=(),
            ok=False,
            reasons=(f"ffmpeg -version exited rc={version_proc.returncode}",),
        )
    stdout = version_proc.stdout

    version_match = _VERSION_RE.search(stdout)
    version = version_match.group(1) if version_match else "unknown"

    config_match = _CONFIG_LINE_RE.search(stdout)
    configuration_line = config_match.group(1) if config_match else ""

    flags = _parse_flags(configuration_line)
    enabled_codecs = tuple(f for f in flags if f.startswith("--enable-"))
    forbidden_present = tuple(f for f in flags if f in FORBIDDEN_LICENSE_FLAGS)
    missing = tuple(f for f in REQUIRED_CODEC_FLAGS if f not in flags)

    license_ = _derive_license(forbidden_present)

    reasons: list[str] = []
    if forbidden_present:
        reasons.append(
            f"forbidden flags present: {sorted(forbidden_present)} "
            f"(binary is {license_.value}, not LGPL)"
        )
    if missing:
        reasons.append(f"missing required codec flags: {sorted(missing)}")

    return FfmpegAuditResult(
        binary_path=binary_path,
        version=version,
        binary_sha256=_sha256_file(binary_path),
        configuration_line=configuration_line,
        license=license_,
        enabled_codecs=enabled_codecs,
        missing_required_codecs=missing,
        forbidden_flags_present=forbidden_present,
        ok=not reasons,
        reasons=tuple(reasons),
    )


def _parse_flags(configuration_line: str) -> frozenset[str]:
    """Tokenise the FFmpeg `configuration:` line into a frozenset of flag tokens.

    Each flag starts with `--``. The FFmpeg configure output uses single-
    dash flags (`--enable-libvpx`, etc.) — we keep both forms for safety.
    """
    if not configuration_line:
        return frozenset()
    tokens = configuration_line.split()
    return frozenset(t for t in tokens if t.startswith("-"))


def _derive_license(forbidden_flags: Iterable[str]) -> FfmpegLicense:
    flags = set(forbidden_flags)
    if "--enable-nonfree" in flags:
        return FfmpegLicense.NON_FREE
    if "--enable-gpl" in flags:
        return FfmpegLicense.GPL
    return FfmpegLicense.LGPL


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
    except OSError:
        return ""
    return h.hexdigest()


__all__ = [
    "FORBIDDEN_LICENSE_FLAGS",
    "REQUIRED_CODEC_FLAGS",
    "FfmpegAuditResult",
    "FfmpegLicense",
    "audit_ffmpeg",
]
