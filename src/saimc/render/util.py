"""Shared helpers for the render modules."""

from __future__ import annotations

import hashlib
from pathlib import Path

# (resolved path, size, mtime_ns) -> hex digest. Immutable assets like
# the soundfont and the ffmpeg binary are hashed on every job otherwise;
# a 1.2 GB SF2 takes seconds of SHA-256 that repeat for no reason.
_HASH_CACHE: dict[tuple[str, int, int], str] = {}


def sha256_file(path: Path, *, cached: bool = False) -> str:
    """SHA-256 over the file's bytes (streaming).

    With `cached=True` the digest is memoized per (path, size, mtime);
    use it only for assets that are effectively immutable while the
    process runs (soundfont, ffmpeg binary) — never for artifacts,
    which can be rewritten between renders.
    """
    stat = path.stat()
    key = (str(path.resolve()), stat.st_size, stat.st_mtime_ns)
    if cached and key in _HASH_CACHE:
        return _HASH_CACHE[key]
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if cached:
        _HASH_CACHE[key] = digest
    return digest


__all__ = ["sha256_file"]
