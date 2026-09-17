"""Canonical JSON serialization for canonical artifacts.

Per `docs/roadmap.md` §6 "Canonical symbolic serialization":

- UTF-8 JSON, sorted object keys, no insignificant whitespace,
  and no NaN/Infinity values;
- integer musical ticks for score positions and durations;
- integer microseconds for realized performance timestamps;
- integer MIDI pitches and velocities;
- arrays remain in musically significant order, with deterministic
  secondary sorting where events share a timestamp;
- an explicit format version in each document.

This module defines the serializer used to hash CompositionSpec,
NotationScore, and PerformancePlan (per §6: these are the canonical,
byte-identical artifacts). Media artifacts (WAV, SVG, MP4/WebM, …)
are not canonical and are verified semantically per §8.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from typing import Any, Final

CANONICAL_FORMAT_VERSION: Final[int] = 1
"""Bump on any change to canonical-JSON encoding rules."""

_SEPARATOR: Final[str] = ":"
"""Format-version separator: {kind}{sep}{v}."""

NA_INF_MESSAGE: Final[str] = (
    "NaN and Infinity are not allowed in canonical JSON artifacts (per §6)."
)


def _reject_nonfinite(value: Any) -> None:
    """Recursively reject float NaN/Infinity, raising ValueError if present."""
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            raise ValueError(NA_INF_MESSAGE)
    elif isinstance(value, Mapping):
        for nested in value.values():
            _reject_nonfinite(nested)
    elif isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value:
            _reject_nonfinite(nested)


def canonical_dumps(value: Any) -> str:
    """Serialize `value` to the canonical UTF-8 JSON form.

    - UTF-8 bytestring compatible (returned as `str`).
    - Object keys sorted lexicographically.
    - No insignificant whitespace (`separators=(",", ":")`).
    - NaN/Infinity rejected.
    """
    _reject_nonfinite(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def canonical_bytes(value: Any) -> bytes:
    """Canonical UTF-8 bytes for `value`."""
    return canonical_dumps(value).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    """Hex SHA-256 over the canonical bytes for `value`."""
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def manifest_document(kind: str, body: Mapping[str, Any]) -> dict[str, Any]:
    """Wrap a canonical body with its format-version tag.

    The format-version tag is `"<kind>:{CANONICAL_FORMAT_VERSION}"`. It is
    written first here, but "first when sorted" is a property of the
    *body*, not of this function: it holds for `NotationScore` and
    `PerformancePlan`, whose keys all sort after `format`, and not for
    `CompositionPlan`, whose `arc_min_reps`, `bass_figures` and
    `cadence_degree` sort before it. §6 requires an explicit version in
    every document, not a leading one, so nothing depends on the order —
    but a consumer that wants to dispatch on the tag without parsing the
    rest has to read it by name rather than assume it comes first.

    This function is unused in production: every canonical document in the
    repo inlines its own `format` string, because each carries a version
    that tracks its own shape rather than the encoding rules.
    """
    if not kind:
        raise ValueError("kind must be a non-empty string")
    if ":" in kind:
        raise ValueError("kind must not contain ':'")
    return {
        "format": f"{kind}{_SEPARATOR}{CANONICAL_FORMAT_VERSION}",
        **body,
    }


__all__ = [
    "CANONICAL_FORMAT_VERSION",
    "canonical_bytes",
    "canonical_dumps",
    "canonical_sha256",
    "manifest_document",
]
