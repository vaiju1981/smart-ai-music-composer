"""Deterministic fallback parser for the Phase 1 mood vocabulary.

Per `docs/roadmap.md` §6: when Ollama is unavailable, rate-limited, or fails
schema validation after 2 repairs, the fallback parser is used for in-vocabulary
requests. Out-of-vocabulary requests fail with a structured `SpecError`.

This parser is intentionally narrow:

- Phase 1 moods only: ``calming | electrifying | sleep``.
- Optional duration as ``<number> <unit>`` (e.g. ``5 min``, ``30s``, ``an hour``,
  ``one minute``); if absent, defaults to 180s.
- No tempo, no key, no famous-piece. Those are out of scope and rejected.
- Multi-language mood keywords are not supported in Phase 1.

The parser is regex-driven, not LLM-driven, so it is deterministic and has
zero external dependencies. Determinism is part of the reproducibility
contract: any prompt it accepts yields the same CompositionSpec every time.
"""

from __future__ import annotations

import re

from saimc.spec import (
    DURATION_SECONDS_DEFAULT,
    DURATION_SECONDS_MAX,
    DURATION_SECONDS_MIN,
    CompositionSpec,
    Mood,
    SpecError,
)

_MOOD_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        Mood.CALMING.value,
        re.compile(
            r"\b(calming|calm|relaxing|relax|soothing|soothe|peaceful|peace|"
            r"gentle|soft|quiet|tranquil|serene|mellow)\b",
            re.IGNORECASE,
        ),
    ),
    (
        Mood.ELECTRIFYING.value,
        re.compile(
            r"\b(electrifying|electric|energetic|energy|intense|exciting|"
            r"excited|dynamic|powerful|epic|thrilling|driving|driven|"
            r"upbeat|lively|vibrant)\b",
            re.IGNORECASE,
        ),
    ),
    (
        Mood.SLEEP.value,
        re.compile(
            r"\b(sleep|sleepy|asleep|lullaby|lull|rest|restful|dreamy|dream|"
            r"ambient|meditation|meditate|nighttime|bedtime)\b",
            re.IGNORECASE,
        ),
    ),
)

# Two patterns are tried in order:
# 1. Numeric value with optional unit: "5 min", "30s", "120" (bare int = seconds).
# 2. Spelled-out value with required unit: "an hour", "one minute" — never bare,
#    so a stray "a" in the prompt can't be mistaken for a duration.
_DURATION_NUMERIC_PATTERN = re.compile(
    r"(?P<num>\d+(?:\.\d+)?)"
    r"(?:\s*(?P<unit>hours?|hrs?|minutes?|mins?|m|seconds?|secs?|s))?\b",
    re.IGNORECASE,
)
_DURATION_SPELLED_PATTERN = re.compile(
    r"(?P<word>an?|one|two|three|four|five|ten|fifteen|twenty|thirty)"
    r"\s+(?P<sunit>hours?|hrs?|minutes?|mins?|seconds?|secs?)\b",
    re.IGNORECASE,
)

_FAMOUS_PIECE_HINTS = re.compile(
    r"\b(canon|fugue|sonata|symphony|concerto|prelude|nocturne|etude|"
    r"rhapsody|op\.\s*\d+|bwv|mvt|movement)\b",
    re.IGNORECASE,
)


def parse_fallback(prompt: str) -> CompositionSpec | SpecError:
    """Parse a Phase 1 prompt deterministically.

    Returns either a CompositionSpec (success) or a SpecError (rejection).
    """
    if not prompt or not prompt.strip():
        return SpecError(
            error_code="empty_prompt",
            message="Prompt is empty.",
            stage="fallback",
        )

    if _FAMOUS_PIECE_HINTS.search(prompt):
        return SpecError(
            error_code="out_of_vocabulary",
            message="Famous-piece requests are not supported in Phase 1.",
            stage="fallback",
        )

    matched_mood: str | None = None
    for mood_value, pattern in _MOOD_PATTERNS:
        if pattern.search(prompt):
            matched_mood = mood_value
            break

    if matched_mood is None:
        return SpecError(
            error_code="out_of_vocabulary",
            message=(
                "No Phase 1 mood keyword found. Phase 1 accepts: calming | electrifying | sleep."
            ),
            stage="fallback",
        )

    duration_seconds = _extract_duration(prompt)
    if duration_seconds is None:
        duration_seconds = DURATION_SECONDS_DEFAULT

    try:
        spec = CompositionSpec(
            duration_seconds=duration_seconds,
            mood=Mood(matched_mood),
        )
    except Exception as exc:
        return SpecError(
            error_code="schema_invalid",
            message=f"Fallback produced an invalid spec: {exc}",
            stage="validating",
        )
    return spec


_SPELLED_NUMBERS: dict[str, int] = {
    "a": 1,
    "an": 1,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "ten": 10,
    "fifteen": 15,
    "twenty": 20,
    "thirty": 30,
}


def _extract_duration(prompt: str) -> int | None:
    """Extract the first numeric duration in seconds, or None.

    Recognises `5 min`, `5m`, `30s`, `30 seconds`, `an hour`, `one minute`.
    Bare integers are treated as seconds. Anything that would fall outside
    the spec's bounds is clamped silently — the schema validator will reject
    out-of-range values and a structured failure will surface there.
    """
    match = _DURATION_NUMERIC_PATTERN.search(prompt)
    if match is not None:
        raw_value = float(match.group("num"))
        unit = match.group("unit")
        if unit is None:
            seconds = int(raw_value)
        else:
            unit = unit.lower()
            if unit.startswith("h"):
                seconds = int(raw_value * 3600)
            elif unit.startswith("m"):
                seconds = int(raw_value * 60)
            else:
                seconds = int(raw_value)
        return _clamp_duration(seconds)

    spelled = _DURATION_SPELLED_PATTERN.search(prompt)
    if spelled is not None:
        word = spelled.group("word").lower()
        sunit = spelled.group("sunit").lower()
        value = _SPELLED_NUMBERS[word]
        seconds = value * 3600 if sunit.startswith("h") else value * 60
        return _clamp_duration(seconds)

    return None


def _clamp_duration(seconds: int) -> int:
    if seconds < DURATION_SECONDS_MIN:
        return DURATION_SECONDS_MIN
    if seconds > DURATION_SECONDS_MAX:
        return DURATION_SECONDS_MAX
    return seconds


__all__ = ["parse_fallback"]
