"""Unit tests for the deterministic fallback parser."""

from __future__ import annotations

import pytest

from saimc.llm.fallback import parse_fallback
from saimc.spec import (
    DURATION_SECONDS_DEFAULT,
    DURATION_SECONDS_MAX,
    DURATION_SECONDS_MIN,
    SPEC_SCHEMA_VERSION,
    CompositionSpec,
    Mood,
    SpecError,
)


class TestParseFallbackAccepts:
    @pytest.mark.parametrize(
        ("prompt", "expected_mood", "expected_duration"),
        [
            ("build me a 5 min electrifying piano piece", Mood.ELECTRIFYING.value, 300),
            ("5 minutes of calming sleep music", Mood.CALMING.value, 300),
            ("some peaceful piano music", Mood.CALMING.value, DURATION_SECONDS_DEFAULT),
            ("30s of thrilling piano", Mood.ELECTRIFYING.value, 30),
            ("a 10 minute dreamy ambient track", Mood.SLEEP.value, 600),
            ("play me something mellow for 2 minutes please", Mood.CALMING.value, 120),
            ("Give me 600 seconds of sleep music", Mood.SLEEP.value, 600),
            ("intense piano track 5m", Mood.ELECTRIFYING.value, 300),
        ],
    )
    def test_supported_prompts(
        self, prompt: str, expected_mood: str, expected_duration: int
    ) -> None:
        out = parse_fallback(prompt)
        assert isinstance(out, CompositionSpec)
        assert out.mood.value == expected_mood
        assert out.duration_seconds == expected_duration
        assert out.instrumentation == "piano"
        assert out.humanization == "light"
        assert out.schema_version == SPEC_SCHEMA_VERSION
        assert out.request_kind.value == "mood_generation"


class TestParseFallbackRejects:
    def test_empty(self) -> None:
        out = parse_fallback("")
        assert isinstance(out, SpecError)
        assert out.error_code == "empty_prompt"
        assert out.stage == "fallback"

    def test_whitespace_only(self) -> None:
        out = parse_fallback("   \n\t  ")
        assert isinstance(out, SpecError)
        assert out.error_code == "empty_prompt"

    def test_no_mood_keyword(self) -> None:
        out = parse_fallback("give me some piano music")
        assert isinstance(out, SpecError)
        assert out.error_code == "out_of_vocabulary"

    def test_famous_piece(self) -> None:
        out = parse_fallback("Canon in D, but in C")
        assert isinstance(out, SpecError)
        assert out.error_code == "out_of_vocabulary"

    def test_unsupported_mood(self) -> None:
        out = parse_fallback("an angry piano piece")
        assert isinstance(out, SpecError)
        assert out.error_code == "out_of_vocabulary"

    def test_indian_classical(self) -> None:
        out = parse_fallback("play Canon in C using Indian classical instruments")
        assert isinstance(out, SpecError)
        assert out.error_code == "out_of_vocabulary"


class TestParseFallbackClamping:
    def test_sub_minimum_clamped_up(self) -> None:
        out = parse_fallback("calming piano, 10 seconds")
        assert isinstance(out, CompositionSpec)
        assert out.duration_seconds == DURATION_SECONDS_MIN

    def test_over_maximum_clamped_down(self) -> None:
        out = parse_fallback("an hour of relaxing piano")
        assert isinstance(out, CompositionSpec)
        assert out.duration_seconds == DURATION_SECONDS_MAX
