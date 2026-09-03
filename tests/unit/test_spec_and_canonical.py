"""Unit tests for saimc.spec and saimc.canonical."""

from __future__ import annotations

import math

import pytest
from pydantic import ValidationError

from saimc.canonical import (
    CANONICAL_FORMAT_VERSION,
    canonical_bytes,
    canonical_dumps,
    canonical_sha256,
    manifest_document,
)
from saimc.spec import (
    DURATION_SECONDS_MAX,
    DURATION_SECONDS_MIN,
    SPEC_SCHEMA_VERSION,
    TEMPO_BPM_MAX,
    TEMPO_BPM_MIN,
    CompositionSpec,
    Mood,
    SpecError,
    TimeSignature,
    WesternKey,
)


class TestCompositionSpec:
    def test_minimal_valid(self) -> None:
        s = CompositionSpec(mood=Mood.CALMING)
        assert s.schema_version == SPEC_SCHEMA_VERSION
        assert s.duration_seconds == 180
        assert s.tempo_bpm is None
        assert s.key is None
        assert s.time_signature == TimeSignature.FOUR_FOUR
        assert s.instrumentation == "piano"
        assert s.humanization == "none"
        assert s.request_kind.value == "mood_generation"

    def test_full_valid(self) -> None:
        s = CompositionSpec(
            duration_seconds=300,
            tempo_bpm=72,
            key=WesternKey.D_MAJOR,
            time_signature=TimeSignature.THREE_FOUR,
            mood=Mood.SLEEP,
            seed=12345,
        )
        dumped = s.model_dump(mode="json")
        assert dumped["mood"] == "sleep"
        assert dumped["key"] == "D"
        assert dumped["time_signature"] == "3/4"

    @pytest.mark.parametrize(
        "duration", [DURATION_SECONDS_MIN - 1, DURATION_SECONDS_MAX + 1, 0, -5]
    )
    def test_duration_out_of_range_rejected(self, duration: int) -> None:
        with pytest.raises(ValidationError):
            CompositionSpec(mood=Mood.CALMING, duration_seconds=duration)

    @pytest.mark.parametrize("tempo", [TEMPO_BPM_MIN - 1, TEMPO_BPM_MAX + 1])
    def test_tempo_out_of_range_rejected(self, tempo: int) -> None:
        with pytest.raises(ValidationError):
            CompositionSpec(mood=Mood.CALMING, tempo_bpm=tempo)

    def test_negative_seed_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CompositionSpec(mood=Mood.CALMING, seed=-1)

    def test_extra_fields_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CompositionSpec.model_validate({"mood": "calming", "instrumentation": "guitar"})

    def test_frozen(self) -> None:
        s = CompositionSpec(mood=Mood.CALMING)
        with pytest.raises(ValidationError):
            s.mood = Mood.SLEEP  # type: ignore[misc]

    def test_unknown_mood_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CompositionSpec.model_validate({"mood": "angsty"})

    def test_wrong_schema_version_rejected(self) -> None:
        with pytest.raises(ValidationError):
            CompositionSpec.model_validate({"mood": "calming", "schema_version": 99})


class TestCanonicalJSON:
    def test_sorted_keys(self) -> None:
        a = canonical_dumps({"b": 1, "a": 2})
        b = canonical_dumps({"a": 2, "b": 1})
        assert a == b
        assert a == '{"a":2,"b":1}'

    def test_no_whitespace(self) -> None:
        out = canonical_dumps({"a": 1, "b": [1, 2, 3]})
        assert " " not in out
        assert "\n" not in out

    def test_unicode_preserved(self) -> None:
        out = canonical_dumps({"label": "Béla"})
        assert "Béla" in out

    def test_nan_rejected(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            canonical_dumps({"x": float("nan")})

    def test_inf_rejected(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            canonical_dumps({"x": math.inf})

    def test_nested_nan_rejected(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            canonical_dumps({"events": [{"time_us": float("nan")}]})

    def test_canonical_bytes_is_utf8(self) -> None:
        b = canonical_bytes({"label": "Béla"})
        assert isinstance(b, bytes)
        assert "Béla".encode() in b

    def test_sha256_stable(self) -> None:
        h1 = canonical_sha256({"a": 1, "b": [1, 2]})
        h2 = canonical_sha256({"b": [1, 2], "a": 1})
        assert h1 == h2
        assert len(h1) == 64

    def test_sha256_changes_with_content(self) -> None:
        h1 = canonical_sha256({"a": 1})
        h2 = canonical_sha256({"a": 2})
        assert h1 != h2


class TestManifestDocument:
    def test_format_key_first_when_sorted(self) -> None:
        d = manifest_document("CompositionSpec", {"mood": "calming", "schema_version": 1})
        assert d["format"] == f"CompositionSpec:{CANONICAL_FORMAT_VERSION}"
        assert next(iter(d.keys())) == "format"

    def test_empty_kind_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            manifest_document("", {})

    def test_kind_with_colon_rejected(self) -> None:
        with pytest.raises(ValueError, match=":"):
            manifest_document("Spec:Evil", {})


class TestSpecError:
    def test_basic(self) -> None:
        e = SpecError(
            error_code="out_of_vocabulary",
            message="mood 'angsty' not in Phase 1 vocabulary",
            stage="parsing",
            attempts=1,
        )
        assert e.error_code == "out_of_vocabulary"
        assert e.attempts == 1

    def test_frozen(self) -> None:
        e = SpecError(error_code="x", message="y", stage="parsing")
        with pytest.raises(ValidationError):
            e.error_code = "z"  # type: ignore[misc]
