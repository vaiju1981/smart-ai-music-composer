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
    ENSEMBLE_MAX_VOICES,
    ROLE_LIMITS,
    SPEC_SCHEMA_VERSION,
    TEMPO_BPM_MAX,
    TEMPO_BPM_MIN,
    CompositionSpec,
    Instrument,
    InstrumentationEntry,
    Mood,
    SpecError,
    TimeSignature,
    VoiceRole,
    WesternKey,
    seed_for_brief,
)


class TestCompositionSpec:
    def test_minimal_valid(self) -> None:
        s = CompositionSpec(mood=Mood.CALMING)
        assert s.schema_version == SPEC_SCHEMA_VERSION
        assert s.duration_seconds == 180
        assert s.tempo_bpm is None
        assert s.key is None
        assert s.time_signature == TimeSignature.FOUR_FOUR
        # The omitted instrumentation field defaults to the mood's piano-led ensemble.
        assert s.instrumentation == [
            InstrumentationEntry(role=VoiceRole.MELODY, instrument=Instrument.PIANO),
            InstrumentationEntry(role=VoiceRole.HARMONY, instrument=Instrument.PIZZICATO_STRINGS),
            InstrumentationEntry(role=VoiceRole.BASS, instrument=Instrument.CELLO),
        ]
        assert s.humanization == "light"
        assert s.request_kind.value == "mood_generation"

    def test_version_1_spec_still_valid(self) -> None:
        # Version-1 specs (humanization locked to "none") remain readable.
        s = CompositionSpec.model_validate(
            {"mood": "calming", "schema_version": 1, "humanization": "none"}
        )
        assert s.schema_version == 1
        assert s.humanization == "none"

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


class TestTheEnsembleCeilingMeetsTheRender:
    """The spec's ceiling and the render's channel budget are one decision.

    They live in two modules because `render` imports `spec` and a dependency
    back would be a cycle, so the numbers are written twice and held together
    here. The spec is written to *meet* the render's bound rather than to guess
    under it: one melody, twelve harmony, one bass and a kit is fifteen, and
    fifteen pitched channels is exactly what MIDI leaves once the kit has
    channel 10.
    """

    def test_every_voice_the_spec_permits_has_a_channel(self) -> None:
        from saimc.compose.score import VOICE_HARMONY
        from saimc.render.audio import MELODIC_CHANNELS, _channel_for_voice

        pitched = ENSEMBLE_MAX_VOICES - ROLE_LIMITS[VoiceRole.PERCUSSION]
        assert pitched <= len(MELODIC_CHANNELS)
        # Voice ids are bass=0, melody=1, percussion=2, harmony=3.., so the
        # highest id a legal ensemble can reach is the last harmony layer's.
        highest = VOICE_HARMONY + ROLE_LIMITS[VoiceRole.HARMONY] - 1
        assert _channel_for_voice(highest) in MELODIC_CHANNELS

    def test_one_more_harmony_voice_than_the_spec_allows_would_have_none(self) -> None:
        """The ceiling is at the bound, not under it: the next voice has no channel."""
        from saimc.compose.score import VOICE_HARMONY
        from saimc.render.audio import AudioRenderError, _channel_for_voice

        with pytest.raises(AudioRenderError):
            _channel_for_voice(VOICE_HARMONY + ROLE_LIMITS[VoiceRole.HARMONY])

    def test_the_largest_legal_ensemble_validates(self) -> None:
        entries = [{"role": "melody", "instrument": "piano"}]
        harmonies = [
            "strings", "choir", "pipe_organ", "celesta", "harp", "flute",
            "clarinet", "oboe", "trumpet", "french_horn", "trombone", "vibraphone",
        ]  # fmt: skip
        assert len(harmonies) == ROLE_LIMITS[VoiceRole.HARMONY]
        entries += [{"role": "harmony", "instrument": i} for i in harmonies]
        entries.append({"role": "bass", "instrument": "cello"})
        entries.append({"role": "percussion", "instrument": "drum_set"})
        spec = CompositionSpec.model_validate(
            {"mood": "calming", "duration_seconds": 60, "instrumentation": entries}
        )
        assert len(spec.instrumentation) == ENSEMBLE_MAX_VOICES


class TestTheBriefDecidesTheSeed:
    """Two different briefs must not be one piece.

    The one-shot path composed every unseeded request against seed 0, so
    "calming, two minutes" was the same key, tempo and melody however
    differently it had been asked for. Deriving the seed from the words fixes
    that without touching the guarantee the field exists for.
    """

    def test_different_briefs_get_different_seeds(self) -> None:
        bare = CompositionSpec.model_validate({"mood": "calming", "duration_seconds": 120})
        one = bare.with_brief_seed("a calming piano piece")
        other = bare.with_brief_seed("something cinematic that builds")
        assert one.seed != other.seed

    def test_the_same_brief_is_always_the_same_piece(self) -> None:
        bare = CompositionSpec.model_validate({"mood": "calming", "duration_seconds": 120})
        assert bare.with_brief_seed("a calming piano piece").seed == (
            bare.with_brief_seed("  a calming piano piece  ").seed
        ), "surrounding whitespace should not be a different request"

    def test_a_seed_that_was_asked_for_is_never_overwritten(self) -> None:
        """An explicit seed is a request for one exact piece."""
        pinned = CompositionSpec.model_validate(
            {"mood": "calming", "duration_seconds": 120, "seed": 5}
        )
        assert pinned.with_brief_seed("whatever the words say").seed == 5

    def test_the_seed_is_stable_across_processes(self) -> None:
        """`hash()` is salted per interpreter; a seed that moved between runs
        would break the reproducibility this field exists for."""
        import subprocess
        import sys

        out = subprocess.run(
            [
                sys.executable,
                "-c",
                "from saimc.spec import seed_for_brief; print(seed_for_brief('x'))",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        assert int(out.stdout.strip()) == seed_for_brief("x")

    def test_two_briefs_compose_two_different_pieces(self) -> None:
        """The point of the change, asserted on notes rather than on numbers."""
        from saimc.compose.engine import compose

        bare = {"mood": "calming", "duration_seconds": 120}
        a = compose(CompositionSpec.model_validate(bare).with_brief_seed("a calm piano piece"))
        b = compose(CompositionSpec.model_validate(bare).with_brief_seed("gentle strings at dusk"))
        assert a.notation_score.compute_hash() != b.notation_score.compute_hash()
