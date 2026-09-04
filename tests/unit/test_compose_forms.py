"""Unit tests for saimc.compose.forms."""

from __future__ import annotations

import pytest

from saimc.compose.forms import (
    MOOD_PROFILES,
    PHRASE_SIZES,
    TEMPO_RANGE_BPM,
    ChordTemplate,
    get_mood_profile,
    get_template_for_form,
    key_root_midi,
    key_signature_from_spec_key,
)
from saimc.compose.score import KeySignature
from saimc.spec import WesternKey


class TestChordTemplate:
    def test_valid_template(self) -> None:
        t = ChordTemplate(name="test", bars=8, chords=((0, 2), (5, 2), (3, 2), (4, 2)))
        assert t.bars == 8
        assert sum(d for _, d in t.chords) == 8


class TestGetTemplateForForm:
    def test_exact_match(self) -> None:
        t = get_template_for_form("calming", 8)
        assert t.bars == 8

    def test_extend_template(self) -> None:
        # No 32-bar calming template; should extend an 8-bar one.
        t = get_template_for_form("calming", 32)
        assert t.bars == 32

    def test_truncate_template(self) -> None:
        # No 4-bar template; should truncate an 8-bar one.
        t = get_template_for_form("calming", 4)
        assert t.bars == 4

    def test_all_moods_have_templates(self) -> None:
        for mood in ("calming", "electrifying", "sleep"):
            for form in PHRASE_SIZES:
                t = get_template_for_form(mood, form)
                assert t.bars == form


class TestTempoRanges:
    def test_calming_range(self) -> None:
        assert TEMPO_RANGE_BPM["calming"] == (50, 80)

    def test_electrifying_range(self) -> None:
        assert TEMPO_RANGE_BPM["electrifying"] == (100, 160)

    def test_sleep_range(self) -> None:
        assert TEMPO_RANGE_BPM["sleep"] == (40, 64)


class TestKeySignatureFromSpecKey:
    def test_none_is_c_major(self) -> None:
        k = key_signature_from_spec_key(None)
        assert k.root == "C"
        assert k.mode == "major"

    def test_major_key(self) -> None:
        k = key_signature_from_spec_key(WesternKey.G_MAJOR)
        assert k.root == "G"
        assert k.mode == "major"

    def test_minor_key(self) -> None:
        k = key_signature_from_spec_key(WesternKey.A_MINOR)
        assert k.root == "A"
        assert k.mode == "minor"


class TestKeyRootMidi:
    def test_c_major_is_60(self) -> None:
        assert key_root_midi(KeySignature(root="C", mode="major")) == 60

    def test_g_major_is_67(self) -> None:
        # G is 7 semitones above C; C is 60, so G4 is 67.
        assert key_root_midi(KeySignature(root="G", mode="major")) == 67

    def test_a_minor_is_69(self) -> None:
        assert key_root_midi(KeySignature(root="A", mode="minor")) == 69

    def test_unknown_root_rejected(self) -> None:
        with pytest.raises(ValueError):
            key_root_midi(KeySignature(root="X", mode="major"))


class TestMoodRegistry:
    def test_profiles_are_the_single_source_of_truth(self) -> None:
        for name, profile in MOOD_PROFILES.items():
            assert profile.name == name
            assert TEMPO_RANGE_BPM[name] == profile.tempo_range_bpm
            assert profile.templates

    def test_unknown_mood_names_the_known_moods(self) -> None:
        with pytest.raises(KeyError) as exc_info:
            get_mood_profile("angsty")
        message = str(exc_info.value)
        assert "angsty" in message
        assert "calming" in message
        assert "sleep" in message
