"""Unit tests for saimc.compose.forms."""

from __future__ import annotations

import pytest

from saimc.compose.forms import (
    MOOD_PROFILES,
    PHRASE_SIZES,
    STEP_MAX_SEMITONES,
    TEMPO_RANGE_BPM,
    ChordSlot,
    ChordTemplate,
    degree_to_midi,
    get_mood_profile,
    get_template_for_form,
    key_root_midi,
    key_scale_pcs,
    key_signature_from_spec_key,
    scale_pitch_offset,
    scale_semitones,
)
from saimc.compose.score import KeySignature
from saimc.spec import WesternKey


class TestChordTemplate:
    def test_valid_template(self) -> None:
        t = ChordTemplate(
            name="test",
            bars=8,
            chords=(ChordSlot(0, 2), ChordSlot(5, 2), ChordSlot(3, 2), ChordSlot(4, 2)),
        )
        assert t.bars == 8
        assert sum(slot.bars for slot in t.chords) == 8

    def test_slot_bars_must_sum_to_template_bars(self) -> None:
        # The engine advances one `bars`-length block per section; a
        # template whose slots overflow it desynchronises every later
        # section's chord map.
        with pytest.raises(ValueError, match="sum to 6"):
            ChordTemplate(name="drift", bars=8, chords=(ChordSlot(0, 2), ChordSlot(5, 4)))


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


class TestScaleTables:
    """The diatonic tables the linter's licence and the melody walk share."""

    def test_the_modes_differ_on_the_third_sixth_and_seventh(self) -> None:
        # Indices 2, 5 and 6 are the degrees the minor scale flattens;
        # everything else is shared. A table that drifted anywhere else
        # would silently move the melodic and harmonic skeleton.
        major = scale_semitones("major")
        minor = scale_semitones("minor")
        assert major[:2] == minor[:2]
        differs = [index for index, (m, n) in enumerate(zip(major, minor, strict=True)) if m != n]
        assert differs == [2, 5, 6]
        assert major[2] - minor[2] == 1
        assert major[5] - minor[5] == 1
        assert major[6] - minor[6] == 1

    def test_scale_pitch_offset_wraps_into_one_octave(self) -> None:
        # Degree 7 is the tonic again, in the same octave — this is the
        # semantics the engine's chord walk relies on.
        assert scale_pitch_offset(0, "major") == scale_pitch_offset(7, "major") == 0

    def test_scale_pitch_offset_accepts_negative_degrees(self) -> None:
        assert scale_pitch_offset(-1, "major") == 11

    def test_degree_to_midi_carries_the_octave(self) -> None:
        assert degree_to_midi(0, 60, "major") == 60
        assert degree_to_midi(7, 60, "major") == 72
        assert degree_to_midi(2, 60, "major") == 64

    def test_degree_to_midi_descends_below_the_tonic(self) -> None:
        # Degree -1 is the leading tone *below* the tonic, not degree 6 of
        # the octave above: a melody has to be able to walk under its
        # starting note without the modulo flipping it up.
        assert degree_to_midi(-1, 60, "major") == 59
        assert degree_to_midi(-7, 60, "major") == 48

    def test_degree_to_midi_follows_the_mode(self) -> None:
        assert degree_to_midi(2, 60, "major") == 64
        assert degree_to_midi(2, 60, "minor") == 63

    def test_key_scale_pcs_covers_the_mode(self) -> None:
        assert key_scale_pcs(KeySignature(root="C", mode="major")) == frozenset(
            (0, 2, 4, 5, 7, 9, 11)
        )
        assert key_scale_pcs(KeySignature(root="A", mode="minor")) == frozenset(
            (9, 11, 0, 2, 4, 5, 7)
        )

    def test_key_scale_pcs_transposes_with_the_root(self) -> None:
        # G major has F# where C major has F natural.
        assert 6 in key_scale_pcs(KeySignature(root="G", mode="major"))
        assert 5 not in key_scale_pcs(KeySignature(root="G", mode="major"))

    def test_a_step_is_a_semitone_or_a_whole_tone(self) -> None:
        # The linter's licence and the quality scorecard both measure
        # against this one number; if it moved, passing tones the licence
        # admits would be scored as leaps.
        assert STEP_MAX_SEMITONES == 2


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
