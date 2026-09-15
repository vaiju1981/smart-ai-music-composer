"""Unit tests for saimc.compose.forms."""

from __future__ import annotations

import pytest

from saimc.compose.forms import (
    LEAP_MIN_SEMITONES,
    MOOD_PROFILES,
    PHRASE_SIZES,
    STEP_MAX_SEMITONES,
    TEMPO_RANGE_BPM,
    ChordSlot,
    ChordTemplate,
    bar_scale_intervals,
    chord_tone_degrees,
    get_mood_profile,
    get_template_for_form,
    key_root_midi,
    key_scale_pcs,
    key_signature_from_spec_key,
    scale_intervals,
    scale_pitch_offset,
    scale_semitones,
    scale_walk,
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

    def test_scale_intervals_covers_one_octave_from_its_root(self) -> None:
        intervals = scale_intervals(0, "major")
        assert intervals == (0, 2, 4, 5, 7, 9, 11)
        assert scale_intervals(0, "minor") == (0, 2, 3, 5, 7, 8, 10)

    def test_scale_intervals_rotates_the_mode_onto_its_new_root(self) -> None:
        # Spelled from the V of major the scale is mixolydian, and from
        # major's sixth it is the natural minor: a bar's chord and the
        # line over it have to be the same seven tones, so the table the
        # walk reads is a rotation of the mode and not the mode itself.
        assert scale_intervals(4, "major") == (0, 2, 4, 5, 7, 9, 10)
        assert scale_intervals(5, "major") == (0, 2, 3, 5, 7, 8, 10)

    def test_scale_intervals_ascends_within_the_octave(self) -> None:
        for mode in ("major", "minor"):
            for degree in range(7):
                intervals = scale_intervals(degree, mode)
                assert intervals[0] == 0
                assert len(intervals) == 7
                assert all(0 <= step < 12 for step in intervals)
                assert list(intervals) == sorted(set(intervals))

    def test_scale_walk_carries_the_octave(self) -> None:
        intervals = scale_intervals(0, "major")
        assert scale_walk(0, 60, intervals) == 60
        assert scale_walk(2, 60, intervals) == 64
        assert scale_walk(7, 60, intervals) == 72

    def test_scale_walk_descends_below_the_root(self) -> None:
        # Degree -1 is the scale tone *below* the root, not degree 6 of
        # the octave above: a melody has to be able to walk under its
        # starting note without the modulo flipping it up.
        intervals = scale_intervals(0, "major")
        assert scale_walk(-1, 60, intervals) == 59
        assert scale_walk(-7, 60, intervals) == 48

    def test_scale_walk_follows_the_mode_it_is_given(self) -> None:
        # The same degree is a semitone lower in minor, which is the
        # whole reason the intervals are a parameter and not a mode name.
        assert scale_walk(2, 60, scale_intervals(0, "major")) == 64
        assert scale_walk(2, 60, scale_intervals(0, "minor")) == 63

    def test_scale_walk_walks_the_bar_scale_from_its_chord_root(self) -> None:
        # A bar on the V: its own scale spelled from its root, so a
        # degree walked over it is a tone of that rotation and not of
        # the key. Degree 0 is the chord root itself.
        key = KeySignature(root="C", mode="major")
        intervals = bar_scale_intervals(4, key)
        assert scale_walk(0, 67, intervals) == 67
        assert scale_walk(1, 67, intervals) == 69
        assert scale_walk(7, 67, intervals) == 79

    def test_a_borrowed_bar_walks_the_parallel_mode(self) -> None:
        # bVI in a major key is built from the parallel minor's table,
        # and the line over it has to read that same table or its steps
        # would leave the chord's own scale.
        key = KeySignature(root="C", mode="major")
        assert bar_scale_intervals(5, key, borrowed=True) == scale_intervals(5, "minor")
        assert bar_scale_intervals(5, key) == scale_intervals(5, "major")

    def test_chord_tone_degrees_are_every_other_degree(self) -> None:
        # The chord tables take a triad's tones from the bar scale's
        # degrees 0, 2, 4 and a seventh's from 0, 2, 4, 6, so "is this
        # note a chord tone" is a question about degree parity and not
        # about pitch classes.
        assert chord_tone_degrees(3) == (0, 2, 4)
        assert chord_tone_degrees(4) == (0, 2, 4, 6)

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

    def test_a_leap_is_a_fourth_or_wider(self) -> None:
        # The generator's recovery pass and the scorecard's leap ratio
        # have to call the same intervals leaps, and a third has to be
        # neither: it is the one interval the two tests cannot disagree
        # about without the disagreement reading as a composition bug.
        assert LEAP_MIN_SEMITONES == 5
        assert LEAP_MIN_SEMITONES > STEP_MAX_SEMITONES + 1


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
