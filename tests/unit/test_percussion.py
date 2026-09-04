"""Unit tests for the drum-pattern library and the percussion voice."""

from __future__ import annotations

import pytest

from saimc.compose.duration import bar_ticks
from saimc.compose.engine import compose
from saimc.compose.percussion import (
    DRUM_KICK,
    DRUM_STYLES,
    PERCUSSION_NOTE_TICKS,
    DrumHit,
    DrumStyle,
    style_for,
)
from saimc.compose.score import VOICE_PERCUSSION
from saimc.spec import Mood, TimeSignature


class TestDrumHit:
    def test_rejects_negative_offset(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            DrumHit(offset_ticks=-1, key=DRUM_KICK)

    def test_rejects_melodic_pitch(self) -> None:
        # 84 (C6) is a melodic pitch, not a GM percussion key (35-81).
        with pytest.raises(ValueError, match="GM percussion key"):
            DrumHit(offset_ticks=0, key=84)

    def test_rejects_out_of_range_velocity(self) -> None:
        with pytest.raises(ValueError, match="velocity"):
            DrumHit(offset_ticks=0, key=DRUM_KICK, velocity=128)


class TestDrumStyle:
    def test_variant_index_wraps(self) -> None:
        rock = DRUM_STYLES["rock"]
        a = rock.pattern("4/4", 0)
        b = rock.pattern("4/4", 1)
        wraps = rock.pattern("4/4", 2)
        assert a is not None
        assert b is not None
        assert wraps == a

    def test_unknown_meter_returns_none(self) -> None:
        rock = DRUM_STYLES["rock"]
        assert rock.pattern("5/4", 0) is None
        assert rock.pattern("7/8", 0) is None

    def test_velocity_scale_bounds(self) -> None:
        with pytest.raises(ValueError, match="velocity_scale"):
            DrumStyle(name="bad", variants={}, velocity_scale=0.0)

    def test_every_style_has_valid_patterns(self) -> None:
        for name, style in DRUM_STYLES.items():
            assert style.variants, f"style {name} has no patterns"
            for meter, variants in style.variants.items():
                ticks_per_bar = bar_ticks(meter)
                for variant in variants:
                    assert variant, f"style {name} meter {meter} has an empty variant"
                    for hit in variant:
                        assert hit.offset_ticks < ticks_per_bar, (
                            f"{name}/{meter}: hit at {hit.offset_ticks} overflows the bar"
                        )


class TestStyleFor:
    @pytest.mark.parametrize("mood", list(Mood))
    def test_every_mood_has_a_4_4_style(self, mood: Mood) -> None:
        assert style_for(mood.value, "4/4") is not None

    def test_waltz_for_3_4_regardless_of_mood(self) -> None:
        for mood in Mood:
            assert style_for(mood.value, "3/4") is DRUM_STYLES["waltz"]

    def test_shuffle_for_6_8(self) -> None:
        assert style_for("calming", "6/8") is DRUM_STYLES["shuffle"]

    def test_exotic_meters_get_no_drums(self) -> None:
        assert style_for("calming", "5/4") is None
        assert style_for("electrifying", "7/8") is None

    def test_sleep_and_calming_share_ballad_but_differ_in_energy(self) -> None:
        assert style_for("sleep", "4/4") is style_for("calming", "4/4")
        from saimc.compose.percussion import MOOD_VELOCITY_SCALE

        assert MOOD_VELOCITY_SCALE.get("sleep", 1.0) < 1.0


class TestEnginePercussionVoice:
    def _compose(self, **kw):
        return compose(_drum_spec(**kw))

    def test_drum_set_piece_contains_a_percussion_voice(self) -> None:
        out = self._compose()
        drum_notes = [n for n in out.notation_score.notes if n.voice_id == VOICE_PERCUSSION]
        assert drum_notes, "drum-set piece must lay a percussion voice"
        assert all(35 <= n.pitch_midi <= 81 for n in drum_notes)

    def test_percussion_hits_every_bar(self) -> None:
        out = self._compose()
        bars = {m.index for m in out.notation_score.measures}
        hit_bars = {
            n.tick // bar_ticks(out.time_signature)
            for n in out.notation_score.notes
            if n.voice_id == VOICE_PERCUSSION
        }
        assert hit_bars == bars

    def test_piano_piece_has_no_percussion_voice(self) -> None:
        out = compose(_drum_spec(instrumentation="piano"))
        assert not [n for n in out.notation_score.notes if n.voice_id == VOICE_PERCUSSION]

    def test_waltz_meter_gets_the_waltz_pattern(self) -> None:
        out = self._compose(time_signature=TimeSignature.THREE_FOUR.value)
        drum_notes = [n for n in out.notation_score.notes if n.voice_id == VOICE_PERCUSSION]
        assert drum_notes
        # Every bar starts with the kick (offset 0), per the waltz template.
        kicks = {n.tick % bar_ticks("3/4") for n in drum_notes if n.pitch_midi == DRUM_KICK}
        assert kicks == {0}

    def test_deterministic_for_the_same_seed(self) -> None:
        a = self._compose(seed=7)
        b = self._compose(seed=7)
        drums_a = [
            (n.pitch_midi, n.tick, n.velocity)
            for n in a.notation_score.notes
            if n.voice_id == VOICE_PERCUSSION
        ]
        drums_b = [
            (n.pitch_midi, n.tick, n.velocity)
            for n in b.notation_score.notes
            if n.voice_id == VOICE_PERCUSSION
        ]
        assert drums_a == drums_b

    def test_percussion_note_durations_are_short_gates(self) -> None:
        out = self._compose()
        for n in out.notation_score.notes:
            if n.voice_id == VOICE_PERCUSSION:
                assert n.duration_ticks == PERCUSSION_NOTE_TICKS


def _drum_spec(**kw):
    from saimc.spec import CompositionSpec, Instrument

    base = {
        "mood": "electrifying",
        "duration_seconds": 30,
        "seed": 42,
        "instrumentation": Instrument.DRUM_SET.value,
    }
    base.update(kw)
    return CompositionSpec.model_validate(base)
