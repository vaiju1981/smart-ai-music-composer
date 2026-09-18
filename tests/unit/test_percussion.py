"""Unit tests for the drum-pattern library and the percussion voice."""

from __future__ import annotations

import pytest

from saimc.compose.duration import bar_ticks
from saimc.compose.engine import compose
from saimc.compose.percussion import (
    DRUM_KICK,
    DRUM_STYLES,
    EIGHTH,
    PERCUSSION_NOTE_TICKS,
    SWING_RATIO_STRAIGHT,
    SWING_RATIO_TRIPLET,
    DrumHit,
    DrumStyle,
    style_for,
)
from saimc.compose.score import PPQ, VOICE_BASS, VOICE_PERCUSSION
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

    def test_percussion_holds_from_the_entry_bar_on(self) -> None:
        from saimc.compose.percussion import PERCUSSION_ENTRY_BAR

        out = self._compose()
        bars = {m.index for m in out.notation_score.measures}
        hit_bars = {
            n.tick // bar_ticks(out.time_signature)
            for n in out.notation_score.notes
            if n.voice_id == VOICE_PERCUSSION
        }
        # The groove is stated before the kit joins it: the entry bars are
        # silent and every bar from there on sounds. A short piece has no
        # intro, so this is the only thing keeping the drums off bar one —
        # before it the kit opened the piece on a crash.
        assert hit_bars == bars - set(range(PERCUSSION_ENTRY_BAR))

    def test_piano_piece_has_no_percussion_voice(self) -> None:
        out = compose(_drum_spec(instrumentation="piano"))
        assert not [n for n in out.notation_score.notes if n.voice_id == VOICE_PERCUSSION]

    def test_waltz_meter_gets_the_waltz_pattern(self) -> None:
        out = self._compose(time_signature=TimeSignature.THREE_FOUR.value)
        ticks = bar_ticks("3/4")
        drum_notes = [n for n in out.notation_score.notes if n.voice_id == VOICE_PERCUSSION]
        assert drum_notes
        # Every bar is kick-led, but the kick is not always on the downbeat:
        # the waltz's second variant states it on the "and" of the first beat
        # (BEAT - EIGHTH = 240 ticks), which is one of the two things the
        # widened vocabulary bought. Which of the two a section plays is the
        # seed's, so this asserts the family rather than the one bar.
        #
        # The pattern's own family is what is asserted, so the bass's
        # onsets — every one of which draws a follow kick now — are added
        # to the allowed set rather than filtered out of the kicks.
        bass_onsets = {
            n.tick % ticks for n in out.notation_score.notes if n.voice_id == VOICE_BASS
        }
        kicks = sorted({n.tick % ticks for n in drum_notes if n.pitch_midi == DRUM_KICK})
        assert set(kicks) <= {0, 240} | bass_onsets
        sounding_bars = {n.tick // ticks for n in drum_notes}
        kick_bars = {n.tick // ticks for n in drum_notes if n.pitch_midi == DRUM_KICK}
        assert kick_bars == sounding_bars

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

    def test_every_bass_attack_in_a_sounding_bar_draws_a_kick(self) -> None:
        """The rhythm section is one section, end to end.

        The percussion pass is written after the bass and reads where it
        attacks, so the bass's own onsets land on a kick — the bass leads
        because it is the voice stating the harmony. Read per bar, because
        a bar the kit rests in is a bar the correlation cannot hold in.
        """
        out = self._compose(duration_seconds=120)
        ticks = bar_ticks(out.time_signature)
        notes = out.notation_score.notes
        bass = [n for n in notes if n.voice_id == VOICE_BASS]
        assert bass, "the piece has no bass line to correlate with"
        drum_bars = {n.tick // ticks for n in notes if n.voice_id == VOICE_PERCUSSION}
        kick_offsets = {
            n.tick % ticks
            for n in notes
            if n.voice_id == VOICE_PERCUSSION and n.pitch_midi == DRUM_KICK
        }
        unrested = sorted({n.tick // ticks for n in bass} & drum_bars)
        assert len(unrested) >= 4, "too few sounding bars to be a correlation"
        for bar in unrested:
            onsets = {n.tick % ticks for n in bass if n.tick // ticks == bar}
            assert onsets <= kick_offsets, f"bar {bar}: {sorted(onsets - kick_offsets)}"

    def test_a_kit_that_writes_no_kick_of_its_own_still_composes(self) -> None:
        """The swing kit is the one style with no kick in any bar template.

        It is unreachable from a mood or a meter — `SetDrumStyle` is the
        only way to it — and it writes every kick the piece has as a
        follow kick, which is the case where the bar's downbeat mark and
        the bass's request can collide at tick 0.
        """
        from dataclasses import replace

        from saimc.compose.percussion import DRUM_STYLES as styles
        from saimc.compose.plan import default_plan

        for swing_ratio in (SWING_RATIO_STRAIGHT, SWING_RATIO_TRIPLET):
            plan = replace(
                default_plan(_drum_spec(duration_seconds=120)),
                drum_style_name="swing",
                swing_ratio=swing_ratio,
            )
            out = compose(_drum_spec(duration_seconds=120), plan=plan)
            drums = [n for n in out.notation_score.notes if n.voice_id == VOICE_PERCUSSION]
            assert drums, "the swing kit wrote nothing"
            kicks = [n for n in drums if n.pitch_midi == DRUM_KICK]
            assert kicks, "the follow kicks did not reach the notation"
            assert [h.offset_ticks for h in styles["swing"].variants["4/4"][0]
                    if h.key == DRUM_KICK] == [], "this style now writes its own kick"

    def test_a_swung_plan_moves_the_kits_offbeats_and_only_those(self) -> None:
        """The ratio reaches the notation end to end, and the bass does not.

        A piece's bass states the harmony on the quarters it always used:
        the feel is the kit's, which is why the knob displaces the written
        pattern rather than re-writing the groove.
        """
        from dataclasses import replace

        from saimc.compose.plan import default_plan

        spec = _drum_spec(duration_seconds=120)
        ticks = bar_ticks(spec.time_signature.value)
        base = compose(spec)
        swung = compose(spec, plan=replace(default_plan(spec), swing_ratio=SWING_RATIO_TRIPLET))
        bass_base = [n.tick for n in base.notation_score.notes if n.voice_id == VOICE_BASS]
        bass_swung = [n.tick for n in swung.notation_score.notes if n.voice_id == VOICE_BASS]
        assert bass_base == bass_swung, "a swing is not the bass's"
        drums_base = {
            (n.tick % ticks, n.pitch_midi)
            for n in base.notation_score.notes
            if n.voice_id == VOICE_PERCUSSION
        }
        drums_swung = {
            (n.tick % ticks, n.pitch_midi)
            for n in swung.notation_score.notes
            if n.voice_id == VOICE_PERCUSSION
        }
        assert drums_base != drums_swung, "the ratio never reached the kit"
        for offset, pitch in drums_swung - drums_base:
            assert offset % PPQ == 2 * PPQ // 3, (offset, pitch)


class TestFillsAndDownbeats:
    """S7 rhythm vocabulary: sections hand off through a fill, and
    every section downbeat is marked with a crash."""

    def _drum_notes(self, out):
        return [n for n in out.notation_score.notes if n.voice_id == VOICE_PERCUSSION]

    def test_last_bar_of_each_section_is_a_fill(self) -> None:
        from saimc.compose.percussion import DRUM_HIGH_TOM, DRUM_LOW_TOM

        out = compose(_drum_spec(duration_seconds=120))
        arr = out.arrangement
        ticks_per_bar = bar_ticks(out.time_signature)
        section_end_bars = {s * arr.form_bars - 1 for s in range(1, arr.repetition_count)}
        fill_hits = [
            n
            for n in self._drum_notes(out)
            if n.tick // ticks_per_bar in section_end_bars
            and n.pitch_midi in (DRUM_HIGH_TOM, DRUM_LOW_TOM)
        ]
        assert fill_hits, "each section's last bar should carry the tom fill"

    def test_final_bar_never_carries_a_fill(self) -> None:
        from saimc.compose.percussion import DRUM_HIGH_TOM, DRUM_LOW_TOM

        out = compose(_drum_spec(duration_seconds=120))
        ticks_per_bar = bar_ticks(out.time_signature)
        final_bar = out.arrangement.total_bars_with_coda - 1
        assert not [
            n
            for n in self._drum_notes(out)
            if n.tick // ticks_per_bar == final_bar
            and n.pitch_midi in (DRUM_HIGH_TOM, DRUM_LOW_TOM)
        ]

    def test_every_sounding_downbeat_has_a_crash(self) -> None:
        from saimc.compose.duration import ARRANGEMENT_ARC_MIN_REPS
        from saimc.compose.percussion import (
            DRUM_CRASH,
            PERCUSSION_ENTRY_BAR,
        )

        out = compose(_drum_spec(duration_seconds=120))
        arr = out.arrangement
        ticks_per_bar = bar_ticks(out.time_signature)
        drums = self._drum_notes(out)
        hit_bars = {n.tick // ticks_per_bar for n in drums}
        crash_bars = {n.tick // ticks_per_bar for n in drums if n.pitch_midi == DRUM_CRASH}
        assert arr.repetition_count >= ARRANGEMENT_ARC_MIN_REPS
        # Every section downbeat the kit actually sounds is marked — a long
        # piece rests the intro, where the first section's own crash lives,
        # and one mid-piece section, so those downbeats are silent with the
        # rest of their bars.
        section_starts = {s * arr.form_bars for s in range(arr.repetition_count)}
        # ...and so is the kit's own entrance, which the form puts no section
        # boundary on. Without it the piece's first cymbal waited for bar
        # `form_bars`, because the only crash before that one is rested.
        assert crash_bars == (section_starts & hit_bars) | {min(hit_bars)}
        assert min(crash_bars) == PERCUSSION_ENTRY_BAR
        assert arr.intro_bars == PERCUSSION_ENTRY_BAR

    def test_long_pieces_rest_the_kit_then_it_returns_with_a_crash(self) -> None:
        from saimc.compose.duration import ARRANGEMENT_ARC_MIN_REPS
        from saimc.compose.percussion import DRUM_CRASH, PERCUSSION_REST_SECTION

        out = compose(_drum_spec(duration_seconds=120))
        arr = out.arrangement
        assert arr.repetition_count >= ARRANGEMENT_ARC_MIN_REPS
        ticks_per_bar = bar_ticks(out.time_signature)
        hit_bars = {
            n.tick // ticks_per_bar for n in self._drum_notes(out)
        }
        # The intro bars and the rest section are silent...
        rested = set(range(arr.intro_bars)) | set(
            range(
                PERCUSSION_REST_SECTION * arr.form_bars,
                (PERCUSSION_REST_SECTION + 1) * arr.form_bars,
            )
        )
        assert not (hit_bars & rested), f"drums sounded in rest bars: {hit_bars & rested}"
        # ...and the kit returns at the next section downbeat with a crash.
        return_bar = (PERCUSSION_REST_SECTION + 1) * arr.form_bars
        assert return_bar in hit_bars
        assert any(
            n.tick == return_bar * ticks_per_bar and n.pitch_midi == DRUM_CRASH
            for n in self._drum_notes(out)
        )

    def test_rotation_holds_a_then_changes_pace(self) -> None:
        from saimc.compose.percussion import rotation_index

        rock = DRUM_STYLES["rock"]
        assert rock.pattern("4/4", rotation_index(0, 2)) == rock.pattern("4/4", 0)
        assert rock.pattern("4/4", rotation_index(1, 2)) == rock.pattern("4/4", 0)
        assert rock.pattern("4/4", rotation_index(2, 2)) == rock.pattern("4/4", 1)
        assert rock.pattern("4/4", rotation_index(3, 2)) == rock.pattern("4/4", 0)
        # A single-variant style always plays it.
        assert rotation_index(5, 1) == 0


class TestSwing:
    """The ratio, read as the displacement it is.

    A swing ratio divides the *beat*: its first eighth takes
    `ratio / (ratio + 1)` of it and the second takes the rest. So the one
    position a ratio moves is the offbeat eighth, and there is no triplet
    grid to hold because the ratio decides where that offbeat goes.
    """

    def test_straight_leaves_every_position_exactly_where_it_was(self) -> None:
        from saimc.compose.percussion import swing_offset

        for offset in range(0, 4 * PPQ, EIGHTH):
            assert swing_offset(offset, SWING_RATIO_STRAIGHT) == offset

    def test_the_triplet_ratio_lands_the_offbeat_on_the_triplet(self) -> None:
        from saimc.compose.percussion import swing_offset

        assert EIGHTH == 240, "this test's arithmetic is written for a 240-tick eighth"
        # A beat is 480 ticks: the first two triplet eighths are 320 and 640.
        assert swing_offset(EIGHTH, SWING_RATIO_TRIPLET) == 320
        assert swing_offset(PPQ + EIGHTH, SWING_RATIO_TRIPLET) == PPQ + 320
        assert swing_offset(7 * PPQ + EIGHTH, SWING_RATIO_TRIPLET) == 7 * PPQ + 320

    @pytest.mark.parametrize("ratio", [1.25, 1.5, 2.0])
    def test_only_the_offbeat_eighth_moves(self, ratio: float) -> None:
        """A hit on a beat, on a sixteenth, or on a dotted value is where the
        pattern put it — the offbeat eighth is the whole vocabulary of a
        swing, which is what keeps a dotted rhythm a written thing rather
        than a ratio past the bound."""
        from saimc.compose.percussion import DOTTED_EIGHTH, swing_offset

        untouched = (
            0,
            PPQ,
            PPQ // 4,
            DOTTED_EIGHTH,
            PPQ + DOTTED_EIGHTH,
            2 * PPQ,
            PPQ - 1,
        )
        for offset in untouched:
            assert swing_offset(offset, ratio) == offset, offset
        assert swing_offset(EIGHTH, ratio) > EIGHTH
        assert swing_offset(EIGHTH, ratio) < PPQ, "a displacement crossed its beat line"

    def test_a_ratio_moves_the_offbeat_later_and_never_earlier(self) -> None:
        """Monotone in the ratio, so a heavier swing is a heavier swing."""
        from saimc.compose.percussion import swing_offset

        offsets = [swing_offset(EIGHTH, r) for r in (1.0, 1.2, 1.5, 1.8, 2.0)]
        assert offsets == sorted(offsets)
        assert len(set(offsets)) == len(offsets)

    def test_the_pattern_comes_back_untouched_at_straight(self) -> None:
        """The identity, and it is the object itself rather than an equal
        copy — which is what makes a straight piece byte-identical to the
        one this module wrote before a ratio existed."""
        from saimc.compose.percussion import swing_pattern

        pattern = DRUM_STYLES["rock"].pattern("4/4", 0)
        assert pattern is not None
        assert swing_pattern(pattern, SWING_RATIO_STRAIGHT) is pattern

    def test_the_pattern_keeps_everything_but_the_offbeat_positions(self) -> None:
        from saimc.compose.percussion import swing_offset, swing_pattern

        pattern = DRUM_STYLES["funk"].pattern("4/4", 0)
        assert pattern is not None
        swung = swing_pattern(pattern, SWING_RATIO_TRIPLET)
        assert len(swung) == len(pattern)
        for was, now in zip(pattern, swung, strict=True):
            assert (now.key, now.velocity) == (was.key, was.velocity), (
                "a swing is a feel, not a different bar"
            )
            assert now.offset_ticks == swing_offset(was.offset_ticks, SWING_RATIO_TRIPLET)

    def test_the_swing_style_has_an_offbeat_for_the_ratio_to_move(self) -> None:
        """The premise of the ratio's reachability: the swing kit writes its
        offbeat ride on the *straight* eighth and the ratio is what moves it
        onto the triplet, so the style and the feel are one vocabulary at
        two ratios rather than two tables."""
        variant = DRUM_STYLES["swing"].pattern("4/4", 0)
        assert variant is not None
        from saimc.compose.percussion import swing_offset

        moved = [h for h in variant if swing_offset(h.offset_ticks, SWING_RATIO_TRIPLET) != h.offset_ticks]
        assert moved, "the swing kit writes no offbeat eighth for a ratio to move"


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
