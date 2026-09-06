"""Unit tests for the duration policy."""

from __future__ import annotations

import pytest

from saimc.compose.duration import (
    ARRANGEMENT_ARC_MIN_REPS,
    DURATION_TOLERANCE,
    INTRO_BARS,
    MAX_REPEATS,
    RITARDANDO_BARS,
    RITARDANDO_FACTOR,
    DurationArrangement,
    DurationUnfulfillableError,
    arrange_for_duration,
    bar_ticks,
    section_seed,
)
from saimc.compose.score import PPQ


class TestBarTicks:
    def test_4_4(self) -> None:
        # 4 * 480 = 1920 ticks.
        assert bar_ticks("4/4") == 1920

    def test_3_4(self) -> None:
        assert bar_ticks("3/4") == 1440

    def test_6_8(self) -> None:
        # Treated as 3 quarter-beat groups in Phase 1.
        assert bar_ticks("6/8") == 1440

    def test_2_4(self) -> None:
        assert bar_ticks("2/4") == 960

    def test_7_8(self) -> None:
        # 3.5 quarter beats: 3*480 + 240 = 1680.
        assert bar_ticks("7/8") == 1680

    def test_unknown_rejected(self) -> None:
        with pytest.raises(ValueError):
            bar_ticks("13/16")


class TestArrangeForDuration:
    @pytest.mark.parametrize("target", [30, 60, 120, 180, 300, 600])
    def test_calming_hits_every_target(self, target: int) -> None:
        arr = arrange_for_duration(
            mood="calming",
            target_duration_seconds=float(target),
            time_signature="4/4",
        )
        assert arr.tempo_bpm > 0
        assert 1 <= arr.repetition_count <= MAX_REPEATS

    @pytest.mark.parametrize("target", [30, 60, 120, 180, 300, 600])
    def test_electrifying_hits_every_target(self, target: int) -> None:
        arr = arrange_for_duration(
            mood="electrifying",
            target_duration_seconds=float(target),
            time_signature="4/4",
        )
        assert arr.tempo_bpm > 0

    @pytest.mark.parametrize("target", [30, 60, 120, 180, 300, 600])
    def test_sleep_hits_every_target(self, target: int) -> None:
        arr = arrange_for_duration(
            mood="sleep",
            target_duration_seconds=float(target),
            time_signature="4/4",
        )
        assert arr.tempo_bpm > 0

    def test_short_calming_picks_8_bar(self) -> None:
        arr = arrange_for_duration(
            mood="calming",
            target_duration_seconds=60.0,
            time_signature="4/4",
        )
        assert arr.form_bars == 8

    def test_short_target_uses_coda(self) -> None:
        """Targets that don't fit a clean (form, repetition) now use a coda."""
        # Calming tempo range 50-80, 8-bar rep 1 caps at 24s,
        # rep 2 caps at 48s — so 45s is in the gap. With a 4-bar coda
        # the policy should hit exactly.
        arr = arrange_for_duration(
            mood="calming",
            target_duration_seconds=45.0,
            time_signature="4/4",
        )
        assert arr.coda_bars == 4
        assert arr.total_bars_with_coda == arr.total_bars + arr.coda_bars

    def test_coda_invalid_when_larger_than_form(self) -> None:
        with pytest.raises(ValueError):
            DurationArrangement(
                form_bars=8,
                template=type("T", (), {})(),
                repetition_count=1,
                total_bars=8,
                tempo_bpm=60.0,
                coda_bars=8,
            )

    def test_coda_invalid_when_negative(self) -> None:
        with pytest.raises(ValueError):
            DurationArrangement(
                form_bars=8,
                template=type("T", (), {})(),
                repetition_count=1,
                total_bars=8,
                tempo_bpm=60.0,
                coda_bars=-1,
            )

    def test_long_calming_picks_16_or_32_bar(self) -> None:
        # 600s target — calming tempo (50-80) means 32 bars at slowest
        # = 153.6s, so we need reps. Form may be 16 or 32 depending on
        # what fits.
        arr = arrange_for_duration(
            mood="calming",
            target_duration_seconds=600.0,
            time_signature="4/4",
        )
        assert arr.form_bars in (16, 32)

    def test_tolerance_is_2_percent(self) -> None:
        # Per §8.
        assert pytest.approx(0.02) == DURATION_TOLERANCE

    def test_max_repeats_is_8(self) -> None:
        # Per §10 #10.
        assert MAX_REPEATS == 8

    def test_invalid_form_rejected(self) -> None:
        with pytest.raises(ValueError):
            arrange_for_duration(
                mood="calming",
                target_duration_seconds=180.0,
                time_signature="4/4",
                base_form_bars=24,
            )

    def test_2_4_long_target_fits_after_form_pick_fixed(self) -> None:
        """Regression: _pick_base_form used to assume 4 beats per bar.

        Calming 2/4 at 300s: the 8-bar form's true max is 8*8*2 beats at
        50bpm = 153.6s, so the policy must pick a larger form (32 bars,
        4 reps at 51.2bpm = 300s) — not silently undershoot.
        """
        arr = arrange_for_duration(
            mood="calming",
            target_duration_seconds=300.0,
            time_signature="2/4",
        )
        realised = (arr.total_bars_with_coda * bar_ticks("2/4") / PPQ) * 60.0 / arr.tempo_bpm
        assert abs(realised - 300.0) / 300.0 <= DURATION_TOLERANCE

    def test_impossible_target_raises_instead_of_undershooting(self) -> None:
        """Per §10 #1: fail loudly, never return an out-of-tolerance arrangement.

        Calming 4/4 at 20s is unreachable: the 8-bar form alone is 24s
        at the fastest tempo, and form+coda is 27s at fastest.
        """
        with pytest.raises(DurationUnfulfillableError):
            arrange_for_duration(
                mood="calming",
                target_duration_seconds=20.0,
                time_signature="4/4",
            )


class TestSectionSeed:
    def test_deterministic(self) -> None:
        assert section_seed(42, 0) == section_seed(42, 0)

    def test_different_for_different_indices(self) -> None:
        assert section_seed(42, 0) != section_seed(42, 1)

    def test_none_seed_uses_index(self) -> None:
        assert section_seed(None, 0) == 0
        assert section_seed(None, 5) == 5


class TestArrangementArc:
    """S8: the arc's extra seconds are part of the duration math."""

    def test_long_pieces_carry_the_ritardando(self) -> None:
        arr = arrange_for_duration(
            mood="calming",
            target_duration_seconds=120.0,
            time_signature="4/4",
        )
        assert arr.repetition_count >= ARRANGEMENT_ARC_MIN_REPS
        assert arr.intro_bars == INTRO_BARS
        assert arr.ritardando_factor == RITARDANDO_FACTOR

    def test_short_pieces_stay_constant_tempo(self) -> None:
        arr = arrange_for_duration(
            mood="calming",
            target_duration_seconds=30.0,
            time_signature="4/4",
        )
        assert arr.repetition_count < ARRANGEMENT_ARC_MIN_REPS
        assert arr.intro_bars == 0
        assert arr.ritardando_factor == 1.0

    def test_pinned_tempo_disables_the_ritardando(self) -> None:
        """The tempo promise outranks the decorative arc: a pinned
        tempo plus the slowdown's extra seconds would leave some
        durations unfulfillable, so a pinned piece plays straight."""
        arr = arrange_for_duration(
            mood="calming",
            target_duration_seconds=180.0,
            time_signature="4/4",
            tempo_bpm=63.0,
        )
        assert arr.tempo_bpm == 63.0
        assert arr.ritardando_factor == 1.0

    def test_realised_math_includes_the_ritardando(self) -> None:
        arr = arrange_for_duration(
            mood="calming",
            target_duration_seconds=120.0,
            time_signature="4/4",
        )
        rit_ticks = RITARDANDO_BARS * bar_ticks("4/4")
        body = (arr.total_bars * bar_ticks("4/4") - rit_ticks) / PPQ * 60.0 / arr.tempo_bpm
        tail = rit_ticks / PPQ * 60.0 / (arr.tempo_bpm * arr.ritardando_factor)
        assert abs(body + tail - 120.0) / 120.0 <= DURATION_TOLERANCE
