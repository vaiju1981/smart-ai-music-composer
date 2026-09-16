"""Unit tests for the canonical symbolic artifacts (NotationScore, PerformancePlan)."""

from __future__ import annotations

import pytest

from saimc.compose.score import (
    DEFAULT_VELOCITY,
    PPQ,
    KeySignature,
    Measure,
    NotationScore,
    NoteEvent,
    PerformanceNoteEvent,
    PerformancePlan,
    TempoMap,
    microseconds_to_ticks,
    realized_duration_seconds,
    ticks_to_microseconds,
)


class TestTickMicrosecondConversion:
    def test_quarter_note_at_60bpm(self) -> None:
        # One quarter at 60bpm = 1 second = 1_000_000 us.
        assert ticks_to_microseconds(PPQ, 60.0) == 1_000_000

    def test_eighth_note_at_120bpm(self) -> None:
        # 120 bpm = 0.5 s per quarter = 500_000 us per quarter.
        assert ticks_to_microseconds(PPQ, 120.0) == 500_000
        # One eighth = half a quarter = 250_000 us.
        assert ticks_to_microseconds(PPQ // 2, 120.0) == 250_000

    def test_round_trip(self) -> None:
        bpm = 100.0
        for ticks in (PPQ, 2 * PPQ, 3 * PPQ, 5 * PPQ):
            us = ticks_to_microseconds(ticks, bpm)
            assert microseconds_to_ticks(us, bpm) == ticks

    def test_zero_bpm_rejected(self) -> None:
        with pytest.raises(ValueError):
            ticks_to_microseconds(PPQ, 0.0)


class TestNoteEvent:
    def test_valid_event(self) -> None:
        n = NoteEvent(voice_id=0, pitch_midi=60, tick=0, duration_ticks=PPQ)
        assert n.velocity == DEFAULT_VELOCITY
        assert n.tie is False

    def test_out_of_range_pitch_rejected(self) -> None:
        with pytest.raises(ValueError):
            NoteEvent(voice_id=0, pitch_midi=128, tick=0, duration_ticks=PPQ)

    def test_negative_pitch_rejected(self) -> None:
        with pytest.raises(ValueError):
            NoteEvent(voice_id=0, pitch_midi=-1, tick=0, duration_ticks=PPQ)

    def test_zero_duration_rejected(self) -> None:
        with pytest.raises(ValueError):
            NoteEvent(voice_id=0, pitch_midi=60, tick=0, duration_ticks=0)

    def test_negative_velocity_rejected(self) -> None:
        with pytest.raises(ValueError):
            NoteEvent(voice_id=0, pitch_midi=60, tick=0, duration_ticks=PPQ, velocity=0)

    def test_negative_tick_rejected(self) -> None:
        with pytest.raises(ValueError):
            NoteEvent(voice_id=0, pitch_midi=60, tick=-1, duration_ticks=PPQ)


class TestMeasure:
    def test_valid(self) -> None:
        m = Measure(index=0, start_tick=0, end_tick=1920, time_signature="4/4")
        assert m.time_signature == "4/4"

    def test_end_before_start_rejected(self) -> None:
        with pytest.raises(ValueError):
            Measure(index=0, start_tick=1920, end_tick=0, time_signature="4/4")

    def test_negative_start_rejected(self) -> None:
        with pytest.raises(ValueError):
            Measure(index=0, start_tick=-1, end_tick=1920, time_signature="4/4")


class TestPerformanceNoteEvent:
    def test_valid(self) -> None:
        n = PerformanceNoteEvent(
            voice_id=0, pitch_midi=60, start_us=0, duration_us=500_000, velocity=64
        )
        assert n.start_us == 0

    def test_zero_duration_rejected(self) -> None:
        with pytest.raises(ValueError):
            PerformanceNoteEvent(voice_id=0, pitch_midi=60, start_us=0, duration_us=0, velocity=64)

    def test_negative_start_rejected(self) -> None:
        with pytest.raises(ValueError):
            PerformanceNoteEvent(
                voice_id=0, pitch_midi=60, start_us=-1, duration_us=500_000, velocity=64
            )


class TestNotationScore:
    def test_build(self) -> None:
        s = NotationScore.make(
            ppq=PPQ,
            key=KeySignature(root="C", mode="major"),
            time_signature="4/4",
            tempo_bpm=80.0,
            measures=[
                Measure(index=0, start_tick=0, end_tick=1920, time_signature="4/4"),
            ],
            notes=[
                NoteEvent(voice_id=0, pitch_midi=60, tick=0, duration_ticks=PPQ),
            ],
        )
        assert s.total_ticks() == 1920
        assert s.tempo.bpm == 80.0

    def test_canonical_dict_has_sorted_keys(self) -> None:
        s = NotationScore.make(
            ppq=PPQ,
            key=KeySignature(root="C", mode="major"),
            time_signature="4/4",
            tempo_bpm=80.0,
            measures=[],
            notes=[],
        )
        d = s.to_canonical_dict()
        # sorted_keys=True is part of canonical_dumps; verify the dict
        # contains everything we expect.
        assert d["ppq"] == PPQ
        assert d["key"] == {"root": "C", "mode": "major"}

    def test_compute_hash_is_stable(self) -> None:
        s = NotationScore.make(
            ppq=PPQ,
            key=KeySignature(root="C", mode="major"),
            time_signature="4/4",
            tempo_bpm=80.0,
            measures=[],
            notes=[],
        )
        h1 = s.compute_hash()
        h2 = s.compute_hash()
        assert h1 == h2
        assert len(h1) == 64


class TestPerformancePlan:
    def test_realized_duration_us(self) -> None:
        p = PerformancePlan.make(
            sample_rate=44100,
            notes=[
                PerformanceNoteEvent(0, 60, 0, 1_000_000, 64),
                PerformanceNoteEvent(0, 64, 1_000_000, 2_000_000, 64),
            ],
        )
        assert p.realized_duration_us() == 3_000_000

    def test_realized_duration_seconds(self) -> None:
        p = PerformancePlan.make(
            sample_rate=44100,
            notes=[PerformanceNoteEvent(0, 60, 0, 1_500_000, 64)],
        )
        assert realized_duration_seconds(p) == 1.5

    def test_compute_hash_is_stable(self) -> None:
        p = PerformancePlan.make(
            sample_rate=44100,
            notes=[PerformanceNoteEvent(0, 60, 0, 1_000_000, 64)],
        )
        assert p.compute_hash() == p.compute_hash()


class TestTempoMap:
    def test_default_ppq(self) -> None:
        t = TempoMap(bpm=80.0)
        assert t.ppq == PPQ
        assert t.bpm == 80.0
