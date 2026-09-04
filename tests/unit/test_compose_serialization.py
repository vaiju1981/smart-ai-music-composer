"""Tests for engine-output sidecar persistence."""

from __future__ import annotations

from pathlib import Path

import pytest

from saimc.compose.duration import DurationArrangement
from saimc.compose.engine import EngineOutput, compose
from saimc.compose.forms import ChordTemplate
from saimc.compose.score import (
    KeySignature,
    Measure,
    NotationScore,
    NoteEvent,
    PerformanceNoteEvent,
    PerformancePlan,
    TempoMap,
)
from saimc.compose.serialization import read_engine_output, write_engine_output
from saimc.spec import CompositionSpec, Mood


def _make_engine_output() -> EngineOutput:
    note = NoteEvent(voice_id=0, pitch_midi=60, tick=0, duration_ticks=480, velocity=64)
    measure = Measure(index=0, start_tick=0, end_tick=1920, time_signature="4/4")
    notation = NotationScore.make(
        ppq=480,
        key=KeySignature(root="C", mode="major"),
        time_signature="4/4",
        tempo_bpm=80.0,
        measures=[measure],
        notes=[note],
    )
    perf_note = PerformanceNoteEvent(
        voice_id=0,
        pitch_midi=60,
        start_us=0,
        duration_us=500_000,
        velocity=64,
    )
    performance = PerformancePlan.make(sample_rate=44100, notes=[perf_note])
    arrangement = DurationArrangement(
        form_bars=1,
        template=ChordTemplate(name="stub_1bar", bars=1, chords=((0, 1),)),
        repetition_count=1,
        total_bars=1,
        tempo_bpm=80.0,
        coda_bars=0,
    )
    return EngineOutput(
        notation_score=notation,
        performance_plan=performance,
        arrangement=arrangement,
        key=KeySignature(root="C", mode="major"),
        time_signature="4/4",
    )


def test_round_trip_preserves_all_fields(tmp_path: Path) -> None:
    """to_sidecar -> write -> read -> from_sidecar reproduces the output."""
    out = _make_engine_output()
    path = tmp_path / "engine_output.json"
    write_engine_output(path, out)
    assert path.exists()

    back = read_engine_output(path)
    assert back == out


def test_round_trip_for_real_engine_output(tmp_path: Path) -> None:
    """The real Phase 1 engine round-trips through the sidecar."""
    spec = CompositionSpec(mood=Mood.CALMING, seed=42, duration_seconds=180)
    out = compose(spec)
    path = tmp_path / "engine_output.json"
    write_engine_output(path, out)
    back = read_engine_output(path)
    # Compare the canonical hashes (a strong equality check).
    assert back.notation_score.compute_hash() == out.notation_score.compute_hash()
    assert back.performance_plan.compute_hash() == out.performance_plan.compute_hash()
    assert back.arrangement == out.arrangement
    assert back.key == out.key
    assert back.time_signature == out.time_signature


def test_read_missing_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_engine_output(tmp_path / "nope.json")


def test_coda_bars_round_trips(tmp_path: Path) -> None:
    """`coda_bars=2` survives the round trip."""
    out = _make_engine_output()
    # Rebuild with a non-zero coda. Coda must be < form_bars, so use a
    # 4-bar form with a 2-bar coda.
    arrangement = DurationArrangement(
        form_bars=4,
        template=ChordTemplate(name="stub_4bar", bars=4, chords=((0, 2), (5, 2))),
        repetition_count=1,
        total_bars=4,
        tempo_bpm=80.0,
        coda_bars=2,
    )
    out2 = EngineOutput(
        notation_score=out.notation_score,
        performance_plan=out.performance_plan,
        arrangement=arrangement,
        key=out.key,
        time_signature=out.time_signature,
    )
    path = tmp_path / "engine_output.json"
    write_engine_output(path, out2)
    back = read_engine_output(path)
    assert back.arrangement.coda_bars == 2


def test_write_is_canonical_json(tmp_path: Path) -> None:
    """The on-disk file is canonical (sorted keys, no extra whitespace)."""
    out = _make_engine_output()
    path = tmp_path / "engine_output.json"
    write_engine_output(path, out)
    text = path.read_text(encoding="utf-8")
    # No insignificant whitespace.
    assert ": " not in text
    assert ", " not in text


def test_to_sidecar_does_not_include_tempo_map_inline() -> None:
    """TempoMap is included under notation_score, not at the top level."""
    out = _make_engine_output()
    sd = out.to_sidecar()
    assert "tempo" in sd["notation_score"]
    assert isinstance(sd["notation_score"]["tempo"], dict)
    assert sd["notation_score"]["tempo"]["bpm"] == 80.0
    assert sd["notation_score"]["tempo"]["ppq"] == 480
    assert "tempo" not in sd  # top-level
    # Sanity: TempoMap is its own dataclass.
    assert isinstance(out.notation_score.tempo, TempoMap)
