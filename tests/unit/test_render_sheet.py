"""Unit tests for the sheet renderer (MusicXML export + CLI wrapper)."""

from __future__ import annotations

import hashlib
import stat
import textwrap
from pathlib import Path

import pytest

from saimc.compose.engine import compose
from saimc.compose.score import KeySignature, Measure, NotationScore, NoteEvent
from saimc.render.sheet import (
    SheetRenderError,
    SheetRenderErrorCode,
    notation_score_to_musicxml,
    render_sheet,
)
from saimc.spec import CompositionSpec, Mood


def _score() -> NotationScore:
    """A tiny hand-built score: 2 bars, bass + melody voices."""
    notes = [
        NoteEvent(voice_id=0, pitch_midi=48, tick=0, duration_ticks=1920, velocity=60),
        NoteEvent(voice_id=1, pitch_midi=64, tick=0, duration_ticks=480, velocity=70),
        NoteEvent(voice_id=1, pitch_midi=67, tick=480, duration_ticks=480, velocity=70),
        NoteEvent(voice_id=1, pitch_midi=72, tick=960, duration_ticks=960, velocity=70),
    ]
    return NotationScore.make(
        ppq=480,
        key=KeySignature(root="C", mode="major"),
        time_signature="4/4",
        tempo_bpm=72.0,
        measures=[
            Measure(index=0, start_tick=0, end_tick=1920, time_signature="4/4"),
            Measure(index=1, start_tick=1920, end_tick=3840, time_signature="4/4"),
        ],
        notes=notes,
    )


def test_export_produces_score_partwise_musicxml() -> None:
    musicxml = notation_score_to_musicxml(_score())
    assert musicxml.lstrip().startswith("<?xml")
    assert "<score-partwise" in musicxml


def test_export_preserves_voice_structure() -> None:
    from music21 import converter

    parsed = converter.parse(notation_score_to_musicxml(_score()), format="musicxml")
    # Two voices -> two piano staves.
    assert len(parsed.parts) == 2

    # Bass voice: one note (bar 1), bar 2 gets a rest.
    bass = parsed.parts[0]
    bass_events = list(bass.recurse().notes)
    assert len(bass_events) == 1
    assert bass_events[0].pitch.midi == 48

    # Melody voice: 3 single notes.
    melody = parsed.parts[1]
    assert len(list(melody.recurse().notes)) == 3


def test_export_collapses_same_tick_notes_into_chords() -> None:
    from music21 import converter

    score = _score()
    chorded = NotationScore(
        format=score.format,
        ppq=score.ppq,
        key=score.key,
        time_signature=score.time_signature,
        tempo=score.tempo,
        measures=score.measures,
        notes=[
            *score.notes,
            NoteEvent(voice_id=1, pitch_midi=76, tick=960, duration_ticks=960, velocity=70),
        ],
    )
    parsed = converter.parse(notation_score_to_musicxml(chorded), format="musicxml")
    melody = parsed.parts[1]
    melody_events = list(melody.recurse().notes)
    assert len(melody_events) == 3  # 3 tick slots; the last is now a 2-note chord
    assert len(melody_events[2].pitches) == 2


def test_export_fills_empty_measures_with_rests() -> None:
    from music21 import converter

    parsed = converter.parse(notation_score_to_musicxml(_score()), format="musicxml")
    # Bar 2 (ticks 1920-3840) has no notes in either voice; the melody
    # part must still carry a whole-measure rest so the bar is complete.
    melody = parsed.parts[1]
    measure2 = melody.getElementsByClass("Measure")[1]
    rests = measure2.getElementsByClass("Rest")
    assert len(rests) == 1
    assert rests[0].quarterLength == 4.0


def test_export_rejects_mixed_time_signatures() -> None:
    score = _score()
    measures = [
        Measure(index=0, start_tick=0, end_tick=1920, time_signature="4/4"),
        Measure(index=1, start_tick=1920, end_tick=3840, time_signature="3/4"),
    ]
    mixed = NotationScore(
        format=score.format,
        ppq=score.ppq,
        key=score.key,
        time_signature=score.time_signature,
        tempo=score.tempo,
        measures=tuple(measures),
        notes=score.notes,
    )
    with pytest.raises(ValueError, match="single time signature"):
        notation_score_to_musicxml(mixed)


def test_export_from_real_engine_round_trips() -> None:
    """The real engine's NotationScore exports and parses back cleanly."""
    from music21 import converter

    spec = CompositionSpec(mood=Mood.CALMING, seed=42, duration_seconds=180)
    output = compose(spec)
    parsed = converter.parse(notation_score_to_musicxml(output.notation_score), format="musicxml")
    assert len(parsed.parts) >= 1


def test_render_sheet_service_missing(tmp_path: Path) -> None:
    with pytest.raises(SheetRenderError) as exc_info:
        render_sheet(tmp_path / "in.musicxml", tmp_path / "out.svg", service_dir=tmp_path)
    assert exc_info.value.code == SheetRenderErrorCode.SERVICE_MISSING


def _stub_node_script(tmp_path: Path, body: str) -> str:
    script = tmp_path / "stub-node"
    script.write_text(textwrap.dedent(body), encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return str(script)


def _service_dir_with_cli(tmp_path: Path) -> Path:
    service_dir = tmp_path / "render-service"
    (service_dir / "dist").mkdir(parents=True)
    (service_dir / "dist" / "cli.js").write_text("// stub\n", encoding="utf-8")
    (service_dir / "package.json").write_text(
        '{"dependencies": {"opensheetmusicdisplay": "2.1.2"}}', encoding="utf-8"
    )
    return service_dir


def test_render_sheet_node_missing(tmp_path: Path) -> None:
    service_dir = _service_dir_with_cli(tmp_path)
    with pytest.raises(SheetRenderError) as exc_info:
        render_sheet(
            tmp_path / "in.musicxml",
            tmp_path / "out.svg",
            service_dir=service_dir,
            node_bin="/nonexistent-node-binary",
        )
    assert exc_info.value.code == SheetRenderErrorCode.NODE_MISSING


def test_render_sheet_cli_failure_surfaces_stderr(tmp_path: Path) -> None:
    service_dir = _service_dir_with_cli(tmp_path)
    stub = _stub_node_script(tmp_path, "#!/bin/sh\necho 'osmd exploded' >&2\nexit 3\n")
    with pytest.raises(SheetRenderError) as exc_info:
        render_sheet(
            tmp_path / "in.musicxml",
            tmp_path / "out.svg",
            service_dir=service_dir,
            node_bin=stub,
        )
    assert exc_info.value.code == SheetRenderErrorCode.RENDER_FAILED
    assert "osmd exploded" in exc_info.value.message


def test_render_sheet_success_writes_artifact(tmp_path: Path) -> None:
    service_dir = _service_dir_with_cli(tmp_path)
    # The stub writes the SVG to the value after --output ($6).
    stub = _stub_node_script(tmp_path, "#!/bin/sh\nprintf '<svg>ok</svg>' > \"$6\"\nexit 0\n")
    musicxml_path = tmp_path / "in.musicxml"
    musicxml_path.write_text("<score-partwise/>", encoding="utf-8")
    out_path = tmp_path / "sheet.svg"

    artifact = render_sheet(musicxml_path, out_path, service_dir=service_dir, node_bin=stub)

    data = out_path.read_bytes()
    assert artifact.sheet_path == out_path
    assert artifact.container == "svg"
    assert artifact.codec == "svg"
    assert artifact.osmd_version == "2.1.2"
    assert artifact.sha256 == hashlib.sha256(data).hexdigest()
    assert artifact.size_bytes == len(data)


def test_render_sheet_reports_missing_output(tmp_path: Path) -> None:
    service_dir = _service_dir_with_cli(tmp_path)
    stub = _stub_node_script(tmp_path, "#!/bin/sh\nexit 0\n")
    with pytest.raises(SheetRenderError) as exc_info:
        render_sheet(
            tmp_path / "in.musicxml",
            tmp_path / "sheet.svg",
            service_dir=service_dir,
            node_bin=stub,
        )
    assert exc_info.value.code == SheetRenderErrorCode.RENDER_FAILED
    assert "wrote no output" in exc_info.value.message
