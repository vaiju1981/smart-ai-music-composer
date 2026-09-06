"""Sheet-music renderer: NotationScore -> MusicXML -> render-service -> SVG.

The engraved path per `docs/roadmap.md` §2/§6: the canonical
`NotationScore` is exported to MusicXML with music21 (gen-time only),
then rendered to SVG by the Node render-service (OpenSheetMusicDisplay
in headless Chromium) launched as a one-shot subprocess. Notation is
driven by the NotationScore — never the PerformancePlan — per §8.
"""

from __future__ import annotations

import json
import os
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING

from saimc.compose.score import Measure, NotationScore, NoteEvent

if TYPE_CHECKING:
    from music21.note import Rest as M21Rest
    from music21.stream import Measure as M21Measure

# The Node CLI entry point lives in the render-service project next to
# the Python package in the repo checkout.
DEFAULT_RENDER_SERVICE_DIR = Path(__file__).resolve().parents[3] / "render-service"

# A stuck Chromium render should never block the job pipeline for long;
# the worker's stage timeouts are the outer bound, this is the inner one.
RENDER_TIMEOUT_SECONDS: int = 300


class SheetRenderErrorCode(StrEnum):
    """Structured error codes surfaced as `JobError.error_code`."""

    SERVICE_MISSING = "render_service_missing"
    NODE_MISSING = "node_missing"
    RENDER_FAILED = "sheet_render_failed"
    TIMEOUT = "sheet_render_timeout"


class SheetRenderError(Exception):
    """Raised by `render_sheet` for every failure mode."""

    def __init__(self, code: SheetRenderErrorCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass(frozen=True)
class SheetArtifact:
    """The rendered sheet plus the provenance the manifest needs."""

    sheet_path: Path
    container: str  # "svg"
    codec: str  # "svg"
    sha256: str
    size_bytes: int
    osmd_version: str


def notation_score_to_musicxml(score: NotationScore) -> str:
    """Export the canonical `NotationScore` to a MusicXML string.

    Each `voice_id` becomes its own staff part (the engine's voices are
    the piano's hands: 0 = bass, 1 = melody), so OSMD engraves the
    notated surface without MusicXML multi-voice machinery. Measures
    come 1:1 from the canonical measures (empty bars get whole rests),
    and simultaneous same-voice notes collapse into chords.
    """
    from music21 import instrument, meter, stream, tempo
    from music21 import key as m21key

    ppq = score.ppq
    time_signatures = {m.time_signature for m in score.measures}
    if len(time_signatures) > 1:
        raise ValueError(
            "Phase 1 engraves a single time signature; score mixes "
            + ", ".join(sorted(time_signatures))
        )
    time_sig = time_signatures.pop() if time_signatures else score.time_signature

    out = stream.Score()
    lowest_voice = min({n.voice_id for n in score.notes})
    for voice_id in sorted({n.voice_id for n in score.notes}):
        part = stream.Part()
        part.insert(instrument.Piano())
        part.insert(m21key.Key(score.key.root, score.key.mode))
        part.insert(meter.TimeSignature(time_sig))

        voice_notes = sorted(
            (n for n in score.notes if n.voice_id == voice_id),
            key=lambda n: n.tick,
        )
        tie_modes = _tie_modes(voice_notes)
        measures_by_start: dict[int, stream.Measure] = {}
        for measure in score.measures:
            m21_measure = _build_m21_measure(
                measure, _notes_in_measure(voice_notes, measure), ppq, tie_modes
            )
            part.insert(measure.start_tick / ppq, m21_measure)
            measures_by_start[measure.start_tick] = m21_measure

        # Tempo markings live inside measures: the MusicXML exporter
        # drops part-level marks when the part carries measures. The
        # base tempo sits in the first bar; the outro ritardando lands
        # in the bar it starts (its ticks are bar-aligned by
        # construction, with the containing-measure fallback for
        # safety). They engrave once, on the first staff.
        if voice_id == lowest_voice:
            measures_by_start[score.measures[0].start_tick].insert(
                0.0, tempo.MetronomeMark(number=score.tempo.bpm)
            )
            for point in score.tempo.changes:
                for measure in score.measures:
                    if measure.start_tick <= point.tick < measure.end_tick:
                        measures_by_start[measure.start_tick].insert(
                            (point.tick - measure.start_tick) / ppq,
                            tempo.MetronomeMark(number=point.bpm),
                        )
                        break
        out.insert(0, part)

    written = Path(out.write("musicxml"))
    return written.read_text(encoding="utf-8")


def _notes_in_measure(notes: list[NoteEvent], measure: Measure) -> list[NoteEvent]:
    return [n for n in notes if measure.start_tick <= n.tick < measure.end_tick]


def _tie_modes(voice_notes: list[NoteEvent]) -> dict[int, str]:
    """Tick -> music21 tie mode for a voice's tied notes.

    `NoteEvent.tie` says a note connects to the next same-voice,
    same-pitch event; the continuation note's marker is inferred from
    that link, so one boolean is enough to engrave start/continue/stop.
    """
    modes: dict[int, str] = {}
    prev: NoteEvent | None = None
    for note in voice_notes:
        linked_from_prev = (
            prev is not None
            and prev.tie
            and prev.pitch_midi == note.pitch_midi
            and prev.tick + prev.duration_ticks == note.tick
        )
        if linked_from_prev and note.tie:
            modes[note.tick] = "continue"
        elif linked_from_prev:
            modes[note.tick] = "stop"
        elif note.tie:
            modes[note.tick] = "start"
        prev = note
    return modes


def _build_m21_measure(
    measure: Measure,
    voice_notes: list[NoteEvent],
    ppq: int,
    tie_modes: dict[int, str] | None = None,
) -> M21Measure:
    """Build one music21 measure, collapsing same-tick notes into chords."""
    from music21 import chord as m21chord
    from music21 import note as m21note
    from music21 import pitch as m21pitch
    from music21 import stream
    from music21 import tie as m21tie

    m21_measure = stream.Measure(number=measure.index + 1)
    grouped: dict[int, list[NoteEvent]] = defaultdict(list)
    for n in _notes_in_measure(voice_notes, measure):
        grouped[n.tick].append(n)

    if not grouped:
        m21_measure.insert(0.0, _rest_for_measure(measure, ppq))
        return m21_measure

    for tick in sorted(grouped):
        offset = (tick - measure.start_tick) / ppq
        at_tick = grouped[tick]
        pitches = sorted(n.pitch_midi for n in at_tick)
        quarter_length = min(n.duration_ticks for n in at_tick) / ppq
        if len(at_tick) == 1:
            element = m21note.Note(m21pitch.Pitch(midi=pitches[0]), quarterLength=quarter_length)
        else:
            element = m21chord.Chord(
                [m21pitch.Pitch(midi=p) for p in pitches], quarterLength=quarter_length
            )
        element.volume.velocity = at_tick[0].velocity
        mode = None if tie_modes is None else tie_modes.get(tick)
        if mode is not None:
            element.tie = m21tie.Tie(mode)
        m21_measure.insert(offset, element)
    return m21_measure


def _rest_for_measure(measure: Measure, ppq: int) -> M21Rest:
    from music21 import note as m21note

    return m21note.Rest(quarterLength=(measure.end_tick - measure.start_tick) / ppq)


def render_sheet(
    musicxml_path: Path,
    out_svg_path: Path,
    *,
    service_dir: Path | None = None,
    node_bin: str | None = None,
    timeout: int = RENDER_TIMEOUT_SECONDS,
) -> SheetArtifact:
    """Render one MusicXML file to SVG via the render-service CLI.

    One-shot subprocess per call: `node <service>/dist/cli.js render …`.
    Exit-code contract with the CLI: 0 success, 2 usage (treated as a
    render failure here — the Python side owns the arguments), 3 render
    failure with the CLI's stderr as the message.
    """
    resolved_dir = (
        service_dir or os.environ.get("SAIMC_RENDER_SERVICE_DIR") or str(DEFAULT_RENDER_SERVICE_DIR)
    )
    service_dir = Path(resolved_dir)
    # tsc keeps the project layout under outDir (src/ and test/ both
    # compile), so the CLI lands at dist/src/cli.js; accept the flatter
    # dist/cli.js too in case the build is restructured later.
    cli_path = service_dir / "dist" / "src" / "cli.js"
    if not cli_path.is_file():
        cli_path = service_dir / "dist" / "cli.js"
    if not cli_path.is_file():
        raise SheetRenderError(
            SheetRenderErrorCode.SERVICE_MISSING,
            f"render-service CLI not found at {cli_path}; run `npm run build` in render-service/",
        )
    node = node_bin or os.environ.get("SAIMC_NODE_BIN") or "node"

    command = [
        node,
        str(cli_path),
        "render",
        "--input",
        str(musicxml_path),
        "--output",
        str(out_svg_path),
    ]
    try:
        proc = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False)
    except FileNotFoundError as exc:
        raise SheetRenderError(
            SheetRenderErrorCode.NODE_MISSING, f"node executable not found: {exc}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise SheetRenderError(
            SheetRenderErrorCode.TIMEOUT,
            f"render-service did not finish within {timeout}s",
        ) from exc

    stderr = (proc.stderr or "").strip()
    if proc.returncode == 3:
        raise SheetRenderError(SheetRenderErrorCode.RENDER_FAILED, stderr or "render failed")
    if proc.returncode != 0:
        raise SheetRenderError(
            SheetRenderErrorCode.RENDER_FAILED,
            f"render-service exited {proc.returncode}: {stderr}",
        )
    if not out_svg_path.is_file():
        raise SheetRenderError(
            SheetRenderErrorCode.RENDER_FAILED,
            f"render-service reported success but wrote no output at {out_svg_path}",
        )

    data = out_svg_path.read_bytes()
    return SheetArtifact(
        sheet_path=out_svg_path,
        container="svg",
        codec="svg",
        sha256=sha256(data).hexdigest(),
        size_bytes=len(data),
        osmd_version=_osmd_version(service_dir),
    )


def _osmd_version(service_dir: Path) -> str:
    """Read the pinned OSMD version from the render-service package.json."""
    try:
        payload = json.loads((service_dir / "package.json").read_text(encoding="utf-8"))
        version = payload.get("dependencies", {}).get("opensheetmusicdisplay")
        return str(version) if version else "unknown"
    except (OSError, json.JSONDecodeError):
        return "unknown"


__all__ = [
    "DEFAULT_RENDER_SERVICE_DIR",
    "RENDER_TIMEOUT_SECONDS",
    "SheetArtifact",
    "SheetRenderError",
    "SheetRenderErrorCode",
    "notation_score_to_musicxml",
    "render_sheet",
]
