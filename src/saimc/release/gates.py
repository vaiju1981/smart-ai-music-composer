"""Release gates per `docs/roadmap.md` §8 (MVP acceptance criteria).

Every gate returns a `GateResult` and asserts one criterion:

- `gate_canonical_reproducibility` — byte-identical canonical artifacts
  for two runs of the same spec + seed (§8 "Spec & reproducibility").
- `gate_spec_round_trip` — every accepted spec round-trips through the
  Pydantic schema validator.
- `gate_composition_correctness` — the theory linter passes (range,
  complete measures, voice-leading collisions).
- `gate_musicxml_structural` — the exported MusicXML parses and its
  structure matches the NotationScore.
- `gate_midi_parseable_and_onsets` — the SMF parses with mido, carries
  one note-on per plan note, and its decoded onsets agree with the
  plan's integer-microsecond schedule within the ±20 ms §8 tolerance
  (the same schedule the animation renderer draws from).
- `gate_duration_tolerance` — realized duration within ±2% of the
  spec's `duration_seconds` (§8/§10 #5).
- `gate_render_time_budget` — a render's wall-clock fits the <=3x
  real-time budget (§8); the real-binary measurement uses this.

The gates deliberately encode the roadmap's thresholds as constants so
a roadmap change forces a code change.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from saimc.canonical import canonical_dumps
from saimc.compose.engine import EngineOutput, compose
from saimc.compose.linter import lint
from saimc.compose.score import PPQ, NotationScore, PerformancePlan, TempoPoint
from saimc.render.audio import build_smf
from saimc.spec import CompositionSpec

DURATION_TOLERANCE: float = 0.02
ONSET_TOLERANCE_US: int = 20_000
RENDER_TIME_BUDGET_FACTOR: float = 3.0


@dataclass(frozen=True)
class GateResult:
    """Outcome of one release gate."""

    name: str
    passed: bool
    detail: str = ""


def gate_canonical_reproducibility(spec: CompositionSpec) -> GateResult:
    """Two engine runs from the same spec produce byte-identical canonical artifacts."""
    first = compose(spec)
    second = compose(spec)
    score_a = canonical_dumps(first.notation_score.to_canonical_dict())
    score_b = canonical_dumps(second.notation_score.to_canonical_dict())
    plan_a = canonical_dumps(first.performance_plan.to_canonical_dict())
    plan_b = canonical_dumps(second.performance_plan.to_canonical_dict())
    if score_a != score_b or plan_a != plan_b:
        return GateResult(
            "canonical_reproducibility",
            False,
            "canonical serialization differs across identical runs",
        )
    if first.notation_score.compute_hash() != second.notation_score.compute_hash():
        return GateResult("canonical_reproducibility", False, "notation score hash differs")
    return GateResult(
        "canonical_reproducibility",
        True,
        f"notation+plan hashes stable: {first.notation_score.compute_hash()[:12]}…",
    )


def gate_spec_round_trip(spec: CompositionSpec) -> GateResult:
    """The spec survives `model_dump(mode="json") -> model_validate` unchanged."""
    payload = spec.model_dump(mode="json")
    round_tripped = CompositionSpec.model_validate(payload)
    if round_tripped != spec:
        return GateResult("spec_round_trip", False, "round-tripped spec differs from original")
    return GateResult("spec_round_trip", True, "schema round-trip is lossless")


def gate_composition_correctness(output: EngineOutput) -> GateResult:
    """The theory linter passes on the engine's NotationScore (§8)."""
    report = lint(output.notation_score, chord_bars=output.chord_bars or None)
    if report.issues:
        summary = "; ".join(f"{i.code}: {i.message}" for i in report.issues[:3])
        return GateResult(
            "composition_correctness", False, f"{len(report.issues)} issue(s): {summary}"
        )
    return GateResult(
        "composition_correctness",
        True,
        f"linter clean over {len(output.notation_score.measures)} measures",
    )


def gate_musicxml_structural(score: NotationScore) -> GateResult:
    """The exported MusicXML parses and matches the NotationScore's structure.

    Structural, not full XSD: it checks the root element, the declared
    MusicXML version, one part per voice, and a 1:1 measure count per
    part. The full MusicXML 4.0 XSD is not vendored yet; the schema
    gate tightens when it is.
    """
    from xml.etree import ElementTree

    from saimc.render.sheet import notation_score_to_musicxml

    root = ElementTree.fromstring(notation_score_to_musicxml(score))
    if not root.tag.endswith("score-partwise"):
        return GateResult("musicxml_structural", False, f"unexpected root element {root.tag}")
    if not root.get("version"):
        return GateResult("musicxml_structural", False, "MusicXML version attribute missing")
    parts = [child for child in root if child.tag.endswith("part")]
    voices = {n.voice_id for n in score.notes}
    if voices and len(parts) != len(voices):
        return GateResult(
            "musicxml_structural",
            False,
            f"{len(parts)} part(s) for {len(voices)} voice(s)",
        )
    for part in parts:
        measures = [child for child in part if child.tag.endswith("measure")]
        if len(measures) != len(score.measures):
            return GateResult(
                "musicxml_structural",
                False,
                f"part has {len(measures)} measures, score has {len(score.measures)}",
            )
    return GateResult(
        "musicxml_structural",
        True,
        f"score-partwise v{root.get('version')}, {len(parts)} part(s), "
        f"{len(score.measures)} measures each",
    )


def gate_midi_parseable_and_onsets(
    plan: PerformancePlan,
    *,
    bpm: float,
    tempo_changes: tuple[TempoPoint, ...] = (),
) -> GateResult:
    """The SMF parses with mido and its onsets match the plan within ±20 ms.

    The SMF encodes the plan's microsecond schedule as PPQ=480 ticks;
    this gate decodes the ticks back to microseconds — following the
    file's own set_tempo events, so a piecewise tempo map (the outro
    ritardando) is verified too — and bounds the quantization error.
    The piano-roll animation draws directly from the same microsecond
    schedule, so this bounds the §8 audio/animation onset tolerance
    end to end.
    """
    import mido

    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "gate.mid"
        build_smf(plan, bpm=bpm, tempo_changes=tempo_changes).save(str(path))
        parsed = mido.MidiFile(str(path))
        if parsed.type != 0 or parsed.ticks_per_beat != PPQ:
            return GateResult(
                "midi_parseable_and_onsets",
                False,
                f"type={parsed.type} ppq={parsed.ticks_per_beat}",
            )

        us_per_tick = 60_000_000 / bpm / PPQ
        # Loaded track messages carry delta ticks in chronological order;
        # accumulate to absolute onset ticks. A set_tempo switches the
        # tick rate for everything after it (its own delta is still
        # covered by the previous rate, which is exactly how the file
        # timelines work).
        onsets_us: list[float] = []
        elapsed_us = 0.0
        for msg in parsed.tracks[0]:
            elapsed_us += msg.time * us_per_tick
            if msg.type == "set_tempo":
                us_per_tick = msg.tempo / PPQ
            elif msg.type == "note_on" and msg.velocity > 0:
                onsets_us.append(elapsed_us)
        if len(onsets_us) != len(plan.notes):
            return GateResult(
                "midi_parseable_and_onsets",
                False,
                f"{len(onsets_us)} note-ons for {len(plan.notes)} plan notes",
            )
        worst = 0.0
        for note, onset_us in zip(
            sorted(plan.notes, key=lambda n: (n.start_us, n.voice_id, n.pitch_midi)),
            onsets_us,
            strict=True,
        ):
            worst = max(worst, abs(onset_us - note.start_us))
        if worst > ONSET_TOLERANCE_US:
            return GateResult(
                "midi_parseable_and_onsets",
                False,
                f"worst onset drift {worst}µs > {ONSET_TOLERANCE_US}µs",
            )
    return GateResult(
        "midi_parseable_and_onsets",
        True,
        f"onsets within {worst}µs (tolerance {ONSET_TOLERANCE_US}µs)",
    )


def gate_duration_tolerance(spec: CompositionSpec, output: EngineOutput) -> GateResult:
    """Realized duration within ±2% of the spec's `duration_seconds` (§10 #5)."""
    target = spec.duration_seconds
    if target is None:
        return GateResult("duration_tolerance", True, "no duration target in spec")
    realized = output.performance_plan.realized_duration_us() / 1_000_000
    delta = abs(realized - target) / target
    if delta > DURATION_TOLERANCE:
        return GateResult(
            "duration_tolerance",
            False,
            f"realized {realized:.2f}s vs target {target}s ({delta * 100:.2f}% off)",
        )
    return GateResult(
        "duration_tolerance",
        True,
        f"realized {realized:.2f}s vs target {target}s ({delta * 100:.2f}%)",
    )


def gate_render_time_budget(elapsed_s: float, piece_seconds: float) -> GateResult:
    """Wall-clock render time fits the <=3x real-time budget (§8/§10 #5)."""
    budget = RENDER_TIME_BUDGET_FACTOR * piece_seconds
    if elapsed_s > budget:
        return GateResult(
            "render_time_budget",
            False,
            f"{elapsed_s:.1f}s > {budget:.0f}s budget for a {piece_seconds:.0f}s piece",
        )
    return GateResult(
        "render_time_budget",
        True,
        f"{elapsed_s:.1f}s of {budget:.0f}s budget for a {piece_seconds:.0f}s piece",
    )


__all__ = [
    "DURATION_TOLERANCE",
    "ONSET_TOLERANCE_US",
    "RENDER_TIME_BUDGET_FACTOR",
    "GateResult",
    "gate_canonical_reproducibility",
    "gate_composition_correctness",
    "gate_duration_tolerance",
    "gate_midi_parseable_and_onsets",
    "gate_musicxml_structural",
    "gate_render_time_budget",
    "gate_spec_round_trip",
    "run_all_gates",
]


def run_all_gates(specs: list[CompositionSpec]) -> list[GateResult]:
    """Run every engine-level gate over a spec matrix; returns all results."""
    results: list[GateResult] = []
    for spec in specs:
        results.append(gate_canonical_reproducibility(spec))
        results.append(gate_spec_round_trip(spec))
        output = compose(spec)
        results.append(gate_composition_correctness(output))
        results.append(gate_musicxml_structural(output.notation_score))
        results.append(
            gate_midi_parseable_and_onsets(
                output.performance_plan,
                bpm=output.arrangement.tempo_bpm,
                tempo_changes=output.notation_score.tempo.changes,
            )
        )
        results.append(gate_duration_tolerance(spec, output))
    return results
