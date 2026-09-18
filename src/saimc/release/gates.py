"""Release gates per `docs/roadmap.md` §8 (MVP acceptance criteria).

Every gate returns a `GateResult` and asserts one criterion:

- `gate_canonical_reproducibility` — byte-identical canonical artifacts
  for two runs of the same spec + seed (§8 "Spec & reproducibility"),
  optionally under a supplied `CompositionPlan`, since §6's versioned
  documents now include the plan the determinism claim rests on.
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
- `gate_musical_quality` — the matrix clears every musical-quality
  threshold in `saimc.quality`: stepwise melody, recovered leaps, a
  melody that sits above its accompaniment, a bass that is not one bar
  looped. This is the gate behind §8's "not a hot pot of instruments".
- `gate_quality_breach_rate` — the share of a *sampled* grid that
  breaches at least one threshold is at or below a recorded ceiling,
  which is the claim the curated matrix above cannot make.

The gates deliberately encode the roadmap's thresholds as constants so
a roadmap change forces a code change.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

from saimc.canonical import canonical_dumps
from saimc.compose.engine import EngineOutput, compose
from saimc.compose.linter import lint
from saimc.compose.plan import CompositionPlan
from saimc.compose.score import PPQ, NotationScore, PerformancePlan, TempoPoint
from saimc.quality import QUALITY_THRESHOLDS, score_corpus, score_piece
from saimc.render.audio import build_smf
from saimc.spec import CompositionSpec

DURATION_TOLERANCE: float = 0.02
ONSET_TOLERANCE_US: int = 20_000
RENDER_TIME_BUDGET_FACTOR: float = 3.0

MAX_THRESHOLD_BREACH_RATE: float = 0.50
"""The ceiling on the share of a sampled grid breaching any threshold.

A measurement, not a target: measured at 27 of 54 cells — 50.0% — across
three moods x 30/120/300 s x six seeds on the tree `gate_quality_breach_rate`
landed with. It is a ratchet, so it may only fall: a change that made more
pieces breach their own bar would otherwise arrive as a lower rate beside a
raised ceiling, which is the one move a ratchet must not allow. Lowering it is
the deliberate act; raising it is a decision that has to be argued for in a
commit, because the number is the whole of the claim.
"""


@dataclass(frozen=True)
class GateResult:
    """Outcome of one gate: a name, a verdict, and the sentence behind it.

    Shared with `saimc.session.gates`, which judges a session's published
    piece against the same threshold table. The vocabulary is deliberately
    one type rather than two — "a named bar, and whether this thing cleared
    it, with a reason" is the same statement about the engine and about the
    harness, and two records for it would be two spellings of one idea. The
    name is `release`'s module path because that is where the engine's gates
    were written; the type itself carries no release-specific concept.
    """

    name: str
    passed: bool
    detail: str = ""


def gate_canonical_reproducibility(
    spec: CompositionSpec, plan: CompositionPlan | None = None
) -> GateResult:
    """Two engine runs from the same `(spec, plan)` produce byte-identical canonical artifacts.

    `plan` is the composition plan to compose under, and `None` means the
    engine's defaults. It is a parameter rather than something the gate
    derives, because the whole point of the plan is that a caller may pin
    the music: a gate that always composes at the defaults would be a gate
    over a path nothing ships on. Pass a non-default plan and the two runs
    below exercise the plan path — including the third check, which is the
    one that catches a plan dropped on the way in, since an ignored plan
    still reproduces itself perfectly.
    """
    first = compose(spec, plan=plan)
    second = compose(spec, plan=plan)
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
    # The composition plan is an artifact the determinism claim is made
    # about (§6), so it is compared for the same two reasons the score is:
    # the runs have to agree, and the agreement has to be about the plan
    # that was asked for.
    if (first.plan is None) != (second.plan is None) or (
        first.plan is not None
        and second.plan is not None
        and first.plan.compute_hash() != second.plan.compute_hash()
    ):
        return GateResult("canonical_reproducibility", False, "composition plan hash differs")
    if plan is not None and first.plan != plan:
        return GateResult(
            "canonical_reproducibility",
            False,
            "the composition plan that was composed under is not the one that was supplied",
        )
    composed_under = "defaults" if first.plan is None else first.plan.compute_hash()[:12]
    return GateResult(
        "canonical_reproducibility",
        True,
        f"notation+plan hashes stable: {first.notation_score.compute_hash()[:12]}… "
        f"(composition plan: {composed_under})",
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
    report = lint(
        output.notation_score,
        chord_bars=output.chord_bars or None,
        bar_keys=output.bar_keys or None,
        voice_instruments={v.voice_id: v.instrument for v in output.voice_instruments},
    )
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


def gate_musical_quality(scores: Sequence[NotationScore]) -> GateResult:
    """The matrix clears every `saimc.quality` threshold (§8, "not a hot pot").

    Judged over the whole matrix rather than one piece at a time: a single
    piece's interval statistics are noisy, and several metrics do not apply
    to every arrangement (a solo has no harmony to separate from), so the
    corpus mean is the honest unit. An empty matrix fails rather than
    passing vacuously.

    What a green run proves is narrower than it reads. The release matrix
    is five hand-picked specs, and the engine does clear every threshold
    on those. It is not a claim about the product: over the sampled grid
    `gate_quality_breach_rate` pins, 27 of 54 pieces breach at least one
    threshold — `texture_hierarchy` and `harmony_pad_coverage` in 18 cells
    each, `leap_recovery_ratio` and `max_leap_semitones` in 5, and
    `step_ratio` in 1 — and the one real end-to-end render in `var/jobs/`
    records `max_leap_semitones: 14.0` against a maximum of 12. A corpus
    mean over five curated specs cannot see any of that, and it is not
    meant to; it is a regression bar for the engine, not a guarantee about
    what a user gets. Read a pass here as "the matrix passes", never as
    "the generator passes": the claim about the generator is the sampled
    gate's, and it is a rate rather than a pass.

    The metric frequencies here are read off that grid and not off a single
    render, which is what they used to be — the paragraph named
    `texture_hierarchy`, `max_leap_semitones` and `leap_recovery_ratio` as
    the frequent three, two of which are among the rarest, and omitted
    `harmony_pad_coverage`, which is tied for first. A frequency read off
    one piece is the same defect the paragraph is about.
    """
    if not scores:
        return GateResult(
            "musical_quality", False, "no pieces to score; the gate needs a non-empty matrix"
        )

    report = score_corpus(
        "release-gate",
        [score_piece(score, piece=f"piece-{index}") for index, score in enumerate(scores)],
    )
    if not report.passed:
        # Every reason is a one-line "metric measured vs bar", so they all
        # fit; naming them is the whole point of the gate.
        missed = len(report.failure_reasons)
        return GateResult(
            "musical_quality",
            False,
            f"{missed} of {len(QUALITY_THRESHOLDS)} thresholds missed over "
            f"{report.piece_count} piece(s): " + "; ".join(report.failure_reasons),
        )
    return GateResult(
        "musical_quality",
        True,
        f"{len(QUALITY_THRESHOLDS)} thresholds cleared over {report.piece_count} piece(s)",
    )


def gate_quality_breach_rate(scores: Sequence[NotationScore]) -> GateResult:
    """The share of a sampled grid breaching any threshold (§8, §F).

    The companion to `gate_musical_quality`, and the two answer different
    questions. That one asks whether the *corpus mean* clears every bar,
    which a curated matrix answers for five pieces. This asks how often a
    piece breaches its own bar, which no corpus mean can express: a metric
    can average clean while half the pieces miss it, and that is exactly the
    state this gate found the generator in.

    **It judges the scores it is handed, so the sampling is the caller's.**
    That is deliberate and it is why `run_all_gates` does not run it: the
    rate is a claim about the reachable spec space, so it has to be measured
    over a grid chosen to span that space rather than over whatever matrix a
    caller happens to be holding. The acceptance suite builds the grid and
    the axes it pins are its subject, not this function's.

    The ceiling is `MAX_THRESHOLD_BREACH_RATE`, measured rather than chosen,
    so a green run means "no worse than the tree that recorded it". An empty
    grid fails: a rate over nothing is not a rate, and `0/0` would read as
    perfect.
    """
    if not scores:
        return GateResult(
            "quality_breach_rate",
            False,
            "no pieces to score; a rate over an empty grid is not a rate",
        )

    missed = [score_piece(score).findings() for score in scores]
    breached = sum(1 for findings in missed if findings)
    rate = breached / len(scores)
    tally = Counter(finding.metric for findings in missed for finding in findings)
    named = ", ".join(f"{metric} ({count})" for metric, count in tally.most_common())
    tail = f"; most often: {named}" if named else ""

    if rate > MAX_THRESHOLD_BREACH_RATE:
        return GateResult(
            "quality_breach_rate",
            False,
            f"{breached} of {len(scores)} pieces breach at least one threshold, "
            f"{rate:.1%} above a ceiling of {MAX_THRESHOLD_BREACH_RATE:.0%}{tail}",
        )
    return GateResult(
        "quality_breach_rate",
        True,
        f"{breached} of {len(scores)} pieces breach at least one threshold, "
        f"{rate:.1%} within a ceiling of {MAX_THRESHOLD_BREACH_RATE:.0%}{tail}",
    )


__all__ = [
    "DURATION_TOLERANCE",
    "MAX_THRESHOLD_BREACH_RATE",
    "ONSET_TOLERANCE_US",
    "RENDER_TIME_BUDGET_FACTOR",
    "GateResult",
    "gate_canonical_reproducibility",
    "gate_composition_correctness",
    "gate_duration_tolerance",
    "gate_midi_parseable_and_onsets",
    "gate_musical_quality",
    "gate_musicxml_structural",
    "gate_quality_breach_rate",
    "gate_render_time_budget",
    "gate_spec_round_trip",
    "run_all_gates",
]


def run_all_gates(
    specs: list[CompositionSpec],
    planned: Sequence[tuple[CompositionSpec, CompositionPlan]] = (),
) -> list[GateResult]:
    """Run every engine-level gate over a spec matrix; returns all results.

    `planned` holds `(spec, plan)` pairs for the same reproducibility gate.
    They are a separate argument because every other gate judges the notes
    and the notes are the engine's either way — it is only the
    reproducibility claim that the *plan* is part of (§6), so it is the
    only gate that has to be handed one.
    """
    results: list[GateResult] = []
    scores: list[NotationScore] = []
    for spec in specs:
        results.append(gate_canonical_reproducibility(spec))
        results.append(gate_spec_round_trip(spec))
        output = compose(spec)
        scores.append(output.notation_score)
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
    for spec, plan in planned:
        results.append(gate_canonical_reproducibility(spec, plan))
    # The quality gate judges the matrix as a whole, so it runs once at the
    # end rather than per spec.
    results.append(gate_musical_quality(scores))
    return results
