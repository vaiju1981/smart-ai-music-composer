"""Release-gate suite per `docs/roadmap.md` §8.

Runs every automatable MVP acceptance gate against a matrix of real
engine outputs. Gates that need external binaries (FluidSynth, the
render-service, the audited FFmpeg) reuse these functions in the
real-binary end-to-end run.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from saimc.compose.engine import compose
from saimc.release import (
    gate_canonical_reproducibility,
    gate_composition_correctness,
    gate_duration_tolerance,
    gate_midi_parseable_and_onsets,
    gate_musicxml_structural,
    gate_render_time_budget,
    gate_spec_round_trip,
)
from saimc.spec import CompositionSpec, Mood

# The §8 gate matrix: every mood at a short, a typical, and the
# §10 #11 cap duration, plus a boundary duration that exercises the
# duration policy's coda logic.
SPEC_MATRIX: list[CompositionSpec] = [
    CompositionSpec(mood=mood, seed=seed, duration_seconds=duration)
    for mood, seed, duration in [
        (Mood.CALMING, 42, 30),
        (Mood.CALMING, 7, 180),
        (Mood.ELECTRIFYING, 3, 300),
        (Mood.SLEEP, 11, 600),
    ]
]


@pytest.mark.parametrize(
    "spec", SPEC_MATRIX, ids=lambda s: f"{s.mood.value}-{s.seed}-{s.duration_seconds}"
)
class TestReleaseGates:
    def test_canonical_artifacts_are_reproducible(self, spec: CompositionSpec) -> None:
        result = gate_canonical_reproducibility(spec)
        assert result.passed, result.detail

    def test_spec_round_trips_through_schema(self, spec: CompositionSpec) -> None:
        result = gate_spec_round_trip(spec)
        assert result.passed, result.detail

    def test_composition_passes_theory_linter(self, spec: CompositionSpec) -> None:
        result = gate_composition_correctness(compose(spec))
        assert result.passed, result.detail

    def test_musicxml_is_structurally_valid(self, spec: CompositionSpec) -> None:
        output = compose(spec)
        result = gate_musicxml_structural(output.notation_score)
        assert result.passed, result.detail

    def test_midi_parseable_with_onsets_within_20ms(self, spec: CompositionSpec) -> None:
        output = compose(spec)
        result = gate_midi_parseable_and_onsets(
            output.performance_plan, bpm=output.arrangement.tempo_bpm
        )
        assert result.passed, result.detail

    def test_duration_within_2_percent(self, spec: CompositionSpec) -> None:
        output = compose(spec)
        result = gate_duration_tolerance(spec, output)
        assert result.passed, result.detail


class TestRenderTimeBudgetGate:
    def test_within_budget_passes(self) -> None:
        result = gate_render_time_budget(elapsed_s=400.0, piece_seconds=300.0)
        assert result.passed

    def test_over_budget_fails(self) -> None:
        result = gate_render_time_budget(elapsed_s=1000.0, piece_seconds=300.0)
        assert not result.passed


class TestModelsRegistryGate:
    """Roadmap §10 #3: MODELS.md exists before any candidate-model request."""

    def test_models_registry_exists_with_required_fields(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        registry = repo_root / "MODELS.md"
        assert registry.is_file(), "MODELS.md must exist before the first model request"
        text = registry.read_text(encoding="utf-8")
        for required in (
            "Model identifier",
            "License",
            "Acceptable-use policy URL",
            "Structured-output capability probe",
            "Decision:",
            "Approved Phase 1 model",
        ):
            assert required in text, f"MODELS.md is missing the {required!r} record field"
