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
from saimc.compose.score import (
    PPQ,
    VOICE_MELODY,
    KeySignature,
    Measure,
    NotationScore,
    NoteEvent,
)
from saimc.release import (
    gate_canonical_reproducibility,
    gate_composition_correctness,
    gate_duration_tolerance,
    gate_midi_parseable_and_onsets,
    gate_musical_quality,
    gate_musicxml_structural,
    gate_render_time_budget,
    gate_spec_round_trip,
)
from saimc.spec import CompositionSpec, Mood

# The §8 gate matrix: every mood at a short, a typical, and the
# §10 #11 cap duration, plus a boundary duration that exercises the
# duration policy's coda logic — and a role-tagged ensemble spec, so
# the gates cover the multi-voice shape the engine now writes.
SPEC_MATRIX: list[CompositionSpec] = [
    CompositionSpec(mood=mood, seed=seed, duration_seconds=duration)
    for mood, seed, duration in [
        (Mood.CALMING, 42, 30),
        (Mood.CALMING, 7, 180),
        (Mood.ELECTRIFYING, 3, 300),
        (Mood.SLEEP, 11, 600),
    ]
] + [
    CompositionSpec.model_validate(
        {
            "mood": "electrifying",
            "duration_seconds": 120,
            "seed": 9,
            "instrumentation": [
                {"role": "melody", "instrument": "piano"},
                {"role": "harmony", "instrument": "strings"},
                {"role": "bass", "instrument": "contrabass"},
                {"role": "percussion", "instrument": "drum_set"},
            ],
        }
    )
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

    def test_harmony_gate_is_active_not_vacuous(self, spec: CompositionSpec) -> None:
        """The gate receives the engine's per-bar chord context, so a
        score that steps off the harmony actually fails it."""
        from dataclasses import replace

        output = compose(spec)
        assert output.chord_bars, "the engine must publish per-bar chord pcs"
        tonic_pc = output.chord_bars[0][0]
        poisoned = replace(
            output, chord_bars=tuple((tonic_pc,) for _ in output.chord_bars)
        )
        result = gate_composition_correctness(poisoned)
        assert not result.passed, "a harmony gate without chord context would pass anything"

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


BAR_TICKS = 1920


def _solo_melody(pitches: list[int]) -> NotationScore:
    """A solo melody, one note per quarter at three note values.

    Written by hand so the gate's answer is known by construction: a solo
    clears the metrics that need an accompaniment by not having one.
    """
    durations = (480, 960, 1920)
    return NotationScore.make(
        ppq=PPQ,
        key=KeySignature(root="C", mode="major"),
        time_signature="4/4",
        tempo_bpm=80.0,
        measures=[
            Measure(
                index=index,
                start_tick=index * BAR_TICKS,
                end_tick=(index + 1) * BAR_TICKS,
                time_signature="4/4",
            )
            for index in range(3)
        ],
        notes=[
            NoteEvent(
                voice_id=VOICE_MELODY,
                pitch_midi=pitch,
                tick=index * 480,
                duration_ticks=durations[index % len(durations)],
            )
            for index, pitch in enumerate(pitches)
        ],
    )


# A stepwise major scale over an octave: every interval is a step, no
# leap to recover from, no repeat, three note values.
_STEPWISE = _solo_melody([60, 62, 64, 65, 67, 69, 71, 72])


class TestMusicalQualityGate:
    def test_a_stepwise_corpus_clears_every_threshold(self) -> None:
        result = gate_musical_quality([_STEPWISE])
        assert result.passed, result.detail

    def test_the_gate_is_active_not_vacuous(self) -> None:
        """A leaping line must fail the same gate the stepwise one passes.

        Without this, a gate that measured nothing — or measured the
        wrong voice — would report success on any input.
        """
        leaping = _solo_melody([60, 84, 60, 84, 60, 84, 60, 84])
        result = gate_musical_quality([leaping])
        assert not result.passed
        assert "step_ratio" in result.detail
        assert "max_leap_semitones" in result.detail

    def test_an_empty_matrix_fails_rather_than_passing_vacuously(self) -> None:
        # A corpus report of nothing meets every bar by having nothing to
        # measure; the gate is the caller that has to refuse that.
        result = gate_musical_quality([])
        assert not result.passed
        assert "non-empty matrix" in result.detail

    def test_the_generator_does_not_clear_the_bar_yet(self) -> None:
        """Documents the current gap as a measured fact, not an aspiration.

        The melody rewrite closed the melodic bars: the line moves by
        step, answers its leaps, is held in a C4-B5 band across the whole
        piece and repeats itself rarely, and `step_ratio`,
        `leap_recovery_ratio`, `repeat_ratio`, `range_semitones`,
        `max_leap_semitones` and `distinct_durations` all clear their
        thresholds now.

        The bass-figure library closed the second one: `bass_onset_patterns`
        measures a bass that played one figure in every bar of every piece,
        and it now reads 1.0 to 3.2 over the gate matrix against a bar of
        3, because the left hand states a figure per chord slot instead of
        repeating one bar for the piece's whole length.

        One threshold is left, and it is the harmony's.
        `tessitura_overlap_semitones` measures the harmony crowding the
        melody's register rather than sitting under it — 19.8 over the
        gate matrix and 12.5 over the pair below, both unchanged by the
        bass, against a bar of 4.

        That one got *worse* under the melody rewrite — 8.25 to 20.25 at
        the time — and the reason is the rewrite itself: the melody used to
        live at 72-102 and never descend past C5, so the harmony folded
        into 48-84 was mostly clear of it by accident, out of reach rather
        than out of the way. Now the line descends into the register the
        harmony occupies, which is the honest picture of two voices sharing
        a band, and the harmony has not yet been moved out of it. Clearing
        it for real is the harmony-clearance commit's work; this test is
        what keeps the gap visible until then — when the whole matrix
        clears, this assertion fails and asks to be read, rather than the
        improvement arriving silently.
        """
        specs = [
            CompositionSpec(mood=Mood.CALMING, seed=42, duration_seconds=30),
            CompositionSpec(mood=Mood.SLEEP, seed=11, duration_seconds=30),
        ]
        scores = [compose(spec).notation_score for spec in specs]
        result = gate_musical_quality(scores)
        assert not result.passed, (
            "the generator now clears the quality bar — re-read this test, "
            "confirm the music genuinely improved, and drop the assertion"
        )
        assert "tessitura_overlap_semitones" in result.detail


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
