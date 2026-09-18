"""Release-gate suite per `docs/roadmap.md` §8.

Runs every automatable MVP acceptance gate against a matrix of real
engine outputs. Gates that need external binaries (FluidSynth, the
render-service, the audited FFmpeg) reuse these functions in the
real-binary end-to-end run.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from saimc.compose.engine import compose
from saimc.compose.plan import CompositionPlan, default_plan
from saimc.compose.score import (
    PPQ,
    VOICE_MELODY,
    KeySignature,
    Measure,
    NotationScore,
    NoteEvent,
)
from saimc.quality import score_corpus, score_piece
from saimc.release import (
    gate_canonical_reproducibility,
    gate_composition_correctness,
    gate_duration_tolerance,
    gate_midi_parseable_and_onsets,
    gate_musical_quality,
    gate_musicxml_structural,
    gate_quality_breach_rate,
    gate_render_time_budget,
    gate_spec_round_trip,
)
from saimc.release.gates import MAX_THRESHOLD_BREACH_RATE
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


PLANNED_SPEC: CompositionSpec = CompositionSpec.model_validate(
    {
        "mood": "electrifying",
        "duration_seconds": 60,
        "seed": 9,
        "instrumentation": [
            {"role": "melody", "instrument": "piano"},
            {"role": "harmony", "instrument": "strings"},
            {"role": "bass", "instrument": "contrabass"},
            {"role": "percussion", "instrument": "drum_set"},
        ],
    }
)
"""The spec of the one planned case: four roles, so the voices and
percussion layers have something to move, at the shortest duration that
still arranges to more than one section."""

RELEASE_PLAN: CompositionPlan = replace(
    default_plan(PLANNED_SPEC),
    # One knob per layer, deliberately: the plan path is gated by this case
    # alone, so a layer whose read was dropped between B2 and B7 should be
    # able to fail it. A single knob would gate a single layer.
    apex_position=0.66,  # melody
    line_band_semitones=16,  # melody
    cadence_degree=5,  # harmony
    modulation_offset=0,  # harmony/modulation
    harmony_pad_velocity=52,  # voices
    harmony_broken_chord=True,  # voices
    percussion_velocity_scale=0.8,  # percussion
    section_crash_velocity=88,  # percussion
    section_energy_peak=1.5,  # sections
    percussion_rest_section=2,  # sections
    intro_bars=2,  # arrangement
    max_repeats=2,  # arrangement
)
"""The plan the release matrix composes under, besides the defaults.

`gate_canonical_reproducibility` compares two runs of the *same build*, so
without a case like this every gate in the matrix would be a gate over a
path nothing ships on: a `compose` that accepted a plan and ignored it
would pass all of them. Two things make this one count — the gate asserts
the plan it composed under is the plan it was handed, and
`test_the_planned_case_is_a_different_piece` asserts the notes moved, which
is what a dropped plan read would break.
"""


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


# A well-shaped line over an octave: mostly steps, with a leap up to the
# octave in it that the next note answers by step, no repeat, three note
# values. It has to carry the leap: a line of nothing but steps now measures
# `step_ratio` 1.00 against a cap of 0.90 and `leap_ratio` 0.00 against a
# floor of 0.01, so the scale this fixture used to be was a clean corpus only
# while the scorecard could see neither fault.
_SHAPED = _solo_melody([60, 62, 64, 65, 72, 71, 72, 74])


class TestMusicalQualityGate:
    def test_a_well_shaped_corpus_clears_every_threshold(self) -> None:
        result = gate_musical_quality([_SHAPED])
        assert result.passed, result.detail

    def test_the_gate_is_active_not_vacuous(self) -> None:
        """A leaping line must fail the same gate the shaped one passes.

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

    def test_the_generator_clears_the_bar(self) -> None:
        """The gap this file pinned open, closed — read and asserted.

        `saimc.quality` measures thirteen properties of the music and this
        gate is the claim that a corpus clears every one of them. When the
        gate landed the engine missed seven. The melody rewrite closed six: the
        line moves by step, answers its leaps, is held in a C4-B5 band
        across the whole piece and repeats itself rarely, so `step_ratio`,
        `leap_recovery_ratio`, `repeat_ratio`, `range_semitones`,
        `max_leap_semitones` and `distinct_durations` all clear. The
        bass-figure library closed `bass_onset_patterns`: a bass that had
        played one figure in every bar of every piece states a figure per
        chord slot instead, and reads 3.0-3.2 against a bar of 3.

        One threshold was left, and this test held it open on purpose.
        `tessitura_overlap_semitones` measured the harmony crowding the
        melody's register rather than sitting under it — 19.8 over the
        gate matrix and 12.5 over the pair below, against a bar of 4 —
        with the instruction that when the whole matrix cleared, the
        assertion was to be re-read and dropped rather than let the
        improvement arrive silently.

        It cleared, in two reads. The melody rewrite made it *worse*, 8.25
        to 20.25: the line used to live at 72-102 and never descend past
        C5, so the harmony folded into 48-84 was clear of it by accident,
        out of reach rather than out of the way. Bringing the tune down
        into a singable band is what put the two voices in one register,
        which is the honest picture of two voices sharing a band — and the
        harmony-clearance pass is what moved the bed out of it: the bed's
        top now sits `HARMONY_MELODY_CLEARANCE` below the lowest note the
        piece's melody reaches, so the bands are disjoint and both corpora
        read 0.0.

        What the reading confirmed, beyond the number: the bed kept its
        own register through the whole piece instead of following the
        tune bar by bar, every voice kept its onsets and its rhythm, and
        the pieces the bed thinned are the ones whose melody sits at the
        bottom of its band — where the alternative to thinning would have
        been the clash the crowding rule exists to prevent.
        """
        specs = [
            CompositionSpec(mood=Mood.CALMING, seed=42, duration_seconds=30),
            CompositionSpec(mood=Mood.SLEEP, seed=11, duration_seconds=30),
        ]
        scores = [compose(spec).notation_score for spec in specs]
        result = gate_musical_quality(scores)
        assert result.passed, result.detail

    def test_the_release_matrix_clears_every_threshold(self) -> None:
        """The same bar, held against the matrix the other gates run.

        The pair above is the one the open loop was pinned on. This is the
        corpus the release gates themselves measure — every mood, a short,
        a typical and the §10 #11 cap duration, plus a role-tagged
        ensemble — so the quality bar is held against the pieces the
        structural gates run on rather than a pair chosen for it.
        """
        report = score_corpus(
            "release matrix",
            [
                score_piece(
                    compose(spec).notation_score,
                    piece=f"{spec.mood.value}-{spec.seed}-{spec.duration_seconds}",
                )
                for spec in SPEC_MATRIX
            ],
        )
        assert report.passed, report.failure_reasons


# The sampled grid the breach-rate gate is measured over. Every axis is a
# literal rather than drawn from `Mood` or from a duration constant: the rate
# is a joint property of the axes, so an axis that followed a table would move
# the number without the engine moving, and the ceiling would then be pinning
# the sample rather than the generator.
QUALITY_GRID_MOODS: tuple[Mood, ...] = (Mood.CALMING, Mood.ELECTRIFYING, Mood.SLEEP)
QUALITY_GRID_DURATIONS: tuple[int, ...] = (30, 120, 300)
QUALITY_GRID_SEEDS: tuple[int, ...] = (1, 2, 3, 4, 5, 6)

QUALITY_GRID: list[CompositionSpec] = [
    CompositionSpec(mood=mood, seed=seed, duration_seconds=duration)
    for mood in QUALITY_GRID_MOODS
    for duration in QUALITY_GRID_DURATIONS
    for seed in QUALITY_GRID_SEEDS
]


@pytest.fixture(scope="module")
def grid_scores() -> list[NotationScore]:
    """The grid composed once: 54 cells is ~1.6 s and three cases read them."""
    return [compose(spec).notation_score for spec in QUALITY_GRID]


class TestTheQualityBreachRate:
    """The quality claim over a sampled grid rather than a curated matrix.

    `gate_musical_quality` judges the corpus mean over five hand-picked specs
    and passes; the same engine breaches at least one threshold in 27 of the
    54 cells below. Both statements are true and only one of them is about the
    product, which is why the second has its own gate.

    Measured on the tree this landed on, composing and scoring all 54 cells in
    about 1.6 s — cheap enough to run on every commit, which is the only cost a
    ratchet may have:

    | | 30 s | 120 s | 300 s | total |
    |---|---|---|---|---|
    | calming | 3/6 | 1/6 | 0/6 | 4/18 |
    | electrifying | 6/6 | 6/6 | 6/6 | 18/18 |
    | sleep | 3/6 | 1/6 | 1/6 | 5/18 |

    27/54 = 50.0%, breaching `texture_hierarchy` 18, `harmony_pad_coverage` 18,
    `leap_recovery_ratio` 5, `max_leap_semitones` 5 and `step_ratio` 1.

    Two things the table says that one number would not, and the second is why
    the grid pins the durations as well as the moods. Every electrifying cell
    breaches at every duration and every seed on the same two metrics — a
    *mood* property, nothing to do with length — so the change that would move
    this rate most is one mood's texture hierarchy and pad coverage rather than
    a general melody or rhythm defect. The other two moods breach mostly at
    30 s and are near-clean at 300 s, so a grid that sampled lengths unevenly
    would move the rate while the engine stood still.

    The ceiling is the measured rate and may only fall. Lowering it as those
    two metrics are fixed is the ordinary edit; raising it is a decision that
    has to be argued for.
    """

    def test_the_grid_spans_every_mood_and_duration(self) -> None:
        """The premises the rate rests on, asserted rather than trusted.

        `set(...) == set(Mood)` is the ratchet on the mood axis: a fourth mood
        fails here rather than quietly leaving the grid, which is the
        difference between a rate over the reachable space and a rate over
        whatever three moods someone last typed.
        """
        assert set(QUALITY_GRID_MOODS) == set(Mood)
        assert len(QUALITY_GRID) == (
            len(QUALITY_GRID_MOODS) * len(QUALITY_GRID_DURATIONS) * len(QUALITY_GRID_SEEDS)
        )
        per_cell = Counter((spec.mood, spec.duration_seconds) for spec in QUALITY_GRID)
        assert set(per_cell.values()) == {len(QUALITY_GRID_SEEDS)}, (
            "the grid is not a full cross product, so the axes are weighted unevenly"
        )

    def test_the_sampled_grid_is_within_the_recorded_ceiling(
        self, grid_scores: list[NotationScore]
    ) -> None:
        result = gate_quality_breach_rate(grid_scores)
        assert result.passed, result.detail

    def test_the_recorded_ceiling_is_the_measured_rate(
        self, grid_scores: list[NotationScore]
    ) -> None:
        """The ceiling is a measurement, so it is pinned to the measurement.

        `MAX_THRESHOLD_BREACH_RATE`'s own docstring says it is a measurement
        rather than a target, and this is what makes that enforceable: the
        constant has to equal what the grid measures. That closes the case a
        plain "at or below" check cannot see — a change that made more pieces
        breach while the ceiling was left where it was. An improvement fails
        here too, deliberately: the ceiling is then lowered in the same commit,
        which is the ordinary edit and the direction a ratchet may move.
        """
        rate = sum(1 for score in grid_scores if score_piece(score).findings()) / len(grid_scores)
        assert rate == MAX_THRESHOLD_BREACH_RATE, (
            f"the recorded ceiling is {MAX_THRESHOLD_BREACH_RATE:.4f} and the grid measures "
            f"{rate:.4f}: lower the ceiling to the measurement, or argue for raising it"
        )

    def test_the_ceiling_is_the_ratchet_and_it_bites(
        self, grid_scores: list[NotationScore], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The guard fired, because a ratchet that cannot fail is a comment.

        Set to the rate the grid actually measures the same grid passes — the
        ceiling is inclusive, so "at or below" is the claim. Lowered by a hair
        it fails, which is the direction the ratchet exists to hold: a change
        that made more pieces breach would otherwise arrive as a higher
        measured rate beside an unmoved ceiling, and nothing would say so.
        """
        rate = sum(1 for score in grid_scores if score_piece(score).findings()) / len(grid_scores)
        ceiling = "saimc.release.gates.MAX_THRESHOLD_BREACH_RATE"

        monkeypatch.setattr(ceiling, rate)
        assert gate_quality_breach_rate(grid_scores).passed, "the ceiling is not inclusive"

        monkeypatch.setattr(ceiling, rate - 0.001)
        result = gate_quality_breach_rate(grid_scores)
        assert not result.passed, "a ratchet that cannot fail guards nothing"
        assert "above a ceiling" in result.detail
        assert "most often:" in result.detail, "a failing rate has to name the bars behind it"

    def test_an_empty_grid_fails_rather_than_passing_vacuously(self) -> None:
        # A rate over nothing is not a rate, and 0/0 would read as perfect.
        result = gate_quality_breach_rate([])
        assert not result.passed
        assert "empty grid" in result.detail


class TestThePlannedCaseIsLive:
    """The one release case that names its own plan.

    Every other gate here composes at the engine's defaults, and a
    `compose` that took a plan and ignored it would pass all of them. These
    are the two checks that make the planned case worth its row: the gate
    is handed the plan and the notes are the ones it asks for.
    """

    def test_the_reproducibility_gate_covers_the_plan_path(self) -> None:
        result = gate_canonical_reproducibility(PLANNED_SPEC, RELEASE_PLAN)
        assert result.passed, result.detail
        assert RELEASE_PLAN.compute_hash()[:12] in result.detail

    def test_the_gate_refuses_a_plan_it_did_not_compose_under(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The check that catches a plan dropped on the way in, fired.

        Comparing two runs against each other cannot catch it: a `compose`
        that discarded its `plan` argument would agree with itself, on the
        wrong music. So the engine is sabotaged into ignoring the plan —
        the one failure mode that leaves both runs identical — and the gate
        has to refuse.
        """

        def _ignoring_engine(spec, *, plan=None):
            return compose(spec)

        monkeypatch.setattr("saimc.release.gates.compose", _ignoring_engine)
        result = gate_canonical_reproducibility(PLANNED_SPEC, RELEASE_PLAN)
        assert not result.passed
        assert "not the one that was supplied" in result.detail

    def test_the_planned_case_is_a_different_piece(self) -> None:
        """The notes moved, so the plan is not merely recorded.

        Compared against the same spec with no plan, which is the one
        comparison that distinguishes "the engine read the plan" from "the
        engine wrote down the plan it was given".
        """
        planned = compose(PLANNED_SPEC, plan=RELEASE_PLAN)
        defaulted = compose(PLANNED_SPEC)
        assert planned.plan == RELEASE_PLAN
        assert planned.plan != default_plan(PLANNED_SPEC)
        assert (
            planned.notation_score.compute_hash() != defaulted.notation_score.compute_hash()
        ), "a plan that changes nothing would make every gate above vacuous"


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
