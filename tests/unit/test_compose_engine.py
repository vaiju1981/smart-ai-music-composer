"""Unit tests for the composition engine."""

from __future__ import annotations

import pytest

from saimc.compose.engine import (
    CompositionEngineError,
    EngineErrorCode,
    compose,
)
from saimc.compose.forms import key_root_midi
from saimc.compose.linter import lint
from saimc.compose.score import NotationScore, PerformancePlan, realized_duration_seconds
from saimc.spec import CompositionSpec, Mood


def _spec(mood: Mood, *, duration: int = 60, seed: int | None = 42, **kw) -> CompositionSpec:
    base = {"mood": mood.value}
    base.update(kw)
    return CompositionSpec.model_validate(
        {"mood": mood.value, "duration_seconds": duration, "seed": seed, **kw}
    )


class TestComposeHappyPath:
    @pytest.mark.parametrize("mood", [Mood.CALMING, Mood.ELECTRIFYING, Mood.SLEEP])
    def test_returns_engine_output(self, mood: Mood) -> None:
        out = compose(_spec(mood, duration=60))
        assert isinstance(out.notation_score, NotationScore)
        assert isinstance(out.performance_plan, PerformancePlan)

    @pytest.mark.parametrize(
        ("mood", "target"),
        [
            (Mood.CALMING, 60),
            (Mood.CALMING, 180),
            (Mood.ELECTRIFYING, 60),
            (Mood.ELECTRIFYING, 300),
            (Mood.SLEEP, 180),
            (Mood.SLEEP, 600),
        ],
    )
    def test_realized_duration_within_tolerance(self, mood: Mood, target: int) -> None:
        out = compose(_spec(mood, duration=target))
        realised = realized_duration_seconds(out.performance_plan)
        delta = abs(realised - target) / target
        assert delta <= 0.02, f"mood={mood.value}, target={target}, realised={realised:.2f}"

    @pytest.mark.parametrize("mood", [Mood.CALMING, Mood.ELECTRIFYING, Mood.SLEEP])
    def test_passes_lint(self, mood: Mood) -> None:
        out = compose(_spec(mood, duration=180))
        report = lint(out.notation_score)
        assert report.passed, f"lint issues: {report.issues}"

    def test_deterministic_for_same_seed(self) -> None:
        out1 = compose(_spec(Mood.CALMING, duration=180, seed=42))
        out2 = compose(_spec(Mood.CALMING, duration=180, seed=42))
        assert out1.notation_score.compute_hash() == out2.notation_score.compute_hash()

    def test_different_seeds_give_different_scores(self) -> None:
        out1 = compose(_spec(Mood.CALMING, duration=180, seed=1))
        out2 = compose(_spec(Mood.CALMING, duration=180, seed=2))
        assert out1.notation_score.compute_hash() != out2.notation_score.compute_hash()

    def test_notes_are_sorted_canonically(self) -> None:
        out = compose(_spec(Mood.CALMING, duration=180))
        for prev, curr in zip(out.notation_score.notes, out.notation_score.notes[1:], strict=False):
            assert (curr.tick, curr.voice_id, curr.pitch_midi) >= (
                prev.tick,
                prev.voice_id,
                prev.pitch_midi,
            )


class TestComposeMultipleVoices:
    def test_has_bass_and_melody(self) -> None:
        out = compose(_spec(Mood.CALMING, duration=60))
        voice_ids = {n.voice_id for n in out.notation_score.notes}
        # At minimum we expect bass (voice 0) and melody (voice 1).
        assert 0 in voice_ids
        assert 1 in voice_ids


class TestComposeFailurePaths:
    def test_invalid_spec_raises(self) -> None:
        # Duration below the schema's 30s minimum is rejected by the
        # schema before reaching the engine.
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            compose(_spec(Mood.CALMING, duration=10))  # below the 30s minimum

    def test_engine_error_has_code(self) -> None:
        # Just assert the error machinery exists by constructing a
        # CompositionEngineError directly (the spec constraints make it
        # hard to provoke one through the public API).
        exc = CompositionEngineError(code=EngineErrorCode.DURATION_UNFULFILLABLE, message="test")
        assert exc.code == EngineErrorCode.DURATION_UNFULFILLABLE

    def test_coda_path_lints_clean(self) -> None:
        """45s calming uses a 4-bar coda; the engine still lints clean."""
        out = compose(_spec(Mood.CALMING, duration=45, seed=42))
        assert out.arrangement.coda_bars == 4
        report = lint(out.notation_score)
        assert report.passed, f"lint issues: {report.issues}"
        # The score includes coda measures on top of the body.
        assert out.notation_score.total_ticks() == (out.arrangement.total_bars_with_coda * 4 * 480)


class TestComposeIntegrationWithWorker:
    """Spot-check the wiring through `compose_stage`'s default-engine path."""

    def test_engine_used_by_stage(self, tmp_path) -> None:
        from datetime import UTC, datetime

        from saimc.jobs.stages import compose_stage
        from saimc.jobs.state import JobState
        from saimc.jobs.storage import Job, JobStorage

        storage = JobStorage(str(tmp_path))
        now = datetime.now(UTC)
        job = Job(
            job_id="test",
            created_at=now,
            updated_at=now,
            state=JobState.PARSING,
            progress=0.0,
            current_stage="parsing",
            input_prompt="calming piano",
            input_spec=_spec(Mood.CALMING, duration=60),
        )
        result = compose_stage(job, storage)
        assert result.next_state == JobState.VALIDATING
        assert result.error is None


class TestMusicalShape:
    """The §10 quality upgrades: real left hand, register split, A/B form."""

    def test_bass_plays_root_fifth_per_bar(self) -> None:
        out = compose(_spec(Mood.CALMING, duration=60))
        bass = [n for n in out.notation_score.notes if n.voice_id == 0]
        bar_ticks_count = out.notation_score.ppq * 4
        first_bar = [n for n in bass if n.tick < bar_ticks_count]
        assert len(first_bar) == 2  # root + fifth, not a held drone
        pitches = {n.pitch_midi for n in first_bar}
        assert len(pitches) == 2
        root, fifth = sorted(pitches)
        assert fifth - root == 7  # perfect fifth above the root

    def test_melody_sits_above_bass(self) -> None:
        out = compose(_spec(Mood.CALMING, duration=60))
        bass_avg = sum(n.pitch_midi for n in out.notation_score.notes if n.voice_id == 0) / len(
            [n for n in out.notation_score.notes if n.voice_id == 0]
        )
        melody_min = min(n.pitch_midi for n in out.notation_score.notes if n.voice_id == 1)
        assert melody_min > bass_avg

    def test_repeated_sections_rotate_template_variants(self) -> None:
        out = compose(_spec(Mood.ELECTRIFYING, duration=300, seed=3))
        arrangement = out.arrangement
        assert arrangement.repetition_count > 1
        # Section harmony differs between rep 0 and rep 1.
        bar_ticks_count = out.notation_score.ppq * 4
        section_ticks = arrangement.form_bars * bar_ticks_count
        # Variant rotation reorders the progression, so the chord
        # sounding at each section's downbeat differs (e.g. I vs vi).
        first_downbeat = next(
            n.pitch_midi for n in out.notation_score.notes if n.voice_id == 0
        )
        second_downbeat = next(
            n.pitch_midi
            for n in out.notation_score.notes
            if n.voice_id == 0 and n.tick >= section_ticks
        )
        assert first_downbeat != second_downbeat

    def test_coda_ends_on_tonic(self) -> None:
        out = compose(_spec(Mood.CALMING, duration=45))
        arrangement = out.arrangement
        assert arrangement.coda_bars > 0
        bar_ticks_count = out.notation_score.ppq * 4
        coda_start = arrangement.total_bars * bar_ticks_count
        coda_bass = [n for n in out.notation_score.notes if n.voice_id == 0 and n.tick >= coda_start]
        last_chord_root = coda_bass[-2].pitch_midi  # the bar's downbeat root
        tonic = key_root_midi(out.key) % 12  # key root pitch class
        assert last_chord_root % 12 == tonic % 12

    def test_tempo_bpm_constraint_is_honoured(self) -> None:
        out = compose(_spec(Mood.CALMING, duration=180, tempo_bpm=63))
        assert out.notation_score.tempo.bpm == 63

    def test_velocity_follows_an_arch(self) -> None:
        out = compose(_spec(Mood.CALMING, duration=60))
        melody = [n for n in out.notation_score.notes if n.voice_id == 1]
        velocities = [n.velocity for n in melody]
        assert max(velocities) - min(velocities) >= 6  # the piece breathes
