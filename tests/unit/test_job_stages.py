"""Unit tests for the per-stage job runners."""

from __future__ import annotations

import sys

import pytest

from saimc.jobs.stages import (
    SubprocessTimeoutError,
    compose_stage,
    parse_stage,
    safe_run,
    transition_to,
)
from saimc.jobs.state import IllegalTransitionError, JobState
from saimc.jobs.storage import JobStorage
from saimc.llm.base import ParseRequest, ParseResult
from saimc.spec import CompositionSpec, Mood


@pytest.fixture
def store(tmp_path):
    return JobStorage(tmp_path)


def _spec_dict() -> dict[str, object]:
    return CompositionSpec(mood=Mood.CALMING, seed=42).model_dump(mode="json")


class _StubLLM:
    def __init__(self, result: ParseResult) -> None:
        self._result = result
        self.calls = 0

    async def parse(self, request: ParseRequest) -> ParseResult:
        self.calls += 1
        return self._result


class TestSafeRun:
    def test_runs_command_successfully(self) -> None:
        proc = safe_run([sys.executable, "-c", "print('hi')"], timeout_s=5.0)
        assert proc.returncode == 0
        assert "hi" in proc.stdout

    def test_timeout_raises(self) -> None:
        with pytest.raises(SubprocessTimeoutError):
            safe_run(
                [sys.executable, "-c", "import time; time.sleep(10)"],
                timeout_s=0.5,
            )

    def test_nonzero_returncode_returned(self) -> None:
        proc = safe_run(
            [sys.executable, "-c", "import sys; sys.exit(3)"],
            timeout_s=5.0,
        )
        assert proc.returncode == 3

    def test_empty_cmd_rejected(self) -> None:
        with pytest.raises(ValueError):
            safe_run([], timeout_s=1.0)


class TestTransitionTo:
    def test_legal_transition(self, store: JobStorage) -> None:
        job = store.create("p")
        transition_to(job, JobState.PARSING)
        assert job.state == JobState.PARSING
        assert job.progress > 0.0

    def test_illegal_transition_raises(self, store: JobStorage) -> None:
        job = store.create("p")
        with pytest.raises(IllegalTransitionError):
            transition_to(job, JobState.COMPLETE)


class TestParseStage:
    def test_success_advances_to_composing(self, store: JobStorage) -> None:
        job = store.create("calming piano")
        transition_to(job, JobState.PARSING)
        spec = CompositionSpec.model_validate(_spec_dict())
        client = _StubLLM(ParseResult(parser_source="llm", spec=spec, extra={"latency_ms": "10"}))
        result = parse_stage(job, client, request_id="t1")
        assert result.next_state == JobState.COMPOSING
        assert result.error is None
        assert result.job.input_spec == spec
        assert result.job.parser_source == "llm"
        assert result.job.seed == 42
        assert client.calls == 1

    def test_failure_returns_failed_state(self, store: JobStorage) -> None:
        job = store.create("angsty piano")
        transition_to(job, JobState.PARSING)
        client = _StubLLM(
            ParseResult(
                parser_source="llm",
                error=__import__("saimc.spec", fromlist=["SpecError"]).SpecError(
                    error_code="schema_invalid",
                    message="bad",
                    stage="validating",
                ),
            )
        )
        result = parse_stage(job, client, request_id="t2")
        assert result.next_state == JobState.FAILED
        assert result.error is not None
        assert result.error.error_code == "schema_invalid"

    def test_parser_exception_is_caught(self, store: JobStorage) -> None:
        class _Boom:
            async def parse(self, request: ParseRequest) -> ParseResult:
                raise RuntimeError("kaboom")

        job = store.create("calming")
        transition_to(job, JobState.PARSING)
        result = parse_stage(job, _Boom(), request_id="t3")  # type: ignore[arg-type]
        assert result.next_state == JobState.FAILED
        assert result.error is not None
        assert result.error.error_code == "parser_exception"


class TestComposeStage:
    def test_without_spec_fails(self, store: JobStorage) -> None:
        job = store.create("p")
        result = compose_stage(job, store, engine=lambda s: (None, None))
        assert result.next_state == JobState.FAILED
        assert result.error is not None
        assert result.error.error_code == "no_spec"

    def test_with_engine_stub_and_spec_advances(self, store: JobStorage) -> None:
        """A stub engine that returns a valid EngineOutput advances."""
        from saimc.compose.duration import DurationArrangement
        from saimc.compose.engine import EngineOutput
        from saimc.compose.forms import ChordTemplate
        from saimc.compose.score import (
            KeySignature,
            Measure,
            NotationScore,
            NoteEvent,
            PerformanceNoteEvent,
            PerformancePlan,
        )

        def _stub_engine(spec):
            # Minimal valid 1-bar NotationScore in C major 4/4.
            note = NoteEvent(
                voice_id=0,
                pitch_midi=60,
                tick=0,
                duration_ticks=480,
                velocity=64,
            )
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
                template=ChordTemplate(
                    name="stub_1bar",
                    bars=1,
                    chords=((0, 1),),
                ),
                repetition_count=1,
                total_bars=1,
                tempo_bpm=80.0,
            )
            return EngineOutput(
                notation_score=notation,
                performance_plan=performance,
                arrangement=arrangement,
                key=KeySignature(root="C", mode="major"),
                time_signature="4/4",
            )

        job = store.create("p")
        job.input_spec = CompositionSpec(mood=Mood.CALMING)
        result = compose_stage(job, store, engine=_stub_engine)
        assert result.next_state == JobState.VALIDATING
        assert result.error is None

    def test_engine_exception_caught(self, store: JobStorage) -> None:
        def _bad_engine(spec):
            raise ValueError("music21 broke")

        job = store.create("p")
        job.input_spec = CompositionSpec(mood=Mood.CALMING)
        result = compose_stage(job, store, engine=_bad_engine)
        assert result.next_state == JobState.FAILED
        assert result.error is not None
        assert result.error.error_code == "compose_failed"
        assert "music21 broke" in (result.error.message or "")

    def test_default_engine_is_real(self, store: JobStorage) -> None:
        """Without an engine argument, the real Phase 1 composer is used."""
        job = store.create("p")
        job.input_spec = CompositionSpec(mood=Mood.CALMING, duration_seconds=60, seed=42)
        result = compose_stage(job, store)
        assert result.next_state == JobState.VALIDATING
        assert result.error is None
