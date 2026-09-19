"""Unit tests for the per-stage job runners."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from saimc.compose.duration import DurationArrangement
from saimc.compose.engine import EngineOutput
from saimc.compose.forms import ChordSlot, ChordTemplate
from saimc.compose.score import (
    KeySignature,
    Measure,
    NotationScore,
    NoteEvent,
    PerformanceNoteEvent,
    PerformancePlan,
)
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
from saimc.spec import CompositionSpec, Instrument, Mood, VoiceRole


@pytest.fixture
def store(tmp_path):
    return JobStorage(tmp_path)


def _spec_dict() -> dict[str, object]:
    return CompositionSpec(mood=Mood.CALMING, seed=42).model_dump(mode="json")


def _stub_engine(spec, *, plan=None) -> EngineOutput:
    """The smallest output `compose_stage` will accept: one bar, one note.

    Module-level because three tests need an engine that composes nothing
    in particular, and only one of them cares what it composes.
    """
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
        template=ChordTemplate(name="stub_1bar", bars=1, chords=(ChordSlot(0, 1),)),
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

    def test_carries_the_model_identifier_off_the_parse_result(self, store: JobStorage) -> None:
        """The adapter reports the model in `extra`; the job keeps it.

        §9 requires the manifest to name the model alongside the parser
        source, and this is the only point at which it is available.
        """
        job = store.create("calming piano")
        transition_to(job, JobState.PARSING)
        spec = CompositionSpec.model_validate(_spec_dict())
        client = _StubLLM(
            ParseResult(
                parser_source="llm",
                spec=spec,
                extra={"model_identifier": "gemma4:31b-cloud"},
            )
        )
        result = parse_stage(job, client, request_id="t-model")
        assert result.job.model == "gemma4:31b-cloud"

    def test_records_no_model_when_the_fallback_wrote_the_spec(self, store: JobStorage) -> None:
        """A hybrid parse is half-LLM, but the spec came from the fallback
        parser, so there is no model to attribute the notes to."""
        job = store.create("calming piano")
        transition_to(job, JobState.PARSING)
        spec = CompositionSpec.model_validate(_spec_dict())
        client = _StubLLM(ParseResult(parser_source="fallback", spec=spec))
        result = parse_stage(job, client, request_id="t-fallback")
        assert result.job.model is None

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
        # "angsty piano" is out-of-vocabulary, so the fallback's
        # rejection (with the vocabulary hint) is surfaced, and the
        # SpecError's own stage propagates instead of a hardcoded one.
        assert result.error.error_code == "out_of_vocabulary"
        assert result.error.stage == "fallback"

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
        result = compose_stage(job, store, engine=lambda s, **_: (None, None))
        assert result.next_state == JobState.FAILED
        assert result.error is not None
        assert result.error.error_code == "no_spec"

    def test_with_engine_stub_and_spec_advances(self, store: JobStorage) -> None:
        """A stub engine that returns a valid EngineOutput advances."""
        job = store.create("p")
        job.input_spec = CompositionSpec(mood=Mood.CALMING)
        result = compose_stage(job, store, engine=_stub_engine)
        assert result.next_state == JobState.VALIDATING
        assert result.error is None

    def test_engine_exception_caught(self, store: JobStorage) -> None:
        def _bad_engine(spec, *, plan=None):
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

    def test_unknown_instrumentation_fails_cleanly(self, store: JobStorage) -> None:
        """An unregistered instrumentation names what IS supported."""
        job = store.create("p")
        spec = CompositionSpec(mood=Mood.CALMING)
        # Bypass validation: a value the Instrument enum would never build.
        forged = SimpleNamespace(value="string_orchestra")
        entry = SimpleNamespace(role=VoiceRole.MELODY, instrument=forged)
        object.__setattr__(spec, "instrumentation", [entry])
        job.input_spec = spec
        result = compose_stage(job, store)
        assert result.next_state == JobState.FAILED
        assert result.error is not None
        assert result.error.error_code == "instrumentation_unsupported"
        assert "piano" in (result.error.message or "")

    def test_drum_set_composes(self, store: JobStorage) -> None:
        """drum_set is a supported instrumentation, not a lookup failure."""
        job = store.create("p")
        job.input_spec = CompositionSpec(
            mood=Mood.ELECTRIFYING,
            instrumentation=Instrument.DRUM_SET,
            duration_seconds=30,
            seed=42,
        )
        result = compose_stage(job, store)
        assert result.next_state == JobState.VALIDATING
        assert result.error is None

    def test_the_jobs_plan_is_handed_to_the_engine(self, store: JobStorage) -> None:
        """The widened call site, proven live rather than assumed.

        `compose_stage` used to call `engine(job.input_spec)`. Threading a
        plan through is a change no other test can see: the engine stub
        would accept it, ignore it, and the stage would still advance. So
        the handed-over object is captured and compared by identity — the
        plan the job stores is the one the engine is given, not a copy, not
        a re-resolution.
        """
        from saimc.compose.plan import default_plan

        handed: list[object] = []

        def _capturing_engine(spec, *, plan=None):
            handed.append(plan)
            return _stub_engine(spec)

        job = store.create("p")
        job.input_spec = CompositionSpec(mood=Mood.CALMING)
        job.input_plan = default_plan(job.input_spec)
        compose_stage(job, store, engine=_capturing_engine)
        assert handed == [job.input_plan]

    def test_a_job_with_no_plan_hands_the_engine_none(self, store: JobStorage) -> None:
        """`None` means "the engine's defaults", and the stage must not
        resolve it into a plan of its own — the plan a piece was composed
        under is the engine's to publish, because the engine is the only
        thing that can say what it actually used."""
        handed: list[object] = []

        def _capturing_engine(spec, *, plan=None):
            handed.append(plan)
            return _stub_engine(spec)

        job = store.create("p")
        job.input_spec = CompositionSpec(mood=Mood.CALMING)
        compose_stage(job, store, engine=_capturing_engine)
        assert handed == [None]

    def test_a_jobs_plan_reaches_the_sidecar_through_the_real_engine(
        self, store: JobStorage
    ) -> None:
        """The whole chain, with nothing stubbed: job → stage → engine →
        sidecar. The render stages read the sidecar rather than
        re-composing, so a plan that stops anywhere short of it is a plan
        the rest of the pipeline cannot see."""
        from dataclasses import replace

        from saimc.compose.plan import default_plan
        from saimc.compose.serialization import read_engine_output

        job = store.create("p")
        job.input_spec = CompositionSpec(mood=Mood.CALMING, duration_seconds=60, seed=42)
        job.input_plan = replace(default_plan(job.input_spec), cadence_degree=5)
        result = compose_stage(job, store)
        assert result.next_state == JobState.VALIDATING

        recorded = read_engine_output(store.job_dir(job.job_id) / "engine_output.json")
        assert recorded.plan == job.input_plan

        # And the plan is not merely recorded: the notes are the ones that
        # cadence produces, not the defaults'. Without this the sidecar
        # could hold the plan while the engine ignored it — which is the
        # failure the whole seam exists to rule out.
        default_job = store.create("p")
        default_job.input_spec = job.input_spec
        compose_stage(default_job, store)
        default_recorded = read_engine_output(
            store.job_dir(default_job.job_id) / "engine_output.json"
        )
        assert (
            recorded.notation_score.compute_hash()
            != default_recorded.notation_score.compute_hash()
        )
