"""Unit tests for the parser benchmark runner and corpus loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from saimc.benchmark import (
    LATENCY_P95_MAX_S,
    QUALITY_FIELD_LEVEL_ACCURACY,
    QUALITY_FIRST_PASS_VALIDITY,
    QUALITY_UNSUPPORTED_REJECTION,
    BenchmarkReport,
    ParseObservation,
    score_corpus,
)
from saimc.benchmark_corpus import BenchmarkRecord, load_corpus
from saimc.spec import Mood


def _obs(
    *,
    mood: str | None = "calming",
    duration: int = 180,
    error_code: str | None = None,
    parser_source: str = "llm",
    attempts: int = 1,
    raw_first_response_valid: bool = True,
    latency_ms: int = 50,
) -> ParseObservation:
    from saimc.spec import CompositionSpec

    if mood is None:
        return ParseObservation(
            spec=None,
            error_code=error_code,
            parser_source=parser_source,
            attempts=attempts,
            structured_output=False,
            latency_ms=latency_ms,
            raw_first_response_valid=raw_first_response_valid,
            cost_usd=0.0,
        )
    spec = CompositionSpec(mood=Mood(mood), duration_seconds=duration)
    return ParseObservation(
        spec=spec,
        error_code=error_code,
        parser_source=parser_source,
        attempts=attempts,
        structured_output=True,
        latency_ms=latency_ms,
        raw_first_response_valid=raw_first_response_valid,
        cost_usd=0.0,
    )


def _rec(
    *,
    record_id: str,
    category: str,
    expected_outcome: str,
    expected_spec: dict[str, object] | None = None,
    expected_error: str | None = None,
) -> BenchmarkRecord:
    return BenchmarkRecord(
        id=record_id,
        category=category,
        prompt=f"prompt-for-{record_id}",
        expected_outcome=expected_outcome,
        expected_spec=expected_spec,
        expected_error=expected_error,
        label_rationale="test record",
    )


def test_corpus_loader_parses_known_file() -> None:
    records, stats = load_corpus()
    assert stats.total == len(records)
    assert stats.total >= 20
    assert stats.accepted_expected + stats.rejected_expected == stats.total
    assert all(
        c in {"supported_paraphrase", "boundary", "malformed", "unsupported"}
        for c in stats.by_category
    )


def test_corpus_loader_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_corpus(tmp_path / "nonexistent.jsonl")


def test_corpus_loader_malformed_record(tmp_path: Path) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"id": "x", "category": "supported_paraphrase"}\n')
    with pytest.raises(ValueError, match="Invalid record"):
        load_corpus(bad)


def test_score_record_accepted_matched_and_field_exact() -> None:
    rec = _rec(
        record_id="r1",
        category="supported_paraphrase",
        expected_outcome="accepted",
        expected_spec={
            "schema_version": 2,
            "request_kind": "mood_generation",
            "duration_seconds": 180,
            "tempo_bpm": None,
            "key": None,
            "time_signature": "4/4",
            "mood": "calming",
            "instrumentation": "piano",
            "seed": None,
            "humanization": "light",
        },
    )
    from saimc.benchmark import score_record

    outcome = score_record(rec, _obs(mood="calming", duration=180))
    assert outcome.matched
    assert outcome.field_exact_match is True


def test_score_record_accepted_wrong_duration() -> None:
    rec = _rec(
        record_id="r2",
        category="supported_paraphrase",
        expected_outcome="accepted",
        expected_spec={
            "schema_version": 2,
            "request_kind": "mood_generation",
            "duration_seconds": 300,
            "tempo_bpm": None,
            "key": None,
            "time_signature": "4/4",
            "mood": "calming",
            "instrumentation": "piano",
            "seed": None,
            "humanization": "light",
        },
    )
    from saimc.benchmark import score_record

    outcome = score_record(rec, _obs(mood="calming", duration=180))
    assert outcome.matched
    assert outcome.field_exact_match is False


def test_score_record_rejected_correctly() -> None:
    rec = _rec(
        record_id="r3",
        category="unsupported",
        expected_outcome="rejected",
        expected_error="out_of_vocabulary",
    )
    from saimc.benchmark import score_record

    outcome = score_record(rec, _obs(mood=None, error_code="out_of_vocabulary"))
    assert outcome.matched
    assert outcome.field_exact_match is None


def test_score_record_rejected_but_accepted() -> None:
    rec = _rec(
        record_id="r4",
        category="unsupported",
        expected_outcome="rejected",
        expected_error="out_of_vocabulary",
    )
    from saimc.benchmark import score_record

    outcome = score_record(rec, _obs(mood="calming"))
    assert not outcome.matched


def test_score_corpus_passes_with_perfect_observations() -> None:
    recs = [
        _rec(
            record_id=f"a{i}",
            category="supported_paraphrase",
            expected_outcome="accepted",
            expected_spec={
                "schema_version": 2,
                "request_kind": "mood_generation",
                "duration_seconds": 180,
                "tempo_bpm": None,
                "key": None,
                "time_signature": "4/4",
                "mood": "calming",
                "instrumentation": "piano",
                "seed": None,
                "humanization": "light",
            },
        )
        for i in range(95)
    ] + [
        _rec(
            record_id=f"u{i}",
            category="unsupported",
            expected_outcome="rejected",
            expected_error="out_of_vocabulary",
        )
        for i in range(5)
    ]
    obs = [
        _obs(mood="calming", duration=180, raw_first_response_valid=True, latency_ms=50)
    ] * 95 + [_obs(mood=None, error_code="out_of_vocabulary", latency_ms=50)] * 5

    report = score_corpus("perfect-client", recs, obs)
    assert isinstance(report, BenchmarkReport)
    assert report.passed
    assert report.first_pass_validity == 1.0
    assert report.field_level_accuracy == 1.0
    assert report.unsupported_rejection_accuracy == 1.0
    assert report.failure_reasons == ()


def test_score_corpus_fails_when_first_pass_validity_too_low() -> None:
    recs = [
        _rec(
            record_id=f"a{i}",
            category="supported_paraphrase",
            expected_outcome="accepted",
            expected_spec={
                "schema_version": 2,
                "request_kind": "mood_generation",
                "duration_seconds": 180,
                "tempo_bpm": None,
                "key": None,
                "time_signature": "4/4",
                "mood": "calming",
                "instrumentation": "piano",
                "seed": None,
                "humanization": "light",
            },
        )
        for i in range(10)
    ]
    obs = [
        _obs(mood="calming", raw_first_response_valid=(i >= 3), latency_ms=50) for i in range(10)
    ]
    report = score_corpus("low-first-pass", recs, obs)
    assert not report.passed
    assert any("first_pass_validity" in r for r in report.failure_reasons)


def test_score_corpus_fails_when_field_level_accuracy_too_low() -> None:
    recs = [
        _rec(
            record_id=f"a{i}",
            category="supported_paraphrase",
            expected_outcome="accepted",
            expected_spec={
                "schema_version": 2,
                "request_kind": "mood_generation",
                "duration_seconds": 180,
                "tempo_bpm": None,
                "key": None,
                "time_signature": "4/4",
                "mood": "calming",
                "instrumentation": "piano",
                "seed": None,
                "humanization": "light",
            },
        )
        for i in range(10)
    ]
    obs = [_obs(mood="calming", duration=180 if i < 8 else 300, latency_ms=50) for i in range(10)]
    report = score_corpus("low-field-accuracy", recs, obs)
    assert not report.passed
    assert any("field_level_accuracy" in r for r in report.failure_reasons)
    assert pytest.approx(0.97) == QUALITY_FIELD_LEVEL_ACCURACY


def test_score_corpus_fails_when_unsupported_rejection_too_low() -> None:
    recs = [
        _rec(
            record_id=f"u{i}",
            category="unsupported",
            expected_outcome="rejected",
            expected_error="out_of_vocabulary",
        )
        for i in range(10)
    ]
    obs = [_obs(mood="calming" if i < 2 else None, latency_ms=50) for i in range(10)]
    report = score_corpus("low-rejection", recs, obs)
    assert not report.passed
    assert any("unsupported_rejection_accuracy" in r for r in report.failure_reasons)
    assert pytest.approx(0.95) == QUALITY_UNSUPPORTED_REJECTION


def test_score_corpus_fails_when_p95_latency_too_high() -> None:
    recs = [
        _rec(
            record_id=f"a{i}",
            category="supported_paraphrase",
            expected_outcome="accepted",
            expected_spec={
                "schema_version": 2,
                "request_kind": "mood_generation",
                "duration_seconds": 180,
                "tempo_bpm": None,
                "key": None,
                "time_signature": "4/4",
                "mood": "calming",
                "instrumentation": "piano",
                "seed": None,
                "humanization": "light",
            },
        )
        for i in range(10)
    ]
    obs = [_obs(mood="calming", latency_ms=50 + i * 1000) for i in range(10)]
    report = score_corpus("slow", recs, obs)
    assert not report.passed
    assert any("latency_p95" in r for r in report.failure_reasons)
    assert pytest.approx(5.0) == LATENCY_P95_MAX_S


def test_score_corpus_first_pass_validity_constant_matches_spec() -> None:
    assert pytest.approx(0.98) == QUALITY_FIRST_PASS_VALIDITY
