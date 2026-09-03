"""Parser benchmark runner.

Per `docs/roadmap.md` §8 (Parser robustness):

- First-response schema validity ≥ 98%
- Post-repair (or fallback) supported-prompt success ≥ 100% (implicit)
- Field-level semantic exact-match ≥ 97%
- Unsupported-request rejection accuracy ≥ 95%
- p95 latency ≤ 5 seconds (Cloud reference)
- A model that misses any quality threshold is disqualified regardless of speed or cost.

This module defines the scoring logic; the actual orchestration (prompt the
adapter, time it, compare against expected) lives in `scripts/benchmark_models.py`
(which we will add in Slice 2.9). Keeping the scorer pure makes it unit-testable
without a live Ollama host.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from statistics import quantiles

from saimc.benchmark_corpus import BenchmarkRecord
from saimc.spec import CompositionSpec

QUALITY_FIRST_PASS_VALIDITY: float = 0.98
QUALITY_FIELD_LEVEL_ACCURACY: float = 0.97
QUALITY_UNSUPPORTED_REJECTION: float = 0.95
LATENCY_P95_MAX_S: float = 5.0
CORPUS_SIZE: int = 100


@dataclass(frozen=True)
class RecordOutcome:
    """One row of the benchmark report."""

    record_id: str
    category: str
    expected_outcome: str
    actual_outcome: str  # "accepted" | "rejected" | "error"
    matched: bool
    field_exact_match: bool | None
    """True iff an accepted spec matched every expected field exactly.
    None for rejected/unmatched records."""

    parser_source: str
    attempts: int
    structured_output: bool
    latency_ms: int
    error_code: str | None = None
    """Stable error code from the adapter or fallback, if any."""


@dataclass(frozen=True)
class BenchmarkReport:
    """Aggregate metrics for a benchmark run over the full corpus."""

    client_label: str
    corpus_size: int
    first_pass_validity: float
    field_level_accuracy: float
    unsupported_rejection_accuracy: float
    latency_p50_s: float
    latency_p95_s: float
    cost_total_usd: float
    passed: bool
    failure_reasons: tuple[str, ...] = ()
    outcomes: tuple[RecordOutcome, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "client_label": self.client_label,
            "corpus_size": self.corpus_size,
            "first_pass_validity": self.first_pass_validity,
            "field_level_accuracy": self.field_level_accuracy,
            "unsupported_rejection_accuracy": self.unsupported_rejection_accuracy,
            "latency_p50_s": self.latency_p50_s,
            "latency_p95_s": self.latency_p95_s,
            "cost_total_usd": self.cost_total_usd,
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
            "outcomes": [
                {
                    "record_id": o.record_id,
                    "category": o.category,
                    "expected_outcome": o.expected_outcome,
                    "actual_outcome": o.actual_outcome,
                    "matched": o.matched,
                    "field_exact_match": o.field_exact_match,
                    "parser_source": o.parser_source,
                    "attempts": o.attempts,
                    "structured_output": o.structured_output,
                    "latency_ms": o.latency_ms,
                    "error_code": o.error_code,
                }
                for o in self.outcomes
            ],
        }


@dataclass(frozen=True)
class ParseObservation:
    """Adapter-or-fallback observation for one corpus record.

    `raw_first_response_valid` is True iff the FIRST attempt produced a
    parseable JSON object that successfully validated against
    CompositionSpec. Subsequent repair attempts are rolled into the final
    `spec` / `error`.
    """

    spec: CompositionSpec | None
    error_code: str | None
    parser_source: str
    attempts: int
    structured_output: bool
    latency_ms: int
    raw_first_response_valid: bool
    cost_usd: float


def score_record(record: BenchmarkRecord, obs: ParseObservation) -> RecordOutcome:
    """Score one record against its expected outcome.

    `actual_outcome` is "accepted" iff `obs.spec` is not None.
    `actual_outcome` is "rejected" iff `obs.spec` is None and there is no
    unexpected internal error (in which case it's "error").

    Field-level exact match compares the spec against the corpus's
    expected_spec after canonical serialization (so key order in the JSONL
    does not affect scoring).
    """
    actual_outcome = "accepted" if obs.spec is not None else "rejected"
    matched: bool
    field_exact_match: bool | None
    error_code: str | None = obs.error_code

    if record.expected_outcome == "accepted":
        matched = obs.spec is not None
        if matched and record.expected_spec is not None:
            assert obs.spec is not None  # narrow for mypy
            field_exact_match = _specs_field_equal(obs.spec, record.expected_spec)
        else:
            field_exact_match = False
    else:
        matched = obs.spec is None
        field_exact_match = None
        if error_code is None:
            error_code = "wrong_outcome"

    return RecordOutcome(
        record_id=record.id,
        category=record.category,
        expected_outcome=record.expected_outcome,
        actual_outcome=actual_outcome,
        matched=matched,
        field_exact_match=field_exact_match,
        parser_source=obs.parser_source,
        attempts=obs.attempts,
        structured_output=obs.structured_output,
        latency_ms=obs.latency_ms,
        error_code=error_code,
    )


def score_corpus(
    client_label: str,
    records: Iterable[BenchmarkRecord],
    observations: Iterable[ParseObservation],
) -> BenchmarkReport:
    """Aggregate per-record outcomes into a §8 report."""
    outcomes = [score_record(r, o) for r, o in zip(records, observations, strict=True)]

    first_pass_denominator = sum(1 for o in outcomes if o.expected_outcome == "accepted")
    first_pass_numerator = sum(
        1
        for o, obs in zip(outcomes, observations, strict=True)
        if o.expected_outcome == "accepted" and obs.raw_first_response_valid
    )
    first_pass_validity = (
        first_pass_numerator / first_pass_denominator if first_pass_denominator else 0.0
    )

    accepted_records = [o for o in outcomes if o.expected_outcome == "accepted"]
    field_exact_count = sum(1 for o in accepted_records if o.field_exact_match)
    field_level_accuracy = field_exact_count / len(accepted_records) if accepted_records else 0.0

    unsupported = [o for o in outcomes if o.expected_outcome == "rejected"]
    unsupported_correct = sum(1 for o in unsupported if o.matched)
    unsupported_rejection_accuracy = unsupported_correct / len(unsupported) if unsupported else 0.0

    latencies_s = sorted(o.latency_ms / 1000.0 for o in outcomes)
    if len(latencies_s) >= 2:
        p50, p95 = _p50_p95(latencies_s)
    elif latencies_s:
        p50 = p95 = latencies_s[0]
    else:
        p50 = p95 = 0.0

    cost_total = sum(obs.cost_usd for obs in observations)

    failures: list[str] = []
    if first_pass_validity < QUALITY_FIRST_PASS_VALIDITY:
        failures.append(
            f"first_pass_validity {first_pass_validity:.2%} < {QUALITY_FIRST_PASS_VALIDITY:.0%}"
        )
    if field_level_accuracy < QUALITY_FIELD_LEVEL_ACCURACY:
        failures.append(
            f"field_level_accuracy {field_level_accuracy:.2%} < {QUALITY_FIELD_LEVEL_ACCURACY:.0%}"
        )
    if unsupported_rejection_accuracy < QUALITY_UNSUPPORTED_REJECTION:
        failures.append(
            f"unsupported_rejection_accuracy {unsupported_rejection_accuracy:.2%} "
            f"< {QUALITY_UNSUPPORTED_REJECTION:.0%}"
        )
    if p95 > LATENCY_P95_MAX_S:
        failures.append(f"latency_p95_s {p95:.2f}s > {LATENCY_P95_MAX_S}s")

    return BenchmarkReport(
        client_label=client_label,
        corpus_size=len(outcomes),
        first_pass_validity=first_pass_validity,
        field_level_accuracy=field_level_accuracy,
        unsupported_rejection_accuracy=unsupported_rejection_accuracy,
        latency_p50_s=p50,
        latency_p95_s=p95,
        cost_total_usd=cost_total,
        passed=not failures,
        failure_reasons=tuple(failures),
        outcomes=tuple(outcomes),
    )


def _p50_p95(sorted_values: list[float]) -> tuple[float, float]:
    """Return the 50th and 95th percentiles via `statistics.quantiles`."""
    if len(sorted_values) == 1:
        return sorted_values[0], sorted_values[0]
    qs = quantiles(sorted_values, n=100, method="inclusive")
    p50 = qs[49]
    p95 = qs[94]
    return float(p50), float(p95)


def _specs_field_equal(actual: CompositionSpec, expected_dict: dict[str, object]) -> bool:
    """Compare a CompositionSpec to a corpus `expected_spec` field-by-field.

    Uses `mode='json'` so enum values serialize as their string values
    (matching the corpus's flat-string format) and orders are ignored.
    """
    actual_dict = actual.model_dump(mode="json")
    return _dict_field_equal(actual_dict, expected_dict)


def _dict_field_equal(actual: object, expected: object) -> bool:
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            return False
        if set(actual.keys()) != set(expected.keys()):
            return False
        return all(_dict_field_equal(actual[k], expected[k]) for k in expected)
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            return False
        return all(_dict_field_equal(a, e) for a, e in zip(actual, expected, strict=True))
    return actual == expected


__all__ = [
    "CORPUS_SIZE",
    "LATENCY_P95_MAX_S",
    "QUALITY_FIELD_LEVEL_ACCURACY",
    "QUALITY_FIRST_PASS_VALIDITY",
    "QUALITY_UNSUPPORTED_REJECTION",
    "BenchmarkReport",
    "ParseObservation",
    "RecordOutcome",
    "score_corpus",
    "score_record",
]
