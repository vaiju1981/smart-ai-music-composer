"""Parser benchmark runner.

Per `docs/roadmap.md` §8 (Parser robustness):

- First-response schema validity ≥ 98% (`QUALITY_FIRST_PASS_VALIDITY`)
- Post-repair (or fallback) supported-prompt success = 100%
  (`QUALITY_SUPPORTED_RESOLUTION`)
- Field-level semantic exact-match ≥ 97% (`QUALITY_FIELD_LEVEL_ACCURACY`)
- Unsupported-request rejection accuracy ≥ 95% (`QUALITY_UNSUPPORTED_REJECTION`)
- p95 latency ≤ 5 seconds (Cloud reference) (`LATENCY_P95_MAX_S`)
- A model that misses any quality threshold is disqualified regardless of speed or cost.

All four quality bars are scored here. The orchestration that feeds them —
prompting the adapter, timing it, comparing against expected — lives in
`saimc/benchmark_cli.py`. Keeping the scorer pure makes it unit-testable without
a live Ollama host, which is why `score_corpus` takes observations rather than a
client.

Two fields here distinguish "measured, and the measurement is zero" from "not
measured at all", and both are `None` in the second case rather than `0.0`:
`ParseObservation.raw_first_response_valid` (no model output was read) and
`ParseObservation.cost_usd` (nothing meters or prices a call in this codebase).
The distinction is the repo's existing convention for this (`PieceQuality.as_dict`)
and it is not cosmetic: a `0.0` that means "free" and a `0.0` that means
"unpriced" read the same to every consumer, and only one of them is evidence.
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
QUALITY_SUPPORTED_RESOLUTION: float = 1.0
"""§8: "after at most two repairs or deterministic fallback, *all* supported
benchmark prompts parse successfully" — a 100% bar, and the reason it is 1.0
rather than "very high"."""

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
    first_pass_validity: float | None
    """`None` when no observation was judgeable — see
    `ParseObservation.raw_first_response_valid`. Not `0.0`: a run in which no
    model answered has not *failed* §8's first bar, it has left it unmeasured,
    and `score_corpus` fails the report with a reason saying so."""

    field_level_accuracy: float
    unsupported_rejection_accuracy: float
    resolution_rate: float
    """Share of supported prompts that ended with a spec at all, by any route.

    §8's "all supported benchmark prompts parse successfully". Distinct from
    both of its neighbours: `field_level_accuracy` asks whether the resulting
    spec was *right*, and `first_pass_validity` asks about attempt one. This one
    asks only whether the prompt resolved, which is the bar the repair loop and
    the fallback exist to meet."""

    latency_p50_s: float
    latency_p95_s: float
    cost_total_usd: float | None
    """The sum over observations, or `None` if any of them was unmetered. A
    partly-priced run's total is unknown, and reporting the priced part as if it
    were the whole is the defect this type is written to prevent."""

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
            "resolution_rate": self.resolution_rate,
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

    **It is `bool | None`, and the `None` is a third state rather than a
    missing value.** `None` means no model output was read at all — the host
    was unreachable, none was configured, or the response body was not a
    model's response — which is `saimc.parser.ParseResult.first_attempt`'s own
    third state seen from here, and the same state a fallback-only run is in for
    every record because it asks no model. It is not `False`: §8's bar is about
    a model's first *response*, and a run with no response cannot fail it. The
    flag was a hardcoded `True` on the fallback-only path until this was fixed,
    which is why the one runnable benchmark reported a validity of `1.0` for a
    run that called no model.
    """

    spec: CompositionSpec | None
    error_code: str | None
    parser_source: str
    attempts: int
    structured_output: bool
    latency_ms: int
    raw_first_response_valid: bool | None
    cost_usd: float | None
    """`None` when nothing measured a price, which is every call today: no
    code in this repo meters or prices a model call, and the configured dev host
    is a local proxy to a cloud provider whose inference is billed to an account
    the app cannot read. A fallback-only run's `0.0` is a genuine zero — it makes
    no call — and that is the distinction the type carries."""


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
    # Materialized once, because the report reads the observations three times
    # (paired with the outcomes, again for the first-pass numerator, and once
    # for the cost total) while the signature promises only an `Iterable`. A
    # generator argument satisfies the annotation and then dies on the second
    # walk with `zip() argument 2 is shorter than argument 1` from the
    # `strict=True` below — a caller-facing lie in a signature, fixed here
    # rather than by narrowing the annotation onto every future caller.
    observations = list(observations)
    outcomes = [score_record(r, o) for r, o in zip(records, observations, strict=True)]

    supported = [
        (outcome, obs)
        for outcome, obs in zip(outcomes, observations, strict=True)
        if outcome.expected_outcome == "accepted"
    ]
    # §8's first bar is scored over the supported prompts that produced
    # something to judge. The reading is deliberate: a model that returns a
    # valid spec for an *unsupported* request has failed rather than passed, so
    # those records belong to the rejection bar and not this one — and a record
    # whose first attempt read no model output belongs to neither, because a
    # network failure is not a first response.
    judged = [obs for _, obs in supported if obs.raw_first_response_valid is not None]
    first_pass_validity: float | None = (
        sum(1 for obs in judged if obs.raw_first_response_valid) / len(judged) if judged else None
    )

    # §8: "all supported benchmark prompts parse successfully" by any route.
    # A supported prompt that produced no spec at all is the failure this bar
    # names, and it is the only one of the four that the repair loop and the
    # fallback exist to hold at 100%.
    resolution_rate = (
        sum(1 for outcome, _ in supported if outcome.matched) / len(supported)
        if supported
        else 0.0
    )

    field_exact_count = sum(1 for outcome, _ in supported if outcome.field_exact_match)
    field_level_accuracy = field_exact_count / len(supported) if supported else 0.0

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

    # A total is `None` unless every observation carried a price: one unmetered
    # call makes the run's cost unknown, and summing the rest would report the
    # priced part as though it were the whole.
    priced = [obs.cost_usd for obs in observations if obs.cost_usd is not None]
    cost_total: float | None = sum(priced) if len(priced) == len(observations) else None

    failures: list[str] = []
    if first_pass_validity is None:
        # A report that passed while its flagship bar went unmeasured would be
        # the hardcoded `1.0` this change exists to remove, one layer up.
        failures.append(
            "first_pass_validity not measured: no supported observation recorded a first attempt"
        )
    elif first_pass_validity < QUALITY_FIRST_PASS_VALIDITY:
        failures.append(
            f"first_pass_validity {first_pass_validity:.2%} < {QUALITY_FIRST_PASS_VALIDITY:.0%}"
        )
    if resolution_rate < QUALITY_SUPPORTED_RESOLUTION:
        failures.append(
            f"resolution_rate {resolution_rate:.2%} < {QUALITY_SUPPORTED_RESOLUTION:.0%}"
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
        resolution_rate=resolution_rate,
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
    "QUALITY_SUPPORTED_RESOLUTION",
    "QUALITY_UNSUPPORTED_REJECTION",
    "BenchmarkReport",
    "ParseObservation",
    "RecordOutcome",
    "score_corpus",
    "score_record",
]
