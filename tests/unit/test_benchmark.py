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
    raw_first_response_valid: bool | None = True,
    cost_usd: float | None = 0.0,
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
            cost_usd=cost_usd,
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
        cost_usd=cost_usd,
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


class TestTheCorpusHash:
    """`corpus_sha256`, and the property that makes it worth recording."""

    def test_it_is_the_sha256_of_the_file_bytes(self, tmp_path: Path) -> None:
        import hashlib

        from saimc.benchmark_corpus import corpus_sha256

        path = tmp_path / "c.jsonl"
        path.write_bytes(b'{"id": "a"}\n')
        assert corpus_sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()

    def test_whitespace_drift_changes_it(self, tmp_path: Path) -> None:
        """The claim that separates a byte digest from a canonical one.

        Both files below parse to the same record — so an implementation that
        re-serialized the parsed records would call them the same corpus — and
        the hash has to say they are not, because the question it is recorded
        to answer is "is this the file the stored score was taken against?".
        """
        from saimc.benchmark_corpus import corpus_sha256

        one = tmp_path / "one.jsonl"
        two = tmp_path / "two.jsonl"
        one.write_text('{"id": "a", "n": 1}\n', encoding="utf-8")
        two.write_text('{"id": "a",   "n": 1}\n', encoding="utf-8")
        assert corpus_sha256(one) != corpus_sha256(two)

    def test_the_canonical_corpus_has_a_stable_hash(self) -> None:
        """A ratchet on the file itself, so a rewrite of it is a declared edit.

        The corpus has been rewritten twice already with no version or hash
        trail, which left the runs either side of those rewrites incomparable
        and nothing saying so. This pin is what makes the next rewrite say so.
        """
        from saimc.benchmark_corpus import DEFAULT_CORPUS_PATH, corpus_sha256

        assert (
            corpus_sha256(DEFAULT_CORPUS_PATH)
            == "53c05b2a3c8163241c791491be93b21bfcef56cedd4fc4d98df10dc9708e2010"
        )


def test_score_record_accepted_matched_and_field_exact() -> None:
    rec = _rec(
        record_id="r1",
        category="supported_paraphrase",
        expected_outcome="accepted",
        expected_spec={
            "schema_version": 3,
            "request_kind": "mood_generation",
            "duration_seconds": 180,
            "tempo_bpm": None,
            "key": None,
            "time_signature": "4/4",
            "mood": "calming",
            "instrumentation": [{"role": "melody", "instrument": "piano"}, {"role": "harmony", "instrument": "pizzicato_strings"}, {"role": "bass", "instrument": "cello"}],
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
            "schema_version": 3,
            "request_kind": "mood_generation",
            "duration_seconds": 300,
            "tempo_bpm": None,
            "key": None,
            "time_signature": "4/4",
            "mood": "calming",
            "instrumentation": [{"role": "melody", "instrument": "piano"}, {"role": "harmony", "instrument": "pizzicato_strings"}, {"role": "bass", "instrument": "cello"}],
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
                "schema_version": 3,
                "request_kind": "mood_generation",
                "duration_seconds": 180,
                "tempo_bpm": None,
                "key": None,
                "time_signature": "4/4",
                "mood": "calming",
                "instrumentation": [{"role": "melody", "instrument": "piano"}, {"role": "harmony", "instrument": "pizzicato_strings"}, {"role": "bass", "instrument": "cello"}],
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
                "schema_version": 3,
                "request_kind": "mood_generation",
                "duration_seconds": 180,
                "tempo_bpm": None,
                "key": None,
                "time_signature": "4/4",
                "mood": "calming",
                "instrumentation": [{"role": "melody", "instrument": "piano"}, {"role": "harmony", "instrument": "pizzicato_strings"}, {"role": "bass", "instrument": "cello"}],
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
                "schema_version": 3,
                "request_kind": "mood_generation",
                "duration_seconds": 180,
                "tempo_bpm": None,
                "key": None,
                "time_signature": "4/4",
                "mood": "calming",
                "instrumentation": [{"role": "melody", "instrument": "piano"}, {"role": "harmony", "instrument": "pizzicato_strings"}, {"role": "bass", "instrument": "cello"}],
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
                "schema_version": 3,
                "request_kind": "mood_generation",
                "duration_seconds": 180,
                "tempo_bpm": None,
                "key": None,
                "time_signature": "4/4",
                "mood": "calming",
                "instrumentation": [{"role": "melody", "instrument": "piano"}, {"role": "harmony", "instrument": "pizzicato_strings"}, {"role": "bass", "instrument": "cello"}],
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


# The spec every supported record in the cases below expects, in the corpus's
# own flat-string form. Written once here because the cases below each build a
# corpus and would otherwise restate ten lines of it apiece.
_EXACT_SPEC: dict[str, object] = {
    "schema_version": 3,
    "request_kind": "mood_generation",
    "duration_seconds": 180,
    "tempo_bpm": None,
    "key": None,
    "time_signature": "4/4",
    "mood": "calming",
    "instrumentation": [
        {"role": "melody", "instrument": "piano"},
        {"role": "harmony", "instrument": "pizzicato_strings"},
        {"role": "bass", "instrument": "cello"},
    ],
    "seed": None,
    "humanization": "light",
}


def _supported(n: int) -> list[BenchmarkRecord]:
    return [
        _rec(
            record_id=f"a{i}",
            category="supported_paraphrase",
            expected_outcome="accepted",
            expected_spec=_EXACT_SPEC,
        )
        for i in range(n)
    ]


class TestFirstPassValidityIsMeasuredOrItIsNotScored:
    """§8's flagship bar, and the three states the number can be in."""

    def test_an_unjudged_cell_is_not_in_the_denominator(self) -> None:
        """`None` is excluded, so a run that partly reached the host is still scorable.

        Eight answered, two never reached it. Reading `None` as a miss would
        report 80% for a model whose every actual response was valid, and
        reading it as a pass would report 100% for a model that answered
        eight times. It is neither: it is out of the denominator.
        """
        recs = _supported(10)
        obs = [
            _obs(mood="calming", raw_first_response_valid=None if i < 2 else True) for i in range(10)
        ]
        report = score_corpus("partly-reached", recs, obs)
        assert report.first_pass_validity == 1.0
        assert report.resolution_rate == 1.0

    def test_a_run_with_nothing_to_judge_refuses_rather_than_scoring_zero(self) -> None:
        """The fallback-only shape: every supported prompt resolved, no model asked."""
        recs = _supported(10)
        obs = [_obs(mood="calming", raw_first_response_valid=None) for _ in range(10)]
        report = score_corpus("fallback", recs, obs)
        assert report.first_pass_validity is None
        assert report.resolution_rate == 1.0
        assert not report.passed
        assert any("not measured" in r for r in report.failure_reasons)

    def test_a_judged_miss_still_scores(self) -> None:
        """The premise of the two above: the bar is real when there is something to read."""
        recs = _supported(10)
        obs = [_obs(mood="calming", raw_first_response_valid=i >= 3) for i in range(10)]
        report = score_corpus("judged", recs, obs)
        assert report.first_pass_validity == pytest.approx(0.7)
        assert any("first_pass_validity 70.00%" in r for r in report.failure_reasons)


class TestResolutionRate:
    """§8's 100% bar over supported prompts, by any route."""

    def test_a_supported_prompt_that_resolves_to_nothing_fails_the_bar(self) -> None:
        recs = _supported(10)
        obs = [
            _obs(
                mood=None,
                error_code="schema_invalid",
                parser_source="fallback",
                attempts=3,
                raw_first_response_valid=False,
            )
        ] + [_obs(mood="calming") for _ in range(9)]
        report = score_corpus("nine-of-ten", recs, obs)
        assert report.resolution_rate == pytest.approx(0.9)
        assert not report.passed
        assert any("resolution_rate 90.00%" in r for r in report.failure_reasons)

    def test_the_bar_is_scored_below_one_hundred_percent(self) -> None:
        """The constant, pinned: a bar that slipped to 0.99 would pass this case."""
        from saimc.benchmark import QUALITY_SUPPORTED_RESOLUTION

        assert QUALITY_SUPPORTED_RESOLUTION == 1.0

    def test_a_prompt_that_resolves_wrongly_still_resolved(self) -> None:
        """The two bars this one sits between, told apart on one corpus.

        Every record here produces a spec, so every supported prompt resolved
        (100%) — and every spec is the *wrong* one, so field-level accuracy is
        zero. Reading this bar off `field_exact_match` instead of `matched`
        would report 0% for a parser that resolved all ten prompts, which is
        the whole reason it is a separate field.
        """
        recs = _supported(10)
        obs = [_obs(mood="sleep", duration=300) for _ in range(10)]
        report = score_corpus("wrong-but-resolved", recs, obs)
        assert all(o.matched for o in report.outcomes)
        assert report.resolution_rate == 1.0
        assert report.field_level_accuracy == 0.0


class TestTheScorerHonoursItsSignature:
    def test_a_generator_of_observations_is_read_three_times_over(self) -> None:
        """`Iterable` in the annotation has to mean it.

        The report walks the observations for the outcomes, again for the
        first-pass numerator and once for the cost total, so an un-materialized
        generator raises `ValueError: zip() argument 2 is shorter than argument
        1` from the `strict=True` on the second walk. Only a generator argument
        reaches that, which is why the case passes one.
        """
        recs = _supported(3)
        obs = [_obs(mood="calming", cost_usd=0.25) for _ in range(3)]
        report = score_corpus("generator", recs, (o for o in obs))
        assert report.corpus_size == 3
        assert report.cost_total_usd == pytest.approx(0.75)


class TestCostIsEitherMeasuredOrItIsNot:
    def test_a_total_over_priced_calls_is_their_sum(self) -> None:
        recs = _supported(4)
        obs = [_obs(mood="calming", cost_usd=0.5) for _ in range(4)]
        report = score_corpus("priced", recs, obs)
        assert report.cost_total_usd == pytest.approx(2.0)

    def test_one_unpriced_call_makes_the_total_unknown(self) -> None:
        """Not the sum of the rest: that reports a part as though it were the whole."""
        recs = _supported(4)
        obs = [_obs(mood="calming", cost_usd=0.5) for _ in range(3)] + [
            _obs(mood="calming", cost_usd=None)
        ]
        report = score_corpus("partly-priced", recs, obs)
        assert report.cost_total_usd is None

    def test_a_genuine_zero_is_a_measured_zero(self) -> None:
        """The offline path's `0.0` and an unpriced call are different statements."""
        recs = _supported(3)
        report = score_corpus("offline", recs, [_obs(mood="calming", cost_usd=0.0) for _ in range(3)])
        assert report.cost_total_usd == 0.0


def test_the_report_dict_carries_every_bar_it_scores() -> None:
    """A bar that exists on the report and is absent from `to_dict` is unreadable.

    The CLI prints the dict, and `MODELS.md` is written from what it prints,
    so a bar missing here is a bar no operator can record.
    """
    report = score_corpus("keys", _supported(2), [_obs(mood="calming") for _ in range(2)])
    payload = report.to_dict()
    for key in (
        "first_pass_validity",
        "resolution_rate",
        "field_level_accuracy",
        "unsupported_rejection_accuracy",
        "latency_p95_s",
        "cost_total_usd",
    ):
        assert key in payload, f"{key} is scored but not reported"
