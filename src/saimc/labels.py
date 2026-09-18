"""The agreement measure behind §10 #3's second labeling pass.

`docs/roadmap.md` §10 #3 selects the parser model from a benchmark run, and
`docs/model-fine-tuning.md` forbids scoring any candidate before the corpus
has been labeled twice, *independently*, with at least 90% pre-adjudication
field-level agreement between the passes. A corpus labeled once measures one
person's reading of the prompts; the agreement rate is what says the labels
are a property of the prompts rather than a property of the labeler.

This module is the comparison and its report. The second pass itself is a
person's, run through `saimc-label` (`saimc/labeling_cli.py`); the file it
writes is validated by the corpus's own loader, because it is a corpus file
in the corpus's own schema.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from saimc.benchmark_corpus import BenchmarkRecord

MIN_FIELD_AGREEMENT: float = 0.90
"""§10 #3's pre-adjudication bar: at least 90% of comparable label fields.

Named here rather than in `benchmark.py` because it is a property of the
*labels*, not of a run: `benchmark.py`'s thresholds judge a model against a
corpus, and this one judges a corpus against itself.
"""

FIXED_BY_SCHEMA: frozenset[str] = frozenset({"schema_version", "request_kind"})
"""Spec keys fixed by the schema or the writer, and therefore not compared.

`request_kind`'s field admits exactly one value in Phase 1 (`famous_piece` is
reserved) and `schema_version`'s is this build's by definition; `saimc-label`
writes both rather than asking for them. Comparing them would be comparing two
values that cannot differ — this repo's name for a check that proves nothing —
while padding the denominator, which is the failure this set exists to prevent:
ten spec keys of which the corpus varies five means a uniform per-key rate lets
a reviewer disagree on a large share of the fields that *are* decisions and
still clear 90%. Both keys are still asserted one-valued by
`tests/unit/test_labels.py`, so widening either vocabulary fails loudly instead
of quietly leaving the key uncompared.

**The rule for this set is "the schema or the writer fixes it", never "the
corpus happens to hold one value".** `time_signature`, `humanization` and `key`
are one value across the whole corpus today, and they stay compared: what is
constant there is a property of these prompts, which is precisely what a second
reading is meant to test. Excluding them would be excluding the measurement.
The `by_field` breakdown is what keeps their weight visible.

`schema_version` is one value across two files that have passed the drift
check, and it is also refused outright when they have not: the field admits
three values, so a difference is real — it says the files are of different
corpus versions, which `docs/parser-benchmark.md:34` says invalidates a
comparison — and scoring it as a disagreement would report a version bump as a
reader's mistake. `compare_labelings` therefore refuses it the way it refuses
prompt drift, and excludes it here for the case where there is nothing to
report.
"""

_OUTCOME = "expected_outcome"
_ERROR = "expected_error"
_SPEC = "expected_spec"


@dataclass(frozen=True)
class FieldDisagreement:
    """One label field the two passes read differently.

    Carries the prompt because adjudicating a disagreement means reading the
    request again, and a reader looking at a report should not have to join
    back to the corpus to find out what was asked.
    """

    record_id: str
    prompt: str
    field: str
    primary: object
    review: object


@dataclass(frozen=True)
class FieldAgreement:
    """One label field's share of the two passes' agreement.

    Reported per field because the uniform rate is dominated by the fields that
    cannot move: a key with one legal value contributes a matched field per
    record whatever the reviewer typed, so a report carrying only the total
    cannot say whether a miss is a disagreement about a decision or a rounding
    of the padding. This is the number that says which field to read the corpus
    about, and it is the reason `FIXED_BY_SCHEMA` exists.
    """

    field: str
    compared: int
    matched: int

    @property
    def rate(self) -> float:
        return self.matched / self.compared if self.compared else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "field": self.field,
            "compared": self.compared,
            "matched": self.matched,
            "rate": self.rate,
        }


@dataclass(frozen=True)
class AgreementReport:
    """How two labelings of the same corpus compare, field by field."""

    record_count: int
    field_count: int
    matched_fields: int
    rate: float
    passed: bool
    failure_reasons: tuple[str, ...] = ()
    disagreements: tuple[FieldDisagreement, ...] = ()
    by_field: tuple[FieldAgreement, ...] = ()

    @property
    def disagreeing_records(self) -> int:
        """How many records contributed at least one disagreement.

        Not `len(disagreements)`: one record can disagree on several fields,
        and the two numbers answer different questions — the rate is over
        fields, the work is over records.
        """
        return len({d.record_id for d in self.disagreements})

    def to_dict(self) -> dict[str, object]:
        return {
            "record_count": self.record_count,
            "field_count": self.field_count,
            "matched_fields": self.matched_fields,
            "rate": self.rate,
            "disagreeing_records": self.disagreeing_records,
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
            "by_field": [entry.to_dict() for entry in self.by_field],
            "disagreements": [
                {
                    "record_id": d.record_id,
                    "prompt": d.prompt,
                    "field": d.field,
                    "primary": d.primary,
                    "review": d.review,
                }
                for d in self.disagreements
            ],
        }


def compare_labelings(
    primary: Iterable[BenchmarkRecord],
    review: Iterable[BenchmarkRecord],
    *,
    minimum: float = MIN_FIELD_AGREEMENT,
) -> AgreementReport:
    """Score two labelings of one corpus against each other.

    Raises `ValueError` when the two files are not labelings *of the same
    corpus* — a mismatched id set, a duplicated id, a record whose `prompt`
    differs, or specs written at different spec versions. Those are drift
    between corpus versions rather than disagreements about a label
    (`docs/parser-benchmark.md`: changing a prompt creates a new corpus version
    and invalidates comparisons), and folding them into the rate would let a
    reworded prompt or a version bump read as a labeling mistake.
    """
    primary_by_id = _by_id(primary, "primary")
    review_by_id = _by_id(review, "review")

    missing = sorted(set(primary_by_id) - set(review_by_id))
    extra = sorted(set(review_by_id) - set(primary_by_id))
    if missing or extra:
        raise ValueError(
            "the two labelings cover different records: "
            f"{len(missing)} labeled but not reviewed ({_preview(missing)}), "
            f"{len(extra)} reviewed but not labeled ({_preview(extra)})"
        )

    primary_versions = _spec_versions(primary_by_id.values())
    review_versions = _spec_versions(review_by_id.values())
    versions = primary_versions | review_versions
    if len(versions) > 1:
        raise ValueError(
            "the labelings name more than one spec version "
            f"({'/'.join(sorted(map(repr, versions)))}): a corpus is one spec "
            "version, and agreement across two of them measures the version"
        )

    totals: Counter[str] = Counter()
    matched: Counter[str] = Counter()
    disagreements: list[FieldDisagreement] = []

    for record_id in sorted(primary_by_id):
        labeled = primary_by_id[record_id]
        reviewed = review_by_id[record_id]
        if labeled.prompt != reviewed.prompt:
            raise ValueError(
                f"corpus drift at {record_id}: the two files carry different prompts "
                f"({labeled.prompt!r} vs {reviewed.prompt!r})"
            )
        for field, (first, second) in _comparable_labels(labeled, reviewed).items():
            totals[field] += 1
            if first == second:
                matched[field] += 1
            else:
                disagreements.append(
                    FieldDisagreement(
                        record_id=record_id,
                        prompt=labeled.prompt,
                        field=field,
                        primary=first,
                        review=second,
                    )
                )

    # Derived from the per-field counters rather than counted alongside them, so
    # the total and the breakdown are one quantity read two ways and cannot
    # disagree — the reason `Draft` stores no derived hash (C3), one level down.
    field_count = sum(totals.values())
    matched_fields = sum(matched.values())
    rate = matched_fields / field_count if field_count else 0.0
    failures: list[str] = []
    if not field_count:
        failures.append("no comparable label fields: both labelings are empty")
    elif rate < minimum:
        failures.append(
            f"field-level agreement {rate:.2%} < {minimum:.0%} "
            f"({len(disagreements)} field(s) across "
            f"{len({d.record_id for d in disagreements})} record(s) disagree)"
        )

    return AgreementReport(
        record_count=len(primary_by_id),
        field_count=field_count,
        matched_fields=matched_fields,
        rate=rate,
        passed=not failures,
        failure_reasons=tuple(failures),
        disagreements=tuple(disagreements),
        by_field=tuple(
            FieldAgreement(field=field, compared=totals[field], matched=matched[field])
            for field in _reading_order(totals)
        ),
    )


def _reading_order(fields: Iterable[str]) -> list[str]:
    """The order a person reads the breakdown in.

    The record's own labels first — the outcome before the error, since the
    error is only meaningful for a rejection — then the spec's keys, which are
    sorted so the order is a property of the corpus rather than of the dict a
    record happened to be parsed into.
    """
    named = [field for field in (_OUTCOME, _ERROR, _SPEC) if field in fields]
    return named + sorted(set(fields) - set(named))


def _spec_versions(records: Iterable[BenchmarkRecord]) -> frozenset[object]:
    """The distinct spec versions these records were written at, raw.

    Raw rather than coerced to `int`, because the value comes off a file: a
    review pass that wrote `"3"` must be *reported* as a version this comparison
    cannot place, and `str(3) == str("3")` is exactly the coercion that would
    hide it. A record with no spec contributes nothing — there is no version
    there to compare — so a file whose every accepted record became a rejection
    is not thereby at odds with the corpus.
    """
    return frozenset(
        (record.expected_spec or {}).get("schema_version", "unset")
        for record in records
        if record.expected_spec is not None
    )


def _comparable_labels(
    labeled: BenchmarkRecord, reviewed: BenchmarkRecord
) -> dict[str, tuple[object, object]]:
    """`field -> (primary, review)` for every label field this pair shares.

    `expected_outcome` and `expected_error` are always comparable. The spec
    is comparable *key by key* when both passes produced one — which is what
    "field-level" means in §8's 97% bar, where a mislabelled tempo and a
    mislabelled mood are the same size of disagreement — and as a single
    `expected_spec` field otherwise, because the ten keys the missing side
    never wrote are one absence rather than ten disagreements.

    Keys in `FIXED_BY_SCHEMA` are skipped in the key-by-key branch, because
    nothing in the comparison could move them. They are *not* scrubbed from the
    whole-document branch: that branch compares two documents rather than two
    fields, and the disagreement there is total — one side has no spec — so
    there is no rate for a constant key to dilute.

    A key absent on one side compares as `None`, which is also how the
    corpus spells an unset optional field (`tempo_bpm`, `key`, `seed`), so
    an omitted optional reads as agreement with an explicit null.
    """
    labels: dict[str, tuple[object, object]] = {
        _OUTCOME: (labeled.expected_outcome, reviewed.expected_outcome),
        _ERROR: (labeled.expected_error, reviewed.expected_error),
    }
    first = labeled.expected_spec
    second = reviewed.expected_spec
    if first is not None and second is not None:
        for key in sorted(set(first) | set(second)):
            if key in FIXED_BY_SCHEMA:
                continue
            labels[key] = (first.get(key), second.get(key))
    else:
        labels[_SPEC] = (first, second)
    return labels


def _by_id(
    records: Iterable[BenchmarkRecord], label: str
) -> dict[str, BenchmarkRecord]:
    """Index by id, refusing a duplicate and naming which file repeats it.

    A resumed labeling pass appends, so an append that ran twice is the
    corruption this can actually see. Indexing would keep the last one
    silently and score a file that is not the one on disk.
    """
    indexed: dict[str, BenchmarkRecord] = {}
    for record in records:
        if record.id in indexed:
            raise ValueError(f"the {label} labeling lists {record.id} more than once")
        indexed[record.id] = record
    return indexed


def _preview(ids: list[str], limit: int = 5) -> str:
    """Name a few ids so a mismatch is diagnosable without a full dump."""
    if len(ids) <= limit:
        return ", ".join(ids) or "none"
    return ", ".join(ids[:limit]) + f", … (+{len(ids) - limit} more)"


__all__ = [
    "FIXED_BY_SCHEMA",
    "MIN_FIELD_AGREEMENT",
    "AgreementReport",
    "FieldAgreement",
    "FieldDisagreement",
    "compare_labelings",
]
