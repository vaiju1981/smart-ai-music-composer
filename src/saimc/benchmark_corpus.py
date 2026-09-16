"""Loader for the Phase 1 parser benchmark corpus.

The corpus lives at `tests/fixtures/parser_benchmark.jsonl`. This loader is
the only code path that reads it; tests and the benchmark CLI both go
through here so format drift is caught in one place.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

DEFAULT_CORPUS_PATH = (
    Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "parser_benchmark.jsonl"
)


class BenchmarkRecord(BaseModel):
    """A single entry in `tests/fixtures/parser_benchmark.jsonl`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    category: str
    prompt: str
    expected_outcome: str  # "accepted" | "rejected"
    expected_spec: dict[str, object] | None
    expected_error: str | None
    label_rationale: str


@dataclass(frozen=True)
class CorpusStats:
    """Top-level counts returned alongside the loaded records."""

    total: int
    accepted_expected: int
    rejected_expected: int
    by_category: dict[str, int]


def load_corpus(
    path: Path | str = DEFAULT_CORPUS_PATH,
) -> tuple[list[BenchmarkRecord], CorpusStats]:
    """Load and validate every record in the corpus JSONL.

    Raises `ValueError` (with the first offending line number) on any
    malformed record, so CI catches corpus drift immediately.
    """
    corpus_path = Path(path)
    records: list[BenchmarkRecord] = []
    by_category: dict[str, int] = {}
    accepted = 0
    rejected = 0

    with corpus_path.open("r", encoding="utf-8") as fh:
        for line_no, raw in enumerate(fh, start=1):
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
                record = BenchmarkRecord.model_validate(obj)
            except (json.JSONDecodeError, ValidationError) as exc:
                raise ValueError(f"Invalid record at {corpus_path}:{line_no}: {exc}") from exc
            records.append(record)
            by_category[record.category] = by_category.get(record.category, 0) + 1
            if record.expected_outcome == "accepted":
                accepted += 1
            elif record.expected_outcome == "rejected":
                rejected += 1
            else:
                raise ValueError(
                    f"Unknown expected_outcome at {corpus_path}:{line_no}: "
                    f"{record.expected_outcome!r}"
                )

    return records, CorpusStats(
        total=len(records),
        accepted_expected=accepted,
        rejected_expected=rejected,
        by_category=by_category,
    )


def iter_records(records: Iterable[BenchmarkRecord]) -> Iterable[BenchmarkRecord]:
    """Passthrough iterator for symmetry with potential future generators."""
    yield from records


__all__ = [
    "DEFAULT_CORPUS_PATH",
    "BenchmarkRecord",
    "CorpusStats",
    "iter_records",
    "load_corpus",
]
