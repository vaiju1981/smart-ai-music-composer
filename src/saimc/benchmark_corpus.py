"""Loader for the Phase 1 parser benchmark corpus.

The corpus lives at `tests/fixtures/parser_benchmark.jsonl`. This loader is
the only code path that reads it; tests and the benchmark CLI both go
through here so format drift is caught in one place.

The corpus has a second file — the independent review pass required by
`docs/model-fine-tuning.md` before any candidate model is scored — and it
is in this same schema, so this loader validates both. See
`DEFAULT_REVIEW_PATH` and `saimc.labels`.

`corpus_sha256` is the one reader here that does *not* parse: see its own
docstring for why the digest is taken over the file's bytes.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, get_args

from pydantic import BaseModel, ConfigDict, ValidationError

_FIXTURES_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures"

DEFAULT_CORPUS_PATH = _FIXTURES_DIR / "parser_benchmark.jsonl"

DEFAULT_REVIEW_PATH = _FIXTURES_DIR / "parser_benchmark_review.jsonl"
"""The second labeling pass, in the primary corpus's own record schema.

Written by `saimc-label` (`saimc/labeling_cli.py`) and committed once the
two passes agree at or above `saimc.labels.MIN_FIELD_AGREEMENT`. It is a
separate file rather than a second set of columns because the two passes
must be *independent*: a reviewer who can see the first label is agreeing
with the screen, not with the corpus.
"""

ExpectedErrorCode = Literal["empty_prompt", "out_of_vocabulary", "schema_invalid"]
"""The closed vocabulary a rejected record's `expected_error` is drawn from.

These are the codes `saimc/llm/fallback.py` names for a request it cannot
honour, and they are what every rejected record in the corpus carries. The
vocabulary is closed here rather than left a free-form string because it is
a *label*: a typo in a review pass would otherwise enter the corpus as a code
no parser can ever emit, and the agreement measure would score it as a
disagreement about the request. The transport codes in `saimc/llm/base.py`
(`llm_unreachable`, `llm_not_configured`) are deliberately not members — they
describe a run rather than a request, and no record may be labelled with one.
"""

REJECTION_ERROR_CODES: frozenset[ExpectedErrorCode] = frozenset(get_args(ExpectedErrorCode))
"""The same vocabulary as a set, for the labeling tool's menu.

Derived from the `Literal` rather than restated, so the menu the tool offers
and the vocabulary the loader enforces are one list: a `Literal` is one
coverage hole at most, where a second hand-kept copy of the same three
strings would be a hole per entry — the rule D2's harmony levels taught. It
is typed over the `Literal` rather than `str` so the menu's answer can go
straight into a record's `expected_error` without a cast: the tool is the
writer, and a writer that has to assert what it is writing is a writer whose
type says less than its code knows.
"""


class BenchmarkRecord(BaseModel):
    """A single entry in `tests/fixtures/parser_benchmark.jsonl`."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    category: str
    prompt: str
    expected_outcome: str  # "accepted" | "rejected"
    expected_spec: dict[str, object] | None
    expected_error: ExpectedErrorCode | None
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


def corpus_sha256(path: Path | str = DEFAULT_CORPUS_PATH) -> str:
    """The corpus file's SHA-256, for the hash every benchmark run records.

    `docs/parser-benchmark.md` requires the corpus hash alongside a run so a
    recorded score can be invalidated when the corpus moves, and nothing
    computed one: the corpus has already been rewritten twice (for schema v2
    and v3) with no version or hash trail, so the two runs either side of those
    rewrites are not comparable and nothing says so.

    **Hashed as bytes, not as a canonical re-serialization of the parsed
    records.** A canonical form would be stable across a reformat, which sounds
    better until the hash has to answer the question it is recorded for — "is
    this the same corpus the stored score was taken against?" — and a canonical
    digest is not the file on disk, so it cannot tell an edited file from a
    re-serialized one. Bytes can, and they answer it the conservative way:
    white-space drift invalidates a comparison rather than silently joining it.
    """
    with Path(path).open("rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


def iter_records(records: Iterable[BenchmarkRecord]) -> Iterable[BenchmarkRecord]:
    """Passthrough iterator for symmetry with potential future generators."""
    yield from records


__all__ = [
    "DEFAULT_CORPUS_PATH",
    "DEFAULT_REVIEW_PATH",
    "REJECTION_ERROR_CODES",
    "BenchmarkRecord",
    "CorpusStats",
    "ExpectedErrorCode",
    "corpus_sha256",
    "iter_records",
    "load_corpus",
]
