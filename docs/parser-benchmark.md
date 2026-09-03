# Parser Benchmark Corpus

This document defines how the Phase 1 prompt-parser benchmark corpus is created, reviewed, frozen, and scored. The corpus lives at `tests/fixtures/parser_benchmark.jsonl` and contains exactly 100 records.

## Record schema

Each JSON Lines record contains:

- `id`: stable unique identifier.
- `category`: `supported_paraphrase | boundary | malformed | unsupported`.
- `prompt`: exact user input.
- `expected_outcome`: `accepted | rejected`.
- `expected_spec`: complete canonical `CompositionSpec` when accepted; otherwise `null`.
- `expected_error`: stable error code when rejected; otherwise `null`.
- `label_rationale`: short explanation of the expected result.

## Composition and labeling

- Cover all three Phase 1 moods, duration boundaries, optional/default fields, spelling and phrasing variation, conflicting instructions, malformed durations, famous-piece requests, unsupported instruments, and unsupported musical traditions.
- Do not copy candidate-model outputs into expected labels.
- The primary labeler assigns every expected field using the frozen `CompositionSpec` and product rules.
- A second human independently reviews every expected outcome and field before seeing the primary rationale.
- Record disagreements, adjudicate them against the roadmap and schema, and update the labeling guide when an ambiguity is discovered.
- Pre-adjudication field-level agreement must be at least 90%. If it is lower, revise the guide and relabel before freezing the corpus.

## Freeze and versioning

- Freeze the labeling guide before scoring candidate models.
- Commit the corpus and record its SHA-256 in `MODELS.md` with every benchmark run.
- Changing a prompt or expected label creates a new corpus version and invalidates comparisons with results from the previous version.
- Keep the final model-selection corpus out of model prompts, training, and repair examples.

## Scoring

Use the thresholds in §8 of [`roadmap.md`](roadmap.md). Report first-response schema validity, post-repair/fallback success, field-level semantic exact match, unsupported-request rejection, p50/p95 latency, and total benchmark cost.
