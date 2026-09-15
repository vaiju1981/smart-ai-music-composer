# Model fine-tuning: what it can and cannot change

This note records the assessment made at the end of the Phase 1 musical-quality
pass, so the question is answered from the architecture rather than re-asked:
**fine-tuning a model cannot improve the music this app produces, and that is a
property of the design, not a measurement.** The claim rests on two facts about
the code — what the model is asked for, and what is reachable from the code that
writes a note — and both are checkable below.

## What the model does today

The LLM's entire job is **prompt → `CompositionSpec`**. It is called from the
job pipeline's parsing stage, its answer is validated against the local schema
before anything downstream consumes it, and it is re-prompted with the
schema-in-context error at most twice before the deterministic fallback parser
takes over. Every accepted prompt is recorded in `manifest.json` with its
`parser_source`, so which path produced a given piece is never a guess.

The spec it produces is ten fields wide, and they are all *requests*:
`schema_version`, `request_kind`, `duration_seconds`, `tempo_bpm`, `key`,
`time_signature`, `mood`, `instrumentation`, `seed`, `humanization`. Not one of
them can carry a note, a phrase, a contour or a chord. The strongest musical
statement the model can make is "calming, 180 seconds, strings and piano" — and
after that it is out of the process.

Everything that follows comes from the rule engine in `src/saimc/compose/`, then
the notation and performance layers, then `src/saimc/render/`. This is verifiable
in one command:

```sh
grep -rn "saimc.llm" src/saimc/compose src/saimc/render src/saimc/release
```

No matches. No LLM is reachable from the code that writes, voices, lints or
renders a note. There is also no model call after composition — the scorecard is
not sent anywhere, and nothing in the app is learning from a render.

## Why fine-tuning cannot change the melody

Two independent reasons, either of which is sufficient:

1. **The note generator contains no model.** Fine-tuning changes the weights of
   a model; if no model participates in choosing pitches, no weight change can
   reach a pitch. The only place a fine-tune could act is spec fidelity: given
   the same prompt, does the parser return *the right spec*. That improves
   *which* piece was asked for, not whether the piece is any good.
2. **The spec is not expressive enough to encode the music.** The engine is a
   deterministic function of the spec and the seed: the same `CompositionSpec`
   composes byte-identical notes every time ([`roadmap.md`](roadmap.md) §8,
   "canonical artifacts are reproducible"). A parser that returned a perfect spec
   and a parser that returned the same spec by luck produce the same music. The
   space a model fine-tune can move in — spec accuracy — is orthogonal to the
   space the melody lives in.

The one place the model has a musical fingerhold is `humanization`, which the
*performance* layer reads for micro-timing and velocity (and, at the expressive
setting, staccato shortening of repeats). That changes how the notes are played,
not which notes they are; the engraved score stays on the grid.

## Where quality feedback actually flows today

The instrument this pass built is model-free and deliberately so. `saimc.quality`
measures ten properties of a composed piece and, in each `QualityThreshold`'s
`hint`, records *which engine knob moves that metric*. Its `findings` therefore
read as instructions to whoever is tuning the generator — or to an automated
repair loop over the generator's parameters — rather than as a verdict on a
prompt. That is the feedback direction that can actually improve the music, and
it points at `src/saimc/compose/`.

## What *would* make model work the lever

A note-generating model — neural, or a hybrid in which a model proposes and the
rule engine disposes. Concretely: a model that, given the spec (and optionally a
musical context), emits pitches and rhythms for the melody and the accompaniment
voices, with the existing linter, scorecard and release gates standing between it
and a render. That is the change that makes fine-tuning meaningful, because it is
the change that puts a weight between a prompt and a note.

It is a Phase 2 conversation, and it arrives with four conditions that the
current architecture does not have to solve:

- **Rights.** A model writing notes has to be fine-tuned on something. The
  training corpus's licence and acceptable-use terms become product obligations,
  the same way every bundled asset's does ([`roadmap.md`](roadmap.md) §4,
  `THIRD_PARTY_NOTICES.md`). §8 already refuses a model that fails a licence
  review regardless of quality.
- **Reproducibility.** The §8 canonical-artifact gate requires byte-identical
  notes for a fixed spec and seed. A stochastic note generator breaks that unless
  it is seeded, greedy, or the gate is amended deliberately — not silently.
- **The acceptance instruments already exist.** `saimc-quality` and
  `gate_musical_quality` measure the music itself; `saimc-benchmark` measures
  parsing. A note-generating model would be scored by the former, which is the
  honest test of whether it made the music better. The thresholds are conventional
  composition practice, set before the engine was measured, so they are not a bar
  the current generator happens to clear.
- **A held-out listening check.** The metrics are a proxy. Any note-generating
  model still has to be listened to before it is preferred over the rule engine.

## The model-side instrument that already exists

The parser side is fully instrumented and is where any model work should start
today:

- `saimc-benchmark` runs a corpus of prompts against a candidate model and
  reports first-response schema validity, post-repair/fallback success,
  field-level semantic exact match, unsupported-request rejection, and p50/p95
  latency.
- `tests/fixtures/parser_benchmark.jsonl` holds the 100 hand-labeled prompts;
  [`parser-benchmark.md`](parser-benchmark.md) defines the schema, label rules,
  freeze process and scoring.
- [`MODELS.md`](../MODELS.md) is the registry §10 #3 requires before a candidate
  model may be requested at all, and it records the decision, the licence and the
  benchmark result. It currently holds no approved model; the built-in
  `DEFAULT_MODEL` in `src/saimc/llm/config.py` is a zero-setup convenience default
  for local development, not an approved choice, and it is overridable by
  `OLLAMA_MODEL`.

## What to restore before any model selection

[`parser-benchmark.md`](parser-benchmark.md) records that pre-adjudication
field-level agreement is **currently SKIPPED**: every label in the corpus is a
single labeler's call, where [`roadmap.md`](roadmap.md) §8 requires at least 90%
agreement between two independent labelings before the corpus is frozen.
Restoring that second pass is a Phase 1 release-gate item — §8's "Parser
robustness" bullet, and §10 #3's precondition for selecting a model — because a
model measured against single-labeler labels is measured against those labels'
errors as much as against the product rules. It is cheap: the corpus is 100
records and the labeling guide is written. It should happen before the first
candidate is scored, not after.

## Decision

No fine-tuning in Phase 1. The measured musical gain of fine-tuning this app's
model is zero by construction: the model cannot write a note, and the spec it
does write cannot describe one. Revisit when — and only when — a note-generating
model enters the pipeline; at that point fine-tuning becomes the lever, and the
conditions above become the work.
