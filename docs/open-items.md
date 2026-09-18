# Open items

Everything known to be unfinished, unmeasured or deliberately deferred, in one
place, so it can be read without reconstructing it from commit messages.

Written 2026-09-18, at the merge point of PR #2 (`feature/parser-and-spec`).
**Keep this honest by deleting lines, not by adding to them**: each item below
either has an owner and a closing action, or a recorded reason for waiting. An
item that closes should leave the file, and an item that changes shape should be
rewritten rather than appended to.

Items are grouped by *what is blocking them*, because that is what decides what
happens next. Verified against the tree at the time of writing; the file:line
citations are the evidence, and each was read rather than recalled.

---

## A. Blocked on the owner's labeling pass

### A1. The second independent labeling (G2)

**What is open.** `docs/model-fine-tuning.md` §10 #3 and `roadmap.md` §8 require at
least 90% pre-adjudication field-level agreement between **two independent**
labelings before the corpus is frozen, and forbid scoring any candidate model
before that. Today the corpus has one labeler's call per record.

**What exists.** The tool (`saimc-label`) is built and is the thing to run:

```sh
saimc-label --corpus tests/fixtures/parser_benchmark.jsonl \
            --review tests/fixtures/parser_benchmark_review.jsonl
```

It shows the prompt only — never the primary label or `category`, so the reviewer
cannot be anchored — writes in the corpus's own `BenchmarkRecord` schema, appends
one flushed line per record, and resumes by skipping ids already present.

**What is deliberately absent.** `tests/fixtures/parser_benchmark_review.jsonl`
does not exist, and there is no acceptance gate asserting the agreement rate
either. Both are absent *by design*: the rule this project holds is that no guard
is written before it can fail, so the gate lands with the file rather than before
it.

**What closing it involves.** Run the wizard to completion, commit the review
file, then add a gate to `tests/acceptance/test_release_gates.py` asserting
`compare_labelings(primary, review).rate >= MIN_FIELD_AGREEMENT` (0.90), with the
premises **asserted rather than trusted**: both files load, 100 ids each, the id
sets match.

**One thing that does not substitute for it.**
`tests/acceptance/test_labeling_round_trip.py` (commit `de804e9`) drives the real
console script over the real 100-record fixture end to end and scores the file it
writes at exactly 1.0. That proves the *tool* works. It cannot discharge §10 #3,
because its keystrokes are generated *from* the primary labels, so its agreement
is 100% by construction. That is why it writes to a tmp dir, is never committed,
and has no release gate asserted against it.

### A2. The live model sweep (G4) — blocked behind A1

**What is open.** `saimc-benchmark --label <model>` has never been run against a
live host. `_run_with_client` raises `NotImplementedError` in the shipped path, so
§8's 98/97/95 bars have never been scored against a model, and `MODELS.md` — which
§10 #3 requires to exist before a candidate may be requested at all — holds no
approved model. `docs/model-fine-tuning.md:112` still says so in prose.

**Why it waits.** Scoring a model against single-labeler labels measures it
against those labels' errors as much as against the product rules. A1 first.

**Two premises the sweep must check rather than assume**, both of which the dev
proxy can violate silently:

- a tag that **cannot generate text** (an embedding-only model) is not a text
  model and must be discarded before it is scored — a proxy's `/api/tags` does not
  distinguish them;
- the **`*-cloud` tag namespace differs** from self-hosted names, so the identifier
  recorded in `MODELS.md` is the one the proxy actually served, not the one
  intended.

**§8's tie-break is not evaluable on this host.** "The least expensive model meeting
every quality and latency threshold" assumes observable prices; nothing in this
codebase meters or prices a call, so `cost_total_usd` comes back `None` for every
candidate rather than a `0.0` that would read as "free" while meaning "unmeasured".
The recorded decision must therefore be justified on quality and latency alone, and
the doc must say that rather than imply a price comparison happened.

**Four documents are false until this lands, and G4 corrects them in the same
commit** — this repo's habit for prose the code outgrew:

| Where | What it claims today |
|---|---|
| `docs/parser-benchmark.md:24` | "Pre-adjudication field-level agreement is currently SKIPPED in Phase 1 … The 90% gate described above is a release-gate to reinstate before Phase 2 model selection" |
| `docs/model-fine-tuning.md:112` | "It currently holds no approved model" |
| `docs/model-fine-tuning.md:118-129` | the whole *What to restore before any model selection* section — "**currently SKIPPED**" |
| `scripts/generate_parser_corpus.py:9-14` | "Treat the corpus as single-labeler until §10 #3 second-pass review happens" |

---

## B. Deliberately deferred, with a recorded reason

### B1. The "show the top 3" presentation (Phase E5)

**What is open.** The fan-out and its ranking exist and are tested; the
*presentation* — draft cards with playable sketches, the scorecard's verdict per
draft, the arbiter's reasons in plain musical language — is not built.

**Why.** `src/saimc/jobs/static/index.html` has **no test witness in the repo at
all**: no test reads it, it is served by `jobs/api.py` and that is the whole of its
coverage. The workspace would be several hundred lines of markup and JS that
nothing in the suite can see, and its behaviour (fetch, render, poll) cannot be
witnessed, since the project has neither a browser nor jsdom. Every endpoint it
needs is built and tested, so this is a surface waiting for a decision about how
the UI is witnessed — not blocked work.

### B2. Draft and session retention is by session age only

**What is open.** Drafts are not pruned by their own age; they go when their
session does. `SESSION_RETENTION_DAYS = 30` (`src/saimc/session/store.py:69`) is the
only horizon, and the session is the unit.

**Why it matters.** It decides whether the preference log becomes a dataset or
stays a trickle — the plan's Open Question 2, whose first half (chain bounds,
`MAX_DELTAS_PER_REVISION` / `MAX_REVISIONS_PER_LINE`) was answered in code and
whose retention half was not.

---

## C. Recorded gaps in the music — named, not silently absorbed

### C1. Two bars breach in every electrifying arrangement (G7's recorded gap)

`texture_hierarchy` and `harmony_pad_coverage` are breached in **18 of 18**
electrifying cells, at every duration and every seed, while the other two moods
breach mostly at 30 s. It is a mood property, not a length or seed property.

The measured rate is pinned as `MAX_THRESHOLD_BREACH_RATE = 0.50`
(`src/saimc/release/gates.py:56`) with the metrics named beside it, so the ratchet
may only fall. **That is the open work: two named bars, in one mood, needing a
music fix** — a specific and bounded thing, which is what the gate was for.
Deciding the fix before the grid measured which bars breach would have been
guessing at the answer the grid exists to produce.

### C2. The tempo and the duration are one decision, not two controls

`arrange_for_duration` re-derives a pinned tempo the piece cannot hold — the length
is a release gate and outranks the tempo. That is deliberate, documented and tested,
so the engine is not wrong. What it means is that `SetTempo` can be answered with a
piece that does not do it, which the delta vocabulary's own rule forbids.

**What D3 did** was make the trade *visible*: `swallowed_tempo`
(`src/saimc/session/deltas.py:826`) refuses with both numbers, the mood's range, and
the nearest legal request. **What is open** is making it not happen. The two honest
resolutions are a duration that yields to a tempo the user pins, or a tempo range
widened per mood. Both are engine changes; the second is a musical judgement.

### C3. The treble-bass walk hole (tracker #42)

A bass instrument can end up with **no note in the walk's register**. Recorded
here because it is recorded **nowhere else** — verified by grep across `src/`,
`docs/` and `tests/`: the tracker row is the only place this fact exists. So the
open item is the recording itself, before it is the fix.

---

## D. Environment and tooling gaps

### D1. The real-binary render path is not exercised in CI, and never has been

FluidSynth → WAV → OGG → SVG → WebM is monkeypatched in every other test, and
`tests/integration/test_sketch_render.py` self-skips without a release-gate FFmpeg
(`pytestmark`, `:120`). CI builds and installs **neither** FFmpeg nor a soundfont.
A2's integration test is the first test in the repo to run them for real, and it
does not run in CI either.

**Closing it is real work**: `scripts/build_ffmpeg.sh` wired into the workflow (a
multi-minute build) plus an audited FluidSynth. Worth doing, but it is a decision
with a cost rather than a switch to flip.

### D2. The pre-push security scan runs in CI, not here

`cycode` is not installed on this machine, so the SAST/secrets/SCA pass over these
commits begins with PR #2's CI run rather than before the push. Install the CLI to
scan the diff locally, as the project rule asks.

### D3. `run.sh`'s bash is syntax-checked, not statically analysed

`shellcheck` is not installed, so the bootstrap section in `run.sh` has only been
through `bash -n`. It is verified by hand in three modes (stale install, missing
render build, unknown service), but not by a linter.

### D4. `ruff format` is not enforced

Measured today: **47 files would be reformatted, 93 already formatted.** CI runs
`ruff check` only. Enforcing the formatter needs a mass-reformat commit whose diff
would bury whatever feature lands beside it, which is why it is a decision rather
than a tidy-up. The number has grown since the same measurement was taken earlier
in the project (25 files), so this drifts upward on its own.

---

## E. Open questions that need a judgement, not work

### E1. Who wins when the user's verdict contradicts a threshold?

The proposed rule is recorded and implemented: correctness (the linter) is
non-negotiable, taste (the user) is final — so a user can never ship an illegal
piece but *can* ship one the scorecard dislikes. That is right only if the
scorecard is a proxy for taste rather than a definition of it, and the slow loop is
what would settle it.

### E2. Does the slow loop move defaults automatically, or under review?

`G6` built the proposal printer (`saimc-preferences`) and it prints the current
order beside the proposed one; a person edits `_REPAIRS`. Whether it should ever
stop waiting for a person is open, and should stay open until the metrics are shown
to track preference.

### E3. Which axes get their own critic first?

Harmony, melody, rhythm, orchestration. The axes and the critics landed together
(the honest answer depended on which measurements exist), but whether those four
are the right first set is unmeasured.

---

## Known limits, deliberate and documented — not open items

Listed so they are not mistaken for gaps. Each is a decision with its reason
written where the code is:

- **The preference log is at-least-once.** A duplicate `POST /verdict` is
  indistinguishable from a second intentional verdict (`at` is server-generated),
  so the log can hold two rows for one judgement. An idempotency key would be new
  surface for a hazard no caller has. (`src/saimc/session/preferences.py:38`)
- **`prune` never deletes a session it cannot read.** A deliberate, visible leak:
  the directory stays where a human can look at it. (`src/saimc/session/store.py:219`)
- **`SessionStorage.list_all` has no production caller.** It exists for the store's
  own completeness and its tests.
- **The composition plan is refused *exactly*, unlike the spec.** An older plan is
  missing fields and a newer one may carry a knob the engine would silently ignore,
  so the version tag is compared with `!=` in both directions — where the spec,
  widened additively, refuses only documents from the future.
- **The arbiter's metrics are a proxy for taste, not a definition of it.** It breaks
  ties and ranks; the user decides. Written into the arbiter's own docstring.
