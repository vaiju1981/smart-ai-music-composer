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
live host, so §8's 98/97/95 bars have never been scored against a model, and
`MODELS.md` — which §10 #3 requires to exist before a candidate may be requested
at all — holds no approved model. `docs/model-fine-tuning.md:112` still says so in
prose.

The runner itself is **not** what is missing: `_run_with_client`
(`src/saimc/benchmark_cli.py:128`) sweeps the corpus, owns and closes the adapter,
and measures latency around the whole `parse_prompt` call. This file said it raised
`NotImplementedError`; that was true until `235d2df` and was already false when
this file was written. What is open is the *run*, not the code that would do it.

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

### B1. Ranking is not shown, and the workspace does not say which draft won

**What is open.** The arbiter ranks a fan-out (`session/arbiter.py`) and the
session workspace does not show that order: the cards are in the order the
drafts were made, and nothing on screen says which one the ratchet would keep.
`SessionResponse` carries no ranking, so the page cannot show one without
inventing it — which is why it shows none.

**What was closed instead.** The presentation itself. `jobs/static/index.html`
now shows the conductor's turns and tool calls, the candidates with playable
sketches, all thirteen measurements per draft with the misses marked and read
back in plain language, the verdict buttons, the edit box and its
applied/refused/unread report, and publish. `tests/ui/` drives all of it in a
browser.

**What closing this involves.** Either `rank` on the session surface (the
arbiter's order, as a list of draft ids with the element that decided each
pair), or a `compare` call the page makes and renders. The first is the honest
one: the order is a property of the drafts, not of a question about them.

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

### C4. Five voices is the ensemble, and a brief can ask for fifteen

**What is open.** `ROLE_LIMITS` is one melody, two harmony, one bass and one
kit — five instruments, and the schema refuses a sixth. A brief asking for "10
to 15 instruments" is therefore asking for something this engine does not
write, and what comes back is a three- or four-piece ensemble.

**What was closed.** The *silence*, not the limit. `/meta` now publishes the
ceiling and the page states it before a brief is typed, built from the same
table `_validate_ensemble` refuses against. An enforced limit nobody is told
about reads, from outside, as the product ignoring what was asked — which is
the deaf-product failure `create_job` names.

**What was then closed.** The ceiling moved to fifteen — one melody, twelve
harmony, a bass and a kit, which is exactly what MIDI carries once the kit has
channel 10 — and `HarmonyVoices.divisi` shares the chord out across the pads so
they voice different inversions instead of one dyad in twelve registers. The
linter's simultaneous-note cap became per *voice*, which is where "a pianist's
two hands" actually applies; counted across the score it was a cap on the size
of the ensemble.

**What is open is that a large ensemble measures worse.** Over 24 cells (3
moods x 2 durations x 4 seeds) at each size, the share of pieces breaching at
least one quality threshold:

| harmony voices | breaching | register_separation | tessitura_overlap |
|---|---|---|---|
| 2 | 8/24 | 3.33 | 0.00 |
| 4 | 11/24 | 3.42 | 0.00 |
| 8 | 11/24 | 3.42 | 0.00 |
| 12 | **22/24** | **0.58** | **5.08** |

The collapse is `settle_harmony_register`: it places each bed against the
finished tune using the instrument's own comfortable range, and a dozen
instruments with overlapping ranges all settle into the same band, on top of
the melody. Divisi does not fix it — it moves `tessitura_overlap` from 5.92 to
5.08 and nothing else — because it distributes *pitch classes*, not registers.

**What closing it involves** is spreading the beds across registers rather than
settling each one independently: the bed's placement has to become a decision
about the whole ensemble, the way `melody_band_for` is already a decision about
the whole piece. Until then a twelve-voice request composes, renders and sounds
crowded, and the numbers above say by how much.

**What is not open:** the parser. It was never failing to adhere — it was being
handed a schema that could not express the request.

### C5. The judge has never been run against a live listener

**What is open.** `saimc-judge` composes the grid, pairs it, renders each piece
blind and scores the two orders against each other — and no sweep has been run
against a model, so `agreement_rate` has never had a value. The number this
project most needs is the one it has built the machinery for and not yet taken.

**What is deliberately absent.** There is no `MIN_AGREEMENT` and no release
gate reading one, for this repo's own rule: no guard is written before it can
fail, and a floor chosen before the first sweep would be a number invented to
be cleared. It lands with the measurement, in the commit that records it.

**Why it matters more than its size suggests.** `release/gates.py` accepts a
50% threshold-breach rate (C1) and nothing can currently say whether that bar
is lax or sensible, because the only reading of "good music" in the project is
the one the bar is made of. E1 — whether the scorecard is a proxy for taste or
a definition of it — is not answerable without this number either.

**What running it involves.** A host, `saimc-judge --model <tag> -o
var/judged.json`, and reading the disagreements by hand: each one is a piece
the arbiter ranked above another that a listener preferred, with the listener's
sentence about why. `docs/judge.md` says how to read the rate and what it
cannot tell you.

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
