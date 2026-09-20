# The listening judge

## Why there is one

`quality.py` measures thirteen things about a piece. Those same thirteen
numbers are:

- what the engine is tuned toward,
- what `session/arbiter.py` ranks candidates by,
- what the session's "critics" report (they are the metrics grouped by axis,
  not a model),
- what `session/repairs.py` moves,
- and what both `release/gates.py` and `session/gates.py` read.

One opinion, five jobs. Nothing in the project was positioned to disagree with
it, so *the music is good* had been answered, every time, by the instrument
that decides what good means. `release/gates.py` currently accepts a **50%
threshold-breach rate** (27 of 54 sampled cells) as passing, and no reading
existed that could say whether that is a lax bar or a sensible one.

`saimc-judge` is the disagreement. It shows a model two pieces, asks which is
the better music, and compares its answer with the arbiter's order over the
same two.

## How to run it

```sh
saimc-judge --dry-run                              # compose, pair, print
saimc-judge --model qwen3:8b -o var/judged.json    # ask a listener
```

`--dry-run` exercises the grid, the pairing and the scorecard's own order and
calls nothing. It is what CI can take.

## What it does

1. **Compose a grid.** mood × duration × seed, through the real engine, with
   `score_piece` taken from the same notation the judge will read.
2. **Pair it.** One seeded shuffle, each piece compared once. A round robin
   would be quadratic in model calls for a linear gain in evidence.
3. **Ask, blind.** `describe_piece` renders the notation as a readable score —
   key, tempo, and every bar's melody, harmony, bass and kit. It takes a
   `NotationScore` and takes nothing else, so it *cannot* leak a measurement;
   `test_judge.py` asserts that over the signature and over real output.
4. **Ask again, swapped.** A model has a position bias. A pair whose two passes
   disagree is counted as *inconsistent* and kept out of the agreement rate.
5. **Report.** Consistency rate, agreement rate, and every disagreement with
   the listener's own sentence about why.

## Reading the number

`agreement_rate` is over the pairs **both** orders can separate. Chance is
**0.5**, not 0.

- Near 0.5 — the scorecard's order carries nothing a listener recognises.
- Near 1.0 — the thirteen metrics are standing in for taste about as well as a
  proxy can.

Read `consistency_rate` first. A judge that is 55% consistent has told you
about itself, not about the music, and its agreement rate is noise.

## What it cannot tell you

- **Not that the arbiter is wrong.** The judge is a model with opinions, not a
  musician of record. A disagreement says the order is not the one this
  listener would give — which is the thing worth knowing, and not the same
  claim.
- **Nothing about the render.** It reads notation, so the soundfont, the mix
  and the master are not in evidence. What is judged is the writing.
- **Nothing beyond bar 16 by default.** A long piece is one repeating form and
  sending all of it would spend the prompt on repeats. The description states
  how many bars it left out.

## What is deliberately absent

**There is no `MIN_AGREEMENT` and no release gate reading one.** This project's
rule is that no guard is written before it can fail, and nothing has run this
against a live model. A threshold chosen now would be a number invented to
match an unmeasured quantity. The floor lands with the first recorded sweep, in
the commit that records it — and `docs/open-items.md` C5 carries the gap
until then.

One failure *is* enforced: a run where the judge never survived a swap exits
non-zero, because it measured its own position bias and nothing about the
music.
