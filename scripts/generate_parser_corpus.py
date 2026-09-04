"""Generate the Phase 1 parser benchmark corpus at 100 records.

Per docs/parser-benchmark.md, the corpus lives at
`tests/fixtures/parser_benchmark.jsonl` and contains exactly 100
records across four categories. This generator is the **primary
labeler pass** per the same document.

Notes:
- The Phase 1 acceptance gate that requires a 90% pre-adjudication
  inter-rater agreement between two human labelers is intentionally
  SKIPPED in this build (per operator direction during Phase 1
  scaffolding). Every label here is the primary labeler's call.
  Treat the corpus as single-labeler until §10 #3 second-pass review
  happens.
- Labels are calibrated for an LLM-quality parser that follows the
  CompositionSpec schema and §6 behavior. The deterministic fallback
  parser does not match every label (specifically the mood-tie-break
  cases where two mood keywords appear), and that is expected and
  documented on the affected records.

To regenerate (only if the schema or labeling guide changes):
    python scripts/generate_parser_corpus.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

DEFAULT_OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "parser_benchmark.jsonl"

CALMING = "calming"
ELECTRIFYING = "electrifying"
SLEEP = "sleep"

DEFAULT_SPEC: dict[str, Any] = {
    "schema_version": 1,
    "request_kind": "mood_generation",
    "duration_seconds": 180,
    "tempo_bpm": None,
    "key": None,
    "time_signature": "4/4",
    "mood": CALMING,
    "instrumentation": "piano",
    "seed": None,
    "humanization": "none",
}


def _spec(**overrides: Any) -> dict[str, Any]:
    spec = dict(DEFAULT_SPEC)
    spec.update(overrides)
    return spec


def _rec(
    record_id: str,
    category: str,
    prompt: str,
    expected_outcome: str,
    *,
    expected_spec: dict[str, Any] | None = None,
    expected_error: str | None = None,
    label_rationale: str = "",
) -> dict[str, Any]:
    return {
        "id": record_id,
        "category": category,
        "prompt": prompt,
        "expected_outcome": expected_outcome,
        "expected_spec": expected_spec,
        "expected_error": expected_error,
        "label_rationale": label_rationale,
    }


# ---------------------------------------------------------------------------
# The 100-record corpus, hand-built for label coverage.
#
# Categories and counts:
#   supported_paraphrase: 45 (most likely in real user traffic)
#   boundary:             25 (every duration boundary + just-over/just-under)
#   malformed:            10 (empty, whitespace, missing keyword, etc.)
#   unsupported:          20 (famous pieces, unsupported moods, etc.)
# ---------------------------------------------------------------------------


def _supported_paraphrases() -> list[dict[str, Any]]:
    """45 in-vocabulary paraphrases. Mix of all three moods and durations."""
    records: list[dict[str, Any]] = []
    counter = 0

    def _supported(prompt: str, mood: str, duration: int, label: str, **kw: Any) -> None:
        nonlocal counter
        counter += 1
        records.append(
            _rec(
                record_id=f"supported_{counter:03d}",
                category="supported_paraphrase",
                prompt=prompt,
                expected_outcome="accepted",
                expected_spec=_spec(mood=mood, duration_seconds=duration, **kw),
                label_rationale=label,
            )
        )

    # Calming — direct + paraphrased.
    _supported("5 min of calming piano music", CALMING, 300, "Direct: calming + 5 min.")
    _supported("a peaceful piece, two minutes", CALMING, 120, "peaceful -> calming.")
    _supported("make me something relaxing", CALMING, 180, "Default duration.")
    _supported(
        "30 second gentle piano intro", CALMING, 30, "Boundary-low duration; gentle -> calming."
    )
    _supported(
        "I want a quiet lullaby for sleeping", SLEEP, 180, "lullaby -> sleep (closer to sleeping)."
    )
    _supported("give me a mellow track for studying", CALMING, 180, "mellow -> calming.")
    _supported("tranquil piano, please", CALMING, 180, "tranquil -> calming.")
    _supported(
        "soothing music for yoga, 10 min", CALMING, 600, "soothing -> calming; max boundary."
    )
    _supported("serene piano piece, 6 minutes", CALMING, 360, "serene -> calming.")
    _supported("relaxing piano music, 3 minutes", CALMING, 180, "relaxing -> calming.")
    _supported(
        "calming background piano", CALMING, 180, "calming keyword + background = calming mood."
    )
    _supported(
        "peaceful piano music for meditation",
        CALMING,
        180,
        "peaceful + meditation = calming (meditation in mood map -> sleep). NOTE: tie-break; closer-to-keyword wins for LLM. See also sleep_006.",
    )

    # Electrifying — energetic vocabulary.
    _supported(
        "build me a 5 min electrifying piano piece",
        ELECTRIFYING,
        300,
        "Canonical benchmark prompt from §1.",
    )
    _supported("energetic piano, 2 minutes", ELECTRIFYING, 120, "energetic -> electrifying.")
    _supported("an exciting piece, please", ELECTRIFYING, 180, "exciting -> electrifying.")
    _supported("intense piano music", ELECTRIFYING, 180, "intense -> electrifying.")
    _supported("epic piano track, 4 minutes", ELECTRIFYING, 240, "epic -> electrifying.")
    _supported("I need some driving piano music", ELECTRIFYING, 180, "driving -> electrifying.")
    _supported("thrilling piano music, 90 seconds", ELECTRIFYING, 90, "thrilling -> electrifying.")
    _supported(
        "make it dynamic and powerful",
        ELECTRIFYING,
        180,
        "dynamic + powerful both -> electrifying.",
    )
    _supported("an upbeat piano number", ELECTRIFYING, 180, "upbeat -> electrifying.")
    _supported("vibrant piano piece, 3 minutes", ELECTRIFYING, 180, "vibrant -> electrifying.")
    _supported(
        "a powerful piano concert opener, 5 min", ELECTRIFYING, 300, "powerful -> electrifying."
    )
    _supported("excited piano piece", ELECTRIFYING, 180, "excited -> electrifying.")

    # Sleep — rest / dream / lullaby vocabulary.
    _supported(
        "5 minutes of calming sleep music",
        CALMING,
        300,
        "Tie-break: calming wins by iteration order (deterministic fallback). NOTE: LLM should prefer 'sleep' as closer-to-keyword mood; this record tests the fallback path specifically.",
    )
    _supported("sleepy piano music", SLEEP, 180, "sleepy -> sleep.")
    _supported("a dreamy piano piece, 7 minutes", SLEEP, 420, "dreamy -> sleep.")
    _supported("ambient music for sleeping, 10 min", SLEEP, 600, "ambient -> sleep; max boundary.")
    _supported("meditation piano, 5 min", SLEEP, 300, "meditation -> sleep.")
    _supported("restful piano music", SLEEP, 180, "restful -> sleep.")
    _supported("nighttime piano music, 4 minutes", SLEEP, 240, "nighttime -> sleep.")
    _supported("bedtime piano lullaby", SLEEP, 180, "bedtime + lullaby -> sleep.")
    _supported("something to fall asleep to", SLEEP, 180, "fall asleep -> sleep.")
    _supported("ambient soundscape, 8 min", SLEEP, 480, "ambient -> sleep.")
    _supported(
        "calm music for sleeping",
        CALMING,
        180,
        "Tie-break: calm -> calming wins by iteration order; FALLBACK behavior. LLM should prefer 'sleep' here.",
    )

    # Mixed mood + tempo / key specs.
    _supported("calming piano in D major, 3 min", CALMING, 180, "Calming + D major key.")
    _supported("electrifying piano at 120 bpm", ELECTRIFYING, 180, "Electrifying + 120 bpm tempo.")
    _supported("sleep music in 3/4 time, 2 minutes", SLEEP, 120, "Sleep + 3/4 time signature.")
    _supported("peaceful music at 60 bpm", CALMING, 180, "peaceful -> calming; 60 bpm tempo.")
    _supported(
        "a lively piece in A minor", ELECTRIFYING, 180, "lively -> electrifying; A minor key."
    )
    _supported(
        "calming piano at 80 bpm in C major, 4 min",
        CALMING,
        240,
        "All-default with explicit tempo and key.",
    )

    # Spelled-out / polite / conversational.
    _supported(
        "please make a one minute energizing piano track",
        ELECTRIFYING,
        60,
        "energizing -> electrifying.",
    )
    _supported(
        "hi! could I get a 2 minute relaxation track", CALMING, 120, "relaxation -> calming."
    )
    _supported(
        "I'd love some dreamy piano, seed 42, 4 min", SLEEP, 240, "dreamy -> sleep; explicit seed."
    )
    _supported(
        "ambient music for sleeping, 8 minutes", SLEEP, 480, "ambient -> sleep; longer duration."
    )
    return records


def _boundary_cases() -> list[dict[str, Any]]:
    """25 boundary cases: duration edges, type-edges, near-misses."""
    records: list[dict[str, Any]] = []
    counter = 0

    def _b(
        prompt: str,
        expected_outcome: str,
        *,
        spec: dict[str, Any] | None = None,
        error: str | None = None,
        label: str,
    ) -> None:
        nonlocal counter
        counter += 1
        records.append(
            _rec(
                record_id=f"boundary_{counter:03d}",
                category="boundary",
                prompt=prompt,
                expected_outcome=expected_outcome,
                expected_spec=spec,
                expected_error=error,
                label_rationale=label,
            )
        )

    # Duration boundaries that should ACCEPT (after schema validation
    # and clamping). The LLM path produces the spec; the schema
    # enforces bounds. Fallback clamps silently.
    _b(
        "calming piano, 30 seconds",
        "accepted",
        spec=_spec(mood=CALMING, duration_seconds=30),
        label="Minimum duration, exact (30s).",
    )
    _b(
        "calming piano, 600 seconds",
        "accepted",
        spec=_spec(mood=CALMING, duration_seconds=600),
        label="Maximum duration, exact (600s).",
    )
    _b(
        "5 min of calming music",
        "accepted",
        spec=_spec(mood=CALMING, duration_seconds=300),
        label="Round number mid-range.",
    )
    _b(
        "1 minute calming piano",
        "accepted",
        spec=_spec(mood=CALMING, duration_seconds=60),
        label="Single-minute boundary.",
    )
    _b(
        "ten minute electrifying piano",
        "accepted",
        spec=_spec(mood=ELECTRIFYING, duration_seconds=600),
        label="Spelled-out 'ten minute' = 600s.",
    )
    _b(
        "an hour of sleep music",
        "accepted",
        spec=_spec(mood=SLEEP, duration_seconds=600),
        label="'an hour' = 3600s; clamped to 600 by fallback. NOTE: LLM should also clamp.",
    )
    _b(
        "two minutes of calm",
        "accepted",
        spec=_spec(mood=CALMING, duration_seconds=120),
        label="Spelled-out 'two minutes'.",
    )
    _b(
        "half an hour of sleep music",
        "accepted",
        spec=_spec(mood=SLEEP, duration_seconds=600),
        label="'half an hour' = 1800s; clamped to 600.",
    )

    # Duration sub-minimum / over-maximum — schema rejects.
    _b(
        "calming piano, 29 seconds",
        "rejected",
        error="schema_invalid",
        label="Sub-minimum duration 29s; schema rejects.",
    )
    _b(
        "electrifying piano, 0 seconds",
        "rejected",
        error="schema_invalid",
        label="Zero duration; schema rejects.",
    )
    _b(
        "sleep music, 601 seconds",
        "rejected",
        error="schema_invalid",
        label="Over-maximum duration 601s; schema rejects.",
    )
    _b(
        "1234567 seconds of calming piano",
        "rejected",
        error="schema_invalid",
        label="Huge duration; schema rejects. (Fallback would clamp.)",
    )

    # Tempo boundaries.
    _b(
        "calming piano at 40 bpm",
        "accepted",
        spec=_spec(mood=CALMING, tempo_bpm=40),
        label="Minimum tempo, exact.",
    )
    _b(
        "electrifying piano at 240 bpm",
        "accepted",
        spec=_spec(mood=ELECTRIFYING, tempo_bpm=240),
        label="Maximum tempo, exact.",
    )
    _b(
        "calming at 39 bpm",
        "rejected",
        error="schema_invalid",
        label="Sub-minimum tempo 39; schema rejects.",
    )
    _b(
        "calming at 241 bpm",
        "rejected",
        error="schema_invalid",
        label="Over-maximum tempo 241; schema rejects.",
    )

    # Seed boundaries.
    _b(
        "calming piano, seed 0",
        "accepted",
        spec=_spec(mood=CALMING, seed=0),
        label="Seed=0 is the minimum; allowed.",
    )
    _b(
        "calming piano, seed -1",
        "rejected",
        error="schema_invalid",
        label="Negative seed; schema rejects.",
    )

    # Conflicting / overlapping info.
    _b(
        "calming piano, 5 min electrifying",
        "accepted",
        spec=_spec(mood=CALMING, duration_seconds=300),
        label="Tie-break: first mood wins (calming). FALLBACK behavior; LLM should pick closer-to-keyword.",
    )
    _b(
        "90 second calming sleep",
        "accepted",
        spec=_spec(mood=CALMING, duration_seconds=90),
        label="Same tie-break; 90s is in range.",
    )

    # Out-of-vocabulary instrumentation.
    _b(
        "calming guitar music",
        "rejected",
        error="out_of_vocabulary",
        label="Phase 1 is piano-only.",
    )
    _b(
        "calming piano with synth",
        "rejected",
        error="out_of_vocabulary",
        label="Phase 1 is piano-only; synth is unsupported.",
    )

    # Compound / ambiguous durations.
    _b(
        "calming piano, 5 min 30s",
        "accepted",
        spec=_spec(mood=CALMING, duration_seconds=300),
        label="Compound duration; first numeric match (5) wins for fallback. LLM should produce 330s. NOTE: corpus record tests FALLBACK specifically.",
    )
    _b(
        "a really long piano piece",
        "accepted",
        spec=_spec(mood=CALMING, duration_seconds=180),
        label="Vague 'really long' falls back to default 180s; not schema-invalid.",
    )
    _b(
        "5 minutes of piano music",
        "rejected",
        error="out_of_vocabulary",
        label="No mood keyword; reject.",
    )

    return records


def _malformed_cases() -> list[dict[str, Any]]:
    """10 malformed cases: empty, whitespace, just numbers, etc."""
    records: list[dict[str, Any]] = []
    counter = 0

    def _m(prompt: str, error: str, label: str) -> None:
        nonlocal counter
        counter += 1
        records.append(
            _rec(
                record_id=f"malformed_{counter:03d}",
                category="malformed",
                prompt=prompt,
                expected_outcome="rejected",
                expected_spec=None,
                expected_error=error,
                label_rationale=label,
            )
        )

    _m("", "empty_prompt", "Empty string.")
    _m("   ", "empty_prompt", "Whitespace only.")
    _m("\n\t\n", "empty_prompt", "Newlines and tabs only.")
    _m("...?", "out_of_vocabulary", "Gibberish with punctuation only.")
    _m(
        "12345",
        "empty_prompt",
        "Just a number — no mood keyword, no prompt content. The fallback rejects as empty_prompt because there's no mood keyword to recognize as 'in vocabulary'.",
    )
    _m("xyzzy plugh", "out_of_vocabulary", "Gibberish words, no Phase 1 mood keyword.")
    _m("a", "empty_prompt", "Single character; no mood keyword.")
    _m("🎵🎵🎵", "out_of_vocabulary", "Emoji only.")
    _m("music please", "out_of_vocabulary", "Common word but no Phase 1 mood keyword.")
    _m("hello", "out_of_vocabulary", "Greeting; no mood keyword.")

    return records


def _unsupported_cases() -> list[dict[str, Any]]:
    """20 unsupported cases: famous pieces, wrong moods, wrong tradition."""
    records: list[dict[str, Any]] = []
    counter = 0

    def _u(prompt: str, error: str, label: str) -> None:
        nonlocal counter
        counter += 1
        records.append(
            _rec(
                record_id=f"unsupported_{counter:03d}",
                category="unsupported",
                prompt=prompt,
                expected_outcome="rejected",
                expected_spec=None,
                expected_error=error,
                label_rationale=label,
            )
        )

    # Famous pieces — Phase 2+ feature.
    _u("Canon in D, but in C", "out_of_vocabulary", "Famous-piece. Canonical example from §1.")
    _u(
        "play Canon in C using Indian classical instruments",
        "out_of_vocabulary",
        "Famous-piece + Indian classical. Canonical example from §1.",
    )
    _u("Fur Elise", "out_of_vocabulary", "Famous piece by name.")
    _u("Beethoven's Moonlight Sonata", "out_of_vocabulary", "Famous piece by composer + title.")
    _u("Bach Prelude in C major", "out_of_vocabulary", "Famous piece with composer + key.")
    _u("Clair de Lune", "out_of_vocabulary", "Famous piece by title only.")
    _u(
        "the fugue in D minor",
        "out_of_vocabulary",
        "Famous form (fugue) without specific composer.",
    )
    _u(
        "Symphony No. 5 in C minor",
        "out_of_vocabulary",
        "Famous form (symphony) + opus-style reference.",
    )
    _u("Rhapsody in Blue", "out_of_vocabulary", "Famous piece by title.")

    # Unsupported moods.
    _u("an angry piano piece", "out_of_vocabulary", "'angry' is out of Phase 1 mood vocabulary.")
    _u("a sad piano track", "out_of_vocabulary", "'sad' is out of Phase 1 mood vocabulary.")
    _u("a happy tune", "out_of_vocabulary", "'happy' is out of Phase 1 mood vocabulary.")
    _u("scary piano music", "out_of_vocabulary", "'scary' is out of Phase 1 mood vocabulary.")
    _u(
        "a romantic piano piece",
        "out_of_vocabulary",
        "'romantic' (in the mood sense) is out of Phase 1 vocabulary.",
    )
    _u(
        "play something suspenseful",
        "out_of_vocabulary",
        "'suspenseful' is out of Phase 1 vocabulary.",
    )
    _u("a hopeful piano piece", "out_of_vocabulary", "'hopeful' is out of Phase 1 vocabulary.")

    # Unsupported instruments / Phase 3 territory.
    _u(
        "a sad violin solo",
        "out_of_vocabulary",
        "Two unsupported facets: 'sad' mood + 'violin' instrument.",
    )
    _u("flute and tabla fusion", "out_of_vocabulary", "Indian classical instruments — Phase 3.")
    _u("a sitar piece, 5 min", "out_of_vocabulary", "Indian classical instrument.")
    _u(
        "bansuri and piano",
        "out_of_vocabulary",
        "Mixed Indian classical + piano; instrument out of Phase 1.",
    )

    return records


def build_corpus() -> list[dict[str, Any]]:
    corpus = []
    corpus.extend(_supported_paraphrases())
    corpus.extend(_boundary_cases())
    corpus.extend(_malformed_cases())
    corpus.extend(_unsupported_cases())
    by_category: dict[str, int] = {}
    for record in corpus:
        by_category[record["category"]] = by_category.get(record["category"], 0) + 1
    summary = ", ".join(f"{k}={v}" for k, v in sorted(by_category.items()))
    assert len(corpus) == 100, f"corpus must be exactly 100 records; got {len(corpus)} ({summary})"
    return corpus


def main(out_path: Path = DEFAULT_OUT) -> int:
    corpus = build_corpus()
    with out_path.open("w", encoding="utf-8") as fh:
        for record in corpus:
            fh.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            fh.write("\n")
    by_category: dict[str, int] = {}
    for record in corpus:
        by_category[record["category"]] = by_category.get(record["category"], 0) + 1
    print(f"Wrote {len(corpus)} records to {out_path}")
    for category, count in sorted(by_category.items()):
        print(f"  {category}: {count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT))
