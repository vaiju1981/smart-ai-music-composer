"""Phase 1 duration policy: target, arrange, fine-tune.

Per `docs/roadmap.md` §10 #1:

- The spec's `duration_seconds` is a target.
- The engine selects a fixed form and mood-appropriate tempo range.
- Repeats or varies complete sections until the target is reached.
- Fine-tunes tempo within the mood's allowed range to reach the target
  within the ±2% tolerance from §8.
- It never truncates a sounding note or emits an incomplete measure.
- If no valid arrangement can satisfy the tolerance, composition fails
  with a structured `duration_unfulfillable` error rather than silently
  returning the wrong duration.
- `PerformancePlan.realized_duration_seconds` records the result.
- No source section may repeat more than eight times (§10 #10).
- Repeated sections receive deterministic, seed-derived variation in
  at least one of accompaniment voicing, register, dynamics, or rhythm
  (§10 #10).

This module is the pure-function implementation. The engine calls
into `arrange_for_duration()` to decide:
- which form (8 / 16 / 32 bar) to use as the base,
- how many repetitions of the base form are allowed,
- what tempo (within the mood's range) hits the target.
"""

from __future__ import annotations

from dataclasses import dataclass

from saimc.compose.forms import PHRASE_SIZES, TEMPO_RANGE_BPM, ChordTemplate, get_template_for_form
from saimc.compose.score import PPQ

DURATION_TOLERANCE: float = 0.02
"""Per §8: ±2% of the spec's `duration_seconds`."""

MAX_REPEATS: int = 8
"""Per §10 #10: maximum 8 repetitions of any source section."""

BAR_DURATIONS_TICKS: dict[str, int] = {
    "4/4": 4 * PPQ,
    "3/4": 3 * PPQ,
    "6/8": 3 * PPQ,  # treated as 3 quarter-beat groups in Phase 1
    "2/4": 2 * PPQ,
    "5/4": 5 * PPQ,
    "7/8": 7 * PPQ // 2,  # 3.5 quarter beats, rounded down; engine handles
}


def bar_ticks(time_signature: str) -> int:
    """PPQ ticks in one measure for the given time signature.

    `7/8` is the only fractional case; the engine treats it as 3.5
    quarter beats (3 * PPQ + PPQ//2 = 1680 ticks at PPQ=480).
    """
    if time_signature == "7/8":
        return 3 * PPQ + PPQ // 2
    if time_signature not in BAR_DURATIONS_TICKS:
        raise ValueError(f"Unknown time signature: {time_signature!r}")
    return BAR_DURATIONS_TICKS[time_signature]


@dataclass(frozen=True)
class DurationArrangement:
    """The chosen form + repetition count + tempo for a piece."""

    form_bars: int
    template: ChordTemplate
    repetition_count: int
    total_bars: int
    tempo_bpm: float

    def __post_init__(self) -> None:
        if self.repetition_count < 1:
            raise ValueError(f"repetition_count must be >= 1; got {self.repetition_count}")
        if self.repetition_count > MAX_REPEATS:
            raise ValueError(
                f"repetition_count exceeds MAX_REPEATS ({MAX_REPEATS}); got {self.repetition_count}"
            )


class DurationUnfulfillableError(Exception):
    """Raised when no valid tempo + repetition combination hits the target."""


def arrange_for_duration(
    *,
    mood: str,
    target_duration_seconds: float,
    time_signature: str,
    base_form_bars: int | None = None,
    variant_index: int = 0,
) -> DurationArrangement:
    """Find (form, repetition_count, tempo) that fits `target_duration_seconds`.

    Strategy:
    1. Pick a base form (8/16/32 bars) — caller-supplied or auto.
    2. For each valid repetition_count in 1..MAX_REPEATS, scan every
       bpm in the mood's range at 1-BPM increments for the smallest
       |realised - target| / target. Return the first combination
       whose realised duration is within ±DURATION_TOLERANCE.
    3. If no combination works, raise `DurationUnfulfillableError`.
    """
    if base_form_bars is None:
        base_form_bars = _pick_base_form(mood, target_duration_seconds)
    if base_form_bars not in PHRASE_SIZES:
        raise ValueError(f"base_form_bars must be one of {PHRASE_SIZES}; got {base_form_bars}")

    template = get_template_for_form(mood, base_form_bars, variant_index=variant_index)
    ticks_per_bar = bar_ticks(time_signature)

    low_bpm, high_bpm = TEMPO_RANGE_BPM[mood]
    tolerance = DURATION_TOLERANCE

    best: tuple[float, int, float, float] | None = None  # (delta, rep, bpm, total_ticks)

    for repetition_count in range(1, MAX_REPEATS + 1):
        total_bars = base_form_bars * repetition_count
        total_ticks = total_bars * ticks_per_bar
        # The exact bpm that hits the target:
        target_bpm = (total_ticks / PPQ) * 60.0 / target_duration_seconds
        # Try the exact bpm if it's in range; else step by 0.5 bpm
        # within the mood range for fine-grained search (the spec
        # stores integer bpm but the engine may realise at fractional
        # tempo so the duration tolerance is actually achievable).
        bpm_candidates: list[float] = []
        if low_bpm <= target_bpm <= high_bpm:
            bpm_candidates.append(target_bpm)
        # 0.5-bpm steps within range.
        bpm_candidates.extend(low_bpm + 0.5 * i for i in range(int((high_bpm - low_bpm) * 2) + 1))
        for bpm in bpm_candidates:
            realised = _realised_seconds(total_ticks, bpm)
            delta = abs(realised - target_duration_seconds) / target_duration_seconds
            if delta <= tolerance:
                # Round to nearest 0.5 bpm for stability; the spec's
                # integer bpm field is honoured by the upstream parser,
                # but the engine's realised tempo is a finer value.
                chosen = round(bpm * 2) / 2
                return DurationArrangement(
                    form_bars=base_form_bars,
                    template=template,
                    repetition_count=repetition_count,
                    total_bars=total_bars,
                    tempo_bpm=chosen,
                )
            if best is None or (delta, repetition_count) < (best[0], best[1]):
                best = (delta, repetition_count, bpm, float(total_ticks))

    # If we got here, no in-tolerance arrangement exists. Surface
    # the closest one as part of the error for diagnostics.
    if best is None:
        raise DurationUnfulfillableError(
            f"no (form, repetition, tempo) combination fits {target_duration_seconds}s "
            f"for mood={mood!r} within ±{tolerance:.0%}; tried forms {PHRASE_SIZES}, "
            f"repetitions 1..{MAX_REPEATS}, bpm {low_bpm}..{high_bpm}"
        )
    closest_delta, closest_rep, closest_bpm, closest_ticks = best
    raise DurationUnfulfillableError(
        f"no (form, repetition, tempo) combination fits {target_duration_seconds}s "
        f"for mood={mood!r} within ±{tolerance:.0%}; closest was form={base_form_bars} "
        f"rep={closest_rep} bpm={closest_bpm} (realised "
        f"{_realised_seconds(int(closest_ticks), float(closest_bpm)):.1f}s, "
        f"delta {closest_delta:.1%})"
    )


def _pick_base_form(mood: str, target_duration_seconds: float) -> int:
    """Pick the smallest form whose max repetition can hit the target duration.

    For each candidate form, compute the maximum seconds achievable at
    MAX_REPEATS x the mood's slowest tempo in 4/4. If that maximum
    meets or exceeds the target, the form is a candidate. We then
    return the smallest such form (smallest repetition count in the
    downstream arrange_for_duration tends to be the cleanest output).
    """
    low_bpm, _high_bpm = TEMPO_RANGE_BPM[mood]
    for form in PHRASE_SIZES:
        # Max achievable seconds for this form: MAX_REPEATS at lowest tempo.
        max_seconds = (form * MAX_REPEATS * 4 / low_bpm) * 60.0
        if max_seconds >= target_duration_seconds:
            return form
    raise DurationUnfulfillableError(
        f"no Phase 1 form can reach {target_duration_seconds}s for mood={mood!r}; "
        f"max achievable is {(PHRASE_SIZES[-1] * MAX_REPEATS * 4 / low_bpm) * 60.0:.1f}s"
    )


def _realised_seconds(total_ticks: int, bpm: float) -> float:
    return (total_ticks / PPQ) * (60.0 / bpm)


def section_seed(spec_seed: int | None, section_index: int) -> int:
    """Derive a deterministic seed for one repeated section.

    Per §10 #10: repeated sections must receive deterministic,
    seed-derived variation. Combining the spec seed with the section
    index (0-based across all repeats) gives the per-section RNG seed.
    """
    if spec_seed is None:
        return section_index
    return spec_seed * 1_000_003 + section_index


__all__ = [
    "BAR_DURATIONS_TICKS",
    "DURATION_TOLERANCE",
    "MAX_REPEATS",
    "PPQ",
    "DurationArrangement",
    "DurationUnfulfillableError",
    "arrange_for_duration",
    "bar_ticks",
    "section_seed",
]
