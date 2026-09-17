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
from typing import Final

from saimc.compose.forms import (
    PHRASE_SIZES,
    ChordTemplate,
    get_mood_profile,
    get_template_for_form,
)
from saimc.compose.score import PPQ

DURATION_TOLERANCE: float = 0.02
"""Per §8: ±2% of the spec's `duration_seconds`."""

MAX_REPEATS: int = 8
"""Per §10 #10: maximum 8 repetitions of any source section."""

ARRANGEMENT_ARC_MIN_REPS: int = 3
"""Long pieces (>= this many repetitions) get the full arc: a
bass-alone intro carved from the first section, a modulation on the
final repetition, and (for the drum set) a percussion rest section."""

INTRO_BARS: int = 2
"""Bars of bass alone that open an arc'd piece (carved from section 0,
so the total bar count and duration math are unchanged)."""

RITARDANDO_FACTOR: float = 0.85
"""The outro slows to this fraction of the piece's tempo. A coda'd
piece slows across its whole coda; a long arc'd piece slows across
its final `RITARDANDO_BARS`. The duration math includes the slowdown
in both branches, so a piece that arrives at its target with a
ritardando lands there honestly."""

RITARDANDO_BARS: int = 2
"""Bars of ritardando on a long (arc'd) piece without a coda: the
final cadence itself eases in."""

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
    """The chosen form + repetition count + tempo for a piece.

    `coda_bars` is the optional shorter tail added when no clean
    (repetition_count, tempo) hits the target. Per §10 #1, a coda is
    "a shorter coda made only of complete measures" — appended after
    the final full-form repetition. `coda_bars=0` means no coda.

    `intro_bars` is the bass-alone opening carved from the first
    section on long pieces (>= `ARRANGEMENT_ARC_MIN_REPS` repetitions);
    it does not add bars. `ritardando_factor` below 1.0 slows the coda
    (the outro) for the final cadence; 1.0 keeps a constant tempo.
    """

    form_bars: int
    template: ChordTemplate
    repetition_count: int
    total_bars: int
    tempo_bpm: float
    coda_bars: int = 0
    intro_bars: int = 0
    ritardando_factor: float = 1.0

    def __post_init__(self) -> None:
        if self.repetition_count < 1:
            raise ValueError(f"repetition_count must be >= 1; got {self.repetition_count}")
        if self.repetition_count > MAX_REPEATS:
            raise ValueError(
                f"repetition_count exceeds MAX_REPEATS ({MAX_REPEATS}); got {self.repetition_count}"
            )
        if self.coda_bars < 0:
            raise ValueError(f"coda_bars must be >= 0; got {self.coda_bars}")
        # Coda must be smaller than the form itself; if it were the same
        # size, the caller should bump repetition_count instead.
        if self.coda_bars > 0 and self.coda_bars >= self.form_bars:
            raise ValueError(f"coda_bars ({self.coda_bars}) must be < form_bars ({self.form_bars})")
        if not 0 <= self.intro_bars < self.form_bars:
            raise ValueError(
                f"intro_bars ({self.intro_bars}) must be in [0, form_bars) "
                f"({self.form_bars})"
            )
        if not 0.0 < self.ritardando_factor <= 1.0:
            raise ValueError(
                f"ritardando_factor must be in (0, 1]; got {self.ritardando_factor}"
            )

    @property
    def total_bars_with_coda(self) -> int:
        return self.total_bars + self.coda_bars


class DurationUnfulfillableError(Exception):
    """Raised when no valid tempo + repetition combination hits the target."""


@dataclass(frozen=True)
class ArrangementKnobs:
    """What the arrangement layer reads from a plan.

    A struct rather than seven parameters because the duration search
    threads them through three functions. It lives here rather than in
    `plan.py` because the plan imports this module for its own defaults,
    so the dependency cannot also run the other way; `CompositionPlan`
    builds one with its `arrangement_knobs()` method.

    The defaults are today's module constants, which is what keeps every
    caller that does not care about the plan — and `arrange_for_duration`
    called with no `knobs` at all — producing exactly the arrangement it
    produced before.
    """

    form_sizes: tuple[int, ...] = PHRASE_SIZES
    max_repeats: int = MAX_REPEATS
    duration_tolerance: float = DURATION_TOLERANCE
    arc_min_reps: int = ARRANGEMENT_ARC_MIN_REPS
    intro_bars: int = INTRO_BARS
    ritardando_factor: float = RITARDANDO_FACTOR
    ritardando_bars: int = RITARDANDO_BARS


DEFAULT_ARRANGEMENT_KNOBS: Final[ArrangementKnobs] = ArrangementKnobs()
"""Today's values, as the object every default reads through."""


def arrange_for_duration(
    *,
    mood: str,
    target_duration_seconds: float,
    time_signature: str,
    base_form_bars: int | None = None,
    variant_index: int = 0,
    tempo_bpm: float | None = None,
    knobs: ArrangementKnobs = DEFAULT_ARRANGEMENT_KNOBS,
) -> DurationArrangement:
    """Find (form, repetition_count, tempo) that fits `target_duration_seconds`.

    Strategy:
    1. Pick a base form (8/16/32 bars) - caller-supplied or auto.
    2. For each valid repetition_count in 1..`knobs.max_repeats`, scan every
       bpm in the mood's range at 0.5-BPM increments. Return the
       first combination whose realised duration is within
       ±DURATION_TOLERANCE. The first fitting repetition is the one
       with the fewest bars, which lands on the slowest tempo that
       reaches the target — a deliberate bias toward the calmer end
       of the mood's range.
    3. If no clean arrangement fits, try adding a coda (a smaller
       tail of complete measures) per §10 #1. The coda is itself
       a sub-form: a positive multiple of the form's bar count
       smaller than the form itself (typically form_bars // 2).
    4. If even a coda doesn't work, raise DurationUnfulfillableError.

    A spec-pinned `tempo_bpm` is honoured whenever a (form,
    repetition) combination exists whose realised duration at that
    exact bpm lands within tolerance. When none does — an exact tempo
    makes the realised duration a step function of the bar count, and
    most (tempo, duration) pairs simply have no step within ±2% — the
    duration promise (the release gate) outranks the tempo request and
    the engine derives the tempo from the mood's range instead,
    reporting the chosen value in the arrangement.
    """
    if tempo_bpm is None:
        return _arrange_at_tempo(
            mood=mood,
            target_duration_seconds=target_duration_seconds,
            time_signature=time_signature,
            base_form_bars=base_form_bars,
            variant_index=variant_index,
            tempo_bpm=None,
            knobs=knobs,
        )
    try:
        return _arrange_at_tempo(
            mood=mood,
            target_duration_seconds=target_duration_seconds,
            time_signature=time_signature,
            base_form_bars=base_form_bars,
            variant_index=variant_index,
            tempo_bpm=tempo_bpm,
            knobs=knobs,
        )
    except DurationUnfulfillableError:
        return _arrange_at_tempo(
            mood=mood,
            target_duration_seconds=target_duration_seconds,
            time_signature=time_signature,
            base_form_bars=base_form_bars,
            variant_index=variant_index,
            tempo_bpm=None,
            knobs=knobs,
        )


def _arrange_at_tempo(
    *,
    mood: str,
    target_duration_seconds: float,
    time_signature: str,
    base_form_bars: int | None,
    variant_index: int,
    tempo_bpm: float | None,
    knobs: ArrangementKnobs,
) -> DurationArrangement:
    """The duration search under one tempo rule: pinned bpm or derived.

    A `tempo_bpm` constraint (from the spec) pins the tempo: only
    (form, repetition) combinations whose realised duration at that
    exact bpm lands within tolerance are eligible.
    """
    if base_form_bars is None:
        base_form_bars = _pick_base_form(mood, target_duration_seconds, time_signature, knobs)
    if base_form_bars not in knobs.form_sizes:
        raise ValueError(
            f"base_form_bars must be one of {knobs.form_sizes}; got {base_form_bars}"
        )

    template = get_template_for_form(mood, base_form_bars, variant_index=variant_index)
    ticks_per_bar = bar_ticks(time_signature)

    low_bpm, high_bpm = get_mood_profile(mood).tempo_range_bpm
    tolerance = knobs.duration_tolerance
    if tempo_bpm is not None and not low_bpm <= tempo_bpm <= high_bpm:
        raise DurationUnfulfillableError(
            f"requested tempo {tempo_bpm}bpm is outside the {mood!r} range {low_bpm}-{high_bpm}bpm"
        )

    best_no_coda: tuple[float, int, float, float] | None = None

    # A spec-pinned tempo is honoured at a constant tempo: the tempo
    # promise outranks the decorative arc, and the slowdown's extra
    # seconds could make a pinned tempo unfulfillable.
    may_ritardando = tempo_bpm is None

    for repetition_count in range(1, knobs.max_repeats + 1):
        total_bars = base_form_bars * repetition_count
        total_ticks = total_bars * ticks_per_bar
        bpm_candidates: list[float] = []
        if tempo_bpm is not None:
            # Spec-pinned tempo: that exact bpm is the only candidate.
            bpm_candidates.append(tempo_bpm)
        else:
            target_bpm = (total_ticks / PPQ) * 60.0 / target_duration_seconds
            if low_bpm <= target_bpm <= high_bpm:
                bpm_candidates.append(target_bpm)
        if tempo_bpm is None:
            bpm_candidates.extend(
                low_bpm + 0.5 * i for i in range(int((high_bpm - low_bpm) * 2) + 1)
            )
        for bpm in bpm_candidates:
            long_piece = repetition_count >= knobs.arc_min_reps
            # A long piece eases in over its final cadence bars; the
            # slowed seconds are part of the math so the ±2% promise
            # holds for the piece the listener actually hears. Short
            # pieces sit too close to the tempo range's edges for the
            # slowdown to be compensable there.
            realised = (
                _realised_rit_seconds(
                    total_ticks, ticks_per_bar, bpm, knobs.ritardando_factor, knobs
                )
                if long_piece and may_ritardando
                else _realised_seconds(total_ticks, bpm)
            )
            delta = abs(realised - target_duration_seconds) / target_duration_seconds
            if delta <= tolerance:
                chosen = round(bpm * 2) / 2
                return DurationArrangement(
                    form_bars=base_form_bars,
                    template=template,
                    repetition_count=repetition_count,
                    total_bars=total_bars,
                    tempo_bpm=chosen,
                    intro_bars=knobs.intro_bars if long_piece else 0,
                    ritardando_factor=knobs.ritardando_factor
                    if long_piece and may_ritardando
                    else 1.0,
                )
            if best_no_coda is None or (delta, repetition_count) < (
                best_no_coda[0],
                best_no_coda[1],
            ):
                best_no_coda = (delta, repetition_count, bpm, float(total_ticks))

    # Try with a coda. The coda is the smallest legal half of the form
    # (must be < form_bars, in whole-bar units). For 8-bar forms the
    # coda is 4 bars; for 16 it's 8; for 32 it's 16.
    coda_bars = base_form_bars // 2
    if coda_bars >= base_form_bars:
        coda_bars = base_form_bars - 4 if base_form_bars >= 4 else 0
    if coda_bars >= 1:
        for repetition_count in range(1, knobs.max_repeats + 1):
            total_bars_no_coda = base_form_bars * repetition_count
            total_ticks_no_coda = total_bars_no_coda * ticks_per_bar
            coda_ticks = coda_bars * ticks_per_bar
            total_ticks = total_ticks_no_coda + coda_ticks
            target_bpm = (total_ticks / PPQ) * 60.0 / target_duration_seconds
            coda_bpm_candidates: list[float] = []
            if tempo_bpm is not None:
                coda_bpm_candidates.append(tempo_bpm)
            else:
                if low_bpm <= target_bpm <= high_bpm:
                    coda_bpm_candidates.append(target_bpm)
                coda_bpm_candidates.extend(
                    low_bpm + 0.5 * i for i in range(int((high_bpm - low_bpm) * 2) + 1)
                )
            for bpm in coda_bpm_candidates:
                # The coda carries the outro ritardando: its seconds are
                # computed at the slowed tempo so the ±2% promise holds
                # for the piece the listener actually hears.
                realised = (
                    _realised_coda_seconds(
                        total_ticks_no_coda, coda_ticks, bpm, knobs.ritardando_factor
                    )
                    if may_ritardando
                    else _realised_seconds(total_ticks_no_coda + coda_ticks, bpm)
                )
                delta = abs(realised - target_duration_seconds) / target_duration_seconds
                if delta <= tolerance:
                    chosen = round(bpm * 2) / 2
                    return DurationArrangement(
                        form_bars=base_form_bars,
                        template=template,
                        repetition_count=repetition_count,
                        total_bars=total_bars_no_coda,
                        tempo_bpm=chosen,
                        coda_bars=coda_bars,
                        intro_bars=knobs.intro_bars
                        if repetition_count >= knobs.arc_min_reps
                        else 0,
                        ritardando_factor=knobs.ritardando_factor
                        if may_ritardando
                        else 1.0,
                    )

    # No in-tolerance arrangement exists — with or without a coda. Per
    # §10 #1 the policy must fail loudly rather than silently return an
    # arrangement that misses the target.
    # Surface the closest no-coda arrangement as part of the error.
    if best_no_coda is None:
        raise DurationUnfulfillableError(
            f"no (form, repetition, tempo) combination fits {target_duration_seconds}s "
            f"for mood={mood!r} within ±{tolerance:.0%}; tried forms {knobs.form_sizes}, "
            f"repetitions 1..{knobs.max_repeats}, bpm {low_bpm}..{high_bpm}"
        )
    closest_delta, closest_rep, closest_bpm, closest_ticks = best_no_coda
    raise DurationUnfulfillableError(
        f"no (form, repetition, tempo, coda) combination fits {target_duration_seconds}s "
        f"for mood={mood!r} within ±{tolerance:.0%}; closest was form={base_form_bars} "
        f"rep={closest_rep} bpm={closest_bpm} (realised "
        f"{_realised_seconds(int(closest_ticks), float(closest_bpm)):.1f}s, "
        f"delta {closest_delta:.1%})"
    )


def _pick_base_form(
    mood: str,
    target_duration_seconds: float,
    time_signature: str,
    knobs: ArrangementKnobs,
) -> int:
    """Pick the smallest form whose max repetition can hit the target duration.

    For each candidate form, compute the maximum seconds achievable at
    `knobs.max_repeats` x the mood's slowest tempo in the spec's time
    signature. If that maximum meets or exceeds the target, the form
    is a candidate. We then return the smallest such form (smallest
    repetition count in the downstream arrange_for_duration tends to
    be the cleanest output).
    """
    low_bpm, _high_bpm = get_mood_profile(mood).tempo_range_bpm
    beats_per_bar = bar_ticks(time_signature) / PPQ
    for form in knobs.form_sizes:
        # Max achievable seconds for this form: max_repeats at lowest tempo.
        max_seconds = (form * knobs.max_repeats * beats_per_bar / low_bpm) * 60.0
        if max_seconds >= target_duration_seconds:
            return form
    raise DurationUnfulfillableError(
        f"no Phase 1 form can reach {target_duration_seconds}s for mood={mood!r} "
        f"in {time_signature}; max achievable is "
        f"{(knobs.form_sizes[-1] * knobs.max_repeats * beats_per_bar / low_bpm) * 60.0:.1f}s"
    )


def _realised_seconds(total_ticks: int, bpm: float) -> float:
    return (total_ticks / PPQ) * (60.0 / bpm)


def _realised_rit_seconds(
    total_ticks: int,
    ticks_per_bar: int,
    bpm: float,
    ritardando_factor: float,
    knobs: ArrangementKnobs,
) -> float:
    """Realized seconds for a piece whose final cadence bars slow down."""
    rit_ticks = min(knobs.ritardando_bars * ticks_per_bar, total_ticks)
    return _realised_coda_seconds(
        total_ticks - rit_ticks, rit_ticks, bpm, ritardando_factor
    )


def _realised_coda_seconds(
    body_ticks: int, coda_ticks: int, bpm: float, ritardando_factor: float
) -> float:
    """Realized seconds for a coda'd piece whose outro slows down."""
    return _realised_seconds(body_ticks, bpm) + _realised_seconds(
        coda_ticks, bpm * ritardando_factor
    )


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
    "ARRANGEMENT_ARC_MIN_REPS",
    "BAR_DURATIONS_TICKS",
    "DEFAULT_ARRANGEMENT_KNOBS",
    "DURATION_TOLERANCE",
    "INTRO_BARS",
    "MAX_REPEATS",
    "PPQ",
    "RITARDANDO_BARS",
    "RITARDANDO_FACTOR",
    "ArrangementKnobs",
    "DurationArrangement",
    "DurationUnfulfillableError",
    "arrange_for_duration",
    "bar_ticks",
    "section_seed",
]
