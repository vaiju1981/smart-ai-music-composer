"""Velocity: the arc across a section, the swell inside a phrase.

Four pure readings of *where* a note sits — in its section, its phrase, its
bar — returning what that position does to its velocity. They are shared by
every voice (`melody`, `harmony`, `percussion`) and by the performance plan,
which is why they live below all of them and depend on none.
"""

from __future__ import annotations

import math
import random

from saimc.compose.duration import (
    DEFAULT_SECTION_ARC,
    SectionArc,
)


def velocity_arc(position: float) -> float:
    """A gentle dynamic arch: quiet entrances and exits, fuller middle.

    `position` is 0..1 across the section. The arc spans roughly 0.8x
    to 1.2x so phrases breathe without any note becoming extreme.
    """
    pos = min(1.0, max(0.0, position))
    return 0.8 + 0.4 * math.sin(math.pi * pos)


def shaped_velocity(
    *, base: int, position: float, tick: int, ticks_per_bar: int, rng_seed: int
) -> int:
    """Base velocity shaped by the section arch + downbeat accent + jitter.

    Deterministic: the jitter draws from a per-tick-seeded RNG so the
    same seed reproduces the exact same performance.
    """
    accent = 6 if tick % ticks_per_bar == 0 else 0
    jitter = random.Random(rng_seed * 31 + tick).randint(-4, 4)
    shaped = base * velocity_arc(position) + accent + jitter
    return max(1, min(127, round(shaped)))


def section_velocity_scale(
    section_idx: int,
    repetition_count: int,
    arc: SectionArc = DEFAULT_SECTION_ARC,
) -> float:
    """Terraced dynamics: the arc is stepped per section, not continuous.

    The opening sits back, the penultimate section peaks, and the
    final one settles slightly for the cadence home. A single-section
    piece has nowhere to move and plays at full. The four terraces are
    the plan's, so a critic can move a section's weight without moving
    the piece's overall level.
    """
    if repetition_count < 2:
        return 1.0
    if section_idx == 0:
        return arc.energy_opening
    if section_idx == repetition_count - 1:
        return arc.energy_final
    if section_idx == repetition_count - 2:
        return arc.energy_peak
    return arc.energy_middle


def phrase_swell(position_in_phrase: float) -> float:
    """One rise-and-fall per phrase, for the controller layer.

    `position_in_phrase` is 0..1 across a `PHRASE_BARS`-bar phrase; the
    swell spans roughly 0.85x to 1.0x so each phrase breathes once.
    """
    pos = min(1.0, max(0.0, position_in_phrase))
    return 0.85 + 0.15 * math.sin(math.pi * pos)


__all__ = [
    "phrase_swell",
    "section_velocity_scale",
    "shaped_velocity",
    "velocity_arc",
]
