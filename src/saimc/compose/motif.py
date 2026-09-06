"""Motif-driven melody generation.

First-year composition craft: instead of re-rolling an arpeggio walk
every bar, each section gets one small motif — 2 to 8 cells, each a
step through the chord tones plus a duration — and every bar derives
its line from that motif through one of the classic operations:
repetition, transposition, sequence, inversion, truncation, ornament.
The RNG chooses which operation a bar uses; the pitches themselves are
always the motif's shape carried onto that bar's chord, so a section
sounds like it is *developing* one idea instead of inventing a new
figure every bar.

Steps live in chord-tone index space (movement across the chord's
tones, wrapping), so the same motif lands correctly on every chord of
the template without knowing any key or mode.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from saimc.compose.score import PPQ

__all__ = [
    "BarSlot",
    "Motif",
    "MotifCell",
    "MotifVariant",
    "apply_rhythm",
    "generate_motif",
    "vary_motif",
]


@dataclass(frozen=True)
class MotifCell:
    """One melodic event: a chord-tone move plus its duration.

    `step` is the movement in chord-tone index space from the previous
    cell (the first cell's step is always 0 — it starts on the bar's
    anchor tone). `length_ticks` is the notated duration.
    """

    step: int
    length_ticks: int


Motif = tuple[MotifCell, ...]


@dataclass(frozen=True)
class MotifVariant:
    """One bar's derived material: which motif to play and how.

    `anchor_offset` shifts the bar's starting chord tone (transposition
    and sequence shift by a tone; the other operations stay home).
    `repeat` asks the renderer to keep playing the motif from the top
    — anchor advancing one tone per cycle — until the bar is full
    (the "sequence" operation).
    """

    motif: Motif
    anchor_offset: int = 0
    repeat: bool = False


# Weights for the bar-level operation draw (repetition first so a motif
# establishes itself before it is worked).
_REPETITION_WEIGHT = 0.35
_TRANSPOSITION_WEIGHT = 0.20
_SEQUENCE_WEIGHT = 0.10
_INVERSION_WEIGHT = 0.15
_TRUNCATION_WEIGHT = 0.10
# Ornament takes the remaining weight.

_STEP_CHOICES: tuple[int, ...] = (-2, -1, 0, 1, 2)
_STEP_WEIGHTS: tuple[float, ...] = (10, 30, 15, 35, 10)


def generate_motif(rng: random.Random, *, bar_ticks: int) -> Motif:
    """Generate the section's motif: 2-8 cells that fit inside one bar.

    Rhythm is drawn per cell from quarter/eighth so a motif can fill
    the bar or leave a natural rest at its end; steps are small walks
    through the chord tones so the line stays singable.
    """
    target_count = rng.randint(2, 8)
    cells: list[MotifCell] = []
    used = 0
    for i in range(target_count):
        length = rng.choice((PPQ, PPQ // 2))
        if used + length > bar_ticks:
            break
        step = 0 if i == 0 else rng.choices(_STEP_CHOICES, weights=_STEP_WEIGHTS, k=1)[0]
        cells.append(MotifCell(step=step, length_ticks=length))
        used += length
    if len(cells) < 2:
        # A pathological draw (a tiny bar, or every draw wanted a full
        # quarter) still owes the section a two-cell motif.
        while len(cells) < 2 and used + PPQ // 2 <= bar_ticks:
            length = PPQ if used + PPQ <= bar_ticks else PPQ // 2
            cells.append(MotifCell(step=rng.choice((-1, 1)), length_ticks=length))
            used += length
    return tuple(cells)


def _invert(motif: Motif) -> Motif:
    """Mirror the contour: every step negated, rhythm unchanged."""
    return tuple(
        MotifCell(step=0 if i == 0 else -cell.step, length_ticks=cell.length_ticks)
        for i, cell in enumerate(motif)
    )


def _truncate(motif: Motif, rng: random.Random) -> Motif:
    """Drop cells from the end, keeping at least one — leaves a rest."""
    keep = rng.randint(1, max(1, len(motif) - 1))
    return motif[:keep]


def _ornament(motif: Motif, rng: random.Random) -> Motif:
    """Split one longer cell into a neighbour-tone pair of shorter notes."""
    splittable = [i for i, cell in enumerate(motif) if cell.length_ticks >= PPQ // 2]
    if not splittable:
        return motif
    index = rng.choice(splittable)
    cell = motif[index]
    half = cell.length_ticks // 2
    neighbour = rng.choice((-1, 1))
    ornamented = list(motif)
    ornamented[index : index + 1] = (
        MotifCell(step=cell.step, length_ticks=half),
        MotifCell(step=neighbour, length_ticks=half),
    )
    return tuple(ornamented)


def vary_motif(motif: Motif, rng: random.Random) -> MotifVariant:
    """Derive one bar's material from the motif via a single operation.

    The draw picks the operation, never raw pitches: repetition plays
    the motif as it stands, transposition starts one chord tone higher,
    sequence repeats it through the bar with the anchor advancing,
    inversion mirrors it, truncation cuts it short (a rest), ornament
    decorates one cell.
    """
    roll = rng.random()
    if roll < _REPETITION_WEIGHT:
        return MotifVariant(motif=motif)
    if roll < _REPETITION_WEIGHT + _TRANSPOSITION_WEIGHT:
        return MotifVariant(motif=motif, anchor_offset=1)
    if roll < _REPETITION_WEIGHT + _TRANSPOSITION_WEIGHT + _SEQUENCE_WEIGHT:
        return MotifVariant(motif=motif, repeat=True)
    if (
        roll
        < _REPETITION_WEIGHT + _TRANSPOSITION_WEIGHT + _SEQUENCE_WEIGHT + _INVERSION_WEIGHT
    ):
        return MotifVariant(motif=_invert(motif))
    if (
        roll
        < _REPETITION_WEIGHT
        + _TRANSPOSITION_WEIGHT
        + _SEQUENCE_WEIGHT
        + _INVERSION_WEIGHT
        + _TRUNCATION_WEIGHT
    ):
        return MotifVariant(motif=_truncate(motif, rng))
    return MotifVariant(motif=_ornament(motif, rng))


# --- Bar-level rhythm vocabulary -------------------------------------------
#
# A motif cell is only ever a quarter or an eighth, so every bar
# inherits the motif's straight rhythm. The rhythm library re-voices a
# bar's slots with the mood's idiomatic figures before the notes are
# resolved: dotted long-short pairs, 16th subdivisions, and ties that
# hold one pitch across a beat (engraved as a tie, played as one note).

BarSlot = tuple[int, int, int, int]
"""One melody event slot: (bar_offset, duration_ticks, tone_index, tie)."""

RHYTHM_WEIGHTS: dict[str, dict[str, float]] = {
    "electrifying": {"straight": 0.30, "dotted": 0.25, "sixteenths": 0.30, "tie": 0.15},
    "calming": {"straight": 0.35, "dotted": 0.30, "tie": 0.35},
    "sleep": {"straight": 0.25, "dotted": 0.35, "tie": 0.40},
}
# Moods without their own profile fall back to a gentle mix.
_DEFAULT_RHYTHM_WEIGHTS: dict[str, float] = {
    "straight": 0.50,
    "dotted": 0.25,
    "tie": 0.25,
}


def _reflow(slots: list[list[int]]) -> list[BarSlot]:
    """Rebuild contiguous offsets after an op changed slot durations."""
    offset = 0
    out: list[BarSlot] = []
    for _start, duration, tone, tie in slots:
        out.append((offset, duration, tone, tie))
        offset += duration
    return out


def _op_dotted(slots: list[list[int]]) -> list[BarSlot]:
    """Long-short: two quarters become a dotted quarter + eighth."""
    for i in range(len(slots) - 1):
        if slots[i][1] == PPQ and slots[i + 1][1] == PPQ:
            slots[i][1] = PPQ + PPQ // 2
            slots[i + 1][1] = PPQ // 2
            break
    return _reflow(slots)


def _op_tie(slots: list[list[int]]) -> list[BarSlot]:
    """Hold one pitch across a beat: mark the first of two same-tone
    neighbours tied (both noteheads stay; the performance layer plays
    them as one sound)."""
    for i in range(len(slots) - 1):
        if slots[i][2] == slots[i + 1][2]:
            slots[i][3] = 1
            break
    return _reflow(slots)


def apply_rhythm(
    slots: list[tuple[int, int, int]],
    *,
    rng: random.Random,
    mood: str,
) -> list[BarSlot]:
    """Re-voice a bar's motif slots through the mood's rhythm library.

    The draw picks one operation for the whole bar — a bar speaks with
    one rhythmic idea, not a new figure on every beat. `straight` keeps
    the motif's own durations; `dotted` renders the long-short pair;
    `sixteenths` subdivides one eighth; `tie` holds a repeated pitch
    across its beat boundary.
    """
    weights = RHYTHM_WEIGHTS.get(mood, _DEFAULT_RHYTHM_WEIGHTS)
    operation = rng.choices(tuple(weights), weights=tuple(weights.values()), k=1)[0]
    if operation == "straight" or len(slots) < 2:
        return [(offset, duration, tone, False) for offset, duration, tone in slots]
    mutable = [[*slot, 0] for slot in slots]
    if operation == "dotted":
        return _op_dotted(mutable)
    if operation == "sixteenths":
        subdividable = [i for i, s in enumerate(mutable) if s[1] == PPQ // 2]
        if subdividable:
            i = rng.choice(subdividable)
            half = mutable[i][1] // 2
            mutable[i : i + 1] = [
                [mutable[i][0], half, mutable[i][2], 0],
                [0, half, mutable[i][2], 0],
            ]
        return _reflow(mutable)
    return _op_tie(mutable)
