"""Motif-driven melody generation.

First-year composition craft: instead of re-rolling an arpeggio walk
every bar, each section gets one small motif — 2 to 8 cells, each a
step plus a duration — and every bar derives its line from that motif
through one of the classic operations: repetition, transposition,
sequence, inversion, truncation, ornament. The RNG chooses which
operation a bar uses; the pitches themselves are always the motif's
shape carried onto that bar's chord, so a section sounds like it is
*developing* one idea instead of inventing a new figure every bar.

Steps live in scale-degree space, one degree per move, so a ±1 step is
a semitone or a whole tone — the conjunct motion a melody is made of.
The renderer spells those degrees in the bar's own scale, so the same
motif lands correctly on every chord of the template while staying
inside the key.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass
from dataclasses import field as dataclass_field

from saimc.compose.score import PPQ
from saimc.instruments import LINE_BAND_SEMITONES

__all__ = [
    "BASS_FIGURES",
    "CHORD_TONE_DEGREES",
    "DEFAULT_APEX_POSITION",
    "DEFAULT_BASS_FIGURES",
    "DEFAULT_MELODY_SHAPE",
    "DEFAULT_RHYTHM_WEIGHTS",
    "DEFAULT_TIE_PROBABILITY",
    "LEAP_DEGREES",
    "MAX_MOTIF_SPAN_DEGREES",
    "MOTIF_OPERATION_WEIGHTS",
    "PLAIN_BASS_FIGURE",
    "RHYTHM_WEIGHTS",
    "STEP_CHOICES",
    "STEP_WEIGHTS",
    "TIE_PROBABILITY",
    "BarSlot",
    "BassFigure",
    "MelodyShape",
    "Motif",
    "MotifCell",
    "MotifVariant",
    "apply_rhythm",
    "draw_bass_figures",
    "generate_motif",
    "recover_leaps",
    "vary_motif",
]


@dataclass(frozen=True)
class MotifCell:
    """One melodic event: a scale-degree move plus its duration.

    `step` is the movement in scale degrees from the previous cell (the
    first cell's step is always 0 — it starts on the bar's anchor tone).
    `length_ticks` is the notated duration.
    """

    step: int
    length_ticks: int


Motif = tuple[MotifCell, ...]

# A move of this many scale degrees or more is a leap: three degrees is a
# fourth in every diatonic scale, which is where a listener hears a gap
# that wants closing. Two degrees is a third — a skip, not a leap.
LEAP_DEGREES: int = 3

# One chord tone is two scale degrees — a triad's tones are its scale's
# degrees 0, 2 and 4, and a seventh chord adds degree 6. This is the step
# that keeps the downbeat on a chord tone, which the passing-tone licence
# requires of every note the harmony rests on.
CHORD_TONE_DEGREES: int = 2


@dataclass(frozen=True)
class MotifVariant:
    """One bar's derived material: which motif to play and how.

    `anchor_offset` starts the bar's walk this many scale degrees above
    the drawn anchor. It is always even, so the downbeat stays a chord
    tone — the anchor's own tone plus one chord tone is a third higher,
    which is what "transposition" means here. `repeat` asks the renderer
    to keep playing the motif from the top — the anchor advancing one
    chord tone per cycle — until the bar is full (the "sequence"
    operation).
    """

    motif: Motif
    anchor_offset: int = 0
    repeat: bool = False


# Weights for the bar-level operation draw (repetition first so a motif
# establishes itself before it is worked). These do not sum to 1: the
# ornament takes whatever weight is left over, which is what keeps the
# table overridable without the caller having to renormalise it.
MOTIF_OPERATION_WEIGHTS: tuple[tuple[str, float], ...] = (
    ("repeat", 0.35),
    ("transpose", 0.20),
    ("sequence", 0.10),
    ("invert", 0.15),
    ("truncate", 0.10),
)

# Step vocabulary, in degrees. Steps dominate (56%) so the line is
# conjunct — `step_ratio` asks for 45% and a melody that is mostly skips
# is a bug, not a style. Thirds (20%) keep the line moving without
# leaping, and the fourths and wider (6%) are what make a leap worth
# answering; a vocabulary without them scores as flat in both directions
# (`interval_diversity` wants six distinct intervals). Repeats (6%) are
# the pedal a phrase rests on, and are capped because `repeat_ratio`
# cannot exceed 25%.
STEP_CHOICES: tuple[int, ...] = (-4, -3, -2, -1, 0, 1, 2, 3, 4)
STEP_WEIGHTS: tuple[float, ...] = (3, 5, 10, 28, 6, 28, 10, 5, 3)
# How many scale degrees a motif may span. The renderer places each bar
# in a tessitura band; a motif wider than this cannot fit inside one, and
# would be folded — which is heard as a glitch, not as a phrase.
MAX_MOTIF_SPAN_DEGREES: int = 8


# --- Bar-level rhythm vocabulary -------------------------------------------
#
# A motif cell is only ever a quarter or an eighth, so every bar
# inherits the motif's straight rhythm. The rhythm library re-voices a
# bar's slots with the mood's idiomatic figures before the notes are
# resolved: dotted long-short pairs, 16th subdivisions, and ties that
# hold one pitch across a beat (engraved as a tie, played as one note).

BarSlot = tuple[int, int, int, int]
"""One melody event slot: (bar_offset, duration_ticks, scale_degree, tie)."""

RHYTHM_WEIGHTS: dict[str, dict[str, float]] = {
    "electrifying": {"straight": 0.30, "dotted": 0.25, "sixteenths": 0.30, "tie": 0.15},
    "calming": {"straight": 0.35, "dotted": 0.30, "tie": 0.35},
    "sleep": {"straight": 0.25, "dotted": 0.35, "tie": 0.40},
}
# Moods without their own profile fall back to a gentle mix.
DEFAULT_RHYTHM_WEIGHTS: dict[str, float] = {
    "straight": 0.50,
    "dotted": 0.25,
    "tie": 0.25,
}


# --- The melody layer, as one value ----------------------------------------

# Cross-bar ties: when two adjacent melody notes share a pitch at the bar
# line, the first is marked tied with this probability (calmer moods hold
# more; electrifying keeps its attacks).
TIE_PROBABILITY: dict[str, float] = {
    "electrifying": 0.18,
    "calming": 0.28,
    "sleep": 0.35,
}
# A mood outside the table holds ties as often as this.
DEFAULT_TIE_PROBABILITY: float = 0.25

# Where the section's melodic apex sits, as a fraction of its bars. Near
# the end rather than at it: the peak arrives, and the line still has the
# closing gesture after it.
DEFAULT_APEX_POSITION: float = 0.6


@dataclass(frozen=True)
class MelodyShape:
    """Every decision the melody layer makes, as one value.

    The vocabulary a line is drawn from (the step table, the motif's span,
    what counts as a leap, how far apart the chord's own tones sit, which
    operation restates the motif, the mood's rhythmic figures), the two
    facts about a phrase's shape that are not draws (where its apex sits,
    how readily a repeated pitch is tied across a bar line), and the width
    of the register window the line is written inside.

    One value rather than ten parameters because they travel together: a
    bar is drawn, varied, snapped and placed by the same vocabulary, and
    the helpers that do it take no plan — the same reason `duration.py`
    has an `ArrangementKnobs` and `SectionArc` exists for the sections.
    Bundling them keeps one tramp argument on the call stack instead of
    six, and keeps `plan.melody_shape()` a mechanical projection of the
    plan's melody fields.

    There is no `__post_init__` here. A shape a caller builds by hand is
    the caller's; a shape that arrives *as a plan* has already been
    checked by `CompositionPlan.__post_init__`, which is where the bounds
    live. Duplicating them would give two places to change one rule.
    """

    step_choices: tuple[int, ...] = STEP_CHOICES
    """The interval vocabulary, in scale degrees, walked by `_draw_step`."""
    step_weights: tuple[float, ...] = STEP_WEIGHTS
    """One weight per `step_choices` entry. Steps dominate; leaping is rare."""
    max_motif_span_degrees: int = MAX_MOTIF_SPAN_DEGREES
    """How far a motif may wander from where it started before it is folded."""
    leap_degrees: int = LEAP_DEGREES
    """The interval, in degrees, from which a step counts as a leap."""
    chord_tone_degrees: int = CHORD_TONE_DEGREES
    """The interval, in degrees, that transposition moves the anchor by."""
    motif_operation_weights: tuple[tuple[str, float], ...] = MOTIF_OPERATION_WEIGHTS
    """Weights for the bar-level operation draw, in draw order. The weight
    left over is the ornament's, so these need not sum to 1."""
    rhythm_weights: tuple[tuple[str, float], ...] = dataclass_field(
        default_factory=lambda: tuple(DEFAULT_RHYTHM_WEIGHTS.items())
    )
    """The mood's rhythmic figures and how often each is chosen.

    A named-weight table rather than the mood it was looked up by: the
    plan carries the resolved figures, so `apply_rhythm` no longer needs
    to know which mood it is dressing.
    """
    tie_probability: float = DEFAULT_TIE_PROBABILITY
    """How readily a repeated pitch is tied across a bar line."""
    apex_position: float = DEFAULT_APEX_POSITION
    """Where the apex bar sits, as a fraction of the section's bars."""
    line_band_semitones: int = LINE_BAND_SEMITONES
    """The register window a melody line is written inside."""


DEFAULT_MELODY_SHAPE: MelodyShape = MelodyShape()
"""The melody layer's defaults: today's vocabulary and phrase shape."""


def generate_motif(
    rng: random.Random, *, bar_ticks: int, shape: MelodyShape = DEFAULT_MELODY_SHAPE
) -> Motif:
    """Generate the section's motif: 2-8 cells that fit inside one bar.

    Rhythm is drawn per cell from quarter/eighth so a motif can fill
    the bar or leave a natural rest at its end; steps are small walks in
    scale degrees, and the walk is kept inside
    `shape.max_motif_span_degrees` of where it started — a motif that
    wanders further than that is a scale exercise, and the bar it lands
    in cannot hold it.
    """
    target_count = rng.randint(2, 8)
    cells: list[MotifCell] = []
    used = 0
    degree = 0
    for i in range(target_count):
        length = rng.choice((PPQ, PPQ // 2))
        if used + length > bar_ticks:
            break
        step = 0 if i == 0 else _draw_step(rng, degree, shape=shape)
        cells.append(MotifCell(step=step, length_ticks=length))
        degree += step
        used += length
    if len(cells) < 2:
        # A pathological draw (a tiny bar, or every draw wanted a full
        # quarter) still owes the section a two-cell motif.
        while len(cells) < 2 and used + PPQ // 2 <= bar_ticks:
            length = PPQ if used + PPQ <= bar_ticks else PPQ // 2
            step = _draw_step(rng, degree, shape=shape) if cells else 0
            cells.append(MotifCell(step=step, length_ticks=length))
            degree += step
            used += length
    motif = tuple(cells)
    steps = recover_leaps([cell.step for cell in motif], shape=shape)
    return _cells(steps, motif)


def _draw_step(
    rng: random.Random, degree: int, *, shape: MelodyShape = DEFAULT_MELODY_SHAPE
) -> int:
    """Draw one step, narrowing the choice at the motif's span limits.

    The vocabulary stays the same at the edges; only the directions that
    would leave the span are dropped, so a motif that has climbed still
    moves by thirds and fourths on the way down.
    """
    within = [
        (step, weight)
        for step, weight in zip(shape.step_choices, shape.step_weights, strict=True)
        if abs(degree + step) <= shape.max_motif_span_degrees
    ]
    return rng.choices(
        [step for step, _weight in within],
        weights=[weight for _step, weight in within],
        k=1,
    )[0]


def recover_leaps(
    steps: Sequence[int], *, shape: MelodyShape = DEFAULT_MELODY_SHAPE
) -> tuple[int, ...]:
    """Answer every leap with a step in the opposite direction.

    A leap that is not answered is the single most audible melodic fault:
    the line arrives nowhere and the listener loses it. Rewriting the
    step *after* a leap is the minimal repair, and it is a repair rather
    than a redraw because it leaves every other interval — including the
    leap itself — exactly as drawn.

    Inversion needs no second pass through this: negating every step
    negates a leap and its recovery together, so the property is
    preserved. Ornament and truncation do not, which is why the renderer
    runs this over the assembled steps of a bar rather than trusting the
    motif's own shape.

    The caller must still answer a leap in the *last* position: that one
    looks across the bar line, which this function cannot see.
    """
    out = list(steps)
    for index, step in enumerate(out):
        if abs(step) < shape.leap_degrees or index + 1 >= len(out):
            continue
        out[index + 1] = -1 if step > 0 else 1
    return tuple(out)


def _cells(steps: Sequence[int], motif: Motif) -> Motif:
    """Re-time `steps` onto `motif`'s durations (same count, same rhythm)."""
    return tuple(
        MotifCell(step=step, length_ticks=cell.length_ticks)
        for step, cell in zip(steps, motif, strict=True)
    )


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


def vary_motif(
    motif: Motif, rng: random.Random, *, shape: MelodyShape = DEFAULT_MELODY_SHAPE
) -> MotifVariant:
    """Derive one bar's material from the motif via a single operation.

    The draw picks the operation, never raw pitches: repetition plays
    the motif as it stands, transposition starts one chord tone higher,
    sequence repeats it through the bar with the anchor advancing,
    inversion mirrors it, truncation cuts it short (a rest), ornament
    decorates one cell.
    """
    roll = rng.random()
    threshold = 0.0
    for operation, weight in shape.motif_operation_weights:
        threshold += weight
        if roll < threshold:
            return _apply_operation(motif, operation, rng, shape=shape)
    return _apply_operation(motif, "ornament", rng, shape=shape)


def _apply_operation(
    motif: Motif, operation: str, rng: random.Random, *, shape: MelodyShape
) -> MotifVariant:
    """The variant one named operation makes of `motif`.

    Named rather than inlined so the operation vocabulary is a value a
    plan can carry: the weights above are drawn against, and the names
    here are what a caller reads and writes. An unknown name raises
    rather than falling back to the motif unchanged — a plan that asks
    for an operation the engine cannot perform must be refused, not
    silently given a repeat.
    """
    if operation == "repeat":
        return MotifVariant(motif=motif)
    if operation == "transpose":
        return MotifVariant(motif=motif, anchor_offset=shape.chord_tone_degrees)
    if operation == "sequence":
        return MotifVariant(motif=motif, repeat=True)
    if operation == "invert":
        return MotifVariant(motif=_invert(motif))
    if operation == "truncate":
        return MotifVariant(motif=_truncate(motif, rng))
    if operation == "ornament":
        return MotifVariant(motif=_ornament(motif, rng))
    raise ValueError(f"unknown motif operation: {operation!r}")


def _reflow(slots: list[list[int]]) -> list[BarSlot]:
    """Rebuild contiguous offsets after an op changed slot durations."""
    offset = 0
    out: list[BarSlot] = []
    for _start, duration, tone, tie in slots:
        out.append((offset, duration, tone, tie))
        offset += duration
    return out


# --- Bass-figure vocabulary ------------------------------------------------
#
# The left hand states the harmony, so every figure is chord tones and
# nothing else. The passing-tone licence admits a stepwise tone in any
# voice, and this voice does not take it up: a bass that leaves the
# harmony stops being the harmony.
#
# A figure is a tuple of onsets, each `(start, length, rung)`, written in
# sixteenths of the bar (16 to the bar). Proportions rather than ticks,
# because the engine composes in every common time signature and one
# figure should land at the same points of the measure in a 3/4 bar as in
# a 4/4 one — and because the rungs, not the pitches, are what let one
# figure serve every chord. `rung` counts up the chord's own tones from
# the one the walk landed on: rung 0 is that tone, 1 the next chord tone
# above it, and so on, so the same shape lands as root-third-fifth on a
# triad and reaches the seventh on a seventh chord. A triad inside an
# octave always holds three tones above any of its own, so rungs 0 to 3
# exist whatever the chord.
#
# Every figure begins on rung 0 at the bar line. The bar's harmony has to
# sound on its downbeat whatever the left hand does afterwards, and the
# walk's register is anchored on that note.

BassFigure = tuple[tuple[int, int, int], ...]

# The plain statement: the tone the walk landed on, then the next chord
# tone above it. Every mood's table contains it, and the final bar of a
# piece plays it whatever its slot drew — a close is stated, not decorated.
PLAIN_BASS_FIGURE: BassFigure = ((0, 8, 0), (8, 8, 1))

# A chord tone per beat: the pulse the electrifying mood is named for.
_DRIVING_BASS_FIGURE: BassFigure = ((0, 4, 0), (4, 4, 1), (8, 4, 2), (12, 4, 3))
# Root, third, then the tone a fifth above held to the bar line.
_ARCHED_BASS_FIGURE: BassFigure = ((0, 4, 0), (4, 4, 1), (8, 8, 2))
# The chord on the downbeat and then space: the bar belongs to the melody
# until the last beat leads back into the harmony.
_SPARSE_BASS_FIGURE: BassFigure = ((0, 4, 0), (12, 4, 1))
# One tone for the whole bar — a pedal under a slow line.
_PEDAL_BASS_FIGURE: BassFigure = ((0, 16, 0),)

BASS_FIGURES: dict[str, tuple[BassFigure, ...]] = {
    "electrifying": (
        _DRIVING_BASS_FIGURE,
        _ARCHED_BASS_FIGURE,
        PLAIN_BASS_FIGURE,
    ),
    "calming": (
        PLAIN_BASS_FIGURE,
        _ARCHED_BASS_FIGURE,
        _SPARSE_BASS_FIGURE,
    ),
    "sleep": (
        _PEDAL_BASS_FIGURE,
        PLAIN_BASS_FIGURE,
        _SPARSE_BASS_FIGURE,
        _ARCHED_BASS_FIGURE,
    ),
}
# Moods without their own profile fall back to the gentle set.
DEFAULT_BASS_FIGURES: tuple[BassFigure, ...] = BASS_FIGURES["calming"]




def draw_bass_figures(
    figures: tuple[BassFigure, ...], *, rng: random.Random, count: int
) -> tuple[BassFigure, ...]:
    """One figure per chord slot, rotating through the vocabulary given.

    A figure belongs to a *slot*, not to a bar: the left hand states a
    figure for as long as its harmony lasts and changes it when the
    harmony changes. That is what the scorecard's `bass_onset_patterns`
    counts — a piece whose bass plays one bar for its whole length scores
    one — and it is the musical reading of the same number, since variety
    that arrived per bar would be noise rather than an accompaniment.

    The vocabulary does the work: each mood's table runs from its most
    characteristic figure to its plainest, and the slot's index takes the
    next one, so a progression is accompanied by a line that changes with
    it. Only where the rotation *starts* is drawn, so two pieces of one
    mood do not open on the same figure.

    The vocabulary arrives as an argument rather than by mood name so that
    it is a value a composition plan carries and a caller can edit — the
    same reason the tables were named. `BASS_FIGURES[mood]` and its
    fallback are still how the *default* plan finds this mood's table.
    """
    if not figures:
        raise ValueError("a bass vocabulary must carry at least one figure")
    start = rng.randrange(len(figures))
    return tuple(figures[(start + index) % len(figures)] for index in range(count))


def _op_dotted(slots: list[list[int]], *, remainders: tuple[int, ...]) -> list[BarSlot]:
    """Long-short: two quarters become a dotted quarter + eighth.

    The lengthened note has to be a chord tone: a dotted quarter is half
    again a quarter, and the passing-tone licence admits a non-chord tone
    only for a quarter or less. Shortening the *second* note of the pair
    would be the dotted figure heard upside down, so a bar with no
    chord-tone quarter in that position keeps its straight rhythm.
    """
    for i in range(len(slots) - 1):
        if (
            slots[i][1] == PPQ
            and slots[i + 1][1] == PPQ
            and slots[i][2] % 7 in remainders
        ):
            slots[i][1] = PPQ + PPQ // 2
            slots[i + 1][1] = PPQ // 2
            break
    return _reflow(slots)


def _op_tie(slots: list[list[int]], *, remainders: tuple[int, ...]) -> list[BarSlot]:
    """Hold one pitch across a beat: mark two neighbouring noteheads at
    one degree tied (both noteheads stay; the performance layer plays
    them as one sound). Equal degrees spell the same pitch in a bar, so
    the degree is the pitch here.

    A tie is only for a chord tone. The licence reads the notation, and
    two noteheads at one pitch are not a step apart, so tying a
    non-chord tone would leave it entered by a step and *left* by
    nothing — the one thing a passing tone may not do.

    A pair already at one degree is the tie to prefer: it changes no
    pitch, and it holds the line where the line already was. A walk that
    only ever moves has no such pair, though, and the tie is the rhythm
    the bar drew — so where the bar has none, its first chord tone whose
    neighbour can join it is brought onto its pitch. That one moves the
    line, so it is drawn only where it leaves the note *after* the tie a
    step away: that note is heard from the tie's own pitch, and a tie
    that leaves a third or more standing there has bought a longer note
    with a worse line.
    """
    for i in range(len(slots) - 1):
        if slots[i][2] == slots[i + 1][2] and slots[i][2] % 7 in remainders:
            slots[i][3] = 1
            return _reflow(slots)
    for i in range(len(slots) - 1):
        if slots[i][2] % 7 not in remainders:
            continue
        if i + 2 < len(slots) and abs(slots[i + 2][2] - slots[i][2]) > 1:
            continue
        slots[i + 1][2] = slots[i][2]
        slots[i][3] = 1
        break
    return _reflow(slots)


def apply_rhythm(
    slots: list[tuple[int, int, int]],
    *,
    rng: random.Random,
    weights: tuple[tuple[str, float], ...],
    remainders: tuple[int, ...] = (0, 2, 4),
) -> list[BarSlot]:
    """Re-voice a bar's motif slots through the melody's rhythm library.

    The draw picks one operation for the whole bar — a bar speaks with
    one rhythmic idea, not a new figure on every beat. `straight` keeps
    the motif's own durations; `dotted` renders the long-short pair;
    `sixteenths` subdivides one eighth; `tie` holds a repeated pitch
    across its beat boundary.

    `remainders` are the bar's chord-tone degrees, which two of the four
    operations have to respect: the licence admits a non-chord tone only
    for a quarter or less, and only where a step leaves it on both sides.
    Both facts are about the harmony, so the rhythm library is told them
    rather than guessing — it moves durations, and a duration can decide
    whether a note is legal at all.

    `weights` is the figure table itself, not a mood to look one up by:
    the plan carries this mood's figures already resolved, and the
    fallback for a mood outside the table is applied where the plan is
    built.
    """
    names, values = zip(*weights, strict=True)
    operation = rng.choices(names, weights=values, k=1)[0]
    if operation == "straight" or len(slots) < 2:
        return [(offset, duration, tone, False) for offset, duration, tone in slots]
    mutable = [[*slot, 0] for slot in slots]
    if operation == "dotted":
        return _op_dotted(mutable, remainders=remainders)
    if operation == "sixteenths":
        # A subdivision has a note to be only where the line is already
        # skipping: a step and a skip are the two halves of a third, so a
        # slot whose next note is two degrees away splits into two steps,
        # which is the figure the ear expects and which the passing-tone
        # licence covers. Splitting a slot whose next note is a step away
        # has no such note to add — it would strike the same pitch twice
        # and, worse, put a unison between a leap and the step that
        # answers it.
        passing = [
            index
            for index in range(len(mutable) - 1)
            if mutable[index][1] == PPQ // 2
            and abs(mutable[index + 1][2] - mutable[index][2]) == 2
        ]
        if not passing:
            return _reflow(mutable)
        index = rng.choice(passing)
        half = mutable[index][1] // 2
        between = (mutable[index][2] + mutable[index + 1][2]) // 2
        mutable[index : index + 1] = [
            [mutable[index][0], half, mutable[index][2], 0],
            [0, half, between, 0],
        ]
        return _reflow(mutable)
    return _op_tie(mutable, remainders=remainders)
