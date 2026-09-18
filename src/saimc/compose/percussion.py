"""Drum-pattern library for the percussion voice.

A drum set renders through GM channel 10, where the MIDI pitch IS the
drum piece (kick=36, snare=38, ...), not a melodic pitch. A "drum
pattern" is therefore a bar template of (offset, drum key, velocity)
hits — the rhythm-pattern equivalent of a keyboard's auto-accompaniment
styles.

`DrumStyle` mirrors how the mood's chord templates work
(`saimc.compose.forms`): each style carries A/B bar variants and the
engine rotates them across repeated sections so repeats differ in
surface rhythm, not just harmony. All patterns are defined in PPQ=480
ticks, matching the NotationScore, so a hit's offset is directly
addable to the bar's start tick.

Every style carries at least two variants per meter and at least one
fill, because the rotation is only a change of pace if there is a second
bar to change to: a single-variant style plays one bar for the whole
piece. Which variant a section takes, and which of a style's fills hands
it over, are read from the piece's seed as a phase of the rotation
(`rotation_index`) — so the written kit differs between two pieces of one
mood and meter, which it did not before.

Licensing/licensing posture: the GM kit sounds come from the general
soundfont (FluidR3_GM, MIT) — no dedicated drum font is needed.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field

from saimc.compose.score import PPQ

# GM percussion key numbers (channel 10 note pitches). Only the keys
# the pattern library uses are named here; the kit has more (toms,
# cymbal variants) and more can be named when a style needs them.
DRUM_KICK: int = 36  # acoustic bass drum
DRUM_SIDE_STICK: int = 37  # rim click — the ballad backbeat
DRUM_SNARE: int = 38
DRUM_HAND_CLAP: int = 39
DRUM_CLOSED_HIHAT: int = 42
DRUM_PEDAL_HIHAT: int = 44  # foot-chick, the jazz 2-and-4
DRUM_OPEN_HIHAT: int = 46
DRUM_CRASH: int = 49
DRUM_HIGH_TOM: int = 50
DRUM_RIDE: int = 51
DRUM_MID_TOM: int = 47
DRUM_LOW_TOM: int = 45

# Bar-relative tick offsets as PPQ fractions, so a pattern reads as its
# musical shape instead of bare tick literals (the numbers are identical
# at PPQ=480; the names are what a reader needs).
BEAT: int = PPQ
EIGHTH: int = PPQ // 2
SIXTEENTH: int = PPQ // 4
DOTTED_BEAT: int = PPQ * 3 // 2
DOTTED_EIGHTH: int = PPQ * 3 // 4
BAR_3_4: int = 3 * PPQ
BAR_4_4: int = 4 * PPQ

# A drum hit renders as a short gate; a full-length note_off would
# make no audible difference, but a uniform short duration keeps the
# PerformancePlan and the engraving honest. A 16th note at 4/4.
PERCUSSION_NOTE_TICKS: int = PPQ // 4

# Velocity ceilings: the kit sits under the melodic voices in the mix.
PERCUSSION_VELOCITY_MAX: int = 96


@dataclass(frozen=True)
class DrumHit:
    """One drum strike inside a bar template.

    `offset_ticks` is the PPQ offset from the bar's start tick
    (`0 <= offset < ticks_per_bar`). `key` is a GM percussion number.
    """

    offset_ticks: int
    key: int
    velocity: int = 80

    def __post_init__(self) -> None:
        if self.offset_ticks < 0:
            raise ValueError(f"offset_ticks must be non-negative; got {self.offset_ticks}")
        if not 1 <= self.velocity <= 127:
            raise ValueError(f"velocity must be in [1, 127]; got {self.velocity}")
        if not 35 <= self.key <= 81:
            raise ValueError(
                f"key {self.key} is not a GM percussion key (35-81); "
                "melodic pitches belong in the melodic voices"
            )


@dataclass(frozen=True)
class DrumStyle:
    """A named rhythm pattern with bar variants per time signature.

    `variants` holds, per time signature, a list of bar templates; the
    first is the style's A pattern and later ones are B patterns the
    engine rotates across sections. `fills` holds, per time signature,
    the transition bars played on a section's last bar (the groove
    hands off to the next section through a fill). `velocity_scale`
    lets a style play soft (sleep ballad) or hard (funk) without
    redefining every hit.
    """

    name: str
    variants: dict[str, tuple[tuple[DrumHit, ...], ...]]
    velocity_scale: float = 1.0
    fills: dict[str, tuple[tuple[DrumHit, ...], ...]] = dataclass_field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 < self.velocity_scale <= 1.5:
            raise ValueError(f"velocity_scale must be in (0, 1.5]; got {self.velocity_scale}")

    def pattern(self, time_signature: str, variant_index: int) -> tuple[DrumHit, ...] | None:
        """The bar template for a meter, or None if the style has none.

        Unknown meters (5/4, 7/8) mean no percussion for that piece —
        an honest skip beats a wrong pattern.
        """
        variants = self.variants.get(time_signature)
        if not variants:
            return None
        return variants[variant_index % len(variants)]

    def fill(self, time_signature: str, variant_index: int) -> tuple[DrumHit, ...] | None:
        """The transition bar for a meter, or None if the style has no fill.

        A style without a fill plays its groove straight through the
        section ending — the honest default.
        """
        fills = self.fills.get(time_signature)
        if not fills:
            return None
        return fills[variant_index % len(fills)]


def _hits(*spec: tuple[int, int, int]) -> tuple[DrumHit, ...]:
    """Shorthand: (offset_ticks, key, velocity) tuples to DrumHits."""
    return tuple(DrumHit(offset, key, velocity) for offset, key, velocity in spec)


# --- 4/4 styles ------------------------------------------------------------

ROCK = DrumStyle(
    name="rock",
    variants={
        "4/4": (
            # A: kick 1+3, snare 2+4, eighths on the hats.
            _hits(
                (0, DRUM_KICK, 92),
                (2 * BEAT, DRUM_KICK, 88),
                (BEAT, DRUM_SNARE, 84),
                (3 * BEAT, DRUM_SNARE, 84),
                *[(o, DRUM_CLOSED_HIHAT, 60) for o in range(0, BAR_4_4, EIGHTH)],
            ),
            # B: same skeleton, crash-riding with an extra push beat.
            _hits(
                (0, DRUM_KICK, 92),
                (2 * BEAT, DRUM_KICK, 88),
                (2 * BEAT + DOTTED_EIGHTH, DRUM_KICK, 76),
                (BEAT, DRUM_SNARE, 88),
                (3 * BEAT, DRUM_SNARE, 88),
                *[(o, DRUM_CLOSED_HIHAT, 66) for o in range(0, BAR_4_4, EIGHTH)],
                (BAR_4_4 - EIGHTH, DRUM_OPEN_HIHAT, 70),
            ),
        )
    },
    fills={
        # F1: 16th snare push, then the toms descend through beats 3-4.
        "4/4": (
            _hits(
                (0, DRUM_KICK, 88),
                (BEAT, DRUM_SNARE, 76),
                (BEAT + SIXTEENTH, DRUM_SNARE, 70),
                (BEAT + 2 * SIXTEENTH, DRUM_SNARE, 72),
                (BEAT + 3 * SIXTEENTH, DRUM_SNARE, 68),
                (2 * BEAT, DRUM_HIGH_TOM, 78),
                (2 * BEAT + EIGHTH, DRUM_MID_TOM, 74),
                (3 * BEAT, DRUM_LOW_TOM, 76),
                (3 * BEAT + EIGHTH, DRUM_LOW_TOM, 70),
            ),
            # F2: a 16th-note snare roll building into the next section.
            _hits(
                (0, DRUM_KICK, 88),
                *[
                    (BEAT + i * SIXTEENTH, DRUM_SNARE, 58 + i * 3)
                    for i in range(12)
                ],
                (3 * BEAT, DRUM_HIGH_TOM, 80),
            ),
        )
    },
)

FUNK = DrumStyle(
    name="funk",
    variants={
        "4/4": (
            # A: syncopated kick, 16th hats, clap ghost on 4&.
            _hits(
                (0, DRUM_KICK, 92),
                (DOTTED_BEAT, DRUM_KICK, 80),
                (3 * BEAT, DRUM_KICK, 84),
                (BEAT, DRUM_SNARE, 84),
                (3 * BEAT + EIGHTH, DRUM_HAND_CLAP, 72),
                *[(o, DRUM_CLOSED_HIHAT, 56) for o in range(0, BAR_4_4, SIXTEENTH)],
            ),
            # B: open-hihat accents on the offbeats.
            _hits(
                (0, DRUM_KICK, 92),
                (DOTTED_BEAT, DRUM_KICK, 80),
                (3 * BEAT, DRUM_KICK, 84),
                (BEAT, DRUM_SNARE, 88),
                (3 * BEAT + EIGHTH, DRUM_HAND_CLAP, 72),
                (EIGHTH, DRUM_OPEN_HIHAT, 64),
                (2 * BEAT + EIGHTH, DRUM_OPEN_HIHAT, 64),
                *[(o, DRUM_CLOSED_HIHAT, 56) for o in range(0, BAR_4_4, SIXTEENTH)],
            ),
        )
    },
    fills={
        # F1: ghosted 16th snare build over the last two beats, landing on
        # the open hat that hands off to the next section.
        "4/4": (
            _hits(
                (0, DRUM_KICK, 88),
                *[(2 * BEAT + i * SIXTEENTH, DRUM_SNARE, 54 + i * 3) for i in range(7)],
                (BAR_4_4 - EIGHTH, DRUM_OPEN_HIHAT, 66),
            ),
            # F2: the kick and the clap trading the back half — the same
            # hand-off without a roll, for a section that should turn
            # rather than build.
            _hits(
                (0, DRUM_KICK, 88),
                (BEAT, DRUM_SNARE, 84),
                (2 * BEAT, DRUM_KICK, 82),
                (2 * BEAT + EIGHTH, DRUM_HAND_CLAP, 70),
                (3 * BEAT, DRUM_SNARE, 86),
                (3 * BEAT + SIXTEENTH, DRUM_SNARE, 66),
                (3 * BEAT + EIGHTH, DRUM_KICK, 78),
            ),
        )
    },
)

BALLAD = DrumStyle(
    name="ballad",
    variants={
        "4/4": (
            # A: cross-stick backbeat, quarter hats, kick on 1 and the
            # pickup into 3.
            _hits(
                (0, DRUM_KICK, 72),
                (2 * BEAT - SIXTEENTH, DRUM_KICK, 60),
                (BEAT, DRUM_SIDE_STICK, 62),
                (3 * BEAT, DRUM_SIDE_STICK, 62),
                (0, DRUM_CLOSED_HIHAT, 44),
                (BEAT, DRUM_CLOSED_HIHAT, 40),
                (2 * BEAT, DRUM_CLOSED_HIHAT, 44),
                (3 * BEAT, DRUM_CLOSED_HIHAT, 40),
            ),
            # B: one bar in eight, dropping the pickup kick.
            _hits(
                (0, DRUM_KICK, 72),
                (BEAT, DRUM_SIDE_STICK, 62),
                (3 * BEAT, DRUM_SIDE_STICK, 62),
                (0, DRUM_CLOSED_HIHAT, 44),
                (BEAT, DRUM_CLOSED_HIHAT, 40),
                (2 * BEAT, DRUM_CLOSED_HIHAT, 44),
                (3 * BEAT, DRUM_CLOSED_HIHAT, 40),
            ),
        )
    },
    fills={
        # A gentle hand-off: cross-stick eighths swelling through the
        # back half of the bar, no toms — the ballad stays soft.
        "4/4": (
            _hits(
                (0, DRUM_KICK, 72),
                (BEAT, DRUM_SIDE_STICK, 60),
                (2 * BEAT, DRUM_SIDE_STICK, 56),
                (2 * BEAT + EIGHTH, DRUM_SIDE_STICK, 58),
                (3 * BEAT, DRUM_SIDE_STICK, 62),
                (3 * BEAT + EIGHTH, DRUM_CLOSED_HIHAT, 48),
            ),
            # The quieter turn: a single cross-stick on the last beat,
            # for a hand-off under a melody that is still singing.
            _hits(
                (0, DRUM_KICK, 70),
                (BEAT, DRUM_CLOSED_HIHAT, 38),
                (2 * BEAT, DRUM_KICK, 60),
                (2 * BEAT, DRUM_CLOSED_HIHAT, 38),
                (3 * BEAT, DRUM_SIDE_STICK, 58),
            ),
        )
    },
)

BOSSA = DrumStyle(
    name="bossa",
    variants={
        "4/4": (
            # A: the classic bossa rim pattern over straight eighths.
            _hits(
                (0, DRUM_KICK, 72),
                (DOTTED_BEAT, DRUM_KICK, 64),
                (2 * BEAT + SIXTEENTH, DRUM_KICK, 64),
                (EIGHTH, DRUM_SIDE_STICK, 66),
                (2 * BEAT, DRUM_SIDE_STICK, 66),
                (2 * BEAT + DOTTED_EIGHTH, DRUM_SIDE_STICK, 60),
                *[(o, DRUM_CLOSED_HIHAT, 48) for o in range(0, BAR_4_4, EIGHTH)],
            ),
            # B: the same clave answered, with the open hat closing the
            # bar and the kick pushed onto the "and" of 2.
            _hits(
                (0, DRUM_KICK, 74),
                (DOTTED_BEAT, DRUM_KICK, 62),
                (3 * BEAT + EIGHTH, DRUM_KICK, 66),
                (EIGHTH, DRUM_SIDE_STICK, 62),
                (2 * BEAT, DRUM_SIDE_STICK, 68),
                (2 * BEAT + DOTTED_EIGHTH, DRUM_SIDE_STICK, 58),
                *[(o, DRUM_CLOSED_HIHAT, 46) for o in range(0, BAR_4_4 - EIGHTH, EIGHTH)],
                (BAR_4_4 - EIGHTH, DRUM_OPEN_HIHAT, 58),
            ),
        )
    },
    fills={
        # The rim answers across the back half of the bar and hands the
        # groove over without a roll — a bossa does not crescendo.
        "4/4": (
            _hits(
                (0, DRUM_KICK, 70),
                (2 * BEAT, DRUM_SIDE_STICK, 58),
                (2 * BEAT + EIGHTH, DRUM_SIDE_STICK, 62),
                (3 * BEAT, DRUM_SIDE_STICK, 66),
                (3 * BEAT + EIGHTH, DRUM_CLOSED_HIHAT, 52),
            ),
        )
    },
)

SWING = DrumStyle(
    name="swing",
    variants={
        "4/4": (
            # A: ride on the quarters with the "and" of 2 and 4 answered on
            # the same cymbal, foot-hihat on 2 and 4. The offbeat ride is
            # written where a *straight* eighth sits, and `swing_ratio` is
            # what moves it onto the triplet — so this is the swung ride at
            # 2.0 and an even eighth ride at 1.0, which is why the quarter
            # pulse no longer has to wait for a triplet grid.
            _hits(
                *[(o, DRUM_RIDE, 64) for o in range(0, BAR_4_4, BEAT)],
                (BEAT + EIGHTH, DRUM_RIDE, 54),
                (3 * BEAT + EIGHTH, DRUM_RIDE, 54),
                (480, DRUM_PEDAL_HIHAT, 56),
                (3 * BEAT, DRUM_PEDAL_HIHAT, 56),
            ),
            # B: the quarters with the snare answering on 4 — the kit
            # whispers the backbeat instead of keeping it on 2 and 4.
            _hits(
                *[(o, DRUM_RIDE, 62) for o in range(0, BAR_4_4, BEAT)],
                (480, DRUM_PEDAL_HIHAT, 56),
                (3 * BEAT, DRUM_SNARE, 54),
            ),
        )
    },
    fills={
        # A pickup on the last beat: the ride keeps the pulse, the snare
        # doubles it and a tom takes the bar over the line.
        "4/4": (
            _hits(
                (0, DRUM_RIDE, 60),
                (2 * BEAT, DRUM_PEDAL_HIHAT, 54),
                (3 * BEAT, DRUM_SNARE, 52),
                (3 * BEAT + EIGHTH, DRUM_SNARE, 58),
                (BAR_4_4 - SIXTEENTH, DRUM_HIGH_TOM, 62),
            ),
        )
    },
    velocity_scale=0.9,
)

MARCH = DrumStyle(
    name="march",
    variants={
        "4/4": (
            # A: oom-pah — kick on 1+3, snare on 2+4, no hats.
            _hits(
                (0, DRUM_KICK, 88),
                (2 * BEAT, DRUM_KICK, 84),
                (BEAT, DRUM_SNARE, 76),
                (3 * BEAT, DRUM_SNARE, 76),
            ),
            # B: the field-drum answer — the snares on the offbeats
            # instead of the beats, which is a different strain of the
            # same march rather than a busier one.
            _hits(
                (0, DRUM_KICK, 88),
                (2 * BEAT, DRUM_KICK, 84),
                (EIGHTH, DRUM_SNARE, 72),
                (2 * BEAT + EIGHTH, DRUM_SNARE, 72),
                (3 * BEAT + EIGHTH, DRUM_SNARE, 76),
            ),
        )
    },
    fills={
        # Military roll: 16ths on the snare through beats 1-2, tom
        # accents answering on beats 3-4.
        "4/4": (
            _hits(
                *[(BEAT + i * SIXTEENTH, DRUM_SNARE, 60 + i * 2) for i in range(8)],
                (2 * BEAT, DRUM_HIGH_TOM, 78),
                (2 * BEAT + BEAT, DRUM_MID_TOM, 78),
                (3 * BEAT, DRUM_LOW_TOM, 80),
            ),
        )
    },
)

# --- 3/4 and 6/8 styles ----------------------------------------------------

WALTZ = DrumStyle(
    name="waltz",
    variants={
        # 3/4: kick on 1, snare on 2+3, hats marking each beat.
        "3/4": (
            _hits(
                (0, DRUM_KICK, 80),
                (BEAT, DRUM_SNARE, 66),
                (2 * BEAT, DRUM_SNARE, 66),
                (0, DRUM_CLOSED_HIHAT, 48),
                (BEAT, DRUM_CLOSED_HIHAT, 40),
                (2 * BEAT, DRUM_CLOSED_HIHAT, 40),
            ),
            # B: the second strain — the kick pushes into 2 and the last
            # beat opens the hat, so a repeated 3/4 section has a place
            # to go that is not another fill.
            _hits(
                (0, DRUM_KICK, 80),
                (BEAT - EIGHTH, DRUM_KICK, 62),
                (BEAT, DRUM_SNARE, 64),
                (2 * BEAT, DRUM_SNARE, 66),
                (0, DRUM_CLOSED_HIHAT, 46),
                (BEAT, DRUM_CLOSED_HIHAT, 40),
                (2 * BEAT, DRUM_OPEN_HIHAT, 52),
            ),
        ),
        # 6/8 has the same bar length as 3/4 (three dotted quarters),
        # so the waltz skeleton maps onto it directly.
        "6/8": (
            _hits(
                (0, DRUM_KICK, 80),
                (BEAT, DRUM_SNARE, 66),
                (2 * BEAT, DRUM_SNARE, 66),
                (0, DRUM_CLOSED_HIHAT, 48),
                (BEAT, DRUM_CLOSED_HIHAT, 40),
                (2 * BEAT, DRUM_CLOSED_HIHAT, 40),
            ),
            _hits(
                (0, DRUM_KICK, 80),
                (BEAT - EIGHTH, DRUM_KICK, 62),
                (BEAT, DRUM_SNARE, 64),
                (2 * BEAT, DRUM_SNARE, 66),
                (0, DRUM_CLOSED_HIHAT, 46),
                (BEAT, DRUM_CLOSED_HIHAT, 40),
                (2 * BEAT, DRUM_OPEN_HIHAT, 52),
            ),
        ),
    },
    fills={
        # A short push into the next 8: snare 16ths across beat 3.
        "3/4": (
            _hits(
                (0, DRUM_KICK, 80),
                *[(2 * BEAT + i * SIXTEENTH, DRUM_SNARE, 54 + i * 2) for i in range(4)],
            ),
        ),
        "6/8": (
            _hits(
                (0, DRUM_KICK, 80),
                *[(2 * BEAT + i * SIXTEENTH, DRUM_SNARE, 54 + i * 2) for i in range(4)],
            ),
        ),
    },
)

SHUFFLE = DrumStyle(
    name="shuffle",
    variants={
        # Both meters share the 3-beat (1440-tick) bar: kick on 1 and the "and"
        # of 2, snare on 2 and 3, hats on the eighths.
        "3/4": (
            _hits(
                (0, DRUM_KICK, 84),
                (DOTTED_BEAT, DRUM_KICK, 72),
                (BEAT, DRUM_SNARE, 72),
                (2 * BEAT, DRUM_SNARE, 72),
                *[(o, DRUM_CLOSED_HIHAT, 52) for o in range(0, BAR_3_4, EIGHTH)],
            ),
            # B: the same bar with the snare answering the "and" of 2 —
            # the shuffle's own syncopation, still on the eighth grid so
            # it stays a shuffle and does not become a swing.
            _hits(
                (0, DRUM_KICK, 84),
                (BEAT + EIGHTH, DRUM_KICK, 70),
                (BEAT, DRUM_SNARE, 72),
                (2 * BEAT, DRUM_SNARE, 72),
                (2 * BEAT + EIGHTH, DRUM_SNARE, 58),
                *[(o, DRUM_CLOSED_HIHAT, 52) for o in range(0, BAR_3_4, EIGHTH)],
            ),
        ),
        "6/8": (
            _hits(
                (0, DRUM_KICK, 84),
                (DOTTED_BEAT, DRUM_KICK, 72),
                (BEAT, DRUM_SNARE, 72),
                (2 * BEAT, DRUM_SNARE, 72),
                *[(o, DRUM_CLOSED_HIHAT, 52) for o in range(0, BAR_3_4, EIGHTH)],
            ),
            _hits(
                (0, DRUM_KICK, 84),
                (BEAT + EIGHTH, DRUM_KICK, 70),
                (BEAT, DRUM_SNARE, 72),
                (2 * BEAT, DRUM_SNARE, 72),
                (2 * BEAT + EIGHTH, DRUM_SNARE, 58),
                *[(o, DRUM_CLOSED_HIHAT, 52) for o in range(0, BAR_3_4, EIGHTH)],
            ),
        ),
    },
    fills={
        # A push into the downbeat: snare 16ths across the last beat.
        "3/4": (
            _hits(
                (0, DRUM_KICK, 84),
                *[(2 * BEAT + i * SIXTEENTH, DRUM_SNARE, 56 + i * 2) for i in range(4)],
            ),
        ),
        "6/8": (
            _hits(
                (0, DRUM_KICK, 84),
                *[(2 * BEAT + i * SIXTEENTH, DRUM_SNARE, 56 + i * 2) for i in range(4)],
            ),
        ),
    },
)

# Section hand-offs: the crash + kick that mark every section downbeat,
# and the long rotation cycle that keeps the B variant a change of pace
# rather than the new normal.
SECTION_CRASH_VELOCITY: int = 90

ROTATION_CYCLE: tuple[int, ...] = (0, 0, 1, 0)

PERCUSSION_REST_SECTION: int = 1
"""The mid-piece section a long piece's kit rests for.

A hole in the texture before it refills, per §10 #10's demand that repeats
differ. It lives here rather than in `engine.py`, where it was declared,
because a plan has to be able to read it and the engine imports the plan.
"""

PERCUSSION_ENTRY_BAR: int = 2
"""The first bar the kit may sound in, counting from zero.

The groove is stated before the drums join it — the bed and the bass carry
the first bars, and the kit arrives on the downbeat of the third. It is a
floor rather than a position: a long piece whose intro outlasts it keeps
resting through the intro, and this is the bar the kit comes back at when
the intro is shorter than the kit's own entrance. Without it a short piece
opened on a crash cymbal over bar one, which is the loudest possible first
impression for a sleep ballad.
"""

PERCUSSION_KICK_VELOCITY: int = 84
"""The kit's own kick level, for the places no pattern names one.

Two places need a kick the template tables do not write: the downbeat mark,
and the kick that follows the bass. Both are the kit playing its own
accent rather than a chosen note, so they share one number.
"""

SWING_RATIO_STRAIGHT: float = 1.0
"""A piece whose eighth notes are even. The identity of `swing_offset`."""

SWING_RATIO_TRIPLET: float = 2.0
"""The ratio that puts a beat's second eighth on the triplet.

A swing ratio is the share of the beat the *first* eighth takes, so 1.0
leaves the pair even, 1.5 is a light swing, and 2.0 is the triplet every
jazz method writes. Past 2.0 the offbeat is later than the triplet, which
is a dotted rhythm rather than a swing — and a dotted rhythm is something
the pattern tables write directly, so the plan's bound stops here.
"""


def swing_offset(offset: int, ratio: float = SWING_RATIO_STRAIGHT) -> int:
    """Where a swing lands a hit the pattern wrote on a beat's second eighth.

    The ratio divides the *beat*, not the bar: its first eighth takes
    `ratio / (ratio + 1)` of the beat and the second the rest, so 1.0
    leaves the hit where it was written and 2.0 lands it on the triplet.
    Everything else is untouched — a hit on the beat, on a sixteenth, or on
    a dotted value is where the pattern put it, and the offbeat eighth is
    the one position a swing displaces. That is why the arithmetic needs no
    triplet grid: the grid would only decide where the offbeat goes, and
    the ratio decides exactly that.
    """
    beat_start = offset - offset % PPQ
    if offset - beat_start != EIGHTH:
        return offset
    return beat_start + round(PPQ * ratio / (ratio + 1.0))


def swing_pattern(
    pattern: tuple[DrumHit, ...], ratio: float = SWING_RATIO_STRAIGHT
) -> tuple[DrumHit, ...]:
    """A bar template with its offbeat eighths displaced by the ratio.

    Returns the pattern itself at `SWING_RATIO_STRAIGHT`, which is what
    makes a straight piece byte-identical to the one this module wrote
    before a ratio existed. A displacement moves a hit later inside its own
    beat and never across a beat line, so the pattern's order survives.
    """
    if ratio == SWING_RATIO_STRAIGHT:
        return pattern
    return tuple(
        replace(hit, offset_ticks=swing_offset(hit.offset_ticks, ratio)) for hit in pattern
    )


def rotation_index(
    section_idx: int,
    variant_count: int,
    *,
    cycle: tuple[int, ...] = ROTATION_CYCLE,
    seed: int = 0,
) -> int:
    """The pattern variant for a section, on a longer cycle than A/B.

    With one variant every section plays it. With more, the cycle holds
    A for two sections, changes pace with B for one, then returns — a
    B pattern every other section would stop feeling like a change.

    The cycle is an argument rather than the module constant because the
    plan carries it, and this module is where the plan and the engine
    both read it. An entry past the last variant wraps rather than going
    silent — `DrumStyle.pattern` is where that is decided.

    `seed` is the piece's, and it enters as a *phase*: the same cycle read
    from a different starting point, so two pieces of one mood and meter
    place their change of pace in different sections instead of sharing a
    drum part. Before this the written kit was byte-identical for every
    seed — the notation, not merely the render — and the drums were the
    one voice the seed did not reach. The plan cannot carry the offset: a
    plan is derived from the spec and does not hold the seed (the arbiter
    depends on that), and which section changes pace is a realisation of
    the plan's cycle rather than a second vocabulary. The default of 0 is
    the cycle as written.
    """
    if variant_count < 2:
        return 0
    return cycle[(section_idx + seed) % len(cycle)]


@dataclass(frozen=True)
class DrumKit:
    """How the kit is written for one piece, as the struct `engine.py` reads.

    One-way bridge, for B2's reason: the engine's percussion pass takes no
    plan, and a plan cannot hold a `DrumStyle` — a style is a table of bar
    templates, and the plan is a canonical JSON document. So the plan
    carries the style's *name* and the four values beside it, and this
    is where the name becomes the style.

    A style with no template for the piece's meter writes no drums at
    all, which is what `style_for` does today for an exotic meter: the
    honest skip beats a wrong pattern. The name is the one thing here the
    plan validates — an unknown one is refused where it is written, not
    silently resolved to silence.

    `style=None` is a piece with no kit, which is the default for a
    caller that never named one.
    """

    style: DrumStyle | None = None
    rotation_cycle: tuple[int, ...] = ROTATION_CYCLE
    velocity_scale: float = 1.0
    crash_velocity: int = SECTION_CRASH_VELOCITY
    swing_ratio: float = SWING_RATIO_STRAIGHT


DEFAULT_DRUM_KIT: DrumKit = DrumKit()
"""No kit: every value the module's own, and no style to write with."""

# --- Mood-driven selection ---------------------------------------------------

# The mood's default 4/4 style. Electrifying drives; calming breathes;
# sleep is a ballad played at the style's lowest energy.
MOOD_STYLES_4_4: dict[str, str] = {
    "electrifying": "rock",
    "calming": "ballad",
    "sleep": "ballad",
}

# Styles registered by name. The engine picks a style by mood + meter
# here; new styles register in this table and (when mood-driven) in
# MOOD_STYLES_4_4.
DRUM_STYLES: dict[str, DrumStyle] = {
    "rock": ROCK,
    "funk": FUNK,
    "ballad": BALLAD,
    "bossa": BOSSA,
    "swing": SWING,
    "march": MARCH,
    "waltz": WALTZ,
    "shuffle": SHUFFLE,
}

# Per-mood energy on top of the style's own velocity_scale: sleep gets
# the same ballad as calming but played noticeably softer.
MOOD_VELOCITY_SCALE: dict[str, float] = {
    "sleep": 0.75,
}

# 6/8 swaps the 4/4 mood style for the shuffle feel; 3/4 is a waltz.
METER_STYLES: dict[str, str] = {"3/4": "waltz", "6/8": "shuffle"}


def style_for(mood: str, time_signature: str) -> DrumStyle | None:
    """Pick the drum style for a mood + meter, or None for no drums.

    Exotic meters (5/4, 7/8) return None — no style claims them yet —
    and the engine skips the percussion voice entirely rather than
    forcing a wrong pattern onto them.
    """
    meter_style = METER_STYLES.get(time_signature)
    if meter_style is not None:
        return DRUM_STYLES.get(meter_style)
    if time_signature != "4/4":
        return None
    name = MOOD_STYLES_4_4.get(mood)
    if name is None:
        return None
    return DRUM_STYLES.get(name)


def style_name_for(mood: str, time_signature: str) -> str | None:
    """The name of the drum style for a mood + meter, or None for no drums.

    The meter decides when it has an opinion — 3/4 is a waltz, 6/8 a
    shuffle — and otherwise 4/4 falls to the mood's style while every
    exotic meter stays silent. The name rather than the `DrumStyle` is
    what a composition plan carries, so the choice is a value a caller
    can read, write, diff and store rather than an object it cannot.
    """
    meter_style = METER_STYLES.get(time_signature)
    if meter_style is not None:
        return meter_style
    if time_signature != "4/4":
        return None
    return MOOD_STYLES_4_4.get(mood)


__all__ = [
    "BALLAD",
    "BAR_3_4",
    "BAR_4_4",
    "BEAT",
    "BOSSA",
    "DEFAULT_DRUM_KIT",
    "DOTTED_BEAT",
    "DOTTED_EIGHTH",
    "DRUM_CLOSED_HIHAT",
    "DRUM_CRASH",
    "DRUM_HAND_CLAP",
    "DRUM_HIGH_TOM",
    "DRUM_KICK",
    "DRUM_LOW_TOM",
    "DRUM_MID_TOM",
    "DRUM_OPEN_HIHAT",
    "DRUM_PEDAL_HIHAT",
    "DRUM_RIDE",
    "DRUM_SIDE_STICK",
    "DRUM_SNARE",
    "DRUM_STYLES",
    "EIGHTH",
    "FUNK",
    "MARCH",
    "METER_STYLES",
    "MOOD_STYLES_4_4",
    "MOOD_VELOCITY_SCALE",
    "PERCUSSION_ENTRY_BAR",
    "PERCUSSION_KICK_VELOCITY",
    "PERCUSSION_NOTE_TICKS",
    "PERCUSSION_REST_SECTION",
    "PERCUSSION_VELOCITY_MAX",
    "ROCK",
    "ROTATION_CYCLE",
    "SECTION_CRASH_VELOCITY",
    "SHUFFLE",
    "SIXTEENTH",
    "SWING",
    "SWING_RATIO_STRAIGHT",
    "SWING_RATIO_TRIPLET",
    "WALTZ",
    "DrumHit",
    "DrumKit",
    "DrumStyle",
    "rotation_index",
    "style_for",
    "style_name_for",
    "swing_offset",
    "swing_pattern",
]
