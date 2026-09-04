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

Licensing/licensing posture: the GM kit sounds come from the general
soundfont (FluidR3_GM, MIT) — no dedicated drum font is needed.
"""

from __future__ import annotations

from dataclasses import dataclass

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
    engine rotates across sections. `velocity_scale` lets a style play
    soft (sleep ballad) or hard (funk) without redefining every hit.
    """

    name: str
    variants: dict[str, tuple[tuple[DrumHit, ...], ...]]
    velocity_scale: float = 1.0

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
                (960, DRUM_KICK, 88),
                (480, DRUM_SNARE, 84),
                (1440, DRUM_SNARE, 84),
                *[(o, DRUM_CLOSED_HIHAT, 60) for o in range(0, 1920, 240)],
            ),
            # B: same skeleton, crash-riding with an extra push beat.
            _hits(
                (0, DRUM_KICK, 92),
                (960, DRUM_KICK, 88),
                (1320, DRUM_KICK, 76),
                (480, DRUM_SNARE, 88),
                (1440, DRUM_SNARE, 88),
                *[(o, DRUM_CLOSED_HIHAT, 66) for o in range(0, 1920, 240)],
                (1920 - 240, DRUM_OPEN_HIHAT, 70),
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
                (720, DRUM_KICK, 80),
                (1440, DRUM_KICK, 84),
                (480, DRUM_SNARE, 84),
                (1680, DRUM_HAND_CLAP, 72),
                *[(o, DRUM_CLOSED_HIHAT, 56) for o in range(0, 1920, 120)],
            ),
            # B: open-hihat accents on the offbeats.
            _hits(
                (0, DRUM_KICK, 92),
                (720, DRUM_KICK, 80),
                (1440, DRUM_KICK, 84),
                (480, DRUM_SNARE, 88),
                (1680, DRUM_HAND_CLAP, 72),
                (240, DRUM_OPEN_HIHAT, 64),
                (1200, DRUM_OPEN_HIHAT, 64),
                *[(o, DRUM_CLOSED_HIHAT, 56) for o in range(0, 1920, 120)],
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
                (840, DRUM_KICK, 60),
                (480, DRUM_SIDE_STICK, 62),
                (1440, DRUM_SIDE_STICK, 62),
                (0, DRUM_CLOSED_HIHAT, 44),
                (480, DRUM_CLOSED_HIHAT, 40),
                (960, DRUM_CLOSED_HIHAT, 44),
                (1440, DRUM_CLOSED_HIHAT, 40),
            ),
            # B: one bar in eight, dropping the pickup kick.
            _hits(
                (0, DRUM_KICK, 72),
                (480, DRUM_SIDE_STICK, 62),
                (1440, DRUM_SIDE_STICK, 62),
                (0, DRUM_CLOSED_HIHAT, 44),
                (480, DRUM_CLOSED_HIHAT, 40),
                (960, DRUM_CLOSED_HIHAT, 44),
                (1440, DRUM_CLOSED_HIHAT, 40),
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
                (720, DRUM_KICK, 64),
                (1080, DRUM_KICK, 64),
                (240, DRUM_SIDE_STICK, 66),
                (960, DRUM_SIDE_STICK, 66),
                (1320, DRUM_SIDE_STICK, 60),
                *[(o, DRUM_CLOSED_HIHAT, 48) for o in range(0, 1920, 240)],
            ),
        )
    },
)

SWING = DrumStyle(
    name="swing",
    variants={
        "4/4": (
            # A: ride on the quarters, foot-hihat on 2 and 4. The swung
            # eighths ride pattern belongs to a triplet grid, so the
            # quarters carry the pulse until a triplet-time slice lands.
            _hits(
                *[(o, DRUM_RIDE, 64) for o in range(0, 1920, 480)],
                (480, DRUM_PEDAL_HIHAT, 56),
                (1440, DRUM_PEDAL_HIHAT, 56),
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
                (960, DRUM_KICK, 84),
                (480, DRUM_SNARE, 76),
                (1440, DRUM_SNARE, 76),
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
                (480, DRUM_SNARE, 66),
                (960, DRUM_SNARE, 66),
                (0, DRUM_CLOSED_HIHAT, 48),
                (480, DRUM_CLOSED_HIHAT, 40),
                (960, DRUM_CLOSED_HIHAT, 40),
            ),
        ),
        # 6/8 has the same bar length as 3/4 (three dotted quarters),
        # so the waltz skeleton maps onto it directly.
        "6/8": (
            _hits(
                (0, DRUM_KICK, 80),
                (480, DRUM_SNARE, 66),
                (960, DRUM_SNARE, 66),
                (0, DRUM_CLOSED_HIHAT, 48),
                (480, DRUM_CLOSED_HIHAT, 40),
                (960, DRUM_CLOSED_HIHAT, 40),
            ),
        ),
    },
)

SHUFFLE = DrumStyle(
    name="shuffle",
    variants={
        # Both meters share the 1440-tick bar: kick on 1 and the "and"
        # of 2, snare on 2 and 3, hats on the eighths.
        "3/4": (
            _hits(
                (0, DRUM_KICK, 84),
                (720, DRUM_KICK, 72),
                (480, DRUM_SNARE, 72),
                (960, DRUM_SNARE, 72),
                *[(o, DRUM_CLOSED_HIHAT, 52) for o in range(0, 1440, 240)],
            ),
        ),
        "6/8": (
            _hits(
                (0, DRUM_KICK, 84),
                (720, DRUM_KICK, 72),
                (480, DRUM_SNARE, 72),
                (960, DRUM_SNARE, 72),
                *[(o, DRUM_CLOSED_HIHAT, 52) for o in range(0, 1440, 240)],
            ),
        ),
    },
)

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


__all__ = [
    "BALLAD",
    "BOSSA",
    "DRUM_CLOSED_HIHAT",
    "DRUM_CRASH",
    "DRUM_HAND_CLAP",
    "DRUM_HIGH_TOM",
    "DRUM_KICK",
    "DRUM_OPEN_HIHAT",
    "DRUM_PEDAL_HIHAT",
    "DRUM_RIDE",
    "DRUM_SIDE_STICK",
    "DRUM_SNARE",
    "DRUM_STYLES",
    "FUNK",
    "MARCH",
    "METER_STYLES",
    "MOOD_STYLES_4_4",
    "MOOD_VELOCITY_SCALE",
    "PERCUSSION_NOTE_TICKS",
    "PERCUSSION_VELOCITY_MAX",
    "ROCK",
    "SHUFFLE",
    "SWING",
    "WALTZ",
    "DrumHit",
    "DrumStyle",
    "style_for",
]
