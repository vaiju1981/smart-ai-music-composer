"""The bass voice: which register it plays in, and the figure it states.

The left hand is the one voice whose register is decided by the *instrument*
rather than by the melody above it — a tuba and a bass guitar do not share a
ladder — so that decision is here, with the walk bounds it has to stay inside.
"""

from __future__ import annotations

from dataclasses import dataclass

from saimc.compose.motif import (
    BassFigure,
)
from saimc.instruments import (
    MelodyBand,
    range_for,
)

# The bass voice's ceiling. A figure's upper rungs climb from the bar's
# landing tone; a rung that reaches past this is taken an octave lower,
# where it is still the same chord tone.
BASS_HIGH_MIDI: int = 67
# The register the walking bass lands in. Both arms of the walk — the one
# that picks the nearest chord tone and the one a template's `bass_degree`
# pins — keep their candidates inside it, so neither can land a bar above
# the point where the figure has nothing left to resolve onto.
BASS_WALK_LOW_MIDI: int = 21
BASS_WALK_HIGH_MIDI: int = 60


@dataclass(frozen=True)
class BassRegister:
    """Where the bass voice may sound: the window its walk lands in, and the
    ceiling its figures climb to.

    Two numbers with one source — the compass of whoever is playing it — so
    they travel together and cannot be narrowed apart. `BASS_WALK_LOW_MIDI`
    and `BASS_HIGH_MIDI` are both the piano's, because the left hand was
    written for one before `instruments.py` gave every voice its own.
    """

    window: MelodyBand
    ceiling: int


def bass_register(instrument: str | None) -> BassRegister:
    """The piano's bass register, narrowed to the compass of the voice that plays it.

    The window is A0 to C4 and the ceiling C4 plus a fifth, which is the
    register the left hand was written for before `instruments.py` gave every
    voice its own. A cello's floor is C2, so the walk's own bottom is a note no
    cello has, and the only thing that kept the bass out of it was that the
    roots it lands on sit above it — until the final repetition is lifted
    *down*. `modulation_offset=-12` is a legal plan (the plan refuses only
    beyond the octave) and it reached the window's floor: 24 of the 144 pieces
    of a 3-mood x 4-duration x 12-seed sweep failed lint at the bound, all of
    them the bass below its own instrument's range. That is the one thing the
    plan states it cannot do — a bound whose far end is a plan the engine
    refuses.

    The ceiling is the same defect reached the other way, and it is why 48 of
    the 66 instruments in the table cannot hold down the bass on this engine.
    A tuba's compass stops at Bb3, so a rung written between C4 and G4 is a
    note no tuba has; the ceiling is what the loop that takes a too-high rung
    an octave lower compares against.

    Narrowing cannot move a note that composed. A landing tone or a rung
    outside the compass is a lint failure whichever window or ceiling chose
    it, so the candidates this removes are exactly the ones that used to
    raise, and the nearest-tone choice is unchanged whenever the tone it
    picked survives the narrowing.
    """
    if instrument is None:
        # No voice was named for the bass, so no compass is known and the
        # linter falls back to its own wide range: the walk's own numbers are
        # what is left to say where it may land and how high it may climb.
        return BassRegister(
            window=MelodyBand(low_midi=BASS_WALK_LOW_MIDI, high_midi=BASS_WALK_HIGH_MIDI),
            ceiling=BASS_HIGH_MIDI,
        )
    compass = range_for(instrument)
    return BassRegister(
        window=MelodyBand(
            low_midi=max(BASS_WALK_LOW_MIDI, compass.low_midi),
            high_midi=min(BASS_WALK_HIGH_MIDI, compass.high_midi),
        ),
        ceiling=min(BASS_HIGH_MIDI, compass.high_midi),
    )


def _bass_ladder(anchor: int, *, chord_root: int, chord_tones: tuple[int, ...]) -> tuple[int, ...]:
    """The chord's own tones rising from the bar's landing tone.

    Rung 0 is the landing tone itself and every later rung is the next
    chord tone above the one before, so a figure written in rungs lands
    as the chord's own intervals wherever the walk put it: a
    root-third-fifth figure is a root-third-fifth over every chord of the
    progression, and the same figure reaches the seventh on a seventh
    chord. A triad fills a rung within every octave, so rungs 0 to 3
    exist for any chord a mood can draw.
    """
    pitch_classes = {(chord_root + tone) % 12 for tone in chord_tones}
    ladder = [anchor]
    pitch = anchor
    for _ in range(len(pitch_classes)):
        pitch = next(p for p in range(pitch + 1, pitch + 13) if p % 12 in pitch_classes)
        ladder.append(pitch)
    return tuple(ladder)


def bass_figure_pitches(
    figure: BassFigure,
    *,
    anchor: int,
    chord_root: int,
    chord_tones: tuple[int, ...],
    ceiling: int,
) -> tuple[tuple[int, int, int], ...]:
    """Resolve a figure's rungs into the bass register above the anchor.

    Returns `(start, length, pitch)` per sounding note, still in
    sixteenths of the bar. A rung that would climb past `ceiling` — the
    bass register's own, narrowed to the compass of the voice playing it —
    is taken an octave lower, and one that still lands below the bar's
    landing tone is dropped: the figure is written for the harmony, not
    for the register the walk happened to land in, and a note under the
    tone the bar opened on is a different figure. Dropping a rung rather
    than rewriting it keeps every remaining onset where the figure put
    it. The landing tone itself is always rung 0 at the bar line, so no
    figure can leave a bar without its harmony on the downbeat.
    """
    ladder = _bass_ladder(anchor, chord_root=chord_root, chord_tones=chord_tones)
    resolved: list[tuple[int, int, int]] = []
    for start, length, rung in figure:
        pitch = ladder[min(rung, len(ladder) - 1)]
        while pitch > ceiling:
            pitch -= 12
        if pitch >= anchor:
            resolved.append((start, length, pitch))
    return tuple(resolved)


__all__ = [
    "BASS_HIGH_MIDI",
    "BASS_WALK_HIGH_MIDI",
    "BASS_WALK_LOW_MIDI",
    "BassRegister",
    "bass_figure_pitches",
    "bass_register",
]
