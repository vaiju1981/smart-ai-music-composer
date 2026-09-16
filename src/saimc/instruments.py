"""Per-instrument compass — what each instrument can actually play.

One row per instrument in `saimc.spec.Instrument`, and the single place
that answers "where does this instrument live". It sits at the package
root rather than under `render/` because both sides of the pipeline need
it and neither owns it: the composer places a melody inside the band, and
the linter refuses a note outside the compass. Keeping it in `render/`
would have made `compose/` import from `render/`, which is the wrong
direction — `render` is downstream of the score, not upstream of it.

Three numbers describe an instrument, and the difference between the
first two is the whole point:

- **Range** (`low_midi`..`high_midi`) is the compass: every note the
  instrument can sound at all. This is what the linter enforces, and it
  is what roadmap §8's "all notes within instrument range" means.
- **Tessitura** (`tessitura_low`..`tessitura_high`) is where the
  instrument is comfortable and sounds like itself — the core of the
  compass a part should be written in. `melody_band` takes a line's
  window from the middle of it and `bed_window` takes an accompaniment's
  from the whole of it. An instrument's extreme notes are reachable but
  not livable: a trumpet *can* play a low F#3, and a tune written down
  there is a bad trumpet tune.
- **Polyphony** is how many notes it can sound at once: 1 for anything
  with a single resonator (every wind, every brass, the solo bowed
  strings), more for keyboards, mallet instruments, plucked strings with
  courses, and the section patches, which are many players and can
  divide.

Values are idiomatic professional compasses in MIDI note numbers (60 =
middle C) at concert pitch — for the transposing instruments that means
*sounding* pitch, because a General MIDI program plays the note it is
given. They are deliberately conservative: the outer semitone or two an
exceptional player can reach is left out, since a generator has no way
to know it has a virtuoso reading the part.

Before this table existed, the composer had one melody band, 64-84, for
all sixty-six instruments: `compose()` returned a byte-identical score
for a tuba and a piccolo, and the range gate was the piano's applied to
every voice. The band now comes from the instrument, so the melody a
tuba gets is a tuba's.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InstrumentRange:
    """One instrument's compass, comfortable core, and polyphony.

    `low_midi`/`high_midi` bound what the linter will accept;
    `tessitura_low`/`tessitura_high` bound what the composer will write
    when the instrument is carrying a line. Both pairs are inclusive.
    """

    low_midi: int
    high_midi: int
    tessitura_low: int
    tessitura_high: int
    polyphony: int

    def __post_init__(self) -> None:
        if not self.low_midi <= self.tessitura_low <= self.tessitura_high <= self.high_midi:
            raise ValueError(
                f"tessitura {self.tessitura_low}-{self.tessitura_high} must sit "
                f"inside the compass {self.low_midi}-{self.high_midi}"
            )
        if self.polyphony < 1:
            raise ValueError(f"polyphony must be at least 1, got {self.polyphony}")

    def contains(self, pitch_midi: int) -> bool:
        """Whether a pitch is inside this instrument's compass."""
        return self.low_midi <= pitch_midi <= self.high_midi


# The width of the window a *line* is written in, in semitones — both the
# floor and the ceiling of it. The floor is the degree walk's: a line
# reaching the band's edge has to have somewhere to enter from, and a band
# narrower than a twelfth leaves a wide line with no octave that fits. The
# ceiling is the music's: the tessitura is everything the instrument is
# comfortable with, which for a piano, a harp or a section patch is three
# octaves — comfortably *loud* across all of it, but not a melody. A tune
# that wanders over three octaves is not a tune, and it is also what left
# the accompaniment nowhere to sit. See the register notes in
# `compose.engine`. An instrument whose whole compass is narrower than
# this gets its compass, since there is nothing else to give it.
LINE_BAND_SEMITONES: int = 21


@dataclass(frozen=True)
class MelodyBand:
    """A window of semitones one instrument's part is written inside."""

    low_midi: int
    high_midi: int

    @property
    def centre_midi(self) -> int:
        """The pitch a placement is measured against when nothing else
        constrains it: the middle of the band."""
        return (self.low_midi + self.high_midi) // 2

    def contains(self, pitch_midi: int) -> bool:
        return self.low_midi <= pitch_midi <= self.high_midi


# Idiomatic compasses in MIDI note numbers. Keys must match
# `saimc.spec.Instrument` 1:1; a drift-guard test asserts it, the same
# way `render.instruments.INSTRUMENT_PROGRAMS` is guarded.
INSTRUMENT_RANGES: dict[str, InstrumentRange] = {
    # Keys. A piano's whole compass is the instrument; the band a melody
    # is written in is the singing middle, which is why the tessitura is
    # not simply the compass.
    "piano": InstrumentRange(21, 108, 48, 84, 10),
    "harpsichord": InstrumentRange(29, 89, 48, 84, 10),
    "celesta": InstrumentRange(60, 108, 72, 96, 4),
    "music_box": InstrumentRange(60, 96, 72, 96, 2),
    # Mallets and bells
    "glockenspiel": InstrumentRange(79, 108, 84, 103, 2),
    "vibraphone": InstrumentRange(53, 89, 60, 84, 4),
    "marimba": InstrumentRange(45, 96, 60, 84, 4),
    "xylophone": InstrumentRange(65, 108, 72, 96, 2),
    "tubular_bells": InstrumentRange(60, 91, 65, 88, 2),
    "dulcimer": InstrumentRange(48, 86, 55, 84, 4),
    # Organs and free reeds
    "pipe_organ": InstrumentRange(36, 96, 48, 84, 10),
    "accordion": InstrumentRange(41, 89, 48, 84, 6),
    "harmonica": InstrumentRange(60, 96, 65, 89, 1),
    # Plucked strings
    "nylon_guitar": InstrumentRange(40, 88, 52, 84, 6),
    "steel_guitar": InstrumentRange(40, 88, 52, 84, 6),
    "banjo": InstrumentRange(50, 88, 55, 84, 5),
    "shamisen": InstrumentRange(57, 88, 60, 84, 3),
    "koto": InstrumentRange(48, 89, 55, 84, 6),
    "sitar": InstrumentRange(48, 88, 52, 84, 4),
    # Bowed strings and ensembles. The section patches are many players
    # and can divide; the solo instruments are one line each.
    "violin": InstrumentRange(55, 103, 62, 93, 2),
    "viola": InstrumentRange(48, 91, 55, 84, 2),
    "cello": InstrumentRange(36, 81, 45, 72, 2),
    "contrabass": InstrumentRange(28, 67, 33, 55, 1),
    "tremolo_strings": InstrumentRange(36, 96, 48, 84, 8),
    "pizzicato_strings": InstrumentRange(36, 96, 48, 84, 8),
    "strings": InstrumentRange(36, 96, 48, 84, 8),
    "fiddle": InstrumentRange(55, 103, 62, 93, 2),
    # Harp and timpani. A timpano is pitched but tuned to the piece and
    # has a handful of notes, so its band is its compass.
    "harp": InstrumentRange(24, 103, 48, 84, 10),
    "timpani": InstrumentRange(38, 57, 38, 57, 2),
    # Choir — an ensemble patch, but a choir part is a line.
    "choir": InstrumentRange(43, 79, 48, 76, 4),
    # Brass. All single-line: one player, one resonator.
    "french_horn": InstrumentRange(34, 77, 41, 65, 1),
    "brass_section": InstrumentRange(34, 84, 45, 76, 4),
    "trumpet": InstrumentRange(52, 82, 58, 77, 1),
    "muted_trumpet": InstrumentRange(52, 82, 58, 77, 1),
    "trombone": InstrumentRange(40, 72, 45, 67, 1),
    "tuba": InstrumentRange(28, 58, 33, 53, 1),
    # Woodwinds. All single-line.
    "flute": InstrumentRange(60, 96, 65, 89, 1),
    "piccolo": InstrumentRange(74, 108, 79, 100, 1),
    "recorder": InstrumentRange(60, 96, 65, 89, 1),
    "pan_flute": InstrumentRange(60, 89, 65, 86, 1),
    "ocarina": InstrumentRange(60, 89, 65, 84, 1),
    "oboe": InstrumentRange(58, 91, 65, 86, 1),
    "english_horn": InstrumentRange(52, 81, 58, 77, 1),
    "bassoon": InstrumentRange(34, 75, 41, 67, 1),
    "clarinet": InstrumentRange(50, 91, 55, 82, 1),
    # Saxophones, at sounding pitch.
    "soprano_sax": InstrumentRange(52, 87, 58, 82, 1),
    "alto_sax": InstrumentRange(49, 80, 55, 77, 1),
    "tenor_sax": InstrumentRange(44, 75, 49, 70, 1),
    "baritone_sax": InstrumentRange(36, 68, 41, 63, 1),
    # World
    "bagpipe": InstrumentRange(55, 86, 60, 84, 1),
    "shakuhachi": InstrumentRange(55, 86, 60, 81, 1),
    "shanai": InstrumentRange(58, 89, 65, 84, 1),
    "kalimba": InstrumentRange(60, 84, 60, 84, 2),
    "steel_drums": InstrumentRange(52, 81, 55, 79, 4),
    "agogo": InstrumentRange(60, 84, 62, 84, 2),
    "woodblock": InstrumentRange(60, 84, 65, 84, 1),
    "taiko": InstrumentRange(36, 60, 38, 58, 3),
    # The kit: pitches here ARE the drum pieces (kick 36, snare 38, …),
    # not a scale, so the band is only a total answer for a voice that
    # never actually carries a melody — `default_ensemble` puts the
    # piano on the melody when a spec asks for a drum kit.
    "drum_set": InstrumentRange(35, 81, 48, 69, 8),
    # Dedicated-font instruments.
    "harmonium": InstrumentRange(36, 84, 48, 79, 4),
    "bansuri": InstrumentRange(60, 91, 65, 86, 1),
    "sarangi": InstrumentRange(48, 84, 55, 79, 1),
    "rudra_veena": InstrumentRange(36, 84, 45, 76, 4),
    "sarasvati_veena": InstrumentRange(48, 84, 52, 79, 4),
    "qanoon": InstrumentRange(48, 89, 55, 84, 6),
    "ud": InstrumentRange(48, 84, 52, 79, 5),
    "kora": InstrumentRange(48, 89, 55, 84, 6),
}


def range_for(instrument: str) -> InstrumentRange:
    """The compass for an instrument name.

    Raises `KeyError` for an unknown name rather than inventing a
    default: a silent fallback to the piano's compass is exactly the bug
    this table exists to end.
    """
    return INSTRUMENT_RANGES[instrument]


def melody_band(instrument: str) -> MelodyBand:
    """The window this instrument's melody is placed inside.

    `LINE_BAND_SEMITONES` wide around the middle of the tessitura, then
    moved — not shrunk — to fit the compass, so the band can never ask for
    a note the range gate would reject and a line still has the twelfth
    the walk needs. An instrument whose whole compass is narrower than
    that gets its compass, since there is nothing else to give it.

    Moving rather than shrinking matters for the instruments whose
    comfortable range is narrower than the band: clamping both ends
    independently would leave a window narrower than the band (a woodblock
    at twenty semitones) and then, by the "narrower than the band" rule,
    hand the whole twenty-four-semitone compass instead of the twelve the
    line wants. Sliding the window into the compass gives exactly the band
    wherever the compass can hold it.

    The clamp is what keeps the two halves of this module agreeing: the
    composer writes inside `melody_band`, the linter enforces
    `range_for`, and `melody_band(i) ⊆ range_for(i)` holds for every
    instrument in the table.

    This is the band a *line* is written in, and it is narrower than the
    tessitura for most instruments on purpose. The accompaniment gets
    `bed_window` instead — the whole comfortable range — because a pad is
    a texture rather than a line, and because the wider window is what
    gives its pitch classes somewhere to fold to.
    """
    span = range_for(instrument)
    if span.high_midi - span.low_midi < LINE_BAND_SEMITONES:
        return MelodyBand(low_midi=span.low_midi, high_midi=span.high_midi)
    centre = (span.tessitura_low + span.tessitura_high) // 2
    low = centre - LINE_BAND_SEMITONES // 2
    low = min(max(low, span.low_midi), span.high_midi - LINE_BAND_SEMITONES)
    return MelodyBand(low_midi=low, high_midi=low + LINE_BAND_SEMITONES)


def bed_window(instrument: str) -> MelodyBand:
    """The window this instrument's accompaniment is written inside.

    The whole comfortable range, which is the right window for a bed for
    two reasons. A pad is a texture, not a line: it has no melody to keep
    in a twelfth, and the register it *sounds* in is settled later against
    the tune, over the finished piece. And the fold that puts a chord tone
    into the bed moves in octaves, so the window has to hold enough of
    them for every pitch class — an instrument's comfortable range always
    spans an octave, so it always does, and a narrower window would leave
    the bed unable to sound a chord tone it is entitled to.

    The range is the tessitura rather than the compass for the same
    reason a melody uses it: the outer semitones an exceptional player can
    reach are not where a pad should be living. A celesta's lowest note is
    C4 and its comfortable range starts an octave above it, which is why a
    celesta pad is written high and sounds high.
    """
    span = range_for(instrument)
    return MelodyBand(low_midi=span.tessitura_low, high_midi=span.tessitura_high)


@dataclass(frozen=True)
class BedRegisters:
    """Everywhere one accompaniment voice may be written.

    Two windows, and the second is the fallback. `comfortable` is
    `bed_window` — the register the bed's chords are voiced in and where
    the instrument sounds like itself. `compass` is `range_for`'s — the
    whole instrument, which the composer falls back to when the tune
    leaves the comfortable range no room clear of it, and which the
    linter accepts either way.

    The fallback exists because a tune can fill its own comfortable range
    and more: a piccolo is written 79-100 and a celesta's comfortable
    range starts at C5, so a celesta pad under a piccolo tune has four
    semitones of room and no accompaniment it can write there. Four
    octaves below its top, the celesta still has sixty semitones of
    compass it can sound, and a pad a fifth under the tune is worth more
    than a pad in the tune's own octave — which is the only alternative
    the comfortable range leaves.
    """

    comfortable: MelodyBand
    compass: MelodyBand


def bed_registers(instrument: str) -> BedRegisters:
    """Both windows an accompaniment for this instrument may be written in."""
    span = range_for(instrument)
    return BedRegisters(
        comfortable=bed_window(instrument),
        compass=MelodyBand(low_midi=span.low_midi, high_midi=span.high_midi),
    )


__all__ = [
    "INSTRUMENT_RANGES",
    "LINE_BAND_SEMITONES",
    "BedRegisters",
    "InstrumentRange",
    "MelodyBand",
    "bed_registers",
    "bed_window",
    "melody_band",
    "range_for",
]
