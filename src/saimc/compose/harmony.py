"""The harmony bed: how many voices sound, where they sit, and how they settle.

The bed's job is to be under the tune and not in it. Every function here is a
way of saying that once: `melody_band_for` finds the window the melody owns,
`settle_harmony_register` drops the bed clear of it, and the private helpers
below decide what "clear" means when the instrument's own range disagrees.
"""

from __future__ import annotations

import random
from collections.abc import Mapping
from dataclasses import replace

from saimc.compose.duration import (
    HARMONY_TEXTURE_CYCLE,
    HARMONY_TEXTURE_FIRST,
    HARMONY_TEXTURE_REST,
)
from saimc.compose.dynamics import shaped_velocity
from saimc.compose.score import (
    PPQ,
    VOICE_HARMONY,
    VOICE_MELODY,
    NoteEvent,
)
from saimc.compose.voices import (
    DEFAULT_HARMONY_VOICES,
    HARMONY_MELODY_CLEARANCE,
    HARMONY_STAB_INSTRUMENTS,
    HarmonyVoices,
)
from saimc.instruments import (
    LINE_BAND_SEMITONES,
    BedRegisters,
    MelodyBand,
    bed_window,
    melody_band,
    range_for,
)

# The harmony voice's constants. The texture it states, the figure that
# texture steps on, the level each sounds at and the clearance between
# the bed and the tune all moved to `saimc.compose.voices`, because the
# plan carries them and the plan cannot import this module. There is no
# register window among this module's remaining constants: the bed is
# folded into its *instrument's* window (`instruments.bed_window`, the
# comfortable range), because "where this instrument sounds like itself"
# does not depend on which role it is playing. Until it moved, the bed
# was folded into a module constant, MIDI 48-84, for every pad
# instrument, so a celesta (lowest note C4 = 60, lives 72-96) had its
# pad written 48-59, a twelfth below the instrument — and the range gate
# could not see it. Which side of the tune the bed settles on is decided
# over the finished piece by `settle_harmony_register`, against the
# tune's band and the room the accompaniment's own instrument has for
# it.
HARMONY_CROWD_INTERVALS: frozenset[int] = frozenset({0, 1, 2, 10, 11})
# The room the accompaniment needs underneath the tune: one octave, so
# that every one of the twelve pitch classes has an octave of the bed's
# own to be folded into. Narrower than this and the bed can sound some
# chord tones under the tune and not others — the pad answers a D with a
# D two octaves up against a C it answers two octaves down, which is a
# hole in the voicing and a leap in a voice that never leaps. The line
# is raised inside its own instrument to leave this room, and a bed with
# less than it goes above the tune instead; see `melody_band_for`.
BED_OCTAVE_SEMITONES: int = 12
# The least room a bed can be written in and still be a bed: half an
# octave, the span that covers seven of the twelve pitch classes. A
# window with less than this on the side the bed has chosen cannot voice
# a chord there — every chord tone with no octave is dropped — so the
# settle pass reaches for the instrument's whole compass instead; see
# `_can_clear_the_tune`.
BED_MIN_ROOM_SEMITONES: int = BED_OCTAVE_SEMITONES // 2


def active_harmony_voices(
    voices: tuple[tuple[int, str], ...],
    *,
    section_index: int,
    long_piece: bool,
    cycle: tuple[str, ...] = HARMONY_TEXTURE_CYCLE,
) -> tuple[tuple[int, str], ...]:
    """Shape long-form density instead of looping one wall of sound.

    The cycle names a group of voices per phase of the arc, repeating
    every four sections: with two harmony colors the opening presents the
    first, the third section becomes a contrasting breakdown led by the
    second, and the intervening sections combine them.

    A group, not an index list, because how many harmony voices a piece
    has is the ensemble's decision — "the leading one" and "every voice
    but it" mean the same thing at two voices and at five.
    """
    if not long_piece or len(voices) < 2:
        return voices
    group = cycle[section_index % len(cycle)]
    if group == HARMONY_TEXTURE_FIRST:
        return voices[:1]
    if group == HARMONY_TEXTURE_REST:
        return voices[1:]
    return voices


def melody_band_for(
    *,
    melody: str,
    bed: str | None,
    band_semitones: int = LINE_BAND_SEMITONES,
    clearance: int = HARMONY_MELODY_CLEARANCE,
) -> MelodyBand:
    """The window the tune is written in, given what plays underneath it.

    An instrument's own band is where its melody would go if it played
    alone. A piece is not that: the accompaniment needs a register of its
    own, and the register it needs is an octave of its instrument's
    comfortable range, under the tune. So the tune is raised inside its
    own range until that room exists — a piano tune over a string pad is
    an F3-C5 band raised to an E♭4-C6 one, which is where the two
    instruments can both be heard — and it keeps the band's width, so the
    line still has the twelfth the walk needs.

    The raise is bounded by the tune instrument's own comfort, and that
    bound is the point rather than a detail. A tuba has nothing above a
    string pad's floor to be raised into, and a cello cannot get clear of
    a piano's; in those pairs the raise does not happen, the bed's
    instrument has no octave underneath the tune, and
    `settle_harmony_register` writes the bed *above* it instead. Which
    side of the tune the bed ends up on is therefore decided here, once,
    by what the two instruments can do — not per bar, which would move a
    held pad note an octave mid-chord.

    With no accompaniment voice the tune sits in its instrument's own
    band, which is what a solo piece is.

    The clearance is an argument for the reason the band's width is one:
    both are the plan's, and neither is a fact about the instrument, so
    this function is handed them rather than reaching for a table.
    """
    band = melody_band(melody, band_semitones=band_semitones)
    if bed is None:
        return band
    window = bed_window(bed)
    floor = window.low_midi + BED_OCTAVE_SEMITONES + clearance
    if floor <= band.low_midi:
        return band
    span = range_for(melody)
    if floor > span.tessitura_high - band_semitones:
        return band
    return MelodyBand(low_midi=floor, high_midi=floor + band_semitones)


def _octaves_in_window(pitch: int, window: MelodyBand) -> list[int]:
    """Every octave of `pitch` inside `window`, lowest first.

    An octave shift is the only move a bed note has: it keeps the pitch
    class, and so the chord tone the note was written as. A window
    narrower than an octave can miss a pitch class entirely, and an
    empty list is the honest answer there — the caller drops the note
    rather than inventing a pitch the instrument cannot sound.

    Only `bed_window` windows are passed, and those span an instrument's
    comfortable range, so every pitch class has an octave in one and the
    empty case is for an instrument whose whole compass is narrower than
    an octave — for which there is no honest note to write.
    """
    octave = pitch
    while octave > window.high_midi:
        octave -= 12
    while octave < window.low_midi:
        octave += 12
    if octave > window.high_midi:
        return []
    while octave - 12 >= window.low_midi:
        octave -= 12
    octaves: list[int] = []
    while octave <= window.high_midi:
        octaves.append(octave)
        octave += 12
    return octaves


def _into_harmony_register(pitch: int, *, window: MelodyBand) -> int:
    """Octave-shift a chord tone into the harmony bed's register.

    The fold moves in the direction it has to: down while the pitch is
    over the window, then up while it is under it, which is what the
    fixed 48-84 fold did when the window was those two constants. An
    octave shift keeps the pitch class (and so the chord tone); a clamp
    would not. A pitch class the window does not contain cannot be
    folded at all and comes back out of the window untouched —
    `settle_harmony_register` is what drops it, because the note is
    unplayable either way and only one of the two is honest about it.
    """
    while pitch > window.high_midi:
        pitch -= 12
    while pitch < window.low_midi:
        pitch += 12
    return pitch


def generate_harmony_section(
    *,
    chords: list[tuple[int, tuple[int, ...], int]],
    section_start_tick: int,
    ticks_per_bar: int,
    rng: random.Random,
    melody_from_bar: int,
    seed_for_variation: int,
    voice_id: int = VOICE_HARMONY,
    instrument: str = "piano",
    window: MelodyBand,
    layer_index: int = 0,
    layer_count: int = 1,
    voices: HarmonyVoices = DEFAULT_HARMONY_VOICES,
) -> list[NoteEvent]:
    """Generate the harmony voice for one section from the resolved chords.

    The plan's texture decides the shape: a sustained pad, a broken-chord
    ostinato, or — for an instrument that does that well — spacious chord
    accents. A broken chord is stated by the leading harmony layer only;
    additional harmony colors form a quieter sustained bed. Intro bars
    stay silent — harmony enters with the melody.

    The bed is written inside `window` — the instrument's own, from
    `saimc.instruments` — so a pad is written where the instrument that
    plays it can sound. Where that window is relative to the tune is
    settled later, over the finished piece, by
    `settle_harmony_register`; nothing here reads the tune, so the RNG
    draws do not depend on what it happens to do.
    """
    notes: list[NoteEvent] = []
    pad = not voices.broken_chord or layer_index > 0
    stabs = voices.broken_chord and instrument in HARMONY_STAB_INSTRUMENTS
    # Which chord tone sits lowest: the rotation (not the bar) decides
    # it, so the section's voicing stays stable instead of churning.
    rotation = rng.randrange(3)
    step_ticks = voices.arpeggio_step_ticks
    arpeggio_steps = max(1, ticks_per_bar // step_ticks)

    cursor = 0
    bar_index = 0
    total_bars = sum(dur for _, _, dur in chords)
    for chord_root, chord_tones, dur in chords:
        for _bar in range(dur):
            bar_tick = section_start_tick + cursor + _bar * ticks_per_bar
            if bar_index < melody_from_bar:
                bar_index += 1
                continue
            position = bar_index / max(1, total_bars)
            if stabs:
                pulse_duration = max(PPQ // 2, min(PPQ, ticks_per_bar // 4))
                for pulse_index, pulse_tick in enumerate((0, ticks_per_bar // 2)):
                    low = (bar_index + rotation + pulse_index) % len(chord_tones)
                    pair = (chord_tones[low], chord_tones[(low + 2) % len(chord_tones)])
                    for tone in pair:
                        notes.append(
                            NoteEvent(
                                voice_id=voice_id,
                                pitch_midi=_into_harmony_register(chord_root + tone, window=window),
                                tick=bar_tick + pulse_tick,
                                duration_ticks=pulse_duration,
                                velocity=shaped_velocity(
                                    base=voices.stab_velocity,
                                    position=position,
                                    tick=bar_tick + pulse_tick,
                                    ticks_per_bar=ticks_per_bar,
                                    rng_seed=seed_for_variation + bar_tick * 103 + pulse_index,
                                ),
                            )
                        )
            elif pad:
                # Divisi: layer *i* starts from the tone *i* steps along the
                # chord, so the pads voice different inversions rather than the
                # same dyad in different registers. A lone pad keeps the dyad —
                # there is nobody to share with — which is what makes this a
                # change to ensembles and not to solo-plus-pad pieces.
                spread = layer_index if voices.divisi else 0
                low = (bar_index + rotation + spread) % len(chord_tones)
                sounding: tuple[int, ...] = (
                    (chord_tones[low],)
                    if voices.divisi and layer_count > 1
                    else (chord_tones[low], chord_tones[(low + 2) % len(chord_tones)])
                )
                for tone in sounding:
                    notes.append(
                        NoteEvent(
                            voice_id=voice_id,
                            pitch_midi=_into_harmony_register(chord_root + tone, window=window),
                            tick=bar_tick,
                            duration_ticks=ticks_per_bar,
                            velocity=shaped_velocity(
                                base=voices.pad_velocity - layer_index * 4,
                                position=position,
                                tick=bar_tick,
                                ticks_per_bar=ticks_per_bar,
                                rng_seed=seed_for_variation + bar_tick,
                            ),
                        )
                    )
            else:
                for step in range(arpeggio_steps):
                    tone = chord_tones[(step + rotation) % len(chord_tones)]
                    notes.append(
                        NoteEvent(
                            voice_id=voice_id,
                            pitch_midi=_into_harmony_register(chord_root + tone, window=window),
                            tick=bar_tick + step * step_ticks,
                            duration_ticks=step_ticks,
                            velocity=shaped_velocity(
                                base=voices.arpeggio_velocity,
                                position=position,
                                tick=bar_tick + step * step_ticks,
                                ticks_per_bar=ticks_per_bar,
                                rng_seed=seed_for_variation + bar_tick * 101 + step,
                            ),
                        )
                    )
            bar_index += 1
        cursor += dur * ticks_per_bar

    # Register and clearance are settled over the finished piece, by
    # `settle_harmony_register`. The bed is written where the instrument
    # sounds and left alone here: a pass in this loop would be fixing a
    # register without knowing the tune it has to clear.
    return notes


def _crowds_melody(note: NoteEvent, melody_notes: list[NoteEvent], pitch: int) -> bool:
    """Does a harmony pitch collide with any simultaneously sounding melody note?"""
    return any(
        melody.tick < note.tick + note.duration_ticks
        and note.tick < melody.tick + melody.duration_ticks
        and abs(melody.pitch_midi - pitch) in HARMONY_CROWD_INTERVALS
        for melody in melody_notes
    )


def settle_harmony_register(
    notes: list[NoteEvent],
    *,
    melody_floor: int,
    melody_ceiling: int,
    registers: Mapping[int, BedRegisters],
    clearance: int = HARMONY_MELODY_CLEARANCE,
) -> list[NoteEvent]:
    """Settle every harmony voice inside its instrument's window, clear of the tune.

    Two constraints, and the first is not negotiable: a note must be
    inside its instrument's window, because a celesta cannot sound a C3
    and the gate that used to check this was the piano's compass applied
    to every voice. The second is that it should stay `clearance` clear
    of the tune's band — a bed note inside the melody's register is what
    `tessitura_overlap_semitones` measures and what a hot pot of
    instruments sounds like.

    Each voice is settled on *one* side of the tune, and which side is
    the voice's decision rather than each note's. Under the tune is
    where a bed belongs, and it is where every voice goes whose window
    can hold an octave down there — an octave being what gives each
    chord tone a place to fold to. A window that cannot (a celesta's
    comfortable range starts at C5, and a cello's melody sits too low
    for the strings under it to reach below) puts the bed above the
    tune instead of losing it. `melody_band_for` is what makes the
    first case the common one — the tune is raised so that an octave
    underneath exists — and this pass is what handles the pairs where
    it cannot be raised.

    The side is the voice's rather than the note's because a bed with
    notes either side of the tune is not a bed: it is the register this
    pass exists to keep out of, and it is what a per-note choice
    produced the moment the octave under the tune happened to rub a
    melody note — that one note jumped over the tune, held there for a
    bar, and left the voice's range closed around the melody's.

    A note the chosen side has no octave for that clears the tune is
    dropped, which is the same accepted outcome the crowding pass has.
    Keeping it instead would put a pad note in the tune's own band, and
    a bed that shares the tune's register is the one thing this pass
    exists to prevent; it is also, measured over the gate matrix, one or
    two notes a piece rather than a hole in the texture, because the
    arrangement has already left the bed an octave to fold into.

    The bounds are the piece's, not the bar's, and deliberately so. The
    tune's floor and ceiling are one number each for the whole piece, so
    a bar whose melody happens to sit high cannot admit a high bed note
    that is inside the melody's band for the piece as a whole. It also
    keeps the bed in one register: a bar-local bound would move one pad
    note an octave between two bars — a leap in a voice that never
    leaps, in the middle of a held chord.

    A note that moves is asked the crowding question again, because an
    octave shift can land it a seventh under a melody note it never met
    where it was.

    It runs over the finished piece, after every section and the coda,
    for the same reason the crowding pass runs after generation: the
    draws must not depend on what the tune did. It changes pitches only —
    the arpeggio's onsets, the pad's held bars and the stabs' pulses are
    all left where they were.
    """
    below_ceiling = melody_floor - clearance
    above_floor = melody_ceiling + clearance
    melody = [note for note in notes if note.voice_id == VOICE_MELODY]
    settled = [note for note in notes if note.voice_id not in registers]
    for voice_id, voice_registers in registers.items():
        settled.extend(
            _settle_voice(
                [note for note in notes if note.voice_id == voice_id],
                registers=voice_registers,
                melody=melody,
                below_ceiling=below_ceiling,
                above_floor=above_floor,
            )
        )
    return settled


def _settle_voice(
    voice_notes: list[NoteEvent],
    *,
    registers: BedRegisters,
    melody: list[NoteEvent],
    below_ceiling: int,
    above_floor: int,
) -> list[NoteEvent]:
    """Settle one harmony voice on one side of the tune.

    The side is decided once, for the voice, by `_bed_goes_above`; the
    notes then fold to the octave of that side nearest the tune that does
    not rub a simultaneously sounding melody note. A note the side has no
    such octave for is dropped — see `settle_harmony_register`.

    The comfortable range is tried first and the whole compass only if the
    tune leaves it no register clear of itself; a voice with neither is
    kept where the harmony pass wrote it rather than emptied.
    """
    for window in (registers.comfortable, registers.compass):
        above = _bed_goes_above(window, below_ceiling=below_ceiling, above_floor=above_floor)
        if _can_clear_the_tune(
            window, below_ceiling=below_ceiling, above_floor=above_floor, above=above
        ):
            break
    else:
        # Neither window has a register clear of the tune — a compass
        # forty semitones wide with a tune filling twenty-one of them
        # leaves nowhere to fold to. The notes are kept where the harmony
        # pass wrote them, because a bed in the tune's register is worth
        # more than no bed at all; the ones that rub the tune still go,
        # since a clash is never what keeping the register is for.
        return [note for note in voice_notes if not _crowds_melody(note, melody, note.pitch_midi)]
    settled: list[NoteEvent] = []
    for note in voice_notes:
        if (
            not above
            and note.pitch_midi <= below_ceiling
            and window.contains(note.pitch_midi)
            and not _crowds_melody(note, melody, note.pitch_midi)
        ):
            # Already under the tune, inside the bed's window, and clear
            # of it: there is nothing to move it to.
            settled.append(note)
            continue
        octaves = _octaves_in_window(note.pitch_midi, window)
        candidates = [
            octave
            for octave in octaves
            if (octave >= above_floor if above else octave <= below_ceiling)
        ]
        # Nearest the tune first — the highest octave under it, the
        # lowest over it — so the bed sits as close to the tune as its
        # instrument allows without either entering it.
        pitch = next(
            (
                octave
                for octave in (candidates if above else reversed(candidates))
                if not _crowds_melody(note, melody, octave)
            ),
            None,
        )
        if pitch is not None:
            settled.append(note if pitch == note.pitch_midi else replace(note, pitch_midi=pitch))
    return settled


def _bed_goes_above(window: MelodyBand, *, below_ceiling: int, above_floor: int) -> bool:
    """Whether this voice's accompaniment sits above the tune rather than under it.

    Under, whenever the window holds a whole octave there: a bed belongs
    under a tune, and an octave is what gives every chord tone a place to
    fold to. `BED_OCTAVE_SEMITONES` is the same octave `melody_band_for`
    raises the tune to leave room for, so a pairing that could be raised
    lands here with exactly an octave of room and goes under.

    Only a window that cannot fit an octave underneath is asked which
    side it prefers, and then it is the roomier one. That is the celesta
    over a low tune — its comfortable range starts above the tune's top,
    so it has no room underneath at all and a bed there sounds high,
    which is where the instrument lives anyway.
    """
    under_room = below_ceiling - window.low_midi
    over_room = window.high_midi - above_floor
    if under_room >= BED_OCTAVE_SEMITONES:
        return False
    return over_room >= BED_OCTAVE_SEMITONES or over_room > under_room


def _can_clear_the_tune(
    window: MelodyBand, *, below_ceiling: int, above_floor: int, above: bool
) -> bool:
    """Whether a bed written in this window has room to clear the tune.

    `BED_MIN_ROOM_SEMITONES` on the side the bed has chosen, which is the
    span that covers seven of the twelve pitch classes. Below that the
    window cannot voice a chord: every chord tone with no octave there is
    dropped, so what is written is not a smaller bed but a bed with holes
    in it, sounding the three or four pitch classes the sliver holds.

    A window with no such room is not used — `_settle_voice` tries the
    instrument's compass next, and keeps the voice where the harmony pass
    wrote it if that has no room either.
    """
    room = below_ceiling - window.low_midi if not above else window.high_midi - above_floor
    return room >= BED_MIN_ROOM_SEMITONES


__all__ = [
    "BED_MIN_ROOM_SEMITONES",
    "BED_OCTAVE_SEMITONES",
    "HARMONY_CROWD_INTERVALS",
    "active_harmony_voices",
    "generate_harmony_section",
    "melody_band_for",
    "settle_harmony_register",
]
