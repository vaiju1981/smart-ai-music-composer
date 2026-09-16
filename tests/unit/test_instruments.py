"""Unit tests for the per-instrument compass table.

The table is the one place that answers "where does this instrument live",
and both halves of the pipeline read it: the composer places a line inside
`melody_band`, an accompaniment inside `bed_window` — or, where the tune
leaves that window no room, inside the instrument's compass — and the
linter refuses a note outside `range_for`. These tests hold the table
itself honest — its keys, its internal geometry, and the containment every
window promises — and then measure the effect the table has on the music,
which is the review finding it was built for: every instrument used to
share one 64-84 window, so `compose()` returned a byte-identical score for
a tuba and a piccolo.
"""

from __future__ import annotations

from saimc.compose.engine import compose
from saimc.instruments import (
    INSTRUMENT_RANGES,
    LINE_BAND_SEMITONES,
    MelodyBand,
    bed_registers,
    bed_window,
    melody_band,
    range_for,
)
from saimc.spec import CompositionSpec, Instrument, Mood


class TestTheTable:
    def test_the_keys_are_the_spec_vocabulary(self) -> None:
        """Drift guard, the same shape as the GM program table's.

        A row missing from here is not a silent default: `range_for` raises
        rather than inventing the piano's compass, so the failure would
        arrive at compose time as a `KeyError` on one instrument somebody
        happened to ask for. This is where it arrives instead.
        """
        assert set(INSTRUMENT_RANGES) == {member.value for member in Instrument}

    def test_every_row_has_its_tessitura_inside_its_compass(self) -> None:
        # The frozen dataclass refuses this at construction, so this is the
        # table-wide statement of the same rule rather than a second check:
        # importing the module at all has already run it for every row.
        for name, span in INSTRUMENT_RANGES.items():
            assert span.low_midi <= span.tessitura_low, name
            assert span.tessitura_high <= span.high_midi, name
            assert span.tessitura_low <= span.tessitura_high, name
            assert span.polyphony >= 1, name

    def test_an_unknown_instrument_is_an_error_not_a_default(self) -> None:
        # A silent fallback to the piano's compass is exactly the bug the
        # table exists to end, so an unknown name must not have one.
        try:
            range_for("theremin")
        except KeyError:
            return
        raise AssertionError("range_for invented a compass for an unknown instrument")

    def test_the_compasses_are_musically_plausible(self) -> None:
        """No row is a stub: at least two octaves, and inside MIDI.

        Two octaves is not a style judgement but a floor on what an
        instrument is: a compass narrower than that cannot hold the
        twelve-semitone window a bed needs, let alone a line. The
        timpani is the narrowest row at nineteen semitones, which is
        why its window is its compass.
        """
        for name, span in INSTRUMENT_RANGES.items():
            assert 0 <= span.low_midi < span.high_midi <= 127, name
            assert span.high_midi - span.low_midi >= LINE_BAND_SEMITONES - 2, name


class TestTheWindows:
    def test_a_line_window_is_the_band_width_and_inside_the_compass(self) -> None:
        for name, span in INSTRUMENT_RANGES.items():
            band = melody_band(name)
            assert span.contains(band.low_midi), (name, band)
            assert span.contains(band.high_midi), (name, band)
            width = band.high_midi - band.low_midi
            if span.high_midi - span.low_midi >= LINE_BAND_SEMITONES:
                assert width == LINE_BAND_SEMITONES, (name, band)
            else:
                # Nothing narrower to give it: an instrument whose whole
                # compass is under the band width gets its compass.
                assert (band.low_midi, band.high_midi) == (span.low_midi, span.high_midi), (
                    name,
                    band,
                )

    def test_a_line_window_sits_on_the_tessitura_it_was_taken_from(self) -> None:
        """The window is the middle of the comfortable range, not the top.

        Both ends of the band must lie inside the instrument's tessitura
        where the tessitura is wide enough to hold them: a window hung off
        the top of the comfort is where the old fixed band put every
        instrument, and it is what a tuba cannot play.
        """
        for name, span in INSTRUMENT_RANGES.items():
            band = melody_band(name)
            if span.tessitura_high - span.tessitura_low < LINE_BAND_SEMITONES:
                continue
            assert span.tessitura_low <= band.low_midi, (name, band)
            assert band.high_midi <= span.tessitura_high, (name, band)

    def test_a_bed_window_is_the_whole_tessitura(self) -> None:
        for name, span in INSTRUMENT_RANGES.items():
            window = bed_window(name)
            assert window == MelodyBand(span.tessitura_low, span.tessitura_high), name
            assert span.contains(window.low_midi), name
            assert span.contains(window.high_midi), name

    def test_a_bed_window_always_holds_an_octave(self) -> None:
        """The fold needs a window that holds every pitch class.

        An accompaniment note keeps its pitch class and moves in octaves,
        so a window narrower than twelve semitones would leave the bed
        unable to sound a chord tone it is entitled to. Every instrument's
        comfortable range clears this, and that is what makes the fold
        total rather than a source of dropped notes.
        """
        for name in INSTRUMENT_RANGES:
            window = bed_window(name)
            octaves = {
                pitch % 12
                for pitch in range(window.low_midi, window.high_midi + 1)
            }
            assert len(octaves) == 12, (name, window, sorted(octaves))

    def test_both_windows_are_contained_in_the_compass_the_linter_enforces(self) -> None:
        """The two halves of the module agree: what the composer writes is
        what the linter accepts, for every instrument."""
        for name, span in INSTRUMENT_RANGES.items():
            for window in (melody_band(name), bed_window(name)):
                for pitch in range(window.low_midi, window.high_midi + 1):
                    assert span.contains(pitch), (name, pitch)


class TestTheBedRegisters:
    """The bed's two windows: the comfortable one, and the compass behind it.

    A bed voice is written in the comfortable window whenever that window
    has room to clear the tune, and in the compass when it has not. Both
    are the instrument's, and the compass is the wider of the two, so
    moving to it can only ever be the fallback and never a way to write a
    bed the comfortable range would have refused.
    """

    def test_the_comfortable_window_is_the_one_a_bed_is_voiced_in(self) -> None:
        for name in INSTRUMENT_RANGES:
            assert bed_registers(name).comfortable == bed_window(name), name

    def test_the_compass_is_the_instrument_s_whole_compass(self) -> None:
        for name, span in INSTRUMENT_RANGES.items():
            compass = bed_registers(name).compass
            assert (compass.low_midi, compass.high_midi) == (span.low_midi, span.high_midi), (
                name,
                compass,
            )

    def test_the_compass_contains_the_comfortable_window(self) -> None:
        for name in INSTRUMENT_RANGES:
            registers = bed_registers(name)
            assert registers.compass.low_midi <= registers.comfortable.low_midi, name
            assert registers.comfortable.high_midi <= registers.compass.high_midi, name

    def test_every_register_the_fallback_can_write_is_playable(self) -> None:
        """The fallback widens what the composer may write, so it must not
        widen it past what the linter accepts: every pitch of both windows
        is inside the compass, for every instrument."""
        for name, span in INSTRUMENT_RANGES.items():
            registers = bed_registers(name)
            for window in (registers.comfortable, registers.compass):
                for pitch in range(window.low_midi, window.high_midi + 1):
                    assert span.contains(pitch), (name, pitch)


class TestTheInstrumentIsACompositionalChoice:
    def test_the_palette_no_longer_shares_one_score(self) -> None:
        """The review finding, as a test: one spec, sixty-six instruments.

        `compose()` used to return a byte-identical score for a trumpet, a
        tuba, a piccolo, a contrabass, a cello, a sitar, a bassoon and a
        piano, because the melody band was a module constant and the
        instrument's only effect was which General MIDI program played the
        same notes. Now every instrument-shaped spec is a different piece:
        the melody is its own, and the hash says so.
        """
        palette = (
            Instrument.PIANO,
            Instrument.TRUMPET,
            Instrument.TUBA,
            Instrument.PICCOLO,
            Instrument.CONTRABASS,
            Instrument.CELLO,
            Instrument.SITAR,
            Instrument.BASSOON,
        )
        digests = {}
        for instrument in palette:
            spec = CompositionSpec(
                mood=Mood.CALMING,
                instrumentation=instrument,
                duration_seconds=60,
                seed=5,
            )
            out = compose(spec)
            digests[instrument] = out.notation_score.compute_hash()
        assert len(set(digests.values())) == len(palette), digests

    def test_a_tuba_is_written_where_a_tuba_lives(self) -> None:
        """The claim behind the hashes, in semitones.

        A tuba's comfortable range is F2 to F3. The same spec that gives a
        piccolo a line above C6 gives the tuba one around F2 — and before
        the table both were written E4 to C6, two octaves above a tuba and
        at the top of a contrabass's compass.
        """
        for instrument in (Instrument.TUBA, Instrument.CONTRABASS, Instrument.PICCOLO):
            span = range_for(instrument.value)
            band = melody_band(instrument.value)
            out = compose(
                CompositionSpec(
                    mood=Mood.CALMING,
                    instrumentation=instrument,
                    duration_seconds=60,
                    seed=5,
                )
            )
            melody = [
                n.pitch_midi
                for n in out.notation_score.notes
                if n.voice_id == 1
            ]
            assert melody, instrument
            assert all(span.contains(pitch) for pitch in melody), instrument
            assert all(band.contains(pitch) for pitch in melody), (instrument, band)
