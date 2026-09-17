"""Unit tests for motif-driven melody generation (S5)."""

from __future__ import annotations

import itertools
import random
from collections import Counter

import pytest

from saimc.compose.engine import _melody_band_for, compose
from saimc.compose.motif import (
    BASS_FIGURES,
    DEFAULT_BASS_FIGURES,
    PLAIN_BASS_FIGURE,
    PPQ,
    Motif,
    MotifCell,
    _op_tie,
    draw_bass_figures,
    generate_motif,
    vary_motif,
)
from saimc.compose.score import VOICE_HARMONY, VOICE_MELODY
from saimc.instruments import MelodyBand, melody_band, range_for
from saimc.quality import score_piece
from saimc.spec import CompositionSpec, Instrument, Mood


def _melody(out) -> list:
    return sorted(
        (n for n in out.notation_score.notes if n.voice_id == VOICE_MELODY),
        key=lambda n: n.tick,
    )


def _melody_band(out) -> MelodyBand:
    """The band the score's own melody voice was written in.

    Read off the engine output rather than hard-coded: the band is the
    melody instrument's, so a test that wants to check the placement
    honours it has to ask which instrument carried the line — and ask the
    engine, not the instrument table, because a melody and an
    accompaniment are placed together: `_melody_band_for` raises the tune
    when the accompaniment's own range needs an octave underneath it, so
    a piano over a strings bed is written in 63-84 and a piano alone in
    56-77. Asserting against `melody_band` alone would fail the pieces
    the arrangement legitimately raised.
    """
    melody = None
    bed = None
    for voice in out.voice_instruments:
        if voice.voice_id == VOICE_MELODY:
            melody = voice.instrument
        elif voice.voice_id == VOICE_HARMONY and bed is None:
            bed = voice.instrument
    if melody is None:
        raise AssertionError("the score has no melody voice instrument")
    return _melody_band_for(melody=melody, bed=bed)


class TestGenerateMotif:
    def test_shape_fits_in_a_bar(self) -> None:
        rng = random.Random(7)
        for _ in range(200):
            motif = generate_motif(rng, bar_ticks=4 * PPQ)
            assert 2 <= len(motif) <= 8
            assert motif[0].step == 0
            total = sum(cell.length_ticks for cell in motif)
            assert total <= 4 * PPQ

    @pytest.mark.parametrize("bar_ticks", [PPQ, 2 * PPQ, 4 * PPQ])
    def test_every_cell_is_a_quarter_or_eighth(self, bar_ticks: int) -> None:
        rng = random.Random(99)
        for _ in range(50):
            motif = generate_motif(rng, bar_ticks=bar_ticks)
            assert motif
            for cell in motif:
                assert cell.length_ticks in (PPQ, PPQ // 2)

    def test_steps_are_small_walks(self) -> None:
        # Steps are scale degrees, not chord-tone indices: ±1 is a
        # semitone or a whole tone, which is what makes the line
        # conjunct. The wider steps are in the vocabulary so a leap exists
        # to be answered, and they stay rare.
        rng = random.Random(3)
        steps: list[int] = []
        for _ in range(100):
            motif = generate_motif(rng, bar_ticks=4 * PPQ)
            for cell in motif[1:]:
                assert cell.step in (-4, -3, -2, -1, 0, 1, 2, 3, 4)
                steps.append(cell.step)
        assert steps
        conjunct = sum(1 for step in steps if abs(step) <= 1)
        assert conjunct / len(steps) > 0.5
        assert any(abs(step) >= 3 for step in steps)

    def test_deterministic(self) -> None:
        a = generate_motif(random.Random(11), bar_ticks=4 * PPQ)
        b = generate_motif(random.Random(11), bar_ticks=4 * PPQ)
        assert a == b


class TestVaryMotif:
    def test_repetition_returns_the_motif_untouched(self) -> None:
        motif: Motif = (
            MotifCell(step=0, length_ticks=PPQ),
            MotifCell(step=1, length_ticks=PPQ),
        )
        # Find a repetition roll.
        rng = random.Random(0)
        for _ in range(200):
            variant = vary_motif(motif, rng)
            if variant.motif == motif and not variant.repeat:
                break
        else:
            pytest.fail("repetition never drawn in 200 rolls")

    def test_all_operations_are_reachable(self) -> None:
        motif: Motif = (
            MotifCell(step=0, length_ticks=PPQ),
            MotifCell(step=1, length_ticks=PPQ),
            MotifCell(step=-1, length_ticks=PPQ // 2),
        )
        seen: set[str] = set()
        for seed in range(500):
            variant = vary_motif(motif, random.Random(seed))
            if variant.repeat:
                seen.add("sequence")
            elif variant.anchor_offset:
                seen.add("transposition")
            elif variant.motif == motif:
                seen.add("repetition")
            elif len(variant.motif) < len(motif):
                seen.add("truncation")
            elif len(variant.motif) > len(motif):
                seen.add("ornament")
            else:
                seen.add("inversion")
        assert seen == {"repetition", "transposition", "sequence", "inversion", "truncation", "ornament"}

    def test_inversion_mirrors_steps(self) -> None:
        motif: Motif = (
            MotifCell(step=0, length_ticks=PPQ),
            MotifCell(step=2, length_ticks=PPQ),
        )
        rng = random.Random(1)
        for _ in range(500):
            variant = vary_motif(motif, rng)
            if variant.motif != motif and len(variant.motif) == len(motif) and not variant.repeat:
                assert all(
                    original.step == -mirrored.step
                    for original, mirrored in zip(motif[1:], variant.motif[1:], strict=True)
                )
                return
        pytest.fail("inversion never drawn in 500 rolls")

    def test_truncation_keeps_at_least_one_cell(self) -> None:
        motif: Motif = tuple(
            MotifCell(step=0 if i == 0 else 1, length_ticks=PPQ) for i in range(6)
        )
        for seed in range(500):
            variant = vary_motif(motif, random.Random(seed))
            assert len(variant.motif) >= 1


class TestOpTie:
    """The tie operation holds a chord tone across its beat boundary."""

    REMAINDERS = (0, 2, 4)

    def test_a_repeated_chord_tone_is_the_tie_to_prefer(self) -> None:
        """It changes no pitch: the line is held where the line already was."""
        tied = _op_tie(
            [[0, 480, 0, 0], [480, 480, 0, 0], [960, 480, 2, 0]],
            remainders=self.REMAINDERS,
        )
        assert tied == [(0, 480, 0, 1), (480, 480, 0, 0), (960, 480, 2, 0)]

    def test_a_walk_with_no_repeat_bends_its_first_chord_tone(self) -> None:
        """A walk that only ever moves has no pair to tie, and the bar drew
        the tie rhythm — so the first chord tone whose neighbour can join it
        is brought onto its pitch. That moves the line, so it is drawn only
        where the note after the tie is still a step away: the note is heard
        from the tie's own pitch, and a tie that leaves a third standing
        there has bought a longer note with a worse line."""
        bent = _op_tie(
            [[0, 480, 0, 0], [480, 480, 1, 0], [960, 480, 1, 0]],
            remainders=self.REMAINDERS,
        )
        assert bent == [(0, 480, 0, 1), (480, 480, 0, 0), (960, 480, 1, 0)]

        declined = _op_tie(
            [[0, 480, 0, 0], [480, 480, 1, 0], [960, 480, 2, 0]],
            remainders=self.REMAINDERS,
        )
        assert declined == [(0, 480, 0, 0), (480, 480, 1, 0), (960, 480, 2, 0)]

    def test_a_bar_with_no_chord_tone_keeps_its_rhythm(self) -> None:
        """A tie is only for a chord tone: two noteheads at one pitch are not
        a step apart, so tying a non-chord tone would leave it entered by a
        step and left by nothing."""
        untied = _op_tie(
            [[0, 480, 1, 0], [480, 480, 1, 0], [960, 480, 3, 0]],
            remainders=self.REMAINDERS,
        )
        assert untied == [(0, 480, 1, 0), (480, 480, 1, 0), (960, 480, 3, 0)]

    def test_a_tie_into_the_last_note_needs_no_note_after_it(self) -> None:
        tied = _op_tie([[0, 480, 2, 0], [480, 960, 2, 0]], remainders=self.REMAINDERS)
        assert tied == [(0, 480, 2, 1), (480, 960, 2, 0)]


class TestBassFigures:
    """The left hand's vocabulary: what a figure is, and how one is drawn."""

    def test_every_figure_states_its_landing_tone_on_the_bar_line(self) -> None:
        """Rung 0 at offset 0, in every figure of every mood.

        The bar's harmony has to sound on its downbeat whatever the left
        hand does afterwards, and the walk is anchored on that note — so
        a figure that opened anywhere else would move the harmony and the
        register the melody is placed against.
        """
        for mood, figures in BASS_FIGURES.items():
            for figure in figures:
                onset, _length, rung = figure[0]
                assert (onset, rung) == (0, 0), mood

    def test_every_figure_stays_inside_the_bar(self) -> None:
        for mood, figures in BASS_FIGURES.items():
            for figure in figures:
                assert all(
                    start >= 0 and start + length <= 16 for start, length, _rung in figure
                ), mood

    def test_no_figure_reaches_past_the_rungs_a_triad_fills(self) -> None:
        """No figure reaches past the third tone above its landing tone.

        A triad fills a rung within every octave, so rungs 0 to 3 exist
        for any chord a mood can draw — this is what lets one figure land
        on a seventh chord without transposing anything.
        """
        for mood, figures in BASS_FIGURES.items():
            for figure in figures:
                assert all(
                    0 <= rung <= 3 for _start, _length, rung in figure
                ), mood

    def test_the_plain_figure_is_in_every_mood(self) -> None:
        # It is what a piece's final bar plays whatever its slot drew, so
        # no mood's vocabulary may be without it.
        for figures in BASS_FIGURES.values():
            assert PLAIN_BASS_FIGURE in figures

    def test_the_draw_is_deterministic_and_rotates(self) -> None:
        """One figure per slot, changing as the harmony changes.

        Deterministic for a seed; and a slot's figure differs from the
        one before it, which is what gives a progression an accompaniment
        that moves with it rather than one bar played twelve times. The
        rotation is why this holds even for a handful of slots over a
        short vocabulary: a plain weighted draw could repeat and leave
        the bar-to-bar reading counting one figure.
        """
        drawn = draw_bass_figures(BASS_FIGURES["calming"], rng=random.Random(4), count=6)
        again = draw_bass_figures(BASS_FIGURES["calming"], rng=random.Random(4), count=6)
        assert drawn == again
        assert len(set(drawn)) >= 3
        assert all(a != b for a, b in itertools.pairwise(drawn))

    def test_the_fallback_vocabulary_is_the_gentle_set(self) -> None:
        """The `.get(mood, default)` shape, now in `default_plan`.

        Every `Mood` has a profile, so nothing reaches the fallback through
        a spec — it is defensive. What can still be checked here is that the
        fallback is the documented one and is drawable, which is what a
        vocabulary handed straight to `draw_bass_figures` has to be.
        """
        assert BASS_FIGURES["calming"] == DEFAULT_BASS_FIGURES
        drawn = draw_bass_figures(DEFAULT_BASS_FIGURES, rng=random.Random(1), count=4)
        assert len(drawn) == 4
        assert set(drawn) <= set(DEFAULT_BASS_FIGURES)

    def test_an_empty_vocabulary_is_refused(self) -> None:
        """A plan cannot carry an empty one — `CompositionPlan` refuses it —
        but a direct caller can, and the refusal names the reason rather
        than raising `randrange` on an empty range."""
        with pytest.raises(ValueError, match="at least one figure"):
            draw_bass_figures((), rng=random.Random(1), count=2)


class TestMotifMelody:
    def test_interval_diversity(self) -> None:
        # The old re-rolled arpeggio walk produced just 3 distinct
        # intervals; the motif walk spans chord-tone space and
        # ornaments, so the histogram must widen.
        for mood in (Mood.CALMING, Mood.ELECTRIFYING, Mood.SLEEP):
            spec = CompositionSpec.model_validate(
                {"mood": mood.value, "duration_seconds": 60, "seed": 42}
            )
            mel = _melody(compose(spec))
            intervals = {
                abs(b.pitch_midi - a.pitch_midi)
                for a, b in itertools.pairwise(mel)
            }
            assert len(intervals) >= 6, (mood, sorted(intervals))

    def test_motif_recurs_across_bars(self) -> None:
        # A section develops one idea: the same melodic contour (sign
        # pattern) must recur across bars of the piece.
        spec = CompositionSpec.model_validate(
            {"mood": Mood.ELECTRIFYING.value, "duration_seconds": 60, "seed": 42}
        )
        mel = _melody(compose(spec))
        contours: Counter[tuple[int, ...]] = Counter()
        bars: dict[int, list[int]] = {}
        for note in mel:
            bars.setdefault(note.tick // (4 * PPQ), []).append(note.pitch_midi)
        for pitches in bars.values():
            contour = tuple(
                (b - a > 0) - (b - a < 0)
                for a, b in itertools.pairwise(pitches)
            )
            if len(contour) >= 3:
                contours[contour] += 1
        assert contours, "no bar carried a long-enough contour"
        assert contours.most_common(1)[0][1] >= 2, "the motif contour never recurs"

    def test_bars_stay_chord_tone_anchored(self) -> None:
        # Every note a motif bar renders is a chord tone of that bar's
        # chord, or a passing/neighbour tone the shared licence admits —
        # which is a stepwise tone entered and left by step between two
        # chord tones, and nothing else. The contract the walk was
        # rewritten to keep is that the line moves *through* the chord:
        # the downbeat is a chord tone and so is everything it lands on,
        # and the tones between them are the ones a step explains. (The
        # integration-level oracle lives in test_compose_engine; the S6
        # walking bass made "interval above the bass note" a wrong oracle
        # here, since the bass now plays inversions too.)
        from saimc.compose.engine import _melody_bar
        from saimc.compose.forms import bar_diatonic_pcs, scale_intervals
        from saimc.compose.linter import legal_non_chord_tone
        from saimc.compose.motif import MotifVariant
        from saimc.compose.score import PPQ, KeySignature
        from saimc.instruments import MelodyBand

        chord_root = 60
        chord_tones = (0, 4, 7, 11)  # Cmaj7
        diatonic = bar_diatonic_pcs(chord_tones, KeySignature(root="C", mode="major"))
        for seed in range(20):
            rng = random.Random(seed)
            motif = generate_motif(rng, bar_ticks=4 * PPQ)
            notes = _melody_bar(
                band=MelodyBand(low_midi=60, high_midi=84),
                variant=MotifVariant(motif=motif),
                chord_root=chord_root,
                chord_tones=chord_tones,
                scale=scale_intervals(0, "major"),
                anchor=0,
                prev_pitch=None,
                start_tick=0,
                bar_ticks=4 * PPQ,
                rng=rng,
                position=0.5,
                ticks_per_bar=4 * PPQ,
                seed_for_variation=seed,
                mood="calming",
            )
            assert notes
            for index, note in enumerate(notes):
                if (note.pitch_midi - chord_root) % 12 in chord_tones:
                    continue
                assert legal_non_chord_tone(
                    note,
                    prev=notes[index - 1] if index else None,
                    nxt=notes[index + 1] if index + 1 < len(notes) else None,
                    bar_start_tick=0,
                    ppq=PPQ,
                    diatonic_pcs=diatonic,
                ), (
                    f"seed {seed}: pitch {note.pitch_midi} at {note.tick} is "
                    "neither a chord tone nor a licensed passing tone"
                )

    def test_deterministic_across_seeds(self) -> None:
        spec = CompositionSpec.model_validate(
            {"mood": Mood.SLEEP.value, "duration_seconds": 60, "seed": 5150}
        )
        out1 = compose(spec)
        out2 = compose(spec)
        assert out1.notation_score.notes == out2.notation_score.notes

    def test_the_line_is_held_in_one_band_for_the_whole_piece(self) -> None:
        """The band, not the phrase, is what the rewrite added.

        The bar's walk is folded to within an octave of the degree it starts
        on and the octave placement fits the bar to the tessitura before it
        weighs anything else, so the melody stays inside one window for the
        whole piece instead of climbing as the section develops. Before the
        rewrite the same pieces spanned 28-30 semitones and never descended
        below C5: the apex was an octave jump, so every section's peak was
        also where the line left its range.

        Which window is the melody instrument's, not a module constant:
        the default ensemble carries the piano, but the sweep below runs
        it for every instrument-shaped mood so the assertion is against
        the band the score itself was written in.

        A band asserted per piece, not on average: it is what the placement
        does with every bar, so a line outside it means the placement
        stopped preferring the band rather than that one piece was
        unlucky.

        One exception is real and is not this test's business to hide: a
        register is a whole octave, and a line wider than the band's
        freedom at its rotation has no octave inside it, so a bar can sit
        a tone past the edge. That is rare — one piece in 4200 over a
        sweep of every mood, 200 seeds and 7 durations, and by two
        semitones — and `test_a_bar_the_band_cannot_hold_keeps_its_line`
        in the engine suite is where it is pinned, along with the line
        that has to survive it.
        """
        for mood in Mood:
            for seed in (42, 7, 11):
                spec = CompositionSpec(mood=mood, seed=seed, duration_seconds=60)
                out = compose(spec)
                band = _melody_band(out)
                pitches = [note.pitch_midi for note in _melody(out)]
                assert pitches
                assert min(pitches) >= band.low_midi, (mood, seed, min(pitches))
                assert max(pitches) <= band.high_midi, (mood, seed, max(pitches))

    def test_the_band_belongs_to_the_melody_instrument(self) -> None:
        """Two instruments, two windows, and the line lands in its own.

        The sweep is over the melody palette, and each piece's melody
        must sit inside *its* instrument's band and — the half that makes
        it a claim about playability rather than about placement — inside
        that instrument's compass. Before the band came from the
        instrument table every one of these wrote the same twenty
        semitones, E4 to C6.

        The window asserted is the one the piece was placed in, which is
        the instrument's own band unless the ensemble's accompaniment
        needed an octave under the tune: the piano here carries the
        default ensemble's strings bed and is therefore written in the
        raised 63-84, while the twenty-one-semitone window itself is what
        the rest of the test reads. Both are asserted, because the raise
        may move the window but it may not widen it.
        """
        palette = (
            Instrument.PIANO,
            Instrument.TRUMPET,
            Instrument.TUBA,
            Instrument.PICCOLO,
            Instrument.CONTRABASS,
            Instrument.CELLO,
            Instrument.FLUTE,
            Instrument.CLARINET,
            Instrument.VIOLIN,
            Instrument.BASSOON,
            Instrument.VIBRAPHONE,
            Instrument.TAIKO,
        )
        for instrument in palette:
            spec = CompositionSpec(
                mood=Mood.CALMING,
                instrumentation=instrument,
                duration_seconds=60,
                seed=1,
            )
            out = compose(spec)
            band = _melody_band(out)
            own = melody_band(instrument.value)
            assert band.high_midi - band.low_midi == own.high_midi - own.low_midi, instrument
            assert band.low_midi >= own.low_midi, instrument
            span = range_for(instrument.value)
            pitches = [note.pitch_midi for note in _melody(out)]
            assert pitches, instrument
            assert min(pitches) >= band.low_midi, (instrument, min(pitches))
            assert max(pitches) <= band.high_midi, (instrument, max(pitches))
            # The band is not the compass: a melody may not leave the
            # instrument either, and `melody_band` is built so it cannot
            # ask for one — this is that invariant, measured on a piece.
            assert span.contains(min(pitches)), (instrument, min(pitches))
            assert span.contains(max(pitches)), (instrument, max(pitches))

    def test_every_voice_of_a_piece_is_inside_its_own_instrument(self) -> None:
        """Not just the melody: the bass and the pad are instruments too.

        §8's "all notes within instrument range" is only true per voice
        once each voice is measured against the instrument it is played
        by, which is what `voice_instruments` is for. The accompaniment
        voices are not the melody instrument, so a check against the
        melody's band would pass a contrabass line an octave too high
        without noticing.
        """
        for instrument in (Instrument.PIANO, Instrument.TUBA, Instrument.PICCOLO):
            spec = CompositionSpec(
                mood=Mood.CALMING,
                instrumentation=instrument,
                duration_seconds=60,
                seed=1,
            )
            out = compose(spec)
            instruments = {v.voice_id: v.instrument for v in out.voice_instruments}
            assert instruments, instrument
            for voice_id, name in instruments.items():
                span = range_for(name)
                pitches = [
                    n.pitch_midi for n in out.notation_score.notes if n.voice_id == voice_id
                ]
                for pitch in pitches:
                    assert span.contains(pitch), (instrument, name, pitch)

    def test_the_piece_scores_as_a_melody(self) -> None:
        """The scorecard's own read of the finished line, over a corpus.

        The per-piece assertions here are the metrics that clear on every
        piece measured; `leap_recovery_ratio` and `max_leap_semitones` are
        corpus *means* (`score_corpus` takes the mean of every metric), and
        one piece may sit outside its bar without the corpus missing it — so
        they are asserted the way the gate measures them rather than
        tightened into an invariant the metric never claimed.

        Both means therefore need a sample big enough to be a mean. The
        sweep is thirty-three seeds of each mood and not the five it used
        to be, because a mean with a heavy tail cannot be read off
        fifteen: that sample happened to draw the two largest leaps in
        the corpus, and its reading flipped by a semitone whenever a
        placement moved. Thirty-three seeds is under a second of work and
        is stable to the semitone.

        The tail is the honest part of this test and it is not asserted
        away. Measured over the ninety-nine pieces, ten carry a leap
        wider than `QUALITY_MAX_LEAP_MAX` — 17 to 20 semitones, every one
        of them at a bar seam in an electrifying piece whose bar is as
        wide as the band it has to sit in, where the band wins the
        ranking and the entrance pays for it. That is the review's
        finding on `max_leap_semitones`, and it is Phase 3's work: this
        test asserts the mean the gate reads, and the mean (11.4) clears
        the bar of 12.

        This is the melodic half of the quality bar. The accompaniment is
        what still misses it, in the acceptance suite's `test_the_generator_
        _does_not_clear_the_bar_yet`.
        """
        reports = []
        for mood in Mood:
            for seed in range(33):
                spec = CompositionSpec(mood=mood, seed=seed, duration_seconds=60)
                report = score_piece(
                    compose(spec).notation_score, piece=f"{mood.value}-{seed}"
                )
                reports.append(report)
                assert report.step_ratio >= 0.45, (mood, seed, report)
                assert 7 <= report.range_semitones <= 24, (mood, seed, report)
                assert report.repeat_ratio <= 0.25, (mood, seed, report)
                assert report.distinct_durations >= 3, (mood, seed, report)

        def mean(metric: str) -> float:
            return sum(getattr(report, metric) for report in reports) / len(reports)

        assert mean("leap_recovery_ratio") >= 0.60
        assert mean("max_leap_semitones") <= 12

    def test_melody_fits_piano_range(self) -> None:
        for mood in Mood:
            spec = CompositionSpec.model_validate(
                {"mood": mood.value, "duration_seconds": 60, "seed": 21}
            )
            mel = _melody(compose(spec))
            assert all(22 <= n.pitch_midi <= 107 for n in mel)
