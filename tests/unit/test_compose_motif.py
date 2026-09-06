"""Unit tests for motif-driven melody generation (S5)."""

from __future__ import annotations

import itertools
import random
from collections import Counter

import pytest

from saimc.compose.engine import compose
from saimc.compose.motif import (
    PPQ,
    Motif,
    MotifCell,
    generate_motif,
    vary_motif,
)
from saimc.compose.score import VOICE_MELODY
from saimc.spec import CompositionSpec, Mood


def _melody(out) -> list:
    return sorted(
        (n for n in out.notation_score.notes if n.voice_id == VOICE_MELODY),
        key=lambda n: n.tick,
    )


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
        rng = random.Random(3)
        for _ in range(100):
            motif = generate_motif(rng, bar_ticks=4 * PPQ)
            for cell in motif[1:]:
                assert cell.step in (-2, -1, 0, 1, 2)

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
            elif variant.anchor_offset == 1:
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
        # chord, whatever the rhythm library did to the durations. (The
        # integration-level oracle lives in test_compose_engine; the
        # S6 walking bass made "interval above the bass note" a wrong
        # oracle here, since the bass now plays inversions too.)
        from saimc.compose.engine import _melody_bar
        from saimc.compose.motif import MotifVariant

        chord_root = 60
        chord_tones = (0, 4, 7, 11)  # Cmaj7
        for seed in range(20):
            rng = random.Random(seed)
            motif = generate_motif(rng, bar_ticks=4 * PPQ)
            notes = _melody_bar(
                variant=MotifVariant(motif=motif),
                chord_root=chord_root,
                chord_tones=chord_tones,
                anchor=0,
                start_tick=0,
                bar_ticks=4 * PPQ,
                rng=rng,
                position=0.5,
                ticks_per_bar=4 * PPQ,
                seed_for_variation=seed,
                mood="calming",
            )
            assert notes
            for note in notes:
                assert (note.pitch_midi - chord_root) % 12 in chord_tones, (
                    f"seed {seed}: pitch {note.pitch_midi} not a chord tone"
                )

    def test_deterministic_across_seeds(self) -> None:
        spec = CompositionSpec.model_validate(
            {"mood": Mood.SLEEP.value, "duration_seconds": 60, "seed": 5150}
        )
        out1 = compose(spec)
        out2 = compose(spec)
        assert out1.notation_score.notes == out2.notation_score.notes

    def test_melody_fits_piano_range(self) -> None:
        for mood in Mood:
            spec = CompositionSpec.model_validate(
                {"mood": mood.value, "duration_seconds": 60, "seed": 21}
            )
            mel = _melody(compose(spec))
            assert all(22 <= n.pitch_midi <= 107 for n in mel)
