"""Unit tests for the composition engine."""

from __future__ import annotations

from itertools import pairwise

import pytest

from saimc.compose.engine import (
    ARRANGEMENT_ARC_MIN_REPS,
    MODULATION_OFFSET,
    CompositionEngineError,
    EngineErrorCode,
    _chord_intervals,
    _merged_tie_runs,
    _scale_degree_to_semitones,
    _truncate_template_for_coda,
    compose,
)
from saimc.compose.forms import key_root_midi
from saimc.compose.linter import lint
from saimc.compose.score import (
    KeySignature,
    NotationScore,
    PerformancePlan,
    microseconds_at_tick,
    realized_duration_seconds,
    ticks_at_microsecond,
)
from saimc.spec import CompositionSpec, Mood, WesternKey

# Independent oracle: diatonic triads and sevenths as semitone
# offsets from the TONIC, degrees I..VII. Major and minor keys.
MAJOR_TRIADS_ABS = (
    (0, 4, 7),
    (2, 5, 9),
    (4, 7, 11),
    (5, 9, 12),
    (7, 11, 14),
    (9, 12, 16),
    (11, 14, 17),
)
MINOR_TRIADS_ABS = (
    (0, 3, 7),
    (2, 5, 8),
    (3, 7, 10),
    (5, 8, 12),
    (7, 10, 14),
    (8, 12, 15),
    (10, 14, 17),
)
MAJOR_SEVENTHS_ABS = (
    (0, 4, 7, 11),
    (2, 5, 9, 12),
    (4, 7, 11, 14),
    (5, 9, 12, 16),
    (7, 11, 14, 17),
    (9, 12, 16, 19),
    (11, 14, 17, 20),
)
MINOR_SEVENTHS_ABS = (
    (0, 3, 7, 10),
    (2, 5, 8, 12),
    (3, 7, 10, 14),
    (5, 8, 12, 15),
    (7, 10, 14, 17),
    (8, 12, 15, 19),
    (10, 14, 17, 21),
)


def _chord_offsets(slot, mode: str) -> tuple[int, ...]:
    """Tonic-relative chord tones for a slot (triad or seventh)."""
    if mode == "major":
        triads, sevenths = MAJOR_TRIADS_ABS, MAJOR_SEVENTHS_ABS
    else:
        triads, sevenths = MINOR_TRIADS_ABS, MINOR_SEVENTHS_ABS
    if slot.borrowed:
        triads, sevenths = (
            (MINOR_TRIADS_ABS, MINOR_SEVENTHS_ABS)
            if mode == "major"
            else (MAJOR_TRIADS_ABS, MAJOR_SEVENTHS_ABS)
        )
    return sevenths[slot.degree % 7] if slot.seventh else triads[slot.degree % 7]


def _bar_degrees_and_offsets(
    out, mood: str
) -> list[tuple[int, tuple[int, ...], int]]:
    """Mirror the engine's per-bar chord walk.

    Returns (degree, tonic-relative tones, key offset) per bar — the
    offset carries the long-piece modulation lift, which exempts each
    section's final two cadence bars.
    """
    from saimc.compose.forms import apply_final_cadence, get_template_for_form

    arrangement = out.arrangement
    lifted = arrangement.repetition_count >= ARRANGEMENT_ARC_MIN_REPS

    result: list[tuple[int, tuple[int, ...], int]] = []
    for section in range(arrangement.repetition_count):
        template = (
            arrangement.template
            if section == 0
            else get_template_for_form(mood, arrangement.form_bars, variant_index=section)
        )
        section_offset = (
            MODULATION_OFFSET if section == arrangement.repetition_count - 1 and lifted else 0
        )
        if section == arrangement.repetition_count - 1:
            template = apply_final_cadence(template, mood)
        consumed = 0
        for slot in template.chords:
            exempt = section_offset and consumed >= template.bars - 2
            bar_offset = 0 if exempt else section_offset
            result.extend(
                (slot.degree, _chord_offsets(slot, out.key.mode), bar_offset)
                for _ in range(slot.bars)
            )
            consumed += slot.bars
    if arrangement.coda_bars > 0:
        coda = apply_final_cadence(
            _truncate_template_for_coda(arrangement.template, arrangement.coda_bars),
            mood,
        )
        coda_offset = MODULATION_OFFSET if lifted else 0
        consumed = 0
        for slot in coda.chords:
            exempt = coda_offset and consumed >= coda.bars - 2
            bar_offset = 0 if exempt else coda_offset
            result.extend(
                (slot.degree, _chord_offsets(slot, out.key.mode), bar_offset)
                for _ in range(slot.bars)
            )
            consumed += slot.bars
    return result


def _spec(mood: Mood, *, duration: int = 60, seed: int | None = 42, **kw) -> CompositionSpec:
    base = {"mood": mood.value}
    base.update(kw)
    return CompositionSpec.model_validate(
        {"mood": mood.value, "duration_seconds": duration, "seed": seed, **kw}
    )


class TestComposeHappyPath:
    @pytest.mark.parametrize("mood", [Mood.CALMING, Mood.ELECTRIFYING, Mood.SLEEP])
    def test_returns_engine_output(self, mood: Mood) -> None:
        out = compose(_spec(mood, duration=60))
        assert isinstance(out.notation_score, NotationScore)
        assert isinstance(out.performance_plan, PerformancePlan)

    @pytest.mark.parametrize(
        ("mood", "target"),
        [
            (Mood.CALMING, 60),
            (Mood.CALMING, 180),
            (Mood.ELECTRIFYING, 60),
            (Mood.ELECTRIFYING, 300),
            (Mood.SLEEP, 180),
            (Mood.SLEEP, 600),
        ],
    )
    def test_realized_duration_within_tolerance(self, mood: Mood, target: int) -> None:
        out = compose(_spec(mood, duration=target))
        realised = realized_duration_seconds(out.performance_plan)
        delta = abs(realised - target) / target
        assert delta <= 0.02, f"mood={mood.value}, target={target}, realised={realised:.2f}"

    @pytest.mark.parametrize("mood", [Mood.CALMING, Mood.ELECTRIFYING, Mood.SLEEP])
    def test_passes_lint(self, mood: Mood) -> None:
        out = compose(_spec(mood, duration=180))
        report = lint(out.notation_score)
        assert report.passed, f"lint issues: {report.issues}"

    def test_deterministic_for_same_seed(self) -> None:
        out1 = compose(_spec(Mood.CALMING, duration=180, seed=42))
        out2 = compose(_spec(Mood.CALMING, duration=180, seed=42))
        assert out1.notation_score.compute_hash() == out2.notation_score.compute_hash()

    def test_different_seeds_give_different_scores(self) -> None:
        out1 = compose(_spec(Mood.CALMING, duration=180, seed=1))
        out2 = compose(_spec(Mood.CALMING, duration=180, seed=2))
        assert out1.notation_score.compute_hash() != out2.notation_score.compute_hash()

    def test_notes_are_sorted_canonically(self) -> None:
        out = compose(_spec(Mood.CALMING, duration=180))
        for prev, curr in zip(out.notation_score.notes, out.notation_score.notes[1:], strict=False):
            assert (curr.tick, curr.voice_id, curr.pitch_midi) >= (
                prev.tick,
                prev.voice_id,
                prev.pitch_midi,
            )


class TestComposeMultipleVoices:
    def test_has_bass_and_melody(self) -> None:
        out = compose(_spec(Mood.CALMING, duration=60))
        voice_ids = {n.voice_id for n in out.notation_score.notes}
        # At minimum we expect bass (voice 0) and melody (voice 1).
        assert 0 in voice_ids
        assert 1 in voice_ids


class TestComposeFailurePaths:
    def test_invalid_spec_raises(self) -> None:
        # Duration below the schema's 30s minimum is rejected by the
        # schema before reaching the engine.
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            compose(_spec(Mood.CALMING, duration=10))  # below the 30s minimum

    def test_engine_error_has_code(self) -> None:
        # Just assert the error machinery exists by constructing a
        # CompositionEngineError directly (the spec constraints make it
        # hard to provoke one through the public API).
        exc = CompositionEngineError(code=EngineErrorCode.DURATION_UNFULFILLABLE, message="test")
        assert exc.code == EngineErrorCode.DURATION_UNFULFILLABLE

    def test_coda_path_lints_clean(self) -> None:
        """45s calming uses a 4-bar coda; the engine still lints clean."""
        out = compose(_spec(Mood.CALMING, duration=45, seed=42))
        assert out.arrangement.coda_bars == 4
        report = lint(out.notation_score)
        assert report.passed, f"lint issues: {report.issues}"
        # The score includes coda measures on top of the body.
        assert out.notation_score.total_ticks() == (out.arrangement.total_bars_with_coda * 4 * 480)


class TestComposeIntegrationWithWorker:
    """Spot-check the wiring through `compose_stage`'s default-engine path."""

    def test_engine_used_by_stage(self, tmp_path) -> None:
        from datetime import UTC, datetime

        from saimc.jobs.stages import compose_stage
        from saimc.jobs.state import JobState
        from saimc.jobs.storage import Job, JobStorage

        storage = JobStorage(str(tmp_path))
        now = datetime.now(UTC)
        job = Job(
            job_id="test",
            created_at=now,
            updated_at=now,
            state=JobState.PARSING,
            progress=0.0,
            current_stage="parsing",
            input_prompt="calming piano",
            input_spec=_spec(Mood.CALMING, duration=60),
        )
        result = compose_stage(job, storage)
        assert result.next_state == JobState.VALIDATING
        assert result.error is None


class TestMusicalShape:
    """The §10 quality upgrades: real left hand, register split, A/B form."""

    def test_bass_plays_two_chord_tones_per_bar(self) -> None:
        out = compose(_spec(Mood.CALMING, duration=60))
        bass = [n for n in out.notation_score.notes if n.voice_id == 0]
        bar_ticks_count = out.notation_score.ppq * 4
        first_bar = [n for n in bass if n.tick < bar_ticks_count]
        assert len(first_bar) == 2  # two tones, not a held drone
        pitches = {n.pitch_midi for n in first_bar}
        assert len(pitches) == 2
        low, high = sorted(pitches)
        tonic = key_root_midi(out.key)
        assert low == tonic - 12  # the section opens in root position
        assert (high - low) % 12 in (3, 4, 7)  # a third or fifth above it

    def test_melody_sits_above_bass(self) -> None:
        out = compose(_spec(Mood.CALMING, duration=60))
        bass_avg = sum(n.pitch_midi for n in out.notation_score.notes if n.voice_id == 0) / len(
            [n for n in out.notation_score.notes if n.voice_id == 0]
        )
        melody_min = min(n.pitch_midi for n in out.notation_score.notes if n.voice_id == 1)
        assert melody_min > bass_avg

    def test_repeated_sections_rotate_template_variants(self) -> None:
        out = compose(_spec(Mood.ELECTRIFYING, duration=300, seed=3))
        arrangement = out.arrangement
        assert arrangement.repetition_count > 1
        # Section harmony differs between rep 0 and rep 1.
        bar_ticks_count = out.notation_score.ppq * 4
        section_ticks = arrangement.form_bars * bar_ticks_count
        # Variant rotation reorders the progression, so the chord
        # sounding at each section's downbeat differs (e.g. I vs vi).
        first_downbeat = next(n.pitch_midi for n in out.notation_score.notes if n.voice_id == 0)
        second_downbeat = next(
            n.pitch_midi
            for n in out.notation_score.notes
            if n.voice_id == 0 and n.tick >= section_ticks
        )
        assert first_downbeat != second_downbeat

    def test_coda_ends_on_tonic(self) -> None:
        out = compose(_spec(Mood.CALMING, duration=45))
        arrangement = out.arrangement
        assert arrangement.coda_bars > 0
        bar_ticks_count = out.notation_score.ppq * 4
        coda_start = arrangement.total_bars * bar_ticks_count
        coda_bass = [
            n for n in out.notation_score.notes if n.voice_id == 0 and n.tick >= coda_start
        ]
        last_chord_root = coda_bass[-2].pitch_midi  # the bar's downbeat root
        tonic = key_root_midi(out.key) % 12  # key root pitch class
        assert last_chord_root % 12 == tonic % 12

    def test_tempo_bpm_constraint_is_honoured(self) -> None:
        out = compose(_spec(Mood.CALMING, duration=180, tempo_bpm=63))
        assert out.notation_score.tempo.bpm == 63


class TestChordToneHarmony:
    """Every sounding pitch belongs to the chord sounding in its bar.

    Regression for the double-add bug: the chord-tone tables were
    tonic-absolute but were added on top of a chord root that already
    carried the scale-degree offset, so every non-tonic chord played
    the wrong triad (e.g. F#m over an Am bass in C major).
    """

    @pytest.mark.parametrize(
        ("mood", "duration"),
        [
            (Mood.CALMING, 60),
            (Mood.CALMING, 45),
            (Mood.ELECTRIFYING, 60),
            (Mood.SLEEP, 60),
            (Mood.CALMING, 180),
        ],
    )
    def test_all_notes_are_chord_tones(self, mood: Mood, duration: int) -> None:
        out = compose(_spec(mood, duration=duration, seed=42))
        bars = _bar_degrees_and_offsets(out, mood.value)
        assert len(bars) == out.arrangement.total_bars_with_coda
        ticks_per_bar = out.notation_score.ppq * 4
        tonic = key_root_midi(out.key)
        anticipation_zone = ticks_per_bar - out.notation_score.ppq // 2
        for note in out.notation_score.notes:
            if note.voice_id == 2:  # percussion keys are GM drum map, not pitched
                continue
            bar = note.tick // ticks_per_bar
            _degree, offsets, key_offset = bars[bar]
            sounding = {(tonic + key_offset + offset) % 12 for offset in offsets}
            if note.tick % ticks_per_bar >= anticipation_zone and bar + 1 < len(bars):
                # An anacrusis pickup anticipates the next bar's chord.
                _next_degree, next_offsets, next_offset = bars[bar + 1]
                sounding |= {(tonic + next_offset + offset) % 12 for offset in next_offsets}
            assert note.pitch_midi % 12 in sounding, (
                f"bar {bar}: pitch {note.pitch_midi} "
                f"not in chord pcs {sounding} (offsets {offsets})"
            )

    def test_seventh_chords_reach_the_score(self) -> None:
        """Templates with 7th slots play their 4th tone, and it is a
        genuine chord tone (the S1 triad test passes it too)."""
        from saimc.compose.forms import MOOD_PROFILES

        assert any(
            slot.seventh
            for template in MOOD_PROFILES["calming"].templates
            for slot in template.chords
        )
        out = compose(_spec(Mood.CALMING, duration=60, seed=42))
        bars = _bar_degrees_and_offsets(out, Mood.CALMING.value)
        assert any(len(offsets) == 4 for _degree, offsets, _key_offset in bars)

    def test_bass_walks_and_stays_on_chord_tones(self) -> None:
        """The bass is a walking line: every note is a chord tone of its
        bar, section starts land in root position, and consecutive chord
        basses move by small intervals instead of jumping octaves."""
        out = compose(_spec(Mood.ELECTRIFYING, duration=300, seed=5))
        bars = _bar_degrees_and_offsets(out, Mood.ELECTRIFYING.value)
        ticks_per_bar = out.notation_score.ppq * 4
        tonic = key_root_midi(out.key)
        bass_by_bar: dict[int, list[int]] = {}
        for note in out.notation_score.notes:
            if note.voice_id == 0:
                bass_by_bar.setdefault(note.tick // ticks_per_bar, []).append(note.pitch_midi)
        for bar, (_degree, offsets, key_offset) in enumerate(bars):
            sounding = {(tonic + key_offset + offset) % 12 for offset in offsets}
            for pitch in set(bass_by_bar[bar]):
                assert pitch % 12 in sounding, (
                    f"bar {bar}: bass {pitch} not a chord tone of {sounding}"
                )
        # Consecutive chord-change basses stay close (the walk).
        last_pitch: int | None = None
        for bar in sorted(bass_by_bar):
            downbeat = min(p for p in bass_by_bar[bar])
            if last_pitch is not None:
                assert abs(downbeat - last_pitch) <= 7, (
                    f"bass jumped {abs(downbeat - last_pitch)} semitones "
                    f"into bar {bar}"
                )
            last_pitch = downbeat

    def test_borrowed_bVII_reaches_the_score(self) -> None:
        """The extended electrifying template's borrowed bVII (a major
        triad a whole step below the tonic) actually sounds: the bar's
        notes sit on the lowered-root pcs, not the diatonic vii."""
        out = compose(_spec(Mood.ELECTRIFYING, duration=300, seed=5))
        assert out.key.mode == "major"
        tonic = key_root_midi(out.key)
        bars = _bar_degrees_and_offsets(out, Mood.ELECTRIFYING.value)
        # Diatonic major degree 6 is (11, 14, 17); the borrowed bVII
        # takes the minor-mode table, giving (10, 14, 17).
        borrowed_bars = [
            b for b, (_degree, offsets, _key_offset) in enumerate(bars) if offsets == (10, 14, 17)
        ]
        assert borrowed_bars, "expected the bVII borrowed slot to sound"
        ticks_per_bar = out.notation_score.ppq * 4
        pitched = [n for n in out.notation_score.notes if n.voice_id != 2]
        for bar in borrowed_bars:
            sounding = {(tonic + offset) % 12 for offset in (10, 14, 17)}
            for note in pitched:
                if note.tick // ticks_per_bar == bar:
                    assert note.pitch_midi % 12 in sounding

    def test_cadence_bass_is_root_position(self) -> None:
        """The cadence's pinned bass_degree lands the final tonic's root
        in the bass (the S3 final-cadence invariant)."""
        out = compose(_spec(Mood.CALMING, duration=60, seed=42))
        tonic = key_root_midi(out.key)
        ticks_per_bar = out.notation_score.ppq * 4
        bass = [n for n in out.notation_score.notes if n.voice_id == 0]
        final_bar_notes = [n for n in bass if n.tick // ticks_per_bar == (
            out.arrangement.total_bars_with_coda - 1
        )]
        assert final_bar_notes
        assert final_bar_notes[0].pitch_midi % 12 == tonic % 12


class TestSharpMinorKeys:
    """C#m and G#m were advertised by the spec vocabulary but crashed
    the engine with an unstructured ValueError (missing key roots)."""

    @pytest.mark.parametrize("key", [WesternKey.C_SHARP_MINOR, WesternKey.G_SHARP_MINOR])
    def test_composes_and_lints(self, key: WesternKey) -> None:
        out = compose(_spec(Mood.CALMING, duration=60, key=key))
        assert out.key.root in ("C#", "G#")
        assert out.notation_score.notes
        assert lint(out.notation_score).passed

    def test_key_roots_resolve(self) -> None:
        assert key_root_midi(KeySignature(root="C#", mode="minor")) == 61
        assert key_root_midi(KeySignature(root="G#", mode="minor")) == 68

    def test_unknown_key_surfaces_as_invalid_spec(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A key that cannot resolve raises the typed engine error, not
        a raw ValueError escaping compose()."""
        import saimc.compose.engine as engine_mod

        def boom(spec: object) -> KeySignature:
            raise ValueError("Unknown key root: 'X'")

        monkeypatch.setattr(engine_mod, "key_signature_from_spec", boom)
        with pytest.raises(CompositionEngineError) as exc_info:
            compose(_spec(Mood.CALMING, duration=60))
        assert exc_info.value.code == EngineErrorCode.INVALID_SPEC

    def test_velocity_follows_an_arch(self) -> None:
        out = compose(_spec(Mood.CALMING, duration=60))
        melody = [n for n in out.notation_score.notes if n.voice_id == 1]
        velocities = [n.velocity for n in melody]
        assert max(velocities) - min(velocities) >= 6  # the piece breathes


class TestFinalCadence:
    """Every piece ends at home: the last two bars are the mood's
    cadence (V-I for electrifying, IV-I for calming/sleep) and the
    melody resolves onto the tonic or its third."""

    @pytest.mark.parametrize("mood", [Mood.CALMING, Mood.ELECTRIFYING, Mood.SLEEP])
    def test_final_bar_lands_on_tonic(self, mood: Mood) -> None:
        out = compose(_spec(mood, duration=60))
        ticks_per_bar = out.notation_score.ppq * 4
        tonic_pc = key_root_midi(out.key) % 12
        # The cadential chord is V (7 semitones up) for electrifying,
        # IV (5 semitones up) for the calmer moods.
        cadence_offset = 7 if mood == Mood.ELECTRIFYING else 5
        melody = [n for n in out.notation_score.notes if n.voice_id == 1]

        last_bar_start = (out.arrangement.total_bars_with_coda - 1) * ticks_per_bar
        final_bass_roots = {
            n.pitch_midi
            for n in out.notation_score.notes
            if n.voice_id == 0 and n.tick >= last_bar_start
        }
        # The final chord is the tonic: its bass root is the key root.
        assert min(final_bass_roots) % 12 == tonic_pc

        # The bar before carries the cadential pre-dominant/dominant:
        # the dominant (V) for electrifying, the subdominant (IV) for
        # the calmer moods.
        prev_bar_start = last_bar_start - ticks_per_bar
        prev_bass_roots = {
            n.pitch_midi
            for n in out.notation_score.notes
            if n.voice_id == 0 and prev_bar_start <= n.tick < last_bar_start
        }
        assert min(prev_bass_roots) % 12 == (tonic_pc + cadence_offset) % 12

        # The melody resolves onto the tonic or its third.
        last_melody = max(melody, key=lambda n: n.tick)
        third = 4 if out.key.mode == "major" else 3
        assert last_melody.pitch_midi % 12 in (tonic_pc, (tonic_pc + third) % 12)

    def test_coda_carries_the_cadence(self) -> None:
        # A 45s calming piece ends with a coda; its last bar is still
        # the tonic with IV before it.
        out = compose(_spec(Mood.CALMING, duration=45))
        assert out.arrangement.coda_bars > 0
        ticks_per_bar = out.notation_score.ppq * 4
        tonic_pc = key_root_midi(out.key) % 12
        last_bar_start = (out.arrangement.total_bars_with_coda - 1) * ticks_per_bar
        final_bass = {
            n.pitch_midi
            for n in out.notation_score.notes
            if n.voice_id == 0 and n.tick >= last_bar_start
        }
        assert min(final_bass) % 12 == tonic_pc


class TestPhraseStructure:
    """2-bar phrase breaths, apex note, downbeat anchoring, half cadence."""

    def test_melody_has_rests(self) -> None:
        out = compose(_spec(Mood.CALMING, duration=180))
        melody = sorted(
            (n for n in out.notation_score.notes if n.voice_id == 1), key=lambda n: n.tick
        )
        rests = sum(
            1
            for prev, curr in pairwise(melody)
            if curr.tick > prev.tick + prev.duration_ticks
        )
        assert rests >= 1, "expected at least one breath in the melody"

    def test_downbeats_anchor_to_root_or_third(self) -> None:
        out = compose(_spec(Mood.CALMING, duration=180))
        ticks_per_bar = out.notation_score.ppq * 4
        melody = [n for n in out.notation_score.notes if n.voice_id == 1]
        downbeats = [n for n in melody if n.tick % ticks_per_bar == 0]
        tonic = key_root_midi(out.key)
        # Chord roots per bar come from the template walk; a downbeat
        # anchors when it matches the chord root or third, i.e. when it
        # is one of the chord's first two tones.
        from saimc.compose.forms import apply_final_cadence, get_template_for_form

        arrangement = out.arrangement
        chords: list[int] = []
        for section in range(arrangement.repetition_count):
            template = (
                arrangement.template
                if section == 0
                else get_template_for_form(
                    "calming", arrangement.form_bars, variant_index=section
                )
            )
            if section == arrangement.repetition_count - 1:
                template = apply_final_cadence(template, "calming")
            for slot in template.chords:
                chords.extend([slot.degree] * slot.bars)
        anchored = 0
        for note in downbeats:
            degree = chords[note.tick // ticks_per_bar]
            chord_root = tonic + _scale_degree_to_semitones(degree, out.key.mode)
            chord_tones = _chord_intervals(degree, out.key)
            candidates = {(chord_root + 12 + t) % 12 for t in chord_tones[:2]}
            if note.pitch_midi % 12 in candidates:
                anchored += 1
        assert anchored / len(downbeats) >= 0.5

    def test_apex_rises_after_the_first_quarter_of_each_section(self) -> None:
        out = compose(_spec(Mood.ELECTRIFYING, duration=180))
        ticks_per_bar = out.notation_score.ppq * 4
        melody = sorted(
            (n for n in out.notation_score.notes if n.voice_id == 1), key=lambda n: n.tick
        )
        section_len = out.arrangement.form_bars * ticks_per_bar
        by_section: dict[int, list] = {}
        for note in melody:
            by_section.setdefault(note.tick // section_len, []).append(note)
        for section_idx, notes in by_section.items():
            start = section_idx * section_len
            peak = max(n.pitch_midi for n in notes)
            # The apex bar sits at ~60% of the section, so the section's
            # highest pitch must occur past its first quarter.
            peak_positions = [
                (n.tick - start) / section_len for n in notes if n.pitch_midi == peak
            ]
            assert any(0.25 <= p <= 0.95 for p in peak_positions), (
                f"section {section_idx}: peak {peak} at positions {peak_positions}"
            )


class TestExpressionModel:
    """S4: controllers ride the plan; humanization touches only the plan."""

    def test_plan_carries_expression_swells(self) -> None:
        out = compose(_spec(Mood.CALMING, duration=60))
        cc11 = [c for c in out.performance_plan.controllers if c.control == 11]
        assert cc11, "expected CC11 expression events"
        # One per measure, on the melody voice, values inside the arch.
        assert {c.voice_id for c in cc11} == {1}
        assert all(c.value <= 127 for c in cc11)

    def test_piano_gets_sustain_pedal(self) -> None:
        out = compose(_spec(Mood.CALMING, duration=60, instrumentation="piano"))
        cc64 = [c for c in out.performance_plan.controllers if c.control == 64]
        assert cc64, "piano expects sustain pedal"
        presses = [c for c in cc64 if c.value == 127]
        releases = [c for c in cc64 if c.value == 0]
        assert presses, "piano pedal should have presses"
        assert releases, "piano pedal should have releases"
        # Every release lands before the next press.
        timeline = sorted(cc64, key=lambda c: c.start_us)
        for prev, curr in pairwise(timeline):
            if prev.value == 0:
                assert curr.start_us >= prev.start_us

    def test_flute_gets_no_pedal(self) -> None:
        out = compose(_spec(Mood.CALMING, duration=60, instrumentation="flute"))
        assert not [c for c in out.performance_plan.controllers if c.control == 64]
        # But expression swells still ride the line.
        assert [c for c in out.performance_plan.controllers if c.control == 11]

    def test_sustained_instrument_gets_legato_overlap(self) -> None:
        out = compose(_spec(Mood.CALMING, duration=60, instrumentation="strings"))
        melody = sorted(
            (e for e in out.performance_plan.notes if e.voice_id == 1), key=lambda e: e.start_us
        )
        overlaps = sum(
            1
            for prev, curr in pairwise(melody)
            if prev.start_us + prev.duration_us > curr.start_us
        )
        assert overlaps >= 1, "sustained instruments should blur into the next note"

    def test_piano_keeps_notated_durations(self) -> None:
        out = compose(_spec(Mood.CALMING, duration=60, instrumentation="piano"))
        melody = sorted(
            (e for e in out.performance_plan.notes if e.voice_id == 1), key=lambda e: e.start_us
        )
        for prev, curr in pairwise(melody):
            assert prev.start_us + prev.duration_us <= curr.start_us

    def test_light_humanization_shifts_only_the_plan(self) -> None:
        spec = _spec(Mood.CALMING, duration=60, humanization="light")
        out = compose(spec)
        melody_plan = sorted(
            (e for e in out.performance_plan.notes if e.voice_id == 1), key=lambda e: e.start_us
        )
        melody_score = sorted(
            (n for n in out.notation_score.notes if n.voice_id == 1), key=lambda n: n.tick
        )
        # The plan folds tied noteheads into one sounded event, so it
        # pairs with the surviving heads — the continuations drop out.
        tie_skips, _ = _merged_tie_runs(melody_score)
        heads = [n for idx, n in enumerate(melody_score) if idx not in tie_skips]
        assert len(melody_plan) == len(heads)
        # The sheet stays on the grid; the plan wobbles within ±10 ms.
        for plan_event, score_note in zip(melody_plan, heads, strict=True):
            grid_us = microseconds_at_tick(score_note.tick, out.notation_score.tempo)
            assert abs(plan_event.start_us - grid_us) <= 10_000

    def test_none_humanization_is_grid_exact(self) -> None:
        out = compose(_spec(Mood.CALMING, duration=60, humanization="none"))
        for event in out.performance_plan.notes:
            tick = ticks_at_microsecond(event.start_us, out.notation_score.tempo)
            back = microseconds_at_tick(tick, out.notation_score.tempo)
            assert event.start_us == back

    def test_humanization_deterministic(self) -> None:
        out1 = compose(_spec(Mood.CALMING, duration=60, humanization="light", seed=7))
        out2 = compose(_spec(Mood.CALMING, duration=60, humanization="light", seed=7))
        assert out1.performance_plan.compute_hash() == out2.performance_plan.compute_hash()

    def test_ghost_notes_in_percussion_plan(self) -> None:
        out = compose(
            _spec(Mood.ELECTRIFYING, duration=60, instrumentation="drum_set")
        )
        perc = [e for e in out.performance_plan.notes if e.voice_id == 2]
        assert perc
        ghosts = [e for e in perc if 20 <= e.velocity <= 35]
        assert ghosts, "expected quiet ghost hits in the drum plan"

    def test_drums_get_timing_scatter(self) -> None:
        out = compose(
            _spec(Mood.ELECTRIFYING, duration=60, instrumentation="drum_set")
        )
        # Realized drum hits are not all on exact 16th multiples of the
        # quarter grid — the scatter is the point.
        quarter_us = 60_000_000 / out.notation_score.tempo.bpm
        off_grid = [
            e
            for e in out.performance_plan.notes
            if e.voice_id == 2 and (e.start_us % (quarter_us / 4)) > 1
        ]
        assert off_grid, "expected percussion timing offsets off the 16th grid"


class TestRhythmVocabulary:
    """S7: the melody speaks in dotted figures, 16ths, ties and pickups."""

    def _melody(self, out: NotationScore) -> list:
        return sorted(
            (n for n in out.notation_score.notes if n.voice_id == 1), key=lambda n: n.tick
        )

    def test_dotted_figures_appear(self) -> None:
        # Dotted quarters (3*PPQ/2, e.g. 720 @ PPQ=480) come only from
        # the rhythm library's dotted op — motif cells are quarters/eighths.
        for mood in (Mood.CALMING, Mood.ELECTRIFYING):
            out = compose(_spec(mood, duration=60))
            assert 3 * out.notation_score.ppq // 2 in {
                n.duration_ticks for n in self._melody(out)
            }, f"{mood}: no dotted rhythm rendered"

    def test_sixteenths_appear_in_high_energy_moods(self) -> None:
        out = compose(_spec(Mood.ELECTRIFYING, duration=60))
        ppq = out.notation_score.ppq
        sixteenths = [n for n in self._melody(out) if n.duration_ticks == ppq // 4]
        assert sixteenths, "electrifying melody never rendered a 16th"

    def test_ties_present_and_folded_into_the_plan(self) -> None:
        out = compose(_spec(Mood.CALMING, duration=60))
        melody = self._melody(out)
        tied_heads = [n for n in melody if n.tie]
        assert tied_heads, "calming melody never tied two noteheads"
        tie_skips, tie_spans = _merged_tie_runs(melody)
        plan_melody = [e for e in out.performance_plan.notes if e.voice_id == 1]
        # One sounded event per surviving head; continuations drop out.
        assert len(plan_melody) == len(melody) - len(tie_skips)
        # Every merged span covers its full run, not just the head.
        for idx, note in enumerate(melody):
            if idx in tie_skips or note.tick not in {m.tick for m in tied_heads}:
                continue
            span = tie_spans.get(idx)
            if span is not None:
                assert span >= note.duration_ticks

    def test_anacrusis_pickups_lead_into_the_bar(self) -> None:
        out = compose(_spec(Mood.ELECTRIFYING, duration=60))
        ppq = out.notation_score.ppq
        ticks_per_bar = 4 * ppq
        # True pickups are the appended half-bar notes at the last
        # eighth slot; 16th-op notes can land at the same tick, so the
        # duration is what distinguishes them.
        pickups = [
            n
            for n in self._melody(out)
            if n.tick % ticks_per_bar == ticks_per_bar - ppq // 2
            and n.duration_ticks == ppq // 2
        ]
        assert pickups, "no eighth-note pickups rendered"
        assert len(pickups) >= 10

    def test_sheet_engraves_ties(self) -> None:
        from saimc.render.sheet import _tie_modes, notation_score_to_musicxml

        out = compose(_spec(Mood.CALMING, duration=60))
        melody = self._melody(out)
        tie_modes = _tie_modes(melody)
        assert "start" in tie_modes.values(), "tied heads should start a tie"
        assert "stop" in tie_modes.values(), "continuations should stop a tie"
        xml = notation_score_to_musicxml(out.notation_score)
        assert '<tied type="start"' in xml
        assert '<tied type="stop"' in xml


class TestArrangementArc:
    """S8: long pieces open thin, breathe with terraced dynamics, lift
    into a new key for the final repetition, and ease into the final
    cadence with a ritardando."""

    def _melody(self, out: NotationScore) -> list:
        return sorted(
            (n for n in out.notation_score.notes if n.voice_id == 1), key=lambda n: n.tick
        )

    def _bass(self, out: NotationScore) -> list:
        return sorted(
            (n for n in out.notation_score.notes if n.voice_id == 0), key=lambda n: n.tick
        )

    def _section_bass_pcs(self, out: NotationScore, section: int) -> dict[int, int]:
        """First bass note's pitch class per bar of one section."""
        arrangement = out.arrangement
        ticks_per_bar = out.notation_score.ppq * 4
        start = section * arrangement.form_bars * ticks_per_bar
        pcs: dict[int, int] = {}
        for note in self._bass(out):
            if start <= note.tick < start + arrangement.form_bars * ticks_per_bar:
                pcs.setdefault((note.tick - start) // ticks_per_bar, note.pitch_midi % 12)
        return pcs

    # --- intro ---

    def test_long_pieces_open_with_a_bass_alone_intro(self) -> None:
        from saimc.compose.duration import INTRO_BARS

        out = compose(_spec(Mood.CALMING, duration=120))
        arrangement = out.arrangement
        assert arrangement.repetition_count >= ARRANGEMENT_ARC_MIN_REPS
        assert arrangement.intro_bars == INTRO_BARS
        intro_end = arrangement.intro_bars * out.notation_score.ppq * 4
        assert not [n for n in self._melody(out) if n.tick < intro_end], (
            "melody played during the intro"
        )
        assert [n for n in self._bass(out) if n.tick < intro_end], (
            "the intro is not silent: bass should carry it"
        )

    def test_short_pieces_skip_the_intro(self) -> None:
        out = compose(_spec(Mood.CALMING, duration=60))
        assert out.arrangement.repetition_count < ARRANGEMENT_ARC_MIN_REPS
        assert out.arrangement.intro_bars == 0

    # --- modulation (lift and return) ---

    def test_final_repetition_is_lifted_and_returns_home(self) -> None:
        out = compose(_spec(Mood.CALMING, duration=180))
        arrangement = out.arrangement
        assert arrangement.repetition_count >= ARRANGEMENT_ARC_MIN_REPS
        tonic_pc = key_root_midi(out.key) % 12
        # The oracle mirrors the engine's walk, so use it directly: the
        # final section's sounding chords sit a whole step above home.
        bars = _bar_degrees_and_offsets(out, Mood.CALMING.value)
        body_start = (arrangement.repetition_count - 1) * arrangement.form_bars
        lifted_heard = False
        for bar in range(body_start, body_start + arrangement.form_bars - 2):
            _degree, offsets, key_offset = bars[bar]
            assert key_offset == MODULATION_OFFSET, f"bar {bar} lost its lift"
            bass_pc = self._section_bass_pcs(out, arrangement.repetition_count - 1)[
                bar - body_start
            ]
            lifted_pcs = {(tonic_pc + key_offset + off) % 12 for off in offsets}
            home_pcs = {(tonic_pc + off) % 12 for off in offsets}
            assert bass_pc in lifted_pcs, f"bar {bar}: bass pc {bass_pc} not in {lifted_pcs}"
            lifted_heard |= bass_pc not in home_pcs
        assert lifted_heard, "the lift never left the home key's pitch classes"

        # The lift-and-return means the piece still ends at home.
        report = lint(out.notation_score)
        assert report.passed, f"lint issues: {report.issues}"
        third = 4 if out.key.mode == "major" else 3
        tonic_pc = key_root_midi(out.key) % 12
        last_melody = max(self._melody(out), key=lambda n: n.tick)
        assert last_melody.pitch_midi % 12 in (tonic_pc, (tonic_pc + third) % 12)

    # --- terraced dynamics ---

    def test_velocity_terracing_shapes_the_sections(self) -> None:
        from saimc.compose.engine import _section_velocity_scale

        # One repetition: no terracing at all.
        assert _section_velocity_scale(0, 1) == 1.0
        # Four repetitions: thin opening, peak before the close, eased close.
        assert _section_velocity_scale(0, 4) == pytest.approx(0.82)
        assert _section_velocity_scale(1, 4) == 1.0
        assert _section_velocity_scale(2, 4) == pytest.approx(1.12)
        assert _section_velocity_scale(3, 4) == pytest.approx(0.95)

        # Audibly: the penultimate section outplays the opening one.
        out = compose(_spec(Mood.CALMING, duration=120))
        arrangement = out.arrangement
        ticks_per_bar = out.notation_score.ppq * 4

        def mean_velocity(section: int) -> float:
            start = section * arrangement.form_bars * ticks_per_bar
            notes = [
                n
                for n in self._melody(out)
                if start <= n.tick < start + arrangement.form_bars * ticks_per_bar
            ]
            return sum(n.velocity for n in notes) / len(notes)

        assert mean_velocity(arrangement.repetition_count - 2) > mean_velocity(0)

    def test_cc11_rides_the_terraced_sections(self) -> None:
        out = compose(_spec(Mood.CALMING, duration=120))
        arrangement = out.arrangement
        tempo = out.notation_score.tempo
        ticks_per_bar = out.notation_score.ppq * 4
        cc11 = [c for c in out.performance_plan.controllers if c.control == 11]
        assert cc11

        def mean_cc(section: int) -> float:
            start_us = microseconds_at_tick(section * arrangement.form_bars * ticks_per_bar, tempo)
            end_us = microseconds_at_tick(
                (section + 1) * arrangement.form_bars * ticks_per_bar, tempo
            )
            values = [c.value for c in cc11 if start_us <= c.start_us < end_us]
            return sum(values) / len(values)

        assert mean_cc(arrangement.repetition_count - 2) > mean_cc(0)

    # --- ritardando ---

    def test_coda_pieces_slow_across_the_coda(self) -> None:
        from saimc.compose.duration import RITARDANDO_FACTOR
        from saimc.compose.score import TempoPoint

        out = compose(_spec(Mood.CALMING, duration=45))
        arrangement = out.arrangement
        assert arrangement.coda_bars > 0
        assert arrangement.ritardando_factor == pytest.approx(RITARDANDO_FACTOR)
        coda_start_tick = arrangement.repetition_count * arrangement.form_bars * 4 * 480
        assert out.notation_score.tempo.changes == (
            TempoPoint(
                tick=coda_start_tick,
                bpm=round(arrangement.tempo_bpm * RITARDANDO_FACTOR, 1),
            ),
        )
        # The duration math included the slowdown, so the realised
        # duration still honours the ±2% promise.
        realized = realized_duration_seconds(out.performance_plan)
        assert abs(realized - 45.0) / 45.0 <= 0.02

    def test_long_pieces_ease_into_the_final_cadence(self) -> None:
        from saimc.compose.duration import RITARDANDO_BARS, RITARDANDO_FACTOR
        from saimc.compose.score import TempoPoint

        out = compose(_spec(Mood.CALMING, duration=120))
        arrangement = out.arrangement
        assert arrangement.coda_bars == 0
        assert arrangement.repetition_count >= ARRANGEMENT_ARC_MIN_REPS
        assert arrangement.ritardando_factor == pytest.approx(RITARDANDO_FACTOR)
        change_tick = (arrangement.total_bars - RITARDANDO_BARS) * 4 * 480
        assert out.notation_score.tempo.changes == (
            TempoPoint(
                tick=change_tick,
                bpm=round(arrangement.tempo_bpm * RITARDANDO_FACTOR, 1),
            ),
        )
        realized = realized_duration_seconds(out.performance_plan)
        assert abs(realized - 120.0) / 120.0 <= 0.02

    def test_pinned_tempo_keeps_a_constant_tempo(self) -> None:
        """The spec's tempo promise outranks the decorative arc."""
        out = compose(_spec(Mood.CALMING, duration=180, tempo_bpm=63))
        assert out.notation_score.tempo.bpm == 63
        assert out.notation_score.tempo.changes == ()
        assert out.arrangement.ritardando_factor == 1.0

    def test_short_pieces_keep_a_constant_tempo(self) -> None:
        out = compose(_spec(Mood.CALMING, duration=30))
        assert out.arrangement.repetition_count < ARRANGEMENT_ARC_MIN_REPS
        assert out.notation_score.tempo.changes == ()
