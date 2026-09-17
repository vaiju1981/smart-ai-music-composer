"""Unit tests for the composition engine."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from itertools import pairwise

import pytest

from saimc.compose.duration import ARRANGEMENT_ARC_MIN_REPS, bar_ticks
from saimc.compose.engine import (
    _RANK_RUBBING,
    _START_REACH_DEGREES,
    _WALK_REACH_DEGREES,
    CompositionEngineError,
    EngineErrorCode,
    EngineOutput,
    _answer_leaps,
    _answered,
    _apex_starts,
    _bass_figure_pitches,
    _bass_ladder,
    _bed_goes_above,
    _bound_walk,
    _can_clear_the_tune,
    _chord_intervals,
    _close_bar,
    _entrance_cost,
    _entry_answer,
    _hold_tied_pairs,
    _into_harmony_register,
    _legal_slots,
    _licit_line,
    _melody_band_for,
    _merged_tie_runs,
    _octaves_in_window,
    _opening_step,
    _place_bar,
    _scale_degree_to_semitones,
    _settle_harmony_register,
    _snap_to_chord,
    _start_offsets,
    _truncate_template_for_coda,
    compose,
)
from saimc.compose.forms import (
    MODULATION_OFFSET,
    PHRASE_BARS,
    STEP_MAX_SEMITONES,
    ChordSlot,
    bar_diatonic_pcs,
    key_root_midi,
    scale_intervals,
    transposed_key,
)
from saimc.compose.linter import LintCode, legal_non_chord_tone, lint
from saimc.compose.motif import BASS_FIGURES, LEAP_DEGREES, PLAIN_BASS_FIGURE
from saimc.compose.plan import default_plan
from saimc.compose.score import (
    VOICE_BASS,
    VOICE_HARMONY,
    VOICE_MELODY,
    VOICE_PERCUSSION,
    KeySignature,
    NotationScore,
    NoteEvent,
    PerformancePlan,
    microseconds_at_tick,
    realized_duration_seconds,
    ticks_at_microsecond,
)
from saimc.compose.voices import HARMONY_MELODY_CLEARANCE
from saimc.instruments import (
    LINE_BAND_SEMITONES,
    BedRegisters,
    MelodyBand,
    bed_window,
    range_for,
)
from saimc.quality import score_piece
from saimc.spec import CompositionSpec, Instrument, Mood, WesternKey

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
    # vii m7b5: the half-diminished seventh on the leading tone (B-D-F-A
    # in C), not a diminished seventh — the A is the key's own sixth, and
    # a dim7 there would put an Ab in a bar that sounds no Ab.
    (11, 14, 17, 21),
)
MINOR_SEVENTHS_ABS = (
    (0, 3, 7, 10),
    (2, 5, 8, 12),
    (3, 7, 10, 14),
    (5, 8, 12, 15),
    (7, 10, 14, 17),
    (8, 12, 15, 19),
    # VII7 in the natural minor (Bb7 in C minor: Bb-D-F-Ab). The seventh
    # of that chord is the key's own sixth degree; a major seventh would
    # be the raised leading tone, which belongs to the harmonic minor's
    # dominant and not to this table.
    (10, 14, 17, 20),
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

    The four values the walk decides by — the modulation offset, the arc's
    repetition floor, and the cadence's degree and seventh — are read from
    `out.plan`, the plan the engine actually composed under, rather than
    from the module constants they used to be. That is the whole point of
    an oracle: a constant here would keep agreeing with the engine for as
    long as both read the same table, and would go on agreeing after one of
    them stopped — which is exactly the drift this test exists to catch.
    The mood stays a parameter because the *variant* template is still a
    fact about the mood's progression tables and not a plan field.
    """
    from saimc.compose.forms import (
        apply_final_cadence,
        get_template_for_form,
    )

    plan = out.plan
    assert plan is not None, "every compose() carries the plan it composed under"
    arrangement = out.arrangement
    lifted = arrangement.repetition_count >= plan.arc_min_reps

    result: list[tuple[int, tuple[int, ...], int]] = []
    for section in range(arrangement.repetition_count):
        template = (
            arrangement.template
            if section == 0
            else get_template_for_form(mood, arrangement.form_bars, variant_index=section)
        )
        section_offset = (
            plan.modulation_offset
            if section == arrangement.repetition_count - 1 and lifted
            else 0
        )
        if section == arrangement.repetition_count - 1:
            template = apply_final_cadence(
                template,
                cadence_degree=plan.cadence_degree,
                seventh=plan.cadence_seventh,
            )
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
            cadence_degree=plan.cadence_degree,
            seventh=plan.cadence_seventh,
        )
        coda_offset = plan.modulation_offset if lifted else 0
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

    def test_the_opening_bar_states_its_chord_with_a_library_figure(self) -> None:
        """The left hand's first bar is one of the mood's figures.

        The bass used to play exactly two tones in every bar of every
        piece, which was the whole of its vocabulary. It now plays a
        figure drawn per chord slot, so what is worth pinning is that the
        bar's onsets are one of the library's figures and that every
        pitch in it is a chord tone of the bar — and that the bar still
        opens in root position, which is what the walk is anchored on.
        Deterministic for a fixed seed, so this is a stable reading.
        """
        out = compose(_spec(Mood.CALMING, duration=60))
        bar_ticks_count = out.notation_score.ppq * 4
        bass = [n for n in out.notation_score.notes if n.voice_id == 0]
        first_bar = sorted(
            (n.tick, n.duration_ticks, n.pitch_midi)
            for n in bass
            if n.tick < bar_ticks_count
        )
        assert first_bar
        onsets = tuple((tick, duration) for tick, duration, _pitch in first_bar)
        shapes = {
            tuple(
                (offset * bar_ticks_count // 16, length * bar_ticks_count // 16)
                for offset, length, _rung in figure
            )
            for figure in BASS_FIGURES[Mood.CALMING.value]
        }
        assert onsets in shapes
        tonic = key_root_midi(out.key)
        assert first_bar[0][2] == tonic - 12  # the section opens in root position
        assert all(pitch % 12 in out.chord_bars[0] for _tick, _duration, pitch in first_bar)

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
        form_bars = arrangement.form_bars
        # Variant rotation reorders the progression, so the harmony
        # sounding at each section's downbeat differs (e.g. I vs vi).
        # Read from the engine's own per-bar chord pcs rather than from
        # the bass's first pitch: the bass used to be a proxy for this,
        # but where the walk lands is a register decision of its own and
        # says nothing about which chord the section opened on.
        assert tuple(out.chord_bars[:form_bars]) != tuple(out.chord_bars[form_bars : 2 * form_bars])
        assert out.chord_bars[0] != out.chord_bars[form_bars]

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
        """Every pitched note is a chord tone of its bar, or a licensed tone.

        The licence is not a loophole: a non-chord tone has to be brief,
        unaccented, diatonic to the bar's own harmony, and entered *and*
        left by step from the notes either side of it in the same voice.
        That is the whole vocabulary the engine composes from, so a note
        outside both is a harmony bug rather than an expressive one.
        """
        out = compose(_spec(mood, duration=duration, seed=42))
        bars = _bar_degrees_and_offsets(out, mood.value)
        assert len(bars) == out.arrangement.total_bars_with_coda
        score = out.notation_score
        ticks_per_bar = score.ppq * 4
        tonic = key_root_midi(out.key)
        anticipation_zone = ticks_per_bar - score.ppq // 2
        by_voice: dict[int, list] = {}
        for note in score.notes:
            by_voice.setdefault(note.voice_id, []).append(note)
        for voice_id, voice_notes in by_voice.items():
            if voice_id == 2:  # percussion keys are GM drum map, not pitched
                continue
            voice_notes.sort(key=lambda n: (n.tick, n.pitch_midi))
            for index, note in enumerate(voice_notes):
                bar = note.tick // ticks_per_bar
                _degree, offsets, key_offset = bars[bar]
                chord = {(tonic + key_offset + offset) % 12 for offset in offsets}
                sounding = set(chord)
                if note.tick % ticks_per_bar >= anticipation_zone and bar + 1 < len(bars):
                    # An anacrusis pickup anticipates the next bar's chord.
                    _next_degree, next_offsets, next_offset = bars[bar + 1]
                    sounding |= {
                        (tonic + next_offset + offset) % 12 for offset in next_offsets
                    }
                if note.pitch_midi % 12 in sounding:
                    continue
                assert legal_non_chord_tone(
                    note,
                    prev=voice_notes[index - 1] if index else None,
                    nxt=(
                        voice_notes[index + 1]
                        if index + 1 < len(voice_notes)
                        else None
                    ),
                    bar_start_tick=bar * ticks_per_bar,
                    ppq=score.ppq,
                    diatonic_pcs=bar_diatonic_pcs(
                        tuple(chord),
                        out.bar_keys[bar] if out.bar_keys else out.key,
                    ),
                ), (
                    f"voice {voice_id} bar {bar}: pitch {note.pitch_midi} at "
                    f"{note.tick} is neither a chord tone of {sounding} "
                    f"(offsets {offsets}) nor a licensed passing tone"
                )

    def test_the_oracle_and_the_engine_spell_the_same_chords(self) -> None:
        """`_bar_degrees_and_offsets` spells each slot's chord from the
        oracle tables above; the engine spells it from `forms`. The oracle
        is only a check on the engine while the two agree, so this pins
        them together — a degree's scale-semitone offset is its chord's
        root, and the chord's intervals rise from there onto the oracle's
        own tones.

        Without it, a drift in the shared scale table would show up as the
        chord-tone tests failing on notes the engine got right.
        """
        for mode, key in (
            ("major", KeySignature(root="C", mode="major")),
            ("minor", KeySignature(root="A", mode="minor")),
        ):
            for degree in range(7):
                for seventh in (False, True):
                    slot = ChordSlot(degree=degree, bars=1, seventh=seventh)
                    tones = _chord_offsets(slot, mode)
                    root = _scale_degree_to_semitones(degree, mode)
                    assert root == tones[0], (mode, degree, seventh)
                    intervals = _chord_intervals(degree, key, seventh=seventh)
                    assert tuple(root + interval for interval in intervals) == tones, (
                        mode,
                        degree,
                        seventh,
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
        harmony spells the lowered-root triad rather than the diatonic
        vii, its downbeats land on that triad, and what the melody plays
        between those downbeats is a chord tone of it or a licensed
        stepwise tone.

        This used to pin every pitched note of the bar to the borrowed
        triad, which was true of the chord-tone walk it was written for.
        The melody now moves in scale degrees, so a bar's line passes
        *through* its harmony — a step is a semitone or a whole tone and
        the tones between the anchors are the ones a step explains. The
        borrowed chord still has to be what sounds, so the anchor half of
        the old assertion stays (via the licence, which admits no
        non-chord tone on a downbeat), and the stepwise tones are checked
        against the licence instead of being forbidden.
        """
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
        score = out.notation_score
        ticks_per_bar = score.ppq * 4
        by_voice: dict[int, list] = {}
        for note in score.notes:
            if note.voice_id != VOICE_PERCUSSION:
                by_voice.setdefault(note.voice_id, []).append(note)
        for voice_notes in by_voice.values():
            voice_notes.sort(key=lambda n: (n.tick, n.pitch_midi))
        for bar in borrowed_bars:
            sounding = {(tonic + offset) % 12 for offset in (10, 14, 17)}
            # The chord the bar publishes is the borrowed triad: the
            # template degree alone would spell the diatonic vii.
            assert set(out.chord_bars[bar]) == sounding, (
                f"bar {bar}: the score sounds {sorted(out.chord_bars[bar])}, "
                f"not the borrowed bVII {sorted(sounding)}"
            )
            diatonic = bar_diatonic_pcs(
                out.chord_bars[bar],
                out.bar_keys[bar] if out.bar_keys else out.key,
            )
            # The bar's final eighth is the anacrusis zone, where a pickup
            # may anticipate the chord it leads into.
            anticipation = (bar + 1) * ticks_per_bar - score.ppq // 2
            into_next = set(out.chord_bars[bar + 1]) if bar + 1 < len(out.chord_bars) else set()
            for voice_notes in by_voice.values():
                for index, note in enumerate(voice_notes):
                    if note.tick // ticks_per_bar != bar:
                        continue
                    if note.pitch_midi % 12 in sounding:
                        continue
                    if note.tick >= anticipation and note.pitch_midi % 12 in into_next:
                        continue
                    assert legal_non_chord_tone(
                        note,
                        prev=voice_notes[index - 1] if index else None,
                        nxt=(
                            voice_notes[index + 1] if index + 1 < len(voice_notes) else None
                        ),
                        bar_start_tick=bar * ticks_per_bar,
                        ppq=score.ppq,
                        diatonic_pcs=diatonic,
                    ), (
                        f"voice {note.voice_id} bar {bar}: pitch {note.pitch_midi} at "
                        f"{note.tick} is neither a tone of the borrowed bVII "
                        f"{sorted(sounding)} nor a licensed stepwise tone"
                    )

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

    def test_every_downbeat_is_a_chord_tone(self) -> None:
        """The bar's first melody note is a tone of the bar's own chord.

        This replaces a check that only half the downbeats were the chord's
        root or third, which the bar-by-bar walk satisfied by accident: the
        old melody was arpeggiated from the chord, so a downbeat could be
        any tone of it, and the third was commoner than the fifth only
        because the walk started there.

        The walk now moves in scale degrees, so what the contract can still
        promise is the anchor: notes *between* downbeats are steps and may
        be licensed non-chord tones, but every note the harmony comes to
        rest on is a tone of the chord underneath it. That is precisely why
        the passing-tone licence admits no non-chord tone on a downbeat, so
        the two rules are one rule read from either side.
        """
        out = compose(_spec(Mood.CALMING, duration=180))
        score = out.notation_score
        ticks_per_bar = score.ppq * 4
        tonic = key_root_midi(out.key)
        bars = _bar_degrees_and_offsets(out, Mood.CALMING.value)
        assert len(bars) == out.arrangement.total_bars_with_coda
        downbeats = [
            n
            for n in score.notes
            if n.voice_id == VOICE_MELODY and n.tick % ticks_per_bar == 0
        ]
        assert downbeats
        for note in downbeats:
            bar = note.tick // ticks_per_bar
            degree, offsets, key_offset = bars[bar]
            chord = {(tonic + key_offset + offset) % 12 for offset in offsets}
            assert note.pitch_midi % 12 in chord, (
                f"bar {bar}: downbeat {note.pitch_midi} is not a tone of the "
                f"bar's chord (degree {degree}, pcs {sorted(chord)})"
            )

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


class TestBassFigures:
    """The left hand's figures, resolved onto a chord and a register.

    A figure is written in rungs, not pitches, so one shape serves every
    chord of a progression: `_bass_ladder` is the chord's own tones
    rising from the bar's landing tone, and `_bass_figure_pitches` turns
    the rungs into pitches on the chord the slot actually has, in the
    register the walk actually landed in. Each rule here is one the
    accompaniment pays for if it goes wrong.
    """

    def test_the_ladder_rises_through_the_chords_own_tones(self) -> None:
        assert _bass_ladder(48, chord_root=60, chord_tones=(0, 4, 7)) == (48, 52, 55, 60)
        # A seventh chord's fourth rung is its seventh, so the same figure
        # reaches further without being rewritten.
        assert _bass_ladder(55, chord_root=67, chord_tones=(0, 4, 7, 10)) == (
            55,
            59,
            62,
            65,
            67,
        )

    def test_a_figure_resolves_to_chord_tones_above_its_landing_tone(self) -> None:
        resolved = _bass_figure_pitches(
            PLAIN_BASS_FIGURE, anchor=48, chord_root=60, chord_tones=(0, 4, 7)
        )
        assert resolved == ((0, 8, 48), (8, 8, 52))
        assert min(pitch for _start, _length, pitch in resolved) == 48

    def test_a_rung_that_cannot_fit_the_register_is_dropped(self) -> None:
        """The figure is written for the harmony, not for the register.

        A seventh chord puts its fourth rung a tenth above the landing
        tone, which on a walk that has landed high sits past the bass
        ceiling. Its octave down falls below the bar's own landing tone —
        a note under the tone the bar opened on is a different figure — so
        it goes, and every rung that does fit keeps the onset the figure
        gave it.
        """
        driving = BASS_FIGURES["electrifying"][0]
        assert _bass_figure_pitches(
            driving, anchor=60, chord_root=60, chord_tones=(0, 4, 7, 10)
        ) == ((0, 4, 60), (4, 4, 64), (8, 4, 67))

    def test_every_figure_lands_on_chord_tones_never_below_its_anchor(self) -> None:
        """What the piece-level walk test reads off the score.

        That test takes each bar's lowest bass note as the bar's landing
        tone and requires the landing tones to move in small steps. This
        is the property that makes it the landing tone: rung 0 sounds the
        anchor on the bar line and no later rung is resolved below it.
        """
        chords = ((60, (0, 4, 7)), (67, (0, 4, 7, 10)), (60, (0, 4, 7, 10)))
        for mood, figures in BASS_FIGURES.items():
            for figure in figures:
                for anchor, (chord_root, tones) in zip((48, 55, 60), chords, strict=True):
                    resolved = _bass_figure_pitches(
                        figure, anchor=anchor, chord_root=chord_root, chord_tones=tones
                    )
                    assert resolved, (mood, figure, anchor)
                    assert resolved[0] == (0, figure[0][1], anchor)
                    assert min(pitch for _start, _length, pitch in resolved) == anchor
                    assert all(
                        pitch % 12 in {(chord_root + tone) % 12 for tone in tones}
                        for _start, _length, pitch in resolved
                    )


class TestMelodyWalk:
    """One bar's line, one pass at a time.

    A bar is written in four passes — fold the walk into one octave, answer
    its leaps, bend the line around the notes the licence cannot cover, then
    choose the octave it sits in — and each of them is the reason some
    metric clears its bar. The piece-level tests elsewhere in this file say
    *that* the melody moved; these say where, so a regression names itself
    instead of arriving as a metric that quietly got worse.
    """

    def test_bound_walk_folds_a_climb_into_one_octave(self) -> None:
        """A sequence climbing a chord tone per replay comes back down.

        The band is centred on the degree the bar starts on — the walk is
        folded to within an octave *of where it is*, not of a fixed register
        — so the same line folded from a different anchor keeps its own
        centre, and the caller keeps the bar's register.
        """
        walk = [0, 3, 6, 9, 12, 15, 18]
        folded = _bound_walk(walk, walk[0])
        assert folded == [0, 3, -1, 2, 5, 1, 4]
        for before, after in zip(walk, folded, strict=True):
            assert after % 7 == before % 7, "a fold is an octave, not a transposition"
        assert all(abs(degree - walk[0]) <= _WALK_REACH_DEGREES for degree in folded)

    def test_a_leap_is_answered_by_a_turn_back(self) -> None:
        """The answer is the whole figure: the step back, and the step that
        leaves it — a passing tone between the chord tones either side."""
        assert _answer_leaps([0, 4, 5, 6]) == [0, 4, 3, 2]

    def test_a_leap_that_lands_off_the_chord_is_undone(self) -> None:
        """A non-chord tone is entered by a step or not at all, so a leap
        landing off the harmony keeps the bar's contour and loses the leap."""
        assert _answer_leaps([0, 5, 6]) == [0, -1, -2]

    def test_a_leap_with_no_room_after_it_is_answered_from_before(self) -> None:
        """The landing is where the bar has to be, so the note before it
        takes the step — the way a cadence is approached."""
        assert _answer_leaps([0, 4, 6]) == [3, 4, 6]
        assert _answer_leaps([0, 4, 5, 6], fixed_tail=1) == [3, 4, 5, 6]

    def test_a_leap_the_bar_cannot_answer_is_left_to_the_next_bar(self) -> None:
        """The pass settles a slot as the approach to its pair, and a
        settled slot is never written again — a bar whose answer has to come
        from before the landing can be re-leapt by the repair of the pair in
        front of it, which is what the engine hung on. So the walk settles
        instead, and the leap it cannot answer is answered by the next bar's
        entrance (`_entry_answer`).

        What this test really asserts is that the call returns at all: a
        regression in the settling rule is an infinite loop, and the harness
        turns that into a timeout here rather than into a hung render.
        """
        walk = [0, 4, 5, 3, 1, 5, 9, 11]
        answered = _answer_leaps(walk)
        assert len(answered) == len(walk), "the pass rewrites, it does not re-time"
        assert answered[1] - answered[0] >= LEAP_DEGREES, (
            "the opening leap has both sides settled and is left standing"
        )

    def test_licit_line_bends_the_note_that_arrives_by_a_third(self) -> None:
        """A non-chord tone owes a step on each side, so the note that
        arrives at one by a third moves by a degree — the one nearer the
        note on its far side."""
        assert _licit_line([0, 1, 5], remainders=(0, 2, 4)) == [0, 1, 2]
        assert _licit_line([0, 1, 5, 6], remainders=(0, 2, 4)) == [0, 1, 2, 3]

    def test_licit_line_never_moves_a_pinned_closing_note(self) -> None:
        """A closing gesture lands where it lands; a line that cannot bend
        toward it is left for the snap pass, which is the one repair allowed
        to move a note the bar has pinned."""
        assert _licit_line([0, 3, 4], remainders=(0, 2, 4), fixed_tail=1) == [0, 3, 4]

    def test_snap_to_chord_stays_a_step_away(self) -> None:
        """A chord scale's tones sit on every other degree, so a snap is a
        move of one degree for every tone that is not already a chord tone."""
        for degree in range(7):
            if degree in (0, 2, 4):
                continue
            for prefer_up in (True, False):
                snapped = _snap_to_chord(degree, tone_count=3, prefer_up=prefer_up)
                assert snapped % 7 in (0, 2, 4)
                assert abs(snapped - degree) == 1, (degree, prefer_up)

    def test_snap_to_chord_weighs_the_neighbours_it_strands(self) -> None:
        """A snap decides the intervals the notes either side are heard on,
        so a chord tone a third from a neighbour the licence covers loses to
        one that keeps that neighbour a step away."""
        snapped = _snap_to_chord(1, tone_count=3, prefer_up=True, neighbours=(0, 3))
        assert snapped == 2, "the downward chord tone (0) strands the neighbour at 3"

    def test_the_licence_keeps_a_passing_tone_and_takes_the_rest(self) -> None:
        """`_legal_slots` is the bar's floor: what the walk could not get
        past the linter, the snap pass rewrites, and it rewrites the degree
        only — a bar keeps its rhythm whatever the licence does to it."""
        scale = scale_intervals(0, "major")

        def bar(degrees, **kw):
            slots = [(index * 480, 480, degree, 0) for index, degree in enumerate(degrees)]
            return _legal_slots(
                slots, tone_count=3, chord_root=60, scale=scale, **kw
            )

        passing = [(0, 480, 0, 0), (480, 480, 1, 0), (960, 480, 2, 0)]
        assert bar((0, 1, 2)) == passing, "entered and left by a step is legal"

        # A non-chord tone on the downbeat is an appoggiatura, which this
        # engine does not model: it snaps to the chord the bar rests on.
        assert bar((1, 1, 2))[0][2] == 0
        # A rub against another voice's pitch class is decided in pitch
        # class, where no octave placement can undo it.
        assert bar((0, 1, 2), avoid_pcs=frozenset({3}))[1][2] == 2
        assert bar((0, 1, 2), avoid_pcs=frozenset({6}))[1][2] == 1
        # A quarter is the longest a non-chord tone may be.
        assert bar((0, 1))[1][2] == 2
        assert _legal_slots(
            [(0, 480, 0, 0), (480, 960, 1, 0)], tone_count=3, chord_root=60, scale=scale
        ) == [(0, 480, 0, 0), (480, 960, 2, 0)]

    def test_a_tied_pair_snaps_as_one_note(self) -> None:
        """Two noteheads at one pitch stand or fall together — snapped apart
        they would be a tie the engraver draws between two heights, and a
        pitch the performance layer, which plays the continuation as
        nothing, drops."""
        snapped = _legal_slots(
            [(0, 480, 1, 1), (480, 480, 1, 0), (960, 480, 2, 0)],
            tone_count=3,
            chord_root=60,
            scale=scale_intervals(0, "major"),
        )
        assert [slot[2] for slot in snapped] == [0, 0, 2]
        assert snapped[0][3] == 1, "the tie itself survives the snap"

    def test_hold_tied_pairs_writes_the_second_notehead_as_the_first(self) -> None:
        """The passes between the rhythm library and the licence rewrite a
        note without knowing which notes are tied to their neighbour, so the
        pair is re-held before the licence reads it."""
        held = _hold_tied_pairs([(0, 480, 0, 1), (480, 240, 1, 0), (720, 240, 4, 0)])
        assert held == [(0, 480, 0, 1), (480, 240, 0, 0), (720, 240, 4, 0)]

    def test_close_bar_writes_the_closing_gesture(self) -> None:
        slots = [(0, 480, 0, 0), (480, 480, 4, 0)]
        assert _close_bar(slots, degree=2, ticks=None) == [
            (0, 480, 0, 0),
            (480, 480, 2, 0),
        ]

    def test_close_bar_shortens_a_breathing_bar(self) -> None:
        """A phrase breathes by halving what it had, not by leaving the
        note there: the rest is what the ear hears as the breath."""
        slots = [(0, 480, 0, 0), (480, 480, 4, 0)]
        assert _close_bar(slots, degree=None, ticks=240) == [
            (0, 480, 0, 0),
            (480, 240, 4, 0),
        ]

    def test_close_bar_yields_a_tie_the_gesture_broke(self) -> None:
        """A tie holds one pitch and the cadence now lands elsewhere, so the
        tie yields — the phrase asked for the gesture, not for the tie."""
        slots = [(0, 480, 0, 1), (480, 480, 4, 0)]
        assert _close_bar(slots, degree=2, ticks=None) == [
            (0, 480, 0, 0),
            (480, 480, 2, 0),
        ]
        kept = [(0, 480, 0, 1), (480, 480, 4, 0)]
        assert _close_bar(kept, degree=4, ticks=None) == kept, (
            "a gesture that lands where the bar already was leaves the tie alone"
        )

    def test_close_bar_is_a_noop_without_a_gesture(self) -> None:
        slots = [(0, 480, 0, 0), (480, 480, 4, 0)]
        assert _close_bar(slots, degree=None, ticks=None) == slots
        assert _close_bar([], degree=2, ticks=None) == []

    def test_opening_step_skips_a_tied_notehead(self) -> None:
        """A tied continuation is not struck, so the interval the ear hears
        first is the one out of the tie — reading it as a repeat would let a
        bar turn a seam it has already answered."""
        slots = [(0, 480, 0, 0), (480, 480, 1, 0), (960, 480, 2, 0)]
        assert _opening_step([60, 62, 64], slots) == 2
        tied = [(0, 480, 0, 1), (480, 480, 1, 0), (960, 480, 2, 0)]
        assert _opening_step([60, 60, 62], tied) == 2, "the move out of the tie"
        assert _opening_step([60], slots[:1]) is None, "one note has no move"
        assert _opening_step([60, 60], [(0, 480, 0, 1), (480, 480, 1, 1)]) is None

    def test_entrance_cost_grades_a_step_above_a_skip_above_a_leap(self) -> None:
        """A step into a bar is what a melody does; a skip still moves and
        still comes back; a repeat is a note the bar did not need; a leap is
        what the listener has to recover from."""
        assert _entrance_cost(None, None) == 0, "the first bar owes nothing"
        assert _entrance_cost(2, None) == 0, "a step"
        assert _entrance_cost(3, None) == _entrance_cost(4, None) == 1, "a skip"
        assert _entrance_cost(0, None) == 2, "a repeat"
        assert _entrance_cost(7, None) == _entrance_cost(12, None) == 3, "a leap"

    def test_a_leap_makes_one_entrance_free_and_the_rest_faults(self) -> None:
        """A leap has to be answered, and the bar's second note is what
        answers it — so a step back the other way is the one way in that
        costs nothing, and a step carrying on the same way is a fault."""
        assert _entrance_cost(-2, 7) == 0, "the step that answers the leap"
        assert _entrance_cost(-7, 7) != 0, "a leap back is still a leap"
        for entrance in (0, 2, 3, -12):
            assert _entrance_cost(entrance, 7) == 3, entrance
        assert _entrance_cost(None, 7) == 0, "a bar after a rest owes no answer"

    def test_answered_and_entry_answer_read_the_same_seam(self) -> None:
        """A seam is answered when the bar's first *sounding* move is a step
        back the way the leap came; `_entry_answer` is the degree step that
        makes that true, and the two must agree on every seam — the ranker
        judges with one and the walk repairs with the other."""
        seams = [
            (None, None),
            (2, None),
            (4, None),
            (7, None),
            (7, 0),
            (7, 2),
            (7, -2),
            (-7, None),
            (-7, 3),
            (-7, -3),
            (12, 5),
        ]
        for entrance, opening in seams:
            answer = _entry_answer(entrance, opening)
            assert (answer is None) == _answered(entrance, opening), (entrance, opening)
            if answer is not None:
                assert answer == (-1 if entrance > 0 else 1), (entrance, opening)

    def test_start_offsets_lead_with_the_drawn_anchor(self) -> None:
        """Every tone of the bar's chord is a place its line can begin, so a
        bar can be restated into another register without rewriting a note —
        and the drawn anchor leads, because the caller keeps it unless
        another placement fits better."""
        for anchor in range(3):
            offsets = _start_offsets(anchor, 3)
            assert offsets[0] == 2 * anchor, "the drawn anchor leads"
            assert len(set(offsets)) == len(offsets)
            for offset in offsets:
                assert offset % 7 in (0, 2, 4), offset
                assert abs(offset) <= _START_REACH_DEGREES, offset

    def test_apex_starts_lift_without_an_octave_jump(self) -> None:
        """The apex is the one bar whose register is chosen rather than
        fitted, and it is chosen *above* where the bar already stands and
        *inside* the octave: an octave lift is the same statement made by a
        leap the listener has to recover from."""
        for anchor in range(3):
            drawn = 2 * anchor
            starts = _apex_starts(anchor, 3)
            assert starts, "any six consecutive degrees hold two tones of a triad"
            assert all(drawn < offset < drawn + 7 for offset in starts), (anchor, starts)

    def test_a_bar_the_band_cannot_hold_keeps_its_line(self) -> None:
        """The band's edge is paid for by a bar's register, never by a note.

        A line wider than the band cannot be fitted at any octave, and
        this is the case the placement has to answer without touching the
        line: the notes left outside are the two ends of it, and every
        interval between the notes is the one the walk wrote. The
        alternative — folding the stray notes an octave one at a time —
        turns a scale into a tear in the middle of the bar, and the
        licence pass has already run by then, so the score fails to lint
        and the whole piece raises.

        The band here is the narrowest the registry can produce (21
        semitones) and the line is 24 wide, so the edge is reached by
        construction rather than by hunting for a seed that reaches it —
        with a per-instrument band no seeded piece does: measured over
        600 pieces across the three moods, every melody bar sits inside
        its own instrument's band.
        """
        band = MelodyBand(low_midi=60, high_midi=81)
        line = [58, 60, 62, 63, 65, 67, 68, 70, 72, 74, 75, 77, 79, 80, 82]
        _rank, shift, outside, _rubbing, _entrance = _place_bar(
            line, band=band, prev_pitch=None, apex=False
        )
        # The first key of the ranking, stated as the property rather than
        # as the number: no octave of the whole bar leaves fewer notes
        # out than the one that was chosen.
        best_possible = min(
            sum(1 for pitch in line if not band.contains(pitch + 12 * octave))
            for octave in range(-3, 4)
        )
        assert outside == best_possible, (outside, best_possible)
        assert outside > 0, "a 24-semitone line does not fit a 21-semitone band"
        shifted = [pitch + shift for pitch in line]
        assert [later - earlier for earlier, later in pairwise(shifted)] == [
            later - earlier for earlier, later in pairwise(line)
        ], "the line comes through the shift unchanged"
        # The excursion is the edge of the line, not a hole in it: the
        # notes inside the band are one unbroken run, so nothing was
        # folded while its neighbours stayed put.
        inside = [band.contains(pitch) for pitch in shifted]
        transitions = sum(
            1 for earlier, later in pairwise(inside) if earlier != later
        )
        assert any(inside), (shifted, inside)
        assert transitions <= 2, (shifted, inside)
        assert shifted[0] < band.low_midi or shifted[-1] > band.high_midi
        assert all(
            abs(later - earlier) <= STEP_MAX_SEMITONES
            for earlier, later in pairwise(shifted)
        ), shifted

    def test_place_bar_counts_rubbing_only_off_the_bar_chord(self) -> None:
        """Two tones of the bar's own chord are a voicing; only the notes off
        it can rub the bass, and those are the ones the licence pass would
        snap away anyway. The count is what steers the octave choice, so an
        exemption that leaked into it would let a real collision through."""
        band = MelodyBand(low_midi=60, high_midi=84)
        _rank, _shift, _outside, rubbing, _entrance = _place_bar(
            [64], band=band, prev_pitch=None, apex=False, bass_pitches=(65,)
        )
        assert rubbing == 1, "64 against a sounding 65 is the m2 the linter refuses"
        _rank, _shift, _outside, exempt, _entrance = _place_bar(
            [64],
            band=band,
            prev_pitch=None,
            apex=False,
            bass_pitches=(65,),
            chord_pcs=frozenset({64 % 12}),
        )
        assert exempt == 0

    def test_place_bar_ranks_both_bar_shapes_on_one_scale(self) -> None:
        """The apex ranks its octaves on its own key order — the height it
        earns comes before the approach — and every other bar on the
        approach. What the two shapes share is the front: the notes left
        outside the band, then the notes left rubbing the bass. The caller
        compares a start's ranking against these, so `_RANK_RUBBING` names
        the index in both and this is what keeps it honest."""
        band = MelodyBand(low_midi=60, high_midi=84)
        for apex in (False, True):
            rank, _shift, outside, rubbing, _entrance = _place_bar(
                [64, 67], band=band, prev_pitch=None, apex=apex, bass_pitches=(65,)
            )
            assert len(rank) == (8 if apex else 7), (apex, rank)
            assert rank[0] == outside, (apex, rank)
            assert rank[_RANK_RUBBING] == rubbing, (apex, rank, rubbing)


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
        """Where the melody leaves its bar's last eighth, the note it puts
        there leads into the bar it abuts: it steps out of the note before
        it and it is a tone of the chord it enters.

        A score cannot tell a pickup from the bar's own last slot — both
        abut the bar line and both are eighth notes — so the two are told
        apart by harmony, which is what distinguishes them in the music.
        A note there that is a tone of the bar it sits in is the bar's own
        last slot and is left alone; one that is not is a pickup on the
        next chord (or a licensed passing tone), and is required to step
        out of its predecessor and to belong to the bar it leads into.

        Asserted over several seeds as a floor rather than a fixed count
        at one. Whether a bar leaves its last eighth free is a rhythm
        draw, and whether the free eighth finds a step onto a tone of the
        next chord is a harmony draw on top of it — so a fixed count
        measures the order the draws happen to fall in, which is what the
        melody rewrite changed, and not the device, which it did not.
        Measured across seeds, roughly a quarter to a half of bars leave
        that eighth and the anacrusis itself lands in several of them.
        """
        for seed in (42, 1, 7, 99):
            out = compose(_spec(Mood.ELECTRIFYING, duration=60, seed=seed))
            score = out.notation_score
            ppq = score.ppq
            ticks_per_bar = 4 * ppq
            melody = self._melody(out)
            last_eighth = [
                (index, note)
                for index, note in enumerate(melody)
                if note.tick % ticks_per_bar == ticks_per_bar - ppq // 2
                and note.duration_ticks == ppq // 2
            ]
            bars = out.arrangement.total_bars_with_coda
            assert len(last_eighth) >= 3, (
                f"seed {seed}: only {len(last_eighth)} of {bars} bars left "
                "their last eighth for a leading note"
            )
            for index, note in last_eighth:
                bar = note.tick // ticks_per_bar
                chord = set(out.chord_bars[bar])
                if note.pitch_midi % 12 in chord:
                    continue  # the bar's own last slot, part of its line
                # Not a tone of the bar it sits in, so it is a pickup on
                # the next chord or a licensed stepwise tone. Both are
                # entered by step, and the harmony it belongs to is the
                # bar it abuts.
                into_next = (
                    set(out.chord_bars[bar + 1]) if bar + 1 < len(out.chord_bars) else set()
                )
                previous = melody[index - 1] if index else None
                stepped = (
                    previous is not None
                    and 0 < abs(note.pitch_midi - previous.pitch_midi) <= STEP_MAX_SEMITONES
                )
                assert stepped, (
                    f"seed {seed}: the note at {note.tick} leaves bar {bar}'s "
                    f"chord {sorted(chord)} without stepping out of {previous}"
                )
                assert note.pitch_midi % 12 in into_next or legal_non_chord_tone(
                    note,
                    prev=previous,
                    nxt=melody[index + 1] if index + 1 < len(melody) else None,
                    bar_start_tick=bar * ticks_per_bar,
                    ppq=ppq,
                    diatonic_pcs=bar_diatonic_pcs(
                        out.chord_bars[bar],
                        out.bar_keys[bar] if out.bar_keys else out.key,
                    ),
                ), (
                    f"seed {seed}: the note at {note.tick} is a tone of neither "
                    f"bar {bar} nor bar {bar + 1} and is not a licensed tone"
                )

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

    def test_the_oracle_reads_the_plan_the_engine_composed_under(self) -> None:
        """The oracle's plan reads, held against the engine's own output.

        `_bar_degrees_and_offsets` used to read `MODULATION_OFFSET` and
        `ARRANGEMENT_ARC_MIN_REPS` from the module. Those are the *default
        plan's* values, so the oracle agreed with the engine for exactly as
        long as both read the same constant — including after one of them
        stopped. It reads `out.plan` now, so the two can only agree here if
        the engine really composed under the plan it published.

        Composed at a lift of 9 semitones rather than the default 2, and
        the engine's bar keys are what the oracle is checked against: a
        `compose` that ignored its plan would publish the plan it was
        handed, lift by 2 anyway, and fail this.
        """
        spec = _spec(Mood.CALMING, duration=180)
        plan = replace(default_plan(spec), modulation_offset=9)
        out = compose(spec, plan=plan)

        arrangement = out.arrangement
        assert arrangement.repetition_count >= plan.arc_min_reps, "expected a lifted piece"
        bars = _bar_degrees_and_offsets(out, Mood.CALMING.value)
        home = out.bar_keys[0]
        final_section = (arrangement.repetition_count - 1) * arrangement.form_bars
        for bar in range(final_section, final_section + arrangement.form_bars - 2):
            _degree, _offsets, key_offset = bars[bar]
            assert key_offset == 9, f"bar {bar}: the oracle forgot the plan's lift"
            assert out.bar_keys[bar] == transposed_key(home, 9), (
                f"bar {bar}: the engine lifted by something other than the plan's 9"
            )

    def test_every_bar_publishes_the_key_its_harmony_belongs_to(self) -> None:
        # The lifted IV of a piece in C is a G major triad, which is also
        # the home key's V — the chord cannot say which key the bar is in,
        # and the linter's passing-tone licence needs to know. So the
        # engine publishes the key per bar, and the last two bars of the
        # section come home with the cadence.
        out = compose(_spec(Mood.CALMING, duration=180))
        arrangement = out.arrangement
        assert len(out.bar_keys) == len(out.chord_bars) == arrangement.total_bars_with_coda
        home = out.bar_keys[0]
        assert home == out.key
        final_section = (arrangement.repetition_count - 1) * arrangement.form_bars
        lifted = {
            bar - final_section
            for bar in range(final_section, arrangement.total_bars)
            if out.bar_keys[bar] != home
        }
        assert lifted == set(range(arrangement.form_bars - 2)), sorted(lifted)
        for bar in range(arrangement.total_bars):
            assert out.bar_keys[bar] in (home, transposed_key(home, MODULATION_OFFSET))

    @pytest.mark.parametrize("duration", [210, 240, 270, 420, 480, 540])
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_a_lifted_piece_composes_and_lints(self, duration: int, seed: int) -> None:
        # These were the specs that failed: the lift was applied to the
        # notes and to the published chords, but the licence read their
        # bars against the home key, so a lifted bar whose chord is also
        # the home key's (a IV read as a V) refused the new key's own
        # notes and the whole compose raised `lint_failed`.
        out = compose(_spec(Mood.ELECTRIFYING, duration=duration, seed=seed))
        report = lint(
            out.notation_score,
            chord_bars=out.chord_bars,
            bar_keys=out.bar_keys,
            voice_instruments={v.voice_id: v.instrument for v in out.voice_instruments},
        )
        assert report.passed, f"{duration}s seed {seed}: {report.issues}"

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


class TestS9Consolidation:
    """S9: the linter's musical gates hold across the generation space."""

    def test_seed_sweep_composes_without_lint_failure(self) -> None:
        """compose() lints internally; a failure raises. The coda's
        final bar used to resolve off-tonic (its _generate_section call
        was missing is_final_section) and surfaced as ENDS_OFF_TONIC on
        unlucky seeds — this sweep is the regression net.

        Both axes have to be swept: the coda's shape comes from the
        duration and the phrase structure from the seed, and a bar's
        register depends on both, so a duration the sweep skips is a
        hole in the net. The band's own edge is where the hole was —
        `electrifying` at seed 99 needed 90 seconds to reach a bar the
        placement could not fit, and 90 was the one typical duration
        missing from the list.
        """
        for mood in Mood:
            for duration in (30, 45, 60, 90, 120, 300):
                for seed in (42, 1, 7, 99):
                    compose(_spec(mood, duration=duration, seed=seed))

    def test_chord_bars_cover_every_bar(self) -> None:
        out = compose(_spec(Mood.CALMING, duration=120, seed=42))
        total_bars = out.arrangement.total_bars_with_coda
        assert len(out.chord_bars) == total_bars
        assert all(bar for bar in out.chord_bars)
        assert all(0 <= pc <= 11 for bar in out.chord_bars for pc in bar)

    def test_melody_never_runs_past_a_phrase(self) -> None:
        for mood in Mood:
            for duration in (30, 120, 300):
                out = compose(_spec(mood, duration=duration, seed=99))
                spans = _melody_span_bars(out)
                assert spans, "the melody should exist"
                assert max(spans) <= PHRASE_BARS

    def test_no_close_dissonant_collisions(self) -> None:
        for mood in Mood:
            for duration in (45, 120):
                out = compose(_spec(mood, duration=duration, seed=1))
                report = lint(out.notation_score, chord_bars=out.chord_bars)
                assert not [
                    i for i in report.issues if i.code == LintCode.DISSONANT_COLLISION
                ]

    def test_coda_piece_ends_at_home(self) -> None:
        # The coda is the piece's true ending: its last melody note
        # must resolve to the tonic or its third.
        out = compose(_spec(Mood.CALMING, duration=45, seed=99))
        assert out.arrangement.coda_bars > 0
        melody = [n for n in out.notation_score.notes if n.voice_id == VOICE_MELODY]
        last = max(melody, key=lambda n: (n.tick, n.pitch_midi))
        tonic_pc = key_root_midi(out.key) % 12
        third_pc = (tonic_pc + (4 if out.key.mode == "major" else 3)) % 12
        assert last.pitch_midi % 12 in (tonic_pc, third_pc)


def _melody_span_bars(out) -> list[float]:
    """Continuous melody spans in bars, anacrusis pickups excluded."""
    score = out.notation_score
    ticks_per_bar = bar_ticks(out.time_signature)
    eighth = score.ppq // 2
    melody = sorted(
        (
            n
            for n in score.notes
            if n.voice_id == VOICE_MELODY
            and not (
                n.tick % ticks_per_bar == ticks_per_bar - eighth
                and n.duration_ticks <= eighth
            )
        ),
        key=lambda n: (n.tick, n.pitch_midi),
    )
    spans: list[float] = []
    if len(melody) < 2:
        return spans
    group_start = melody[0].tick
    prev = melody[0]
    for note in melody[1:]:
        if note.tick > prev.tick + prev.duration_ticks:
            spans.append((prev.tick + prev.duration_ticks - group_start) / ticks_per_bar)
            group_start = note.tick
        prev = note
    spans.append((prev.tick + prev.duration_ticks - group_start) / ticks_per_bar)
    return spans


def _bed_note(pitch: int, *, tick: int = 0, duration: int = 1920) -> NoteEvent:
    return NoteEvent(
        voice_id=VOICE_HARMONY,
        pitch_midi=pitch,
        tick=tick,
        duration_ticks=duration,
        velocity=46,
    )


def _melody_note(pitch: int, *, tick: int = 0, duration: int = 1920) -> NoteEvent:
    return NoteEvent(
        voice_id=VOICE_MELODY,
        pitch_midi=pitch,
        tick=tick,
        duration_ticks=duration,
        velocity=70,
    )


class TestHarmonyRegisterPass:
    """The bed is settled into its instrument's window, clear of the tune."""

    # A pad instrument whose window is the old fixed 48-84, so the
    # cases below read as the behaviour this pass always had.
    _WINDOW = MelodyBand(low_midi=48, high_midi=84)

    def _settle(
        self,
        notes: list[NoteEvent],
        *,
        window: MelodyBand | None = None,
        compass: MelodyBand | None = None,
        melody_floor: int,
        melody_ceiling: int | None = None,
    ) -> list[NoteEvent]:
        """Settle one voice, with the comfortable window on both sides.

        The default compass *is* the comfortable window, so the cases
        below read as the behaviour the pass always had: a bed with one
        window and no fallback behind it. `compass` is passed only by the
        cases that are about the fallback.
        """
        comfortable = window or self._WINDOW
        return _settle_harmony_register(
            notes,
            melody_floor=melody_floor,
            melody_ceiling=melody_floor if melody_ceiling is None else melody_ceiling,
            registers={
                VOICE_HARMONY: BedRegisters(
                    comfortable=comfortable,
                    compass=compass or comfortable,
                )
            },
        )

    def test_a_fold_lands_on_the_octave_the_window_holds(self) -> None:
        # 76 is already in the window; 100 folds down to 76, and 40 folds
        # up to 52 — the fold moves the way it has to, which is what the
        # fixed 48-84 fold did when the window was those two constants.
        assert _octaves_in_window(76, self._WINDOW) == [52, 64, 76]
        assert _into_harmony_register(76, window=self._WINDOW) == 76
        assert _into_harmony_register(100, window=self._WINDOW) == 76
        assert _into_harmony_register(40, window=self._WINDOW) == 52

    def test_a_fold_never_leaves_the_window(self) -> None:
        for pitch in range(0, 128):
            folded = _into_harmony_register(pitch, window=self._WINDOW)
            assert self._WINDOW.contains(folded), (pitch, folded)

    def test_a_window_narrower_than_an_octave_may_miss_a_pitch_class(self) -> None:
        """An empty octave list is the honest answer for a window that has
        no octave of the pitch in it — the caller drops the note rather
        than the fold inventing one the instrument cannot sound."""
        narrow = MelodyBand(low_midi=60, high_midi=66)
        assert _octaves_in_window(61, narrow) == [61]
        # G: every octave of it is outside a window of C..F#.
        assert _octaves_in_window(67, narrow) == []
        assert not narrow.contains(_into_harmony_register(67, window=narrow))

    def test_a_note_clear_of_the_tune_is_left_alone(self) -> None:
        note = _bed_note(60)
        assert self._settle([note], melody_floor=64) == [note]

    def test_a_note_inside_the_melody_band_folds_under_it(self) -> None:
        # The melody's floor is 64, so the ceiling is 61 and the bed takes
        # 76 down to 64 — which is still over it — and on to 52.
        settled = self._settle([_bed_note(76), _melody_note(64)], melody_floor=64)
        assert [n.pitch_midi for n in settled if n.voice_id == VOICE_HARMONY] == [52]

    def test_a_fold_that_would_crowd_the_melody_takes_the_next_octave(self) -> None:
        """A note that moves is asked about the melody again.

        Where the note was, an octave above the melody, it never met it.
        Folded to 61 it is a major seventh under a melody note at 72 —
        a rubbed interval, not a cleared one — so the bed takes the
        octave below that instead of arriving in the tune's face.
        """
        settled = self._settle([_bed_note(73), _melody_note(72)], melody_floor=64)
        assert [n.pitch_midi for n in settled if n.voice_id == VOICE_HARMONY] == [49]

    def test_a_note_with_no_clear_octave_is_dropped(self) -> None:
        """Both homes refused, so the note goes.

        A B above the ceiling has only 59 under it, and a melody note at
        70 leaves that a minor seventh — the pitch class of a whole tone
        — from the tune. The tune's ceiling of 84 also leaves the window
        (48-84) no octave above it to move to, so there is nowhere for
        the note to go and it is dropped rather than written in the
        tune's face.
        """
        settled = self._settle(
            [_bed_note(71), _melody_note(70)], melody_floor=64, melody_ceiling=84
        )
        assert not [n for n in settled if n.voice_id == VOICE_HARMONY]

    def test_a_voice_that_goes_under_does_not_leap_above_the_tune(self) -> None:
        """The side is the voice's decision, so one note cannot reverse it.

        The note below the tune is crowded in every octave the window
        holds, and the window has a whole octave clear underneath it. The
        voice's side is therefore *under*, and a note with no clearing
        octave there is dropped rather than sent over the tune. Deciding
        the side per note is what put a bed's notes on both sides of the
        tune — the accompaniment straddling the melody it is supposed to
        leave alone, which is the register breach this pass exists to
        prevent.
        """
        settled = self._settle([_bed_note(71), _melody_note(70)], melody_floor=64)
        assert not [n for n in settled if n.voice_id == VOICE_HARMONY]

    def test_a_bar_whose_bed_still_has_a_home_keeps_sounding(self) -> None:
        settled = self._settle(
            [_bed_note(71), _bed_note(61), _melody_note(70)],
            melody_floor=64,
            melody_ceiling=84,
        )
        assert [n.pitch_midi for n in settled if n.voice_id == VOICE_HARMONY] == [61]

    def test_the_same_note_folds_the_same_way_wherever_it_sounds(self) -> None:
        """The bounds are the piece's, so the fold does not move per bar.

        A bar-local bound would fold this note to 61 in a bar whose tune
        sat high and to 52 in one where it descended — the pad leaping an
        octave between two bars of a held chord.
        """
        settled = self._settle(
            [_bed_note(64, tick=0), _bed_note(64, tick=1920), _melody_note(64, duration=3840)],
            melody_floor=64,
        )
        assert [n.pitch_midi for n in settled if n.voice_id == VOICE_HARMONY] == [52, 52]

    def test_a_pad_that_cannot_sit_under_the_tune_goes_above_it(self) -> None:
        """A celesta's floor is C5, so a piano tune descending to C3 leaves
        it nothing underneath — and a celesta above a low tune is where
        the instrument belongs anyway. Before the window was the
        instrument's, this pad was written 48-59, a twelfth below the
        celesta's lowest note, and the range gate was the piano's so it
        could not see it.
        """
        window = bed_window(Instrument.CELESTA.value)
        assert window.low_midi > 45, "the celesta has no register under a C3 tune"
        settled = self._settle(
            [_bed_note(window.low_midi), _melody_note(48), _melody_note(84)],
            window=window,
            melody_floor=48,
            melody_ceiling=84,
        )
        pads = [n.pitch_midi for n in settled if n.voice_id == VOICE_HARMONY]
        # The lowest octave of C5 that clears the tune's top: C7, which
        # is the top of the celesta's own window.
        assert pads == [96], pads
        assert window.contains(pads[0])

    def test_a_pad_goes_under_the_tune_when_the_window_reaches(self) -> None:
        """The preference is below, wherever below has the room.

        The same tune, and a window with a whole octave of clear compass
        under the tune's floor and less than that above its ceiling: the
        bed lands under the tune rather than over it. Under is the
        preference and not the rule — it wins whenever it has an octave —
        because a pad under a tune is the sound of accompaniment, and one
        written above is a descant.
        """
        window = MelodyBand(low_midi=30, high_midi=96)
        settled = self._settle(
            [_bed_note(40), _melody_note(48), _melody_note(84)],
            window=window,
            melody_floor=48,
            melody_ceiling=84,
        )
        pads = [n.pitch_midi for n in settled if n.voice_id == VOICE_HARMONY]
        assert pads == [40], pads
        assert window.contains(pads[0])

    def test_a_pad_the_comfortable_window_cannot_place_falls_back_to_the_compass(
        self,
    ) -> None:
        """A celesta under a piccolo tune has nowhere comfortable to be.

        The celesta's comfortable range starts at C5 and the piccolo's
        band runs 79-100, so the comfortable window clears the tune's
        floor by four semitones — under the six the pass asks for — and
        reaches nothing above its ceiling. With only that window behind
        it, a note it cannot hold clear is dropped: here eleven semitones
        above the tune's floor — a major seventh from it — with no octave
        of it the window can write clear. The compass is a celesta's whole
        sixty semitones, and a pad an octave below the tune is worth more
        than a lost note. So the pass falls back to it, which is the
        difference this test measures: the same note, the same tune, and
        the only change is whether the compass is behind the comfortable
        window.
        """
        comfortable = bed_window(Instrument.CELESTA.value)
        note = _bed_note(90)
        tune = [_melody_note(79), _melody_note(100)]

        without = self._settle(
            [note, *tune],
            window=comfortable,
            melody_floor=79,
            melody_ceiling=100,
        )
        assert not [n for n in without if n.voice_id == VOICE_HARMONY], (
            "with no compass behind it the pass has no home for the note"
        )

        compass = range_for(Instrument.CELESTA.value)
        settled = self._settle(
            [note, *tune],
            window=comfortable,
            compass=MelodyBand(compass.low_midi, compass.high_midi),
            melody_floor=79,
            melody_ceiling=100,
        )
        pads = [n.pitch_midi for n in settled if n.voice_id == VOICE_HARMONY]
        # Thirteen semitones under the tune's floor — an octave and a
        # semitone, so clear of it — inside the celesta's compass and
        # below its comfortable range: a register only the fallback can
        # write, since the comfortable window is the whole of 72-96.
        assert pads == [66], pads
        assert compass.contains(pads[0])
        assert pads[0] < comfortable.low_midi

    def test_a_voice_no_window_can_place_still_loses_the_notes_that_clash(
        self,
    ) -> None:
        """Keeping the register is not keeping the clash.

        A window thirty-one semitones wide under a tune that fills
        twenty-three of them leaves neither side the six semitones the
        pass asks for — three under the tune's floor and none above its
        ceiling — so the voice is kept where the harmony pass wrote it,
        a bed in the tune's register being worth more than no bed. What
        survives that is only the notes that do not rub the tune: a clash
        is not a register, and the pass drops it either way.
        """
        window = MelodyBand(low_midi=60, high_midi=91)
        clashing = _bed_note(77)  # eleven semitones from either tune note
        clear = _bed_note(74)
        settled = self._settle(
            [clashing, clear, _melody_note(66), _melody_note(88)],
            window=window,
            melody_floor=66,
            melody_ceiling=88,
        )
        assert [n.pitch_midi for n in settled if n.voice_id == VOICE_HARMONY] == [74]

    def test_every_pad_note_of_a_piece_is_in_its_own_instrument(self) -> None:
        """The property, over the whole palette: every voice is written
        where the instrument that plays it can play.

        Three claims, in descending strength. The compass — `range_for`,
        which the linter enforces — holds for every note of every voice.
        An accompaniment is written inside `bed_window`, its instrument's
        whole comfortable range, whenever that window has room to clear
        the tune — and inside the compass when it does not, which is the
        fallback a narrow instrument needs. That condition is not a
        detail of this test: it is the same `_can_clear_the_tune` the
        pass itself decides with, so a bed that has moved outside
        `bed_window` for any other reason fails here. And every melody
        note is inside the window the engine actually placed it in:
        `melody_band` for a melody on its own, or the one
        `_melody_band_for` raises when the accompaniment needs an octave
        under the tune — which is why this asserts against that function
        rather than the instrument's own band.

        This is the defect the whole table exists to end: a tuba's tune
        written two octaves above a tuba, and a byte-identical score for
        every instrument in the palette.
        """
        for mood in Mood:
            for instrument in Instrument:
                out = compose(
                    CompositionSpec(
                        mood=mood,
                        instrumentation=instrument,
                        duration_seconds=60,
                        seed=42,
                    )
                )
                instruments = {v.voice_id: v.instrument for v in out.voice_instruments}
                melody_band_here = _melody_band_for(
                    melody=instruments[VOICE_MELODY],
                    bed=instruments.get(VOICE_HARMONY),
                )
                melody = [
                    n for n in out.notation_score.notes if n.voice_id == VOICE_MELODY
                ]
                melody_pitches = [n.pitch_midi for n in melody]
                below_ceiling = min(melody_pitches) - HARMONY_MELODY_CLEARANCE
                above_floor = max(melody_pitches) + HARMONY_MELODY_CLEARANCE
                for voice_id, name in instruments.items():
                    span = range_for(name)
                    notes = [n for n in out.notation_score.notes if n.voice_id == voice_id]
                    assert notes
                    pitches = sorted({n.pitch_midi for n in notes})
                    assert all(span.contains(n.pitch_midi) for n in notes), (
                        mood,
                        name,
                        pitches,
                    )
                    if voice_id >= VOICE_HARMONY:
                        comfortable = bed_window(name)
                        above = _bed_goes_above(
                            comfortable,
                            below_ceiling=below_ceiling,
                            above_floor=above_floor,
                        )
                        if _can_clear_the_tune(
                            comfortable,
                            below_ceiling=below_ceiling,
                            above_floor=above_floor,
                            above=above,
                        ):
                            assert all(comfortable.contains(n.pitch_midi) for n in notes), (
                                mood,
                                name,
                                comfortable,
                                pitches,
                            )
                    if voice_id == VOICE_MELODY:
                        assert all(
                            melody_band_here.contains(n.pitch_midi) for n in notes
                        ), (mood, name, melody_band_here, pitches)
                        assert max(pitches) - min(pitches) <= LINE_BAND_SEMITONES, (
                            mood,
                            name,
                            min(pitches),
                            max(pitches),
                        )



class TestHarmonyVoice:
    """P2: the harmony voice — pad for calm moods, arpeggio for energy."""

    def test_harmony_voice_present_and_sidecar_named(self) -> None:
        out = compose(_spec(Mood.CALMING, duration=60))
        harmony = [n for n in out.notation_score.notes if n.voice_id == 3]
        assert harmony, "the default ensemble carries a harmony voice"
        sidecar = {v.voice_id: v.instrument for v in out.voice_instruments}
        assert sidecar == {0: "cello", 1: "piano", 3: "pizzicato_strings"}

    def test_two_requested_harmonies_both_reach_the_score_and_sidecar(self) -> None:
        out = compose(
            _spec(
                Mood.ELECTRIFYING,
                duration=300,
                instrumentation=[
                    {"role": "melody", "instrument": "trumpet"},
                    {"role": "harmony", "instrument": "brass_section"},
                    {"role": "harmony", "instrument": "strings"},
                    {"role": "bass", "instrument": "contrabass"},
                    {"role": "percussion", "instrument": "drum_set"},
                ],
            )
        )
        sidecar = {v.voice_id: v.instrument for v in out.voice_instruments}
        assert sidecar == {
            0: "contrabass",
            1: "trumpet",
            2: "drum_set",
            3: "brass_section",
            4: "strings",
        }
        voices = {n.voice_id for n in out.notation_score.notes}
        assert voices == {0, 1, 2, 3, 4}
        # Brass punctuates; the secondary strings form the long bed.
        assert {n.duration_ticks for n in out.notation_score.notes if n.voice_id == 3} == {
            out.notation_score.ppq
        }
        assert {n.duration_ticks for n in out.notation_score.notes if n.voice_id == 4} == {
            bar_ticks(out.time_signature)
        }

    def test_drum_set_piece_has_no_harmony_voice(self) -> None:
        out = compose(
            _spec(Mood.ELECTRIFYING, duration=30, instrumentation="drum_set")
        )
        voices = {n.voice_id for n in out.notation_score.notes}
        assert 3 not in voices
        assert voices == {0, 1, 2}

    @pytest.mark.parametrize("mood", [Mood.CALMING, Mood.SLEEP])
    def test_calm_moods_get_a_sustained_pad(self, mood: Mood) -> None:
        out = compose(_spec(mood, duration=60))
        harmony = [n for n in out.notation_score.notes if n.voice_id == 3]
        ticks_per_bar = bar_ticks(out.time_signature)
        # The pad holds each note across a bar, quietly.
        assert all(n.duration_ticks == ticks_per_bar for n in harmony)
        assert all(30 <= n.velocity <= 60 for n in harmony)

    def test_electrifying_gets_a_broken_chord_arpeggio(self) -> None:
        out = compose(_spec(Mood.ELECTRIFYING, duration=60))
        harmony = [n for n in out.notation_score.notes if n.voice_id == 3]
        eighth = out.notation_score.ppq // 2
        assert all(n.duration_ticks == eighth for n in harmony)
        assert len(harmony) > 4 * out.arrangement.total_bars, (
            "the arpeggio moves faster than one note per beat"
        )

    def test_harmony_stays_in_its_register(self) -> None:
        """The register is the instrument's, not a module constant.

        The bed is written across the instrument's whole comfortable
        range — `bed_window` — so a sleep piece's celesta pad sits at
        72-96 and a calming piece's pizzicato strings at 48-84. The one
        48-84 window this test used to assert was the piano's comfort
        applied to every accompaniment voice in the product.
        """
        for mood in (Mood.CALMING, Mood.ELECTRIFYING, Mood.SLEEP):
            out = compose(_spec(mood, duration=60))
            instruments = {v.voice_id: v.instrument for v in out.voice_instruments}
            harmony = [n for n in out.notation_score.notes if n.voice_id == VOICE_HARMONY]
            assert harmony
            window = bed_window(instruments[VOICE_HARMONY])
            assert all(window.contains(n.pitch_midi) for n in harmony), (
                instruments[VOICE_HARMONY],
                window,
                sorted({n.pitch_midi for n in harmony}),
            )
            assert all(range_for(instruments[VOICE_HARMONY]).contains(n.pitch_midi)
                       for n in harmony)

    def test_harmony_never_rubs_against_the_melody(self) -> None:
        for mood in (Mood.CALMING, Mood.ELECTRIFYING, Mood.SLEEP):
            out = compose(_spec(mood, duration=60))
            melody = [n for n in out.notation_score.notes if n.voice_id == 1]
            harmony = [n for n in out.notation_score.notes if n.voice_id == 3]
            for h in harmony:
                for m in melody:
                    if m.tick < h.tick + h.duration_ticks and h.tick < m.tick + m.duration_ticks:
                        assert abs(m.pitch_midi - h.pitch_midi) not in (0, 1, 2, 10, 11), (
                            f"harmony {h.pitch_midi} crowds melody {m.pitch_midi} ({mood})"
                        )

    def test_the_bed_sits_under_the_melody_for_the_piece(self) -> None:
        """The bed's register is the tune's floor, held for the whole piece.

        Every bed note is at least `HARMONY_MELODY_CLEARANCE` clear of the
        tune's range — under its floor, or over its ceiling when the
        instrument's window has nothing clear underneath, which is the
        celesta's case. Either way the two ranges are disjoint, which is
        the register half of "not a hot pot of instruments": the pad has a
        register of its own instead of sharing the tune's. Measured before
        this pass over the gate matrix, the bed's top sat at 84 against a
        tune whose floor was 64 — twenty-one semitones of shared band, and
        19.8 by the metric against a bar of 4.

        Both register metrics now read ranges rather than means, so they
        are the assertion: `register_separation_semitones` is the distance
        between the nearest accompaniment voice and the tune, and
        `tessitura_overlap_semitones` is how many semitones of the tune's
        own range the harmony sounds in.
        """
        for mood in (Mood.CALMING, Mood.ELECTRIFYING, Mood.SLEEP):
            out = compose(_spec(mood, duration=60))
            notes = out.notation_score.notes
            melody = [n.pitch_midi for n in notes if n.voice_id == VOICE_MELODY]
            harmony = [n.pitch_midi for n in notes if n.voice_id == VOICE_HARMONY]
            assert harmony
            assert melody
            assert all(
                pitch <= min(melody) - HARMONY_MELODY_CLEARANCE
                or pitch >= max(melody) + HARMONY_MELODY_CLEARANCE
                for pitch in harmony
            ), (mood, min(melody), max(melody), sorted(set(harmony)))
            report = score_piece(out.notation_score, piece=f"{mood.value}-under-the-tune")
            assert report.tessitura_overlap_semitones == 0, (mood, report)
            assert report.register_separation_semitones >= HARMONY_MELODY_CLEARANCE, (
                mood,
                report,
            )

    def test_a_cleared_bed_keeps_a_note_in_every_bar_it_opens(self) -> None:
        """The bed thins; it does not vanish.

        A bed note the melody leaves no legal octave for is dropped, so a
        bar whose bed had a single note can lose it. That must not become
        the texture: a pad that stops sounding for a bar mid-piece is a
        hole, not a rest, and the bar's melody and bass do not fill it.

        Measured over the release gate matrix, the pass leaves 307 of the
        334 bed bars untouched, thins 23 and empties 4 — all four in the
        one 600-second sleep piece, each a bar whose single bed note has
        no octave clear of a melody sitting at the foot of its band. That
        is why the assertion below is scoped to the 60-second pieces: it
        is the floor for the pieces it composes, and the long piece's
        thinning is the crowding rule's verdict on a bed that was already
        one note thin, not this pass inventing a hole.
        """
        for mood in (Mood.CALMING, Mood.ELECTRIFYING, Mood.SLEEP):
            out = compose(_spec(mood, duration=60))
            notes = out.notation_score.notes
            ticks = bar_ticks(out.time_signature)
            bed_bars = {n.tick // ticks for n in notes if n.voice_id == VOICE_HARMONY}
            assert bed_bars
            # The intro bars are the one place the bed is meant to be
            # silent: it enters with the melody.
            assert min(bed_bars) >= out.arrangement.intro_bars
            assert bed_bars == set(range(min(bed_bars), out.arrangement.total_bars_with_coda)), mood

    def test_harmony_enters_with_the_melody(self) -> None:
        # A long calming piece opens bass alone; the pad waits for it.
        out = compose(_spec(Mood.CALMING, duration=600))
        assert out.arrangement.intro_bars > 0
        intro_end = out.arrangement.intro_bars * bar_ticks(out.time_signature)
        harmony = [n for n in out.notation_score.notes if n.voice_id == 3]
        assert all(n.tick >= intro_end for n in harmony)

    def test_harmony_cc11_only_when_sustained(self) -> None:
        # Calming's pad is pizzicato strings — plucked, no sustain to shape.
        out = compose(_spec(Mood.CALMING, duration=60))
        cc11_voices = {c.voice_id for c in out.performance_plan.controllers if c.control == 11}
        assert cc11_voices == {1}
        # Electrifying's arpeggio rides sustained strings: it swells too.
        out = compose(_spec(Mood.ELECTRIFYING, duration=60))
        cc11_voices = {c.voice_id for c in out.performance_plan.controllers if c.control == 11}
        assert cc11_voices == {1, 3}

    def test_harmony_gets_pedal_when_its_instrument_reads_one(self) -> None:
        # A celesta pad (sleep's scalar coercion) reads a pedal.
        out = compose(_spec(Mood.SLEEP, duration=60))
        cc64_voices = {c.voice_id for c in out.performance_plan.controllers if c.control == 64}
        assert cc64_voices == {1, 3}

    def test_only_the_melody_moved_in_the_drum_set_plan(self) -> None:
        """Per-voice pins: the kit and the bass are byte-identical to the
        layout they had before the instrument table, and the melody is not.

        The SHA-256 of each voice's canonical notes pins that voice down
        to the note: no melody legato or pedal, kit humanization only,
        exactly as the scalar-drum spec played before the ensemble landed.
        A whole-plan hash would cover all of that too but could not say
        *which* voice moved, and the instrument table is a change to the
        melody's register alone — so the pin is per voice, and a future
        accidental re-timing names its voice in the failure.

        The kit and the bass read *identically* to the pre-table plan:
        the kit's 207 percussion events and the bass's 46 notes are
        unchanged, as are the melody's 84 onsets, durations and
        velocities. What moved is the melody's pitch — from the fixed
        64-84 window every instrument used to share to the piano's own
        band, 60-77 — which is the whole point of the table and the one
        voice this pin is re-based for. It has been re-based three times
        now, twice before this: once when the melody walk was rewritten
        to move in scale degrees, and once when the bass replaced its
        one hard-coded figure with the figure library. Both readings were
        checked voice by voice the same way.
        """
        out = compose(
            _spec(Mood.ELECTRIFYING, duration=30, instrumentation="drum_set")
        )
        plan = out.performance_plan
        pinned = {
            VOICE_BASS: (
                "3759d87e70d0d346e0ffb4088a1ddfbd5598ff7595f927c6a5412281504cf4dc",
                46,
            ),
            VOICE_MELODY: (
                "b89953fc0a976956f0868113dde9ea2ea03b9e0e7db71f467dc3caaa17294def",
                84,
            ),
            VOICE_PERCUSSION: (
                "f1a2f2631018d91de93e382997e23d4f55f422b88638a3230be978df674840fb",
                207,
            ),
        }
        for voice_id, (digest, count) in pinned.items():
            events = [asdict(n) for n in plan.notes if n.voice_id == voice_id]
            assert len(events) == count, (voice_id, len(events))
            payload = json.dumps(events, sort_keys=True)
            assert hashlib.sha256(payload.encode()).hexdigest() == digest, voice_id

    def test_sidecar_round_trips_voice_instruments(self) -> None:
        out = compose(_spec(Mood.ELECTRIFYING, duration=60, instrumentation="flute"))
        payload = out.to_sidecar()
        restored = EngineOutput.from_sidecar(payload)
        assert restored.voice_instruments == out.voice_instruments
        assert {v.voice_id: v.instrument for v in restored.voice_instruments} == {
            0: "contrabass",
            1: "flute",
            3: "strings",
        }

    def test_from_sidecar_tolerates_missing_voice_instruments(self) -> None:
        out = compose(_spec(Mood.CALMING, duration=60))
        payload = out.to_sidecar()
        del payload["voice_instruments"]
        restored = EngineOutput.from_sidecar(payload)
        assert restored.voice_instruments == ()
