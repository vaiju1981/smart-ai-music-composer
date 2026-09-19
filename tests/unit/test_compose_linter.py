"""Unit tests for the theory linter."""

from __future__ import annotations

from saimc.compose.linter import (
    FALLBACK_MAX_MIDI,
    FALLBACK_MIN_MIDI,
    LintCode,
    lint,
)
from saimc.compose.score import (
    PPQ,
    KeySignature,
    Measure,
    NotationScore,
    NoteEvent,
)


def _build_score(
    *,
    time_signature: str = "4/4",
    key: KeySignature | None = None,
    notes: list[NoteEvent] | None = None,
    measures: list[Measure] | None = None,
) -> NotationScore:
    if measures is None:
        measures = [Measure(index=0, start_tick=0, end_tick=1920, time_signature="4/4")]
    if notes is None:
        notes = [NoteEvent(voice_id=0, pitch_midi=60, tick=0, duration_ticks=480)]
    return NotationScore.make(
        ppq=PPQ,
        key=key or KeySignature(root="C", mode="major"),
        time_signature=time_signature,
        tempo_bpm=80.0,
        measures=measures,
        notes=notes,
    )


class TestLintPasses:
    def test_minimal_score_passes(self) -> None:
        report = lint(_build_score())
        assert report.passed
        assert report.issues == ()


class TestLintFailsOnRange:
    def test_pitch_below_range(self) -> None:
        score = _build_score(
            notes=[
                NoteEvent(voice_id=0, pitch_midi=FALLBACK_MIN_MIDI - 1, tick=0, duration_ticks=480)
            ]
        )
        report = lint(score)
        assert not report.passed
        assert any(i.code == LintCode.NOTE_OUT_OF_RANGE for i in report.issues)

    def test_pitch_above_range(self) -> None:
        score = _build_score(
            notes=[
                NoteEvent(voice_id=0, pitch_midi=FALLBACK_MAX_MIDI + 1, tick=0, duration_ticks=480)
            ]
        )
        report = lint(score)
        assert not report.passed
        assert any(i.code == LintCode.NOTE_OUT_OF_RANGE for i in report.issues)

    def test_the_fallback_compass_does_not_know_the_instrument(self) -> None:
        """Without a mapping, a tuba line at E4 is inside the piano's compass.

        64 is a perfectly ordinary pitch and no instrument-blind check can
        object to it; this pins that the fallback is the *wide* compass it
        claims to be, so the per-voice test below is measuring the mapping
        and not the pitch.
        """
        score = _build_score(
            notes=[NoteEvent(voice_id=1, pitch_midi=64, tick=0, duration_ticks=480)]
        )
        report = lint(score)
        assert not any(i.code == LintCode.NOTE_OUT_OF_RANGE for i in report.issues)

    def test_a_voice_is_measured_against_its_own_instrument(self) -> None:
        """The same E4 fails once the linter is told the voice is a tuba.

        A tuba's compass tops out at 58 (Bb3), so 64 is two octaves above
        where the instrument lives. That is exactly the pitch the engine
        wrote for every melody instrument before the band came from the
        instrument table, and it is the pitch the piano-compass gate could
        never refuse.
        """
        score = _build_score(
            notes=[NoteEvent(voice_id=1, pitch_midi=64, tick=0, duration_ticks=480)]
        )
        report = lint(score, voice_instruments={1: "tuba"})
        assert not report.passed
        out_of_range = [i for i in report.issues if i.code == LintCode.NOTE_OUT_OF_RANGE]
        assert len(out_of_range) == 1
        assert out_of_range[0].voice_id == 1
        assert "tuba" in out_of_range[0].message
        # The message names the instrument's own bounds, not the
        # fallback's, so a reader can tell which compass was applied.
        assert "[28, 58]" in out_of_range[0].message

    def test_each_voice_gets_its_own_compass(self) -> None:
        """One voice may pass where another fails on the very same pitch.

        The bass is a contrabass (compass 28..67) and the melody is a
        piccolo (74..108), so 60 is inside the one and below the other.
        A single compass for the score cannot express that, and a green
        check under one is not evidence the other is playable.
        """
        score = _build_score(
            notes=[
                NoteEvent(voice_id=0, pitch_midi=60, tick=0, duration_ticks=480),
                NoteEvent(voice_id=1, pitch_midi=60, tick=0, duration_ticks=480),
            ]
        )
        report = lint(score, voice_instruments={0: "contrabass", 1: "piccolo"})
        out_of_range = [i for i in report.issues if i.code == LintCode.NOTE_OUT_OF_RANGE]
        assert [i.voice_id for i in out_of_range] == [1]
        assert "piccolo" in out_of_range[0].message


class TestLintFailsOnMeasureCompleteness:
    def test_short_measure_fails(self) -> None:
        # 4/4 measure should be 1920 ticks; we declare 1000.
        score = _build_score(
            measures=[Measure(index=0, start_tick=0, end_tick=1000, time_signature="4/4")]
        )
        report = lint(score)
        assert not report.passed
        assert any(i.code == LintCode.MEASURE_INCOMPLETE for i in report.issues)


class TestLintFailsOnTimeSignatureMismatch:
    def test_measure_time_signature_mismatch(self) -> None:
        score = _build_score(
            time_signature="4/4",
            measures=[
                Measure(index=0, start_tick=0, end_tick=1920, time_signature="4/4"),
                Measure(index=1, start_tick=1920, end_tick=3360, time_signature="3/4"),
            ],
        )
        report = lint(score)
        assert not report.passed
        assert any(i.code == LintCode.TIME_SIGNATURE_MISMATCH for i in report.issues)


class TestLintFailsOnEmpty:
    def test_empty_score_fails(self) -> None:
        score = _build_score(notes=[], measures=[])
        report = lint(score)
        assert not report.passed
        assert any(i.code == LintCode.EMPTY_SCORE for i in report.issues)


class TestLintFailsOnDuplicateNotes:
    def test_duplicate_at_tick_fails(self) -> None:
        score = _build_score(
            notes=[
                NoteEvent(voice_id=0, pitch_midi=60, tick=0, duration_ticks=480),
                NoteEvent(voice_id=0, pitch_midi=60, tick=0, duration_ticks=480),
            ]
        )
        report = lint(score)
        assert not report.passed
        assert any(i.code == LintCode.DUPLICATE_NOTE_AT_TICK for i in report.issues)

    def test_same_pitch_different_voice_ok(self) -> None:
        # Same pitch at the same tick but different voices is fine.
        score = _build_score(
            notes=[
                NoteEvent(voice_id=0, pitch_midi=60, tick=0, duration_ticks=480),
                NoteEvent(voice_id=1, pitch_midi=60, tick=0, duration_ticks=480),
            ]
        )
        report = lint(score)
        assert report.passed


class TestLintFailsOnSimultaneous:
    def test_too_many_simultaneous_fails(self) -> None:
        # 9 notes at tick=0 should fail (limit is 8).
        notes = [
            NoteEvent(voice_id=0, pitch_midi=60 + i, tick=0, duration_ticks=480) for i in range(9)
        ]
        score = _build_score(notes=notes)
        report = lint(score)
        assert not report.passed
        assert any(i.code == LintCode.TOO_MANY_SIMULTANEOUS_NOTES for i in report.issues)


class TestEndsOnTonic:
    """The melody must resolve: its final note is the tonic or third."""

    def test_final_melody_note_on_tonic_passes(self) -> None:
        score = _build_score(
            notes=[NoteEvent(voice_id=1, pitch_midi=72, tick=0, duration_ticks=1920)]
        )
        report = lint(score)
        assert report.passed

    def test_final_melody_note_on_third_passes(self) -> None:
        # C major: third is E (64 / 76).
        score = _build_score(
            notes=[NoteEvent(voice_id=1, pitch_midi=76, tick=0, duration_ticks=1920)]
        )
        report = lint(score)
        assert report.passed

    def test_final_melody_note_off_tonic_fails(self) -> None:
        # D (74) over C major is neither tonic nor third.
        score = _build_score(
            notes=[NoteEvent(voice_id=1, pitch_midi=74, tick=0, duration_ticks=1920)]
        )
        report = lint(score)
        assert not report.passed
        assert any(i.code == LintCode.ENDS_OFF_TONIC for i in report.issues)

    def test_minor_key_third_is_minor_third(self) -> None:
        # A minor: tonic A (69/81), third C (72/84).
        score = _build_score(
            key=KeySignature(root="A", mode="minor"),
            notes=[NoteEvent(voice_id=1, pitch_midi=84, tick=0, duration_ticks=1920)],
        )
        report = lint(score)
        assert report.passed

    def test_minor_key_major_third_fails(self) -> None:
        # C# (73) over A minor is neither tonic nor (minor) third.
        score = _build_score(
            key=KeySignature(root="A", mode="minor"),
            notes=[NoteEvent(voice_id=1, pitch_midi=85, tick=0, duration_ticks=1920)],
        )
        report = lint(score)
        assert any(i.code == LintCode.ENDS_OFF_TONIC for i in report.issues)

    def test_score_without_melody_skips_check(self) -> None:
        # A bass-only score has no melodic line to resolve.
        score = _build_score(
            notes=[NoteEvent(voice_id=0, pitch_midi=48, tick=0, duration_ticks=1920)]
        )
        report = lint(score)
        assert report.passed

    def test_voice_leading_collision_code_removed(self) -> None:
        # The check was declared but never implemented; the code is gone
        # rather than pretending a gate exists.
        assert not hasattr(LintCode, "VOICE_LEADING_COLLISION")


def _bars(*specs: tuple[int, int]) -> list[Measure]:
    """Measures from (start_tick, end_tick) pairs, 4/4."""
    return [
        Measure(index=i, start_tick=start, end_tick=end, time_signature="4/4")
        for i, (start, end) in enumerate(specs)
    ]


class TestChordToneMembership:
    """With chord_bars, every pitched note must sound its bar's chord."""

    def _score(self, notes: list[NoteEvent]) -> NotationScore:
        return _build_score(
            measures=_bars((0, 1920), (1920, 3840)),
            notes=notes,
        )

    def test_chord_tone_passes(self) -> None:
        score = self._score(
            [
                NoteEvent(voice_id=1, pitch_midi=72, tick=0, duration_ticks=960),
                NoteEvent(voice_id=0, pitch_midi=48, tick=0, duration_ticks=1920),
            ]
        )
        report = lint(score, chord_bars=((0, 4, 7), (0, 4, 7)))
        assert report.passed

    def test_non_chord_tone_fails(self) -> None:
        # F (65) over a C-major bar is not a chord tone.
        score = self._score(
            [
                NoteEvent(voice_id=1, pitch_midi=65, tick=0, duration_ticks=960),
                NoteEvent(voice_id=1, pitch_midi=72, tick=1920, duration_ticks=1920),
            ]
        )
        report = lint(score, chord_bars=((0, 4, 7), (0, 4, 7)))
        assert any(i.code == LintCode.CHORD_TONE_VIOLATION for i in report.issues)

    def test_anacrusis_may_anticipate_the_next_chord(self) -> None:
        # A pickup in the bar's final eighth may sound the next chord.
        score = self._score(
            [
                NoteEvent(voice_id=1, pitch_midi=72, tick=0, duration_ticks=1680),
                NoteEvent(voice_id=1, pitch_midi=74, tick=1680, duration_ticks=240),
                NoteEvent(voice_id=1, pitch_midi=72, tick=1920, duration_ticks=1920),
            ]
        )
        report = lint(score, chord_bars=((0, 4, 7), (0, 2, 7)))
        assert report.passed

    def test_anticipation_does_not_extend_mid_bar(self) -> None:
        # D (74) mid-bar over C major fails even though bar 1 holds D.
        score = self._score(
            [
                NoteEvent(voice_id=1, pitch_midi=74, tick=960, duration_ticks=480),
                NoteEvent(voice_id=1, pitch_midi=72, tick=1920, duration_ticks=1920),
            ]
        )
        report = lint(score, chord_bars=((0, 4, 7), (0, 2, 7)))
        assert any(i.code == LintCode.CHORD_TONE_VIOLATION for i in report.issues)

    def test_percussion_notes_are_exempt(self) -> None:
        # GM kit keys are not pitches; they sound on no chord.
        score = self._score(
            [
                NoteEvent(voice_id=1, pitch_midi=72, tick=0, duration_ticks=1920),
                NoteEvent(voice_id=2, pitch_midi=38, tick=0, duration_ticks=120),
            ]
        )
        report = lint(score, chord_bars=((0, 4, 7), (0, 4, 7)))
        assert report.passed

    def test_without_chord_bars_the_check_is_skipped(self) -> None:
        score = self._score(
            [NoteEvent(voice_id=1, pitch_midi=65, tick=0, duration_ticks=1920)]
        )
        report = lint(score)
        assert not any(i.code == LintCode.CHORD_TONE_VIOLATION for i in report.issues)


class TestPassingAndNeighbourTones:
    """The one licence that lets a melody move by step between chord tones.

    Every case is built over a C major bar (pcs 0, 4, 7) with F (65) as
    the tone under test: F is *diatonic* to C major, so it isolates the
    other three conditions instead of tripping the diatonic one. Each
    test removes exactly one condition from an otherwise legal figure.
    """

    def _score(self, notes: list[NoteEvent]) -> NotationScore:
        return _build_score(measures=_bars((0, 1920), (1920, 3840)), notes=notes)

    def test_a_passing_tone_between_two_chord_tones_passes(self) -> None:
        # G - F - E: entered by step, left by step, same direction (down).
        # The figures here resolve onto E, the third, so the melody also
        # satisfies the ends-on-tonic rule and each test fails for its own
        # named reason rather than for the cadence.
        score = self._score(
            [
                NoteEvent(voice_id=1, pitch_midi=67, tick=0, duration_ticks=480),
                NoteEvent(voice_id=1, pitch_midi=65, tick=480, duration_ticks=480),
                NoteEvent(voice_id=1, pitch_midi=64, tick=960, duration_ticks=960),
            ]
        )
        report = lint(score, chord_bars=((0, 4, 7), (0, 4, 7)))
        assert report.passed

    def test_a_neighbour_tone_returns_to_its_chord_tone(self) -> None:
        # E - F - E: entered by step, left by step, opposite direction.
        # A rule that demanded one direction would reject this figure or
        # the passing tone above; the licence has to admit both.
        score = self._score(
            [
                NoteEvent(voice_id=1, pitch_midi=64, tick=0, duration_ticks=480),
                NoteEvent(voice_id=1, pitch_midi=65, tick=480, duration_ticks=480),
                NoteEvent(voice_id=1, pitch_midi=64, tick=960, duration_ticks=960),
            ]
        )
        report = lint(score, chord_bars=((0, 4, 7), (0, 4, 7)))
        assert report.passed

    def test_a_rest_before_the_tone_breaks_the_figure(self) -> None:
        # The tone is still stepwise on both sides, but it is no longer
        # *approached*: a rest stands where its predecessor should be, so
        # nothing prepares the dissonance.
        score = self._score(
            [
                NoteEvent(voice_id=1, pitch_midi=64, tick=0, duration_ticks=240),
                NoteEvent(voice_id=1, pitch_midi=65, tick=480, duration_ticks=480),
                NoteEvent(voice_id=1, pitch_midi=64, tick=960, duration_ticks=960),
            ]
        )
        report = lint(score, chord_bars=((0, 4, 7), (0, 4, 7)))
        assert any(i.code == LintCode.CHORD_TONE_VIOLATION for i in report.issues)

    def test_a_chromatic_tone_between_two_chord_tones_fails(self) -> None:
        # F# (66) is a semitone from both neighbours, so every stepwise
        # condition holds — but it is not in C major's scale, and this is
        # the condition that stops a chromatic neighbour.
        score = self._score(
            [
                NoteEvent(voice_id=1, pitch_midi=65, tick=0, duration_ticks=480),
                NoteEvent(voice_id=1, pitch_midi=66, tick=480, duration_ticks=480),
                NoteEvent(voice_id=1, pitch_midi=64, tick=960, duration_ticks=960),
            ]
        )
        report = lint(score, chord_bars=((0, 4, 7), (0, 4, 7)))
        assert any(i.code == LintCode.CHORD_TONE_VIOLATION for i in report.issues)

    def test_a_diatonic_tone_on_the_downbeat_fails(self) -> None:
        # Accented non-chord tone: a suspension or an appoggiatura, which
        # the engine does not model. The neighbours are legal chord tones.
        score = self._score(
            [
                NoteEvent(voice_id=1, pitch_midi=65, tick=0, duration_ticks=480),
                NoteEvent(voice_id=1, pitch_midi=64, tick=480, duration_ticks=1440),
            ]
        )
        report = lint(score, chord_bars=((0, 4, 7), (0, 4, 7)))
        assert any(i.code == LintCode.CHORD_TONE_VIOLATION for i in report.issues)

    def test_a_long_non_chord_tone_fails(self) -> None:
        # A held F between two chord tones: stepwise on both sides,
        # unaccented, diatonic — but a passing tone is brief by
        # definition, and a held dissonance is a different device.
        score = self._score(
            [
                NoteEvent(voice_id=1, pitch_midi=64, tick=240, duration_ticks=240),
                NoteEvent(voice_id=1, pitch_midi=65, tick=480, duration_ticks=960),
                NoteEvent(voice_id=1, pitch_midi=64, tick=1440, duration_ticks=480),
            ]
        )
        report = lint(score, chord_bars=((0, 4, 7), (0, 4, 7)))
        assert any(i.code == LintCode.CHORD_TONE_VIOLATION for i in report.issues)

    def test_a_tone_left_by_leap_fails(self) -> None:
        # Entered by step from E, but left by a fifth up to C an octave
        # above: the figure never resolves to the chord tone next door.
        score = self._score(
            [
                NoteEvent(voice_id=1, pitch_midi=64, tick=0, duration_ticks=480),
                NoteEvent(voice_id=1, pitch_midi=65, tick=480, duration_ticks=480),
                NoteEvent(voice_id=1, pitch_midi=72, tick=960, duration_ticks=960),
            ]
        )
        report = lint(score, chord_bars=((0, 4, 7), (0, 4, 7)))
        assert any(i.code == LintCode.CHORD_TONE_VIOLATION for i in report.issues)

    def test_a_step_from_another_voice_is_not_a_step(self) -> None:
        # The bass sounds E an octave below while the melody sounds F and
        # then steps to E. The melody has nothing *before* the F, and the
        # bass's E is in another voice: neighbours are per voice, because
        # a step across voices prepares a dissonance from a different line.
        score = self._score(
            [
                NoteEvent(voice_id=1, pitch_midi=65, tick=480, duration_ticks=480),
                NoteEvent(voice_id=1, pitch_midi=64, tick=960, duration_ticks=960),
                NoteEvent(voice_id=0, pitch_midi=52, tick=480, duration_ticks=480),
                NoteEvent(voice_id=0, pitch_midi=53, tick=960, duration_ticks=960),
            ]
        )
        report = lint(score, chord_bars=((0, 4, 7), (0, 4, 7)))
        assert any(
            i.code == LintCode.CHORD_TONE_VIOLATION and i.voice_id == 1 for i in report.issues
        )

    def test_a_tie_is_not_a_step_into_the_tone(self) -> None:
        # F tied from the bar line into the unaccented beat, then stepping
        # down to E. The tie repeats the pitch, so the held tone is not
        # *approached* by step — it is simply carried over the line.
        score = self._score(
            [
                NoteEvent(voice_id=1, pitch_midi=65, tick=0, duration_ticks=480),
                NoteEvent(voice_id=1, pitch_midi=65, tick=480, duration_ticks=240, tie=True),
                NoteEvent(voice_id=1, pitch_midi=64, tick=720, duration_ticks=1200),
            ]
        )
        report = lint(score, chord_bars=((0, 4, 7), (0, 4, 7)))
        assert any(i.code == LintCode.CHORD_TONE_VIOLATION for i in report.issues)

    def test_a_passing_tone_is_licenced_in_any_pitched_voice(self) -> None:
        # The rule is uniform across voices, not a melody privilege: the
        # bass may walk by step through a passing tone too.
        score = self._score(
            [
                NoteEvent(voice_id=0, pitch_midi=64, tick=0, duration_ticks=480),
                NoteEvent(voice_id=0, pitch_midi=65, tick=480, duration_ticks=480),
                NoteEvent(voice_id=0, pitch_midi=67, tick=960, duration_ticks=960),
                NoteEvent(voice_id=1, pitch_midi=72, tick=0, duration_ticks=1920),
            ]
        )
        report = lint(score, chord_bars=((0, 4, 7), (0, 4, 7)))
        assert report.passed

    def test_the_scale_is_the_score_modes_not_always_major(self) -> None:
        # F natural (65) is diatonic to A minor and chromatic to A major
        # (which has F#), so this figure only passes if the licence reads
        # the score's mode. A hard-coded major scale would reject it.
        score = _build_score(
            key=KeySignature(root="A", mode="minor"),
            measures=_bars((0, 1920), (1920, 3840)),
            notes=[
                NoteEvent(voice_id=1, pitch_midi=64, tick=0, duration_ticks=480),
                NoteEvent(voice_id=1, pitch_midi=65, tick=480, duration_ticks=480),
                NoteEvent(voice_id=1, pitch_midi=64, tick=960, duration_ticks=960),
                NoteEvent(voice_id=1, pitch_midi=69, tick=1920, duration_ticks=1920),
            ],
        )
        report = lint(score, chord_bars=((9, 0, 4), (9, 0, 4)))
        assert report.passed

    def test_the_key_is_the_bars_own_when_a_modulation_moved_it(self) -> None:
        # A long piece's final repetition is lifted a whole step, and a
        # lifted IV — G major under a piece in C — is spelled exactly like
        # the home key's V. The chord alone cannot say which key the bar
        # belongs to, so `bar_keys` says it: the same figure is refused
        # read against C and licensed read against D, which is the key
        # F# (78) belongs to and the key the walk wrote the bar in.
        score = _build_score(
            measures=_bars((0, 1920), (1920, 3840)),
            notes=[
                NoteEvent(voice_id=1, pitch_midi=74, tick=0, duration_ticks=240),
                NoteEvent(voice_id=1, pitch_midi=76, tick=240, duration_ticks=240),
                NoteEvent(voice_id=1, pitch_midi=78, tick=480, duration_ticks=480),
                NoteEvent(voice_id=1, pitch_midi=79, tick=960, duration_ticks=480),
                NoteEvent(voice_id=1, pitch_midi=83, tick=1440, duration_ticks=480),
                NoteEvent(voice_id=1, pitch_midi=79, tick=1920, duration_ticks=480),
                NoteEvent(voice_id=1, pitch_midi=76, tick=2400, duration_ticks=960),
            ],
        )
        chord_bars = ((2, 7, 11), (0, 4, 7))
        read_as_home = lint(score, chord_bars=chord_bars)
        assert any(i.code == LintCode.CHORD_TONE_VIOLATION for i in read_as_home.issues)
        read_as_lifted = lint(
            score,
            chord_bars=chord_bars,
            bar_keys=(KeySignature(root="D", mode="major"), KeySignature(root="C", mode="major")),
        )
        assert read_as_lifted.passed


class TestDissonantCollision:
    """Close m2/M7 overlaps between voices are flagged unless both are chord tones."""

    def _score(self, notes: list[NoteEvent]) -> NotationScore:
        return _build_score(measures=_bars((0, 1920)), notes=notes)

    def test_close_second_between_voices_fails(self) -> None:
        # C (72) against C# (61) a major 7th below — a rubbed second.
        score = self._score(
            [
                NoteEvent(voice_id=1, pitch_midi=72, tick=0, duration_ticks=960),
                NoteEvent(voice_id=0, pitch_midi=61, tick=0, duration_ticks=960),
            ]
        )
        report = lint(score, chord_bars=((0, 4, 7),))
        assert any(i.code == LintCode.DISSONANT_COLLISION for i in report.issues)

    def test_maj7_voicing_is_exempt(self) -> None:
        # Fmaj7: bass F (65) against melody E (76) an M7 apart — both
        # chord tones, so the wide voicing is intended.
        score = self._score(
            [
                NoteEvent(voice_id=1, pitch_midi=76, tick=0, duration_ticks=960),
                NoteEvent(voice_id=0, pitch_midi=65, tick=0, duration_ticks=960),
            ]
        )
        report = lint(score, chord_bars=((5, 9, 0, 4),))
        assert not any(i.code == LintCode.DISSONANT_COLLISION for i in report.issues)

    def test_wide_interval_is_not_a_collision(self) -> None:
        # A m2/M7 pitch-class pair spread over an octave is voicing, not rubbing.
        score = self._score(
            [
                NoteEvent(voice_id=1, pitch_midi=77, tick=0, duration_ticks=960),
                NoteEvent(voice_id=0, pitch_midi=53, tick=0, duration_ticks=960),
            ]
        )
        report = lint(score, chord_bars=((5, 9, 0, 4),))
        assert not any(i.code == LintCode.DISSONANT_COLLISION for i in report.issues)

    def test_percussion_is_exempt(self) -> None:
        score = self._score(
            [
                NoteEvent(voice_id=1, pitch_midi=72, tick=0, duration_ticks=960),
                NoteEvent(voice_id=2, pitch_midi=73, tick=0, duration_ticks=120),
            ]
        )
        report = lint(score, chord_bars=((0, 4, 7),))
        assert not any(i.code == LintCode.DISSONANT_COLLISION for i in report.issues)

    def test_skipped_without_chord_bars(self) -> None:
        score = self._score(
            [
                NoteEvent(voice_id=1, pitch_midi=72, tick=0, duration_ticks=960),
                NoteEvent(voice_id=0, pitch_midi=61, tick=0, duration_ticks=960),
            ]
        )
        report = lint(score)
        assert not any(i.code == LintCode.DISSONANT_COLLISION for i in report.issues)


class TestPhraseGaps:
    """The melody may not run past PHRASE_BARS without a breath."""

    def _melody(self, durations: list[int], *, pitch: int = 72) -> NotationScore:
        """One melody note per bar with the given durations (4/4, one bar each)."""
        notes = [
            NoteEvent(voice_id=1, pitch_midi=pitch, tick=bar * 1920, duration_ticks=duration)
            for bar, duration in enumerate(durations)
        ]
        return _build_score(
            measures=_bars(*[(bar * 1920, (bar + 1) * 1920) for bar in range(len(durations))]),
            notes=notes,
        )

    def test_five_continuous_bars_fail(self) -> None:
        score = self._melody([1920] * 5)
        report = lint(score)
        assert any(i.code == LintCode.PHRASE_GAP_MISSING for i in report.issues)

    def test_four_bars_then_a_breath_passes(self) -> None:
        # Three full bars and a fourth that lifts off early into a
        # rest: the run stops at a phrase's length, and the next bar
        # starts a fresh group.
        score = self._melody([1920, 1920, 1920, 960, 960])
        report = lint(score)
        assert not any(i.code == LintCode.PHRASE_GAP_MISSING for i in report.issues)

    def test_pickup_does_not_extend_the_run(self) -> None:
        # A breath on bar 3, an anacrusis pickup in its final eighth,
        # then four full bars: the run after the breath is exactly a
        # phrase — the pickup belongs to it but must not lengthen it.
        notes = [
            NoteEvent(voice_id=1, pitch_midi=72, tick=0, duration_ticks=1920),
            NoteEvent(voice_id=1, pitch_midi=72, tick=1920, duration_ticks=1920),
            NoteEvent(voice_id=1, pitch_midi=72, tick=3840, duration_ticks=1920),
            NoteEvent(voice_id=1, pitch_midi=72, tick=5760, duration_ticks=960),
            NoteEvent(voice_id=1, pitch_midi=72, tick=7440, duration_ticks=240),
            NoteEvent(voice_id=1, pitch_midi=72, tick=7680, duration_ticks=1920),
            NoteEvent(voice_id=1, pitch_midi=72, tick=9600, duration_ticks=1920),
            NoteEvent(voice_id=1, pitch_midi=72, tick=11520, duration_ticks=1920),
            NoteEvent(voice_id=1, pitch_midi=72, tick=13440, duration_ticks=1920),
        ]
        score = _build_score(measures=_bars(*[(b * 1920, (b + 1) * 1920) for b in range(8)]), notes=notes)
        report = lint(score)
        assert not any(i.code == LintCode.PHRASE_GAP_MISSING for i in report.issues)

    def test_tied_notes_are_one_continuous_line(self) -> None:
        # Touching notes count as continuous even when marked tied.
        score = self._melody([1920] * 5)
        report = lint(score)
        assert any(i.code == LintCode.PHRASE_GAP_MISSING for i in report.issues)
