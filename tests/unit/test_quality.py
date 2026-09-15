"""Unit tests for the musical-quality scorecard.

Each metric is exercised on a hand-written score whose answer is known by
construction, so a metric that silently starts measuring something else
fails here rather than in a release gate.
"""

from __future__ import annotations

from saimc.compose.score import (
    PPQ,
    VOICE_BASS,
    VOICE_HARMONY,
    VOICE_MELODY,
    VOICE_PERCUSSION,
    KeySignature,
    Measure,
    NotationScore,
    NoteEvent,
)
from saimc.quality import (
    QUALITY_TESSITURA_OVERLAP_MAX,
    QUALITY_THRESHOLDS,
    PieceQuality,
    score_corpus,
    score_piece,
)

BAR_TICKS = 1920


def _score(notes: list[NoteEvent], *, bars: int = 1) -> NotationScore:
    return NotationScore.make(
        ppq=PPQ,
        key=KeySignature(root="C", mode="major"),
        time_signature="4/4",
        tempo_bpm=80.0,
        measures=[
            Measure(
                index=index,
                start_tick=index * BAR_TICKS,
                end_tick=(index + 1) * BAR_TICKS,
                time_signature="4/4",
            )
            for index in range(bars)
        ],
        notes=notes,
    )


def _melody(pitches: list[int], *, duration: int = 480) -> list[NoteEvent]:
    """One melody note per quarter, starting on beat 1."""
    return [
        NoteEvent(
            voice_id=VOICE_MELODY,
            pitch_midi=pitch,
            tick=index * 480,
            duration_ticks=duration,
        )
        for index, pitch in enumerate(pitches)
    ]


class TestMelodicIntervals:
    def test_a_stepwise_line_scores_all_steps(self) -> None:
        # C D E F G — every interval is a major second.
        piece = score_piece(_score(_melody([60, 62, 64, 65, 67])))
        assert piece.step_ratio == 1.0
        assert piece.repeat_ratio == 0.0
        assert piece.max_leap_semitones == 2
        assert piece.range_semitones == 7

    def test_an_all_thirds_line_scores_no_steps(self) -> None:
        # C E G C E — every interval is a third or wider.
        piece = score_piece(_score(_melody([60, 64, 67, 72, 76])))
        assert piece.step_ratio == 0.0

    def test_repeats_do_not_count_as_conjunct_motion(self) -> None:
        # C C C D E: two repeats then two steps. Conjunct motion is
        # measured over the moves only, so it is 2/2, not 2/4 — a stuck
        # generator must not earn step credit for standing still.
        piece = score_piece(_score(_melody([60, 60, 60, 62, 64])))
        assert piece.repeat_ratio == 0.5
        assert piece.step_ratio == 1.0

    def test_a_tied_hold_is_not_a_repeat(self) -> None:
        # C (tied C) D: one attack pair plus the tie. Counting the tie
        # onset as an interval would manufacture a repetition.
        notes = [
            NoteEvent(voice_id=VOICE_MELODY, pitch_midi=60, tick=0, duration_ticks=960),
            NoteEvent(
                voice_id=VOICE_MELODY, pitch_midi=60, tick=960, duration_ticks=480, tie=True
            ),
            NoteEvent(voice_id=VOICE_MELODY, pitch_midi=62, tick=1440, duration_ticks=480),
        ]
        piece = score_piece(_score(notes))
        assert piece.repeat_ratio == 0.0
        assert piece.step_ratio == 1.0

    def test_leap_recovery_needs_a_step_back(self) -> None:
        # C up a fifth to G, back down a second to F: recovered.
        recovered = score_piece(_score(_melody([60, 67, 65])))
        assert recovered.leap_recovery_ratio == 1.0

    def test_leap_continued_in_the_same_direction_is_not_recovered(self) -> None:
        # C up a fifth to G, up a second to A: the leap keeps climbing.
        continued = score_piece(_score(_melody([60, 67, 69])))
        assert continued.leap_recovery_ratio == 0.0

    def test_a_leap_answered_by_a_repeat_is_not_recovered(self) -> None:
        # C up a fifth to G, then G again: stalled, not answered.
        stalled = score_piece(_score(_melody([60, 67, 67])))
        assert stalled.leap_recovery_ratio == 0.0

    def test_no_leap_means_no_recovery_to_report(self) -> None:
        piece = score_piece(_score(_melody([60, 62, 64])))
        assert piece.leap_recovery_ratio is None

    def test_a_single_note_has_no_intervals(self) -> None:
        piece = score_piece(_score(_melody([60])))
        assert piece.step_ratio is None
        assert piece.repeat_ratio is None
        assert piece.max_leap_semitones is None
        assert piece.range_semitones == 0
        assert piece.distinct_durations == 1

    def test_distinct_durations_counts_note_values(self) -> None:
        notes = [
            NoteEvent(voice_id=VOICE_MELODY, pitch_midi=60, tick=0, duration_ticks=480),
            NoteEvent(voice_id=VOICE_MELODY, pitch_midi=62, tick=480, duration_ticks=480),
            NoteEvent(voice_id=VOICE_MELODY, pitch_midi=64, tick=960, duration_ticks=960),
        ]
        piece = score_piece(_score(notes, bars=2))
        assert piece.distinct_durations == 2


class TestTexture:
    def _with_accompaniment(self, melody: list[NoteEvent], harmony_count: int) -> NotationScore:
        harmony = [
            NoteEvent(
                voice_id=VOICE_HARMONY,
                pitch_midi=55,
                tick=index * 480,
                duration_ticks=480,
            )
            for index in range(harmony_count)
        ]
        return _score(melody + harmony, bars=2)

    def test_melody_above_its_accompaniment_scores_above_one(self) -> None:
        piece = score_piece(self._with_accompaniment(_melody([60, 62, 64, 65]), 2))
        assert piece.texture_hierarchy == 2.0

    def test_a_denser_accompaniment_fails_the_hierarchy(self) -> None:
        piece = score_piece(self._with_accompaniment(_melody([60, 62]), 8))
        assert piece.texture_hierarchy == 0.25

    def test_a_solo_piece_has_no_hierarchy_to_judge(self) -> None:
        piece = score_piece(_score(_melody([60, 62, 64])))
        assert piece.texture_hierarchy is None

    def test_percussion_is_not_an_accompaniment_to_beat(self) -> None:
        # A kit voice is a stream of short hits; out-counting it is not a
        # musical claim, so the hierarchy must be measured against the
        # harmony (4/2), not against 16 drum hits (4/16).
        drums = [
            NoteEvent(
                voice_id=VOICE_PERCUSSION, pitch_midi=38, tick=index * 240, duration_ticks=120
            )
            for index in range(16)
        ]
        harmony = [
            NoteEvent(voice_id=VOICE_HARMONY, pitch_midi=55, tick=index * 480, duration_ticks=480)
            for index in range(2)
        ]
        score = _score(_melody([60, 62, 64, 65]) + harmony + drums, bars=2)
        assert score_piece(score).texture_hierarchy == 2.0

    def test_percussion_alone_leaves_no_hierarchy_to_judge(self) -> None:
        drums = [
            NoteEvent(
                voice_id=VOICE_PERCUSSION, pitch_midi=38, tick=index * 240, duration_ticks=120
            )
            for index in range(16)
        ]
        score = _score(_melody([60, 62, 64, 65]) + drums, bars=2)
        assert score_piece(score).texture_hierarchy is None


class TestRegisters:
    def test_harmony_below_the_melody_separates(self) -> None:
        notes = [
            *_melody([72, 74, 76]),
            NoteEvent(voice_id=VOICE_HARMONY, pitch_midi=55, tick=0, duration_ticks=1920),
        ]
        piece = score_piece(_score(notes))
        assert piece.register_separation_semitones == 19.0
        assert piece.tessitura_overlap_semitones == 0

    def test_a_harmony_reaching_into_the_melody_overlaps(self) -> None:
        notes = [
            *_melody([72, 74, 76]),
            NoteEvent(voice_id=VOICE_HARMONY, pitch_midi=75, tick=0, duration_ticks=1920),
        ]
        piece = score_piece(_score(notes))
        # The harmony sits on one pitch inside the melody's 72..76 band,
        # so one semitone is shared — not a zero-width span.
        assert piece.tessitura_overlap_semitones == 1
        assert piece.register_separation_semitones == -1.0

    def test_a_solo_piece_has_no_register_metrics(self) -> None:
        piece = score_piece(_score(_melody([60, 62, 64])))
        assert piece.register_separation_semitones is None
        assert piece.tessitura_overlap_semitones is None


class TestBassFigures:
    def test_one_bar_repeated_is_one_figure(self) -> None:
        notes = [
            NoteEvent(voice_id=VOICE_BASS, pitch_midi=36, tick=bar * BAR_TICKS, duration_ticks=1920)
            for bar in range(4)
        ]
        assert score_piece(_score(notes, bars=4)).bass_onset_patterns == 1

    def test_distinct_figures_are_counted_separately(self) -> None:
        notes = [
            # Bar 0: a whole note. Bar 1: two half notes. Bar 2: another
            # whole note (same figure as bar 0).
            NoteEvent(voice_id=VOICE_BASS, pitch_midi=36, tick=0, duration_ticks=1920),
            NoteEvent(voice_id=VOICE_BASS, pitch_midi=36, tick=BAR_TICKS, duration_ticks=960),
            NoteEvent(voice_id=VOICE_BASS, pitch_midi=36, tick=BAR_TICKS + 960, duration_ticks=960),
            NoteEvent(voice_id=VOICE_BASS, pitch_midi=36, tick=2 * BAR_TICKS, duration_ticks=1920),
        ]
        assert score_piece(_score(notes, bars=3)).bass_onset_patterns == 2

    def test_a_piece_without_bass_has_no_pattern_count(self) -> None:
        assert score_piece(_score(_melody([60, 62]))).bass_onset_patterns is None


class TestFindings:
    def test_a_clean_piece_reports_no_findings(self) -> None:
        # Whole-tone steps up an octave, three note values, melody alone:
        # every metric that applies to a solo clears its bar.
        notes = [
            NoteEvent(voice_id=VOICE_MELODY, pitch_midi=pitch, tick=tick, duration_ticks=duration)
            for pitch, tick, duration in (
                (60, 0, 480),
                (62, 480, 480),
                (64, 960, 960),
                (66, 1440, 480),
                (68, 1920, 1920),
                (70, 2400, 480),
                (72, 2880, 1920),
            )
        ]
        piece = score_piece(_score(notes, bars=3))
        assert piece.step_ratio == 1.0
        assert piece.range_semitones == 12
        assert piece.distinct_durations == 3
        assert piece.findings() == ()

    def test_every_finding_names_a_threshold_and_a_remedy(self) -> None:
        piece = score_piece(_score(_melody([60, 72, 60, 84])))
        findings = piece.findings()
        metrics = {t.metric for t in QUALITY_THRESHOLDS}
        assert findings
        for finding in findings:
            assert finding.metric in metrics
            assert finding.hint
            assert finding.rationale
            assert finding.metric in finding.message()

    def test_the_leaping_line_is_flagged_for_both_leap_metrics(self) -> None:
        # Two octaves up then back: no step, a leap past the cap, and no
        # recovery — the line only touches the two extremes.
        piece = score_piece(_score(_melody([60, 84, 60])))
        flagged = {f.metric for f in piece.findings()}
        assert {"step_ratio", "max_leap_semitones", "leap_recovery_ratio"} <= flagged

    def test_a_threshold_miss_reports_the_measured_and_target(self) -> None:
        piece = score_piece(_score(_melody([60, 63, 96])))
        step = next(f for f in piece.findings() if f.metric == "step_ratio")
        assert step.direction == "min"
        assert step.measured == 0.0
        assert step.target > 0


class TestCorpusReport:
    def _pieces(self) -> list[PieceQuality]:
        """One stepwise piece with a repeated bass bar, and one wild one."""
        steady_bass = [
            NoteEvent(voice_id=VOICE_BASS, pitch_midi=36, tick=bar * BAR_TICKS, duration_ticks=1920)
            for bar in range(2)
        ]
        return [
            score_piece(
                _score(_melody([60, 62, 64, 66, 68]) + steady_bass, bars=2), piece="steady"
            ),
            score_piece(_score(_melody([60, 84, 60, 84])), piece="wild"),
        ]

    def test_the_report_averages_each_metric_over_the_pieces_that_report_it(self) -> None:
        report = score_corpus("test", self._pieces())
        assert report.piece_count == 2
        # Only the "clean" piece has a bass voice, so the mean is its own.
        assert report.metrics["bass_onset_patterns"] == 1.0
        # 1.0 for the stepwise line, 0.0 for the two-octave one.
        assert report.metrics["step_ratio"] == 0.5

    def test_a_metric_no_piece_reports_is_not_judged(self) -> None:
        solo = [score_piece(_score(_melody([60, 62, 64])), piece="solo")]
        report = score_corpus("solo", solo)
        assert report.metrics["texture_hierarchy"] is None
        assert report.metrics["register_separation_semitones"] is None
        assert report.metrics["tessitura_overlap_semitones"] is None

    def test_the_corpus_fails_when_a_mean_misses_a_bar(self) -> None:
        report = score_corpus("test", self._pieces())
        assert not report.passed
        # One repeated bass figure averages to 1 < 3, and the two pieces'
        # 3 and 1 distinct durations average to 2 < 3.
        assert any("bass_onset_patterns" in reason for reason in report.failure_reasons)
        assert any("distinct_durations" in reason for reason in report.failure_reasons)
        # Every reason names the mean it came from, so a mixed corpus does
        # not read as a single bad piece.
        assert all("mean over 2 pieces" in reason for reason in report.failure_reasons)

    def test_findings_carry_the_offending_pieces_not_the_mean(self) -> None:
        report = score_corpus("test", self._pieces())
        assert report.findings
        # The stepwise piece never has a zero step ratio; only the wild
        # one does, and it is reported individually.
        assert any(f.metric == "step_ratio" and f.measured == 0.0 for f in report.findings)
        assert not any(f.metric == "step_ratio" and f.measured == 0.5 for f in report.findings)

    def test_to_dict_is_json_ready_and_keeps_the_feedback(self) -> None:
        import json

        payload = json.loads(json.dumps(score_corpus("test", self._pieces()).to_dict()))
        assert payload["passed"] is False
        assert payload["piece_count"] == 2
        assert payload["findings"][0]["hint"]
        assert {p["piece"] for p in payload["pieces"]} == {"steady", "wild"}

    def test_an_empty_corpus_passes_vacuously(self) -> None:
        # There is nothing to be wrong about; the gate that consumes this
        # is responsible for requiring a non-empty matrix.
        report = score_corpus("empty", [])
        assert report.passed
        assert report.piece_count == 0


class TestThresholdTable:
    def test_every_threshold_bounds_at_least_one_direction(self) -> None:
        for threshold in QUALITY_THRESHOLDS:
            assert threshold.minimum is not None or threshold.maximum is not None, threshold.metric

    def test_the_two_sided_threshold_catches_both_sides(self) -> None:
        # Only a range has both ends, and it must reject a line narrower
        # than a fifth as well as one wider than two octaves.
        range_bar = next(t for t in QUALITY_THRESHOLDS if t.metric == "range_semitones")
        assert range_bar.minimum is not None
        assert range_bar.maximum is not None
        assert range_bar.violated_by(range_bar.minimum - 1)
        assert range_bar.violated_by(range_bar.maximum + 1)
        assert not range_bar.violated_by(range_bar.minimum)

    def test_a_minimum_miss_reads_as_less_than(self) -> None:
        bar = next(t for t in QUALITY_THRESHOLDS if t.metric == "step_ratio")
        assert bar.describe(0.1) == f"step_ratio 0.10 < {bar.minimum:.2f}"

    def test_a_maximum_miss_reads_as_greater_than(self) -> None:
        bar = next(t for t in QUALITY_THRESHOLDS if t.metric == "repeat_ratio")
        assert bar.describe(0.9) == f"repeat_ratio 0.90 > {bar.maximum:.2f}"

    def test_every_threshold_explains_itself_and_its_remedy(self) -> None:
        for threshold in QUALITY_THRESHOLDS:
            assert threshold.rationale, threshold.metric
            assert threshold.hint, threshold.metric

    def test_every_metric_has_a_threshold(self) -> None:
        measured = set(
            PieceQuality(
                piece="x",
                melody_notes=0,
                melody_bars=0,
                step_ratio=None,
                repeat_ratio=None,
                leap_recovery_ratio=None,
                max_leap_semitones=None,
                range_semitones=None,
                distinct_durations=None,
                texture_hierarchy=None,
                register_separation_semitones=None,
                tessitura_overlap_semitones=None,
                bass_onset_patterns=None,
            ).as_dict()
        )
        assert {t.metric for t in QUALITY_THRESHOLDS} == measured

    def test_the_tessitura_bar_leaves_room_for_a_major_third(self) -> None:
        # The docstring and the constant must agree: the harmony may
        # occupy a major third (4 semitones), not a fourth.
        assert QUALITY_TESSITURA_OVERLAP_MAX == 4
