"""Unit tests for the musical-quality scorecard.

Each metric is exercised on a hand-written score whose answer is known by
construction, so a metric that silently starts measuring something else
fails here rather than in a release gate.
"""

from __future__ import annotations

from dataclasses import replace

from saimc.compose.engine import compose
from saimc.compose.motif import FIGURES_BY_MOTION
from saimc.compose.plan import default_plan
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
    _LOCALISERS,
    AXES,
    QUALITY_HARMONY_PAD_COVERAGE_MIN,
    QUALITY_TESSITURA_OVERLAP_MAX,
    QUALITY_THRESHOLDS,
    BarSpan,
    PieceQuality,
    QualityFinding,
    _spans,
    localize,
    score_corpus,
    score_piece,
)
from saimc.spec import CompositionSpec, Mood

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


def _clean_line() -> list[NoteEvent]:
    """A solo line every metric that applies to a solo clears.

    Whole-tone steps up an octave over three note values. It is the fixture
    for "this part has nothing to say": a test that needs a clean axis adds
    its offending part to this line, so whatever is reported can be
    attributed to what it added rather than to the tune underneath.
    """
    return [
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
            NoteEvent(voice_id=VOICE_MELODY, pitch_midi=60, tick=960, duration_ticks=480, tie=True),
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
        # The distance between the ranges: the melody starts at 72, the
        # harmony ends at 55, so 17 semitones of clear air between them.
        assert piece.register_separation_semitones == 17.0
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
        # And no separation at all: the two ranges interlock, so the
        # distance is zero rather than a negative "mean below mean".
        assert piece.register_separation_semitones == 0.0

    def test_a_far_off_mean_cannot_hide_a_voice_in_the_melody_s_octave(self) -> None:
        # The failure the mean-based metric could not report: a harmony
        # part that spends most of its time an octave and a half below the
        # tune but reaches into its band. Three voices doing this averaged
        # 19.12 semitones of separation while piling into one octave.
        notes = [
            *_melody([72, 74, 76]),
            *[
                NoteEvent(
                    voice_id=VOICE_HARMONY, pitch_midi=pitch, tick=index * 480, duration_ticks=480
                )
                for index, pitch in enumerate([45, 45, 45, 45, 45, 45, 73])
            ],
        ]
        piece = score_piece(_score(notes, bars=2))
        assert piece.register_separation_semitones == 0.0
        assert piece.tessitura_overlap_semitones == 1

    def test_the_nearest_harmony_voice_sets_the_separation(self) -> None:
        # One voice folded well below the tune is not evidence that the
        # other one is clear of it: the minimum over voices is the number.
        notes = [
            *_melody([72, 74, 76]),
            NoteEvent(voice_id=VOICE_HARMONY, pitch_midi=45, tick=0, duration_ticks=1920),
            NoteEvent(voice_id=VOICE_HARMONY + 1, pitch_midi=64, tick=0, duration_ticks=1920),
        ]
        piece = score_piece(_score(notes))
        assert piece.register_separation_semitones == 8.0

    def test_a_pad_spanning_the_melody_sounds_none_of_it(self) -> None:
        # An arpeggio whose compass brackets the tune but folds below it:
        # the ranges intersect, the notes do not. The overlap counts what
        # is heard.
        notes = [
            *_melody([72, 74, 76]),
            *[
                NoteEvent(
                    voice_id=VOICE_HARMONY, pitch_midi=pitch, tick=index * 480, duration_ticks=480
                )
                for index, pitch in enumerate([36, 48, 60, 48, 36, 48, 60, 48])
            ],
        ]
        piece = score_piece(_score(notes, bars=2))
        assert piece.tessitura_overlap_semitones == 0
        assert piece.register_separation_semitones == 12.0

    def test_a_solo_piece_has_no_register_metrics(self) -> None:
        piece = score_piece(_score(_melody([60, 62, 64])))
        assert piece.register_separation_semitones is None
        assert piece.tessitura_overlap_semitones is None

    def test_the_bass_is_not_one_of_the_harmony_voices(self) -> None:
        # A bass line belongs under the tune and a low ensemble may share
        # its register legitimately; the register metric is about the
        # accompaniment that has a choice.
        notes = [
            *_melody([72, 74, 76]),
            NoteEvent(voice_id=VOICE_BASS, pitch_midi=45, tick=0, duration_ticks=1920),
        ]
        piece = score_piece(_score(notes))
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


class TestTheHarmonyBed:
    """The bed's coverage: does the accompaniment hold a chord anywhere?

    A hand-written score per case, so a metric that starts measuring
    something else — the bars the harmony *sounds* in rather than the bars
    it holds through — fails here rather than in a release gate.
    """

    def test_a_chord_held_through_its_bar_covers_it(self) -> None:
        notes = [NoteEvent(voice_id=VOICE_HARMONY, pitch_midi=60, tick=0, duration_ticks=BAR_TICKS)]
        assert score_piece(_score(notes)).harmony_pad_coverage == 1.0

    def test_a_broken_chord_covers_nothing(self) -> None:
        # The arpeggio's own step is a quarter of a bar: a chord stated and
        # released leaves the bar unsustained however many notes state it.
        notes = [
            NoteEvent(
                voice_id=VOICE_HARMONY,
                pitch_midi=60 + step,
                tick=step * 240,
                duration_ticks=240,
            )
            for step in range(4)
        ]
        assert score_piece(_score(notes)).harmony_pad_coverage == 0.0

    def test_half_a_bar_is_a_sustained_note(self) -> None:
        """The boundary the metric turns on, asserted rather than assumed."""
        notes = [
            NoteEvent(voice_id=VOICE_HARMONY, pitch_midi=60, tick=0, duration_ticks=BAR_TICKS // 2)
        ]
        assert score_piece(_score(notes)).harmony_pad_coverage == 1.0

    def test_the_share_is_of_the_piece_and_not_of_the_bars_it_sounds_in(self) -> None:
        notes = [NoteEvent(voice_id=VOICE_HARMONY, pitch_midi=60, tick=0, duration_ticks=BAR_TICKS)]
        assert score_piece(_score(notes, bars=4)).harmony_pad_coverage == 0.25

    def test_the_last_bar_is_measured_to_the_end_of_the_score(self) -> None:
        """The final bar has no start after it, so its length comes from the
        score's own end — a case no middle bar can reach."""
        notes = [
            NoteEvent(
                voice_id=VOICE_HARMONY, pitch_midi=60, tick=2 * BAR_TICKS, duration_ticks=BAR_TICKS
            )
        ]
        assert score_piece(_score(notes, bars=3)).harmony_pad_coverage == 1 / 3

    def test_a_solo_piece_has_no_bed_to_sustain(self) -> None:
        assert score_piece(_score(_melody([60, 62]))).harmony_pad_coverage is None

    def test_a_held_bass_note_is_a_pedal_and_not_a_bed(self) -> None:
        notes = [NoteEvent(voice_id=VOICE_BASS, pitch_midi=36, tick=0, duration_ticks=BAR_TICKS)]
        assert score_piece(_score(notes)).harmony_pad_coverage is None

    def test_a_bed_that_never_sustains_is_reported_with_the_knob_that_moves_it(self) -> None:
        notes = _melody([60, 62, 64, 65])
        notes.append(NoteEvent(voice_id=VOICE_HARMONY, pitch_midi=55, tick=0, duration_ticks=240))
        bed = next(
            finding
            for finding in score_piece(_score(notes)).findings()
            if finding.metric == "harmony_pad_coverage"
        )
        assert bed.direction == "min"
        assert bed.target == QUALITY_HARMONY_PAD_COVERAGE_MIN
        assert "harmony_broken_chord" in bed.hint


class TestTheBedBarIsReachable:
    """The bar is fired against a real piece, not asserted.

    A threshold no plan can move is a guard that cannot fail, which is the
    one thing this repo keeps finding and deleting. So the metric's own knob
    is exercised end to end — the default electrifying texture sustains
    nothing, and the plan field the finding's hint names is what fixes it.
    """

    def _coverage(self, *, broken_chord: bool) -> float | None:
        spec = CompositionSpec(mood=Mood.ELECTRIFYING, duration_seconds=30, seed=5)
        plan = replace(default_plan(spec), harmony_broken_chord=broken_chord)
        return score_piece(compose(spec, plan=plan).notation_score).harmony_pad_coverage

    def test_the_default_texture_misses_the_bar(self) -> None:
        assert self._coverage(broken_chord=True) == 0.0
        assert QUALITY_HARMONY_PAD_COVERAGE_MIN > 0.0

    def test_the_bar_clears_once_the_plan_stops_breaking_the_chord(self) -> None:
        coverage = self._coverage(broken_chord=False)
        assert coverage is not None
        assert coverage >= QUALITY_HARMONY_PAD_COVERAGE_MIN


class TestWhatTheScoreCannotMeasure:
    """The premises behind the metrics this module deliberately does not have.

    Each is a claim in the module docstring about a table the score does not
    carry, pinned here so that a change to the table fails a case instead of
    quietly making the note false.
    """

    def test_every_bass_figure_states_the_bar_s_root_on_its_downbeat(self) -> None:
        for motion, figure in FIGURES_BY_MOTION.items():
            assert figure[0][0] == 0, motion
            assert figure[0][2] == 0, motion


_AXIS_BY_METRIC = {
    "step_ratio": "melody",
    "leap_recovery_ratio": "melody",
    "repeat_ratio": "melody",
    "range_semitones": "melody",
    "max_leap_semitones": "melody",
    "distinct_durations": "melody",
    "texture_hierarchy": "accompaniment",
    "register_separation_semitones": "accompaniment",
    "tessitura_overlap_semitones": "accompaniment",
    "harmony_pad_coverage": "accompaniment",
    "bass_onset_patterns": "bass",
}

_UNLOCALISED_METRICS = (
    "range_semitones",
    "distinct_durations",
    "register_separation_semitones",
    "tessitura_overlap_semitones",
)
"""The four metrics no bar can hold, named so that the partition is a ratchet."""


class TestTheAxisTable:
    """One axis per metric, and the axes are the parts a remedy can move.

    Pinned as a literal, in the idiom the arbiter's metric tiers are pinned
    in: the axis is a judgement recorded once, so re-deciding which critic
    owns a bar is a declared edit rather than a swap that leaves every test
    green while the melody's report fills with the accompaniment's misses.
    """

    def test_the_table_is_the_partition_it_is_written_as(self) -> None:
        assert {t.metric: t.axis for t in QUALITY_THRESHOLDS} == _AXIS_BY_METRIC

    def test_the_axes_are_exactly_the_parts_the_table_names(self) -> None:
        assert set(_AXIS_BY_METRIC.values()) == set(AXES)

    def test_no_metric_pretends_to_say_where_it_is_when_no_bar_holds_it(self) -> None:
        """The localised metrics and the four that count a relation partition
        the table, so a fifth absent one arrives as a failed test rather than
        as a finding silently reported without bars."""
        assert set(_LOCALISERS) | set(_UNLOCALISED_METRICS) == set(_AXIS_BY_METRIC)
        assert not set(_LOCALISERS) & set(_UNLOCALISED_METRICS)


class TestBarSpans:
    def test_no_bars_is_no_spans(self) -> None:
        assert _spans([]) == ()

    def test_a_run_of_bars_reads_as_one_span(self) -> None:
        assert _spans([0, 1, 3]) == (BarSpan(1, 2), BarSpan(4, 4))

    def test_indices_are_sorted_and_deduplicated_before_they_are_merged(self) -> None:
        # A localiser collects bars in a set, so neither order nor repetition
        # is information — but both arrive from a reading rather than a
        # choice, and a span built from either would be a span built from
        # iteration order.
        assert _spans([3, 0, 3, 1]) == (BarSpan(1, 2), BarSpan(4, 4))

    def test_a_bar_is_numbered_the_way_a_reader_counts_it(self) -> None:
        assert BarSpan(7, 7).label() == "bar 7"
        assert BarSpan(3, 6).label() == "bars 3-6"


def _finding_for(metric: str) -> QualityFinding:
    """A finding named for one metric, so a localisation case names its own.

    Only the metric selects a localiser, so the numbers are placeholders and
    the axis and the remedy come from the shipped table — what is under test
    is a reading of the *score*, and a case that first had to make a metric
    breach its bar would be testing the bar as well as the bars it names.
    """
    bar = next(threshold for threshold in QUALITY_THRESHOLDS if threshold.metric == metric)
    return QualityFinding(
        metric=bar.metric,
        measured=0.0,
        target=0.0,
        direction="min",
        rationale=bar.rationale,
        hint=bar.hint,
        axis=bar.axis,
    )


def _bars(score: NotationScore, metric: str) -> tuple[BarSpan, ...]:
    """The bars `localize` reports for one metric on a hand-written score."""
    (localised,) = localize(score, [_finding_for(metric)])
    return localised.bars


class TestLocalisation:
    """Where a miss sits, read off the score its metrics were taken from.

    Each case is a hand-written score whose answer is known by construction,
    and by the same reading the metric itself makes: the bars named are the
    bars the metric's own counted events sit in, so a localisation that
    disagreed with its measurement would fail here.
    """

    def test_a_leap_is_placed_in_the_bar_it_lands_in(self) -> None:
        # Four quarters of steps, then a fifth: the interval is heard where
        # it arrives, which is beat 1 of bar 2.
        score = _score(_melody([60, 62, 64, 65, 72]), bars=2)
        assert [span.label() for span in _bars(score, "step_ratio")] == ["bar 2"]

    def test_a_repeat_is_placed_in_the_bar_it_sits_in(self) -> None:
        score = _score(_melody([60, 62, 62, 64]))
        assert [span.label() for span in _bars(score, "repeat_ratio")] == ["bar 1"]

    def test_a_repeat_is_not_an_offence_against_the_step_bar(self) -> None:
        # `step_ratio` measures the line's *moves*, and a repeat is not one —
        # it is scored by `repeat_ratio` — so the same interval must not be
        # reported against both bars.
        score = _score(_melody([60, 60, 62, 64]))
        assert _bars(score, "step_ratio") == ()

    def test_a_leap_past_the_cap_is_placed_where_it_lands(self) -> None:
        score = _score(_melody([60, 84]))
        assert [span.label() for span in _bars(score, "max_leap_semitones")] == ["bar 1"]

    def test_an_answered_leap_is_not_placed_at_all(self) -> None:
        score = _score(_melody([60, 67, 65]))
        assert _bars(score, "leap_recovery_ratio") == ()

    def test_an_unanswered_leap_is_placed_in_the_bar_it_lands_in(self) -> None:
        score = _score(_melody([60, 67, 69]))
        assert [span.label() for span in _bars(score, "leap_recovery_ratio")] == ["bar 1"]

    def test_a_leap_at_the_very_end_of_the_line_is_not_placed(self) -> None:
        # `_leap_recovery` does not count a closing leap — nothing follows it
        # to answer it — so the localisation must not name a bar for one.
        score = _score(_melody([60, 67]))
        assert _bars(score, "leap_recovery_ratio") == ()

    def test_consecutive_offending_bars_read_as_one_span(self) -> None:
        # Steps for two bars, then thirds for two: every interval of bars 3
        # and 4 is offending, and the two bars are one address.
        line = [60, 62, 64, 65, 67, 69, 71, 72, 76, 80, 84, 88, 84, 80, 76, 72]
        score = _score(_melody(line), bars=4)
        assert _bars(score, "step_ratio") == (BarSpan(3, 4),)

    def test_an_outcounting_voice_is_placed_in_the_bar_it_outcounts_in(self) -> None:
        # The bed plays four notes in bar 1 and holds its tongue in bar 2, so
        # the bar the tune is crowded in is the bar with the figure in it.
        notes = [
            NoteEvent(voice_id=VOICE_MELODY, pitch_midi=60, tick=0, duration_ticks=960),
            NoteEvent(voice_id=VOICE_MELODY, pitch_midi=62, tick=BAR_TICKS, duration_ticks=960),
            *(
                NoteEvent(voice_id=VOICE_HARMONY, pitch_midi=55, tick=tick, duration_ticks=480)
                for tick in (0, 480, 960, 1440)
            ),
        ]
        score = _score(notes, bars=2)
        assert [span.label() for span in _bars(score, "texture_hierarchy")] == ["bar 1"]

    def test_an_unsustained_bar_is_placed_and_not_the_bed_s_own_bar(self) -> None:
        notes = _melody([60, 62, 64, 65, 67, 69, 71, 72])
        notes.append(NoteEvent(voice_id=VOICE_HARMONY, pitch_midi=55, tick=0, duration_ticks=480))
        notes.append(
            NoteEvent(
                voice_id=VOICE_HARMONY, pitch_midi=57, tick=BAR_TICKS, duration_ticks=BAR_TICKS
            )
        )
        score = _score(notes, bars=2)
        # The bed holds bar 2 through and only states bar 1, so bar 1 is the
        # bar with nothing held — the complement of the metric's own reading.
        assert _bars(score, "harmony_pad_coverage") == (BarSpan(1, 1),)

    def test_the_bars_the_bass_repeats_its_commonest_figure_in(self) -> None:
        notes = [
            NoteEvent(voice_id=VOICE_BASS, pitch_midi=36, tick=0, duration_ticks=BAR_TICKS),
            NoteEvent(voice_id=VOICE_BASS, pitch_midi=36, tick=BAR_TICKS, duration_ticks=BAR_TICKS),
            NoteEvent(voice_id=VOICE_BASS, pitch_midi=36, tick=2 * BAR_TICKS, duration_ticks=960),
            NoteEvent(
                voice_id=VOICE_BASS, pitch_midi=36, tick=2 * BAR_TICKS + 960, duration_ticks=960
            ),
        ]
        score = _score(notes, bars=3)
        assert [span.label() for span in _bars(score, "bass_onset_patterns")] == ["bars 1-2"]

    def test_a_metric_that_counts_a_relation_is_placed_nowhere(self) -> None:
        """The four absences, asserted rather than left to the docstring.

        No bar holds a relation between the line's two extremes, so naming
        bars for one would name the bars where the piece is legal.
        """
        score = _score(_melody([60, 84, 60]))
        for metric in _UNLOCALISED_METRICS:
            assert _bars(score, metric) == (), metric
        # The contrast, so that "no bars" cannot be "nobody asked for the
        # score": the widest leap in the same line is placed.
        assert _bars(score, "max_leap_semitones")

    def test_a_scorecard_alone_carries_no_bars(self) -> None:
        # A `PieceQuality` is a block of numbers and has no notes to
        # attribute, which is the other cause of an empty `bars` — and the
        # difference between "nowhere" and "not asked".
        piece = score_piece(_score(_melody([60, 84, 60])))
        assert piece.findings()
        assert all(finding.bars == () for finding in piece.findings())

    def test_localizing_a_finding_moves_nothing_but_the_bars(self) -> None:
        finding = _finding_for("step_ratio")
        score = _score(_melody([60, 62, 64, 65, 72]), bars=2)
        assert localize(score, [finding]) == (replace(finding, bars=(BarSpan(2, 2),)),)

    def test_a_localized_message_names_the_bars_it_speaks_of(self) -> None:
        score = _score(_melody([60, 62, 64, 65, 72]), bars=2)
        (finding,) = localize(score, [_finding_for("step_ratio")])
        assert "(bar 2)" in finding.message()


class TestFindings:
    def test_a_clean_piece_reports_no_findings(self) -> None:
        piece = score_piece(_score(_clean_line(), bars=3))
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


class TestFindingsByAxis:
    """One critic per axis, and every axis answers — a clean one included.

    A report that listed only the axes which found something could not be
    told from one whose other critics never ran, and "the melody is clean" is
    exactly what a user asking about the melody wants to be told.
    """

    def test_every_axis_answers_even_when_it_has_nothing_to_say(self) -> None:
        piece = score_piece(_score(_melody([60, 84, 60])))
        grouped = piece.findings_by_axis()
        assert set(grouped) == set(AXES)
        assert grouped["bass"] == ()
        # The melody is the axis this line offends, so the finding lands
        # there and nowhere else: the two assertions together are what make
        # "clean" a reading rather than a silence.
        assert {f.metric for f in grouped["melody"]} == {f.metric for f in piece.findings()}

    def test_a_finding_sits_under_the_part_its_remedy_moves(self) -> None:
        # A clean tune, so the only thing that can be reported is the bed
        # added underneath it — a broken chord, which sustains nothing.
        notes = _clean_line()
        notes.extend(
            NoteEvent(voice_id=VOICE_HARMONY, pitch_midi=55, tick=step * 240, duration_ticks=240)
            for step in range(4)
        )
        grouped = score_piece(_score(notes, bars=3)).findings_by_axis()
        assert [f.metric for f in grouped["accompaniment"]] == ["harmony_pad_coverage"]
        assert grouped["melody"] == ()
        assert grouped["bass"] == ()


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
        assert all("mean over" in reason for reason in report.failure_reasons)

    def test_a_reason_names_a_partial_mean_as_partial(self) -> None:
        # Only one of the two pieces has a bass voice, so the bass mean is
        # over half the corpus — reporting it as "mean over 2 pieces" would
        # overstate what the number was taken from.
        report = score_corpus("test", self._pieces())
        bass = next(r for r in report.failure_reasons if "bass_onset_patterns" in r)
        assert "mean over the 1 of 2 piece(s) with this metric" in bass
        # Every piece reports distinct durations, so that mean covers all.
        durations = next(r for r in report.failure_reasons if "distinct_durations" in r)
        assert "mean over 2 pieces" in durations

    def test_a_single_piece_reason_reads_as_one_piece(self) -> None:
        solo = score_piece(_score(_melody([60, 63, 96])))
        report = score_corpus("solo", [solo])
        assert any("mean over 1 piece)" in reason for reason in report.failure_reasons)

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
                harmony_pad_coverage=None,
            ).as_dict()
        )
        assert {t.metric for t in QUALITY_THRESHOLDS} == measured

    def test_the_tessitura_bar_leaves_room_for_a_major_third(self) -> None:
        # The docstring and the constant must agree: the harmony may
        # occupy a major third (4 semitones), not a fourth.
        assert QUALITY_TESSITURA_OVERLAP_MAX == 4
