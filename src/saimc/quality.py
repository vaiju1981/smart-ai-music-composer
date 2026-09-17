"""Musical-quality scorecard — measuring the music, not the plumbing.

`saimc.benchmark` scores the *parser* (did the prompt become the right
spec?). This module scores the *composition*: given a `NotationScore`,
does what came out behave like a melody over an accompaniment, or like a
hot pot of instruments? It is the measurable half of "the music has to be
good"; the linter next door checks that the score is *legal*, which is a
different and much weaker claim.

Two shapes, mirroring `benchmark.py` so the two reports read alike:

- `score_piece(score)` measures one piece into `PieceQuality`.
- `score_corpus(label, pieces)` averages a corpus into `QualityReport`
  with `passed` / `failure_reasons`.

`QUALITY_THRESHOLDS` is the single place the bars live. Each entry names
the metric, the direction it must not cross, why the bar is where it is,
and — the feedback half — which engine knob moves it. A report's
`findings` therefore read as instructions to a repair loop or to whoever
is tuning the generator, not just as a pass/fail verdict.

Thresholds are conventional, not reverse-engineered from the current
engine: a step-dominant line, a leap answered by a step, a melody more
active than its accompaniment. Today's engine misses several of them —
that is the point, and it is what makes the gate in
`saimc.release.gates` non-vacuous.

Three axes a critic might ask for are deliberately absent, and the first
two for one reason: **this module measures a `NotationScore`, which
carries notes and no tables.** *Bass root motion* — the share of chord
changes the left hand leaves the root on — needs the bar's chord. This
paragraph used to claim the reading was unreachable as well, and Phase F1
falsified both halves of that. `EngineOutput.chord_bars` and `bar_keys`
carry the bar's chord and the key it belongs to, so every caller that
holds the engine's output rather than a score — the repair loop, the
critics, the conductor — can take the reading, and under the plan's
`bass_root_motion` knob the two settings measure 100% and 32% of chord
changes on the root. What survives is narrower and still true:
`score_piece` takes a `NotationScore`, so an axis that needs the chord
table has no home in this file. *Rhythm-section variety* needs the style's
own variant count: a waltz repeats one bar through 0.80-0.90 of a piece,
and a rock groove whose rotation the plan switched off repeats it through
0.78-0.90, so a bar over that reading fires on a legal one-variant style.
*Contour* is measured and left alone, because no bar could act on it:
across the palette grid and seven mutations of the melody's vocabulary the
share of turning points stays inside 0.37-0.55. What a contour complaint is
actually about — a line that only steps, a leap that is never answered —
is `step_ratio` and `leap_recovery_ratio`. The defects behind the first
two are reported too: one bass figure for a whole piece is
`bass_onset_patterns`, and a kit with too few bars to play is the style's
vocabulary, not a bar that would call a waltz wrong.

Two things turn a measurement into a review comment, and both are here
rather than in the session that reads them. `QualityThreshold.axis` names
the part of the piece a remedy for the bar would move — the melody's line,
the accompaniment under it, or the bass — so the critics can be asked one
part at a time rather than all eleven numbers at once. `localize` then says
*where*: the bars the finding's own counted events sit in, so a
`repeat_ratio` miss arrives as "the melody answers itself in bars 3-6 and
11" rather than as a piece-wide number with no address. Bars and not
sections, because a score carries measure boundaries and no section table —
the same reason three axes above are absent — and 1-based, the way a reader
counts them, with `score.measures` indexed from 0 for a caller mapping back.

A localisation is not a second way of taking a metric, and it cannot
disagree with one: the bars it names are the bars the metric's own counted
events sit in, and the piece-wide value stays the piece-wide value. What it
deliberately is *not* is the metric re-taken over each bar. A bar of a
four-note melody reads 0.00, 0.33 or 0.50, and a bar-level breach would
fire on legal bars: measured, a 120-second calming piece whose own
`step_ratio` clears its bar at 0.71 has six of its thirty bars with moves
below 0.45. Four metrics have no localisation at all, and the rule that
excludes them is what admits the other seven — the metric's counted *thing*
has to land in a bar. `range_semitones` counts semitones between two
extreme notes, `distinct_durations` counts note values, and
`register_separation_semitones` and `tessitura_overlap_semitones` count
semitones and pitches across two voices' ranges: a difference or a relation
between the whole piece's extremes rather than an event, so there is no bar
to name and neither of them reports one.
"""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Final, Literal

from saimc.compose.forms import LEAP_MIN_SEMITONES, STEP_MAX_SEMITONES
from saimc.compose.score import (
    VOICE_BASS,
    VOICE_HARMONY,
    VOICE_MELODY,
    VOICE_PERCUSSION,
    NotationScore,
    NoteEvent,
)

# `STEP_MAX_SEMITONES` — "a step is a major second or less" — and
# `LEAP_MIN_SEMITONES` — "a leap is a fourth or wider" — are defined in
# `forms.py` and re-exported here, so the scorecard, the linter's
# passing-tone licence and the generator's leap-recovery pass cannot
# disagree about which intervals are steps and which are leaps.

QUALITY_STEP_RATIO_MIN: float = 0.45
"""At least this share of the melody's *moving* intervals must be steps.

Measured over the intervals that actually move — repeats are excluded
and scored separately by `repeat_ratio`, so a stuck generator cannot
earn conjunct-motion credit for standing still. A singable line moves
mostly by step; one built from thirds and fifths has nothing to sing.
Roughly half is the conventional floor for tonal melody.
"""

QUALITY_LEAP_RECOVERY_MIN: float = 0.60
"""After a leap of a fourth or more, the next move must be a step in the
opposite direction. Without it the line does not have a contour, it has
a zigzag."""

QUALITY_REPEAT_RATIO_MAX: float = 0.25
"""At most this share of melodic intervals may repeat the previous pitch.

Repetition is a device; a third of all intervals is a stuck generator.
"""

QUALITY_MELODIC_RANGE_MIN: int = 7
"""A melody should span at least a fifth, or it is an accompaniment
figure rather than a tune."""

QUALITY_MELODIC_RANGE_MAX: int = 24
"""Two octaves is the widest a sung line comfortably reaches."""

QUALITY_MAX_LEAP_MAX: int = 12
"""No single melodic interval wider than an octave."""

QUALITY_DISTINCT_DURATIONS_MIN: int = 3
"""A melody needs at least three distinct note values; fewer is a
metronome with pitch."""

QUALITY_TEXTURE_HIERARCHY_MIN: float = 1.0
"""The melody must be at least as active as the busiest accompaniment
voice, or nothing is leading and the texture is a hot pot."""

QUALITY_REGISTER_SEPARATION_MIN: float = 3.0
"""The nearest harmony voice must keep a minor third between its own range
and the melody's — no harmony pitch inside the melody's compass at all.
Voices written in the same register are heard as one blurred part, and
this is a range distance rather than a difference of averages, so a wide
accompaniment cannot hide an overlap behind a far-away mean. The engine
establishes it wherever the instrument has room: the register pass places
the accompaniment under (or over) the finished tune with
`HARMONY_MELODY_CLEARANCE` between them, falling back to the instrument's
whole compass. It does not hold for the instruments whose compass cannot
hold both a line and a bed clear of it — measured over the palette grid,
four of sixty-six — and there the scorecard reports the overlap rather
than the pass pretending to have placed it."""

QUALITY_TESSITURA_OVERLAP_MAX: int = 4
"""The harmony may sound at most a major third of the melody's band.

Two voices whose ranges interlock this way are heard as one crowded
line even when their averages are an octave apart. Counted in semitones
the harmony actually sounds inside the melody's range, not in the width
of the intersection of two ranges: an accompaniment spanning the
melody's range while sounding none of it — a pad folded below the tune —
is not crowding anything. It holds wherever the separation above does,
and where the instrument leaves the register pass nowhere to put the bed
it is the metric that reports it."""

QUALITY_BASS_ONSET_PATTERNS_MIN: int = 3
"""The bass must show at least three distinct onset patterns across its
bars. One pattern is a loop, not a bass line."""

QUALITY_HARMONY_PAD_COVERAGE_MIN: float = 0.25
"""At least this share of the bars must carry a harmony note held through
half of the bar.

A bed is what an accompaniment *is*. An arpeggio and a stab state the chord
and move on, so a piece made only of those has a harmony that never sustains
a note anywhere — and the accompaniment's figure is chosen for the whole
piece, so nothing in a piece can be holding a chord while the rest stabs.
A quarter of the bars is the least a bed can be and still be one; measured
over the palette grid, a pad clears 0.92-1.00 of them, so the bar is a
statement about the bed rather than a number today's pieces happen to pass.
"""


Axis = Literal["melody", "accompaniment", "bass"]
"""The part of a piece the metrics are grouped by, and the parts are closed.

A closed vocabulary rather than a free string, because the axis is what a
critic is asked for by name: a session asking for the melody's critic has to
be told when it has asked for something that does not exist, rather than
being handed an empty report that reads like a clean piece.
"""

AXES: Final[tuple[Axis, ...]] = ("melody", "accompaniment", "bass")
"""Every axis, in the order a report reads them: the tune, then its bed."""


@dataclass(frozen=True)
class BarSpan:
    """A run of consecutive bars, numbered the way a reader counts them."""

    first_bar: int
    last_bar: int

    def label(self) -> str:
        """`bar 7` for one bar, `bars 3-6` for a run."""
        if self.first_bar == self.last_bar:
            return f"bar {self.first_bar}"
        return f"bars {self.first_bar}-{self.last_bar}"


@dataclass(frozen=True)
class QualityThreshold:
    """One bar a piece must clear, with the reason and the remedy."""

    metric: str
    minimum: float | None
    maximum: float | None
    rationale: str
    hint: str
    axis: Axis
    """The part of the piece a remedy for this bar would move.

    Read off the `hint` rather than chosen: every hint names the code that
    writes the offending notes, and the part that code writes is the part
    the bar is about. So this is a fact about the metric, not a taste, and
    it is what lets one critic own one part — six melody-line metrics, four
    about the accompaniment's density and placement, and the bass figure.

    Three rather than the four specialists the harness set out to write,
    and the two missing ones are E1's finding seen from here: nothing
    measures the ensemble's orchestration at all, and nothing measures the
    drum kit's groove, because a score carries no style table to read a
    variation count from. A part with no metric has no critic.
    """

    def violated_by(self, measured: float) -> bool:
        """True iff `measured` is on the wrong side of this bar."""
        if self.minimum is not None and measured < self.minimum:
            return True
        return self.maximum is not None and measured > self.maximum

    def describe(self, measured: float) -> str:
        """One line naming the miss, for the report's failure list."""
        if self.minimum is not None and measured < self.minimum:
            return f"{self.metric} {measured:.2f} < {self.minimum:.2f}"
        return f"{self.metric} {measured:.2f} > {self.maximum:.2f}"


QUALITY_THRESHOLDS: tuple[QualityThreshold, ...] = (
    QualityThreshold(
        metric="step_ratio",
        minimum=QUALITY_STEP_RATIO_MIN,
        maximum=None,
        rationale="a singable line moves mostly by step",
        hint=(
            "the melody's interval vocabulary is set in "
            "saimc/compose/motif.py by STEP_CHOICES/STEP_WEIGHTS, which "
            "walk CHORD-TONE INDICES: a step of 1 is a third and 2 is a "
            "fifth in semitones. Weight the walk in semitones instead."
        ),
        axis="melody",
    ),
    QualityThreshold(
        metric="leap_recovery_ratio",
        minimum=QUALITY_LEAP_RECOVERY_MIN,
        maximum=None,
        rationale="a leap must be answered by a step back",
        hint=(
            "apply the leap-recovery rule where the walk is generated "
            "(saimc/compose/motif.py): after an interval of a fourth or "
            "more, force the next step to be a small one in the opposite "
            "direction."
        ),
        axis="melody",
    ),
    QualityThreshold(
        metric="repeat_ratio",
        minimum=None,
        maximum=QUALITY_REPEAT_RATIO_MAX,
        rationale="repetition is a device, not the default move",
        hint=(
            "the 0-step in STEP_WEIGHTS carries 15% weight but repeated "
            "notes cluster at phrase starts; make repetition conditional "
            "on position in the phrase."
        ),
        axis="melody",
    ),
    QualityThreshold(
        metric="range_semitones",
        minimum=QUALITY_MELODIC_RANGE_MIN,
        maximum=QUALITY_MELODIC_RANGE_MAX,
        rationale="a tune spans a fifth to two octaves",
        hint=(
            "widen or narrow the melody's register window — the melody "
            "voice is pinned to chord tones over the bar's chord, which "
            "bounds its span."
        ),
        axis="melody",
    ),
    QualityThreshold(
        metric="max_leap_semitones",
        minimum=None,
        maximum=QUALITY_MAX_LEAP_MAX,
        rationale="no melodic interval wider than an octave",
        hint=(
            "cap the walk's step size (saimc/compose/motif.py) and check "
            "the register wrap at the chord-tone boundary, which is where "
            "an unbounded index step can jump an octave."
        ),
        axis="melody",
    ),
    QualityThreshold(
        metric="distinct_durations",
        minimum=QUALITY_DISTINCT_DURATIONS_MIN,
        maximum=None,
        rationale="a melody needs a rhythm, not one note value",
        hint=(
            "the melody's durational palette is chosen in "
            "saimc/compose/engine.py's _melody_bar; add at least a long "
            "and a short value per phrase rather than per mood."
        ),
        axis="melody",
    ),
    QualityThreshold(
        metric="texture_hierarchy",
        minimum=QUALITY_TEXTURE_HIERARCHY_MIN,
        maximum=None,
        rationale="the melody must be the most active voice",
        hint=(
            "pull the harmony's note count below the melody's: the "
            "harmony generator in saimc/compose/engine.py chooses pad / "
            "arpeggio / stab per bar, and the arpeggio figure is denser "
            "than the melody it is supposed to support."
        ),
        axis="accompaniment",
    ),
    QualityThreshold(
        metric="register_separation_semitones",
        minimum=QUALITY_REGISTER_SEPARATION_MIN,
        maximum=None,
        rationale="voices sharing a register blur into one part",
        hint=(
            "a harmony voice has climbed into the melody's range. The "
            "register pass in saimc/compose/engine.py "
            "(_settle_harmony_register) places the bed relative to the "
            "finished tune and drops a note it cannot place; check that "
            "the voice's instrument window (saimc/instruments.py "
            "bed_window) still spans an octave below the melody's band, "
            "which _melody_band_for is what arranges."
        ),
        axis="accompaniment",
    ),
    QualityThreshold(
        metric="tessitura_overlap_semitones",
        minimum=None,
        maximum=QUALITY_TESSITURA_OVERLAP_MAX,
        rationale="the harmony's range must not climb into the melody's",
        hint=(
            "the harmony is sounding notes inside the melody's range, so "
            "the tune has no register of its own. Either the melody's band "
            "is too wide for its instrument (LINE_BAND_SEMITONES in "
            "saimc/instruments.py) or the bed window in the same module "
            "puts the accompaniment where the tune is."
        ),
        axis="accompaniment",
    ),
    QualityThreshold(
        metric="bass_onset_patterns",
        minimum=QUALITY_BASS_ONSET_PATTERNS_MIN,
        maximum=None,
        rationale="a bass line repeats a figure, it does not repeat one bar",
        hint=(
            "vary the bass figure across the section form in "
            "saimc/compose/engine.py rather than repeating one bar's "
            "onset pattern for the whole piece."
        ),
        axis="bass",
    ),
    QualityThreshold(
        metric="harmony_pad_coverage",
        minimum=QUALITY_HARMONY_PAD_COVERAGE_MIN,
        maximum=None,
        rationale="the accompaniment has to sustain its chords somewhere",
        hint=(
            "the bed's figure is chosen for the whole piece rather than per "
            "section, so a mood whose texture is the broken chord plays "
            "arpeggio or stab in every bar it sounds in: setting "
            "`harmony_broken_chord` on the composition plan lets the bed "
            "hold its chords. Per-section control is the texture knob "
            "saimc/compose/plan.py does not carry yet."
        ),
        axis="accompaniment",
    ),
)


@dataclass(frozen=True)
class QualityFinding:
    """One threshold a piece missed, with the measured value and the remedy."""

    metric: str
    measured: float
    target: float
    direction: str  # "min" | "max"
    rationale: str
    hint: str
    axis: Axis
    bars: tuple[BarSpan, ...] = ()
    """The bars the miss sits in, and empty when nothing localises it.

    Empty has two causes and they are different things to a reader: the
    metric counts something no bar holds (`range_semitones`), or nobody has
    asked `localize` for the score. A finding built by `findings()` alone
    carries no bars, because a `PieceQuality` is a block of numbers and has
    no notes to attribute; `localize(score, findings)` is what fills them.
    """

    def message(self) -> str:
        """One readable line: what missed, by how much, and what to change."""
        comparator = "<" if self.direction == "min" else ">"
        where = f" ({', '.join(span.label() for span in self.bars)})" if self.bars else ""
        return (
            f"{self.axis} — {self.metric} {self.measured:.2f} {comparator} "
            f"{self.target:.2f}{where}: {self.rationale}. To move it: {self.hint}"
        )


@dataclass(frozen=True)
class PieceQuality:
    """Raw measurements for one piece.

    Every field is a measurement, never a judgement; the thresholds turn
    them into findings. A `None` means the metric does not apply to this
    piece (a solo has no harmony to separate from, so its
    `register_separation_semitones` is None and is not checked).

    Interval metrics are computed over the melody voice's notes in onset
    order, ignoring ties (a tie is one note held, not a repeated attack).
    """

    piece: str
    melody_notes: int
    melody_bars: int
    step_ratio: float | None
    repeat_ratio: float | None
    leap_recovery_ratio: float | None
    max_leap_semitones: int | None
    range_semitones: int | None
    distinct_durations: int | None
    texture_hierarchy: float | None
    register_separation_semitones: float | None
    tessitura_overlap_semitones: int | None
    bass_onset_patterns: int | None
    harmony_pad_coverage: float | None

    def as_dict(self) -> dict[str, float | None]:
        """Metric name -> value, for the threshold loop and the report.

        Ints are widened to float so the threshold table has one type to
        compare against.
        """
        measured: dict[str, int | float | None] = {
            "step_ratio": self.step_ratio,
            "repeat_ratio": self.repeat_ratio,
            "leap_recovery_ratio": self.leap_recovery_ratio,
            "max_leap_semitones": self.max_leap_semitones,
            "range_semitones": self.range_semitones,
            "distinct_durations": self.distinct_durations,
            "texture_hierarchy": self.texture_hierarchy,
            "register_separation_semitones": self.register_separation_semitones,
            "tessitura_overlap_semitones": self.tessitura_overlap_semitones,
            "bass_onset_patterns": self.bass_onset_patterns,
            "harmony_pad_coverage": self.harmony_pad_coverage,
        }
        return {name: None if value is None else float(value) for name, value in measured.items()}

    def entry(self) -> dict[str, object]:
        """The full per-piece record: identity, note counts, and metrics.

        This is the shape a report's `pieces` list and a manifest's
        `quality` block both carry, so a trend can tell a piece with no
        melody voice apart from one whose melody measured badly.
        """
        return {
            "piece": self.piece,
            "melody_notes": self.melody_notes,
            "melody_bars": self.melody_bars,
            **self.as_dict(),
        }

    def findings(self) -> tuple[QualityFinding, ...]:
        """Every threshold this piece missed, in table order."""
        measured = self.as_dict()
        found: list[QualityFinding] = []
        for threshold in QUALITY_THRESHOLDS:
            value = measured.get(threshold.metric)
            if value is None or not threshold.violated_by(value):
                continue
            if threshold.minimum is not None and value < threshold.minimum:
                target, direction = threshold.minimum, "min"
            else:
                assert threshold.maximum is not None  # violated_by guarantees one
                target, direction = threshold.maximum, "max"
            found.append(
                QualityFinding(
                    metric=threshold.metric,
                    measured=value,
                    target=target,
                    direction=direction,
                    rationale=threshold.rationale,
                    hint=threshold.hint,
                    axis=threshold.axis,
                )
            )
        return tuple(found)

    def findings_by_axis(self) -> dict[Axis, tuple[QualityFinding, ...]]:
        """Every axis, with the findings it owns — a clean axis included.

        Every axis and not only the offending ones: a report that lists the
        axes which found something cannot be told from one whose other
        critics never ran, and "the melody is clean" is the thing a user
        asking for the melody wants to be told.
        """
        found = self.findings()
        return {axis: tuple(finding for finding in found if finding.axis == axis) for axis in AXES}


@dataclass(frozen=True)
class QualityReport:
    """Corpus-level scorecard: the mean of every metric plus the verdict."""

    label: str
    piece_count: int
    metrics: dict[str, float | None]
    passed: bool
    failure_reasons: tuple[str, ...] = ()
    findings: tuple[QualityFinding, ...] = ()
    pieces: tuple[PieceQuality, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "label": self.label,
            "piece_count": self.piece_count,
            "metrics": self.metrics,
            "passed": self.passed,
            "failure_reasons": list(self.failure_reasons),
            "findings": [
                {
                    "metric": f.metric,
                    "measured": f.measured,
                    "target": f.target,
                    "direction": f.direction,
                    "rationale": f.rationale,
                    "hint": f.hint,
                    "axis": f.axis,
                }
                for f in self.findings
            ],
            "pieces": [p.entry() for p in self.pieces],
        }


def score_piece(score: NotationScore, *, piece: str = "piece") -> PieceQuality:
    """Measure one NotationScore into a `PieceQuality`."""
    melody = _voice_notes(score, VOICE_MELODY)
    intervals = _intervals(melody)
    moves = [step for step in intervals if step != 0]

    return PieceQuality(
        piece=piece,
        melody_notes=len(melody),
        melody_bars=_bars_covered(score, melody),
        step_ratio=_ratio(sum(1 for step in moves if abs(step) <= STEP_MAX_SEMITONES), len(moves)),
        repeat_ratio=_ratio(len(intervals) - len(moves), len(intervals)),
        leap_recovery_ratio=_leap_recovery(intervals),
        max_leap_semitones=max((abs(step) for step in moves), default=None),
        range_semitones=_range_semitones(melody),
        distinct_durations=_distinct_durations(melody),
        texture_hierarchy=_texture_hierarchy(score, melody),
        register_separation_semitones=_register_separation(score, melody),
        tessitura_overlap_semitones=_tessitura_overlap(score),
        bass_onset_patterns=_bass_onset_patterns(score),
        harmony_pad_coverage=_harmony_pad_coverage(score),
    )


def score_corpus(label: str, pieces: Iterable[PieceQuality]) -> QualityReport:
    """Average per-piece measurements into a corpus scorecard.

    A metric is averaged over the pieces it applies to; a metric no piece
    reports (no harmony anywhere, say) stays None and is not judged. The
    verdict fails when any threshold is missed by the corpus mean, and
    `findings` carries the offending pieces' individual misses so the
    report says *which* piece and *which* knob.
    """
    measured_pieces = tuple(pieces)
    metrics: dict[str, float | None] = {}
    counts: dict[str, int] = {}
    for threshold in QUALITY_THRESHOLDS:
        values = [
            value
            for value in (p.as_dict().get(threshold.metric) for p in measured_pieces)
            if value is not None
        ]
        counts[threshold.metric] = len(values)
        metrics[threshold.metric] = sum(values) / len(values) if values else None

    failures: list[str] = []
    for threshold in QUALITY_THRESHOLDS:
        value = metrics.get(threshold.metric)
        if value is None or not threshold.violated_by(value):
            continue
        scope = _mean_scope(counts[threshold.metric], len(measured_pieces))
        failures.append(f"{threshold.describe(value)} ({scope})")

    findings = tuple(f for p in measured_pieces for f in p.findings())
    return QualityReport(
        label=label,
        piece_count=len(measured_pieces),
        metrics=metrics,
        passed=not failures,
        failure_reasons=tuple(failures),
        findings=findings,
        pieces=measured_pieces,
    )


def _mean_scope(measured: int, corpus: int) -> str:
    """How many pieces a corpus mean was taken over.

    A metric only applies to some arrangements — a solo piece has no bass
    figure to count — so a mean over one of four pieces must not read as a
    corpus-wide number.
    """
    if measured == corpus:
        return f"mean over {measured} {'piece' if measured == 1 else 'pieces'}"
    return f"mean over the {measured} of {corpus} piece(s) with this metric"


def _voice_notes(score: NotationScore, voice_id: int) -> list[NoteEvent]:
    """One voice's notes in onset order, ties dropped.

    A tie is the same pitch held across a boundary, not a new attack, so
    counting it as an interval would manufacture a repeat.
    """
    notes = [n for n in score.notes if n.voice_id == voice_id and not n.tie]
    return sorted(notes, key=lambda n: (n.tick, n.pitch_midi))


def _onsets(notes: list[NoteEvent]) -> list[tuple[int, int]]:
    """One (tick, lowest pitch) per onset, in tick order.

    At a shared tick (a chord in a single voice) the lowest note carries
    the line, so an onset is one tick and one pitch rather than an
    arbitrary member of the chord. This is the sequence both the interval
    metrics and the localisations are taken over, which is what keeps a
    finding's bars and its measurement from reading different notes.
    """
    by_tick: dict[int, list[int]] = {}
    for note in notes:
        by_tick.setdefault(note.tick, []).append(note.pitch_midi)
    return [(tick, min(by_tick[tick])) for tick in sorted(by_tick)]


def _intervals(notes: list[NoteEvent]) -> list[int]:
    """Semitone deltas between successive onsets of one voice."""
    return [later - earlier for (_, earlier), (_, later) in pairwise(_onsets(notes))]


def _ratio(numerator: int, denominator: int) -> float | None:
    """`numerator / denominator`, or None when there is nothing to divide."""
    return numerator / denominator if denominator else None


def _leap_recovery(intervals: list[int]) -> float | None:
    """Share of leaps answered by a step in the opposite direction.

    None when the line contains no leap at all — nothing to recover from.
    A leap whose recovery interval is a repeat (delta 0) has not been
    recovered: it has stalled, and the rule asks for a step *back*.
    """
    leaps = [
        index
        for index, step in enumerate(intervals)
        if abs(step) >= LEAP_MIN_SEMITONES and index + 1 < len(intervals)
    ]
    if not leaps:
        return None
    recovered = sum(
        1
        for index in leaps
        if 0 < abs(intervals[index + 1]) <= STEP_MAX_SEMITONES
        and intervals[index + 1] * intervals[index] < 0
    )
    return recovered / len(leaps)


def _range_semitones(notes: list[NoteEvent]) -> int | None:
    """Highest minus lowest pitch, or None for an empty voice."""
    if not notes:
        return None
    pitches = [n.pitch_midi for n in notes]
    return max(pitches) - min(pitches)


def _distinct_durations(notes: list[NoteEvent]) -> int | None:
    """How many distinct note values the voice uses."""
    if not notes:
        return None
    return len({n.duration_ticks for n in notes})


def _bars_covered(score: NotationScore, notes: list[NoteEvent]) -> int:
    """How many of the score's bars the voice sounds in."""
    layout = _BarLayout.of(score)
    return len({layout.index_of(n.tick) for n in notes} - {None})


@dataclass(frozen=True)
class _BarLayout:
    """Measure boundaries, so a tick maps to a bar without rescanning."""

    starts: tuple[int, ...]
    end_tick: int

    @classmethod
    def of(cls, score: NotationScore) -> _BarLayout:
        return cls(
            starts=tuple(m.start_tick for m in score.measures),
            end_tick=score.measures[-1].end_tick if score.measures else 0,
        )

    def index_of(self, tick: int) -> int | None:
        """The bar sounding at `tick`.

        An onset before the first bar (an anacrusis pickup) belongs to the
        first bar; a tick past the last bar has no bar.
        """
        if not self.starts or tick >= self.end_tick:
            return None
        return max(0, bisect_right(self.starts, tick) - 1)

    def length_of(self, index: int) -> int:
        """How many ticks bar `index` lasts.

        Bars are read from their starts, so the last one is measured to the
        score's end rather than to a start that is not there.
        """
        end = self.starts[index + 1] if index + 1 < len(self.starts) else self.end_tick
        return end - self.starts[index]


def _accompaniment_voice_ids(score: NotationScore) -> list[int]:
    """Every voice that is neither the tune nor the kit, in a stable order.

    The bass is one of them: `texture_hierarchy` asks whether the melody
    leads the busiest thing anywhere under it, and a bass line busier than
    the tune crowds it the way a pad busier than the tune does. Percussion
    is out for `_harmony_voice_ids`' reason — a drum map is not a note count
    the tune can lead.
    """
    skip = {VOICE_MELODY, VOICE_PERCUSSION}
    return sorted({n.voice_id for n in score.notes if n.voice_id not in skip})


def _texture_hierarchy(score: NotationScore, melody: list[NoteEvent]) -> float | None:
    """Melody note count over the busiest accompaniment voice's count.

    Percussion is excluded: a kit voice is a stream of short hits, and
    comparing a melody's note count to a hi-hat's is not a musical claim.
    None when the piece is melody alone.
    """
    accompaniment = {
        voice_id: len(_voice_notes(score, voice_id)) for voice_id in _accompaniment_voice_ids(score)
    }
    busiest = max(accompaniment.values(), default=0)
    if not busiest or not melody:
        return None
    return len(melody) / busiest


def _harmony_voice_ids(score: NotationScore) -> list[int]:
    """The voices this module calls harmony, in a stable order.

    Voice 0 is the bass and is deliberately not one of them. A bass line
    is placed in its own instrument's compass and is *supposed* to be the
    low end — a cello melody over a contrabass shares its register
    legitimately, and a metric that called that a blur would fail a legal
    low-string ensemble. Percussion is out for the usual reason: a drum
    map is not a pitch range.
    """
    return sorted({n.voice_id for n in score.notes if n.voice_id >= VOICE_HARMONY})


def _register_separation(score: NotationScore, melody: list[NoteEvent]) -> float | None:
    """How far the *closest* harmony voice stays from the melody, in semitones.

    The distance between two ranges — zero when they interlock at any
    pitch, however far apart their centres are — and the minimum over the
    harmony voices, so one accompaniment crowding the tune is reported
    however roomy the others are.

    This replaced a mean-melody-minus-mean-harmony figure that could not
    see the thing it is for. Three accompaniment voices piling into the
    melody's octave reported 19.12 semitones of separation because their
    *averages* were low, and a celesta pad written above a piano tune
    reported -17.9 for being high. A range distance reports both as the
    overlap they are.

    None when the piece is melody alone or has no harmony voice.
    """
    if not melody:
        return None
    melody_low = min(n.pitch_midi for n in melody)
    melody_high = max(n.pitch_midi for n in melody)
    distances: list[int] = []
    for voice_id in _harmony_voice_ids(score):
        notes = _voice_notes(score, voice_id)
        if not notes:
            continue
        low = min(n.pitch_midi for n in notes)
        high = max(n.pitch_midi for n in notes)
        distances.append(max(melody_low - high, low - melody_high, 0))
    return float(min(distances)) if distances else None


def _tessitura_overlap(score: NotationScore) -> int | None:
    """Semitones of the melody's band in which the harmony actually sounds.

    A count of distinct pitches rather than of the band's width, because
    the band is not what is heard: an accompaniment whose two extreme
    notes bracket the melody's range shares the whole of it in this
    measure only if it is *sounding* notes there, and a pad folded an
    octave below the tune shares none of it however wide its own compass.

    None when the piece is melody alone or has no harmony voice.
    """
    melody_band = _band(score, {VOICE_MELODY})
    if melody_band is None:
        return None
    harmony_voices = _harmony_voice_ids(score)
    if not harmony_voices:
        return None
    low, high = melody_band
    sounded = {
        n.pitch_midi
        for n in score.notes
        if n.voice_id in harmony_voices and low <= n.pitch_midi <= high
    }
    return len(sounded)


def _band(score: NotationScore, voice_ids: set[int]) -> tuple[int, int] | None:
    """Lowest and highest pitch across `voice_ids`, or None if silent."""
    pitches = [n.pitch_midi for n in score.notes if n.voice_id in voice_ids]
    return (min(pitches), max(pitches)) if pitches else None


def _bass_figures_by_bar(score: NotationScore) -> dict[int, tuple[tuple[int, int], ...]] | None:
    """The bass voice's onset/duration figure for each bar, or None with no bass.

    Onsets are measured from the bar's own start, so the same figure played
    two bars apart is the same figure. Shared by the metric and its
    localisation, so the count a piece is judged by and the bars a finding
    names cannot come apart.
    """
    bass = _voice_notes(score, VOICE_BASS)
    if not bass:
        return None
    layout = _BarLayout.of(score)
    per_bar: dict[int, list[tuple[int, int]]] = {index: [] for index in range(len(layout.starts))}
    for note in bass:
        index = layout.index_of(note.tick)
        if index is not None:
            per_bar[index].append((note.tick - layout.starts[index], note.duration_ticks))
    return {index: tuple(figure) for index, figure in per_bar.items()}


def _bass_onset_patterns(score: NotationScore) -> int | None:
    """Distinct onset/duration figures the bass voice plays, per bar.

    None when the piece has no bass voice. A piece whose bass repeats one
    bar for its whole length scores 1.
    """
    figures = _bass_figures_by_bar(score)
    if figures is None:
        return None
    return len(set(figures.values()))


def _covered_bars(score: NotationScore) -> set[int] | None:
    """The bars a harmony voice holds a note through, or None if it has none.

    A note covers its bar when it sounds for at least half of it — the line
    between a bed and a figure, which is measured rather than chosen: over
    the palette grid a pad clears 0.92-1.00 of a piece's bars and the
    broken-chord figures play through none of them, so nothing sits near
    the half-bar boundary for the bar to be a coin flip about.

    None when the piece has no bars or is melody and bass alone: a solo has
    no bed to sustain. Shared by the metric and its localisation.
    """
    layout = _BarLayout.of(score)
    voices = set(_harmony_voice_ids(score))
    if not layout.starts or not voices:
        return None
    covered: set[int] = set()
    for note in score.notes:
        if note.voice_id not in voices:
            continue
        index = layout.index_of(note.tick)
        if index is not None and note.duration_ticks * 2 >= layout.length_of(index):
            covered.add(index)
    return covered


def _harmony_pad_coverage(score: NotationScore) -> float | None:
    """Share of the bars a harmony voice holds a note through.

    None when there is no bed to sustain, and reporting zero there would
    read as a miss rather than as a metric that does not apply.
    """
    covered = _covered_bars(score)
    if covered is None:
        return None
    return len(covered) / len(score.measures)


def _melody_intervals_by_bar(score: NotationScore) -> list[tuple[int | None, int]]:
    """Each melodic interval with the bar its arrival lands in.

    The arrival and not the departure: an interval is heard where it lands.
    A bar of `None` is an arrival past the last measure — the piece-wide
    metric still counts that interval, and this reading has no bar to put
    it in rather than dropping it silently.
    """
    layout = _BarLayout.of(score)
    onsets = pairwise(_onsets(_voice_notes(score, VOICE_MELODY)))
    return [(layout.index_of(tick), later - earlier) for (_, earlier), (tick, later) in onsets]


def _interval_bars(score: NotationScore, offending: Callable[[int], bool]) -> set[int]:
    """The bars holding a melodic interval `offending` accepts."""
    return {
        bar
        for bar, delta in _melody_intervals_by_bar(score)
        if bar is not None and offending(delta)
    }


def _count_by_bar(layout: _BarLayout, notes: list[NoteEvent]) -> dict[int, int]:
    """How many of `notes` onset in each bar."""
    counts: dict[int, int] = {}
    for note in notes:
        index = layout.index_of(note.tick)
        if index is not None:
            counts[index] = counts.get(index, 0) + 1
    return counts


def _unrecovered_leap_bars(score: NotationScore) -> set[int]:
    """The bars an unanswered leap lands in.

    A leap at the very end of the line is not examined, because
    `_leap_recovery` does not count it as a leap either: the localisation
    has to place the leaps the metric counted and no others.
    """
    placed = _melody_intervals_by_bar(score)
    bars: set[int] = set()
    for (bar, delta), (_, following) in pairwise(placed):
        if bar is None or abs(delta) < LEAP_MIN_SEMITONES:
            continue
        answered = 0 < abs(following) <= STEP_MAX_SEMITONES and following * delta < 0
        if not answered:
            bars.add(bar)
    return bars


def _outcounted_bars(score: NotationScore) -> set[int]:
    """The bars where an accompaniment voice plays more notes than the melody.

    The busiest voice *in that bar* rather than the piece's busiest,
    because this is a bar's own reading: `texture_hierarchy` claims the tune
    leads the busiest thing under it, and that claim inside one bar is made
    against whichever voice leads in that bar.
    """
    layout = _BarLayout.of(score)
    melody = _count_by_bar(layout, _voice_notes(score, VOICE_MELODY))
    per_voice = {
        voice_id: _count_by_bar(layout, _voice_notes(score, voice_id))
        for voice_id in _accompaniment_voice_ids(score)
    }
    return {
        bar
        for bar in range(len(layout.starts))
        if per_voice
        and melody.get(bar, 0) < max(counts.get(bar, 0) for counts in per_voice.values())
    }


def _unsustained_bars(score: NotationScore) -> set[int]:
    """The bars no harmony note holds through."""
    covered = _covered_bars(score)
    if covered is None:
        return set()
    return set(range(len(score.measures))) - covered


def _looping_bass_bars(score: NotationScore) -> set[int]:
    """The bars the bass repeats its commonest figure in.

    `bass_onset_patterns` counts distinct figures, so the bars that answer
    for a low count are the ones sharing the figure the piece falls back on.
    A tie for commonest goes to the figure heard first, which is bar order
    and therefore stable.
    """
    figures = _bass_figures_by_bar(score)
    if figures is None:
        return set()
    commonest, _ = Counter(figures.values()).most_common(1)[0]
    return {bar for bar, figure in figures.items() if figure == commonest}


_LOCALISERS: dict[str, Callable[[NotationScore], set[int]]] = {
    "step_ratio": lambda score: _interval_bars(
        score, lambda delta: abs(delta) > STEP_MAX_SEMITONES
    ),
    "repeat_ratio": lambda score: _interval_bars(score, lambda delta: delta == 0),
    "leap_recovery_ratio": _unrecovered_leap_bars,
    "max_leap_semitones": lambda score: _interval_bars(
        score, lambda delta: abs(delta) > QUALITY_MAX_LEAP_MAX
    ),
    "texture_hierarchy": _outcounted_bars,
    "harmony_pad_coverage": _unsustained_bars,
    "bass_onset_patterns": _looping_bass_bars,
}
"""The metrics whose counted events land in a bar, and how to find them.

The four absent from this table are absent for one reason, which is the
same reason three axes are absent from this module's own docstring: the
metric counts a relation rather than an event, so naming bars for it would
name the bars where the piece is *legal*. `range_semitones` counts
semitones between the whole line's two extreme notes, `distinct_durations`
counts note values, and `register_separation_semitones` and
`tessitura_overlap_semitones` count semitones and pitches across two
voices' ranges.
"""


def _spans(bars: Iterable[int]) -> tuple[BarSpan, ...]:
    """Merge bar indices into runs, numbered the way a reader counts them.

    The score indexes its measures from 0 and a finding is prose read by a
    user, so the conversion happens here, once, rather than at each of the
    seven readings that find the bars.
    """
    runs: list[list[int]] = []
    for index in sorted(set(bars)):
        if runs and index == runs[-1][-1] + 1:
            runs[-1].append(index)
        else:
            runs.append([index])
    return tuple(BarSpan(first_bar=run[0] + 1, last_bar=run[-1] + 1) for run in runs)


def localize(
    score: NotationScore, findings: Iterable[QualityFinding]
) -> tuple[QualityFinding, ...]:
    """Attach to each finding the bars its own counted events sit in.

    Takes the score as well as the findings because a `PieceQuality` is a
    block of numbers: it does not carry the notes, so nothing in it can say
    where the offenders are. A metric with no localiser keeps its finding
    unchanged with empty bars rather than being handed a bar that would be
    an invention.
    """
    localized: list[QualityFinding] = []
    for finding in findings:
        locate = _LOCALISERS.get(finding.metric)
        localized.append(replace(finding, bars=() if locate is None else _spans(locate(score))))
    return tuple(localized)


__all__ = [
    "AXES",
    "LEAP_MIN_SEMITONES",
    "QUALITY_BASS_ONSET_PATTERNS_MIN",
    "QUALITY_DISTINCT_DURATIONS_MIN",
    "QUALITY_HARMONY_PAD_COVERAGE_MIN",
    "QUALITY_LEAP_RECOVERY_MIN",
    "QUALITY_MAX_LEAP_MAX",
    "QUALITY_MELODIC_RANGE_MAX",
    "QUALITY_MELODIC_RANGE_MIN",
    "QUALITY_REGISTER_SEPARATION_MIN",
    "QUALITY_REPEAT_RATIO_MAX",
    "QUALITY_STEP_RATIO_MIN",
    "QUALITY_TESSITURA_OVERLAP_MAX",
    "QUALITY_TEXTURE_HIERARCHY_MIN",
    "QUALITY_THRESHOLDS",
    "STEP_MAX_SEMITONES",
    "Axis",
    "BarSpan",
    "PieceQuality",
    "QualityFinding",
    "QualityReport",
    "QualityThreshold",
    "localize",
    "score_corpus",
    "score_piece",
]
