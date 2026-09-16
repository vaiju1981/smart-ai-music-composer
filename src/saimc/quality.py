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
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise

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


@dataclass(frozen=True)
class QualityThreshold:
    """One bar a piece must clear, with the reason and the remedy."""

    metric: str
    minimum: float | None
    maximum: float | None
    rationale: str
    hint: str

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

    def message(self) -> str:
        """One readable line: what missed, by how much, and what to change."""
        comparator = "<" if self.direction == "min" else ">"
        return (
            f"{self.metric} {self.measured:.2f} {comparator} {self.target:.2f} "
            f"— {self.rationale}. To move it: {self.hint}"
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
                )
            )
        return tuple(found)


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
        step_ratio=_ratio(
            sum(1 for step in moves if abs(step) <= STEP_MAX_SEMITONES), len(moves)
        ),
        repeat_ratio=_ratio(len(intervals) - len(moves), len(intervals)),
        leap_recovery_ratio=_leap_recovery(intervals),
        max_leap_semitones=max((abs(step) for step in moves), default=None),
        range_semitones=_range_semitones(melody),
        distinct_durations=_distinct_durations(melody),
        texture_hierarchy=_texture_hierarchy(score, melody),
        register_separation_semitones=_register_separation(score, melody),
        tessitura_overlap_semitones=_tessitura_overlap(score),
        bass_onset_patterns=_bass_onset_patterns(score),
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


def _intervals(notes: list[NoteEvent]) -> list[int]:
    """Semitone deltas between successive onsets of one voice.

    At a shared tick (a chord in a single voice) the lowest note carries
    the line, so the delta is taken against the previous onset's lowest
    pitch rather than against an arbitrary member of the chord.
    """
    by_tick: dict[int, list[int]] = {}
    for note in notes:
        by_tick.setdefault(note.tick, []).append(note.pitch_midi)
    pitches = [min(by_tick[tick]) for tick in sorted(by_tick)]
    return [b - a for a, b in pairwise(pitches)]


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


def _texture_hierarchy(score: NotationScore, melody: list[NoteEvent]) -> float | None:
    """Melody note count over the busiest accompaniment voice's count.

    Percussion is excluded: a kit voice is a stream of short hits, and
    comparing a melody's note count to a hi-hat's is not a musical claim.
    None when the piece is melody alone.
    """
    accompaniment = {
        voice_id: len(_voice_notes(score, voice_id))
        for voice_id in {n.voice_id for n in score.notes}
        if voice_id not in {VOICE_MELODY, VOICE_PERCUSSION}
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


def _bass_onset_patterns(score: NotationScore) -> int | None:
    """Distinct onset/duration figures the bass voice plays, per bar.

    None when the piece has no bass voice. A piece whose bass repeats one
    bar for its whole length scores 1.
    """
    bass = _voice_notes(score, VOICE_BASS)
    if not bass:
        return None
    layout = _BarLayout.of(score)
    per_bar: dict[int, list[tuple[int, int]]] = {index: [] for index in range(len(layout.starts))}
    for note in bass:
        index = layout.index_of(note.tick)
        if index is None:
            continue
        per_bar[index].append((note.tick - layout.starts[index], note.duration_ticks))
    return len({tuple(figure) for figure in per_bar.values()})


__all__ = [
    "LEAP_MIN_SEMITONES",
    "QUALITY_BASS_ONSET_PATTERNS_MIN",
    "QUALITY_DISTINCT_DURATIONS_MIN",
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
    "PieceQuality",
    "QualityFinding",
    "QualityReport",
    "QualityThreshold",
    "score_corpus",
    "score_piece",
]
