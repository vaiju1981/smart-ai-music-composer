"""Canonical symbolic artifacts: NotationScore and PerformancePlan.

Per `docs/roadmap.md` §2 and §6:

- `NotationScore` is the engraved surface: pitches, durations, measures,
  dynamics. Drives MusicXML -> OSMD -> SVG/PDF.
- `PerformancePlan` is the playback surface: realized integer-microsecond
  timestamps, velocities, articulations, humanization. Drives MIDI ->
  FluidSynth -> WAV and the piano-roll animation.

Both are canonical artifacts per §6: their canonical-JSON serialization
is byte-identical for a given spec + seed + engine version. Media
artifacts (WAV, SVG, WebM) are derived renderings and are verified
semantically per §8.

Integer types per §6:
- score positions and durations: integer musical ticks (PPQ-resolution).
  Phase 1 uses PPQ=480, so a quarter note = 480 ticks.
- realized performance timestamps: integer microseconds.
- pitches: integer MIDI numbers (0-127).
- velocities: integer 0-127.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from saimc.canonical import (
    CANONICAL_FORMAT_VERSION,
    canonical_sha256,
)

# Pulses per quarter-note. music21's default PPQ is not standardised,
# so we anchor to a known resolution that downstream renderers can
# target. 480 is the SMF standard and gives us 1920 ticks/measure at 4/4,
# enough resolution for the rhythm-density Phase 1 ships.
PPQ: int = 480

# Microseconds per minute. Used to convert bpm -> microseconds per quarter.
MICROSECONDS_PER_MINUTE: int = 60_000_000

# Default MIDI velocity for Phase 1. The composition engine produces
# velocities in [40, 100] for humanized feel; this is the neutral default.
DEFAULT_VELOCITY: int = 64

# Canonical voice IDs. The engine writes bass into voice 0 and melody
# into voice 1; renderers key instrument choice off these. Voice 2 is
# the percussion kit: its `pitch_midi` values are GM percussion keys
# (kick=36, snare=38, ...) and the renderer routes the voice to MIDI
# channel 10. Phase 2's orchestra adds more voices after these.
VOICE_BASS: int = 0
VOICE_MELODY: int = 1
VOICE_PERCUSSION: int = 2


def ticks_to_microseconds(ticks: int, bpm: float, ppq: int = PPQ) -> int:
    """Convert an integer tick count to integer microseconds at `bpm`."""
    if bpm <= 0:
        raise ValueError(f"bpm must be positive; got {bpm}")
    quarter_micros = MICROSECONDS_PER_MINUTE / bpm
    return round(ticks * quarter_micros / ppq)


def microseconds_to_ticks(microseconds: int, bpm: float, ppq: int = PPQ) -> int:
    """Convert integer microseconds to integer ticks at `bpm`."""
    if bpm <= 0:
        raise ValueError(f"bpm must be positive; got {bpm}")
    quarter_micros = MICROSECONDS_PER_MINUTE / bpm
    return round(microseconds * ppq / quarter_micros)


@dataclass(frozen=True)
class NoteEvent:
    """A single note in the NotationScore.

    `pitch_midi` is the integer MIDI number. `velocity` is the default
    velocity (1-127). `tick` is the integer PPQ offset from the start of
    the piece (0-based, ascending). `duration_ticks` is the integer
    PPQ length. `tie` connects to the next event with the same
    `voice_id` + `pitch_midi` if True; otherwise the note ends within
    its own duration.
    """

    voice_id: int
    pitch_midi: int
    tick: int
    duration_ticks: int
    velocity: int = DEFAULT_VELOCITY
    tie: bool = False

    def __post_init__(self) -> None:
        if not 0 <= self.pitch_midi <= 127:
            raise ValueError(f"pitch_midi must be in [0, 127]; got {self.pitch_midi}")
        if not 1 <= self.velocity <= 127:
            raise ValueError(f"velocity must be in [1, 127]; got {self.velocity}")
        if self.duration_ticks <= 0:
            raise ValueError(f"duration_ticks must be positive; got {self.duration_ticks}")
        if self.tick < 0:
            raise ValueError(f"tick must be non-negative; got {self.tick}")


@dataclass(frozen=True)
class Measure:
    """A measure in the NotationScore.

    `start_tick` and `end_tick` are integer PPQ offsets. `time_signature`
    is the spec's time signature (the engine is responsible for ensuring
    `end_tick - start_tick` matches it on every measure).
    """

    index: int
    start_tick: int
    end_tick: int
    time_signature: str  # "4/4" | "3/4" | "6/8" | ...

    def __post_init__(self) -> None:
        if self.start_tick < 0:
            raise ValueError(f"start_tick must be non-negative; got {self.start_tick}")
        if self.end_tick <= self.start_tick:
            raise ValueError(
                f"end_tick must exceed start_tick; got {self.end_tick} <= {self.start_tick}"
            )


@dataclass(frozen=True)
class TempoMap:
    """Realized tempo(s) for the score.

    Phase 1 uses a single tempo for the whole piece; the field is a
    list to leave room for tempo changes without changing the type.
    """

    bpm: float
    ppq: int = PPQ


@dataclass(frozen=True)
class KeySignature:
    """Root note name and mode for the NotationScore."""

    root: str  # "C", "G", "D", ... (matches saimc.spec.WesternKey root)
    mode: Literal["major", "minor"]


@dataclass(frozen=True)
class NotationScore:
    """The canonical engraved-surface artifact.

    `format` is the manifest-document format-version tag for
    `canonical_dumps` compatibility. `hash` is the canonical SHA-256
    of the canonical-JSON serialization; computed lazily by
    `compute_hash()` (not in `__post_init__` to avoid hashing during
    construction).
    """

    format: str
    ppq: int
    key: KeySignature
    time_signature: str
    tempo: TempoMap
    measures: tuple[Measure, ...]
    notes: tuple[NoteEvent, ...]

    @staticmethod
    def make(
        *,
        ppq: int,
        key: KeySignature,
        time_signature: str,
        tempo_bpm: float,
        measures: list[Measure],
        notes: list[NoteEvent],
    ) -> NotationScore:
        return NotationScore(
            format=f"NotationScore:{CANONICAL_FORMAT_VERSION}",
            ppq=ppq,
            key=key,
            time_signature=time_signature,
            tempo=TempoMap(bpm=tempo_bpm, ppq=ppq),
            measures=tuple(measures),
            notes=tuple(notes),
        )

    def total_ticks(self) -> int:
        return self.measures[-1].end_tick if self.measures else 0

    def compute_hash(self) -> str:
        return canonical_sha256(self.to_canonical_dict())

    def to_canonical_dict(self) -> dict[str, object]:
        return {
            "format": self.format,
            "ppq": self.ppq,
            "key": {"root": self.key.root, "mode": self.key.mode},
            "time_signature": self.time_signature,
            "tempo": {"bpm": self.tempo.bpm, "ppq": self.tempo.ppq},
            "measures": [
                {
                    "index": m.index,
                    "start_tick": m.start_tick,
                    "end_tick": m.end_tick,
                    "time_signature": m.time_signature,
                }
                for m in self.measures
            ],
            "notes": [
                {
                    "voice_id": n.voice_id,
                    "pitch_midi": n.pitch_midi,
                    "tick": n.tick,
                    "duration_ticks": n.duration_ticks,
                    "velocity": n.velocity,
                    "tie": n.tie,
                }
                for n in self.notes
            ],
        }


@dataclass(frozen=True)
class PerformanceNoteEvent:
    """A realized note in the PerformancePlan.

    `start_us` and `duration_us` are integer microseconds. Phase 1
    carries one velocity per note; no continuous controllers.
    """

    voice_id: int
    pitch_midi: int
    start_us: int
    duration_us: int
    velocity: int
    tie: bool = False

    def __post_init__(self) -> None:
        if self.start_us < 0:
            raise ValueError(f"start_us must be non-negative; got {self.start_us}")
        if self.duration_us <= 0:
            raise ValueError(f"duration_us must be positive; got {self.duration_us}")
        if not 0 <= self.pitch_midi <= 127:
            raise ValueError(f"pitch_midi must be in [0, 127]; got {self.pitch_midi}")
        if not 1 <= self.velocity <= 127:
            raise ValueError(f"velocity must be in [1, 127]; got {self.velocity}")


@dataclass(frozen=True)
class ControllerEvent:
    """A MIDI controller change in the PerformancePlan.

    `control` is the CC number (7 volume, 10 pan, 11 expression,
    64 sustain pedal, 91 reverb send), `value` the 0-127 value, and
    `start_us` the realized time the change lands on the same integer
    microsecond clock as the notes. `voice_id` selects the MIDI
    channel via the same voice->channel mapping the notes use.
    """

    voice_id: int
    control: int
    value: int
    start_us: int

    def __post_init__(self) -> None:
        if not 0 <= self.control <= 127:
            raise ValueError(f"control must be in [0, 127]; got {self.control}")
        if not 0 <= self.value <= 127:
            raise ValueError(f"value must be in [0, 127]; got {self.value}")
        if self.start_us < 0:
            raise ValueError(f"start_us must be non-negative; got {self.start_us}")


@dataclass(frozen=True)
class PitchBendEvent:
    """A MIDI pitch-wheel movement in the PerformancePlan.

    `bend` is the 14-bit signed MIDI value: 0 is centre, ±8191/8192 is
    the full ±2 semitones the default bend range gives.
    """

    voice_id: int
    start_us: int
    bend: int

    def __post_init__(self) -> None:
        if not -8192 <= self.bend <= 8191:
            raise ValueError(f"bend must be in [-8192, 8191]; got {self.bend}")
        if self.start_us < 0:
            raise ValueError(f"start_us must be non-negative; got {self.start_us}")


@dataclass(frozen=True)
class PerformancePlan:
    """The canonical playback-surface artifact.

    `controllers` and `pitch_bends` carry the expression layer (S4):
    CC11 swells, sustain pedal, and later vibrato. Both are optional
    so plans written before the expression model deserialize
    unchanged.
    """

    format: str
    sample_rate: int
    notes: tuple[PerformanceNoteEvent, ...]
    controllers: tuple[ControllerEvent, ...] = ()
    pitch_bends: tuple[PitchBendEvent, ...] = ()

    @staticmethod
    def make(
        *,
        sample_rate: int,
        notes: list[PerformanceNoteEvent],
        controllers: list[ControllerEvent] | None = None,
        pitch_bends: list[PitchBendEvent] | None = None,
    ) -> PerformancePlan:
        return PerformancePlan(
            format=f"PerformancePlan:{CANONICAL_FORMAT_VERSION}",
            sample_rate=sample_rate,
            notes=tuple(notes),
            controllers=tuple(controllers or ()),
            pitch_bends=tuple(pitch_bends or ()),
        )

    def realized_duration_us(self) -> int:
        if not self.notes:
            return 0
        return max(n.start_us + n.duration_us for n in self.notes)

    def compute_hash(self) -> str:
        return canonical_sha256(self.to_canonical_dict())

    def to_canonical_dict(self) -> dict[str, object]:
        return {
            "format": self.format,
            "sample_rate": self.sample_rate,
            "notes": [
                {
                    "voice_id": n.voice_id,
                    "pitch_midi": n.pitch_midi,
                    "start_us": n.start_us,
                    "duration_us": n.duration_us,
                    "velocity": n.velocity,
                    "tie": n.tie,
                }
                for n in self.notes
            ],
            "controllers": [
                {
                    "voice_id": c.voice_id,
                    "control": c.control,
                    "value": c.value,
                    "start_us": c.start_us,
                }
                for c in self.controllers
            ],
            "pitch_bends": [
                {
                    "voice_id": b.voice_id,
                    "start_us": b.start_us,
                    "bend": b.bend,
                }
                for b in self.pitch_bends
            ],
        }


def realized_duration_seconds(plan: PerformancePlan) -> float:
    """Convenience: realized duration in seconds (float), for §8 tolerance checks."""
    return plan.realized_duration_us() / 1_000_000


def merge_performance_plan(events: Iterable[PerformanceNoteEvent]) -> PerformancePlan:
    """Wrap a sequence of PerformanceNoteEvent into a PerformancePlan at 44.1 kHz.

    The 44.1 kHz sample rate is the Phase 1 default; Phase 2 may add
    a config knob if we want higher fidelity. FluidSynth will resample
    the synthesized audio to whatever the output container requires.
    """
    return PerformancePlan.make(sample_rate=44100, notes=list(events))


def merge_notation_score(events: Iterable[NoteEvent]) -> tuple[NoteEvent, ...]:
    """Sort a sequence of NoteEvents by (tick, voice_id, pitch_midi).

    Used by the engine to produce deterministically-ordered output. Per
    §6 canonical-serialization rules, arrays stay in musically significant
    order with deterministic secondary sorting.
    """
    return tuple(sorted(events, key=lambda n: (n.tick, n.voice_id, n.pitch_midi)))


__all__ = [
    "DEFAULT_VELOCITY",
    "MICROSECONDS_PER_MINUTE",
    "PPQ",
    "ControllerEvent",
    "KeySignature",
    "Measure",
    "NotationScore",
    "NoteEvent",
    "PerformanceNoteEvent",
    "PerformancePlan",
    "PitchBendEvent",
    "TempoMap",
    "merge_notation_score",
    "merge_performance_plan",
    "microseconds_to_ticks",
    "realized_duration_seconds",
    "ticks_to_microseconds",
]
