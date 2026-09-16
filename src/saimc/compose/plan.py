"""The composition plan: the surface an agent reads and edits.

`CompositionSpec` has ten fields and none of them can carry a bar-level
decision, so "drums come in earlier" or "the bass walks instead of
pedalling" is inexpressible today — it is a literal inside `motif.py` or
`engine.py`, reachable only by editing the code. The plan is the missing
middle layer: the thing a conductor, a specialist or a user edits, and
the engine reads.

It is what a working tree is to a coding harness — inspectable,
diffable, revertible — and, like the score and the performance plan, a
**canonical artifact**: frozen, versioned, hashed, and stored. That is
what lets "the conductor is free; the artifact is not" be true. An LLM
may write a plan; the engine is deterministic in it, so `(plan, seed) ->
notes` is byte-identical and replayable exactly as `(spec, seed) ->
notes` is.

**Materialized, never a delta.** Every field here is a value, not an
override with the real value left in a module table. A plan that carried
overrides would make `(plan, seed) -> notes` a claim about `motif.py`
rather than about the artifact: edit the table and every stored plan
would replay differently while the manifest still promised
reproducibility. So a stored plan is complete, and a change to one is
`dataclasses.replace`, not a diff to be re-applied later.

**Scope.** The knobs here are the ones that already exist as a named
value in a module `plan.py` is allowed to import — `motif.py`,
`forms.py`, `duration.py`, `percussion.py`, `instruments.py` — plus the
ones `saimc.quality.QUALITY_THRESHOLDS` names in its `hint` fields, which
are precisely the knobs a critic loop can move. Constants that happen to
live in `engine.py` (the harmony velocities, the register clearances, the
tie probability, the modulation offset) join as their layer is wired,
because `engine.py` imports this module and the move has to go the other
way. Derivation constants and internal search parameters are left out on
purpose: `BASS_HIGH_MIDI`, `_WALK_REACH_DEGREES`, `_RANK_RUBBING` and
their kin are not musical decisions anyone has an opinion about.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from saimc.canonical import canonical_sha256
from saimc.compose.duration import (
    ARRANGEMENT_ARC_MIN_REPS,
    DURATION_TOLERANCE,
    INTRO_BARS,
    MAX_REPEATS,
    RITARDANDO_BARS,
    RITARDANDO_FACTOR,
)
from saimc.compose.forms import PHRASE_SIZES, cadence_degree_for
from saimc.compose.motif import (
    BASS_FIGURES,
    CHORD_TONE_DEGREES,
    DEFAULT_BASS_FIGURES,
    DEFAULT_RHYTHM_WEIGHTS,
    LEAP_DEGREES,
    MAX_MOTIF_SPAN_DEGREES,
    MOTIF_OPERATION_WEIGHTS,
    RHYTHM_WEIGHTS,
    STEP_CHOICES,
    STEP_WEIGHTS,
    BassFigure,
)
from saimc.compose.percussion import (
    DRUM_STYLES,
    MOOD_VELOCITY_SCALE,
    ROTATION_CYCLE,
    SECTION_CRASH_VELOCITY,
    style_name_for,
)
from saimc.instruments import LINE_BAND_SEMITONES
from saimc.spec import CompositionSpec

PLAN_SCHEMA_VERSION: Final[int] = 1
"""Bump when the plan's field set changes.

Deliberately not `CANONICAL_FORMAT_VERSION`, which moves only when the
*encoding* rules move (see `saimc.canonical`). The score and the
performance plan tag themselves with that constant, so adding a field to
either would leave its format tag alone and a reader holding an old
document with no way to tell. A plan is new, so it gets the version that
actually tracks its shape.
"""

PLAN_FORMAT_PREFIX: Final[str] = "CompositionPlan"


def _require(condition: bool, message: str) -> None:
    """Raise `PlanError` unless `condition`. Keeps `__post_init__` readable."""
    if not condition:
        raise PlanError(message)


class PlanError(ValueError):
    """A plan that cannot be honoured, and why.

    A `ValueError` because a plan arrives from outside the engine — from a
    conductor, an agent or a user — and a caller has to be able to catch
    it without catching a programming error. The message is the reason:
    the product's rule is that a request the engine cannot honour is
    **refused with a reason**, never silently downgraded to a no-op.
    """


def _weight_pairs(name: str, pairs: tuple[tuple[str, float], ...]) -> None:
    """Validate a named-weight table: non-empty, unique names, weights >= 0."""
    _require(bool(pairs), f"{name} must carry at least one entry")
    seen: set[str] = set()
    for key, weight in pairs:
        _require(bool(key), f"{name} has an entry with an empty name")
        _require(key not in seen, f"{name} names {key!r} twice")
        _require(weight >= 0.0, f"{name}.{key} is negative ({weight})")
        seen.add(key)


@dataclass(frozen=True)
class CompositionPlan:
    """A complete, materialized set of the engine's musical decisions.

    Every field is required, because a field left unset is a field whose
    real value lives in a module table — which is the thing this type
    exists to stop being true. `format` is set by `default_plan` and is
    the document's version tag: present in the canonical form, though not
    its leading key, since `canonical_dumps` sorts and several of this
    document's keys sort before `format`.
    """

    format: str

    # --- Melody: the walk, and the motif operations that vary it --------
    step_choices: tuple[int, ...]
    """The interval vocabulary, in scale degrees, walked by `_draw_step`."""
    step_weights: tuple[float, ...]
    """One weight per `step_choices` entry. Steps dominate; leaping is rare."""
    max_motif_span_degrees: int
    """How far a motif may wander from where it started before it is folded."""
    leap_degrees: int
    """The interval, in degrees, from which a step counts as a leap."""
    chord_tone_degrees: int
    """The interval, in degrees, that transposition moves the anchor by."""
    motif_operation_weights: tuple[tuple[str, float], ...]
    """Weights for the bar-level operation draw, in draw order. The weight
    left over is the ornament's, so these need not sum to 1."""
    rhythm_weights: tuple[tuple[str, float], ...]
    """The mood's rhythmic figures and how often each is chosen."""

    # --- Harmony --------------------------------------------------------
    bass_figures: tuple[BassFigure, ...]
    """The mood's bass vocabulary, most characteristic figure first. A
    figure belongs to a chord slot: the left hand states one for as long
    as its harmony lasts, which is what `bass_onset_patterns` counts."""
    cadence_degree: int
    """The scale degree the final cadence approaches the tonic from."""

    # --- Arrangement ----------------------------------------------------
    form_sizes: tuple[int, ...]
    """The phrase lengths a form may be built from, shortest first."""
    intro_bars: int
    """Bars the melody rests at the top of a long piece's first section."""
    max_repeats: int
    """The most times a form may repeat while filling a duration."""
    duration_tolerance: float
    """How far a realised duration may sit from the one that was asked for."""
    ritardando_factor: float
    """The tempo the close slows to, as a fraction of the piece's."""
    ritardando_bars: int
    """The bars the slowdown is spread over."""
    arc_min_reps: int
    """Repeats from which a piece counts as long, and gets an intro, a
    ritardando and a modulation."""

    # --- Voices ---------------------------------------------------------
    line_band_semitones: int
    """The register window a melody line is written inside. Wide enough to
    hold a tune, narrow enough to leave the accompaniment its own room."""

    # --- Percussion -----------------------------------------------------
    drum_style_name: str | None
    """The chosen kit's name, or None for a piece with no drums (every
    meter but 4/4, 3/4 and 6/8, and the percussion voice is then skipped)."""
    rotation_cycle: tuple[int, ...]
    """Which pattern variant each section takes, cycling by section index."""
    percussion_velocity_scale: float
    """The mood's scaling of every drum hit's velocity."""
    section_crash_velocity: int
    """The velocity of the crash that marks a section downbeat."""

    def __post_init__(self) -> None:
        _require(
            self.format.startswith(f"{PLAN_FORMAT_PREFIX}:"),
            f"format must be {PLAN_FORMAT_PREFIX}:<version>, got {self.format!r}",
        )

        _require(
            len(self.step_choices) == len(self.step_weights),
            "step_choices and step_weights must be the same length "
            f"({len(self.step_choices)} vs {len(self.step_weights)})",
        )
        _require(bool(self.step_choices), "step_choices must not be empty")
        _require(
            all(weight >= 0.0 for weight in self.step_weights),
            "step_weights must not be negative",
        )
        _require(
            any(weight > 0.0 for weight in self.step_weights),
            "step_weights must carry some weight, or no step can be drawn",
        )
        _require(
            self.max_motif_span_degrees > 0, "max_motif_span_degrees must be positive"
        )
        _require(self.leap_degrees > 0, "leap_degrees must be positive")
        _require(self.chord_tone_degrees > 0, "chord_tone_degrees must be positive")
        _weight_pairs("motif_operation_weights", self.motif_operation_weights)
        _weight_pairs("rhythm_weights", self.rhythm_weights)

        _require(bool(self.bass_figures), "bass_figures must carry at least one figure")
        for figure in self.bass_figures:
            _require(bool(figure), "a bass figure must carry at least one note")
        _require(
            0 <= self.cadence_degree <= 6,
            f"cadence_degree is out of the scale ({self.cadence_degree})",
        )

        _require(bool(self.form_sizes), "form_sizes must not be empty")
        _require(
            all(size > 0 for size in self.form_sizes),
            "every form size must be a positive number of bars",
        )
        _require(self.intro_bars >= 0, "intro_bars must not be negative")
        _require(self.max_repeats >= 1, "max_repeats must allow at least one repeat")
        _require(
            0.0 < self.duration_tolerance < 1.0,
            f"duration_tolerance must sit in (0, 1), got {self.duration_tolerance}",
        )
        _require(
            0.0 < self.ritardando_factor <= 1.0,
            "ritardando_factor must sit in (0, 1] — it is a fraction of the tempo",
        )
        _require(self.ritardando_bars >= 0, "ritardando_bars must not be negative")
        _require(self.arc_min_reps >= 1, "arc_min_reps must allow at least one repeat")

        _require(self.line_band_semitones > 0, "line_band_semitones must be positive")

        _require(
            self.drum_style_name is None or self.drum_style_name in DRUM_STYLES,
            f"unknown drum style {self.drum_style_name!r}; known styles are "
            f"{sorted(DRUM_STYLES)}",
        )
        _require(bool(self.rotation_cycle), "rotation_cycle must not be empty")
        _require(
            all(variant >= 0 for variant in self.rotation_cycle),
            "a rotation_cycle entry names a pattern variant, so it cannot be negative",
        )
        _require(
            self.percussion_velocity_scale > 0.0,
            "percussion_velocity_scale must be positive",
        )
        _require(
            self.section_crash_velocity > 0, "section_crash_velocity must be positive"
        )

    def compute_hash(self) -> str:
        """The plan's canonical digest. Two identical plans hash alike."""
        return canonical_sha256(self.to_canonical_dict())

    def to_canonical_dict(self) -> dict[str, Any]:
        """The plan as the canonical JSON document it is stored as.

        Tuples become lists because JSON has no tuple, and the nested
        weight tables become lists of `{name, weight}` objects rather
        than pairs: a pair reads as a two-element array whose meaning a
        consumer has to be told, while a named object does not.
        """
        return {
            "format": self.format,
            "step_choices": list(self.step_choices),
            "step_weights": list(self.step_weights),
            "max_motif_span_degrees": self.max_motif_span_degrees,
            "leap_degrees": self.leap_degrees,
            "chord_tone_degrees": self.chord_tone_degrees,
            "motif_operation_weights": [
                {"operation": name, "weight": weight}
                for name, weight in self.motif_operation_weights
            ],
            "rhythm_weights": [
                {"figure": name, "weight": weight} for name, weight in self.rhythm_weights
            ],
            "bass_figures": [[list(note) for note in figure] for figure in self.bass_figures],
            "cadence_degree": self.cadence_degree,
            "form_sizes": list(self.form_sizes),
            "intro_bars": self.intro_bars,
            "max_repeats": self.max_repeats,
            "duration_tolerance": self.duration_tolerance,
            "ritardando_factor": self.ritardando_factor,
            "ritardando_bars": self.ritardando_bars,
            "arc_min_reps": self.arc_min_reps,
            "line_band_semitones": self.line_band_semitones,
            "drum_style_name": self.drum_style_name,
            "rotation_cycle": list(self.rotation_cycle),
            "percussion_velocity_scale": self.percussion_velocity_scale,
            "section_crash_velocity": self.section_crash_velocity,
        }


def _named_weights(table: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
    """A mood's weight table as the ordered pairs a plan stores."""
    return tuple(table.items())


def default_plan(spec: CompositionSpec) -> CompositionPlan:
    """The plan the engine behaves as today, for this spec.

    Every value is read from the table that held it before the plan
    existed, so this is a *description* of the current engine rather than
    a new set of choices — which is what lets `compose(spec)` keep
    producing byte-identical output while the plan is threaded through it.

    Only the values that vary by mood are `spec`-dependent; the rhythm
    weights and the bass vocabulary are the two that matter, and a mood
    outside the table falls back to the same gentle defaults the engine
    already uses.
    """
    mood = spec.mood.value
    return CompositionPlan(
        format=f"{PLAN_FORMAT_PREFIX}:{PLAN_SCHEMA_VERSION}",
        step_choices=STEP_CHOICES,
        step_weights=STEP_WEIGHTS,
        max_motif_span_degrees=MAX_MOTIF_SPAN_DEGREES,
        leap_degrees=LEAP_DEGREES,
        chord_tone_degrees=CHORD_TONE_DEGREES,
        motif_operation_weights=MOTIF_OPERATION_WEIGHTS,
        rhythm_weights=_named_weights(RHYTHM_WEIGHTS.get(mood, DEFAULT_RHYTHM_WEIGHTS)),
        bass_figures=BASS_FIGURES.get(mood, DEFAULT_BASS_FIGURES),
        cadence_degree=cadence_degree_for(mood),
        form_sizes=PHRASE_SIZES,
        intro_bars=INTRO_BARS,
        max_repeats=MAX_REPEATS,
        duration_tolerance=DURATION_TOLERANCE,
        ritardando_factor=RITARDANDO_FACTOR,
        ritardando_bars=RITARDANDO_BARS,
        arc_min_reps=ARRANGEMENT_ARC_MIN_REPS,
        line_band_semitones=LINE_BAND_SEMITONES,
        drum_style_name=style_name_for(mood, spec.time_signature.value),
        rotation_cycle=ROTATION_CYCLE,
        percussion_velocity_scale=MOOD_VELOCITY_SCALE.get(mood, 1.0),
        section_crash_velocity=SECTION_CRASH_VELOCITY,
    )


def resolve_plan(
    spec: CompositionSpec, plan: CompositionPlan | None = None
) -> CompositionPlan:
    """The plan the engine composes under: `plan`, or this spec's default.

    The seam `compose` calls. Passing `None` resolves to exactly today's
    values, which is what keeps the plan invisible until something
    consumes it.

    There is no second "resolved" type. A `CompositionPlan` is already
    complete and self-validating — its `__post_init__` refuses a plan that
    could not be honoured — so a wrapper carrying the same fields would be
    ceremony, and the spec-derived context the engine needs alongside the
    plan (the meter, the key, the arrangement) is the spec's and the
    arrangement's to hold, not the plan's to duplicate. If a layer turns
    out to need them bundled, that is the moment to add the type, with the
    layer as the evidence.
    """
    if plan is None:
        return default_plan(spec)
    return plan


__all__ = [
    "PLAN_FORMAT_PREFIX",
    "PLAN_SCHEMA_VERSION",
    "CompositionPlan",
    "PlanError",
    "default_plan",
    "resolve_plan",
]
