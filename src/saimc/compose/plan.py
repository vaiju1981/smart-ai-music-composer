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
modulation offset) either move to the module that owns the value or join
here as their layer is wired, because `engine.py` imports this module and
the import cannot go the other way. Derivation constants and internal
search parameters are left out on purpose: `BASS_HIGH_MIDI`,
`_WALK_REACH_DEGREES`, `_RANK_RUBBING` and their kin are not musical
decisions anyone has an opinion about.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

from saimc.canonical import canonical_sha256
from saimc.compose.duration import (
    ARRANGEMENT_ARC_MIN_REPS,
    DURATION_TOLERANCE,
    HARMONY_TEXTURE_CYCLE,
    HARMONY_TEXTURE_GROUPS,
    INTRO_BARS,
    MAX_REPEATS,
    RITARDANDO_BARS,
    RITARDANDO_FACTOR,
    SECTION_VELOCITY_FINAL,
    SECTION_VELOCITY_MIDDLE,
    SECTION_VELOCITY_OPENING,
    SECTION_VELOCITY_PEAK,
    ArrangementKnobs,
    SectionArc,
)
from saimc.compose.forms import (
    DEFAULT_SECTION_CLOSE,
    MODULATION_OFFSET,
    PHRASE_SIZES,
    SECTION_CLOSES,
    cadence_degree_for,
    cadence_seventh_for,
    key_pool_for,
)
from saimc.compose.motif import (
    BASS_FIGURES,
    CHORD_TONE_DEGREES,
    DEFAULT_APEX_POSITION,
    DEFAULT_BASS_FIGURES,
    DEFAULT_RHYTHM_WEIGHTS,
    DEFAULT_TIE_PROBABILITY,
    LEAP_DEGREES,
    MAX_MOTIF_SPAN_DEGREES,
    MOTIF_OPERATION_WEIGHTS,
    RHYTHM_WEIGHTS,
    STEP_CHOICES,
    STEP_WEIGHTS,
    TIE_PROBABILITY,
    BassFigure,
    MelodyShape,
)
from saimc.compose.percussion import (
    DRUM_STYLES,
    MOOD_VELOCITY_SCALE,
    PERCUSSION_REST_SECTION,
    ROTATION_CYCLE,
    SECTION_CRASH_VELOCITY,
    DrumKit,
    style_name_for,
)
from saimc.compose.voices import (
    BROKEN_CHORD_MOODS,
    HARMONY_ARPEGGIO_STEP_TICKS,
    HARMONY_ARPEGGIO_VELOCITY,
    HARMONY_MELODY_CLEARANCE,
    HARMONY_PAD_VELOCITY,
    HARMONY_STAB_VELOCITY,
    HarmonyVoices,
)
from saimc.instruments import LINE_BAND_SEMITONES
from saimc.spec import CompositionSpec, WesternKey

PLAN_SCHEMA_VERSION: Final[int] = 5
"""Bump when the plan's field set changes.

Deliberately not `CANONICAL_FORMAT_VERSION`, which moves only when the
*encoding* rules move (see `saimc.canonical`). The score and the
performance plan tag themselves with that constant, so adding a field to
either would leave its format tag alone and a reader holding an old
document with no way to tell. A plan is new, so it gets the version that
actually tracks its shape.
"""

PLAN_FORMAT_PREFIX: Final[str] = "CompositionPlan"

PLAN_FORMAT: Final[str] = f"{PLAN_FORMAT_PREFIX}:{PLAN_SCHEMA_VERSION}"
"""The tag every plan document carries, in §6's `{kind}:{version}` form.

A constant rather than a literal in two places, because the writer and
the reader are the pair that has to agree on it: `default_plan` stamps it
and `from_canonical_dict` refuses anything else, so building it twice
would be two chances to disagree.
"""


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


class UnsupportedPlanVersionError(PlanError):
    """A stored plan document this build cannot read.

    Distinct from a plain `PlanError` because the situation is different
    from a plan the engine refused: this one is *unreadable*, so the reader
    has to say which container it was found in and what to do about it.
    The message carries that; the class does not, because the same
    document found in a job directory or a session directory is the same
    refusal with a different place to look.

    It lives beside `CompositionPlan` rather than beside either reader,
    for the reason this repo keeps re-learning: a value reached through a
    module that merely re-exports it is coupled to that module. Both
    `jobs/storage.py` and `session/models.py` raise this, which is the
    evidence that neither of them owns it.
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


_KEY_NAMES: Final[frozenset[str]] = frozenset(key.value for key in WesternKey)
"""Every key name a pool may contain, read off the type rather than listed.

A pool names keys as strings so the plan document stays flat, which means
the document can carry a name `WesternKey` does not admit — and the
engine would then refuse it as an unknown root, at compose time, in
whatever mood the piece happened to be in. Derived from the enum rather
than written out, so widening the key vocabulary cannot leave this list
behind; a second literal would be a coverage hole per new key."""


def _key_pool(pool: tuple[str, ...]) -> None:
    """Validate a key pool: non-empty, no repeats, every name a real key."""
    _require(bool(pool), "key_pool must name at least one key")
    _require(
        len(set(pool)) == len(pool),
        f"key_pool names a key twice, so the seed walks it unevenly ({', '.join(pool)})",
    )
    unknown = [name for name in pool if name not in _KEY_NAMES]
    _require(not unknown, f"key_pool names keys that are not keys: {unknown}")


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
    tie_probability: float
    """How readily a repeated pitch is tied across a bar line.

    Cross-bar ties were a mood lookup inside `_generate_section`
    (`TIE_PROBABILITY.get(mood, 0.25)`) — real musical values, but not a
    value a plan could state. The table now lives in `motif.py` and the
    plan carries the resolved probability.
    """
    apex_position: float
    """Where the section's apex bar sits, as a fraction of its bars.

    It was the literal `0.6` in `_generate_section`'s apex placement.
    The bar is clamped so the apex can never coincide with the section's
    final bar, which is why the plan bounds this below 1.0 rather than
    leaving the clamp to absorb it.
    """
    line_band_semitones: int
    """The register window a melody line is written inside. Wide enough to
    hold a tune, narrow enough to leave the accompaniment its own room.

    It belongs to the melody rather than to the voices: it is the window
    the *tune* is written in, not a harmony voice's placement, and the
    bed settles under it (`_settle_harmony_register`) rather than the
    other way round.
    """

    # --- Harmony --------------------------------------------------------
    key_pool: tuple[str, ...]
    """The keys this piece may be written in, when the spec names none.

    A spec that names a key gets it and this is dead. A spec that does not
    used to get C major, at every mood, duration, ensemble and seed — the
    parser leaves the key unset by default, so "engine chooses" named a
    constant, and a large share of generated pieces were in C. Measured
    over 64 seeds of each of the three moods: one answer.

    **The plan carries the pool and not the key**, because the plan is
    seed-independent by construction — the arbiter's sixth element exists
    only because two candidates of one fan-out share a plan hash — so a
    chosen key stored here would put the seed back into the plan and make
    that element cover a case nothing reaches. The engine picks
    `pool[seed % len(pool)]` at compose time instead, which is also what
    makes a fan-out's consecutive seeds land on different keys rather than
    rolling against the same one.

    Each mood's pool is a taste call, not a finding: a 225-cell sweep
    (every key `WesternKey` admits, against three moods and three
    duration, ensemble and seed points) found every one of them composes
    clean, so no key is this engine's to refuse. The pools themselves live
    in `forms.py` beside the key arithmetic, and the plan reads one per
    mood through `key_pool_for` the way it reads a cadence.
    """
    bass_figures: tuple[BassFigure, ...]
    """The mood's bass vocabulary, most characteristic figure first. A
    figure belongs to a chord slot: the left hand states one for as long
    as its harmony lasts, which is what `bass_onset_patterns` counts."""
    bass_root_motion: bool
    """Whether the left hand lands on the chord's *root* at each change.

    The walk used to land on the chord tone nearest the line it joined,
    which is what makes it move stepwise through inversions — and what
    leaves the bar's harmony unstated. Measured over 162 pieces (three
    moods, three durations, three ensembles, six seeds) the landing tone
    is the root at 32% of the 2322 chord changes and does not move at all
    at 43%: the left hand is a counter-line, and the bar's chord is
    asserted by the right hand alone. Set, the walk lands on the root
    instead — in the octave nearest the line, so it still moves by the
    smallest step the harmony allows — and a change of chord is audible
    where it happens.

    Off is the pre-plan behaviour, and the two differ in nothing but which
    of the chord's tones the landing may choose from: `chord_bars` and
    `bar_keys` read identically under either policy. What it is *not* is a
    change to the bass alone, which is what this field first claimed. The
    melody reads the bar's sounding bass — `_melody_bar` refuses a
    candidate that would rub against it — so a bass that moves where the
    harmony moves carries the tune with it, and `_settle_harmony_register`
    moves the bed under the tune in turn. Over the same matrix the bass
    moves in every one of the 162 pieces, the tune in 44 and the bed in
    33. The kit's written notes never move; its realized fills did, with the
    tune's note count, through the humanization stream it shared with the
    melodic voices — a coupling that could not be attributed, and that Phase
    F2a ended by giving the kit a stream of its own.
    """
    cadence_degree: int
    """The scale degree the final cadence approaches the tonic from."""
    cadence_seventh: bool
    """Whether the cadence chord is a seventh, as electrifying's V7 is.

    The other half of the cadence: it was `mood == "electrifying"` inline,
    so a plan could not state a cadence without the mood deciding half of
    it. A value now, so `(plan, seed) -> notes` holds for the cadence as
    it does for everything else the plan carries.
    """
    section_close: str
    """How a section that is not the piece's last one closes.

    One of `forms.SECTION_CLOSES`. Measured before it was a plan value: over
    108 pieces the engine wrote 384 interior sections and not one closed —
    each ended with the template's last chord sounding into the final bar,
    so a piece had exactly one cadence, at its end. The default is `half`, a
    ii-V that points home and resolves across the section boundary; `hold`
    is that older behaviour, kept as a legal value; `full` gives every
    section the plan's own final cadence.

    The piece's *last* section and the coda are not governed by this: they
    always close with `cadence_degree` and `cadence_seventh`, because a
    piece that does not end at home is a different request than this
    vocabulary has a name for. This field is read at one call site, the
    interior-section branch, which is why the closed vocabulary matters —
    a name outside it would be a section closing some third way no test
    looks at.
    """
    harmonic_rhythm: tuple[int, ...] | None
    """The chord durations a template's progression is re-cut into, in bars.

    A pattern rather than a rate, and it cycles: `(1,)` walks the progression
    a chord a bar, `(4,)` holds each for four, and `(2, 1, 1)` is both faster
    than the default two-bar slot and *varied*, which is the thing
    `harmonic_rhythm_variety` measures. `apply_harmonic_rhythm` cuts it into
    the template before the close rewrites the last two bars, so the cadence
    a section ends on survives whatever the pattern says.

    **A one-bar pattern is legal here and refused by the engine on any piece
    with a coda**, and both halves of that are deliberate. It is a pulse at
    any rate, so it reads 0.0 on `harmonic_rhythm_variety` — measured, and
    the reason the vocabulary's "faster" is `(2, 1, 1)`. And it is the only
    pattern that makes `_truncate_template_for_coda`'s forced tone sound as a
    bar of its own, where the bass walk writes a non-chord tone: composed, it
    raises `lint_failed` rather than producing a piece. A plan is thus a
    document that can be *read* and cannot be *honoured* for this one value,
    which is the plan's own rule — the refusal is named, and the tool surface
    carries it to the user per candidate — but it is a fact a writer of this
    field has to know, so it is here rather than in the phase that found it.
    See `test_a_one_bar_pattern_is_refused_where_the_codas_forced_tonic_sounds`.

    **`None` is this plan's one deliberate deferral, and it is structural.**
    It means "the template's own rhythm" — the rate the hand-coded
    progression already has — and it cannot be materialized into a tuple the
    way every other field here is, because *which* template a piece uses is
    the duration search's to decide at compose time: it picks the form, the
    repetition count and the variant, and the template is then truncated or
    extended to fit. Measured: `sleep`'s two eight-bar templates are `(2, 2,
    2, 2)` and `(4, 4)`, so no single tuple stands for the default even
    within one mood. The degrees, the slot colours and the bar counts are
    deferred the same way and for the same reason; what this field adds is
    that a piece whose rhythm was *asked for* replays it from the plan rather
    than from `forms.py`, the way a piece whose key was asked for reads it
    from the spec.
    """
    modulation_offset: int
    """Semitones the final repetition of a long piece is lifted by.

    `MODULATION_OFFSET`, which lived in `engine.py` and therefore joins the
    plan here, as its layer is wired. Zero is a legal choice — a long piece
    that returns home instead of lifting — it is simply not this engine's.
    """

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

    # --- Sections -------------------------------------------------------
    section_energy_opening: float
    """The first section's dynamic, as a fraction of the piece's own."""
    section_energy_peak: float
    """The penultimate section's dynamic — the piece's loudest step."""
    section_energy_final: float
    """The last section's dynamic, which settles so the cadence lands."""
    section_energy_middle: float
    """Every other section's dynamic. The piece plays at its own level."""
    harmony_texture_cycle: tuple[str, ...]
    """Which harmony voices sound in each phase of the long-piece arc.

    One of `first`, `rest` or `all`: the leading voice alone, every voice
    but it, or the full section. Groups rather than voice indices, because
    how many harmony voices a piece has is the ensemble's to decide.
    """
    percussion_rest_section: int
    """The section a long piece's kit rests for, counting from zero."""

    # --- Voices: how the accompaniment under the tune is written ---------
    harmony_broken_chord: bool
    """Whether the leading harmony layer states a broken chord.

    It was `mood == "electrifying"` inline in `_generate_harmony_section`,
    so the texture was the mood's to decide and a critic could not ask
    for it: this is the knob that answers `texture_hierarchy`, which
    measures the accompaniment's note count against the melody's.
    """
    harmony_arpeggio_step_ticks: int
    """The interval the broken-chord figure steps on, an eighth by default.

    `PPQ // 2` inline, and the reason `texture_hierarchy` can miss: the
    figure's onsets per bar are `ticks_per_bar // this`, so the value is
    the accompaniment's density rather than a tempo or a note length.
    """
    harmony_pad_velocity: int
    """The sustained bed's level: the quietest of the three textures."""
    harmony_arpeggio_velocity: int
    """The broken-chord figure's level, a notch above the pad's."""
    harmony_stab_velocity: int
    """A stabbed chord's level: the loudest, because it is an accent."""
    harmony_melody_clearance: int
    """How far under the melody the bed is held.

    Read by two passes — `_melody_band_for`, which raises the tune until
    the room exists, and `_settle_harmony_register`, which places each
    voice against the finished tune. Below the crowding window's width
    the two registers stop being disjoint, which is what
    `register_separation_semitones` and `tessitura_overlap_semitones`
    measure.
    """

    # --- Percussion ------------------------------------------------------
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
        _require(
            0.0 <= self.tie_probability <= 1.0,
            "tie_probability is a probability, so it must sit in [0, 1]; "
            f"got {self.tie_probability}",
        )
        # The apex bar is clamped to the section's penultimate bar, so a
        # position of 1.0 would not move the apex past the final bar — it
        # would silently land it where the clamp already put it. The bound
        # refuses the value that cannot mean anything rather than
        # accepting it and behaving as if it were smaller.
        _require(
            0.0 < self.apex_position < 1.0,
            "apex_position is a fraction of the section, and the apex may "
            f"not be its final bar, so it must sit in (0, 1); got {self.apex_position}",
        )

        _key_pool(self.key_pool)
        _require(bool(self.bass_figures), "bass_figures must carry at least one figure")
        for figure in self.bass_figures:
            _require(bool(figure), "a bass figure must carry at least one note")
        _require(
            0 <= self.cadence_degree <= 6,
            "cadence_degree is out of the scale, which runs 0..6 "
            f"({self.cadence_degree})",
        )
        # A closed vocabulary, refused by name: the value reaches the notes
        # through `apply_section_close`, which raises on a name outside it —
        # but that raise would arrive from inside a composition, where a
        # plan is the thing that should have refused it, and the engine's
        # four callers would each have to handle it.
        _require(
            self.section_close in SECTION_CLOSES,
            f"unknown section close {self.section_close!r}; the closes are "
            f"{', '.join(SECTION_CLOSES)}.",
        )
        _require(
            self.harmonic_rhythm is None
            or (bool(self.harmonic_rhythm) and all(bars >= 1 for bars in self.harmonic_rhythm)),
            "harmonic_rhythm is a pattern of chord durations in bars, each at least "
            f"one, or None for the templates' own rhythm; got {self.harmonic_rhythm}",
        )
        _require(
            abs(self.modulation_offset) <= 12,
            "modulation_offset is a lift of the key, so it cannot exceed an "
            f"octave ({self.modulation_offset})",
        )

        _require(bool(self.form_sizes), "form_sizes must not be empty")
        _require(
            all(size > 0 for size in self.form_sizes),
            "every form size must be a positive number of bars",
        )
        _require(self.intro_bars >= 0, "intro_bars must not be negative")
        # The intro is carved from the first section, so it has to be
        # shorter than the shortest form the arrangement may pick. Without
        # this the failure surfaces as a bare ValueError from inside the
        # duration search rather than as the refusal this type exists to
        # give — the plan is where both numbers are known.
        _require(
            self.intro_bars < min(self.form_sizes),
            f"intro_bars ({self.intro_bars}) must be shorter than the shortest form "
            f"({min(self.form_sizes)}), or there is no section left to carve it from",
        )
        _require(self.max_repeats >= 1, "max_repeats must allow at least one repeat")
        _require(
            self.max_repeats <= MAX_REPEATS,
            f"max_repeats may not exceed {MAX_REPEATS} — §10 #10 caps how often a "
            f"source section may repeat, and the plan may lower that cap, not raise it "
            f"(got {self.max_repeats})",
        )
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

        for name in (
            "section_energy_opening",
            "section_energy_peak",
            "section_energy_final",
            "section_energy_middle",
        ):
            _require(getattr(self, name) > 0.0, f"{name} is a dynamic, so it must be positive")
        _require(bool(self.harmony_texture_cycle), "harmony_texture_cycle must not be empty")
        for group in self.harmony_texture_cycle:
            _require(
                group in HARMONY_TEXTURE_GROUPS,
                f"unknown harmony texture {group!r}; the groups are "
                f"{sorted(HARMONY_TEXTURE_GROUPS)}",
            )
        # The rest is a section index. A value past the piece's last section
        # rests nothing rather than misplacing the rest, so the plan only has
        # to refuse the one that cannot mean anything.
        _require(
            self.percussion_rest_section >= 0,
            "percussion_rest_section is a section index, so it cannot be negative",
        )

        _require(self.line_band_semitones > 0, "line_band_semitones must be positive")

        _require(
            self.harmony_arpeggio_step_ticks > 0,
            "harmony_arpeggio_step_ticks is the interval the figure steps on, "
            f"so it must be positive; got {self.harmony_arpeggio_step_ticks}",
        )
        for name in (
            "harmony_pad_velocity",
            "harmony_arpeggio_velocity",
            "harmony_stab_velocity",
        ):
            velocity = getattr(self, name)
            # A velocity is a MIDI value, so it lives in 1..127: nothing at
            # the bottom is a note that never sounds, and nothing at the top
            # is a number the port drops. Both are refused here rather than
            # clamped silently later, which is the rule for a delta the
            # engine cannot honour.
            _require(
                1 <= velocity <= 127,
                f"{name} is a MIDI velocity, so it must sit in 1..127; got {velocity}",
            )
        _require(
            self.harmony_melody_clearance > 0,
            "harmony_melody_clearance is a distance between two registers, so it "
            f"must be positive; got {self.harmony_melody_clearance}",
        )

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
            1 <= self.section_crash_velocity <= 127,
            "section_crash_velocity is a MIDI velocity, so it must sit in 1..127; "
            f"got {self.section_crash_velocity}",
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
            "tie_probability": self.tie_probability,
            "apex_position": self.apex_position,
            "line_band_semitones": self.line_band_semitones,
            "key_pool": list(self.key_pool),
            "bass_figures": [[list(note) for note in figure] for figure in self.bass_figures],
            "bass_root_motion": self.bass_root_motion,
            "cadence_degree": self.cadence_degree,
            "cadence_seventh": self.cadence_seventh,
            "section_close": self.section_close,
            "harmonic_rhythm": (
                None if self.harmonic_rhythm is None else list(self.harmonic_rhythm)
            ),
            "modulation_offset": self.modulation_offset,
            "form_sizes": list(self.form_sizes),
            "intro_bars": self.intro_bars,
            "max_repeats": self.max_repeats,
            "duration_tolerance": self.duration_tolerance,
            "ritardando_factor": self.ritardando_factor,
            "ritardando_bars": self.ritardando_bars,
            "arc_min_reps": self.arc_min_reps,
            "section_energy_opening": self.section_energy_opening,
            "section_energy_peak": self.section_energy_peak,
            "section_energy_final": self.section_energy_final,
            "section_energy_middle": self.section_energy_middle,
            "harmony_texture_cycle": list(self.harmony_texture_cycle),
            "percussion_rest_section": self.percussion_rest_section,
            "harmony_broken_chord": self.harmony_broken_chord,
            "harmony_arpeggio_step_ticks": self.harmony_arpeggio_step_ticks,
            "harmony_pad_velocity": self.harmony_pad_velocity,
            "harmony_arpeggio_velocity": self.harmony_arpeggio_velocity,
            "harmony_stab_velocity": self.harmony_stab_velocity,
            "harmony_melody_clearance": self.harmony_melody_clearance,
            "drum_style_name": self.drum_style_name,
            "rotation_cycle": list(self.rotation_cycle),
            "percussion_velocity_scale": self.percussion_velocity_scale,
            "section_crash_velocity": self.section_crash_velocity,
        }

    @classmethod
    def from_canonical_dict(cls, payload: Mapping[str, Any]) -> CompositionPlan:
        """Rebuild a plan from the document `to_canonical_dict` writes.

        The `format` tag is read first and refused unless it names exactly
        the version this build writes. §6 requires an explicit version in
        every canonical document, and a plan is the one document whose
        version tracks its *shape*: an older one is missing fields a
        materialized plan cannot do without, and a newer one may carry a
        knob this engine would silently ignore — which is the failure §6
        exists to prevent, because the plan is the artifact the
        determinism claim is made about. Refusing exactly, rather than
        only refusing versions newer than this build, is what makes that
        hold in both directions.

        Every other value is read as written: no field is optional and
        none falls back to a module table. A fallback would make the
        reloaded plan a different plan from the one that was stored,
        which is the whole thing "materialized, never a delta" rules out.
        """
        document_format = str(payload.get("format", ""))
        if document_format != PLAN_FORMAT:
            raise PlanError(
                f"plan document names format {document_format!r}, but this build writes "
                f"and understands {PLAN_FORMAT!r}; a plan is stored materialized, so an "
                "older document is missing fields this build cannot supply and a newer "
                "one may carry values it would silently ignore"
            )
        return cls(
            format=PLAN_FORMAT,
            step_choices=tuple(payload["step_choices"]),
            step_weights=tuple(payload["step_weights"]),
            max_motif_span_degrees=payload["max_motif_span_degrees"],
            leap_degrees=payload["leap_degrees"],
            chord_tone_degrees=payload["chord_tone_degrees"],
            motif_operation_weights=tuple(
                (entry["operation"], entry["weight"])
                for entry in payload["motif_operation_weights"]
            ),
            rhythm_weights=tuple(
                (entry["figure"], entry["weight"]) for entry in payload["rhythm_weights"]
            ),
            tie_probability=payload["tie_probability"],
            apex_position=payload["apex_position"],
            line_band_semitones=payload["line_band_semitones"],
            key_pool=tuple(payload["key_pool"]),
            bass_figures=tuple(
                tuple(tuple(note) for note in figure) for figure in payload["bass_figures"]
            ),
            bass_root_motion=payload["bass_root_motion"],
            cadence_degree=payload["cadence_degree"],
            cadence_seventh=payload["cadence_seventh"],
            section_close=payload["section_close"],
            harmonic_rhythm=(
                None
                if payload["harmonic_rhythm"] is None
                else tuple(payload["harmonic_rhythm"])
            ),
            modulation_offset=payload["modulation_offset"],
            form_sizes=tuple(payload["form_sizes"]),
            intro_bars=payload["intro_bars"],
            max_repeats=payload["max_repeats"],
            duration_tolerance=payload["duration_tolerance"],
            ritardando_factor=payload["ritardando_factor"],
            ritardando_bars=payload["ritardando_bars"],
            arc_min_reps=payload["arc_min_reps"],
            section_energy_opening=payload["section_energy_opening"],
            section_energy_peak=payload["section_energy_peak"],
            section_energy_final=payload["section_energy_final"],
            section_energy_middle=payload["section_energy_middle"],
            harmony_texture_cycle=tuple(payload["harmony_texture_cycle"]),
            percussion_rest_section=payload["percussion_rest_section"],
            harmony_broken_chord=payload["harmony_broken_chord"],
            harmony_arpeggio_step_ticks=payload["harmony_arpeggio_step_ticks"],
            harmony_pad_velocity=payload["harmony_pad_velocity"],
            harmony_arpeggio_velocity=payload["harmony_arpeggio_velocity"],
            harmony_stab_velocity=payload["harmony_stab_velocity"],
            harmony_melody_clearance=payload["harmony_melody_clearance"],
            drum_style_name=payload["drum_style_name"],
            rotation_cycle=tuple(payload["rotation_cycle"]),
            percussion_velocity_scale=payload["percussion_velocity_scale"],
            section_crash_velocity=payload["section_crash_velocity"],
        )

    def arrangement_knobs(self) -> ArrangementKnobs:
        """The arrangement layer, as the struct `duration.py` reads.

        The plan cannot hand `duration.py` the plan: this module imports
        that one for its defaults, so the dependency cannot also run the
        other way. This struct is the one-way bridge, and the arrangement
        layer is the first to need one.
        """
        return ArrangementKnobs(
            form_sizes=self.form_sizes,
            max_repeats=self.max_repeats,
            duration_tolerance=self.duration_tolerance,
            arc_min_reps=self.arc_min_reps,
            intro_bars=self.intro_bars,
            ritardando_factor=self.ritardando_factor,
            ritardando_bars=self.ritardando_bars,
        )

    def section_arc(self) -> SectionArc:
        """The sections layer, as the struct `engine.py` reads.

        The second one-way bridge, on the same terms as `arrangement_knobs`.
        """
        return SectionArc(
            energy_opening=self.section_energy_opening,
            energy_peak=self.section_energy_peak,
            energy_final=self.section_energy_final,
            energy_middle=self.section_energy_middle,
            texture_cycle=self.harmony_texture_cycle,
        )

    def melody_shape(self) -> MelodyShape:
        """The melody layer, as the struct `motif.py` and `engine.py` read.

        The third one-way bridge. The melody is the layer with the most
        readers and the deepest call stack — `_draw_step`, `vary_motif`,
        `apply_rhythm`, `_bent_step`, `_walk_shape` and their kin are
        leaves that take no plan — so the seven values they share travel
        as one argument rather than seven.
        """
        return MelodyShape(
            step_choices=self.step_choices,
            step_weights=self.step_weights,
            max_motif_span_degrees=self.max_motif_span_degrees,
            leap_degrees=self.leap_degrees,
            chord_tone_degrees=self.chord_tone_degrees,
            motif_operation_weights=self.motif_operation_weights,
            rhythm_weights=self.rhythm_weights,
            tie_probability=self.tie_probability,
            apex_position=self.apex_position,
            line_band_semitones=self.line_band_semitones,
        )


    def harmony_voices(self) -> HarmonyVoices:
        """The voices layer, as the struct `engine.py` reads.

        The fourth one-way bridge, on the same terms as the other three.
        Unlike them it is not the plan's own imports that force a struct
        — `voices.py` is importable from here — but the call stack: two
        of these values are read inside `_generate_harmony_section` and
        forwarded there through `_generate_section`, both of which take
        no plan.
        """
        return HarmonyVoices(
            broken_chord=self.harmony_broken_chord,
            arpeggio_step_ticks=self.harmony_arpeggio_step_ticks,
            pad_velocity=self.harmony_pad_velocity,
            arpeggio_velocity=self.harmony_arpeggio_velocity,
            stab_velocity=self.harmony_stab_velocity,
            melody_clearance=self.harmony_melody_clearance,
        )

    def drum_kit(self) -> DrumKit:
        """The percussion layer, as the struct `engine.py` reads.

        The fifth one-way bridge, and the only one whose plan field is a
        *name*: a `DrumStyle` is a table of bar templates, which no
        canonical document can carry. So the plan stores the name the
        meter and mood resolve to (`style_name_for`) and this is where it
        becomes the style, with the three values beside it.
        """
        return DrumKit(
            style=DRUM_STYLES.get(self.drum_style_name) if self.drum_style_name else None,
            rotation_cycle=self.rotation_cycle,
            velocity_scale=self.percussion_velocity_scale,
            crash_velocity=self.section_crash_velocity,
        )


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
        format=PLAN_FORMAT,
        step_choices=STEP_CHOICES,
        step_weights=STEP_WEIGHTS,
        max_motif_span_degrees=MAX_MOTIF_SPAN_DEGREES,
        leap_degrees=LEAP_DEGREES,
        chord_tone_degrees=CHORD_TONE_DEGREES,
        motif_operation_weights=MOTIF_OPERATION_WEIGHTS,
        rhythm_weights=_named_weights(RHYTHM_WEIGHTS.get(mood, DEFAULT_RHYTHM_WEIGHTS)),
        tie_probability=TIE_PROBABILITY.get(mood, DEFAULT_TIE_PROBABILITY),
        apex_position=DEFAULT_APEX_POSITION,
        line_band_semitones=LINE_BAND_SEMITONES,
        key_pool=key_pool_for(mood),
        bass_figures=BASS_FIGURES.get(mood, DEFAULT_BASS_FIGURES),
        # The first plan value that is not a description of the engine as
        # it behaved before the plan existed. Every other field here was
        # read out of a table so that `compose(spec)` would stay
        # byte-identical while the plan was threaded through; this one is
        # a change to the music, made deliberately and carried as a knob
        # so a critic or a user can turn it the other way.
        bass_root_motion=True,
        cadence_degree=cadence_degree_for(mood),
        cadence_seventh=cadence_seventh_for(mood),
        section_close=DEFAULT_SECTION_CLOSE,
        # The templates' own rhythm, which is the only deferral in this
        # document and cannot be avoided — see the field's own docstring.
        harmonic_rhythm=None,
        modulation_offset=MODULATION_OFFSET,
        form_sizes=PHRASE_SIZES,
        intro_bars=INTRO_BARS,
        max_repeats=MAX_REPEATS,
        duration_tolerance=DURATION_TOLERANCE,
        ritardando_factor=RITARDANDO_FACTOR,
        ritardando_bars=RITARDANDO_BARS,
        arc_min_reps=ARRANGEMENT_ARC_MIN_REPS,
        section_energy_opening=SECTION_VELOCITY_OPENING,
        section_energy_peak=SECTION_VELOCITY_PEAK,
        section_energy_final=SECTION_VELOCITY_FINAL,
        section_energy_middle=SECTION_VELOCITY_MIDDLE,
        harmony_texture_cycle=HARMONY_TEXTURE_CYCLE,
        percussion_rest_section=PERCUSSION_REST_SECTION,
        harmony_broken_chord=mood in BROKEN_CHORD_MOODS,
        harmony_arpeggio_step_ticks=HARMONY_ARPEGGIO_STEP_TICKS,
        harmony_pad_velocity=HARMONY_PAD_VELOCITY,
        harmony_arpeggio_velocity=HARMONY_ARPEGGIO_VELOCITY,
        harmony_stab_velocity=HARMONY_STAB_VELOCITY,
        harmony_melody_clearance=HARMONY_MELODY_CLEARANCE,
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
    "PLAN_FORMAT",
    "PLAN_FORMAT_PREFIX",
    "PLAN_SCHEMA_VERSION",
    "CompositionPlan",
    "PlanError",
    "UnsupportedPlanVersionError",
    "default_plan",
    "resolve_plan",
]
