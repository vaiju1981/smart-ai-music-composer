"""The typed change: what a user, a conductor or a critic may ask for.

If feedback becomes "append this text and recompose", the user is rerolling
with more words. So a request is a **delta** — a small typed value naming one
change — and this module is the whole vocabulary of them, plus the applier
that folds a chain of them into a piece.

Three rules hold it together.

**A delta writes one of two documents, and which one is not the caller's
choice.** `SetTempo` writes a field of `CompositionSpec`; `SetBassMotion`
writes a field of `CompositionPlan`. A subclass overrides exactly one of
`spec_changes` and `plan_changes`, and a test holds every registered type to
that — so a delta that forgot to override either is a failing test rather
than a request that quietly does nothing. A request that does nothing is the
one outcome the product may never produce: a silent no-op teaches the user
that the product is deaf.

**The spec is the substrate and the plan is the patch laid on it.** A chain is
folded in two passes: every spec delta in order, giving the spec the piece is
written for; then the plan is derived from that spec and every plan delta is
applied to it, in order. So the *position* of a spec delta among the plan
deltas does not matter — "hold the bass, and set the mood to sleep" means the
same thing either way round, and the pedal survives the mood change because it
is a patch on the plan the new mood derives. That is also what keeps the
recipe complete: `(spec, deltas) -> plan` is a pure function, so a draft that
stores its chain can be rebuilt exactly, and the chain *is* the plan's
provenance rather than a note about it.

**A request the engine cannot honour is refused with a reason.** Every delta's
domain is the document's own — the spec's field constraints, the plan's
`__post_init__` — and this module deliberately does not restate them, because
a second copy of a bound is a second thing to keep in step. So a delta is
applied by asking the document to accept the change, and a refusal is the
document's own message with the knob's value in front of it. Two consequences
are worth stating rather than discovering: a spec built by
`model_copy(update=...)` is *not* validated (Pydantic skips validation on a
copy), which is why the spec pass builds a new spec through `model_validate`
instead; and the applier never composes, so it cannot tell you whether the
piece still lints. That gate is `compose`, which raises rather than returning
a score the linter refused, and it fires where the notes are made.

**The vocabulary is bounded by what the plan actually carries**, which is the
design's own scope rule rather than a shortfall of this module. The design
named `SetHarmonicRhythm`, `SetSwing`, `SetDrumEntry` and `SetRegister`; plan
v1 has no knob for any of them. They are in `UNCARRIED` by name with what to
ask for instead, because a request the engine cannot honour must not be
answered with "I did not understand you": the request was understood, and it
is unbuilt.

Where a refusal can name a nearest legal request it does, and the bound is
read off the constraint that failed — Pydantic's `{"le": 240}` becomes
`SetTempo(240)` — rather than written down here a second time. Where the
document's own message already names the alternative ("percussion role
requires the drum_set instrument") the message *is* the alternative and
`nearest` stays empty, for the same reason.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, replace
from enum import StrEnum
from typing import Any, Final, Literal, TypeAlias

from pydantic import ValidationError

from saimc.compose.motif import FIGURES_BY_MOTION, BassMotion
from saimc.compose.plan import CompositionPlan, PlanError, default_plan
from saimc.spec import (
    CompositionSpec,
    Instrument,
    Mood,
    TimeSignature,
    VoiceRole,
    WesternKey,
)

Reason: TypeAlias = Literal["violates_the_spec", "violates_the_plan", "unknown_knob"]
"""Why a request was not honoured.

Three, and they are told apart because the sentence the user needs differs.
`violates_the_spec` and `violates_the_plan` are the same situation against
two documents — the request was understood and the piece cannot hold it —
while `unknown_knob` is a request naming something the engine has no knob
for at all. The last one is only ever built by `refuse_uncarried`, since a
knob that does not exist has no value to construct.
"""

Level: TypeAlias = Literal["none", "light", "expressive"]
"""The spec's `humanization` vocabulary, named once for the delta that sets it."""

Texture: TypeAlias = Literal["pad", "arpeggio", "stab"]
"""The accompaniment's three textures, one per harmony level the plan carries."""

Terrace: TypeAlias = Literal["opening", "peak", "final", "middle"]
"""One step of the dynamic arc, named as the plan's field suffix."""


@dataclass(frozen=True)
class DeltaRefusal:
    """A request the engine would not honour, why, and what to ask for instead.

    `request` is the knob's name rather than the delta, because a refusal
    outlives the value it was about: the preference log records them, and a
    name is what a reader can look up in `DELTA_TYPES`. The message carries
    the value as well, so nothing is lost by not keeping the delta itself.
    """

    request: str
    reason: Reason
    message: str
    nearest: str | None = None
    """The closest request the engine could honour, when there is one to name.

    Empty is a real answer rather than a gap: a request refused for a reason
    whose message already names the alternative needs no second copy of it,
    and a request refused because there is no nearby legal value has no
    alternative to offer. Inventing one would be advice the engine cannot
    honour either.
    """


@dataclass(frozen=True)
class DeltaApplication:
    """A chain of deltas folded into the pair of documents it describes.

    `applied` and `refused` are the chain, in order, split — so a caller can
    tell the user what happened to each thing they asked for, and the chain
    that produced a plan is recoverable from the two together. A chain that
    honoured three of four is a better answer than one that honoured none,
    which is the rule the fan-out already follows for a candidate the engine
    refused.
    """

    spec: CompositionSpec
    plan: CompositionPlan
    applied: tuple[Delta, ...]
    refused: tuple[DeltaRefusal, ...]

    @property
    def ok(self) -> bool:
        """Whether every request in the chain was honoured."""
        return not self.refused


@dataclass(frozen=True)
class Delta:
    """One typed change to a piece, and the name it is asked for by.

    A frozen dataclass with no fields of its own, so that every concrete
    delta is one too by inheritance rather than by a rule a reader has to
    be told: `fields()` and the generated `__eq__` are then the type
    system's promise about a delta, and a subclass that forgot the
    decorator would not be one. It is deliberately absent from
    `DELTA_TYPES`, which is what enumerates the vocabulary — so nothing can
    be built from a stored document that is not one of the requests the
    engine knows.

    A subclass overrides exactly one of `spec_changes` and `plan_changes`.
    The other stays empty, and that is how the applier knows which document
    a delta belongs to without being told twice.
    """

    def spec_changes(self, spec: CompositionSpec) -> Mapping[str, Any]:
        """The spec fields this delta sets. Empty for a delta of the plan."""
        return {}

    def plan_changes(self, plan: CompositionPlan) -> Mapping[str, Any]:
        """The plan fields this delta sets. Empty for a delta of the spec."""
        return {}

    @property
    def knob(self) -> str:
        """The name the tool call, the preference log and the UI all use."""
        return type(self).__name__

    def describe(self) -> str:
        """One line, for the turn log and the orchestration disclosure.

        Written from the dataclass's own fields rather than per type, so a
        field added to a delta cannot leave the description behind. A
        `StrEnum` field reads as its value — `mood=sleep`, not
        `mood=Mood.SLEEP` — because the reader is a user being told what
        changed, not a debugger.
        """
        rendered = ", ".join(
            f"{field.name}={_value(getattr(self, field.name))}" for field in fields(self)
        )
        return f"{self.knob}({rendered})"


# --- Tier 1: the spec. No model call is needed to author one of these. ---


@dataclass(frozen=True)
class SetTempo(Delta):
    """How fast the piece is played, in BPM."""

    tempo_bpm: int

    def spec_changes(self, spec: CompositionSpec) -> Mapping[str, Any]:
        return {"tempo_bpm": self.tempo_bpm}


@dataclass(frozen=True)
class SetMood(Delta):
    """The mood, which is the one spec field the derived plan comes from.

    Changing it re-derives the plan — the rhythm vocabulary, the bass
    figures, the tie probability, the broken-chord texture and the kit's
    style are all the mood's — and any plan delta in the chain is then
    applied to the plan the new mood produced.
    """

    mood: Mood

    def __post_init__(self) -> None:
        _canonical(self, "mood", Mood)

    def spec_changes(self, spec: CompositionSpec) -> Mapping[str, Any]:
        return {"mood": self.mood}


@dataclass(frozen=True)
class SetKey(Delta):
    """The key, or `None` to leave the choice to the engine."""

    key: WesternKey | None

    def __post_init__(self) -> None:
        if self.key is not None:
            _canonical(self, "key", WesternKey)

    def spec_changes(self, spec: CompositionSpec) -> Mapping[str, Any]:
        return {"key": self.key}


@dataclass(frozen=True)
class SetTimeSignature(Delta):
    """The meter, which the kit's style is resolved against as well."""

    time_signature: TimeSignature

    def __post_init__(self) -> None:
        _canonical(self, "time_signature", TimeSignature)

    def spec_changes(self, spec: CompositionSpec) -> Mapping[str, Any]:
        return {"time_signature": self.time_signature}


@dataclass(frozen=True)
class SetDuration(Delta):
    """The target length in seconds. The arrangement search fills it."""

    duration_seconds: int

    def spec_changes(self, spec: CompositionSpec) -> Mapping[str, Any]:
        return {"duration_seconds": self.duration_seconds}


@dataclass(frozen=True)
class SetInstrument(Delta):
    """Which instrument plays one of the ensemble's roles.

    The role's first entry is replaced where the role is already carried
    and a new entry is added where it is not: an ensemble may hold two
    harmony voices, and a request naming a role without naming which of
    them means the one the accompaniment leads with. Whether the result is
    an ensemble the spec accepts — a duplicated instrument, a percussion
    voice that is not the kit — is the spec's to say, and its refusal is
    what the caller is told.
    """

    role: VoiceRole
    instrument: Instrument

    def __post_init__(self) -> None:
        _canonical(self, "role", VoiceRole)
        _canonical(self, "instrument", Instrument)

    def spec_changes(self, spec: CompositionSpec) -> Mapping[str, Any]:
        entries = [entry.model_dump() for entry in spec.instrumentation]
        for entry in entries:
            if entry["role"] == self.role:
                entry["instrument"] = self.instrument
                break
        else:
            entries.append({"role": self.role, "instrument": self.instrument})
        return {"instrumentation": entries}


@dataclass(frozen=True)
class SetHumanization(Delta):
    """How much timing and velocity variation the performance layer applies."""

    level: Level

    def spec_changes(self, spec: CompositionSpec) -> Mapping[str, Any]:
        return {"humanization": self.level}


@dataclass(frozen=True)
class ReRoll(Delta):
    """The seed, which is the same piece's material composed again.

    It is the one delta that moves the music without touching a musical
    value, and the reason the plan carries no seed: the arrangement, the
    harmony and the vocabulary stay exactly as they were drawn, and only
    the draws that follow them are new.
    """

    seed: int

    def spec_changes(self, spec: CompositionSpec) -> Mapping[str, Any]:
        return {"seed": self.seed}


# --- Tier 2: the plan. Authored from a sentence, one model call away. ---


@dataclass(frozen=True)
class SetMelodyBand(Delta):
    """The register window a melody line is written in, in semitones.

    `range_semitones` and `tessitura_overlap_semitones` are the two bars
    this moves: a wider band lets the tune span more, and a band wider than
    the instrument leaves the bed nowhere outside it.
    """

    semitones: int

    def plan_changes(self, plan: CompositionPlan) -> Mapping[str, Any]:
        return {"line_band_semitones": self.semitones}


@dataclass(frozen=True)
class SetHarmonyClearance(Delta):
    """How far under the tune the harmony bed is held, in semitones."""

    semitones: int

    def plan_changes(self, plan: CompositionPlan) -> Mapping[str, Any]:
        return {"harmony_melody_clearance": self.semitones}


@dataclass(frozen=True)
class SetAccompanimentDensity(Delta):
    """How often the broken-chord figure steps, in ticks.

    Smaller is denser — the onsets per bar are `ticks_per_bar // step_ticks`
    — so this is the accompaniment's density rather than a note length, and
    it is the knob `texture_hierarchy` measures.
    """

    step_ticks: int

    def plan_changes(self, plan: CompositionPlan) -> Mapping[str, Any]:
        return {"harmony_arpeggio_step_ticks": self.step_ticks}


@dataclass(frozen=True)
class SetHarmonyTexture(Delta):
    """Whether the leading harmony voice states a broken chord."""

    broken_chord: bool

    def plan_changes(self, plan: CompositionPlan) -> Mapping[str, Any]:
        return {"harmony_broken_chord": self.broken_chord}


@dataclass(frozen=True)
class SetHarmonyLevel(Delta):
    """How loud one of the accompaniment's three textures is played.

    A level rather than a change to a level, because it is a MIDI velocity
    and the user is asking for one: "the pad is too loud" means a number.
    """

    texture: Texture
    velocity: int

    def plan_changes(self, plan: CompositionPlan) -> Mapping[str, Any]:
        return {_HARMONY_LEVEL_FIELDS[self.texture]: self.velocity}


@dataclass(frozen=True)
class SetMotifVariation(Delta):
    """How readily the melody varies its motif, as a factor on the weights.

    A factor rather than a value, because the weights are a table: 1.0 is
    the piece as it stands, and a larger number gives the operations that
    vary the motif more of the draw at the ornament's expense. The weights
    are not probabilities and need not sum to one — the weight left over is
    the ornament's — so scaling them is exactly the request.
    """

    factor: float

    def plan_changes(self, plan: CompositionPlan) -> Mapping[str, Any]:
        return {
            "motif_operation_weights": tuple(
                (name, weight * self.factor) for name, weight in plan.motif_operation_weights
            )
        }


@dataclass(frozen=True)
class SetBassMotion(Delta):
    """Which left-hand figure leads the bass vocabulary.

    The chosen figure is put first and the piece's own remaining figures
    are kept behind it, because the plan's vocabulary is a *set* of figures
    to draw from rather than one to repeat: a delta that replaced it with a
    single figure would state the same bar for the whole piece and miss
    `bass_onset_patterns`, which is the bar that measures exactly that.
    """

    motion: BassMotion

    def __post_init__(self) -> None:
        _canonical(self, "motion", BassMotion)

    def plan_changes(self, plan: CompositionPlan) -> Mapping[str, Any]:
        chosen = FIGURES_BY_MOTION[self.motion]
        rest = tuple(figure for figure in plan.bass_figures if figure != chosen)
        return {"bass_figures": (chosen, *rest)}


@dataclass(frozen=True)
class SetCadence(Delta):
    """The degree the final cadence approaches from, and whether it is a seventh."""

    degree: int
    seventh: bool

    def plan_changes(self, plan: CompositionPlan) -> Mapping[str, Any]:
        return {"cadence_degree": self.degree, "cadence_seventh": self.seventh}


@dataclass(frozen=True)
class SetModulation(Delta):
    """The semitones a long piece's final repetition is lifted by.

    Zero is a legal choice — a long piece that returns home instead of
    lifting — and it is not this engine's default.
    """

    semitones: int

    def plan_changes(self, plan: CompositionPlan) -> Mapping[str, Any]:
        return {"modulation_offset": self.semitones}


@dataclass(frozen=True)
class SetIntroBars(Delta):
    """How many bars the melody rests at the top of a long piece.

    Bounded by the shortest form the arrangement may pick, since the intro
    is carved from the first section.
    """

    bars: int

    def plan_changes(self, plan: CompositionPlan) -> Mapping[str, Any]:
        return {"intro_bars": self.bars}


@dataclass(frozen=True)
class SetSectionEnergy(Delta):
    """One terrace of the dynamic arc, as a factor on the piece's level.

    A factor rather than a value: the four terraces are velocities relative
    to the piece's own dynamic, so "bring the peak up a little" is a factor
    whatever the piece's level happens to be.
    """

    role: Terrace
    factor: float

    def plan_changes(self, plan: CompositionPlan) -> Mapping[str, Any]:
        # The terrace's name is the plan's field suffix: peak -> section_energy_peak.
        terrace = f"section_energy_{self.role}"
        return {terrace: getattr(plan, terrace) * self.factor}


@dataclass(frozen=True)
class SetDrumStyle(Delta):
    """Which kit the percussion voice plays, or none for a piece without drums.

    The style is a name rather than a table because a `DrumStyle` is a set
    of bar templates, which no stored document holds — and because whether
    a style has a template for this meter is a fact about the style, so a
    name asked to play a meter it has none for writes no drums at all
    rather than a wrong groove.
    """

    name: str | None

    def plan_changes(self, plan: CompositionPlan) -> Mapping[str, Any]:
        return {"drum_style_name": self.name}


@dataclass(frozen=True)
class SetPercussionRest(Delta):
    """Which section a long piece's kit rests for, counting from zero."""

    section: int

    def plan_changes(self, plan: CompositionPlan) -> Mapping[str, Any]:
        return {"percussion_rest_section": self.section}


DELTA_TYPES: Final[dict[str, type[Delta]]] = {
    "SetTempo": SetTempo,
    "SetMood": SetMood,
    "SetKey": SetKey,
    "SetTimeSignature": SetTimeSignature,
    "SetDuration": SetDuration,
    "SetInstrument": SetInstrument,
    "SetHumanization": SetHumanization,
    "ReRoll": ReRoll,
    "SetMelodyBand": SetMelodyBand,
    "SetHarmonyClearance": SetHarmonyClearance,
    "SetAccompanimentDensity": SetAccompanimentDensity,
    "SetHarmonyTexture": SetHarmonyTexture,
    "SetHarmonyLevel": SetHarmonyLevel,
    "SetMotifVariation": SetMotifVariation,
    "SetBassMotion": SetBassMotion,
    "SetCadence": SetCadence,
    "SetModulation": SetModulation,
    "SetIntroBars": SetIntroBars,
    "SetSectionEnergy": SetSectionEnergy,
    "SetDrumStyle": SetDrumStyle,
    "SetPercussionRest": SetPercussionRest,
}
"""Every delta, by the name a tool call and a stored document use.

The registry *is* the vocabulary: a knob absent from it is a request
nothing can build, which is why `Delta` itself is not a member and why the
uncarried requests below are not either. A test holds the keys to the
classes that actually exist, so a delta added without a name — or a name
left behind by one that was removed — fails rather than drifting.
"""

_HARMONY_LEVEL_FIELDS: Final[dict[str, str]] = {
    "pad": "harmony_pad_velocity",
    "arpeggio": "harmony_arpeggio_velocity",
    "stab": "harmony_stab_velocity",
}


@dataclass(frozen=True)
class Uncarried:
    """A request the design named and plan v1 has no knob for."""

    request: str
    why: str
    instead: str


UNCARRIED: Final[tuple[Uncarried, ...]] = (
    Uncarried(
        request="SetRegister",
        why=(
            "the tune's register is settled against the instrument's compass and the "
            "bed settles under it, so there is no knob that moves one voice's register "
            "on its own"
        ),
        instead="SetMelodyBand, the window the tune is written in, or SetHarmonyClearance",
    ),
    Uncarried(
        request="SetHarmonicRhythm",
        why="how often the harmony changes is the progression template's, and the plan carries no knob for it",
        instead="SetBassMotion, or SetAccompanimentDensity for the figure's own rate",
    ),
    Uncarried(
        request="SetSwing",
        why="a swing ratio needs a triplet grid, which the engine does not have",
        instead="SetDrumStyle, for the styles it does carry",
    ),
    Uncarried(
        request="SetDrumEntry",
        why="the kit's entry is a section rather than a bar, which is what the plan carries",
        instead="SetPercussionRest",
    ),
    Uncarried(
        request="ExtendSection",
        why=(
            "a section's length comes from the arrangement search over the phrase "
            "sizes, so there is no per-section bar count to set"
        ),
        instead="SetDuration, or SetIntroBars for the melody's entrance",
    ),
)
"""The requests the vocabulary cannot honour, and what to ask for instead.

Written down rather than left to the translator's default, because
otherwise every one of these would be refused as an unknown word. The
request was understood; it is unbuilt, and those are different things to
tell a user. `refuse_uncarried` is how a caller tells them apart.
"""


def apply_deltas(spec: CompositionSpec, deltas: Iterable[Delta]) -> DeltaApplication:
    """Fold a chain of deltas into the spec and the plan they describe.

    Two passes and the order between them is the design — see the module
    docstring. A refused delta changes nothing and the chain carries on.
    """
    chain = tuple(deltas)
    refusals: dict[int, DeltaRefusal] = {}

    for index, delta in enumerate(chain):
        changes = delta.spec_changes(spec)
        if not changes:
            continue
        try:
            spec = CompositionSpec.model_validate({**spec.model_dump(), **changes})
        except ValidationError as error:
            refusals[index] = _spec_refusal(delta, error)

    plan = default_plan(spec)
    for index, delta in enumerate(chain):
        changes = delta.plan_changes(plan)
        if not changes:
            continue
        try:
            plan = replace(plan, **changes)
        except PlanError as error:
            refusals[index] = _plan_refusal(delta, error)

    return DeltaApplication(
        spec=spec,
        plan=plan,
        applied=tuple(delta for index, delta in enumerate(chain) if index not in refusals),
        refused=tuple(refusals[index] for index in sorted(refusals)),
    )


def refuse_uncarried(request: str) -> DeltaRefusal | None:
    """The refusal for a request no knob carries, or `None` if it is not one.

    `None` for anything absent from `UNCARRIED` is what lets a caller ask
    "is this an unbuilt knob, or a word I do not know?" and answer with a
    different sentence — which is the whole reason the table exists.
    """
    entry = next((each for each in UNCARRIED if each.request == request), None)
    if entry is None:
        return None
    return DeltaRefusal(
        request=entry.request,
        reason="unknown_knob",
        message=(
            f"{entry.request} is not a knob this engine carries yet: {entry.why}. "
            f"The closest is {entry.instead}."
        ),
        nearest=entry.instead,
    )


def delta_to_dict(delta: Delta) -> dict[str, Any]:
    """The delta as the document the preference log stores.

    The knob's name is the key a reader dispatches on and every field is
    written as its plain value, so the document carries no type names —
    which is what makes a log of these readable later without the class
    that wrote it, and what makes two identical requests one entry.
    """
    return {
        "knob": delta.knob,
        **{field.name: _plain(getattr(delta, field.name)) for field in fields(delta)},
    }


def delta_from_dict(document: Mapping[str, Any]) -> Delta:
    """Rebuild a delta from `delta_to_dict`'s document.

    The knob has to be one this build knows: a document naming another is
    refused by name, with the vocabulary listed, rather than skipped. A
    value the dataclass cannot hold is refused by the constructor — the
    enums are held as members, so a name outside one fails here rather
    than becoming a string nothing else in this module expects.
    """
    knob = document.get("knob")
    if knob not in DELTA_TYPES:
        known = ", ".join(sorted(DELTA_TYPES))
        raise ValueError(f"unknown knob {knob!r}; this build knows {known}")
    return DELTA_TYPES[knob](**{name: value for name, value in document.items() if name != "knob"})


def _spec_refusal(delta: Delta, error: ValidationError) -> DeltaRefusal:
    """The refusal for a delta the spec would not accept.

    The message is Pydantic's own, which names the constraint that failed,
    with the request's name and value in front of it. Terser than a
    hand-written sentence and better, because it states the constraint the
    spec actually enforced rather than a description of it that could
    drift away from it.
    """
    failing = error.errors()[0]
    return DeltaRefusal(
        request=delta.knob,
        reason="violates_the_spec",
        message=f"{delta.describe()} cannot be honoured: {failing['msg']}",
        nearest=_nearest_from(delta.knob, failing),
    )


def _plan_refusal(delta: Delta, error: PlanError) -> DeltaRefusal:
    """The refusal for a delta the plan would not accept.

    A plan's refusals are written sentences naming the bound — "intro_bars
    (16) must be shorter than the shortest form (8)" — so there is nothing
    to add but the request. Where one names the alternative too, the
    message is the alternative and no `nearest` is invented beside it.
    """
    return DeltaRefusal(
        request=delta.knob,
        reason="violates_the_plan",
        message=f"{delta.describe()} cannot be honoured: {error}",
    )


def _nearest_from(knob: str, failing: Mapping[str, Any]) -> str | None:
    """The closest request the spec would accept, read off the failed constraint.

    Pydantic puts the bound in the error's `ctx`, so the alternative is the
    constraint that fired rather than a second copy of it written here.
    Only the inclusive bounds are read: the nearest legal value of an
    exclusive one is not the bound itself, and no field in this vocabulary
    is bounded exclusively by Pydantic.
    """
    context = failing.get("ctx") or {}
    for key in ("le", "ge"):
        if key in context:
            return f"{knob}({context[key]})"
    return None


def _canonical(instance: Delta, name: str, enum: type[StrEnum]) -> None:
    """Hold an enum field as the member it names rather than as the string.

    A delta is built from three directions — a UI control, a model's tool
    call, and a stored document — and the last two arrive with a string
    where this module's types say member. Coercing on construction rather
    than at each reader means the string form is *equal* to the member form
    from the moment the delta exists, so a round trip through the log
    compares equal and the same request asked twice is one entry.
    """
    object.__setattr__(instance, name, enum(getattr(instance, name)))


def _plain(value: Any) -> Any:
    """A field's value as JSON holds it: an enum is written as its value."""
    return value.value if isinstance(value, StrEnum) else value


def _value(value: Any) -> str:
    """A field's value as prose. `None` reads as "none" rather than "None"."""
    return "none" if value is None else str(value)


__all__ = [
    "DELTA_TYPES",
    "UNCARRIED",
    "Delta",
    "DeltaApplication",
    "DeltaRefusal",
    "Level",
    "ReRoll",
    "Reason",
    "SetAccompanimentDensity",
    "SetBassMotion",
    "SetCadence",
    "SetDrumStyle",
    "SetDuration",
    "SetHarmonyClearance",
    "SetHarmonyLevel",
    "SetHarmonyTexture",
    "SetHumanization",
    "SetInstrument",
    "SetIntroBars",
    "SetKey",
    "SetMelodyBand",
    "SetModulation",
    "SetMood",
    "SetMotifVariation",
    "SetPercussionRest",
    "SetSectionEnergy",
    "SetTempo",
    "SetTimeSignature",
    "Terrace",
    "Texture",
    "Uncarried",
    "apply_deltas",
    "delta_from_dict",
    "delta_to_dict",
    "refuse_uncarried",
]
