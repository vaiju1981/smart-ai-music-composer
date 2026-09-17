"""The delta vocabulary: a request writes one document, a refusal names its reason, a chain folds.

The tests are written three ways because the module has three jobs.

The **vocabulary** is tested as a closed set. A walk of `Delta.__subclasses__()`
holds the registry to the classes that exist, and a table of one example per
knob holds every registered type to "writes exactly one of the two documents",
"names a field that document actually has", "survives a round trip through its
stored form" and "moves the thing it says it moves". The examples are shared
between those tests deliberately: a knob added without an example fails the
table's own premise, so the ratchet cannot be satisfied by writing a new type
and nothing else.

The **fold** is tested as two passes over one chain, because that order is the
design and not an implementation detail: the spec is the substrate, the plan is
the patch laid on it, and a chain that says "hold the bass, and set the mood to
sleep" has to mean the same thing read either way round. Only one test is posed
on a value the vocabulary cannot express — the count of `default_plan` calls,
made with a wrapper — and it is there because the fold's *shape* is otherwise
invisible: a version that re-derived the plan after every delta would produce
the same two documents for every chain in this file.

The **refusals** are tested against the sentences the documents write, not
against sentences repeated here. So the assertions read "the message contains
the constraint" rather than "the message is this string": a bound restated in a
test is a second copy of it, which is the drift the module refuses to introduce
in the first place.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import FrozenInstanceError, fields
from enum import StrEnum
from typing import get_args

import pytest

from saimc.compose.duration import DurationArrangement, arrange_for_duration
from saimc.compose.engine import compose
from saimc.compose.motif import FIGURES_BY_MOTION, BassMotion
from saimc.compose.plan import CompositionPlan, default_plan
from saimc.session import deltas
from saimc.session.deltas import (
    DELTA_TYPES,
    UNCARRIED,
    Delta,
    DeltaApplication,
    DeltaRefusal,
    ReRoll,
    SetAccompanimentDensity,
    SetBassMotion,
    SetCadence,
    SetDrumStyle,
    SetDuration,
    SetHarmonyClearance,
    SetHarmonyLevel,
    SetHarmonyTexture,
    SetHumanization,
    SetInstrument,
    SetIntroBars,
    SetKey,
    SetMelodyBand,
    SetModulation,
    SetMood,
    SetMotifVariation,
    SetPercussionRest,
    SetSectionClose,
    SetSectionEnergy,
    SetTempo,
    SetTimeSignature,
    Texture,
    apply_deltas,
    delta_from_dict,
    delta_to_dict,
    refusal_line,
    refuse_uncarried,
    swallowed_tempo,
)
from saimc.spec import (
    CompositionSpec,
    Instrument,
    Mood,
    TimeSignature,
    VoiceRole,
    WesternKey,
)

_SPEC = CompositionSpec(mood=Mood.CALMING, duration_seconds=30)
"""The piece every case starts from: one mood, one duration, nothing else said.

Calming at 30 seconds because it is the cheapest spec the engine composes and
because its defaults are the ones each example below was checked against — a
`SetCadence(4, True)` moves nothing if the default is already a fourth, and a
`SetIntroBars(2)` would be the identity.
"""

_BASE_PLAN = default_plan(_SPEC)

_EXAMPLES: dict[str, Delta] = {
    "SetTempo": SetTempo(tempo_bpm=90),
    "SetMood": SetMood(mood=Mood.SLEEP),
    "SetKey": SetKey(key=WesternKey.G_MAJOR),
    "SetTimeSignature": SetTimeSignature(time_signature=TimeSignature.THREE_FOUR),
    "SetDuration": SetDuration(duration_seconds=60),
    "SetInstrument": SetInstrument(role=VoiceRole.MELODY, instrument=Instrument.FLUTE),
    "SetHumanization": SetHumanization(level="expressive"),
    "ReRoll": ReRoll(seed=7),
    "SetMelodyBand": SetMelodyBand(semitones=19),
    "SetHarmonyClearance": SetHarmonyClearance(semitones=6),
    "SetAccompanimentDensity": SetAccompanimentDensity(step_ticks=120),
    "SetHarmonyTexture": SetHarmonyTexture(broken_chord=True),
    "SetHarmonyLevel": SetHarmonyLevel(texture="pad", velocity=64),
    "SetMotifVariation": SetMotifVariation(factor=1.5),
    "SetBassMotion": SetBassMotion(motion=BassMotion.PEDAL),
    "SetCadence": SetCadence(degree=4, seventh=True),
    "SetSectionClose": SetSectionClose(close="full"),
    "SetModulation": SetModulation(semitones=5),
    "SetIntroBars": SetIntroBars(bars=3),
    "SetSectionEnergy": SetSectionEnergy(role="peak", factor=1.1),
    "SetDrumStyle": SetDrumStyle(name="funk"),
    "SetPercussionRest": SetPercussionRest(section=2),
}
"""One request per registered knob, and every one of them differs from the default.

The value matters as much as the presence: a delta set to the piece's own
current value moves nothing, and the tests below that assert a move would pass
without one. Each was checked against `_SPEC`'s defaults before it was written
down — `SetBassMotion(PEDAL)` rather than `ROOT`, because a calming piece
already leads with the root, which is finding 3 in the plan's Phase D block.
"""

_SPEC_EFFECTS: dict[str, dict[str, object]] = {
    "SetTempo": {"tempo_bpm": 90},
    "SetMood": {"mood": Mood.SLEEP},
    "SetKey": {"key": WesternKey.G_MAJOR},
    "SetTimeSignature": {"time_signature": TimeSignature.THREE_FOUR},
    "SetDuration": {"duration_seconds": 60},
    "SetHumanization": {"humanization": "expressive"},
    "ReRoll": {"seed": 7},
}
"""What each spec delta writes, as the value the spec must end up holding."""

_PLAN_EFFECTS: dict[str, dict[str, object]] = {
    "SetMelodyBand": {"line_band_semitones": 19},
    "SetHarmonyClearance": {"harmony_melody_clearance": 6},
    "SetAccompanimentDensity": {"harmony_arpeggio_step_ticks": 120},
    "SetHarmonyTexture": {"harmony_broken_chord": True},
    "SetHarmonyLevel": {"harmony_pad_velocity": 64},
    "SetCadence": {"cadence_degree": 4, "cadence_seventh": True},
    "SetSectionClose": {"section_close": "full"},
    "SetModulation": {"modulation_offset": 5},
    "SetIntroBars": {"intro_bars": 3},
    "SetSectionEnergy": {"section_energy_peak": pytest.approx(1.12 * 1.1)},
    "SetDrumStyle": {"drum_style_name": "funk"},
    "SetPercussionRest": {"percussion_rest_section": 2},
}
"""What each plan delta writes. The three whose value is *derived* are exempt.

`SetInstrument` writes a whole ensemble, `SetMotifVariation` scales a table and
`SetBassMotion` reorders a vocabulary, so each has its own class below where the
derivation can be stated rather than tabulated. The premise test holds the union
of the three tables to the registry, so an exempt knob cannot be quietly absent.
"""


_SPEC_KNOBS: set[str] = set(_SPEC_EFFECTS) | {"SetInstrument"}
_PLAN_KNOBS: set[str] = set(_PLAN_EFFECTS) | {"SetMotifVariation", "SetBassMotion"}
"""Which document each knob writes, for the tests that have to pose one side.

The two sets together are the whole vocabulary, and that is asserted rather than
trusted: a knob in neither would be a delta no table covers, which is the state
the exempt-three paragraph above is about.
"""

_HARMONY_LEVELS: dict[str, str] = {
    "pad": "harmony_pad_velocity",
    "arpeggio": "harmony_arpeggio_velocity",
    "stab": "harmony_stab_velocity",
}
"""The three textures and the plan field each one is played through."""

_DERIVING_KNOBS: set[str] = {"SetMood", "SetTimeSignature"}
"""The two spec deltas the derived plan reads, so that the other six cannot move it.

Both are in `_SPEC_KNOBS`, so naming them here rather than subtracting them from
the registry is a claim about *which* spec fields `default_plan` reads — the
mood and the meter, from `style_name_for` — and a third one joining them fails
`test_a_spec_delta_that_derives_nothing_leaves_the_plan_alone` loudly.
"""


def _registered_deltas() -> set[str]:
    """Every `Delta` subclass the module defines, found by walking the classes.

    Walking rather than listing, because a list is a second copy of the
    vocabulary: a delta registered and checked by a table this file also has to
    maintain would be a delta whose absence from the list is unnoticeable.
    """
    found: set[str] = set()
    pending = list(Delta.__subclasses__())
    while pending:
        klass = pending.pop()
        found.add(klass.__name__)
        pending.extend(klass.__subclasses__())
    return found


class TestTheVocabularyIsClosed:
    def test_the_registry_names_every_delta_that_exists(self) -> None:
        assert set(DELTA_TYPES) == _registered_deltas()

    def test_the_base_is_deliberately_not_a_delta(self) -> None:
        assert "Delta" not in DELTA_TYPES

    def test_every_name_is_the_knob_the_class_reports(self) -> None:
        for name, klass in DELTA_TYPES.items():
            assert isinstance(_EXAMPLES[name], klass)
            assert _EXAMPLES[name].knob == name

    def test_every_registered_type_has_an_example(self) -> None:
        assert set(_EXAMPLES) == set(DELTA_TYPES)

    def test_every_example_is_checked_by_one_of_the_tables(self) -> None:
        assert set(DELTA_TYPES) == _SPEC_KNOBS | _PLAN_KNOBS
        assert not _SPEC_KNOBS & _PLAN_KNOBS

    def test_the_harmony_level_cases_cover_the_textures_the_type_declares(self) -> None:
        assert set(_HARMONY_LEVELS) == set(get_args(Texture))
        assert set(_HARMONY_LEVELS.values()) <= {field.name for field in fields(CompositionPlan)}

    @pytest.mark.parametrize("knob", list(_EXAMPLES))
    def test_every_example_writes_exactly_one_document(self, knob: str) -> None:
        delta = _EXAMPLES[knob]
        writes_spec = bool(delta.spec_changes(_SPEC))
        writes_plan = bool(delta.plan_changes(_BASE_PLAN))
        assert writes_spec != writes_plan, f"{knob} writes both documents, or neither"

    @pytest.mark.parametrize("knob", list(_EXAMPLES))
    def test_every_example_moves_the_document_it_writes(self, knob: str) -> None:
        application = apply_deltas(_SPEC, [_EXAMPLES[knob]])
        assert application.ok, application.refused
        assert application.spec != _SPEC or application.plan != _BASE_PLAN

    @pytest.mark.parametrize("knob", list(_EXAMPLES))
    def test_every_spec_delta_names_a_field_the_spec_has(self, knob: str) -> None:
        changes = _EXAMPLES[knob].spec_changes(_SPEC)
        assert set(changes) <= set(CompositionSpec.model_fields)

    @pytest.mark.parametrize("knob", list(_EXAMPLES))
    def test_every_plan_delta_names_a_field_the_plan_has(self, knob: str) -> None:
        # The fold patches the plan with `replace`, which raises `TypeError` for a
        # name the plan does not carry — an uncaught crash rather than a refusal. So
        # the names are checked against the dataclass here instead of discovered by
        # composing something.
        changes = _EXAMPLES[knob].plan_changes(_BASE_PLAN)
        assert set(changes) <= {field.name for field in fields(CompositionPlan)}

    @pytest.mark.parametrize("knob", list(_EXAMPLES))
    def test_a_delta_cannot_be_edited_in_place(self, knob: str) -> None:
        delta = _EXAMPLES[knob]
        first = fields(delta)[0].name
        with pytest.raises(FrozenInstanceError):
            setattr(delta, first, None)


class TestTheDeltaDescribesItself:
    @pytest.mark.parametrize("knob", list(_EXAMPLES))
    def test_the_description_names_the_knob_and_every_field(self, knob: str) -> None:
        delta = _EXAMPLES[knob]
        described = delta.describe()
        assert described.startswith(f"{knob}(")
        assert described.endswith(")")
        for field in fields(delta):
            assert f"{field.name}=" in described

    @pytest.mark.parametrize("knob", list(_EXAMPLES))
    def test_the_description_carries_no_type_names(self, knob: str) -> None:
        described = _EXAMPLES[knob].describe()
        assert "<" not in described
        assert "Mood." not in described
        assert "VoiceRole." not in described

    def test_an_enum_reads_as_its_value(self) -> None:
        assert SetMood(mood=Mood.SLEEP).describe() == "SetMood(mood=sleep)"
        assert SetBassMotion(motion=BassMotion.PEDAL).describe() == "SetBassMotion(motion=pedal)"

    def test_a_missing_value_reads_as_none_rather_than_None(self) -> None:
        assert SetKey(key=None).describe() == "SetKey(key=none)"
        assert SetDrumStyle(name=None).describe() == "SetDrumStyle(name=none)"

    def test_a_delta_of_two_fields_names_both(self) -> None:
        assert SetCadence(degree=4, seventh=True).describe() == "SetCadence(degree=4, seventh=True)"


class TestASpecDeltaWritesTheSpec:
    @pytest.mark.parametrize("knob", list(_SPEC_EFFECTS))
    def test_the_named_fields_end_up_holding_the_requested_values(self, knob: str) -> None:
        application = apply_deltas(_SPEC, [_EXAMPLES[knob]])
        assert application.ok, application.refused
        for field, expected in _SPEC_EFFECTS[knob].items():
            assert getattr(application.spec, field) == expected

    @pytest.mark.parametrize("knob", sorted(_SPEC_KNOBS - _DERIVING_KNOBS))
    def test_a_spec_delta_that_derives_nothing_leaves_the_plan_alone(self, knob: str) -> None:
        # `default_plan` reads the mood and the meter, and nothing else — so a spec
        # delta of any other field cannot move the plan, which is what makes the
        # fold's second pass a patch rather than a second derivation.
        assert apply_deltas(_SPEC, [_EXAMPLES[knob]]).plan == _BASE_PLAN

    @pytest.mark.parametrize("knob", sorted(_DERIVING_KNOBS))
    def test_the_two_spec_deltas_the_plan_reads_do_move_it(self, knob: str) -> None:
        application = apply_deltas(_SPEC, [_EXAMPLES[knob]])
        assert application.ok, application.refused
        assert application.plan != _BASE_PLAN


class TestSetInstrument:
    def test_a_role_the_ensemble_does_not_carry_is_added(self) -> None:
        application = apply_deltas(
            _SPEC, [SetInstrument(role=VoiceRole.PERCUSSION, instrument=Instrument.DRUM_SET)]
        )
        roles = [entry.role for entry in application.spec.instrumentation]
        assert application.ok, application.refused
        assert VoiceRole.PERCUSSION in roles
        assert len(roles) == len(_SPEC.instrumentation) + 1

    def test_a_role_the_ensemble_carries_is_replaced_where_it_stands(self) -> None:
        application = apply_deltas(_SPEC, [_EXAMPLES["SetInstrument"]])
        assert application.ok, application.refused
        assert application.spec.instrumentation[0] == _SPEC.instrumentation[0].model_copy(
            update={"instrument": Instrument.FLUTE}
        )
        assert [entry.role for entry in application.spec.instrumentation] == [
            entry.role for entry in _SPEC.instrumentation
        ]

    def test_only_the_first_entry_of_a_role_is_replaced(self) -> None:
        # An ensemble may hold two harmony voices, and a request naming a role
        # without naming which of them means the one the accompaniment leads with.
        five = CompositionSpec(
            mood=Mood.CALMING,
            duration_seconds=30,
            instrumentation=[
                {"role": "melody", "instrument": "piano"},
                {"role": "harmony", "instrument": "pizzicato_strings"},
                {"role": "harmony", "instrument": "cello"},
                {"role": "bass", "instrument": "contrabass"},
                {"role": "percussion", "instrument": "drum_set"},
            ],
        )
        harmony = [entry for entry in five.instrumentation if entry.role == VoiceRole.HARMONY]
        assert len(harmony) == 2, "this case asserts nothing unless the ensemble holds two"
        application = apply_deltas(
            five, [SetInstrument(role=VoiceRole.HARMONY, instrument=Instrument.FLUTE)]
        )
        assert application.ok, application.refused
        after = [
            entry for entry in application.spec.instrumentation if entry.role == VoiceRole.HARMONY
        ]
        assert [entry.instrument for entry in after] == [Instrument.FLUTE, harmony[1].instrument]

    def test_an_instrument_the_ensemble_already_gives_another_role_is_refused(self) -> None:
        # The spec's own rule, reached through this delta: a cello melody on a piece
        # whose bass is already a cello. Nothing here restates the rule — the
        # message is the spec's — and the case exists because this is the delta
        # whose whole job is to change an ensemble.
        (refusal,) = apply_deltas(
            _SPEC, [SetInstrument(role=VoiceRole.MELODY, instrument=Instrument.CELLO)]
        ).refused
        assert refusal.reason == "violates_the_spec"
        assert "at most once" in refusal.message

    def test_the_refusal_keeps_the_ensembles_own_sentence_and_offers_no_nearest(self) -> None:
        application = apply_deltas(
            _SPEC, [SetInstrument(role=VoiceRole.PERCUSSION, instrument=Instrument.FLUTE)]
        )
        (refusal,) = application.refused
        assert refusal.reason == "violates_the_spec"
        assert "drum_set" in refusal.message
        assert refusal.nearest is None, "the message already names the alternative"
        assert application.spec == _SPEC


class TestTheHarmonyLevels:
    """One case per texture, because the texture-to-field pairing is a table.

    `SetSectionEnergy`'s terrace-to-field pairing is a formula — the terrace's
    name *is* the suffix — so one case proves it for all four. The three harmony
    levels are three literal strings, so each needs its own: a `pad` request
    writing the arpeggio's field is a mistake no formula can make and no single
    case can witness.

    That is the whole of the argument for *coverage*, and it was read as the
    whole of the argument for the field: D3 found by using the type that a
    formula over an unvalidated value is not the same thing as a checked one —
    an unknown terrace and an unknown texture each raised from inside the fold
    rather than refusing. The check is a `Literal`'s members at construction, and
    `test_a_value_outside_a_literal_is_refused_at_construction` is where that
    lives. **A formula needs no case per entry; it does need the value checked.**
    """

    @pytest.mark.parametrize(("texture", "field"), sorted(_HARMONY_LEVELS.items()))
    def test_a_texture_is_written_to_its_own_level(self, texture: str, field: str) -> None:
        application = apply_deltas(_SPEC, [SetHarmonyLevel(texture=texture, velocity=64)])
        assert application.ok, application.refused
        assert getattr(application.plan, field) == 64
        for other in set(_HARMONY_LEVELS.values()) - {field}:
            assert getattr(application.plan, other) == getattr(_BASE_PLAN, other)


class TestSetMotifVariation:
    def test_the_weights_are_scaled_by_the_factor(self) -> None:
        base = dict(_BASE_PLAN.motif_operation_weights)
        assert base["repeat"] == pytest.approx(0.35), "the piece's own weight is the premise"
        application = apply_deltas(_SPEC, [_EXAMPLES["SetMotifVariation"]])
        assert dict(application.plan.motif_operation_weights)["repeat"] == pytest.approx(0.35 * 1.5)

    def test_the_operations_and_their_order_are_untouched(self) -> None:
        application = apply_deltas(_SPEC, [_EXAMPLES["SetMotifVariation"]])
        assert [name for name, _ in application.plan.motif_operation_weights] == [
            name for name, _ in _BASE_PLAN.motif_operation_weights
        ]

    def test_a_factor_of_one_is_the_piece_as_it_stands(self) -> None:
        # Deliberately the identity rather than a refusal: the request is honoured,
        # and what it asked for is what the piece already does.
        application = apply_deltas(_SPEC, [SetMotifVariation(factor=1.0)])
        assert application.ok
        assert application.plan == _BASE_PLAN


class TestSetBassMotion:
    def test_the_chosen_motion_leads_the_vocabulary(self) -> None:
        application = apply_deltas(_SPEC, [_EXAMPLES["SetBassMotion"]])
        assert application.plan.bass_figures[0] == FIGURES_BY_MOTION[BassMotion.PEDAL]

    def test_the_rest_of_the_pieces_own_vocabulary_is_kept_behind_it(self) -> None:
        chosen = FIGURES_BY_MOTION[BassMotion.PEDAL]
        assert chosen not in _BASE_PLAN.bass_figures, "this piece must not already hold the pedal"
        application = apply_deltas(_SPEC, [_EXAMPLES["SetBassMotion"]])
        assert application.plan.bass_figures == (
            chosen,
            *_BASE_PLAN.bass_figures,
        )

    def test_a_motion_the_vocabulary_already_leads_with_is_the_identity(self) -> None:
        # Calming's own vocabulary leads with the root, so this asks for what is
        # already true. Honoured rather than refused — the alternative would refuse
        # a request the user cannot tell is redundant.
        assert FIGURES_BY_MOTION[BassMotion.ROOT] == _BASE_PLAN.bass_figures[0]
        application = apply_deltas(_SPEC, [SetBassMotion(motion=BassMotion.ROOT)])
        assert application.ok
        assert application.plan == _BASE_PLAN


class TestSetSectionClose:
    def test_the_close_the_piece_already_ends_on_is_the_identity(self) -> None:
        # The shipped default closes on the half cadence, so this asks for what
        # the piece already does. Honoured rather than refused, like the other
        # identities here: the request is true, so the piece is unchanged.
        assert _BASE_PLAN.section_close == "half"
        application = apply_deltas(_SPEC, [SetSectionClose(close="half")])
        assert application.ok
        assert application.plan == _BASE_PLAN

    def test_a_close_outside_the_plans_own_list_is_refused_by_the_plan(self) -> None:
        (refusal,) = apply_deltas(_SPEC, [SetSectionClose(close="hanging")]).refused
        assert refusal.reason == "violates_the_plan"
        assert "SetSectionClose(close=hanging)" in refusal.message
        # The plan's own sentence names the closes it knows, which is why the
        # delta does not restate them — and why no `nearest` is invented beside
        # a message that already lists the alternatives.
        assert "hold, half, full" in refusal.message
        assert refusal.nearest is None


class TestAPlanDeltaWritesThePlan:
    @pytest.mark.parametrize("knob", sorted(_PLAN_EFFECTS))
    def test_the_named_fields_end_up_holding_the_requested_values(self, knob: str) -> None:
        application = apply_deltas(_SPEC, [_EXAMPLES[knob]])
        assert application.ok, application.refused
        for field, expected in _PLAN_EFFECTS[knob].items():
            assert getattr(application.plan, field) == expected

    @pytest.mark.parametrize("knob", sorted(_PLAN_KNOBS))
    def test_a_plan_delta_leaves_the_spec_alone(self, knob: str) -> None:
        assert apply_deltas(_SPEC, [_EXAMPLES[knob]]).spec is _SPEC


class TestTheChainFolds:
    def test_an_empty_chain_is_the_piece_the_default_plan_describes(self) -> None:
        application = apply_deltas(_SPEC, [])
        assert application.ok
        assert application.applied == ()
        assert application.refused == ()
        assert application.spec is _SPEC
        assert application.plan == _BASE_PLAN

    def test_the_spec_pass_and_the_plan_pass_commute(self) -> None:
        chain = [SetMood(mood=Mood.SLEEP), _EXAMPLES["SetBassMotion"], SetTempo(tempo_bpm=90)]
        forwards = apply_deltas(_SPEC, chain)
        backwards = apply_deltas(_SPEC, list(reversed(chain)))
        assert forwards.spec == backwards.spec
        assert forwards.plan == backwards.plan

    def test_a_plan_delta_survives_a_mood_change(self) -> None:
        # The whole reason the plan is a patch on the spec rather than a second
        # substrate: the mood re-derives the plan, so a delta applied to a parent's
        # materialized plan would have been overwritten by tables the new mood owns.
        sleep = _SPEC.model_copy(update={"mood": Mood.SLEEP})
        derived = default_plan(sleep)
        assert derived.rhythm_weights != _BASE_PLAN.rhythm_weights, "the mood must move something"
        application = apply_deltas(_SPEC, [_EXAMPLES["SetBassMotion"], SetMood(mood=Mood.SLEEP)])
        assert application.plan.bass_figures[0] == FIGURES_BY_MOTION[BassMotion.PEDAL]
        assert application.plan.rhythm_weights == derived.rhythm_weights
        assert application.plan.tie_probability == derived.tie_probability

    def test_within_the_spec_the_last_word_wins_and_both_are_honoured(self) -> None:
        application = apply_deltas(
            _SPEC,
            [SetTempo(tempo_bpm=90), SetDuration(duration_seconds=60), SetTempo(tempo_bpm=120)],
        )
        assert application.spec.tempo_bpm == 120
        assert application.spec.duration_seconds == 60
        assert len(application.applied) == 3, "the earlier tempo was honoured, then overridden"

    def test_within_the_plan_every_patch_stands(self) -> None:
        application = apply_deltas(
            _SPEC, [_EXAMPLES["SetCadence"], _EXAMPLES["SetModulation"], _EXAMPLES["SetIntroBars"]]
        )
        assert application.plan.cadence_degree == 4
        assert application.plan.cadence_seventh is True
        assert application.plan.modulation_offset == 5
        assert application.plan.intro_bars == 3

    def test_the_plan_is_derived_once_from_the_final_spec(self) -> None:
        # The fold's shape, which the documents cannot show: a version that re-derived
        # the plan after every delta would produce these same two documents for every
        # chain in this file. So the count is posed — and the spec it was handed is
        # asserted to be the chain's *last* word rather than any intermediate.
        seen: list[CompositionSpec] = []
        real = deltas.default_plan

        def counting(spec: CompositionSpec) -> CompositionPlan:
            seen.append(spec)
            return real(spec)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(deltas, "default_plan", counting)
            chain = [
                SetTempo(tempo_bpm=90),
                _EXAMPLES["SetBassMotion"],
                SetMood(mood=Mood.SLEEP),
                SetDuration(duration_seconds=60),
            ]
            application = apply_deltas(_SPEC, chain)

        assert len(seen) == 1
        assert seen[0].mood is Mood.SLEEP
        assert seen[0].tempo_bpm == 90
        assert application.plan.bass_figures[0] == FIGURES_BY_MOTION[BassMotion.PEDAL]

    def test_the_fold_is_a_pure_function_of_the_spec_and_the_chain(self) -> None:
        chain = [
            SetMood(mood=Mood.SLEEP),
            _EXAMPLES["SetBassMotion"],
            _EXAMPLES["SetSectionEnergy"],
        ]
        first = apply_deltas(_SPEC, chain)
        second = apply_deltas(_SPEC, chain)
        assert first.spec == second.spec
        assert first.plan == second.plan

    def test_the_spec_it_was_handed_is_not_touched(self) -> None:
        before = _SPEC.model_dump()
        apply_deltas(_SPEC, [_EXAMPLES["SetInstrument"], SetTempo(tempo_bpm=90)])
        assert _SPEC.model_dump() == before

    def test_applied_and_refused_partition_the_chain_in_order(self) -> None:
        chain = [
            SetTempo(tempo_bpm=90),
            SetTempo(tempo_bpm=1000),
            SetDuration(duration_seconds=60),
            SetIntroBars(bars=16),
            _EXAMPLES["SetCadence"],
        ]
        application = apply_deltas(_SPEC, chain)
        assert application.applied == (chain[0], chain[2], chain[4])
        assert not application.ok


class TestARefusalNamesItsReason:
    @pytest.mark.parametrize(
        ("delta", "nearest"),
        [
            (SetTempo(tempo_bpm=1000), "SetTempo(240)"),
            (SetTempo(tempo_bpm=10), "SetTempo(40)"),
            (SetDuration(duration_seconds=10), "SetDuration(30)"),
            (SetDuration(duration_seconds=10_000), "SetDuration(600)"),
            (ReRoll(seed=-1), "ReRoll(0)"),
        ],
    )
    def test_a_failed_bound_names_the_nearest_legal_request(
        self, delta: Delta, nearest: str
    ) -> None:
        (refusal,) = apply_deltas(_SPEC, [delta]).refused
        assert refusal.reason == "violates_the_spec"
        assert refusal.nearest == nearest

    def test_the_message_carries_the_request_and_the_documents_own_words(self) -> None:
        (refusal,) = apply_deltas(_SPEC, [SetTempo(tempo_bpm=1000)]).refused
        assert "SetTempo(tempo_bpm=1000)" in refusal.message
        assert "less than or equal to 240" in refusal.message

    def test_a_plan_bound_refuses_with_the_plans_own_sentence(self) -> None:
        (refusal,) = apply_deltas(_SPEC, [SetIntroBars(bars=16)]).refused
        assert refusal.reason == "violates_the_plan"
        assert "SetIntroBars(bars=16)" in refusal.message
        assert "must be shorter than the shortest form" in refusal.message

    def test_a_plan_refusal_whose_message_names_the_alternative_offers_no_nearest(self) -> None:
        (refusal,) = apply_deltas(_SPEC, [SetDrumStyle(name="polka")]).refused
        assert refusal.reason == "violates_the_plan"
        assert "ballad" in refusal.message, "the known styles are the alternative"
        assert refusal.nearest is None

    def test_a_refused_delta_changes_nothing_and_the_chain_carries_on(self) -> None:
        application = apply_deltas(
            _SPEC,
            [
                SetTempo(tempo_bpm=90),
                SetTempo(tempo_bpm=1000),
                SetDuration(duration_seconds=60),
                _EXAMPLES["SetCadence"],
            ],
        )
        assert application.spec.tempo_bpm == 90
        assert application.spec.duration_seconds == 60
        assert application.plan.cadence_degree == 4
        assert not application.ok

    def test_a_chain_that_honours_nothing_is_still_a_piece(self) -> None:
        # The applier is not a legality gate and not a composer: it folds what it
        # can and reports the rest, which is the same shape a fan-out has for a
        # candidate the engine refused.
        application = apply_deltas(
            _SPEC,
            [SetTempo(tempo_bpm=1000), SetIntroBars(bars=16), SetDrumStyle(name="polka")],
        )
        assert not application.ok
        assert application.spec == _SPEC
        assert application.plan == default_plan(_SPEC)

    def test_a_refusal_names_the_request_and_the_failing_kind(self) -> None:
        (refusal,) = apply_deltas(_SPEC, [SetTempo(tempo_bpm=1000)]).refused
        assert refusal == DeltaRefusal(
            request="SetTempo",
            reason="violates_the_spec",
            message=refusal.message,
            nearest="SetTempo(240)",
        )


class TestARefusalReadAsAWholeSentence:
    """`refusal_line`: for the two callers where a refusal *is* the answer.

    The conductor's `revise` when it honoured none of its requests, and the
    studio's feedback box when it read nothing it could act on — both answer with
    one refusal rather than a list, so `nearest` has to be folded into the
    sentence instead of being rendered beside it. Everywhere else the two travel
    apart, which is why this is a function and not a field.
    """

    def test_a_failed_bound_is_followed_by_the_request_that_satisfies_it(self) -> None:
        """ "Less than or equal to 240" is a bound; `SetTempo(240)` is the request."""
        (refusal,) = apply_deltas(_SPEC, [SetTempo(tempo_bpm=1000)]).refused

        assert refusal_line(refusal) == (
            f"{refusal.message} The nearest request this piece can honour is SetTempo(240)."
        )

    def test_a_refusal_with_no_nearest_stays_its_own_sentence(self) -> None:
        """A message that already names the alternative is not given a second one."""
        (refusal,) = apply_deltas(_SPEC, [SetDrumStyle(name="polka")]).refused

        assert refusal.nearest is None
        assert refusal_line(refusal) == refusal.message


class TestTheUnbuiltKnobs:
    @pytest.mark.parametrize("unbuilt", [entry.request for entry in UNCARRIED])
    def test_an_unbuilt_request_refuses_by_name(self, unbuilt: str) -> None:
        refusal = refuse_uncarried(unbuilt)
        assert refusal is not None
        assert refusal.reason == "unknown_knob"
        assert refusal.request == unbuilt
        assert refusal.message.startswith(f"{unbuilt} is not a knob this engine carries yet")
        assert refusal.nearest

    @pytest.mark.parametrize("unbuilt", [entry.request for entry in UNCARRIED])
    def test_an_unbuilt_request_offers_an_alternative_that_exists(self, unbuilt: str) -> None:
        (entry,) = [each for each in UNCARRIED if each.request == unbuilt]
        assert any(name in entry.instead for name in DELTA_TYPES), (
            "an alternative that names no request is not an alternative"
        )

    def test_the_unbuilt_requests_are_the_five_the_scope_rule_left_out(self) -> None:
        # A ratchet, not evidence: the design's Tier-2 list named these, plan v1 has
        # no knob for them, and re-opening or shortening the table should be a
        # declared edit rather than a quiet one.
        assert {entry.request for entry in UNCARRIED} == {
            "SetRegister",
            "SetHarmonicRhythm",
            "SetSwing",
            "SetDrumEntry",
            "ExtendSection",
        }

    @pytest.mark.parametrize("knob", list(DELTA_TYPES))
    def test_a_knob_the_engine_carries_is_not_reported_as_unbuilt(self, knob: str) -> None:
        assert refuse_uncarried(knob) is None

    def test_a_word_outside_the_table_is_not_an_unbuilt_request(self) -> None:
        # The distinction the table exists for: an unbuilt request was understood and
        # is unbuilt, where an unknown word was not understood at all. A caller that
        # could not tell them apart would answer one with the other's sentence.
        assert refuse_uncarried("MakeItSoundBetter") is None


class TestTheStoredDocument:
    @pytest.mark.parametrize("knob", list(_EXAMPLES))
    def test_a_delta_round_trips_through_its_document(self, knob: str) -> None:
        delta = _EXAMPLES[knob]
        assert delta_from_dict(delta_to_dict(delta)) == delta

    @pytest.mark.parametrize("knob", list(_EXAMPLES))
    def test_the_document_carries_plain_values_and_no_type_names(self, knob: str) -> None:
        document = delta_to_dict(_EXAMPLES[knob])
        assert document["knob"] == knob
        assert not any(isinstance(value, StrEnum) for value in document.values())

    def test_a_document_written_by_strings_rebuilds_the_same_delta(self) -> None:
        # The log, the tool call and the UI all arrive as JSON, and the enums are
        # held as members on construction — which is what makes the round trip
        # compare equal rather than becoming two entries for one request.
        assert delta_from_dict({"knob": "SetMood", "mood": "sleep"}) == SetMood(mood=Mood.SLEEP)
        rebuilt = delta_from_dict(
            {"knob": "SetInstrument", "role": "melody", "instrument": "flute"}
        )
        assert rebuilt == _EXAMPLES["SetInstrument"]
        assert rebuilt.describe() == _EXAMPLES["SetInstrument"].describe()

    def test_a_string_field_is_held_as_the_member_it_names(self) -> None:
        delta = SetMood(mood="sleep")
        assert delta.mood is Mood.SLEEP
        assert SetInstrument(role="melody", instrument="flute").role is VoiceRole.MELODY
        assert SetKey(key="G").key is WesternKey.G_MAJOR

    def test_a_name_outside_an_enum_is_refused_at_construction(self) -> None:
        with pytest.raises(ValueError):
            SetMood(mood="grumpy")

    @pytest.mark.parametrize(
        ("build", "named"),
        [
            (lambda: SetHumanization(level="loudish"), "'none', 'light', 'expressive'"),
            (lambda: SetHarmonyLevel(texture="strings", velocity=64), "'pad'"),
            (lambda: SetSectionEnergy(role="verse", factor=2.0), "'opening', 'peak'"),
        ],
    )
    def test_a_value_outside_a_literal_is_refused_at_construction(
        self, build: Callable[[], Delta], named: str
    ) -> None:
        """A `Literal` is a promise the interpreter does not keep, so the record does.

        Found by using the type rather than by reading it: `SetSectionEnergy`'s
        terrace names a *plan field* and `SetHarmonyLevel`'s texture names a
        *level field*, so an unknown one raised an `AttributeError` and a
        `KeyError` from inside the fold — a `tool_crashed` reaching the user
        where the vocabulary's whole rule is a named refusal. The three fields
        are the `Literal`s the vocabulary has, and the members named in the
        message are read off the alias rather than restated, so a fourth texture
        is offered without an edit here.
        """
        with pytest.raises(ValueError) as caught:
            build()

        assert named in str(caught.value)

    def test_a_document_naming_an_unknown_knob_is_refused_with_the_vocabulary(self) -> None:
        with pytest.raises(ValueError) as caught:
            delta_from_dict({"knob": "SetVibe", "vibe": "good"})
        assert "SetVibe" in str(caught.value)
        assert "SetTempo" in str(caught.value)

    def test_a_document_with_no_knob_is_refused(self) -> None:
        with pytest.raises(ValueError):
            delta_from_dict({"tempo_bpm": 90})


class TestTheChainMovesTheMusic:
    def test_the_applied_plan_is_the_plan_the_engine_composes_under(self) -> None:
        application = apply_deltas(
            _SPEC, [_EXAMPLES["SetBassMotion"], _EXAMPLES["SetAccompanimentDensity"]]
        )
        output = compose(application.spec, plan=application.plan)
        assert output.plan is application.plan
        assert output.notation_score.compute_hash() != compose(_SPEC).notation_score.compute_hash()

    def test_a_draft_that_stores_its_chain_rebuilds_the_piece_it_heard(self) -> None:
        chain = [SetMood(mood=Mood.SLEEP), _EXAMPLES["SetCadence"], _EXAMPLES["SetSectionEnergy"]]
        first = apply_deltas(_SPEC, chain)
        second = apply_deltas(_SPEC, chain)
        assert compose(first.spec, plan=first.plan).notation_score.compute_hash() == (
            compose(second.spec, plan=second.plan).notation_score.compute_hash()
        )

    def test_the_application_carries_the_pair_the_pair_of_documents_it_folded(self) -> None:
        application = apply_deltas(_SPEC, [_EXAMPLES["SetCadence"]])
        assert application == DeltaApplication(
            spec=application.spec,
            plan=application.plan,
            applied=(_EXAMPLES["SetCadence"],),
            refused=(),
        )


class TestTheTempoTheEngineTradedAway:
    """The one request whose honour is only knowable from a composition.

    Every other refusal in this module is a document's own bound, which is why the
    applier can raise it while folding. A tempo is not: `arrange_for_duration`
    declares a pinned tempo it cannot fit unfulfillable and then catches its own
    refusal, because the length is a release gate and outranks the tempo. That
    policy is deliberate — it is asserted in `test_compose_duration.py` — and it
    leaves a request the user made answered with a piece that does not do it.

    So the check takes the arrangement, and the arrangement only exists where a
    composition happened. These cases build one with the real search rather than a
    hand-made struct, because the question being asked is what the search does.
    """

    def _arrangement(self, *, duration_seconds: float, tempo_bpm: float) -> DurationArrangement:
        return arrange_for_duration(
            mood="calming",
            target_duration_seconds=duration_seconds,
            time_signature="4/4",
            tempo_bpm=tempo_bpm,
        )

    def test_a_tempo_the_arrangement_kept_is_not_refused(self) -> None:
        """The half without which the check would refuse every tempo there is."""
        spec = CompositionSpec(mood=Mood.CALMING, duration_seconds=60)
        arrangement = self._arrangement(duration_seconds=60.0, tempo_bpm=80.0)
        assert arrangement.tempo_bpm == 80.0, "the premise: the search kept the pin"
        assert swallowed_tempo([SetTempo(80)], spec=spec, arrangement=arrangement) is None

    def test_a_tempo_the_length_outranked_is_refused_with_both_numbers(self) -> None:
        spec = CompositionSpec(mood=Mood.CALMING, duration_seconds=30)
        arrangement = self._arrangement(duration_seconds=30.0, tempo_bpm=96.0)
        refusal = swallowed_tempo([SetTempo(96)], spec=spec, arrangement=arrangement)
        assert refusal is not None
        assert (refusal.request, refusal.reason) == ("SetTempo", "tempo_not_honoured")
        assert "SetTempo(96)" in refusal.message, "the request"
        assert f"{arrangement.tempo_bpm:g} BPM" in refusal.message, "and what it plays instead"
        assert "50-80 BPM" in refusal.message, "and the range that would have reached it"
        assert refusal.nearest == f"SetTempo({int(arrangement.tempo_bpm)})"

    def test_a_chain_is_judged_by_the_last_tempo_it_asks_for(self) -> None:
        """The spec ends up carrying the last request, so that is the one answered."""
        spec = CompositionSpec(mood=Mood.CALMING, duration_seconds=30)
        arrangement = self._arrangement(duration_seconds=30.0, tempo_bpm=96.0)
        refusal = swallowed_tempo([SetTempo(70), SetTempo(96)], spec=spec, arrangement=arrangement)
        assert refusal is not None
        assert "SetTempo(96)" in refusal.message
        assert "SetTempo(70)" not in refusal.message

    def test_a_half_bpm_arrangement_names_no_request_to_make(self) -> None:
        """`SetTempo` takes an int, so a 53.5 BPM piece has no nearest to offer.

        Naming one would be advice the vocabulary cannot express — a round trip
        through `delta_from_dict` would refuse it for not being an integer — which
        is the same reason the field is documented as nullable at all.
        """
        spec = CompositionSpec(mood=Mood.CALMING, duration_seconds=180)
        arrangement = self._arrangement(duration_seconds=180.0, tempo_bpm=50.0)
        assert not arrangement.tempo_bpm.is_integer(), "the premise: the search landed on a half"
        refusal = swallowed_tempo([SetTempo(50)], spec=spec, arrangement=arrangement)
        assert refusal is not None
        assert refusal.nearest is None

    def test_a_chain_that_asks_for_no_tempo_has_nothing_to_judge(self) -> None:
        spec = CompositionSpec(mood=Mood.CALMING, duration_seconds=30)
        arrangement = self._arrangement(duration_seconds=30.0, tempo_bpm=96.0)
        assert (
            swallowed_tempo([_EXAMPLES["SetBassMotion"]], spec=spec, arrangement=arrangement)
            is None
        )
