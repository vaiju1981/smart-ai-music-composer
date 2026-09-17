"""The composition plan: its defaults, its canonical form, its refusals.

The plan exists so that `(plan, seed) -> notes` can be a claim about an
artifact rather than a claim about the module tables. That only works if
the artifact is **complete** — every knob a value, nothing left for the
engine to look up elsewhere — and if its digest covers every one of those
values. So the load-bearing tests here are the two that read from one
table: that the field set is exactly the one expected, and that changing
any single field changes the hash. A field missing from
`to_canonical_dict` would otherwise be a field a stored plan silently
does not pin, which is the whole failure this type is built to prevent.

`test_the_defaults_describe_todays_engine` is the other kind of test, and
it is weaker than it looks: the defaults agree with the tables by
construction, so what it catches is a hand-edit that replaces a table read
with a typed-in number. What proves `compose(spec)` still produces today's
music is the frozen hash corpus in `tests/fixtures/golden_hashes.json`.

That corpus proves only that nothing broke. It cannot prove the plan is
*read* — a `compose` that ignored its `plan` argument would satisfy every
one of its hashes, since it composes with no plan at all. The other half of
the argument lives in `tests/unit/test_compose_plan_seam.py`, which
composes under non-default plans and asserts the output moves.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import fields, replace
from typing import Any

import pytest

from saimc.canonical import canonical_dumps
from saimc.compose import forms, motif, percussion, voices
from saimc.compose import plan as plan_module
from saimc.compose.duration import (
    DEFAULT_ARRANGEMENT_KNOBS,
    DEFAULT_SECTION_ARC,
    MAX_REPEATS,
)
from saimc.compose.plan import (
    PLAN_FORMAT,
    PLAN_SCHEMA_VERSION,
    CompositionPlan,
    PlanError,
    default_plan,
    resolve_plan,
)
from saimc.compose.voices import (
    BROKEN_CHORD_MOODS,
    DEFAULT_HARMONY_VOICES,
)
from saimc.instruments import LINE_BAND_SEMITONES
from saimc.spec import CompositionSpec

MOODS = ("calming", "electrifying", "sleep")
DURATIONS = (30, 180, 600)
KEYS = ("C", "Am", "G")
TIME_SIGNATURES = ("4/4", "3/4", "6/8", "5/4", "7/8")
"""Every meter the spec allows.

The exotic ones are here because they are the only way `drum_style_name`
is None, and a plan that named a kit for 7/8 would have the percussion
voice rendered onto a meter no style claims.
"""


def _spec_matrix() -> list[CompositionSpec]:
    """A grid over the axes the plan's defaults branch on."""
    return [
        CompositionSpec(
            mood=mood,
            duration_seconds=duration,
            seed=index,
            key=key,
            time_signature=time_signature,
        )
        for index, (mood, duration, key, time_signature) in enumerate(
            (mood, duration, key, time_signature)
            for mood in MOODS
            for duration in DURATIONS
            for key in KEYS
            for time_signature in TIME_SIGNATURES
        )
    ]


_CALMING = CompositionSpec(mood="calming", duration_seconds=120, seed=7)


def _base() -> CompositionPlan:
    return default_plan(_CALMING)


_MUTATIONS: dict[str, Any] = {
    "format": f"CompositionPlan:{PLAN_SCHEMA_VERSION + 1}",
    # Reversed rather than truncated: the choices and the weights have to
    # stay the same length as each other, and a mutation that trips an
    # invariant would prove nothing about the hash. `step_weights` is
    # swapped rather than reversed because the table it comes from is a
    # palindrome, so reversing it would be a mutation in name only.
    "step_choices": tuple(reversed(_base().step_choices)),
    "step_weights": (5, 3, 10, 28, 6, 28, 10, 5, 3),
    "max_motif_span_degrees": 9,
    "leap_degrees": 4,
    "chord_tone_degrees": 3,
    "motif_operation_weights": (
        ("repeat", 0.30),
        ("transpose", 0.20),
        ("sequence", 0.10),
        ("invert", 0.15),
        ("truncate", 0.10),
    ),
    "rhythm_weights": (("straight", 0.40), ("dotted", 0.30), ("tie", 0.30)),
    # The base is calming, whose ties are drawn at 0.28, so this is a
    # change and not a restatement of the value the mood already holds.
    "tie_probability": 0.40,
    # Apex before the halfway point rather than after it: the base's 0.6 is
    # the late peak, and a mutation has to move the phrase's shape.
    "apex_position": 0.45,
    "bass_figures": tuple(reversed(_base().bass_figures)),
    "cadence_degree": 4,
    # The base is calming, whose cadence is a plain triad, so `True` is the
    # change and the electrifying V7 is the value it does not already hold.
    "cadence_seventh": True,
    "modulation_offset": 3,
    "form_sizes": (4, 8, 16, 32),
    "intro_bars": 3,
    # Lowered, not raised: §10 #10 caps how often a source section may
    # repeat, so 8 (MAX_REPEATS) is the ceiling and the plan may only
    # tighten it. `test_a_plan_above_the_repetition_cap_is_refused`
    # covers the other direction.
    "max_repeats": 7,
    "duration_tolerance": 0.03,
    "ritardando_factor": 0.80,
    "ritardando_bars": 3,
    "arc_min_reps": 4,
    "section_energy_opening": 0.70,
    "section_energy_peak": 1.20,
    "section_energy_final": 0.90,
    "section_energy_middle": 1.05,
    # The cycle has to keep the arc's own shape to be a mutation of it
    # rather than a differently-lengthed one: four phases, with the two
    # thinning phases swapped so the breakdown arrives second.
    "harmony_texture_cycle": ("all", "rest", "first", "all"),
    "percussion_rest_section": 2,
    # The base is calming, whose harmony sustains, so a broken chord is the
    # change — as it is for the plan the electrifying mood would default to.
    "harmony_broken_chord": True,
    # A quarter rather than the eighth note the broken-chord figure steps on
    # by default: half the onsets per bar, which is the density itself.
    "harmony_arpeggio_step_ticks": 480,
    "harmony_pad_velocity": 50,
    "harmony_arpeggio_velocity": 56,
    "harmony_stab_velocity": 62,
    "harmony_melody_clearance": 4,
    "line_band_semitones": 22,
    "drum_style_name": "funk",
    "rotation_cycle": (0, 1, 0),
    "percussion_velocity_scale": 0.8,
    "section_crash_velocity": 91,
}
"""One valid, different value per field.

Doubles as the field inventory: `test_every_field_is_in_the_inventory`
fails when a field is added without a mutation, so the two guards below
cannot silently stop covering a new field. Every value has to satisfy
`__post_init__`, or the mutation would raise before the hash was taken.
"""


class TestTheCanonicalHashes:
    """What a stored plan pins, and how completely."""

    def test_every_field_is_in_the_inventory(self) -> None:
        declared = {field.name for field in fields(CompositionPlan)}
        assert declared == set(_MUTATIONS), (
            "a plan field has no mutation, so nothing below asserts that the "
            "canonical hash covers it. Add one to _MUTATIONS."
        )

    @pytest.mark.parametrize("field_name", sorted(_MUTATIONS))
    def test_changing_one_field_changes_the_hash(self, field_name: str) -> None:
        """The digest covers every field, not just the ones it remembers.

        A field left out of `to_canonical_dict` hashes to the same value
        as its default, so two plans that differ in it would store alike
        and replay alike — silently, with the manifest still claiming the
        plan was pinned.
        """
        base = _base()
        changed = replace(base, **{field_name: _MUTATIONS[field_name]})
        assert changed.compute_hash() != base.compute_hash(), (
            f"{field_name} is not reflected in the canonical hash, so a stored "
            "plan does not pin it"
        )

    def test_two_identical_plans_hash_alike(self) -> None:
        """The other direction, without which the test above is satisfied
        by a hash that simply never repeats."""
        for spec in _spec_matrix():
            assert default_plan(spec).compute_hash() == default_plan(spec).compute_hash()

    def test_the_canonical_form_is_json_a_canonical_document(self) -> None:
        """`canonical_dumps` is the serializer that rejects NaN/Infinity and
        sorts keys. A plan it cannot serialize could not be stored, so this
        is the check that the plan is a canonical artifact at all."""
        for spec in _spec_matrix():
            document = default_plan(spec).to_canonical_dict()
            text = canonical_dumps(document)
            assert json.loads(text) == json.loads(canonical_dumps(document))

    def test_the_format_tag_is_present_and_versioned(self) -> None:
        """The §6 rule: an explicit format version in each document.

        Present, not necessarily first. `canonical_dumps` sorts keys, and
        `arc_min_reps`, `bass_figures` and `cadence_degree` all sort
        before `format`, so this document does not lead with its tag the
        way `NotationScore` does (whose key set happens to sort after it).
        `canonical.manifest_document`'s docstring claims the tag is always
        first; that holds for the documents written so far and not for
        this one, so the claim is load-bearing only as an accident.
        """
        document = _base().to_canonical_dict()
        assert document["format"] == f"CompositionPlan:{PLAN_SCHEMA_VERSION}"
        assert '"format":"CompositionPlan:' in canonical_dumps(document)


_ROUND_TRIP_SPECS = _spec_matrix()
"""Every cell of the plan's own grid, because the reader is field-by-field.

A spot check would cover the fields the cells happen to differ on, which
is not the same set: `drum_style_name` only moves with the meter and
`percussion_velocity_scale` only with the mood.
"""


class TestTheStoredPlanIsReadBack:
    """The canonical document survives a round trip through storage.

    `to_canonical_dict` is what gets written; `from_canonical_dict` is what
    reads it back. The pair has to name the same fields, and nothing else
    in the suite would notice if it did not — the writer's own tests only
    look at what it writes.
    """

    @pytest.mark.parametrize(
        "spec",
        _ROUND_TRIP_SPECS,
        ids=lambda s: f"{s.mood.value}-{s.duration_seconds}-{s.key}-{s.time_signature.value}",
    )
    def test_the_document_round_trips_to_a_plan_that_hashes_alike(
        self, spec: CompositionSpec
    ) -> None:
        """Equality over every field, and equality of the digest.

        Equality is what catches a field the reader forgets: the substitute
        would be accepted by `__post_init__` and the plan would come back
        subtly different. The hash is the form a manifest makes the
        provenance claim in — the digest the plan was stored under is the
        digest a reader recomputes.
        """
        plan = default_plan(spec)
        reloaded = CompositionPlan.from_canonical_dict(plan.to_canonical_dict())
        assert reloaded == plan
        assert reloaded.compute_hash() == plan.compute_hash()

    def test_the_reloaded_document_survives_a_json_round_trip(self) -> None:
        """Not just equal in memory: the reload survives being written and
        parsed again, which is what a sidecar and a `job.json` both do."""
        plan = _base()
        text = canonical_dumps(plan.to_canonical_dict())
        reloaded = CompositionPlan.from_canonical_dict(json.loads(text))
        assert reloaded == plan
        assert canonical_dumps(reloaded.to_canonical_dict()) == text

    @pytest.mark.parametrize(
        "document_format",
        ["CompositionPlan:2", "CompositionPlan:0", "CompositionPlan", "NotationScore:1", ""],
        ids=["newer", "older", "untagged", "wrong-kind", "absent"],
    )
    def test_a_document_of_another_version_is_refused(self, document_format: str) -> None:
        """Refused exactly, in both directions, rather than coerced.

        An older plan is missing fields a materialized plan cannot do
        without, and a newer one may carry a knob this build would ignore —
        which is the failure §6's explicit version exists to prevent, since
        the plan is the artifact the determinism claim is made about.
        """
        document = _base().to_canonical_dict()
        document["format"] = document_format
        with pytest.raises(PlanError) as exc_info:
            CompositionPlan.from_canonical_dict(document)
        assert PLAN_FORMAT in str(exc_info.value)

    def test_the_refusal_names_the_document_it_could_not_read(self) -> None:
        """The plan type's own rule: refused *with a reason*. A reader
        holding a file needs to know which version it is holding."""
        document = _base().to_canonical_dict()
        document["format"] = "CompositionPlan:7"
        with pytest.raises(PlanError) as exc_info:
            CompositionPlan.from_canonical_dict(document)
        message = str(exc_info.value)
        assert "CompositionPlan:7" in message
        assert PLAN_FORMAT in message

    def test_a_dropped_field_is_not_replaced_by_a_default(self) -> None:
        """The reader has no fallbacks at all, deliberately.

        A fallback to a module table would make a stored plan replay as a
        plan nobody wrote — the thing "materialized, never a delta" rules
        out — so a document missing a field is a `KeyError`, loudly, rather
        than a working plan with a quiet substitution.
        """
        document = _base().to_canonical_dict()
        del document["harmony_pad_velocity"]
        with pytest.raises(KeyError):
            CompositionPlan.from_canonical_dict(document)


class TestTheDefaultsDescribeTodaysEngine:
    """`default_plan` is a description, not a new set of choices.

    Weaker than it reads — see the module docstring. It catches a table
    read replaced by a typed-in number, which is the one way these can
    diverge while nothing consumes the plan.
    """

    def test_the_mood_independent_knobs_are_the_constants(self) -> None:
        plan = _base()
        assert plan.step_choices == motif.STEP_CHOICES
        assert plan.step_weights == motif.STEP_WEIGHTS
        assert plan.max_motif_span_degrees == motif.MAX_MOTIF_SPAN_DEGREES
        assert plan.leap_degrees == motif.LEAP_DEGREES
        assert plan.chord_tone_degrees == motif.CHORD_TONE_DEGREES
        assert plan.motif_operation_weights == motif.MOTIF_OPERATION_WEIGHTS
        assert plan.form_sizes == forms.PHRASE_SIZES
        assert plan.line_band_semitones == LINE_BAND_SEMITONES
        assert plan.harmony_arpeggio_step_ticks == voices.HARMONY_ARPEGGIO_STEP_TICKS
        assert plan.harmony_pad_velocity == voices.HARMONY_PAD_VELOCITY
        assert plan.harmony_arpeggio_velocity == voices.HARMONY_ARPEGGIO_VELOCITY
        assert plan.harmony_stab_velocity == voices.HARMONY_STAB_VELOCITY
        assert plan.harmony_melody_clearance == voices.HARMONY_MELODY_CLEARANCE
        assert plan.rotation_cycle == percussion.ROTATION_CYCLE
        assert plan.section_crash_velocity == percussion.SECTION_CRASH_VELOCITY
        assert plan.cadence_degree == forms.cadence_degree_for("calming")

    @pytest.mark.parametrize("mood", MOODS)
    def test_the_mood_knobs_are_the_moods_row(self, mood: str) -> None:
        plan = default_plan(CompositionSpec(mood=mood, duration_seconds=120, seed=7))
        assert plan.rhythm_weights == tuple(motif.RHYTHM_WEIGHTS[mood].items())
        assert plan.bass_figures == motif.BASS_FIGURES[mood]
        assert plan.harmony_broken_chord == (mood in voices.BROKEN_CHORD_MOODS)
        assert plan.percussion_velocity_scale == percussion.MOOD_VELOCITY_SCALE.get(
            mood, 1.0
        )
        assert plan.cadence_degree == forms.CADENCE_DEGREE.get(
            mood, forms.DEFAULT_CADENCE_DEGREE
        )

    @pytest.mark.parametrize("time_signature", TIME_SIGNATURES)
    def test_the_meter_decides_the_kit(self, time_signature: str) -> None:
        """3/4 is a waltz whatever the mood; 5/4 and 7/8 have no style at
        all, and the percussion voice is skipped rather than forced."""
        plan = default_plan(
            CompositionSpec(
                mood="calming", duration_seconds=120, seed=1, time_signature=time_signature
            )
        )
        assert plan.drum_style_name == percussion.style_name_for("calming", time_signature)
        if time_signature in percussion.METER_STYLES:
            assert plan.drum_style_name == percussion.METER_STYLES[time_signature]
        if time_signature not in ("4/4", "3/4", "6/8"):
            assert plan.drum_style_name is None

    def test_a_mood_without_a_rhythm_profile_falls_back(self) -> None:
        """The `.get(mood, default)` shape, not `[mood]` — the engine has a
        gentle fallback and the plan has to describe it rather than raise."""
        plan = default_plan(CompositionSpec(mood="calming", duration_seconds=120, seed=1))
        assert plan.rhythm_weights == tuple(
            motif.RHYTHM_WEIGHTS.get("calming", motif.DEFAULT_RHYTHM_WEIGHTS).items()
        )


class TestResolutionIsInvisibleByDefault:
    def test_an_absent_plan_resolves_to_the_default(self) -> None:
        for spec in _spec_matrix():
            assert resolve_plan(spec) == default_plan(spec)
            assert resolve_plan(spec, None) == default_plan(spec)

    def test_a_supplied_plan_is_passed_through_unchanged(self) -> None:
        """Not re-deriveable, not re-stamped: the caller's plan is the
        artifact. Anything else would mean the plan a user edited is not
        the plan the engine composed under."""
        supplied = replace(_base(), line_band_semitones=18)
        assert resolve_plan(_CALMING, supplied) is supplied


class TestTheLayersReadThePlanThroughBridges:
    """A layer reads its knobs through a struct, because of the cycle.

    `plan.py` imports `duration.py` for its own defaults, so `duration.py`
    cannot import `plan.py` back. The struct is the one-way bridge, and it
    is only a bridge if it actually carries the plan's values — a bridge
    that dropped a field would give the engine a default the plan never
    named, silently, while the manifest still claimed the plan was pinned.
    """

    def test_the_default_plan_bridges_to_todays_arrangement(self) -> None:
        """`plan=None` has to reach the very same struct the module's own
        default is, or byte-identity under `plan=None` is a coincidence."""
        for spec in _spec_matrix():
            assert default_plan(spec).arrangement_knobs() == DEFAULT_ARRANGEMENT_KNOBS

    def test_the_bridge_carries_every_arrangement_field(self) -> None:
        """Every `ArrangementKnobs` field, checked by name rather than by
        equality against the default — equality would also hold for a
        bridge that dropped a field and let the struct's default stand."""
        plan = replace(
            _base(),
            form_sizes=(4, 8),
            max_repeats=3,
            duration_tolerance=0.01,
            arc_min_reps=4,
            intro_bars=1,
            ritardando_factor=0.7,
            ritardando_bars=1,
        )
        knobs = plan.arrangement_knobs()
        assert knobs.form_sizes == (4, 8)
        assert knobs.max_repeats == 3
        assert knobs.duration_tolerance == 0.01
        assert knobs.arc_min_reps == 4
        assert knobs.intro_bars == 1
        assert knobs.ritardando_factor == 0.7
        assert knobs.ritardando_bars == 1

    def test_the_default_plan_bridges_to_todays_section_arc(self) -> None:
        for spec in _spec_matrix():
            assert default_plan(spec).section_arc() == DEFAULT_SECTION_ARC

    def test_the_bridge_carries_every_section_field(self) -> None:
        """The same by-name check for `SectionArc`.

        Its three velocity fields are checked as the defaults *are* today,
        because `energy_middle` is 1.0 — the one value whose absence from
        the bridge a comparison against the default would not catch, since
        a dropped field would leave the struct's own default standing.
        """
        plan = replace(
            _base(),
            section_energy_opening=0.7,
            section_energy_peak=1.2,
            section_energy_final=0.9,
            section_energy_middle=1.05,
            harmony_texture_cycle=("all", "rest", "first", "all"),
        )
        arc = plan.section_arc()
        assert arc.energy_opening == 0.7
        assert arc.energy_peak == 1.2
        assert arc.energy_final == 0.9
        assert arc.energy_middle == 1.05
        assert arc.texture_cycle == ("all", "rest", "first", "all")

    def test_the_default_plan_bridges_to_todays_harmony_voices(self) -> None:
        """The texture is the mood's, so the default is not one struct: a
        plan that reached the wrong one would state the other mood's
        accompaniment while the plan itself looked untouched."""
        for spec in _spec_matrix():
            expected = replace(
                DEFAULT_HARMONY_VOICES,
                broken_chord=spec.mood.value in BROKEN_CHORD_MOODS,
            )
            assert default_plan(spec).harmony_voices() == expected

    def test_the_bridge_carries_every_voice_field(self) -> None:
        """The same by-name check for `HarmonyVoices`.

        `broken_chord` is the field a comparison against the default would
        not catch, because the struct's own default is `False` — the value
        a dropped field would leave standing for a mood that wanted a
        broken chord.
        """
        plan = replace(
            _base(),
            harmony_broken_chord=True,
            harmony_arpeggio_step_ticks=480,
            harmony_pad_velocity=50,
            harmony_arpeggio_velocity=56,
            harmony_stab_velocity=62,
            harmony_melody_clearance=4,
        )
        voices = plan.harmony_voices()
        assert voices.broken_chord is True
        assert voices.arpeggio_step_ticks == 480
        assert voices.pad_velocity == 50
        assert voices.arpeggio_velocity == 56
        assert voices.stab_velocity == 62
        assert voices.melody_clearance == 4


class TestAPlanThatCannotBeHonouredIsRefused:
    """Refused with a reason, never silently downgraded.

    The product's rule for a request the engine cannot perform. A plan is
    written by an outside caller — a conductor, an agent, a user — so the
    failure has to be a typed, catchable one rather than a wrong piece.
    """

    @pytest.mark.parametrize(
        ("changes", "fragment"),
        [
            ({"step_choices": (0, 1)}, "same length"),
            ({"step_weights": (-1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)}, "negative"),
            ({"step_weights": (0.0,) * 9}, "some weight"),
            ({"max_motif_span_degrees": 0}, "positive"),
            ({"leap_degrees": 0}, "positive"),
            ({"chord_tone_degrees": 0}, "positive"),
            ({"motif_operation_weights": ()}, "at least one entry"),
            ({"rhythm_weights": ()}, "at least one entry"),
            ({"tie_probability": 1.5}, "probability"),
            ({"tie_probability": -0.1}, "probability"),
            ({"apex_position": 0.0}, "fraction of the section"),
            ({"apex_position": 1.0}, "fraction of the section"),
            (
                {"motif_operation_weights": (("repeat", 0.5), ("repeat", 0.5))},
                "twice",
            ),
            ({"bass_figures": ()}, "at least one figure"),
            ({"bass_figures": ((),)}, "at least one note"),
            ({"cadence_degree": 9}, "out of the scale"),
            ({"modulation_offset": 13}, "cannot exceed an octave"),
            ({"modulation_offset": -13}, "cannot exceed an octave"),
            ({"form_sizes": ()}, "must not be empty"),
            ({"form_sizes": (8, 0)}, "positive number of bars"),
            ({"intro_bars": -1}, "not be negative"),
            ({"intro_bars": 16}, "shorter than the shortest form"),
            ({"max_repeats": 0}, "at least one repeat"),
            ({"max_repeats": MAX_REPEATS + 1}, "may not exceed"),
            ({"duration_tolerance": 0.0}, r"\(0, 1\)"),
            ({"duration_tolerance": 1.0}, r"\(0, 1\)"),
            ({"ritardando_factor": 0.0}, r"\(0, 1\]"),
            ({"ritardando_factor": 1.5}, r"\(0, 1\]"),
            ({"ritardando_bars": -1}, "not be negative"),
            ({"arc_min_reps": 0}, "at least one repeat"),
            ({"section_energy_opening": 0.0}, "must be positive"),
            ({"section_energy_peak": -0.1}, "must be positive"),
            ({"harmony_texture_cycle": ()}, "must not be empty"),
            ({"harmony_texture_cycle": ("all", "quiet")}, "unknown harmony texture"),
            ({"percussion_rest_section": -1}, "cannot be negative"),
            ({"harmony_arpeggio_step_ticks": 0}, "must be positive"),
            ({"harmony_pad_velocity": 0}, "1..127"),
            ({"harmony_pad_velocity": 128}, "1..127"),
            ({"harmony_arpeggio_velocity": 0}, "1..127"),
            ({"harmony_stab_velocity": 200}, "1..127"),
            ({"harmony_melody_clearance": 0}, "must be positive"),
            ({"line_band_semitones": 0}, "positive"),
            ({"drum_style_name": "theremin"}, "unknown drum style"),
            ({"rotation_cycle": ()}, "must not be empty"),
            ({"rotation_cycle": (-1, 0)}, "cannot be negative"),
            ({"percussion_velocity_scale": 0.0}, "positive"),
            ({"section_crash_velocity": 0}, "1..127"),
            ({"section_crash_velocity": 128}, "1..127"),
            ({"format": "SomethingElse:1"}, "CompositionPlan"),
        ],
    )
    def test_it_raises_with_the_reason(self, changes: dict[str, Any], fragment: str) -> None:
        with pytest.raises(PlanError, match=fragment):
            replace(_base(), **changes)

    def test_the_guard_lets_a_valid_plan_through(self) -> None:
        """Without this, a `__post_init__` that raised unconditionally — or a
        message pattern that matched everything — would pass every case
        above."""
        for spec in _spec_matrix():
            assert default_plan(spec) is not None

    @pytest.mark.parametrize("field_name", sorted(_MUTATIONS))
    def test_every_mutation_is_itself_legal(self, field_name: str) -> None:
        """The inventory's values have to be honourable, or the hash tests
        above would be asserting on a refusal rather than on a hash."""
        replace(_base(), **{field_name: _MUTATIONS[field_name]})

    def test_planerror_is_a_valueerror(self) -> None:
        """A caller catching `ValueError` for a bad input catches this too."""
        assert issubclass(PlanError, ValueError)


class TestThePlanCannotReachTheEngine:
    """`plan.py` must not import `engine.py`, which imports it.

    Not a style rule: `engine.py` imports this module, so an import in the
    other direction is a cycle. The import would fail loudly in a fresh
    interpreter, but only in the order that closes it — a run that had
    already loaded `engine` would not notice, which is exactly the
    situation a test suite is in. So this checks a fresh one.
    """

    def test_importing_the_plan_does_not_load_the_engine(self) -> None:
        probe = (
            "import sys; import saimc.compose.plan; "
            "assert 'saimc.compose.engine' not in sys.modules, "
            "'saimc.compose.plan pulled in saimc.compose.engine'; print('ok')"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "ok"

    def test_the_probe_can_fail(self) -> None:
        """A subprocess check that cannot fail guards nothing: this is the
        same probe against a module that does import the engine."""
        probe = (
            "import sys; import saimc.compose.engine; "
            "assert 'saimc.compose.engine' not in sys.modules, 'wrong'; print('ok')"
        )
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode != 0
        assert "wrong" in result.stderr

    def test_plan_module_exposes_what_it_claims(self) -> None:
        assert plan_module.__all__ == [
            "PLAN_FORMAT",
            "PLAN_FORMAT_PREFIX",
            "PLAN_SCHEMA_VERSION",
            "CompositionPlan",
            "PlanError",
            "default_plan",
            "resolve_plan",
        ]
