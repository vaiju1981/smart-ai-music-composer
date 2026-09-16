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
it is weaker than it looks: at this phase nothing reads the plan, so the
defaults agree with the tables by construction. It guards a hand-edit
that replaces a table read with a typed-in number. What actually proves
`compose(spec)` still produces today's music is the frozen hash corpus in
`tests/fixtures/golden_hashes.json`, which no plan field can reach yet.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import fields, replace
from typing import Any

import pytest

from saimc.canonical import canonical_dumps
from saimc.compose import forms, motif, percussion
from saimc.compose import plan as plan_module
from saimc.compose.plan import (
    PLAN_SCHEMA_VERSION,
    CompositionPlan,
    PlanError,
    default_plan,
    resolve_plan,
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
    "bass_figures": tuple(reversed(_base().bass_figures)),
    "cadence_degree": 4,
    "form_sizes": (4, 8, 16, 32),
    "intro_bars": 3,
    "max_repeats": 9,
    "duration_tolerance": 0.03,
    "ritardando_factor": 0.80,
    "ritardando_bars": 3,
    "arc_min_reps": 4,
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
        assert plan.rotation_cycle == percussion.ROTATION_CYCLE
        assert plan.section_crash_velocity == percussion.SECTION_CRASH_VELOCITY
        assert plan.cadence_degree == forms.cadence_degree_for("calming")

    @pytest.mark.parametrize("mood", MOODS)
    def test_the_mood_knobs_are_the_moods_row(self, mood: str) -> None:
        plan = default_plan(CompositionSpec(mood=mood, duration_seconds=120, seed=7))
        assert plan.rhythm_weights == tuple(motif.RHYTHM_WEIGHTS[mood].items())
        assert plan.bass_figures == motif.BASS_FIGURES[mood]
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
            (
                {"motif_operation_weights": (("repeat", 0.5), ("repeat", 0.5))},
                "twice",
            ),
            ({"bass_figures": ()}, "at least one figure"),
            ({"bass_figures": ((),)}, "at least one note"),
            ({"cadence_degree": 9}, "out of the scale"),
            ({"form_sizes": ()}, "must not be empty"),
            ({"form_sizes": (8, 0)}, "positive number of bars"),
            ({"intro_bars": -1}, "not be negative"),
            ({"max_repeats": 0}, "at least one repeat"),
            ({"duration_tolerance": 0.0}, r"\(0, 1\)"),
            ({"duration_tolerance": 1.0}, r"\(0, 1\)"),
            ({"ritardando_factor": 0.0}, r"\(0, 1\]"),
            ({"ritardando_factor": 1.5}, r"\(0, 1\]"),
            ({"ritardando_bars": -1}, "not be negative"),
            ({"arc_min_reps": 0}, "at least one repeat"),
            ({"line_band_semitones": 0}, "positive"),
            ({"drum_style_name": "theremin"}, "unknown drum style"),
            ({"rotation_cycle": ()}, "must not be empty"),
            ({"rotation_cycle": (-1, 0)}, "cannot be negative"),
            ({"percussion_velocity_scale": 0.0}, "positive"),
            ({"section_crash_velocity": 0}, "positive"),
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
            "PLAN_FORMAT_PREFIX",
            "PLAN_SCHEMA_VERSION",
            "CompositionPlan",
            "PlanError",
            "default_plan",
            "resolve_plan",
        ]
