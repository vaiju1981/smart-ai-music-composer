"""The arbiter's three properties: it is total, its sequence is the argument, and it does not veto an edit.

The tests are written two ways because the module has two kinds of reader.
`arbiter_order` takes values, so each of its six elements is posed directly —
which is the only way to vary an element that a real draft cannot vary, and the
reason the function takes values at all. `rank` and `regression` take records,
so those run on drafts the engine really composed.

One thing is posed on those records, and it is the quality: no composition
lands on "exactly one bar missed, by a fifth of it", and the ratchet's branches
are about exactly those shapes. That is not the hand-built draft C4 declined to
test — there, a production *branch* had no input and could be deleted; here the
shape is one the record type admits and the ratchet must answer for, so the
posed value is an input rather than a fiction.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from saimc.compose.engine import compose
from saimc.compose.plan import default_plan
from saimc.quality import QUALITY_THRESHOLDS, PieceQuality, QualityFinding
from saimc.session.arbiter import (
    ELEMENTS,
    METRIC_TIERS,
    _breach,
    arbiter_order,
    deciding_element,
    draft_key,
    musical_key,
    musical_order,
    rank,
    regression,
)
from saimc.session.models import Draft
from saimc.session.tools import _draft_from
from saimc.spec import CompositionSpec, Mood

_SPEC = CompositionSpec(mood=Mood.CALMING, duration_seconds=30, seed=5)
_ELECTRIFYING = CompositionSpec(mood=Mood.ELECTRIFYING, duration_seconds=30, seed=5)
"""A second plan at the same seed, for the element no measurement can reach."""

_CLEAN: PieceQuality = PieceQuality(
    piece="baseline",
    melody_notes=64,
    melody_bars=16,
    step_ratio=0.60,
    repeat_ratio=0.05,
    leap_recovery_ratio=0.80,
    max_leap_semitones=7,
    range_semitones=12,
    distinct_durations=4,
    texture_hierarchy=1.5,
    register_separation_semitones=6.0,
    tessitura_overlap_semitones=1,
    bass_onset_patterns=4,
)
"""A piece that clears every bar, the starting point for every case.

Every value here is inside its threshold, and that is *asserted* rather than
trusted: a baseline that quietly breached one bar would turn every "one miss"
case into a two-miss case, and the tests would read as passing while measuring
something else.
"""


def _quality(**overrides: float | None) -> PieceQuality:
    """The clean baseline with named metrics moved."""
    return replace(_CLEAN, **overrides)


def _draft(seed: int, **posed: object) -> Draft:
    """A real draft, built the way the `draft` tool builds one.

    The quality is posed by default rather than always, so a case can pose its
    own: `compose`'s report is a measurement of the seed's draw, which is not
    what the ratchet's branches are about.
    """
    spec = _SPEC.model_copy(update={"seed": seed})
    posed.setdefault("quality", _CLEAN)
    return replace(_draft_from(f"draft-{seed}", spec, compose(spec)), **posed)


def _key(quality: PieceQuality) -> tuple[int, int, float, int, str, int]:
    """`arbiter_order` for a quality, with the tiebreakers filled in and constant."""
    return arbiter_order(quality, lint_passed=True, plan_hash="e" * 64, seed=0)


def _rehash_above(parent: Draft) -> Draft:
    """A revision of `parent` that measures identically and whose hash sorts above its own.

    Which *direction* an edit moves a plan hash is data rather than a choice, so
    the child a test of "the tiebreakers are ignored" needs cannot be posed: it
    has to be searched for. A child whose hash sorts *lower* is worthless here,
    because a ratchet that consulted the hash would allow it too.
    """
    for offset in range(-12, 13):
        for degree in range(7):
            child = replace(
                parent,
                plan=replace(parent.plan, modulation_offset=offset, cadence_degree=degree),
                draft_id="draft-1a",
            )
            if child.plan_hash > parent.plan_hash:
                return child
    raise AssertionError("no cadence/offset pair raised the hash above the parent's")


def _draft_of(spec: CompositionSpec, draft_id: str, *, seed: int = 1) -> Draft:
    """A real draft composed at another spec, with the posed clean quality.

    `_draft` is built on one plan, and one element of the order can only be
    reached by two of them — see `_pair_for`. The seed is stamped the same way
    `_draft` stamps it, so the pair differs on its plan and on nothing else.
    """
    stamped = spec.model_copy(update={"seed": seed})
    return replace(_draft_from(draft_id, stamped, compose(stamped)), quality=_CLEAN)


def _elements_before(left: Draft, right: Draft, element: str) -> bool:
    """Whether two drafts tie on every element that decides ahead of `element`.

    This is the premise a case for one element has to carry. Ties *after* the
    named element are inevitable rather than suspicious — a pair differing in how
    many bars it missed differs in the magnitude and the tier too, and the point
    is that it did not have to be asked about either.
    """
    decided = ELEMENTS.index(element)
    return draft_key(left)[:decided] == draft_key(right)[:decided]


_ELEMENT_CASES = ("legality", "misses", "breach", "tier", "plan_hash", "seed")
"""The six elements, once each, as the case for it is named here.

A literal table rather than something read off `ELEMENTS`, for the reason D2
recorded about `_HARMONY_LEVEL_FIELDS`: a case per element is what closes the
hole, and one derived from the tuple under test would close it by agreeing with
it. `test_every_element_has_a_case_of_its_own` is what holds the two together —
`ELEMENTS` gaining a seventh makes this line the failing one.
"""


def _pair_for(element: str) -> tuple[Draft, Draft]:
    """Two drafts differing on `element` and tying exactly on all before it.

    Each pair is chosen for the element it is a case *for*, which is the whole
    difficulty: the breach and the tier are only reached once the count ties, so
    both are built as one miss against one miss, and the tier's two bars are
    missed by the same fraction of themselves — `(24 - 12) / 12` and
    `(48 - 24) / 24` are the same 1.0, and `pytest.approx` would not be enough,
    because a near-tie is not a tie and the element below would decide.
    """
    if element == "legality":
        passed = _draft(1)
        return passed, replace(passed, lint=replace(passed.lint, passed=False))
    if element == "misses":
        return _draft(1), _draft(1, quality=_quality(step_ratio=0.10))
    if element == "breach":
        return (
            _draft(1, quality=_quality(max_leap_semitones=13)),
            _draft(1, quality=_quality(max_leap_semitones=14)),
        )
    if element == "tier":
        return (
            _draft(1, quality=_quality(range_semitones=48)),
            _draft(1, quality=_quality(max_leap_semitones=24)),
        )
    if element == "plan_hash":
        return _draft(1), _draft_of(_ELECTRIFYING, "draft-1b")
    if element == "seed":
        return _draft(1), _draft(2)
    raise AssertionError(f"no case is written for the element {element!r}")


def test_the_clean_baseline_clears_every_bar() -> None:
    """The premise every other case rests on, asserted rather than assumed."""
    assert _CLEAN.findings() == ()


def test_the_tier_table_names_exactly_the_metrics_the_thresholds_do() -> None:
    """A written-down table is what keeps the order current, and this is what keeps it true.

    A derived order would re-order itself the day a threshold moved in the
    file; a written one goes stale the same day. So the two are held together
    here, and a new threshold fails this rather than joining the order ranked.
    """
    assert set(METRIC_TIERS) == {threshold.metric for threshold in QUALITY_THRESHOLDS}
    assert len(METRIC_TIERS) == len(set(METRIC_TIERS))


def test_the_table_is_the_order_it_is_written_as() -> None:
    """The order itself, pinned, so that re-ordering a judgement is a declared edit.

    Not evidence that the order is *right* — nothing can be, because it is a
    judgement — but evidence that changing it was deliberate rather than
    incidental. The two properties above hold for any permutation; without this
    the whole interior of the table is unpinned, and swapping two middle entries
    would move which bar the arbiter calls the higher one with every test still
    green.
    """
    assert METRIC_TIERS == (
        "step_ratio",
        "leap_recovery_ratio",
        "max_leap_semitones",
        "range_semitones",
        "repeat_ratio",
        "distinct_durations",
        "bass_onset_patterns",
        "texture_hierarchy",
        "register_separation_semitones",
        "tessitura_overlap_semitones",
    )


def test_the_default_plan_is_seed_independent() -> None:
    """The premise the seed tiebreaker rests on, asserted rather than believed.

    If a plan carried the seed, a fan-out's candidates would already differ on
    the plan hash and the seed element would be covering a case nothing reaches.
    """
    assert default_plan(_SPEC) == default_plan(_SPEC.model_copy(update={"seed": 99}))


class TestLegality:
    """Element one: nothing trades against it."""

    def test_an_illegal_draft_ranks_below_a_legal_one_that_misses_three_bars(self) -> None:
        """Not "below an equally good one" — below one that missed three."""
        illegal = arbiter_order(_quality(), lint_passed=False, plan_hash="a" * 64, seed=1)
        worse = _quality(step_ratio=0.10, range_semitones=4, max_leap_semitones=20)
        legal = arbiter_order(worse, lint_passed=True, plan_hash="a" * 64, seed=1)

        assert len(worse.findings()) == 3
        assert illegal > legal

    def test_the_element_is_a_value_a_caller_can_pose(self) -> None:
        """Why `arbiter_order` takes `lint_passed` rather than reading a `Draft`.

        `compose` raises on a lint failure, so no draft the tool surface makes
        will ever carry this — and an element nothing can set is an element
        nothing can witness. Taken as an argument it is a value, which is what
        makes the case above possible at all.
        """
        clean = _quality()

        assert arbiter_order(clean, lint_passed=False, plan_hash="h", seed=0) > arbiter_order(
            clean, lint_passed=True, plan_hash="h", seed=0
        )


class TestTheSequenceIsTheArgument:
    """Each element decides only where the one before it left a tie."""

    def test_more_misses_lose_to_fewer_misses(self) -> None:
        one = _quality(step_ratio=0.10)
        three = _quality(step_ratio=0.10, range_semitones=4, max_leap_semitones=20)

        assert len(one.findings()) == 1
        assert len(three.findings()) == 3
        assert _key(three) > _key(one)

    def test_a_smaller_breach_wins_when_the_count_ties(self) -> None:
        """Two misses either way; the one further from its bars loses."""
        shallow = _quality(step_ratio=0.44, repeat_ratio=0.26)
        deep = _quality(step_ratio=0.10, repeat_ratio=0.90)

        assert len(shallow.findings()) == len(deep.findings()) == 2
        assert _key(deep) > _key(shallow)

    def test_the_magnitudes_add_up_rather_than_reducing_to_the_worst(self) -> None:
        """Two misses either side, and the two readings disagree about which piece is better.

        "How far it missed them, as a fraction of *each* bar" is about the misses
        together: a piece over one bar by 0.88 and another by 0.02 has missed
        less in total than one over two bars by half each, and it has the worse
        *single* miss. So `max` is one substitution away from passing every other
        case here while being a different element wearing the same name.
        """
        spread = _quality(step_ratio=0.054, repeat_ratio=0.255)  # 0.88 + 0.02
        level = _quality(step_ratio=0.225, repeat_ratio=0.3675)  # 0.50 + 0.47

        assert len(spread.findings()) == len(level.findings()) == 2
        assert _key(spread)[3] == _key(level)[3], "the same bar missed highest, so the tier ties"
        assert _key(spread)[2] < _key(level)[2], "less missed in all"
        assert _key(spread) < _key(level)

        assert max(_breach(f) for f in spread.findings()) > max(
            _breach(f) for f in level.findings()
        ), "and the worse single miss, which is what `max` would rank it by"

    def test_the_tier_decides_only_after_the_count_and_the_magnitude_tie(self) -> None:
        """A one-semitone miss on the leap ceiling against two on the range.

        The bars are 12 and 24 semitones, so those misses are the *same*
        fraction of their own bar — which is the whole reason the magnitude is
        normalised. With the count and the magnitude equal, what separates the
        two pieces is which bar they missed, and the leap ceiling is the higher.
        """
        leap = _quality(max_leap_semitones=13)
        span = _quality(range_semitones=26)

        assert [len(leap.findings()), len(span.findings())] == [1, 1]
        assert _key(leap)[1] == _key(span)[1]
        assert _key(leap)[2] == pytest.approx(_key(span)[2])
        assert _key(leap)[3] > _key(span)[3]

    def test_the_ends_of_the_table_are_the_ends_of_the_order(self) -> None:
        """Half the bar missed either way — the tune's, or the texture's."""
        tune = _quality(step_ratio=0.225)
        texture = _quality(tessitura_overlap_semitones=6)

        assert _key(tune)[1] == _key(texture)[1]
        assert _key(tune)[2] == pytest.approx(_key(texture)[2])
        assert _key(tune)[3] > _key(texture)[3]

    def test_a_clean_piece_ranks_below_every_tier(self) -> None:
        """It has already won on the count, so the tier must not put it behind.

        Both elements are asserted, because the count alone does not say the
        second one is right: a piece with no misses compares as "better" on the
        count whatever the tier says, so a default that ranked it as if it had
        missed the *highest* bar would pass a test that only read the count —
        which is what this test did until a sabotage walked through it.
        """
        missed = _quality(tessitura_overlap_semitones=6)

        assert _key(_quality())[1] < _key(missed)[1]
        assert _key(_quality())[3] < _key(missed)[3], "below the lowest tier, not above it"
        assert _key(_quality()) < _key(missed)


class TestTheBreachFraction:
    """The normalisation, which is what makes two metrics comparable at all."""

    def test_the_miss_is_a_fraction_of_the_bar(self) -> None:
        """Missing 0.09 against a bar of 0.45 is a fifth of the bar."""
        key = _key(_quality(step_ratio=0.36))

        assert key[1] == 1
        assert key[2] == pytest.approx(0.09 / 0.45)

    def test_a_maximum_bar_measures_its_miss_the_same_way(self) -> None:
        """Over the ceiling by 2 of 24 is a twelfth, not a 2."""
        key = _key(_quality(range_semitones=26))

        assert key[1] == 1
        assert key[2] == pytest.approx(2 / 24)

    def test_a_bar_of_zero_measures_its_miss_raw(self) -> None:
        """ "Never" is a bar with no scale to be a fraction of.

        No threshold in the table has a target of 0, so this arm is unreachable
        through `findings()` — and it is *not* deleted the way C4's dead branch
        was, because deleting it would not remove the case: it would move it
        into a `ZeroDivisionError` raised from inside a sort the day a "must
        never happen" bar is added. `_breach` is a pure function of one value,
        so the case is posed on the value.
        """
        never = QualityFinding(
            metric="texture_hierarchy",
            measured=3.0,
            target=0.0,
            direction="max",
            rationale="",
            hint="",
        )

        assert _breach(never) == 3.0


class TestTheTiebreakers:
    """Elements five and six: arbitrary, deterministic, and needed."""

    def test_the_plan_hash_settles_two_drafts_no_measurement_can(self) -> None:
        low = arbiter_order(_quality(), lint_passed=True, plan_hash="a" * 64, seed=1)
        high = arbiter_order(_quality(), lint_passed=True, plan_hash="b" * 64, seed=1)

        assert low[:4] == high[:4]
        assert low < high

    def test_the_seed_settles_the_tie_the_plan_hash_cannot(self) -> None:
        """One plan, several seeds — which is what a fan-out is.

        A plan is derived from the spec and does not carry the seed, so the
        candidates of one `draft(n=k)` call share a plan hash. Without this
        element they are indistinguishable, and their order would fall back to
        the order they arrived in.
        """
        same_plan = "c" * 64
        first = arbiter_order(_quality(), lint_passed=True, plan_hash=same_plan, seed=3)
        second = arbiter_order(_quality(), lint_passed=True, plan_hash=same_plan, seed=9)

        assert first[:5] == second[:5]
        assert first < second

    def test_an_unseeded_draft_sorts_before_every_seeded_one(self) -> None:
        """`CompositionSpec.seed` is optional, so the type admits this case.

        The alternative to ordering it is not skipping it: it is a `TypeError`
        from inside `sorted`, which is a worse answer than an arbitrary one.
        """
        unseeded = arbiter_order(_quality(), lint_passed=True, plan_hash="d" * 64, seed=None)
        seeded = arbiter_order(_quality(), lint_passed=True, plan_hash="d" * 64, seed=0)

        assert unseeded < seeded


class TestWhatDecidedIt:
    """The order is also an explanation, and the sentence has to be the ranking's own.

    `deciding_element` reads the first element the two keys differ on, which is
    the one `sorted` read — so the reason a draft is shown above another cannot
    disagree with the ranking that put it there.
    """

    def test_every_element_has_a_case_of_its_own(self) -> None:
        """D2's rule: a table of literals is one coverage hole per entry."""
        assert sorted(_ELEMENT_CASES) == sorted(ELEMENTS)

    def test_the_names_are_as_many_as_the_elements_the_order_has(self) -> None:
        """One name per element, so a seventh element cannot go unnamed.

        The length is read against a real key rather than against a number: the
        claim is that the table names the elements `OrderKey` *has*, and a count
        written down here would be a third copy of the arity.
        """
        assert len(ELEMENTS) == len(draft_key(_draft(1)))
        assert len(set(ELEMENTS)) == len(ELEMENTS), "distinct, or two elements share a name"

    def test_the_names_are_the_order_they_decide_in(self) -> None:
        """A ratchet on a judgement, in the shape the table itself uses.

        Nothing can prove the sequence is right — it is a judgement, and the
        module says so. What this pins is that re-ordering it is a *declared*
        edit rather than a silent one, the same shape as the `__all__` and
        catalogue pins.
        """
        assert ELEMENTS == ("legality", "misses", "breach", "tier", "plan_hash", "seed")

    @pytest.mark.parametrize("element", _ELEMENT_CASES)
    def test_the_first_element_that_differs_is_the_one_named(self, element: str) -> None:
        leader, follower = _pair_for(element)

        assert _elements_before(leader, follower, element), "the pair ties on everything before it"
        assert draft_key(leader) < draft_key(follower), "and is ordered the way it reads"
        assert deciding_element(leader, follower) == element

    def test_two_drafts_that_share_every_element_have_no_deciding_one(self) -> None:
        """The type admits it and a caller should not expect it.

        Two drafts equal on all six elements are one plan at one seed, so there
        is no reason to give and `None` is the answer rather than a name.
        """
        assert deciding_element(_draft(1), _draft(1)) is None


class TestTotality:
    """Element zero of the contract: any two drafts compare, however they arrived."""

    def test_ranking_does_not_depend_on_the_order_the_drafts_arrived_in(self) -> None:
        drafts = [_draft(seed) for seed in range(4)]

        assert [draft.draft_id for draft in rank(drafts)] == [
            draft.draft_id for draft in rank(reversed(drafts))
        ]

    def test_a_real_fan_out_is_ranked_by_what_it_measured(self) -> None:
        """The premise is asserted, so this cannot pass by ranking nothing."""
        drafts = [_draft(seed) for seed in range(4)]

        assert len(drafts) == 4
        assert len({draft.plan_hash for draft in drafts}) == 1, "the fan-out shares a plan"
        assert len({draft_key(draft) for draft in drafts}) == 4, "every candidate is told apart"

    def test_best_first_is_the_order_the_ranking_is_read_in(self) -> None:
        good = _draft(1)
        bad = _draft(2, quality=_quality(step_ratio=0.10))

        assert [draft.draft_id for draft in rank([bad, good])] == [good.draft_id, bad.draft_id]


class TestTheRatchet:
    """The rule from the design: a revision may not be worse than what it revises."""

    @pytest.fixture
    def parent(self) -> Draft:
        return _draft(1)

    def test_an_identical_draft_is_allowed(self, parent: Draft) -> None:
        """Equal is not worse. Refusing a tie would veto every edit that moves no metric."""
        assert regression(parent, parent) is None

    def test_a_better_revision_is_allowed(self, parent: Draft) -> None:
        worse = replace(parent, quality=_quality(step_ratio=0.10, repeat_ratio=0.90))
        better = replace(parent, quality=_quality(), draft_id="draft-1a")

        assert regression(worse, better) is None

    def test_a_revision_that_misses_more_bars_is_refused_and_counted(self, parent: Draft) -> None:
        child = replace(
            parent,
            quality=_quality(step_ratio=0.10, range_semitones=4, max_leap_semitones=20),
            draft_id="draft-1a",
        )

        reason = regression(parent, child)

        assert reason is not None
        assert "misses 3 quality bar(s)" in reason
        assert f"{parent.draft_id} misses 0" in reason
        assert "may not be worse" in reason, "the rule is stated, not only the numbers"

    def test_a_revision_that_misses_the_same_bars_by_more_is_refused(self, parent: Draft) -> None:
        was = replace(parent, quality=_quality(step_ratio=0.44))
        now = replace(parent, quality=_quality(step_ratio=0.10), draft_id="draft-1a")

        reason = regression(was, now)

        assert reason is not None
        assert "the same 1 bar(s), but by more" in reason

    def test_a_revision_that_misses_a_higher_bar_is_refused_and_named(self, parent: Draft) -> None:
        """Same count, same fraction of a bar, a bar that matters more."""
        was = replace(parent, quality=_quality(range_semitones=26))
        now = replace(parent, quality=_quality(max_leap_semitones=13), draft_id="draft-1a")

        assert _key(was.quality)[1] == _key(now.quality)[1]
        assert _key(was.quality)[2] == pytest.approx(_key(now.quality)[2]), "the magnitude ties"

        reason = regression(was, now)

        assert reason is not None
        assert "misses max_leap_semitones" in reason
        assert "range_semitones" in reason

    def test_the_ratchet_ignores_the_tiebreakers(self, parent: Draft) -> None:
        """A revision is a different plan by construction, so its hash always differs.

        If the ratchet compared the whole key, a revision that measured
        identically would be kept or rejected on whether its new hash happened
        to sort higher — a coin flip deciding whether the user's edit survives.
        Hence a child posed to lose that tiebreak: see `_rehash_above`.
        """
        child = _rehash_above(parent)

        assert child.plan_hash > parent.plan_hash, "the child loses the tiebreaker"
        assert musical_key(child) == musical_key(parent), "and it measured identically"
        assert regression(parent, child) is None

    def test_the_tier_names_the_bar_that_matters_most_not_the_one_that_spells_first(
        self, parent: Draft
    ) -> None:
        """Two misses either side, and the tiers disagree with the alphabet.

        `_top_miss` is a `min` over the metrics, so its `key` is the whole of
        what it does: without it the sentence names whichever metric sorts first
        as a *string*, which is a coincidence of spelling rather than a claim
        about the piece. So the case is built so that the two disagree — on each
        side the alphabet picks `distinct_durations` and the tiers pick the other
        one — which means `distinct_durations` is shared by both, and the two
        one-miss-each distinctor can be *posed* to breach their bars by the same
        quarter.

        The magnitude has to tie exactly rather than closely, because the key is
        a tuple read left to right: a piece that missed less in total has already
        won, and the element after it is never consulted. That is not an accident
        of this test — it is the order the module documents, element by element.
        """
        was = _quality(repeat_ratio=0.3125, distinct_durations=2)
        now = _quality(max_leap_semitones=15, distinct_durations=2)

        assert len(was.findings()) == len(now.findings()) == 2
        assert _key(now)[2] == _key(was)[2], "a quarter of a bar each, exactly"
        assert _key(now)[3] > _key(was)[3], "and the child's highest miss is the higher bar"

        reason = regression(
            replace(parent, quality=was), replace(parent, quality=now, draft_id="draft-1a")
        )

        assert reason is not None
        assert "misses max_leap_semitones" in reason
        assert "repeat_ratio" in reason

    def test_the_ratchet_compares_exactly_the_musical_elements(self) -> None:
        """The two readers of the split cannot come to disagree about the line.

        The line is a function boundary rather than a slice index now, so what is
        held together here is that the boundary is still where the ratchet reads:
        the musical key *is* the key's own prefix, and exactly two elements — the
        plan hash and the seed — follow it. The third assertion is the other
        reader: `musical_key` is `musical_order` read off a record, and nothing
        else.
        """
        draft = _draft(1)
        key = draft_key(draft)
        music = musical_key(draft)

        assert key[: len(music)] == music
        assert len(key) == len(music) + 2, "the plan hash and the seed, and nothing else"
        assert musical_order(draft.quality, lint_passed=draft.lint_passed) == music


def test_the_module_exports_what_it_claims() -> None:
    """A ratchet on the public surface, in every other module's shape."""
    from saimc.session import arbiter

    assert set(arbiter.__all__) == {
        "ELEMENTS",
        "METRIC_TIERS",
        "MusicalKey",
        "OrderKey",
        "arbiter_order",
        "deciding_element",
        "draft_key",
        "musical_key",
        "musical_order",
        "rank",
        "regression",
    }
