"""Unit tests for saimc.compose.forms."""

from __future__ import annotations

import pytest

from saimc.compose.forms import (
    DEFAULT_KEY_POOL,
    HALF_CADENCE_APPROACH_DEGREE,
    HALF_CADENCE_TARGET_DEGREE,
    LEAP_MIN_SEMITONES,
    MOOD_KEY_POOLS,
    MOOD_PROFILES,
    PHRASE_SIZES,
    SECTION_CLOSES,
    STEP_MAX_SEMITONES,
    TEMPO_RANGE_BPM,
    ChordSlot,
    ChordTemplate,
    apply_final_cadence,
    apply_half_cadence,
    apply_harmonic_rhythm,
    apply_section_close,
    bar_scale_intervals,
    chord_root_offset,
    chord_tone_degrees,
    chosen_key,
    get_mood_profile,
    get_template_for_form,
    key_pool_for,
    key_root_midi,
    key_scale_pcs,
    key_signature_from_spec_key,
    scale_intervals,
    scale_pitch_offset,
    scale_semitones,
    scale_walk,
    transposed_key,
)
from saimc.compose.score import KeySignature
from saimc.spec import WesternKey


class TestChordTemplate:
    def test_valid_template(self) -> None:
        t = ChordTemplate(
            name="test",
            bars=8,
            chords=(ChordSlot(0, 2), ChordSlot(5, 2), ChordSlot(3, 2), ChordSlot(4, 2)),
        )
        assert t.bars == 8
        assert sum(slot.bars for slot in t.chords) == 8

    def test_slot_bars_must_sum_to_template_bars(self) -> None:
        # The engine advances one `bars`-length block per section; a
        # template whose slots overflow it desynchronises every later
        # section's chord map.
        with pytest.raises(ValueError, match="sum to 6"):
            ChordTemplate(name="drift", bars=8, chords=(ChordSlot(0, 2), ChordSlot(5, 4)))


class TestTheTablesTheBassWalkIsWrittenAgainst:
    """What the tables admit, swept rather than read.

    Two claims about the chord tables are what make two lines in
    `_generate_bass` what they are, and neither can be checked by composing a
    piece: a pinned degree is never borrowed, and the two modes' tables are
    the ones `chord_root_offset` swaps between. A claim about what the tables
    cannot reach needs a sweep of the tables, so both are swept here.
    """

    def _templates(self) -> list[ChordTemplate]:
        templates = [
            template for profile in MOOD_PROFILES.values() for template in profile.templates
        ]
        assert templates, "the sweep found no templates"
        return templates

    def test_a_pinned_bass_degree_is_never_a_borrowed_chord(self) -> None:
        """The premise `_generate_bass`'s pinned arm rests on.

        That arm reads `chord_root_offset(degree, key, borrowed=slot.borrowed)`,
        because a pinned degree belongs to the chord's own table — so a
        *borrowed* chord would pin the other mode's degree. Nothing reaches it:
        every pin in the tables is a cadence pin, built unborrowed by
        `apply_final_cadence`, and no template puts a pin on a borrowed slot.
        So the argument is correct by construction and inert on every input the
        tables admit today, and this case is what keeps that honest — a
        template that pins a borrowed bass degree fails here, where the wrong
        root would otherwise arrive without a symptom. Both halves are
        asserted: the pins exist at all (or the arm has no site to be inert
        at), and none of them is borrowed.
        """
        templates = self._templates()
        pinned = [
            (template.name, slot)
            for template in templates
            for slot in apply_final_cadence(
                template, cadence_degree=4, seventh=True
            ).chords
            if slot.bass_degree is not None
        ]
        assert pinned, "nothing pins a bass degree, so the arm this guards is dead"
        borrowed = [entry for entry in pinned if entry[1].borrowed]
        assert borrowed == [], borrowed

    def test_the_two_modes_disagree_on_exactly_the_three_flattened_degrees(self) -> None:
        """The whole of what `chord_root_offset`'s swap can move.

        A borrowed chord's root reads off the parallel mode's table, so the two
        tables agreeing on a degree means the swap is invisible there. Degrees
        2, 5 and 6 are the ones the minor scale flattens, and every other degree
        in 0..6 must agree — this is why the borrowed-root defect was reachable
        at all (a major key's borrowed slot could only ever sit on one of the
        three) and why the fix had to be symmetric rather than an adjustment
        applied in one direction.
        """
        major = KeySignature(root="C", mode="major")
        minor = KeySignature(root="C", mode="minor")
        disagree = {
            degree
            for degree in range(7)
            if chord_root_offset(degree, major) != chord_root_offset(degree, minor)
        }
        assert disagree == {2, 5, 6}


class TestGetTemplateForForm:
    def test_exact_match(self) -> None:
        t = get_template_for_form("calming", 8)
        assert t.bars == 8

    def test_extend_template(self) -> None:
        # No 32-bar calming template; should extend an 8-bar one.
        t = get_template_for_form("calming", 32)
        assert t.bars == 32

    def test_truncate_template(self) -> None:
        # No 4-bar template; should truncate an 8-bar one.
        t = get_template_for_form("calming", 4)
        assert t.bars == 4

    def test_all_moods_have_templates(self) -> None:
        for mood in ("calming", "electrifying", "sleep"):
            for form in PHRASE_SIZES:
                t = get_template_for_form(mood, form)
                assert t.bars == form

    def test_a_truncation_that_lands_on_a_boundary_leaves_an_empty_slot(self) -> None:
        """The artifact the walk's zero-length guard exists for, swept.

        `_truncate_template` closes a truncation with
        `slot._replace(bars=target_bars - consumed)`, and when the break
        lands exactly on a slot boundary that difference is zero — so the
        template gains a slot that occupies no bar. `_generate_bass` skips
        such a slot (`if dur == 0: continue`), because stepping the line
        onto a chord that never sounds is a leap on a bar that does not
        exist. Swept rather than illustrated, and in both directions: the
        artifact has to be *reached* or that guard is dead, and it must not
        appear where it should not, or the guard is doing work the
        truncation already did. The characterization is exact because the
        base templates are uniform two-bar slots — asserted as this sweep's
        premise rather than assumed — so an empty slot is what a whole
        number of slots remaining leaves behind and nothing else does.
        """
        reached: list[str] = []
        for mood in ("calming", "electrifying", "sleep"):
            base = get_mood_profile(mood).templates[0]
            assert {slot.bars for slot in base.chords} == {2}, (
                f"{mood}'s base template is not uniform two-bar slots, so this sweep's "
                "arithmetic no longer describes it"
            )
            for form in range(2, 33):
                t = get_template_for_form(mood, form)
                assert t.bars == form, (mood, form)
                assert min(slot.bars for slot in t.chords) >= 0, (mood, form)
                empty = any(slot.bars == 0 for slot in t.chords)
                remainder = form % base.bars
                assert empty == (remainder != 0 and remainder % 2 == 0), (mood, form)
                if empty:
                    reached.append(f"{mood}/{form}")
        assert reached, "no truncation leaves an empty slot, so the walk's guard is dead"
        assert "calming/4" in reached, reached
        assert "calming/5" not in reached, reached


class TestTheSectionClose:
    """The three ways a section can end, and what each one may move.

    `apply_section_close` is the one site the plan's `section_close`
    reaches, so the three arms are pinned here rather than left to the seam
    tests — which read the knob through a composed piece and cannot tell
    one arm's arithmetic from another's. What every arm shares is the
    interesting half: each preserves the bar count and leaves the body
    before the last two bars exactly as the template wrote it. A close
    rewrites the ending and nothing else.
    """

    def _template(self) -> ChordTemplate:
        return get_template_for_form("calming", 8)

    @staticmethod
    def _bar_by_bar(template: ChordTemplate) -> tuple[ChordSlot, ...]:
        """One entry per bar, so a zero-length slot contributes nothing.

        The production templates have uniform slots and the truncations
        leave an empty one behind (see the test above), so comparing slot
        tuples would compare the truncation's arithmetic rather than the
        chords that sound.
        """
        return tuple(slot._replace(bars=1) for slot in template.chords for _ in range(slot.bars))

    def test_the_three_arms_are_three_different_endings(self) -> None:
        template = self._template()
        closes = {
            name: apply_section_close(template, close=name, cadence_degree=4, seventh=True)
            for name in SECTION_CLOSES
        }
        assert closes["hold"] is template, "hold rebuilt the template it was asked to leave alone"
        assert closes["half"] != template, "the half close rebuilt nothing"
        assert closes["full"] != template, "the full close rebuilt nothing"
        assert closes["half"] != closes["full"], (
            "the half and the full close write the same template, so the plan's choice is inert"
        )
        for name, closed in closes.items():
            assert closed.bars == template.bars, name
            assert self._bar_by_bar(closed)[:-2] == self._bar_by_bar(template)[:-2], (
                f"the {name} close moved a bar before the last two"
            )

    def test_a_half_cadence_approaches_the_dominant_and_does_not_arrive(self) -> None:
        """The two chords a half cadence is: root position, one bar each.

        A half cadence points home and stops — a predominant, then the
        dominant. That it does not arrive is what separates this arm from
        the full cadence, so it is asserted rather than implied by the
        constants: a retuning that landed either chord on the tonic would
        quietly turn this arm into the other one.

        The degrees are pinned twice, on purpose. The literals pin the pair
        itself — nothing derives a half cadence from a table, so this is not
        evidence the pair is right (it cannot be; that is a judgement) but
        the ratchet that makes retuning it a declared edit, the way the
        arbiter's metric order is pinned. And the slots pin the *wiring*: a
        slot hardcoded to a degree the constant no longer holds would pass
        the first assertion alone.
        """
        approach, target = apply_half_cadence(self._template()).chords[-2:]
        assert (HALF_CADENCE_APPROACH_DEGREE, HALF_CADENCE_TARGET_DEGREE) == (1, 4)
        assert (approach.degree, target.degree) == (1, 4)
        assert (approach.bars, target.bars) == (1, 1)
        assert approach.bass_degree == approach.degree, "the approach is not in root position"
        assert target.bass_degree == target.degree, "the dominant is not in root position"

    def test_the_full_close_writes_the_pair_the_plan_gave_it(self) -> None:
        """`cadence_degree` and `seventh` are read only by the `full` arm.

        Asserted at 3 and at `seventh=True` — not at the module's own default
        of 4 — because a close that fell back to the default pair would agree
        with an assertion written at the default and disagree with this one.
        """
        closed = apply_section_close(self._template(), close="full", cadence_degree=3, seventh=True)
        resolution, tonic = closed.chords[-2:]
        assert resolution.degree == 3
        assert resolution.seventh is True
        assert resolution.bass_degree == 3, "the cadence is not in root position"
        assert (tonic.degree, tonic.bass_degree) == (0, 0)

    def test_a_template_too_short_to_close_is_returned_untouched(self) -> None:
        """Two bars is the shortest template the truncation can make.

        A cadence needs two bars of its own, so a shorter template has no
        body left to rewrite. Both arms return it *by identity* rather than
        by an equal copy, which is what says no slot was rebuilt.
        """
        short = get_template_for_form("calming", 2)
        assert short.bars == 2, "this test's premise is that a two-bar template exists"
        for close in ("half", "full"):
            closed = apply_section_close(short, close=close, cadence_degree=4, seventh=True)
            assert closed is short, close

    def test_an_unknown_close_is_refused_by_name(self) -> None:
        """The arm a plan cannot reach, and the refusal a direct call still needs.

        `CompositionPlan` refuses a `section_close` outside the vocabulary,
        so this raise is unreachable through a plan — which is why it is
        pinned here: a direct call is the only way to reach it, and a
        template silently left alone would be the unnamed no-op the deltas'
        rule forbids.
        """
        with pytest.raises(ValueError) as raised:
            apply_section_close(self._template(), close="cadential", cadence_degree=4, seventh=True)
        message = str(raised.value)
        assert "cadential" in message
        assert all(name in message for name in SECTION_CLOSES), message


class TestTheHarmonicRhythm:
    """A progression re-cut at the plan's rate: what it may move, and what not.

    The plan carries the durations and the template carries the chords, so
    the whole of this function is the arithmetic between them — and the two
    properties every caller relies on are that the bar count is unchanged
    (the arrangement's own bar arithmetic, and the duration the piece was
    asked for, are built on it) and that no chord is cut to zero bars (a
    zero-bar slot is a chord that never sounds, which `_generate_section`
    has to skip by name).

    What the re-cut must *not* move is the part of a slot that is not its
    length: a cadence's seventh and a forced bass pin are musical decisions
    the templates made, and a rate is not a reason to drop either.
    """

    def _template(self) -> ChordTemplate:
        return get_template_for_form("calming", 8)

    @staticmethod
    def _bar_by_bar(template: ChordTemplate) -> tuple[ChordSlot, ...]:
        """One entry per bar, so the durations are read as they sound."""
        return tuple(slot._replace(bars=1) for slot in template.chords for _ in range(slot.bars))

    def test_the_pattern_decides_how_long_each_degree_lasts(self) -> None:
        """`(4,)` holds each chord of the default progression twice as long.

        The default is four two-bar slots, so a four-bar pattern states the
        first two of them and stops — the progression is not shortened, it
        is *slowed*, and the chords that no longer fit are the ones the
        cadence is about to rewrite anyway.
        """
        template = self._template()
        re_cut = apply_harmonic_rhythm(template, (4,))
        assert [(slot.degree, slot.bars) for slot in re_cut.chords] == [(0, 4), (5, 4)]
        assert re_cut.bars == template.bars
        assert [slot.seventh for slot in re_cut.chords] == [
            slot.seventh for slot in template.chords[:2]
        ], "the re-cut dropped a slot's seventh, which its length has nothing to do with"
        assert all(slot.bass_degree is None for slot in re_cut.chords)

    def test_a_one_bar_pattern_walks_the_progression_a_chord_a_bar(self) -> None:
        template = self._template()
        re_cut = apply_harmonic_rhythm(template, (1,))
        degrees = [slot.degree for slot in re_cut.chords]
        assert degrees == [0, 5, 3, 4, 0, 5, 3, 4], (
            "a one-bar pattern does not walk the progression: the degrees are "
            "not the template's own, in their own order, cycling"
        )
        assert all(slot.bars == 1 for slot in re_cut.chords)

    @pytest.mark.parametrize("pattern", [(1,), (4,), (2, 1, 1), (3,), (5, 3), (16,), (2, 2, 2, 2)])
    def test_the_bar_count_holds_and_no_chord_is_cut_to_nothing(self, pattern: tuple[int, ...]) -> None:
        """The two invariants every caller's arithmetic rests on.

        `(16,)` is here because the clamp is what makes it legal: a pattern
        whose durations run past the template's own length would otherwise
        write a chord longer than the section, and `(3,)` because a pattern
        that does not divide the bar count is the arm that shortens the last
        slot — the one place a clamp to zero could appear.
        """
        template = self._template()
        re_cut = apply_harmonic_rhythm(template, pattern)
        assert re_cut.bars == template.bars
        assert sum(slot.bars for slot in re_cut.chords) == template.bars
        assert all(slot.bars >= 1 for slot in re_cut.chords), (
            "a chord was cut to zero bars, which is a chord that never sounds"
        )
        # The degrees are the template's own, in their own order, cycling —
        # a rate moves how long a chord lasts and not which chord it is. The
        # tail is *not* pinned: the pattern stops when the bars run out, so
        # a slow pattern ends on an earlier degree than the template does,
        # and `apply_final_cadence` rewrites those last two bars anyway.
        template_degrees = [slot.degree for slot in template.chords]
        assert [slot.degree for slot in re_cut.chords] == [
            template_degrees[index % len(template_degrees)] for index in range(len(re_cut.chords))
        ]

    def test_a_pattern_that_does_not_fit_is_shortened_at_the_end_and_not_by_dropping_a_chord(
        self,
    ) -> None:
        """`(3,)` over eight bars is 3, 3 and 2 — not 3, 3 and nothing.

        A `while consumed < bars` loop that stopped when the next chord did
        not fit would leave the piece two bars short of its section.
        """
        re_cut = apply_harmonic_rhythm(self._template(), (3,))
        assert [slot.bars for slot in re_cut.chords] == [3, 3, 2]

    def test_the_rhythm_a_uniform_template_already_has_is_the_pattern_that_restates_it(
        self,
    ) -> None:
        """The premise behind `None`, stated where it holds and where it does not.

        For the two-bar templates a pattern of two-bar slots writes the
        template back, chord for chord — which is why `None` needs the
        explanation the plan's field carries rather than being spelled as a
        tuple the templates would agree with.
        """
        template = self._template()
        assert all(slot.bars == 2 for slot in template.chords), (
            "this test's premise is that calming's base template is uniform two-bar slots"
        )
        assert apply_harmonic_rhythm(template, (2, 2, 2, 2)).chords == template.chords

    def test_no_single_pattern_stands_for_the_templates_own_rhythm_in_one_mood(self) -> None:
        """Which is the whole reason the plan defers to `None`.

        `sleep` ships both a four-bar pair and a four-slot two-bar
        progression, and the duration search decides which one a piece uses
        at compose time — so a `default_plan` that wrote one of them as a
        pattern would be pinning an arrangement decision the plan does not
        make. Asserted as a premise rather than described: a future
        `forms.py` in which every template of a mood had the same slot
        lengths would make the field's deferral unnecessary, and this test
        is where that would be noticed.
        """
        slots_per_template = {
            template.name: tuple(slot.bars for slot in template.chords)
            for template in MOOD_PROFILES["sleep"].templates
        }
        assert len(set(slots_per_template.values())) > 1, (
            f"every sleep template is cut the same way ({slots_per_template}), so no "
            "pattern would need to be deferred — the plan's `None` is now a choice"
        )
        assert slots_per_template["sleep_drone"] == (4, 4)
        assert slots_per_template["sleep_lullaby"] == (2, 2, 2, 2)

    @pytest.mark.parametrize("pattern", [(), (0,), (2, 0), (2, -1), (-1, 2)])
    def test_a_pattern_that_is_not_whole_bars_each_is_refused_by_name(
        self, pattern: tuple[int, ...]
    ) -> None:
        """Refused here as well as at the plan, because a direct call is a caller.

        The plan refuses these too, so this raise is unreachable through a
        plan — which is why it is pinned: `_with_harmonic_rhythm` is one
        call site and a script composing through `forms` is another, and a
        pattern silently treated as a pulse would be the unnamed no-op the
        deltas' rule forbids.

        The guard is also load-bearing against a *hang*, which is more than
        the rule asks of it. A zero in the pattern makes `duration` zero, so
        `consumed` never advances and the loop appends chord slots until the
        process dies — measured when the guard was removed to check this
        test: 2.6 GB of RSS before it was killed. So the reading "a pattern
        that does not fit is shortened at the end", which holds for every
        whole-bar pattern, is not the property being defended here; the
        function not returning at all is.
        """
        with pytest.raises(ValueError) as raised:
            apply_harmonic_rhythm(self._template(), pattern)
        message = str(raised.value)
        assert str(tuple(pattern)) in message
        assert "each at least one" in message


class TestTempoRanges:
    def test_calming_range(self) -> None:
        assert TEMPO_RANGE_BPM["calming"] == (50, 80)

    def test_electrifying_range(self) -> None:
        assert TEMPO_RANGE_BPM["electrifying"] == (100, 160)

    def test_sleep_range(self) -> None:
        assert TEMPO_RANGE_BPM["sleep"] == (40, 64)


class TestKeySignatureFromSpecKey:
    def test_major_key(self) -> None:
        k = key_signature_from_spec_key(WesternKey.G_MAJOR)
        assert k.root == "G"
        assert k.mode == "major"

    def test_minor_key(self) -> None:
        k = key_signature_from_spec_key(WesternKey.A_MINOR)
        assert k.root == "A"
        assert k.mode == "minor"

    def test_every_key_the_type_admits_resolves(self) -> None:
        """A key the vocabulary names but this cannot resolve is unreachable.

        Not a style point: `WesternKey` is what a spec, a plan document and
        a delta all carry, so a value in it that raises here would be a
        request the type accepts and the engine refuses. The count is the
        vocabulary's own distinct roots, derived rather than listed.
        """
        resolved = {name: key_signature_from_spec_key(name) for name in WesternKey}
        assert all(k.mode in ("major", "minor") for k in resolved.values())
        assert {k.root for k in resolved.values()} == {
            name.value[:-1] if name.value.endswith("m") else name.value for name in WesternKey
        }
        assert all(
            (name.value.endswith("m")) == (k.mode == "minor") for name, k in resolved.items()
        )


class TestKeyPools:
    """The table a mood's default key is drawn from.

    A musical judgement rather than a finding — a 225-cell sweep found every
    key `WesternKey` admits composes clean at every mood, duration, ensemble
    and seed — so these pin the table's *shape* and its literal contents,
    the way the arbiter's metric order is pinned. Nothing here says the
    pools are the right taste; it says the taste is declared.
    """

    def test_every_mood_has_a_pool_and_no_others_do(self) -> None:
        assert set(MOOD_KEY_POOLS) == set(MOOD_PROFILES)

    def test_every_pool_names_distinct_real_keys(self) -> None:
        for mood, pool in (*MOOD_KEY_POOLS.items(), ("<default>", DEFAULT_KEY_POOL)):
            assert pool, mood
            assert len(set(pool)) == len(pool), (mood, pool)
            unknown = [name for name in pool if name not in {k.value for k in WesternKey}]
            assert not unknown, (mood, unknown)

    def test_the_pools_are_the_taste_they_are_written_as(self) -> None:
        """The literal, so re-ordering or re-picking is a declared edit.

        The *order* is load-bearing: the engine picks `pool[seed % len]`, so
        the first entry is the key a seedless spec falls back to, and the
        test below reads that as the contract rather than as an accident.
        """
        assert MOOD_KEY_POOLS == {
            "calming": ("C", "F", "G", "Bb", "Am", "Dm"),
            "sleep": ("F", "Bb", "Eb", "Ab", "Dm", "Gm"),
            "electrifying": ("A", "E", "D", "G", "Am", "Em"),
        }
        assert DEFAULT_KEY_POOL == ("C", "G", "F", "D", "Em", "Am")

    def test_each_pool_varies_the_root_and_the_mode(self) -> None:
        """The two axes a key varies on, and the pair A1's corpus holds fixed."""
        for mood, pool in (*MOOD_KEY_POOLS.items(), ("<default>", DEFAULT_KEY_POOL)):
            modes = {name.endswith("m") for name in pool}
            assert modes == {False, True}, (mood, pool)
            assert len({name.removesuffix("m") for name in pool}) >= 4, (mood, pool)

    def test_an_unlisted_mood_gets_the_default_pool(self) -> None:
        assert key_pool_for("sleepy") is DEFAULT_KEY_POOL
        assert key_pool_for("calming") == MOOD_KEY_POOLS["calming"]


class TestChosenKey:
    """Which key a piece is written in: the spec's, or the seed's pick."""

    _POOL = ("C", "G", "Am")

    def test_a_named_key_is_the_key_and_the_pool_is_dead(self) -> None:
        for seed in (0, 1, 2, 3, 17):
            k = chosen_key(WesternKey.E_FLAT_MAJOR, pool=self._POOL, seed=seed)
            assert (k.root, k.mode) == ("Eb", "major")

    def test_an_unnamed_key_walks_the_pool_in_order(self) -> None:
        """The pick is `pool[seed % len(pool)]`, so consecutive seeds differ."""
        picked = [
            chosen_key(None, pool=self._POOL, seed=seed).root for seed in range(len(self._POOL) * 2)
        ]
        assert picked == ["C", "G", "A", "C", "G", "A"]

    def test_a_spec_naming_no_key_and_no_seed_gets_the_pool_head(self) -> None:
        """Which is how each table declares its mood's fallback.

        A seedless spec resolves to seed 0 here as it does at the engine's
        other seed reads, so "the mood's fallback key" and "the pool's first
        entry" are the same statement.
        """
        k = chosen_key(None, pool=self._POOL, seed=None)
        assert k.root == self._POOL[0]

    def test_a_minor_pool_entry_resolves_to_a_minor_key(self) -> None:
        k = chosen_key(None, pool=self._POOL, seed=2)
        assert (k.root, k.mode) == ("A", "minor")

    def test_an_empty_pool_is_refused_rather_than_dividing_by_zero(self) -> None:
        # The plan refuses an empty pool, so the engine cannot reach this; a
        # direct call can, and a bare ZeroDivisionError is not a named refusal.
        with pytest.raises(ValueError, match="at least one key"):
            chosen_key(None, pool=(), seed=0)

    def test_a_pool_naming_a_non_key_is_refused_by_name(self) -> None:
        with pytest.raises(ValueError, match="H"):
            chosen_key(None, pool=("C", "H"), seed=1)


class TestKeyRootMidi:
    def test_c_major_is_60(self) -> None:
        assert key_root_midi(KeySignature(root="C", mode="major")) == 60

    def test_g_major_is_67(self) -> None:
        # G is 7 semitones above C; C is 60, so G4 is 67.
        assert key_root_midi(KeySignature(root="G", mode="major")) == 67

    def test_a_minor_is_69(self) -> None:
        assert key_root_midi(KeySignature(root="A", mode="minor")) == 69

    def test_unknown_root_rejected(self) -> None:
        with pytest.raises(ValueError):
            key_root_midi(KeySignature(root="X", mode="major"))


class TestScaleTables:
    """The diatonic tables the linter's licence and the melody walk share."""

    def test_the_modes_differ_on_the_third_sixth_and_seventh(self) -> None:
        # Indices 2, 5 and 6 are the degrees the minor scale flattens;
        # everything else is shared. A table that drifted anywhere else
        # would silently move the melodic and harmonic skeleton.
        major = scale_semitones("major")
        minor = scale_semitones("minor")
        assert major[:2] == minor[:2]
        differs = [index for index, (m, n) in enumerate(zip(major, minor, strict=True)) if m != n]
        assert differs == [2, 5, 6]
        assert major[2] - minor[2] == 1
        assert major[5] - minor[5] == 1
        assert major[6] - minor[6] == 1

    def test_scale_pitch_offset_wraps_into_one_octave(self) -> None:
        # Degree 7 is the tonic again, in the same octave — this is the
        # semantics the engine's chord walk relies on.
        assert scale_pitch_offset(0, "major") == scale_pitch_offset(7, "major") == 0

    def test_scale_pitch_offset_accepts_negative_degrees(self) -> None:
        assert scale_pitch_offset(-1, "major") == 11

    def test_scale_intervals_covers_one_octave_from_its_root(self) -> None:
        intervals = scale_intervals(0, "major")
        assert intervals == (0, 2, 4, 5, 7, 9, 11)
        assert scale_intervals(0, "minor") == (0, 2, 3, 5, 7, 8, 10)

    def test_scale_intervals_rotates_the_mode_onto_its_new_root(self) -> None:
        # Spelled from the V of major the scale is mixolydian, and from
        # major's sixth it is the natural minor: a bar's chord and the
        # line over it have to be the same seven tones, so the table the
        # walk reads is a rotation of the mode and not the mode itself.
        assert scale_intervals(4, "major") == (0, 2, 4, 5, 7, 9, 10)
        assert scale_intervals(5, "major") == (0, 2, 3, 5, 7, 8, 10)

    def test_scale_intervals_ascends_within_the_octave(self) -> None:
        for mode in ("major", "minor"):
            for degree in range(7):
                intervals = scale_intervals(degree, mode)
                assert intervals[0] == 0
                assert len(intervals) == 7
                assert all(0 <= step < 12 for step in intervals)
                assert list(intervals) == sorted(set(intervals))

    def test_scale_walk_carries_the_octave(self) -> None:
        intervals = scale_intervals(0, "major")
        assert scale_walk(0, 60, intervals) == 60
        assert scale_walk(2, 60, intervals) == 64
        assert scale_walk(7, 60, intervals) == 72

    def test_scale_walk_descends_below_the_root(self) -> None:
        # Degree -1 is the scale tone *below* the root, not degree 6 of
        # the octave above: a melody has to be able to walk under its
        # starting note without the modulo flipping it up.
        intervals = scale_intervals(0, "major")
        assert scale_walk(-1, 60, intervals) == 59
        assert scale_walk(-7, 60, intervals) == 48

    def test_scale_walk_follows_the_mode_it_is_given(self) -> None:
        # The same degree is a semitone lower in minor, which is the
        # whole reason the intervals are a parameter and not a mode name.
        assert scale_walk(2, 60, scale_intervals(0, "major")) == 64
        assert scale_walk(2, 60, scale_intervals(0, "minor")) == 63

    def test_scale_walk_walks_the_bar_scale_from_its_chord_root(self) -> None:
        # A bar on the V: its own scale spelled from its root, so a
        # degree walked over it is a tone of that rotation and not of
        # the key. Degree 0 is the chord root itself.
        key = KeySignature(root="C", mode="major")
        intervals = bar_scale_intervals(4, key)
        assert scale_walk(0, 67, intervals) == 67
        assert scale_walk(1, 67, intervals) == 69
        assert scale_walk(7, 67, intervals) == 79

    def test_a_borrowed_bar_walks_the_parallel_mode(self) -> None:
        # bVI in a major key is built from the parallel minor's table,
        # and the line over it has to read that same table or its steps
        # would leave the chord's own scale.
        key = KeySignature(root="C", mode="major")
        assert bar_scale_intervals(5, key, borrowed=True) == scale_intervals(5, "minor")
        assert bar_scale_intervals(5, key) == scale_intervals(5, "major")

    def test_chord_tone_degrees_are_every_other_degree(self) -> None:
        # The chord tables take a triad's tones from the bar scale's
        # degrees 0, 2, 4 and a seventh's from 0, 2, 4, 6, so "is this
        # note a chord tone" is a question about degree parity and not
        # about pitch classes.
        assert chord_tone_degrees(3) == (0, 2, 4)
        assert chord_tone_degrees(4) == (0, 2, 4, 6)

    def test_key_scale_pcs_covers_the_mode(self) -> None:
        assert key_scale_pcs(KeySignature(root="C", mode="major")) == frozenset(
            (0, 2, 4, 5, 7, 9, 11)
        )
        assert key_scale_pcs(KeySignature(root="A", mode="minor")) == frozenset(
            (9, 11, 0, 2, 4, 5, 7)
        )

    def test_key_scale_pcs_transposes_with_the_root(self) -> None:
        # G major has F# where C major has F natural.
        assert 6 in key_scale_pcs(KeySignature(root="G", mode="major"))
        assert 5 not in key_scale_pcs(KeySignature(root="G", mode="major"))

    def test_a_step_is_a_semitone_or_a_whole_tone(self) -> None:
        # The linter's licence and the quality scorecard both measure
        # against this one number; if it moved, passing tones the licence
        # admits would be scored as leaps.
        assert STEP_MAX_SEMITONES == 2

    def test_a_leap_is_a_fourth_or_wider(self) -> None:
        # The generator's recovery pass and the scorecard's leap ratio
        # have to call the same intervals leaps, and a third has to be
        # neither: it is the one interval the two tests cannot disagree
        # about without the disagreement reading as a composition bug.
        assert LEAP_MIN_SEMITONES == 5
        assert LEAP_MIN_SEMITONES > STEP_MAX_SEMITONES + 1


class TestTransposedKey:
    """`transposed_key` names the key a modulation lift lands on."""

    def test_a_whole_step_above_c_major_is_d_major(self) -> None:
        assert transposed_key(KeySignature(root="C", mode="major"), 2) == KeySignature(
            root="D", mode="major"
        )

    def test_the_mode_is_carried(self) -> None:
        assert transposed_key(KeySignature(root="A", mode="minor"), 2) == KeySignature(
            root="B", mode="minor"
        )

    def test_no_offset_is_the_same_key(self) -> None:
        key = KeySignature(root="Bb", mode="major")
        assert transposed_key(key, 0) is key

    def test_the_tonic_moves_by_exactly_the_offset(self) -> None:
        # Every root in the table has to spell a key the table can read
        # back, and the lifted tonic has to be the old one plus the lift —
        # the chord roots, the bass and the licence are all read off it.
        for root in ("C", "G", "D", "A", "E", "B", "F#", "C#", "F", "Bb", "Eb", "Ab"):
            key = KeySignature(root=root, mode="major")
            for semitones in (1, 2, 5, 7, 11):
                lifted = transposed_key(key, semitones)
                assert key_root_midi(lifted) % 12 == (key_root_midi(key) + semitones) % 12


class TestMoodRegistry:
    def test_profiles_are_the_single_source_of_truth(self) -> None:
        for name, profile in MOOD_PROFILES.items():
            assert profile.name == name
            assert TEMPO_RANGE_BPM[name] == profile.tempo_range_bpm
            assert profile.templates

    def test_unknown_mood_names_the_known_moods(self) -> None:
        with pytest.raises(KeyError) as exc_info:
            get_mood_profile("angsty")
        message = str(exc_info.value)
        assert "angsty" in message
        assert "calming" in message
        assert "sleep" in message
