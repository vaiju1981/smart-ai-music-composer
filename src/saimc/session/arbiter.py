"""One order over drafts, written down, and the ratchet that keeps a revision honest.

The linter says whether a piece may ship. It cannot say which of two shippable
pieces is better, and no amount of tightening it would make it able to: it
judges legality, and taste has no legality. So the harness needs a second thing
— a total order over drafts that are all legal — and this module is it. It is
deliberately small and deliberately in code, because the alternative is an order
that lives in a prompt, where it would depend on who spoke last and on nothing
else.

Three properties hold this together, and the rest of the design follows from
them.

**The order is total, and it does not depend on how the drafts arrived.** Any
two drafts compare, and ranking a fan-out gives the same sequence whichever
order its candidates are handed in. This is the engine's guarantee one level up:
`(plan, seed) -> notes` is byte-identical, and `drafts -> ranked` is too.

**The sequence is the argument.** Each element is a strictly more negotiable
thing than the one before it, so a piece that wins an earlier element cannot be
overturned by a later one:

1. **Legality.** A draft the linter refused never outranks one it accepted,
   whatever else it measured. Nothing below this line trades against it.
2. **How many bars it missed.** One miss is better than three.
3. **How far it missed them, as a fraction of each bar.** Missing by a hair
   three times is not the same as missing by half once, and the normalisation is
   what makes a semitone deficit comparable to a ratio deficit at all.
4. **Which bars it missed.** A fixed priority over the eleven metrics. This is the
   only element that is a judgement rather than an arithmetic, so it is a
   written-down table with its reasoning beside it, below.
5. **The plan's hash.** Arbitrary, and deterministic: it settles two drafts that
   measured identically and differ only in what produced them.
6. **The seed.** For the case the plan hash cannot reach — see `arbiter_order`.

**A revision may not be worse than the draft it revises.** `regression` is that
rule. It returns a sentence rather than a flag, because the conductor reads it
and the user is told it.

The honest limit belongs here rather than in a review note: **these metrics are
a proxy for taste, not a definition of it.** The linter judges shipability, the
user judges taste, and this order sits between them as a way of *presenting* a
choice. `release/gates.py` says the same thing about the same numbers. This
module ranks; the user decides.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Final, TypeAlias

from saimc.quality import PieceQuality, QualityFinding
from saimc.session.models import Draft

OrderKey: TypeAlias = tuple[int, int, float, int, str, int]
"""One draft's place in the order: lower is better, and it sorts as a tuple."""

MusicalKey: TypeAlias = tuple[int, int, float, int]
"""The part of an `OrderKey` that is about the piece: legality, misses, breach, tier.

The other two elements are tiebreakers, and a tiebreaker must not gate an edit: a
revision *is* a different plan, so its hash differs by construction, and a ratchet
comparing the whole key would reject a revision that measured identically
whenever the new hash happened to sort higher. That is a coin flip deciding
whether the user's edit survives. So the two parts are two functions — see
`musical_order` — rather than one function and a slice, and the boundary between
them is the only place the ratchet reads.
"""

METRIC_TIERS: Final[tuple[str, ...]] = (
    # What a listener hears as "the tune": a line that moves mostly by step,
    # answers its leaps, and never jumps further than an octave.
    "step_ratio",
    "leap_recovery_ratio",
    "max_leap_semitones",
    # The tune's shape — how far it spans, and how much of it is repetition.
    "range_semitones",
    "repeat_ratio",
    # Its rhythm. A melody with one note length is a metronome, and a bass with
    # one onset pattern is a loop.
    "distinct_durations",
    "bass_onset_patterns",
    # How the parts sit together. Last, and not because a bad texture is
    # inaudible — it is the opposite — but because the bed settles *under* the
    # tune: a fix here cannot make an unsingable melody singable, and a piece
    # can clear every one of these and still be dull.
    "texture_hierarchy",
    "register_separation_semitones",
    "tessitura_overlap_semitones",
    # And whether the bed is there at all. A texture that sustains nothing in
    # any bar is the thinnest thing a piece can do and still have an
    # accompaniment, so it ranks below the two ways a bed crowds the tune:
    # a crowded tune is harder to hear, an absent bed is harder to notice.
    "harmony_pad_coverage",
)
"""The eleven metrics, most important first. The order the arbiter ranks a miss by.

This is a judgement and it is written as one. It is *not* derived from
`QUALITY_THRESHOLDS`, because a derived order is not a written-down order: the
table would silently re-order itself the day a threshold moved in the file. So
the two are held together by a test instead — the metrics here are exactly the
metrics there, once each — which fails loudly whichever side changes first.
"""


def musical_order(quality: PieceQuality, *, lint_passed: bool) -> MusicalKey:
    """The musical half of the order, which is also the whole of the ratchet.

    This is where the line between the piece and its provenance is drawn, and it
    is drawn once: `arbiter_order` appends the tiebreakers to it and
    `musical_key` reads it off a record, so the two cannot come to disagree about
    which elements decide whether a revision is worse.
    """
    findings = quality.findings()
    return (
        0 if lint_passed else 1,
        len(findings),
        sum(_breach(finding) for finding in findings),
        _tier_rank(findings),
    )


def arbiter_order(
    quality: PieceQuality,
    *,
    lint_passed: bool,
    plan_hash: str,
    seed: int | None,
) -> OrderKey:
    """Where one draft sits in the order, as a tuple Python can sort.

    Takes the four things the order is *made of* rather than the `Draft` they
    came from, so each element can be varied on its own. `lint_passed` is why:
    `compose` raises rather than returning a score that fails lint, so no draft
    the tool surface produces will ever have it false — and an element no input
    can reach is an element no test can witness. Taken as an argument it is a
    value a caller can hold, which is what lets the element below the line be
    checked rather than asserted.

    The last two elements are two answers to one question — "these drafts are
    otherwise indistinguishable; which comes first?" — and both are needed. A
    plan is derived from the spec and does *not* carry the seed, so the
    candidates of one `draft(n=k)` call, which differ in nothing else, share a
    plan hash and would otherwise tie.
    """
    return (*musical_order(quality, lint_passed=lint_passed), plan_hash, _comparable_seed(seed))


def rank(drafts: Iterable[Draft]) -> list[Draft]:
    """The drafts, best first, by `draft_key`.

    Best first because that is how the ranking is read: the UI shows the top
    few, and the conductor is told which one leads. `sorted` is stable, and
    with a total key stability never shows — two drafts sharing every element
    are the same plan at the same seed, and either order is the same answer.
    """
    return sorted(drafts, key=draft_key)


ELEMENTS: Final[tuple[str, ...]] = (
    "legality",
    "misses",
    "breach",
    "tier",
    "plan_hash",
    "seed",
)
"""The six elements, in the order they decide, named as they are spoken to a user.

One name per element of `OrderKey`, in the same order, and it is a *literal*
table rather than something derived from `musical_order`: an explanation is
words, and a name generated from an index would be a name nobody chose. The
two are held together by a test that reads both — the length, and a case per
name — so a seventh element cannot be added without an eighth line here.
"""


def deciding_element(better: Draft, worse: Draft) -> str | None:
    """Which element of the order put `better` ahead of `worse`.

    The first element on which the two keys differ, which is the one `sorted`
    read, so the sentence and the ranking cannot disagree about why a draft sits
    where it does. `None` when they differ on nothing — two drafts that share
    every element are the same plan at the same seed, so it is a return the type
    allows rather than one a caller should expect.
    """
    ahead = draft_key(better)
    behind = draft_key(worse)
    for name, left, right in zip(ELEMENTS, ahead, behind, strict=True):
        if left != right:
            return name
    return None


def draft_key(draft: Draft) -> OrderKey:
    """One draft's place in the order, read off the record."""
    return (*musical_key(draft), draft.plan_hash, _comparable_seed(draft.spec.seed))


def musical_key(draft: Draft) -> MusicalKey:
    """The part of a draft's key that decides whether a revision is worse.

    `draft_key` without the tiebreakers, and with them the coin flip — see
    `MusicalKey`. The record-level reader of `musical_order`, for the same reason
    `draft_key` is the record-level reader of `arbiter_order`.
    """
    return musical_order(draft.quality, lint_passed=draft.lint_passed)


def regression(parent: Draft, child: Draft) -> str | None:
    """Why `child` may not replace `parent`, or `None` when it may.

    A sentence rather than a flag, because the conductor reads it and the user
    is told it: naming how many bars moved and which one moved furthest is
    actionable, and `True` is not.

    Only *worse* is refused. Equal is allowed, and deliberately: a revision
    that moves the music without moving any measurement is exactly what "the
    same piece with a different detail" means, and refusing it would turn the
    ratchet into a veto on editing. For the same reason it compares the
    musical elements only, which is `musical_key` and nothing else.
    """
    was = musical_key(parent)
    now = musical_key(child)
    if now <= was:
        return None

    before = parent.quality.findings()
    after = child.quality.findings()
    if now[1] > was[1]:
        moved = f"misses {len(after)} quality bar(s) where {parent.draft_id} misses {len(before)}"
    elif now[2] > was[2]:
        moved = f"misses the same {len(after)} bar(s), but by more"
    else:
        moved = (
            f"misses {_top_miss(after)}, which sits above {_top_miss(before)} — "
            f"the highest bar {parent.draft_id} misses"
        )
    return (
        f"this revision {moved}, and a revision may not be worse than the draft it came "
        f"from. Keep {parent.draft_id}, or change something that moves the bar."
    )


def _breach(finding: QualityFinding) -> float:
    """How far past its bar a piece went, as a fraction of the bar itself.

    A fraction, because the eleven metrics are not in the same units: a semitone
    deficit and a ratio deficit cannot be summed, and adding them raw would let
    whichever metric happens to carry the largest numbers decide the order. The
    divisor is `abs(target)` — a bar of 0 says "never", and a piece past it has
    no scale to be measured against, so its miss is left as the raw value.

    The `abs` is the identity on every threshold `QUALITY_THRESHOLDS` holds,
    since all eleven bars are positive, so no value through `findings()` can
    witness it; it is here for the bar that is added one day with a negative
    one, which would otherwise invert its own miss.
    """
    miss = (
        finding.target - finding.measured
        if finding.direction == "min"
        else finding.measured - finding.target
    )
    return miss / abs(finding.target) if finding.target else miss


def worst_finding(findings: tuple[QualityFinding, ...]) -> QualityFinding | None:
    """The highest bar in a set of misses, or `None` when the piece is clean.

    The tier table decides, so this is the same rule `_tier_rank` ranks by and
    the same one `_top_miss` names a sentence from — which is why it is the
    owner of the comparison and both of those read it. Ties go to the first miss
    in `findings()` order, which is the quality report's own order, so a caller
    aiming at the worst bar aims at the same one twice.
    """
    if not findings:
        return None
    return min(findings, key=lambda finding: METRIC_TIERS.index(finding.metric))


def _tier_rank(findings: tuple[QualityFinding, ...]) -> int:
    """How important the *most* important bar this piece missed is. Higher is worse.

    The minimum index, inverted so that a worse miss ranks higher: a piece that
    misses step_ratio and tessitura is a piece with an unsingable melody,
    whatever else it got right. A clean piece ranks below every tier, which is
    correct — it has already won on how many bars it missed, and this element
    only ever decides between two pieces that missed the same number.
    """
    worst = worst_finding(findings)
    if worst is None:
        return 0
    return len(METRIC_TIERS) - METRIC_TIERS.index(worst.metric)


def _top_miss(findings: tuple[QualityFinding, ...]) -> str:
    """The name of the highest bar in a set of misses, for a sentence."""
    worst = worst_finding(findings)
    return worst.metric if worst is not None else "nothing"


def _comparable_seed(seed: int | None) -> int:
    """A seed that sorts, for a draft that may not have one.

    `CompositionSpec.seed` is optional — the parser leaves it unset — while
    `Draft.spec.seed` is concrete for every draft the `draft` tool makes, since
    it stamps each candidate with the seed it composed at. So the `None` arm is
    here because the *type* says it can be, not because a tool produces it, and
    it sorts an unseeded draft before every seeded one: there is no seed to
    order it by, and a number invented for it would be a claim about the
    artifact. Omitting the arm entirely would not remove the case, it would
    move it into a `TypeError` raised from inside `sorted`.
    """
    return seed if seed is not None else -1


__all__ = [
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
    "worst_finding",
]
