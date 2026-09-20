"""The melody line: one bar at a time, inside what the plan permits.

`melody_bar` is the entry point and the rest is the vocabulary it draws on —
the walk, the leap and the step that answers it, the chord tone a bar closes
on. Every knob these read comes from `MelodyShape` (the plan), never from a
constant here, which is what the plan-seam tests hold them to.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from saimc.compose.dynamics import shaped_velocity
from saimc.compose.forms import (
    BREATH_TICKS,
    LEAP_MIN_SEMITONES,
    STEP_MAX_SEMITONES,
    chord_tone_degrees,
    scale_walk,
)
from saimc.compose.linter import DISSONANT_INTERVALS
from saimc.compose.motif import (
    DEFAULT_MELODY_SHAPE,
    BarSlot,
    MelodyShape,
    MotifVariant,
    apply_rhythm,
)
from saimc.compose.score import (
    DEFAULT_VELOCITY,
    PPQ,
    VOICE_MELODY,
    NoteEvent,
)
from saimc.instruments import (
    MelodyBand,
)


def downbeat_anchor(rng: random.Random, tone_count: int) -> int:
    """Choose a bar's starting chord tone: root (50%), third (30%), fifth (20%)."""
    roll = rng.random()
    if roll < 0.5:
        return 0
    if roll < 0.8:
        return 1 % tone_count
    return 2 % tone_count


# The melody's tessitura is the melody instrument's, and it is looked up
# per piece: `saimc.instruments.melody_band` turns the instrument into
# the window `_place_bar` places every bar inside. There is no module
# constant any more. Until it was removed this was `MELODY_LOW_MIDI = 64`
# to `MELODY_HIGH_MIDI = 84` (E4 to C6) for all sixty-six instruments, so
# a tuba, a piccolo and a piano were written the same twenty semitones
# and `compose()` returned a byte-identical score for each.
#
# What the band is *for* is unchanged: every bar is placed inside it by
# `_place_bar`, which is what bounds the piece's range — a bar is moved
# to a register, never clamped note by note into one. It is wide enough
# for a phrase peak and for a line that moves, and narrow enough that the
# accompaniment has a register of its own below it. How wide that has to
# be is `instruments.LINE_BAND_SEMITONES`, and the window is placed at
# the middle of the tessitura there; this is the same constraint seen
# from the other end.
#
# The floor and ceiling are only reachable by octave placement, and a
# line wider than the band divided by 12 rotations is not guaranteed a
# register inside it: a twelve-semitone line sits in 9 of the 12 octaves
# it could be written in, so 3 of them have no register inside the band.
# A bar against them has nowhere left to go — so it sounds where the
# placement put it, up to a tone or two past the edge, rather than being
# displaced note by note into a tear. `_place_bar` ranks placements by
# how many notes each leaves outside, so a bar that *can* fit does; the
# few that cannot are the price of an octave-quantised register, paid at
# the edge by a tone. A narrow instrument pays it more often, which is
# the honest cost of writing for it rather than for the piano.
# How far a bar's walk may reach from its anchor before it is folded
# back an octave. Ten degrees is at most 18 semitones in either diatonic
# mode, so a bar that stays inside this window fits the band above and
# is never folded for reasons the motif did not already imply.
_WALK_REACH_DEGREES: int = 5
# How far from the drawn anchor a bar may be restated, in scale degrees:
# an octave each way. The band is narrower than that in every direction
# that matters, so a wider lattice would only offer placements the
# tessitura refuses — but a narrower one would leave the octave grid
# (`_place_bar`) as the only way to move a line, and an octave step is
# twelve semitones when the entrance wanted three.
_START_REACH_DEGREES: int = 8
# The widest interval a bar may be entered on. A leap is recovered by the
# step that follows it, and past an octave there is no answer the ear
# accepts: the line is lost before the recovery arrives. It binds in one
# place — a bar whose line fits the band in a single register and whose
# every restatement rubs the bass has no way in but a leap, and this says
# which leap. Inside the octave, size is still the caller's tiebreak.
# Which rank position it occupies is what makes it bind, and the apex
# needs its own answer: the bound sits behind the register everywhere
# else (the band is a fact about the instrument and a rub is a refusal)
# but ahead of it at the apex, which is the one bar per section that was
# otherwise free to ignore it.
_MAX_ENTRANCE_SEMITONES: int = 12
# Which field of `_place_bar`'s rank tuple counts the notes left rubbing
# the bass. It is the second field of both rank shapes — the apex's and
# every other bar's — and it is read by `melody_bar`, which retries a bar
# whose winner rubs.
_RANK_RUBBING: int = 1


def _bound_walk(degrees: list[int], centre_degree: int) -> list[int]:
    """Fold a bar's degree walk into one octave of where it starts.

    A sequence climbs a chord tone per replay, and a long bar replays the
    motif many times, so without this the line walks out of the
    instrument. Folding is an octave displacement — what a sequence does
    at its seam anyway — and the leap-answering pass that follows treats
    it as the leap it is.
    """
    low = centre_degree - _WALK_REACH_DEGREES
    high = centre_degree + _WALK_REACH_DEGREES
    folded: list[int] = []
    for degree in degrees:
        while degree > high:
            degree -= 7
        while degree < low:
            degree += 7
        folded.append(degree)
    return folded


def _answer_leaps(
    degrees: list[int],
    *,
    fixed_tail: int = 0,
    remainders: tuple[int, ...] = (0, 2, 4),
    shape: MelodyShape = DEFAULT_MELODY_SHAPE,
) -> list[int]:
    """Answer every leap in a bar's degree walk with a turn back.

    A leap that is not answered is the fault a listener notices first,
    and the answer has to survive the licence pass to be heard at all: a
    single step back lands on a non-chord tone, and the licence covers
    that only when it is entered *and* left by a step. So the answer is
    the whole figure — the step back, and the step that leaves it, which
    is a passing tone between the two chord tones either side of it.

    The turn comes back the way the leap went, so a phrase that has
    climbed answers downward and one that has dived answers up.

    The leap itself is kept whenever it can be: a leap the licence can
    cover lands *on* the harmony — three degrees from the chord's fifth
    is its root — and a leap that lands off it cannot be licensed at all,
    because a non-chord tone is entered by a step or not at all. So a leap
    whose landing is off the harmony is re-aimed at it rather than undone.
    The chord scale puts a chord tone on either side of any landing, the
    side further along the leap is a degree wider than the motif drew it,
    and a degree is the whole of the change — where the motif drew a
    fourth the bar sounds a fifth, and the leap it drew is heard. Only a
    landing with no chord tone beyond it narrows, and one with neither is
    undone as it always was: the landing steps back to where the line came
    from, which leaves the bar's contour intact and one leap poorer rather
    than one dissonance richer.

    Re-aiming happens here rather than where the motif is drawn, and the
    reason is the frame: `_draw_step` chooses a step from the motif's own
    start, and the chord tones it can see are the ones congruent to *that*
    degree, while this pass reads a landing against the bar's anchor — a
    different chord tone, so a landing the draw called consonant is one
    this pass calls a dissonance. Constraining the draw to chord tones
    measured at half the leaps drawn and *fewer* of them rendered, because
    the two frames disagree in one case in three.

    `fixed_tail` is how many of the bar's last slots the answer may not
    move — the closing gesture's note, which has to land where it lands.
    A leap into one of those is answered from the other side instead:
    the note *before* the landing takes the step, which is how a cadence
    is approached in the first place. Rewriting the note before a
    landing can leave a leap before *that* one, so the loop walks back
    over the slots it has rewritten.

    A slot the pass has written as the *approach* to its pair is settled,
    and settled slots are never written again. Without that, two repairs
    that face each other undo one another on every pass and the walk never
    finishes: a bar whose answer has to come from before the landing can
    be re-leapt by the repair of the pair before it — the engine hung on
    exactly that. Every walk back settles one more slot at a lower index
    than the last, so the walk cannot go round, and a leap whose both
    sides are settled is left to the bar after this one, whose entrance
    answers it (`_entry_answer`).

    A leap into the bar's *last* slot is the same case one step along,
    and it is the case this pass used to get wrong. There is no room
    after such a landing for any answer — the turn needs three slots —
    so the landing was treated as a fault and the leap undone. But a leap
    the bar cannot answer is a leap the bar does not have to answer when
    the bar after it can: `_entry_answer` turns that bar's opening step
    back the way the leap came, and the placement ranks an answered
    entrance above an unanswered one. So the leap is kept, re-aimed at
    the harmony if it needs to be, and handed over — which is what makes
    the seam's own machinery reachable at all. Measured over the 840-piece
    grid (3 moods x 7 durations x 40 seeds), counting the pass at its own
    boundary so "keeps" is its share rather than a re-derivation of it:
    it keeps 58% of the leaps the motif draws where it kept 34%, the
    pieces whose melody ends with no leap in any bar fall from 254 to
    194, and the pieces carrying a move wider than an octave between
    onsets fall from 101 to 83. The direction is the durable claim — the
    counts move with the melody written above this pass, so they are
    recorded with their measure rather than kept as a target.
    """
    out = list(degrees)
    last_mutable = len(out) - fixed_tail - 1
    settled = [False] * len(out)
    index = 0
    while index + 1 < len(out):
        leap = out[index + 1] - out[index]
        if abs(leap) < shape.leap_degrees:
            index += 1
            continue
        back = -1 if leap > 0 else 1
        landing = index + 1
        if not settled[landing]:
            # Room for the whole answer, or room for none of it — but the
            # bar's last slot is not room *less*: it is the one landing the
            # bar after this one answers, so the seam's turn counts here.
            seam = fixed_tail == 0 and landing == last_mutable
            room = seam or (
                index + 3 <= last_mutable and not settled[index + 2] and not settled[index + 3]
            )
            if out[landing] % 7 not in remainders:
                # The landing needs the licence and cannot have it where it
                # sits: a step is the only way into a non-chord tone. Re-aim
                # it at the harmony, one degree out, and the leap is kept —
                # but only where the answer that licenses it has room to
                # follow, because a leap the bar cannot answer is the fault
                # this pass exists to remove, and keeping one here would be
                # that fault rather than a repair of it.
                aimed = _aim_leap(out[landing], back, remainders=remainders) if room else None
                if aimed is None or abs(aimed - out[index]) < shape.leap_degrees:
                    # Neither side of the landing is a chord tone the leap
                    # survives on — a third from the chord's third is the
                    # case — so the line keeps its shape and loses the leap.
                    out[landing] = out[index] + back
                    if index + 2 <= last_mutable and not settled[index + 2]:
                        out[index + 2] = out[landing] + back
                    index += 1
                    continue
                out[landing] = aimed
            if room:
                if seam:
                    # Nothing left in this bar to write the answer on, and
                    # the bar after it owes the turn.
                    index += 1
                    continue
                # Landing on a chord tone, with room for the whole answer.
                out[index + 2] = out[landing] + back
                out[index + 3] = out[index + 2] + back
                index += 3
                continue
        if settled[index]:
            # Nothing left on this pair that may be written, so the leap
            # stands and the next bar's entrance is what answers it.
            index += 1
            continue
        # No room after the landing for the turn: the landing is where
        # the bar has to be, so the note before it steps into it.
        out[index] = out[landing] + back
        settled[index] = True
        index = max(index - 1, 0)
    return out


def _aim_leap(landing: int, back: int, *, remainders: tuple[int, ...]) -> int | None:
    """The chord degree a leap is re-aimed at, or `None` if there is none.

    A chord built in thirds puts a chord tone on either side of any degree,
    so this is a choice between two and not a search. `back` is the
    direction the line came from, so `landing - back` is the side further
    along the leap and is always the one a leap wants: it is a degree
    wider than the motif drew, so the leap survives by construction. The
    nearer side is the fallback, and it is the caller's to check — one
    degree in can shorten a fourth to a third, and a leap that is no
    longer a leap is the fault this pass exists to answer.
    """
    wide = landing - back
    if wide % 7 in remainders:
        return wide
    narrow = landing + back
    return narrow if narrow % 7 in remainders else None


def _licit_line(
    degrees: list[int],
    *,
    remainders: tuple[int, ...],
    fixed_tail: int = 0,
    shape: MelodyShape = DEFAULT_MELODY_SHAPE,
) -> list[int]:
    """Reshape a walk so every non-chord tone is a passing or neighbour tone.

    A non-chord tone owes the licence a single degree on each side: it is
    entered by a step and left by a step, and one degree is a semitone or
    a whole tone in every diatonic mode. That makes the licence a
    statement about the *walk* — a line that arrives at a non-chord tone
    by a third has already broken the rule, and one that leaves by a
    third breaks it again — and it makes the repair a local one: the
    offending note moves by a degree, the way it was already going.

    Only the notes around a non-chord tone move, and they move by a
    single degree, so a bar's rhythm is untouched and its contour barely
    shifts. Of the two degrees the moved note could take, the one nearer
    the note on its far side wins, which is what keeps the repair from
    opening a gap of its own.

    An earlier version of this repair snapped the non-chord tone onto the
    chord instead. That is the right answer for a note with nowhere to
    go, but as a general repair it moves the *wrong* note: it puts the
    decorated tone on the harmony and leaves the harmony's own tones a
    third or a fourth apart, which is the very leap the licence exists to
    prevent. Bending the line around the non-chord tone keeps the music
    and removes the fault.

    `fixed_tail` is the bar's closing gesture, which has to land where it
    lands; a line that cannot bend toward it is left for the snap pass,
    which is the one repair allowed to move a note the bar has pinned.
    """
    out = list(degrees)
    count = len(out)
    last_mutable = count - fixed_tail - 1
    for index in range(1, count):
        if out[index] % 7 in remainders and out[index - 1] % 7 in remainders:
            continue
        gap = out[index] - out[index - 1]
        if abs(gap) == 1:
            continue
        # A repeat is as unwalkable here as a leap: the non-chord tone
        # has to be left by a degree, and staying still is not one.
        upward = gap > 0 if gap else _was_rising(out, index)
        far = out[index + 1] if index + 1 < count else None
        if index <= last_mutable:
            out[index] = _bent_step(out[index - 1], upward, far, shape=shape)
        elif index - 1 <= last_mutable:
            out[index - 1] = _bent_step(
                out[index], not upward, out[index - 2] if index > 1 else None, shape=shape
            )
    return out


def _bent_step(
    anchor: int, upward: bool, far: int | None, *, shape: MelodyShape = DEFAULT_MELODY_SHAPE
) -> int:
    """One degree from `anchor`, in the direction the line was going.

    A bend is the licence's repair, and a leap is the fault that licence
    exists to prevent, so a candidate that leaves `far` a step away
    outranks one that is merely nearer to it — a bend that closes the gap
    it was called for and opens a fourth on the far side has repaired
    nothing. Where neither is a step away, the other candidate wins only
    when it leaves `far` strictly nearer, which is the older rule and
    still the one that keeps a line's intervals from widening.
    """
    up, down = anchor + 1, anchor - 1
    if far is not None:
        ordered = (up, down) if upward else (down, up)
        for step in ordered:
            if abs(far - step) <= shape.chord_tone_degrees:
                return step
        if abs(far - ordered[1]) < abs(far - ordered[0]):
            return ordered[1]
    return up if upward else down


def _was_rising(degrees: list[int], index: int) -> bool:
    """Whether the line was rising before a repeated note."""
    for position in range(index - 1, 0, -1):
        step = degrees[position] - degrees[position - 1]
        if step:
            return step > 0
    return True


def _snap_to_chord(
    degree: int,
    *,
    tone_count: int,
    prefer_up: bool,
    neighbours: tuple[int, ...] = (),
    shape: MelodyShape = DEFAULT_MELODY_SHAPE,
) -> int:
    """The bar's nearest chord degree to `degree`, preferring one direction.

    A chord scale's tones sit on every other degree, so a tone that is
    not a chord tone always has one within two degrees — which is what
    makes this a snap and not a search.

    `neighbours` are the degrees the snapped note sits between, and a
    snap decides the intervals *they* are heard on, so what the chord
    tone costs them is weighed before how far the note itself moves:

    - A neighbour the licence covers is a step away, and it has to stay
      one. Snapping a note onto a chord tone a third from such a
      neighbour leaves the neighbour a note the licence cannot cover any
      more, so the pass snaps that one too — and two snaps facing away
      from each other widen a step into a leap, which is the fault the
      whole line is written to avoid.
    - A chord-tone neighbour needs no licence, but a leap between two
      chord tones is still a leap: the line has to answer it, and the
      room to answer it may not be there. So a leap behind the note
      costs less than a stranded neighbour and more than neither.

    Only among chord tones that are equally kind to the neighbours does
    the smaller move win, and then the direction the line was going.
    """
    remainders = chord_tone_degrees(tone_count)
    candidates = [
        degree + offset for offset in (1, -1, 2, -2) if (degree + offset) % 7 in remainders
    ]

    def cost(candidate: int) -> tuple[int, int]:
        """How many neighbours the move leaves stranded, and how many leaping."""
        stranded = leapt = 0
        for neighbour in neighbours:
            gap = abs(neighbour - candidate)
            if neighbour % 7 in remainders:
                leapt += gap >= shape.leap_degrees
            else:
                stranded += gap != 1
        return stranded, leapt

    return min(
        candidates,
        key=lambda candidate: (
            *cost(candidate),
            abs(candidate - degree),
            0 if (candidate > degree) == prefer_up else 1,
        ),
    )


def _hold_tied_pairs(slots: list[BarSlot]) -> list[BarSlot]:
    """Write a tied pair as the one pitch it is.

    A tie joins two noteheads into a single sound, so the second of them
    has to carry the first's degree. The passes between the rhythm
    library and the licence each rewrite a note of the line without
    knowing which notes are tied to their neighbour — the entry turn
    writes the bar's opening two moves outright — and a pair left at two
    pitches is not a tie at all: the engraver draws a slur between
    different heights, and the performance layer, which plays a tied
    continuation as nothing, drops a pitch the line meant to sound.

    Both halves are chord tones of the bar's chord scale when the rhythm
    library drew the tie, so holding the pair together invents no
    dissonance; it also cannot open a leap, since the tie's own pitch is
    the one already there.
    """
    out = list(slots)
    for index, slot in enumerate(out[:-1]):
        if slot[3]:
            offset, duration, _degree, tie = out[index + 1]
            out[index + 1] = (offset, duration, slot[2], tie)
    return out


def _legal_slots(
    slots: list[BarSlot],
    *,
    tone_count: int,
    chord_root: int,
    scale: tuple[int, ...],
    avoid_pcs: frozenset[int] = frozenset(),
    shape: MelodyShape = DEFAULT_MELODY_SHAPE,
) -> list[BarSlot]:
    """Snap every slot the passing-tone licence cannot cover to a chord tone.

    A slot sounding a chord tone of the bar (`forms.chord_tone_degrees`)
    needs no licence. A slot that does not is a non-chord tone, and the
    licence covers it only when it is unaccented, no longer than a
    quarter, and entered and left by a step that abuts it on both sides
    (`linter.legal_non_chord_tone`). Whatever fails those is snapped to
    the nearest chord degree in the direction the line was already
    moving, so the bar keeps its rhythm and its contour and the note
    sounds the chord tone it was decorating instead of one the harmony
    forbids.

    `avoid_pcs` are the pitch classes of the bar's other sounding
    voices, and a non-chord tone a semitone from one of them is snapped
    whatever its surroundings: the collision check flags a m2 and a M7
    between two sounding notes, and those are the same pair of pitch
    classes an octave apart, so the pass decides in pitch class where no
    octave placement can undo it. Only non-chord tones are affected — a
    chord tone against a chord tone of the same bar is a voicing, which
    the check exempts.

    A tie holds one pitch across two noteheads, so a tied pair stands or
    falls together and is snapped as a unit — otherwise the two halves
    could snap opposite ways and the engraver would draw a tie between
    two different pitches.
    """
    degrees = [slot[2] for slot in slots]
    count = len(degrees)
    if count == 0:
        return slots
    tied = [bool(slot[3]) for slot in slots]
    # A tie is a group of two, and the caller's line ends inside the bar:
    # a forward tie on the last slot names a partner that is not there,
    # and the group's walk below would take a degree from past the end.
    assert not tied[-1], "a tie on the bar's last slot holds into no slot"
    remainders = chord_tone_degrees(tone_count)

    def pitch_class(index: int) -> int:
        """The slot's pitch class — what the octave placement cannot change."""
        return (chord_root + scale[degrees[index] % 7]) % 12

    def clashes(index: int) -> bool:
        """Whether a non-chord tone here would rub another voice."""
        return any((pitch_class(index) - other) % 12 in DISSONANT_INTERVALS for other in avoid_pcs)

    def needs_snapping(index: int) -> bool:
        if degrees[index] % 7 in remainders:
            return False
        if clashes(index):
            return True
        # The bar's first slot falls on the downbeat, and its last has no
        # note inside the bar to step away to; either is unlicensable.
        if index == 0 or index == count - 1:
            return True
        if slots[index][1] > PPQ:
            return True
        if tied[index] or tied[index - 1]:
            return True
        # Both neighbours must be a single degree away: one degree is a
        # semitone or a whole tone, two is a third and no step.
        return not (
            abs(degrees[index] - degrees[index - 1]) == 1
            and abs(degrees[index + 1] - degrees[index]) == 1
        )

    changed = True
    while changed:
        changed = False
        for index in range(count):
            if not needs_snapping(index):
                continue
            start = index - 1 if index > 0 and tied[index - 1] else index
            stop = start + 1 if tied[start] else start
            prefer_up = start > 0 and degrees[start] >= degrees[start - 1]
            neighbours = tuple(
                degrees[position] for position in (start - 1, stop + 1) if 0 <= position < count
            )
            for member in range(start, stop + 1):
                degrees[member] = _snap_to_chord(
                    degrees[member],
                    tone_count=tone_count,
                    prefer_up=prefer_up,
                    neighbours=neighbours,
                    shape=shape,
                )
            changed = True
    return [(slot[0], slot[1], degrees[index], slot[3]) for index, slot in enumerate(slots)]


def _walk_shape(
    variant: MotifVariant,
    *,
    bar_ticks: int,
    tone_count: int,
    closing_degree: int | None = None,
    shape: MelodyShape = DEFAULT_MELODY_SHAPE,
) -> tuple[list[int], list[int]]:
    """Walk one bar's motif: scale degrees and durations, in slot order.

    The degrees are relative to the bar's anchor chord tone, and the
    shape does not depend on which tone that is: every candidate start
    sits a whole number of chord tones above the others, so the same
    shape serves all of them, shifted.

    `closing_degree` is written over the walk's last slot before the
    leap answering runs, so the answer can step into the note the bar
    has to land on rather than turn away from it. `_close_bar` writes
    the same degree again once the start's offset is applied — the walk
    here only has to know where the bar is going.

    The walk is folded to within an octave of where it starts, its leaps
    are answered by a turn back, and the line is bent so every note off
    the chord is a step from both its neighbours. Those three passes are
    what let the bar reach the licence already legal: the generator and
    the linter have to agree note for note, and the safest way to agree
    is for the walk to be written the way the linter reads it.
    """
    degrees: list[int] = []
    durations: list[int] = []
    offset = 0
    degree = variant.anchor_offset
    while offset < bar_ticks:
        for cell in variant.motif:
            if offset >= bar_ticks:
                break
            # The step is the move *into* the note, so it is taken before
            # the note is emitted: the motif's first step is always 0 —
            # "start on the bar's anchor tone" — and taking it afterwards
            # sounded that step as a repeat of the anchor and dropped the
            # last step the motif actually drew.
            degree += cell.step
            degrees.append(degree)
            durations.append(min(cell.length_ticks, bar_ticks - offset))
            offset += cell.length_ticks
        if not variant.repeat:
            break
        # The sequence advances one chord tone per replay, so each replay
        # starts on the next tone of the chord.
        degree += shape.chord_tone_degrees
    if not degrees:
        return degrees, durations
    if closing_degree is not None:
        degrees[-1] = closing_degree
    remainders = chord_tone_degrees(tone_count)
    return (
        _licit_line(
            _answer_leaps(
                _bound_walk(degrees, degrees[0]),
                fixed_tail=1 if closing_degree is not None else 0,
                remainders=remainders,
                shape=shape,
            ),
            remainders=remainders,
            shape=shape,
        ),
        durations,
    )


def _closing_tone(
    closing_degree: int,
    offset: int,
    *,
    half_cadence: bool,
    shape: MelodyShape = DEFAULT_MELODY_SHAPE,
) -> int | None:
    """The degree a bar closes on when its walk starts `offset` away.

    The gesture is written in the bar's own frame — the walk was shaped
    around it, so the last note steps into it — and a bar restated a
    chord tone higher therefore closes one chord tone higher, with the
    approach intact. What the gesture may not do is land somewhere the
    phrase has not asked for: the half cadence rests on the root of the
    V, the final bar on the tonic or its third (`closing_degree`, whose
    chord degrees are 0 and 2). A start whose closing tone falls outside
    that has no closing degree, and the caller drops it.
    """
    degree = closing_degree + offset
    allowed = (0,) if half_cadence else (0, shape.chord_tone_degrees)
    return degree if degree % 7 in allowed else None


def _close_bar(
    slots: list[BarSlot],
    *,
    degree: int | None,
    ticks: int | None,
) -> list[BarSlot]:
    """Give a bar's last slot the phrase's closing gesture.

    A half cadence lands on the chord's root, the final bar on the tonic
    or its third, a breathing bar simply shortens what it had. The
    gesture is applied to the *degrees*, before the licence pass runs,
    so the pass approves the notes that are actually sounded — writing
    the closing pitch in afterwards is what left a licensed passing tone
    a leap away from the note it was licensed to step into.

    `degree` arrives in the bar's own frame, the one the walk was shaped
    in (see `_closing_tone`), not in the chord's: the degrees the slots
    carry are already shifted by whatever tone of the chord the bar
    starts on, and a closing degree that ignored that shift would be
    written a chord tone or two below the line it has to step out of.
    """
    if not slots or (degree is None and ticks is None):
        return slots
    offset, duration, last_degree, tie = slots[-1]
    closing = last_degree if degree is None else degree
    out = list(slots)
    if closing != last_degree and len(out) >= 2 and out[-2][3]:
        # The gesture moved the note the bar ends on, so a tie into it is
        # off: a tie holds one pitch and this one now lands elsewhere. The
        # cadence is what the phrase asked for, so the tie yields.
        head = out[-2]
        out[-2] = (head[0], head[1], head[2], 0)
    out[-1] = (offset, duration if ticks is None else ticks, closing, tie)
    return out


def _land_on_chord(
    degrees: list[int],
    durations: list[int],
    *,
    tone_count: int,
    closing_degree: int | None,
) -> tuple[list[int], list[int]]:
    """Add the step that carries a bar's last note onto a chord tone.

    A bar ends on the harmony: its last note has nothing after it to be
    a passing tone *to*, so the licence cannot cover a non-chord tone
    there. The composer's move is not to rewrite that note but to keep
    walking — a line that has stepped down to A over a C chord takes one
    more step to G — so the repair is one extra note, and the note it
    was built to serve keeps the pitch the motif gave it.

    One step always suffices: a non-chord tone is one degree from a
    chord tone in a chord scale, so the landing is the note one degree
    further along the line. A bar that already ends on a chord tone —
    including every bar whose closing gesture put it there — is left
    alone.
    """
    if not degrees or closing_degree is not None:
        return degrees, durations
    remainders = chord_tone_degrees(tone_count)
    last = degrees[-1]
    if last % 7 in remainders:
        return degrees, durations
    forward = -1 if len(degrees) < 2 or last <= degrees[-2] else 1
    landing = last + forward
    if landing % 7 not in remainders:
        landing = last - forward
    half = durations[-1] // 2
    return [*degrees, landing], [*durations[:-1], durations[-1] - half, half]


def _start_offsets(
    anchor: int, tone_count: int, *, shape: MelodyShape = DEFAULT_MELODY_SHAPE
) -> tuple[int, ...]:
    """Every chord tone a bar could be restated on, the drawn one first.

    A bar is one line on one chord, and every tone of that chord is a
    place the line can begin: restating it from the third or the fifth is
    how a composer moves a phrase into another register without rewriting
    a note of it. The offsets are the chord's own degrees — the ones
    congruent to a chord tone modulo 7 — so the restatement keeps every
    note of the line a chord tone of the bar (`_legal_slots` then has
    nothing to snap and the line survives intact), and its reach is an
    octave either way, which is as far as the tessitura band can use.

    The drawn anchor leads the tuple because the caller keeps it when
    nothing else fits better; the rest of the lattice is what lets a bar
    come in by step when its own register would have made it leap.
    """
    drawn = shape.chord_tone_degrees * anchor
    return (drawn, *(o for o in _chord_lattice(anchor, tone_count) if o != drawn))


def _apex_starts(
    anchor: int, tone_count: int, *, shape: MelodyShape = DEFAULT_MELODY_SHAPE
) -> tuple[int, ...]:
    """The higher tones a section's peak bar may be restated on.

    The apex is the one bar whose register is chosen rather than fitted,
    and a *lift* is what makes it the peak: the line goes up a tone or
    two of its own chord — a third or a fourth, the interval a phrase
    rises by — while every other bar is placed where its entrance is
    plainest. That bound is the whole point. Restating the bar an octave
    up is the same statement made by a leap the listener has to recover
    from, and it is what used to make the section's peak arrive as a
    twelve-semitone jump rather than as the top of a climb.

    The set is therefore the lattice's offsets strictly above the drawn
    anchor and strictly inside the octave: never empty (any six
    consecutive degrees hold two tones of a triad), and never a jump.
    """
    drawn = shape.chord_tone_degrees * anchor
    return tuple(o for o in _chord_lattice(anchor, tone_count) if drawn < o < drawn + 7)


def _chord_lattice(anchor: int, tone_count: int) -> tuple[int, ...]:
    """The degrees congruent to a tone of the bar's chord, within reach."""
    remainders = chord_tone_degrees(tone_count)
    reach = _START_REACH_DEGREES
    return tuple(offset for offset in range(-reach, reach + 1) if offset % 7 in remainders)


def _entrance_cost(entrance: int | None, prev_leap: int | None) -> int:
    """What it costs a bar to be entered the way a shift makes it enter.

    A step into the bar is what a melody does and it is free. A skip — a
    third or a fourth — still moves and still comes back easily, so it
    is next. A repeat leaves the line where it was, which is a note the
    bar did not need and a repeat the score counts. A leap is the one
    entrance a bar should avoid, because a leap is what the *listener*
    has to recover from.

    When what ran into this bar was itself a leap, the entrance is the
    note that answers it, so a step back the other way is the one
    entrance that is free and every other way in is a fault: a step
    carrying on the same way never answers, and a repeat is the line
    refusing to move at all, which the score reads the same way. A skip
    is no better there — a leap wants a step, and a third is not one.

    The caller ranks equal costs by the size of the entrance, so the
    grades are deliberately few: what matters is that a skip outranks a
    repeat and both outrank a leap, not that the numbers are spaced.
    """
    if entrance is None:
        return 0
    distance = abs(entrance)
    if prev_leap is not None and abs(prev_leap) >= LEAP_MIN_SEMITONES:
        if 0 < distance <= STEP_MAX_SEMITONES and (entrance > 0) != (prev_leap > 0):
            return 0
        return 3
    if distance == 0:
        return 2
    if distance <= STEP_MAX_SEMITONES:
        return 0
    return 1 if distance < LEAP_MIN_SEMITONES else 3


def _opening_step(pitches: Sequence[int], slots: Sequence[BarSlot]) -> int | None:
    """The bar's first *sounding* move, or None if it has only one note.

    A tied continuation is not struck, so the interval the ear hears
    first is the one out of the tie, not the one between the tied
    noteheads. Both the seam's answer (`_entry_answer`) and the
    placement's ranking ask whether the bar's opening move answers the
    leap it came in on, and a tied opening read as a repeat answers
    nothing — which would have the bar turn a seam it has already
    answered and leave the answer unheard.
    """
    if len(pitches) < 2:
        return None
    index = 1
    while index < len(pitches) and slots[index - 1][3]:
        index += 1
    return None if index >= len(pitches) else pitches[index] - pitches[0]


def _answered(entrance: int | None, opening: int | None) -> bool:
    """Whether a bar's opening move answers the leap it was entered on.

    The rule the score measures is the same one a listener hears: a leap
    is recovered by the step after it, and the step has to go back the
    way the leap came. `opening` is the bar's own first interval, so a
    bar entered by a step — or entered by nothing, at the top of the
    piece — has nothing to answer and is trivially fine.
    """
    if entrance is None or abs(entrance) < LEAP_MIN_SEMITONES:
        return True
    if opening is None:
        return False
    return 0 < abs(opening) <= STEP_MAX_SEMITONES and (opening > 0) != (entrance > 0)


def _entry_answer(entrance: int | None, opening: int | None) -> int | None:
    """The degree step that answers a leap into the bar, or None if none is owed.

    A leap across a bar line is a leap: the line has to come back, and
    the note that comes back is the bar's second. One degree the other
    way is a semitone or a whole tone in the opposite direction, which is
    exactly what `_answered` asks for — the same repair `_answer_leaps`
    makes inside a bar, applied to the one seam that pass cannot see.
    """
    if _answered(entrance, opening):
        return None
    assert entrance is not None
    return -1 if entrance > 0 else 1


def _place_bar(
    pitches: list[int],
    *,
    band: MelodyBand,
    prev_pitch: int | None,
    apex: bool,
    bass_pitches: tuple[int, ...] = (),
    chord_pcs: frozenset[int] = frozenset(),
    prev_leap: int | None = None,
    opening: int | None = None,
) -> tuple[tuple[float, ...], int, int, int, int]:
    """Choose the octave a bar's line sits in, and how well the bar fits it.

    The shift is applied to the whole bar, so the bar's intervals — its
    motif — come through unchanged. The only interval the choice can
    damage is the one into the bar from the previous bar's last note,
    which is why the octaves are ranked by that interval once the
    register is right (`_entrance_cost`).

    `bass_pitches` is what the bar sounds against. A melody note a
    semitone or a major seventh from a sounding bass note is the one
    collision the linter refuses, and register is the honest way to
    settle it: the note keeps its pitch class and moves away from the
    bass an octave at a time, which no rewrite of the line can do. So a
    shift that clears those is preferred to one that does not, after the
    tessitura and before the approach — the band is not negotiable, the
    collision is.

    `chord_pcs` are the pitch classes of the bar's own chord, and they
    are what keeps that count honest: the linter exempts a note that is a
    chord tone of its bar, because two tones of the bar's chord are a
    voicing and not a clash — a seventh chord may sound its own seventh
    against its root. Only the notes off the chord are counted, and those
    are the ones the licence pass would snap away from the bass anyway.

    `opening` is the bar's first *sounding* move, which is the caller's
    to know: a tied continuation is not struck, so the move the ear hears
    is the one out of the tie. It is the interval the seam is judged on
    and the octave cannot change it, so it is passed in rather than read
    off the first two notes, which a tie leaves at one pitch.

    An `apex` bar is the section's peak, and its height is already in
    its start (`_apex_starts` lifts the line by a tone or two of its own
    chord), so what is left for the octave here is only the band. What
    it weighs, after the band and the collision: a bounded entrance,
    then the smallest displacement that fits, then whether the entrance
    is answered, then the height, then the entrance's grade, then the
    smallest entrance.

    Weighing the displacement *before* the entrance's own grade and size
    is the one key that differs from every other bar's order, and it is a
    correction rather than a refinement: it used to sit last, the way it
    does for every other bar, which made the apex the one bar per section
    that would rather speak from the register it was written in than
    enter quietly from an octave away. Re-measured over the 210 pieces of
    the 3-mood x 7-duration x 10-seed grid by putting the displacement
    back last, counting the moves of more than an octave taken *across a
    rest* — the entrances a breath frames, which is where a misplaced
    apex is heard: they rise from 10 to 46, carried by 10 pieces instead
    of 42. Every one of them lands in the body rather than the apex — the
    apex bar carries none in either tree, and none on any of the 60
    accepted specs of the parser corpus either
    — because what the apex's own entrance decides is the register the
    section's later bars are written from. The interval it may not be
    bought with is still the one wider than any answer can cover, and the
    band and a rub still outrank the entrance both: the tessitura is a
    fact about the instrument and a collision is a linter refusal.

    The height used to be weighed above the displacement here, on the
    argument that an apex is *built* rather than fitted: among the
    octaves that fit the band, clear the bass and enter within the
    octave, the highest is the one the phrase is climbing to. That is
    true of the bar's *start*, and it is why `_apex_starts` draws it
    from the lattice above the anchor rather than taking it as drawn.
    For the octave it was measured and reverted, and it is left reverted
    here on a trade rather than on the reading that first settled it.
    Swapping the two keys over the 960 sections of the 3-mood x
    7-duration x 10-seed grid *raises* all three peak readings the
    displacement-first order was kept for: the apex bar holds the
    section's top in 54.7% of sections against 45.4%, the section's own
    top rises from 79.89 to 80.10 on a band running 63..84, and the
    share whose top lands in the section's later third (its bars from
    `2 * form // 3` on) goes from 77.6% to 82.1%. That contradicts the
    reading the revert was argued from — 43.5% / 75.9% / 79.59 — and
    dates it: it was taken before the breathing bar was written a breath
    short, and that change reversed its effect, because the tail a
    breathing bar gives up is often the part of its line that was
    highest. What the swap costs is the recovery, and that is what the
    order is kept for now: over the same grid the leap-recovery ratio
    falls from 0.8297 to 0.7776 and the pieces breaching its floor rise
    from 13 to 21 of 210 — the peak bought with the breath this phase
    wrote to answer the leaps. Until that trade is decided, the octave
    goes back to the smallest displacement that fits, and the height
    stays the last word it always was among the placements everything
    else has already admitted.

    Whether the entrance *is* answered is weighed before the height,
    though, and that is not the same key as the entrance's grade: a
    whole bar's line is the choice here, so a variant that comes back
    from the leap into it at the price of a semitone or two of the
    bar's top has bought the one thing the apex owes the phrase — the
    peak is a peak because it is arrived at and left, not because it
    is the highest note in a line that stalled on it.

    Every other bar weighs its entrance first: a step is free, a repeat
    is a note the bar did not need, a leap is what the listener has to
    recover from — and a leap the bar's own opening step answers is
    better than one it does not.

    Returns the ranking the placement earned, in the order the keys were
    weighed, then the shift itself, the number of notes it leaves outside
    the band — zero for every bar the walk wrote inside it, and
    occasionally one or two for a line too wide to sit in the band at any
    octave — the number of notes it still leaves rubbing the bass, and
    what its entrance costs. The ranking is handed back so the caller
    that chooses between *starts* ranks them on the same scale the
    octaves were ranked on, rather than on a second one of its own.
    """
    best: tuple[tuple[float, ...], int, int, int, int] | None = None
    for octave in range(-3, 4):
        shift = 12 * octave
        shifted = [pitch + shift for pitch in pitches]
        outside = sum(1 for pitch in shifted if not band.contains(pitch))
        rubbing = sum(
            1
            for pitch in shifted
            if pitch % 12 not in chord_pcs
            for bass in bass_pitches
            if abs(pitch - bass) in DISSONANT_INTERVALS
        )
        step_into_bar = None if prev_pitch is None else shifted[0] - prev_pitch
        entrance = _entrance_cost(step_into_bar, prev_leap)
        # Among entrances of the same grade the smaller one wins: the
        # octave grid can leave a bar with nothing but leaps to choose
        # between, and a bar entered a fifth away is a bar entered well
        # next to one entered a tenth away.
        gap = 0 if step_into_bar is None else abs(step_into_bar)
        # An entrance wider than an octave is a fault whatever else is on
        # offer: the line is lost before the step that recovers it can
        # arrive. So it is weighed before the entrance's grade — a repeat
        # the score counts is the smaller price — and, for the apex,
        # before the height, which may not be bought at that price.
        within = 0 if gap <= _MAX_ENTRANCE_SEMITONES else 1
        answered = 0 if _answered(step_into_bar, opening) else 1
        if apex:
            rank: tuple[float, ...] = (
                outside,
                rubbing,
                within,
                abs(octave),
                answered,
                -max(shifted),
                entrance,
                gap,
            )
        else:
            rank = (
                outside,
                rubbing,
                within,
                entrance,
                answered,
                gap,
                abs(shifted[0] - band.centre_midi),
            )
        if best is None or rank < best[0]:
            best = (rank, shift, outside, rubbing, entrance)
    assert best is not None
    return best[0], best[1], best[2], best[3], best[4]


def _pickup_pitch(
    *,
    band: MelodyBand,
    root: int,
    tones: tuple[int, ...],
    nearby: int | None,
    bass_pitches: tuple[int, ...],
) -> int | None:
    """The anacrusis pitch leading into the next bar, or None if none fits.

    A pickup is a chord tone of the bar it leads into, placed in the
    tessitura band near the note it follows — the register it must
    approach from, not an octave above it. Candidates that would sound a
    close m2/M7 against the sounding bass are skipped; when every
    candidate clashes the pickup is dropped rather than played against a
    clash.

    Among the playable ones the pickup is a *step* from the note it
    follows — never that note itself, and never a leap. A pickup that
    cannot step is not a pickup: the figure exists to leave the melody
    before the downbeat, and a chord tone a third or more from the note
    it follows would be a leap into the bar line with nothing after it
    to answer it, which is worse than no anacrusis at all.
    """
    target = nearby if nearby is not None else band.centre_midi
    candidates = sorted(
        (root + tone + 12 * octave for tone in tones for octave in range(-3, 4)),
        key=lambda pitch: abs(pitch - target),
    )
    steps = [
        candidate
        for candidate in candidates
        if band.contains(candidate)
        and 0 < abs(candidate - target) <= STEP_MAX_SEMITONES
        and all(abs(candidate - bass) not in DISSONANT_INTERVALS for bass in bass_pitches)
    ]
    return steps[0] if steps else None


def _final_closing_degree(rng: random.Random, *, shape: MelodyShape = DEFAULT_MELODY_SHAPE) -> int:
    """The degree the piece's last bar lands on: the tonic, or its third.

    Home twice as often as its third, because a resolution onto the third
    is a colour and one onto the tonic is an ending. Which third is not a
    constant: the chord tone away from the tonic is the plan's, so a plan
    that states a different chord spelling states its own close.

    Named rather than written inline so the read is a thing a test can
    hold. It is the only reader of `chord_tone_degrees` that is not a
    helper taking a shape, and an inline expression sharing its field with
    six other sites cannot be shown to read the plan at all — reverting
    this one to the constant leaves every test green. The probability
    itself stays a literal: it is a musical decision, but not one the
    quality thresholds name, which is the rule the plan's scope follows.
    """
    return 0 if rng.random() < 0.6 else shape.chord_tone_degrees


def _voiced_bar(
    variant: MotifVariant,
    *,
    rng: random.Random,
    bar_ticks: int,
    chord_tones: tuple[int, ...],
    closing_degree: int | None,
    shape: MelodyShape,
    is_final_bar: bool,
    short: bool,
) -> tuple[list[BarSlot], int | None]:
    """How long the bar is written, and how its slots are voiced.

    Returns the bar's slots and the length its closing note is held to — or
    `None` where nothing closes. Everything here happens in degree space: a
    slot's pitch depends on the bar's octave, which depends on the whole bar,
    so the walk is settled — walked, answered, landed on a chord tone — before
    any pitch is spelled, and `melody_bar` does the spelling.

    `short` is a bar that ends its phrase: a half cadence or a breath.
    """
    # A bar that ends its phrase is a breath shorter than its bar line,
    # and it is *written* that short: the walk, the rhythm library and
    # the closing gesture all see `written_ticks`, so the phrase's own
    # ending is inside the room the phrase has. Truncating a full bar
    # afterwards was the earlier shape of this and it took that ending
    # with it — the walk answers a leap with the note after it, so a bar
    # whose last slot was dropped ended on whatever interval was left
    # over. Re-measured over the 3-mood x 7-duration x 10-seed grid by
    # reconstructing the truncation: it drops 1444 slots the rhythm
    # library had voiced across 4311 bars, and leaves the mean
    # `leap_recovery_ratio` at 0.8256 against 0.8297 written short, with
    # the same 13 pieces of 210 below the bar. So the reason to write the
    # bar short is the dropped notes rather than where the ratio lands — a
    # note the bar never plays is a note the phrase never had.
    written_ticks = bar_ticks - BREATH_TICKS if not is_final_bar and short else bar_ticks
    degrees, durations = _walk_shape(
        variant,
        bar_ticks=written_ticks,
        tone_count=len(chord_tones),
        closing_degree=closing_degree,
        shape=shape,
    )
    degrees, durations = _land_on_chord(
        degrees,
        durations,
        tone_count=len(chord_tones),
        closing_degree=closing_degree,
    )
    slots = [
        (sum(durations[:index]), duration, degrees[index])
        for index, duration in enumerate(durations)
    ]
    rhythm_slots: list[BarSlot]
    if is_final_bar:
        # The closing bar keeps the motif's own rhythm: the resolution
        # is the one event that should not be dressed up.
        rhythm_slots = [(o, d, t, False) for o, d, t in slots]
    else:
        rhythm_slots = apply_rhythm(
            slots,
            rng=rng,
            weights=shape.rhythm_weights,
            remainders=chord_tone_degrees(len(chord_tones)),
        )

    # The gesture's durations are the last word on the bar, so they are
    # taken after the rhythm library has re-voiced it — and for the bar
    # that ends the piece, so is the bar's length.
    #
    # A bar that half-closes or breathes has to leave silence the ear
    # reads as air, and the silence is a named length rather than a
    # fraction of whatever the bar happened to end on. Halving the last
    # slot was the first shape of this and it left the breath to the
    # rhythm library's mercy: a bar closing on a 32nd left a 32nd of
    # rest, which is a seam, not a breath. The marking pass recorded the
    # phrase as ended and the listener heard it run straight on. The
    # named length is now the room `written_ticks` leaves, and the rest
    # is what is left over — nothing is removed after the bar is written,
    # so every note the rhythm library voiced is a note the bar plays.
    closing_ticks: int | None = None
    if rhythm_slots:
        last_offset, _last_duration, _last_degree, _last_tie = rhythm_slots[-1]
        if is_final_bar:
            # Held to the bar line.
            closing_ticks = bar_ticks - last_offset
        # A tie on the bar's last slot holds it into a slot the bar does
        # not have: the tie that crosses a bar line is drawn later, by
        # the pass that pairs one bar's last note with the next bar's
        # first. Left set, the licence pass reads a tie as a group of
        # two, so the walk leaves the bar looking for a slot beyond it.
        #
        # Measured: the rhythm library does not currently produce one.
        # `_op_tie` marks only the head of a pair, so the final slot is
        # never a head; removing this clearing leaves all 1080 pieces of
        # a 3-mood x 9-duration x 40-seed sweep byte-identical, and no
        # plan rhythm weighting tried made it fire either. It is kept as
        # the normalisation that makes the bar well-formed by
        # construction — `_legal_slots`' assert is the matching invariant
        # — rather than as a repair for a fault that is being reached.
        if rhythm_slots[-1][3]:
            offset, duration, degree, _tie = rhythm_slots[-1]
            rhythm_slots[-1] = (offset, duration, degree, False)

    return rhythm_slots, closing_ticks


def melody_bar(
    *,
    band: MelodyBand,
    variant: MotifVariant,
    chord_root: int,
    chord_tones: tuple[int, ...],
    scale: tuple[int, ...],
    anchor: int,
    prev_pitch: int | None,
    start_tick: int,
    bar_ticks: int,
    rng: random.Random,
    position: float,
    ticks_per_bar: int,
    seed_for_variation: int,
    shape: MelodyShape = DEFAULT_MELODY_SHAPE,
    is_final_bar: bool = False,
    half_cadence: bool = False,
    apex: bool = False,
    breathe: bool = False,
    pickup: tuple[int, tuple[int, ...]] | None = None,
    bass_pitches: tuple[int, ...] = (),
    prev_leap: int | None = None,
) -> list[NoteEvent]:
    """Render one bar of melody from a motif variant.

    The motif is walked in *scale degrees* of `scale` — the bar's own
    chord scale, spelled from `chord_root` — so a step is a semitone or a
    whole tone and the same shape lands correctly on every chord of the
    template. When `repeat` is set (the sequence operation) the motif
    keeps replaying from the top, the walk advancing one chord tone per
    cycle, until the bar is full. The bar's slots are then re-voiced
    through the melody's rhythm library (`motif.apply_rhythm`): dotted
    figures, 16th subdivisions, ties, which move durations and never
    pitches.

    Which of the chord's tones the bar starts on is then chosen: the
    drawn `anchor` when it leaves the bar sounding, and another tone of
    the same chord when it would force a leap into the bar from the
    previous one. Each candidate is walked, snapped to the passing-tone
    licence (`_legal_slots`) and placed in the tessitura, so the choice
    is made on a bar that is already legal and in register.

    Phrase shape:
    - an `apex` bar is placed at the top of the tessitura, with a
      velocity lift — the section's melodic peak;
    - a `half_cadence` bar ends early on the chord's root, leaving a
      rest (the phrase breathes on the V);
    - a breathing bar is written a breath short of its bar line, so the
      rest the phrase ends on is room the bar never wrote into (the
      length, and why it is a named one, is below);
    - the `is_final_bar` of the piece resolves onto the tonic or its
      third, held to the bar line;
    - when the bar leaves at least an eighth of space at its end and
      `pickup` is given (the next bar's root and tones), an anacrusis
      pickup note sounds on the last eighth, leading into the next bar.
      A bar that breathes or half-closes is the exception, and it is the
      shape rather than an oversight: the room at its end *is* the
      phrase's rest, so a pickup there puts a note where the rest is
      (the read says what the lint cannot see about the run that makes).

    `prev_leap` is the interval the previous bar ended on, when it was a
    leap: this bar's entrance is the note that answers it, so the choice
    of octave is told which way the answer has to go.
    """
    # The closing gesture's degree is decided before the walk is built:
    # the walk has to know which note the bar lands on so its leap
    # answering can approach that note by step. Its durations come later,
    # because the rhythm library re-voices the bar's slots first.
    closing_degree: int | None = None
    if is_final_bar:
        closing_degree = _final_closing_degree(rng, shape=shape)
    elif half_cadence:
        closing_degree = 0

    # Degrees first: a slot's pitch depends on the bar's octave, which
    # depends on the whole bar. So the walk is settled in degree space
    # before any pitch is spelled — walked, answered, and landed on a
    # chord tone, which is what most of the passing-tone licence needs
    # and what the rhythm library then dresses.
    #
    rhythm_slots, closing_ticks = _voiced_bar(
        variant,
        rng=rng,
        bar_ticks=bar_ticks,
        chord_tones=chord_tones,
        closing_degree=closing_degree,
        shape=shape,
        is_final_bar=is_final_bar,
        short=half_cadence or breathe,
    )

    # An apex bar is placed by height, not by its approach: it is the
    # section's peak, and the top of the band is worth a wide interval
    # into it. Every other bar tries each of the chord's tones as its
    # start, cheapest approach winning, and keeps the drawn one when
    # none of them makes the approach cheaper.
    #
    # The bar is placed before it is repaired. Register is the cheap fix
    # for a rub against the sounding bass — a note a semitone from the
    # bass a tenth below is the same note a semitone from it an octave
    # up — and moving the whole bar never touches the line. Only a bar
    # that rubs at every octave the tessitura allows is handed to the
    # licence pass with the bass's pitch classes to steer around, and
    # that pass rewrites the line, so a bar reaches it only when nothing
    # else can be done.
    lattice = _start_offsets(anchor, len(chord_tones), shape=shape)
    # The drawn start is the lattice's first offset, and it is kept as the
    # fallback below: a bar whose every start is barred by the closing
    # gesture is restated where it was drawn. Read from the lattice rather
    # than recomputed, so the plan's chord tone is read in one place.
    drawn = lattice[0]
    starts = _apex_starts(anchor, len(chord_tones), shape=shape) if apex else lattice
    if closing_degree is not None:
        # The closing gesture is a chord tone of the bar, and the bar it
        # closes is one of the chord's tones tall. Which *one* is not
        # free: the half cadence rests on the root of the V and the
        # final bar on the tonic or its third, so a start whose closing
        # tone lands elsewhere is not a candidate. This is also what
        # keeps the landing a step away: the walk was shaped around the
        # closing degree, so shifting the bar by the start's own offset
        # moves the two together and the approach survives.
        starts = tuple(
            offset
            for offset in starts
            if _closing_tone(closing_degree, offset, half_cadence=half_cadence, shape=shape)
            is not None
        ) or (drawn,)
    bass_pcs = frozenset(pitch % 12 for pitch in bass_pitches)
    chord_pcs = frozenset((chord_root + tone) % 12 for tone in chord_tones)
    remainders = chord_tone_degrees(len(chord_tones))

    def build(
        start: int,
        entry: int | None,
        avoid_pcs: frozenset[int],
    ) -> tuple[tuple[float, ...], int, list[BarSlot], int | None, int | None]:
        """One start's bar: its line, its register, and how well it fits.

        `entry` forces the bar's opening two moves (see `_entry_answer`) —
        the turn that answers a leap into the bar — and is applied before
        the closing gesture and the licence pass, so what is placed and
        ranked is the line the bar will sound. A tie the rhythm library
        drew is held across those rewrites (`_hold_tied_pairs`), so a
        turn written onto a tied note comes out as the move *out of* the
        tie, which is the only move the bar has there. The last two
        elements are the interval the bar is entered on and the interval
        it opens with, which together say whether the entrance was
        answered.
        """
        closing = (
            None
            if closing_degree is None
            else _closing_tone(closing_degree, start, half_cadence=half_cadence, shape=shape)
        )
        degrees = [degree + start for _offset, _duration, degree, _tie in rhythm_slots]
        if entry is not None and len(degrees) > 2:
            if rhythm_slots[0][3]:
                # The bar opens on a tie: its first notehead sounds
                # through the second, so the seam hears one move where two
                # are written, and the turn goes on the move out of the
                # tie. Writing it twice would put the turn's second step
                # on a note the ear never hears struck and leave a third
                # standing at the seam with nothing to answer it.
                degrees[2] = degrees[0] + entry
            else:
                degrees[1] = degrees[0] + entry
                degrees[2] = degrees[1] + entry
        closed = _close_bar(
            [
                (offset, duration, degrees[index], tie)
                for index, (offset, duration, _degree, tie) in enumerate(rhythm_slots)
            ],
            degree=closing,
            ticks=closing_ticks,
        )
        # The closing gesture is written onto the last slot after the
        # line has been shaped, which can leave the note before it a
        # third away rather than a step. Bending the line toward the
        # landing is the same repair the walk took, and it is the only
        # one the bar's last note is allowed to need.
        bent = _licit_line(
            [slot[2] for slot in closed],
            remainders=remainders,
            fixed_tail=1 if closing_degree is not None else 0,
            shape=shape,
        )
        candidate = _legal_slots(
            _hold_tied_pairs(
                [
                    (offset, duration, bent[index], tie)
                    for index, (offset, duration, _degree, tie) in enumerate(closed)
                ]
            ),
            tone_count=len(chord_tones),
            chord_root=chord_root,
            scale=scale,
            avoid_pcs=avoid_pcs,
            shape=shape,
        )
        pitches = [scale_walk(degree, chord_root, scale) for _o, _d, degree, _t in candidate]
        opening = _opening_step(pitches, candidate)
        rank, shift, _outside, _rubbing, _entrance = _place_bar(
            pitches,
            band=band,
            prev_pitch=prev_pitch,
            apex=apex,
            bass_pitches=bass_pitches,
            chord_pcs=chord_pcs,
            prev_leap=prev_leap,
            opening=opening,
        )
        approach = None if prev_pitch is None else pitches[0] + shift - prev_pitch
        # Everything the seam is judged on is already in the placement's
        # ranking; the only key left is which tone of the chord the bar
        # was drawn on, and it is last because a restatement is a device,
        # not a preference — it is kept when nothing about the bar's fit
        # makes it worse.
        return (
            (*rank, 0 if start == drawn else 1),
            shift,
            candidate,
            approach,
            opening,
        )

    def choose(
        avoid_pcs: frozenset[int],
    ) -> tuple[tuple[float, ...], int, list[BarSlot], int | None, int | None]:
        best: tuple[tuple[float, ...], int, list[BarSlot], int | None, int | None] | None = None
        for start in starts:
            here = build(start, None, avoid_pcs)
            entry = _entry_answer(here[3], here[4])
            if entry is not None:
                # The bar came in on a leap it does not answer, and the
                # step that would answer it is a cheap, local repair —
                # so the bar is built twice and the better of the two
                # kept, which leaves the placement free to prefer the
                # natural line when that is the better one.
                turned = build(start, entry, avoid_pcs)
                if turned[0] < here[0]:
                    here = turned
            if best is None or here[0] < best[0]:
                best = here
        assert best is not None
        return best

    chosen = choose(frozenset())
    if chosen[0][_RANK_RUBBING] and bass_pcs:
        # A note rubbing the bass is the fault no voicing can undo: the
        # rank weighs the line's own shape ahead of it, and the line was
        # shaped without knowing what the bass plays under it. So when the
        # winner rubs, the same bar is built once more with the licence
        # pass steered around the bass's pitch classes, and the better of
        # the two is kept. A bar that needed no steering comes back as the
        # same line, so this pass reaches only the bars that had no other
        # way out.
        steered = choose(bass_pcs)
        if steered[0] < chosen[0]:
            chosen = steered
    shift, rhythm_slots = chosen[1], chosen[2]

    notes: list[NoteEvent] = []
    for bar_offset, duration, degree, tie in rhythm_slots:
        tick = start_tick + bar_offset
        notes.append(
            NoteEvent(
                voice_id=VOICE_MELODY,
                # The bar's register is the placement's shift, applied to
                # the whole bar and to nothing else. Folding an
                # out-of-band *note* an octave instead would move it
                # against the line it belongs to — a note at the band's
                # floor lifted an octave is a twelve-semitone tear in the
                # middle of a phrase the walk wrote as a step — and no
                # later pass repairs an interval that no longer matches
                # the line the licence was checked against. A bar the
                # placement could not fit therefore sits at the band's
                # edge, which is where `_place_bar` already ranks it: the
                # count of notes left outside is the first key of the
                # ranking, so the octave that leaves the fewest is the
                # octave that wins.
                pitch_midi=scale_walk(degree, chord_root, scale) + shift,
                tick=tick,
                duration_ticks=duration,
                velocity=shaped_velocity(
                    base=DEFAULT_VELOCITY + 8 + (10 if apex else 0),
                    position=position,
                    tick=tick,
                    ticks_per_bar=ticks_per_bar,
                    rng_seed=seed_for_variation + tick,
                ),
                tie=bool(tie),
            )
        )

    # Anacrusis: the bar left room at its end, so an eighth-note pickup
    # on the next chord leads into the next downbeat.
    #
    # A bar that breathes, or that half-closes, is the one case where the
    # room is the point rather than the opportunity. Its notes stop a
    # breath before the bar line so the phrase can end on silence, and a
    # pickup on that bar's last eighth puts a note where the rest is: the
    # closing was marked, the gap was not, and the phrase ran on unbroken.
    # Measured over the 3-mood x 7-duration x 10-seed grid with the
    # suppression removed, 313 of 960 sections carry a run longer than
    # `PHRASE_BARS`, out to 15.9 bars — and *no* piece is refused for it,
    # because `_is_anacrusis_pickup` excludes the pickup from the linter's
    # run grouping. So the run the ear hears is one the lint cannot see,
    # which is why the suppression is written here rather than left to the
    # span rule to catch. The rest is what the ear hears the phrase end
    # on, and the pickup belongs to a bar that is still going somewhere.
    if (
        pickup is not None
        and not is_final_bar
        and not breathe
        and not half_cadence
        and notes
        and notes[-1].tick + notes[-1].duration_ticks <= start_tick + bar_ticks - PPQ // 2
    ):
        pickup_pitch = _pickup_pitch(
            band=band,
            root=pickup[0],
            tones=pickup[1],
            nearby=notes[-1].pitch_midi,
            bass_pitches=bass_pitches,
        )
        if pickup_pitch is not None:
            pickup_tick = start_tick + bar_ticks - PPQ // 2
            notes.append(
                NoteEvent(
                    voice_id=VOICE_MELODY,
                    pitch_midi=pickup_pitch,
                    tick=pickup_tick,
                    duration_ticks=PPQ // 2,
                    velocity=max(
                        1,
                        shaped_velocity(
                            base=DEFAULT_VELOCITY + 8,
                            position=position,
                            tick=pickup_tick,
                            ticks_per_bar=ticks_per_bar,
                            rng_seed=seed_for_variation + start_tick,
                        )
                        - 8,
                    ),
                )
            )
    return notes


__all__ = [
    "downbeat_anchor",
    "melody_bar",
]
