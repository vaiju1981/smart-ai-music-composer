"""The harmony voices: how the accompaniment under the tune is written.

`motif.py` is the melodic material's vocabulary, `percussion.py` the
kit's, and this module the third: the values that write the harmony
voice — which texture its leading layer states, the figure that texture
steps on, and the level each texture sounds at — with the clearance the
bed keeps from the tune's floor.

They live here rather than in `engine.py` because the plan carries them
and the plan cannot import the engine (the engine imports the plan), so
every value a plan field defaults to has to live somewhere both may
read. `bed_window` stays in `instruments.py`, and the split is the
honest one: an instrument's comfortable range is a fact about the
instrument, while where the bed sits relative to the tune — and how it
is written once it is there — is a decision about the voice.
"""

from __future__ import annotations

from dataclasses import dataclass

from saimc.compose.score import PPQ

HARMONY_STAB_INSTRUMENTS: frozenset[str] = frozenset(
    {
        "brass_section",
        "french_horn",
        "trumpet",
        "muted_trumpet",
        "trombone",
        "tuba",
    }
)
"""Instruments that state spacious chord accents rather than an ostinato.

A stabbed chord is two chord tones on each half of the bar. Brass does
that; an instrument that can sustain does not, so a broken-chord mood
whose harmony is strings or a piano gets the broken chord instead.
Which instruments these are is a fact about the instruments, so it is
not a plan field — the plan carries which texture the leading layer
states, not which instruments sound like themselves playing it.
"""

BROKEN_CHORD_MOODS: frozenset[str] = frozenset({"electrifying"})
"""The moods whose leading harmony layer states a broken chord.

The one place the old inline `mood != "electrifying"` survives, as a
value rather than a comparison buried in the generator: a plan states
the texture, and this table is only how the default plan looks one up.
"""

HARMONY_ARPEGGIO_STEP_TICKS: int = PPQ // 2
"""The interval the broken-chord figure steps on, an eighth note.

With it a bar holds `ticks_per_bar // step` onsets, so a longer step is
a sparser figure — a whole-bar step sounds one chord tone per bar, the
same shape as a stab but held. The plan carries it because the arpeggio
is the note-densest thing the accompaniment does, and it is what
`texture_hierarchy` measures the melody against.
"""

HARMONY_PAD_VELOCITY: int = 46
"""The level of the sustained bed. The quietest of the three: it is the
accompaniment, and a pad that competes with the tune stops being one."""

HARMONY_ARPEGGIO_VELOCITY: int = 52
"""The level of the broken-chord figure, a notch above the pad's.

Between the pad and the stab for the same reason its texture is: a
broken chord is more articulated than a held one and less emphatic than
an accent, so it sits between them in level as well.
"""

HARMONY_STAB_VELOCITY: int = 58
"""The level of a stabbed chord: the loudest of the three, because a
stab is an accent and the whole point of the texture is the attack."""

HARMONY_MELODY_CLEARANCE: int = 3
"""How far under the melody the bed is held.

Its top sits this many semitones below the lowest note the melody
reaches anywhere in the piece, so the two registers are disjoint and no
bed note is ever within the crowding window *above* of the tune's
floor. Three, not two, because two is a semitone count the crowd set
itself calls a rub — a bed held exactly that far under the tune's floor
could be legal by register and illegal by interval at the same time.
"""


@dataclass(frozen=True)
class HarmonyVoices:
    """Every decision the harmony voice's writing makes, as one value.

    Which texture the leading layer states, the interval its figure
    steps on, the level each texture sounds at, and the clearance the
    bed keeps from the tune. One value rather than six parameters
    because they travel together — `_generate_section` forwards them to
    `_generate_harmony_section` untouched — the same reason
    `ArrangementKnobs`, `SectionArc` and `MelodyShape` exist.

    No `__post_init__`: a shape a caller builds by hand is the caller's,
    and a shape that arrives as a plan has already been checked by
    `CompositionPlan.__post_init__`, which is where the bounds live.
    """

    broken_chord: bool = False
    """Whether the leading harmony layer states a broken chord.

    False here because a sustained pad is the calm default, which is two
    moods of three; which mood takes the other one is
    `BROKEN_CHORD_MOODS`, read by the default plan. The layers under the
    leading one always sustain, whatever this says — a piece with two
    harmony voices gets its broken chord on one of them, not both.
    """
    arpeggio_step_ticks: int = HARMONY_ARPEGGIO_STEP_TICKS
    """The interval the broken-chord figure steps on."""
    pad_velocity: int = HARMONY_PAD_VELOCITY
    """The level of the sustained bed."""
    arpeggio_velocity: int = HARMONY_ARPEGGIO_VELOCITY
    """The level of the broken-chord figure."""
    stab_velocity: int = HARMONY_STAB_VELOCITY
    """The level of a stabbed chord."""
    melody_clearance: int = HARMONY_MELODY_CLEARANCE
    """How far under the melody the bed is held."""


DEFAULT_HARMONY_VOICES: HarmonyVoices = HarmonyVoices()
"""The voices layer's defaults: today's texture, figure and levels."""


__all__ = [
    "BROKEN_CHORD_MOODS",
    "DEFAULT_HARMONY_VOICES",
    "HARMONY_ARPEGGIO_STEP_TICKS",
    "HARMONY_ARPEGGIO_VELOCITY",
    "HARMONY_MELODY_CLEARANCE",
    "HARMONY_PAD_VELOCITY",
    "HARMONY_STAB_INSTRUMENTS",
    "HARMONY_STAB_VELOCITY",
    "HarmonyVoices",
]
