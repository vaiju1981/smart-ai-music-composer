"""A second opinion about the music, and whether the first one was taste.

`quality.py` measures thirteen things about a piece. Those same thirteen
numbers are then what the engine is tuned toward, what `session/arbiter.py`
ranks candidates by, what the critics report, what the repair table moves, and
what both release gates read. One opinion with five jobs, and nothing in the
project positioned to disagree with it — so "the music is good" has been
answered, every time, by the instrument that defines what good means.

This module is the disagreement. A model that has never seen a threshold is
shown two pieces and asked which is the better music; the arbiter's order over
the same two is already known; and the number worth having is how often the two
agree. That number is what turns `arbiter.py`'s own admission — *these metrics
are a proxy for taste, not a definition of it* — from a disclaimer into a
measurement, and it is the evidence the plan's Open Question 1 asks for before
the scorecard is allowed to overrule anybody.

Four decisions hold it together.

**Pairwise, not a mark out of ten.** A model asked to score a piece 0-100 is
being asked to hold a scale steady across a corpus, which it cannot do and
which nothing here could check. Asked which of two pieces is better it is being
asked the question the arbiter also answers, so the two answers are directly
comparable and the comparison needs no calibration at all.

**Blind, and structurally so.** `describe_piece` takes a `NotationScore` and
nothing else. It *cannot* leak a measurement into the prompt, because it is
never handed one — which is a stronger claim than a prompt that promises not to
mention them, and `test_judge.py` holds it.

**Every pair twice, with the sides swapped.** A model has a position bias, and a
judge whose answer depends on which piece came first is not evidence of
anything. A pair whose two passes disagree is counted as *inconsistent* and
excluded from the agreement rate rather than silently resolved — and the
consistency rate is reported beside it, because a judge that is 55% consistent
has told you about itself, not about the music.

**No bar yet, deliberately.** There is no `MIN_AGREEMENT` here and no release
gate reading one. This project's rule is that no guard is written before it can
fail, and nothing has run this against a live model: a threshold chosen now
would be a number invented to match an unmeasured quantity. The floor lands
with the first recorded sweep, in the commit that records it
(`docs/open-items.md` C5).

Two things this cannot tell you, listed so the number is not over-read. It
cannot say the arbiter is *wrong* when they disagree — the judge is a model with
opinions, not a musician of record — only that the order is not the one a
listener would give, which is the thing worth knowing. And it reads notation
rather than audio: what is judged is the writing, not the performance, so
nothing about the render, the soundfont or the mix is in evidence.
"""

from __future__ import annotations

import json
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any, Final, Literal

from saimc.compose.percussion import (
    DRUM_CLOSED_HIHAT,
    DRUM_CRASH,
    DRUM_HAND_CLAP,
    DRUM_HIGH_TOM,
    DRUM_KICK,
    DRUM_LOW_TOM,
    DRUM_MID_TOM,
    DRUM_OPEN_HIHAT,
    DRUM_PEDAL_HIHAT,
    DRUM_RIDE,
    DRUM_SIDE_STICK,
    DRUM_SNARE,
)
from saimc.compose.score import (
    PPQ,
    VOICE_BASS,
    VOICE_HARMONY,
    VOICE_MELODY,
    VOICE_PERCUSSION,
    NotationScore,
    NoteEvent,
)
from saimc.llm.base import ChatClient, ChatRequest, Message
from saimc.quality import PieceQuality
from saimc.session.arbiter import musical_order

Side = Literal["first", "second", "neither"]
"""What the judge was asked to answer, in the order it was shown."""

Preference = Literal["left", "right", "neither"]
"""What an answer means about the pair itself, once the sides are unswapped."""

DESCRIBED_BARS: Final[int] = 16
"""How many bars of a piece the judge is shown.

A five-minute piece is a hundred-odd bars of one repeating form, and sending
all of them would spend most of the prompt on the repeats. Sixteen covers the
form and its first return for every form size the engine builds. The
description *says* how many bars it left out, so the model is not being
told it has the whole piece.
"""

NOTE_NAMES: Final[tuple[str, ...]] = (
    "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B",
)  # fmt: skip

_DURATIONS: Final[tuple[tuple[float, str], ...]] = (
    (4.0, "whole"), (3.0, "dotted-half"), (2.0, "half"), (1.5, "dotted-quarter"),
    (1.0, "quarter"), (0.75, "dotted-eighth"), (0.5, "eighth"), (0.25, "sixteenth"),
)  # fmt: skip

VOICE_NAMES: Final[dict[int, str]] = {
    VOICE_MELODY: "Melody",
    VOICE_HARMONY: "Harmony",
    VOICE_BASS: "Bass",
    VOICE_PERCUSSION: "Drums",
}
"""Voice id -> the word a reader knows it by, keyed off the engine's constants.

Written out of `score.py`'s names rather than as the four integers, because
four integers here would be a second copy of an ordering that lives there —
and the first draft of this file had them in the wrong order, which labelled
the bass as the melody and read as music in both directions.
"""

VOICE_ORDER: Final[tuple[int, ...]] = (
    VOICE_MELODY,
    VOICE_HARMONY,
    VOICE_BASS,
    VOICE_PERCUSSION,
)
"""Top down, the way a score is read, not the order the ids happen to be in."""

DRUM_NAMES: Final[dict[int, str]] = {
    DRUM_KICK: "kick",
    DRUM_SIDE_STICK: "rim",
    DRUM_SNARE: "snare",
    DRUM_HAND_CLAP: "clap",
    DRUM_CLOSED_HIHAT: "hat",
    DRUM_PEDAL_HIHAT: "hat-pedal",
    DRUM_OPEN_HIHAT: "hat-open",
    DRUM_CRASH: "crash",
    DRUM_LOW_TOM: "low tom",
    DRUM_MID_TOM: "mid tom",
    DRUM_HIGH_TOM: "high tom",
    DRUM_RIDE: "ride",
}
"""On channel 10 the pitch *is* the drum, so a kit bar written as note names
would read as a tune nobody played. Named instead, and a piece the table does
not know is reported by its number rather than as a pitch."""

SYSTEM_PROMPT: Final[str] = """\
You are a listener with a trained ear and no stake in how either piece was made.

You will be shown two short pieces written out as notation. Read them as music:
the shape of the line, whether it goes anywhere and comes back, whether the
accompaniment supports the tune or fights it, whether the rhythm has life in it,
whether the harmony moves.

Say which is the better music. "Better" is your judgement and nothing here will
argue with it; there is no rubric and no scoring guide. If they are genuinely
of a piece, say so rather than picking one to be decisive.

Answer with JSON and nothing else:

  {"prefers": "first" | "second" | "neither", "reason": "<one or two sentences>"}

The reason is for a human reading your answer later. Name what you heard —
a phrase, a bar, a line that did or did not go somewhere — not a general
quality word."""


def note_name(pitch_midi: int) -> str:
    """`C4` for middle C, sharps throughout.

    Spelled with sharps rather than from the key signature: the engine's own
    key handling is interval tables over pitch classes, so a flat spelling here
    would be a second opinion about enharmonics that nothing else in the
    project holds.
    """
    return f"{NOTE_NAMES[pitch_midi % 12]}{pitch_midi // 12 - 1}"


def duration_name(duration_ticks: int, ppq: int = PPQ) -> str:
    """The nearest named value, or the tick count when nothing is near."""
    beats = duration_ticks / ppq
    for value, name in _DURATIONS:
        if abs(beats - value) < 0.02:
            return name
    return f"{beats:.2f}-beat"


def describe_piece(score: NotationScore, *, bars: int = DESCRIBED_BARS) -> str:
    """One piece, written out for a reader who has not been told anything else.

    Takes a `NotationScore` and takes nothing else — no `PieceQuality`, no
    findings, no plan. That is the blindness, and it is a property of the
    signature rather than a promise in the prompt.
    """
    lines = [
        f"Key: {score.key.root} {score.key.mode}",
        f"Time signature: {score.time_signature}",
        f"Tempo: {score.tempo.bpm:.0f} bpm",
        f"Length: {len(score.measures)} bars",
    ]
    shown = score.measures[:bars]
    if len(score.measures) > len(shown):
        lines.append(
            f"Written out below: bars 1-{len(shown)} of {len(score.measures)}. "
            "The rest repeats the same form."
        )
    lines.append("")
    for index, measure in enumerate(shown, start=1):
        in_bar = [
            note for note in score.notes if measure.start_tick <= note.tick < measure.end_tick
        ]
        lines.append(f"Bar {index}")
        present = {note.voice_id for note in in_bar}
        for voice_id in (*VOICE_ORDER, *sorted(present - set(VOICE_ORDER))):
            if voice_id not in present:
                continue
            voice = [note for note in in_bar if note.voice_id == voice_id]
            name = VOICE_NAMES.get(voice_id, f"Voice {voice_id}")
            written = (
                _kit_line(voice, score.ppq)
                if voice_id == VOICE_PERCUSSION
                else _voice_line(voice, score.ppq)
            )
            lines.append(f"  {name}: {written}")
    return "\n".join(lines)


def _voice_line(notes: Sequence[NoteEvent], ppq: int) -> str:
    """One voice's bar: simultaneous notes as a chord, in onset order."""
    by_tick: dict[int, list[NoteEvent]] = {}
    for note in sorted(notes, key=lambda n: (n.tick, n.pitch_midi)):
        by_tick.setdefault(note.tick, []).append(note)
    parts = []
    for tick in sorted(by_tick):
        together = by_tick[tick]
        pitches = "+".join(note_name(note.pitch_midi) for note in together)
        parts.append(f"{pitches} ({duration_name(together[0].duration_ticks, ppq)})")
    return " ".join(parts) if parts else "silent"


def _kit_line(notes: Sequence[NoteEvent], ppq: int) -> str:
    """One kit bar as beats and pieces: `1 kick+hat, 2.5 hat, 3 snare`."""
    by_tick: dict[int, list[NoteEvent]] = {}
    for note in sorted(notes, key=lambda n: (n.tick, n.pitch_midi)):
        by_tick.setdefault(note.tick, []).append(note)
    if not by_tick:
        return "silent"
    first = min(by_tick)
    parts = []
    for tick in sorted(by_tick):
        beat = (tick - first) / ppq + 1
        pieces = "+".join(
            DRUM_NAMES.get(note.pitch_midi, f"drum {note.pitch_midi}") for note in by_tick[tick]
        )
        parts.append(f"{beat:g} {pieces}")
    return ", ".join(parts)


@dataclass(frozen=True)
class JudgedPiece:
    """One candidate: what the judge reads, and what the scorecard measured.

    The two halves never meet. `notation` is what `describe_piece` renders;
    `quality` and `lint_passed` are only ever read by `scorecard_prefers`,
    which runs after the judge has answered.
    """

    label: str
    notation: NotationScore
    quality: PieceQuality
    lint_passed: bool = True


@dataclass(frozen=True)
class Pairing:
    """Two pieces the same brief could have produced, and the order to beat."""

    pair_id: str
    left: JudgedPiece
    right: JudgedPiece

    def scorecard_prefers(self) -> Preference:
        """Which side the arbiter's musical order puts first, if either.

        `musical_order` and not `arbiter_order`: the last two elements of the
        full key are a plan hash and a seed, which settle a tie by arithmetic
        that is deliberately arbitrary. Asking a listener to agree with a hash
        would be asking them to lose, so a pair the *music* cannot separate is
        reported as `neither` and counted as incomparable.
        """
        left = musical_order(self.left.quality, lint_passed=self.left.lint_passed)
        right = musical_order(self.right.quality, lint_passed=self.right.lint_passed)
        if left == right:
            return "neither"
        return "left" if left < right else "right"


@dataclass(frozen=True)
class JudgeVerdict:
    """One answer, exactly as the judge gave it."""

    prefers: Side
    reason: str
    error: str | None = None

    @property
    def usable(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class PairJudgement:
    """One pair, judged both ways round, and what that came to.

    `prefers` is `None` when the two passes disagreed. That is not a failure to
    record — it is the judge telling you its answer moved with the order, and
    the honest thing to do with it is to keep it out of the agreement rate and
    count it in the consistency rate instead.
    """

    pair_id: str
    forward: JudgeVerdict
    reverse: JudgeVerdict
    scorecard: Preference

    @property
    def consistent(self) -> bool:
        if not (self.forward.usable and self.reverse.usable):
            return False
        return self.forward.prefers == _flip(self.reverse.prefers)

    @property
    def prefers(self) -> Preference | None:
        if not self.consistent:
            return None
        return _as_preference(self.forward.prefers)

    @property
    def comparable(self) -> bool:
        """Both sides have a strict preference, so agreement means something."""
        return self.prefers not in (None, "neither") and self.scorecard != "neither"

    @property
    def agrees(self) -> bool | None:
        return self.prefers == self.scorecard if self.comparable else None


def _flip(side: Side) -> Side:
    if side == "first":
        return "second"
    return "first" if side == "second" else "neither"


def _as_preference(side: Side) -> Preference:
    if side == "first":
        return "left"
    return "right" if side == "second" else "neither"


@dataclass(frozen=True)
class JudgeReport:
    """What a sweep came to, in the shape `BenchmarkReport` established.

    Deliberately without `passed`. The other two reports in this project carry
    a verdict because their thresholds were argued for before the code was
    written; this one has no threshold, because nothing has measured what a
    reasonable agreement rate looks like and a bar invented before the first
    run would be a number chosen to be cleared.
    """

    label: str
    model: str
    pairs: int
    consistent_pairs: int
    comparable_pairs: int
    agreeing_pairs: int
    failed_calls: int
    judgements: tuple[PairJudgement, ...] = field(repr=False, default=())

    @property
    def consistency_rate(self) -> float:
        """Share of pairs the judge answered the same way both ways round."""
        return self.consistent_pairs / self.pairs if self.pairs else 0.0

    @property
    def agreement_rate(self) -> float:
        """Of the pairs both can separate, the share they separate alike.

        Chance is 0.5 here, not 0. A rate near half says the scorecard's order
        carries no information a listener recognises; a rate near one says the
        thirteen metrics are standing in for taste about as well as a proxy
        can. Neither reading is available from the metrics alone, which is the
        whole reason this number exists.
        """
        return self.agreeing_pairs / self.comparable_pairs if self.comparable_pairs else 0.0

    def disagreements(self) -> tuple[PairJudgement, ...]:
        """The comparable pairs where the two orders parted company.

        The material worth reading by hand: each one is a piece the arbiter
        would have ranked above another that a listener preferred, with the
        listener's own sentence about why.
        """
        return tuple(j for j in self.judgements if j.agrees is False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "model": self.model,
            "pairs": self.pairs,
            "consistent_pairs": self.consistent_pairs,
            "consistency_rate": round(self.consistency_rate, 4),
            "comparable_pairs": self.comparable_pairs,
            "agreeing_pairs": self.agreeing_pairs,
            "agreement_rate": round(self.agreement_rate, 4),
            "failed_calls": self.failed_calls,
            "disagreements": [
                {
                    "pair_id": j.pair_id,
                    "listener_preferred": j.prefers,
                    "scorecard_preferred": j.scorecard,
                    "reason": j.forward.reason,
                }
                for j in self.disagreements()
            ],
        }


def _request(first: str, second: str) -> ChatRequest:
    return ChatRequest(
        messages=(
            Message(role="system", content=SYSTEM_PROMPT),
            Message(
                role="user",
                content=f"PIECE ONE\n\n{first}\n\n\nPIECE TWO\n\n{second}",
            ),
        ),
    )


def read_verdict(content: str) -> JudgeVerdict:
    """The model's reply, read as an answer or as a named failure.

    A reply that is not the JSON it was asked for is recorded as an error
    rather than guessed at: a judge that has to be interpreted is one whose
    agreement rate is partly the interpreter's.
    """
    text = content.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1].rsplit("```", 1)[0]
    try:
        payload = json.loads(text)
    except ValueError:
        return JudgeVerdict("neither", "", error="the reply was not JSON")
    prefers = payload.get("prefers")
    if prefers not in ("first", "second", "neither"):
        return JudgeVerdict("neither", "", error=f"unreadable preference {prefers!r}")
    return JudgeVerdict(prefers, str(payload.get("reason", "")))


async def judge_pair(client: ChatClient, pairing: Pairing) -> PairJudgement:
    """Ask about one pair twice, with the sides swapped the second time."""
    left = describe_piece(pairing.left.notation)
    right = describe_piece(pairing.right.notation)
    forward = await _ask(client, left, right)
    reverse = await _ask(client, right, left)
    return PairJudgement(
        pair_id=pairing.pair_id,
        forward=forward,
        reverse=reverse,
        scorecard=pairing.scorecard_prefers(),
    )


async def _ask(client: ChatClient, first: str, second: str) -> JudgeVerdict:
    result = await client.chat(_request(first, second))
    if result.error is not None:
        return JudgeVerdict("neither", "", error=result.error.error_code)
    return read_verdict(result.content or "")


async def judge_corpus(
    client: ChatClient,
    pairings: Iterable[Pairing],
    *,
    label: str,
    model: str,
) -> JudgeReport:
    """Judge every pairing and total up what the two orders came to."""
    judgements = []
    for pairing in pairings:
        judgements.append(await judge_pair(client, pairing))
    return score_judgements(judgements, label=label, model=model)


def score_judgements(judgements: Sequence[PairJudgement], *, label: str, model: str) -> JudgeReport:
    """Total a set of judgements. Separate from the asking, so it is testable."""
    return JudgeReport(
        label=label,
        model=model,
        pairs=len(judgements),
        consistent_pairs=sum(1 for j in judgements if j.consistent),
        comparable_pairs=sum(1 for j in judgements if j.comparable),
        agreeing_pairs=sum(1 for j in judgements if j.agrees is True),
        failed_calls=sum(1 for j in judgements for v in (j.forward, j.reverse) if not v.usable),
        judgements=tuple(judgements),
    )


def pairs_from(pieces: Sequence[JudgedPiece], *, seed: int = 0) -> list[Pairing]:
    """Every piece paired with one other, once, in a shuffled order.

    A round-robin would be quadratic in model calls for a linear gain in
    evidence; one shuffled pass gives every piece a comparison and costs
    `len(pieces) // 2` pairs. The shuffle is seeded, so a sweep is as
    reproducible as everything else here.
    """
    order = list(pieces)
    random.Random(seed).shuffle(order)
    return [
        Pairing(pair_id=f"{a.label}|{b.label}", left=a, right=b)
        for a, b in zip(order[::2], order[1::2], strict=False)
    ]


__all__ = [
    "DESCRIBED_BARS",
    "DRUM_NAMES",
    "SYSTEM_PROMPT",
    "VOICE_NAMES",
    "VOICE_ORDER",
    "JudgeReport",
    "JudgeVerdict",
    "JudgedPiece",
    "PairJudgement",
    "Pairing",
    "Preference",
    "Side",
    "describe_piece",
    "duration_name",
    "judge_corpus",
    "judge_pair",
    "note_name",
    "pairs_from",
    "read_verdict",
    "score_judgements",
]
