"""The words a listener types, read into the requests the engine carries.

`deltas.py` is what the engine can be asked for; this is the other side of that
boundary. One model call reads the sentence when there is a model, and a table of
phrases reads it when there is not — because *"with the LLM disabled, every Tier-1
delta and every UI-control path still works"* is a promise about the product rather
than a hope about a server. A studio whose feedback box goes dead when the model
does is a studio where half the surface is a demo.

**The fallback is not a degraded reader; it is a narrower one, and the difference
is written down rather than discovered.** It reads requests that name their own
value — "calmer", "no drums", "hold the bass" — and it does not read one whose
value is a *change* it would have to measure the piece to compute. So "make it
longer" is read (this piece's length is always known, and the spec has a bound to
refuse it against) and "faster" is not: whether a tempo is this piece's or the
engine's is knowable only by composing, and a fallback that invented a base would
be writing the request rather than reading it. What it cannot read it reports; see
`unread` below, which is the whole reason a partial reading is not a silent one.

**Three outcomes and they are three because the sentences differ.** A phrase can
become a request (a `Delta`), be recognised as a request the engine cannot honour
(a `DeltaRefusal` from `refuse_uncarried` — "swing" is understood and unbuilt),
or name nothing at all (the words are reported as unread). The middle one is the
distinction the vocabulary's `UNCARRIED` table exists for, and collapsing either
of the other two into it would tell the user something false.

**Nothing here composes, and the one question that needs a composition is
therefore not asked here.** `swallowed_tempo` needs the arrangement, so whether a
requested tempo survived the duration search is `revise`'s to answer — the
translator reads words into requests and hands them on, which is also why it can
be tested without a note being written.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final, TypeAlias

from saimc.compose.motif import BassMotion
from saimc.llm.base import ChatClient, ChatRequest, Message, ToolCall, ToolSpec
from saimc.session.conductor import spec_line
from saimc.session.deltas import (
    DELTA_TYPES,
    Delta,
    DeltaRefusal,
    RequestSource,
    ReRoll,
    SetBassMotion,
    SetDrumStyle,
    SetDuration,
    SetHarmonyTexture,
    SetHumanization,
    SetMood,
    SetSectionClose,
    delta_from_dict,
    refuse_uncarried,
)
from saimc.session.tools import request_schema
from saimc.spec import CompositionSpec, Mood

REQUEST_TOOL: Final[str] = "request_delta"
"""The one tool the model is offered, named for what it does."""

SYSTEM_PROMPT: Final[str] = """\
You read one listener's feedback about a piece of music and say which changes it
asks for. You do not write music and you do not decide how anything sounds:
calling the feedback tool is the only way to change anything, and a deterministic
engine applies whatever you name and owns every value's legal range.

- One call per change the feedback asks for. A sentence asking for two things is
  two calls, in the order the listener said them.
- Name only requests the tool's schema offers, and give each only its own
  arguments. A value outside the range the engine accepts comes back refused with
  the range it would have taken, so naming the value you mean is better than
  rounding it yourself.
- Never invent a request for something the vocabulary has no knob for. Make no
  call for it, and say in a sentence what the engine can do instead.
- If the feedback asks for nothing — a question, a remark, praise — answer in
  prose and make no calls. Saying nothing is also an answer, and the listener
  reads what you say.
"""

REQUEST_TOOL_DESCRIPTION: Final[str] = (
    "Name one change the feedback asks for: the knob, and that knob's own "
    "arguments. Call it once per change."
)

_SECONDS_PER_STEP: Final[int] = 30
"""How much length one 'longer' or 'shorter' asks for."""

_READ_BY_KEYWORD: Final[str] = "read by keyword"
_NO_MODEL: Final[str] = f"no language model is configured, so the feedback was {_READ_BY_KEYWORD}"
"""What a translation with no model behind it says about itself.

The note rather than an error, because the reading is real: the words were read,
the requests are the ones the table names, and what is missing is the model's
reading of everything the table does not name. The caller decides what to do with
a partial reading; this module does not pretend it is a failure.
"""


@dataclass(frozen=True)
class Translation:
    """One piece of feedback, and everything that was made of it.

    `deltas` are the requests to apply, in the order they were read. `refusals`
    are requests that were recognised and cannot be honoured, each with the
    reason and the nearest thing to ask for instead. `unread` is what was left:
    the words of the sentence no phrase covered, or the calls the model made that
    could not be turned into a request at all. And `note` is a sentence — the
    model's own words when it answered in prose, or what the keyword reader is.

    The two buckets are kept apart because they are different things to tell a
    user: "I understood you and the engine cannot do it" against "I did not
    understand you". A translation with requests *and* an unread remainder is the
    ordinary case for a sentence asking for two things when only one is readable,
    which is why the remainder is reported rather than dropped — a partial
    reading that says nothing about its missing half is the silent ignore this
    whole vocabulary exists to forbid.
    """

    text: str
    # The vocabulary is `deltas.RequestSource` and not one of this module's own:
    # it names four paths and this module is two of them, so a second alias here
    # would be a second place the values are spelled and the two could disagree
    # about what "model" means.
    source: RequestSource
    deltas: tuple[Delta, ...] = ()
    refusals: tuple[DeltaRefusal, ...] = ()
    unread: tuple[str, ...] = ()
    note: str = ""

    @property
    def read(self) -> bool:
        """Whether anything was read as a change to the piece."""
        return bool(self.deltas)


Reader: TypeAlias = Callable[[CompositionSpec], Delta | DeltaRefusal]


@dataclass(frozen=True)
class Phrase:
    """One thing the keyword reader can read: its words, and what they ask for.

    `read` is a function rather than a value because three of the requests the
    table carries are relative to the piece — "longer" is its own length plus
    thirty seconds, "a different take" is the next seed — and a table that could
    only hold finished values could not carry them. A reader that names its own
    value ignores its argument, which is why one signature serves both.
    """

    phrases: tuple[str, ...]
    read: Reader


def _fixed(request: Delta | DeltaRefusal) -> Reader:
    """A reader for a phrase whose request is the same whatever the piece is."""

    def read(_spec: CompositionSpec) -> Delta | DeltaRefusal:
        return request

    return read


def _unbuilt(request: str) -> Reader:
    """A reader for a phrase the vocabulary understands and cannot honour.

    The refusal is built here, at import, so a phrase naming a request that is
    *not* in `UNCARRIED` fails when the module loads rather than turning into an
    unread phrase at the moment a user says it.
    """
    refusal = refuse_uncarried(request)
    if refusal is None:
        raise ValueError(f"{request!r} is not in UNCARRIED, so no phrase can refuse it by name")
    return _fixed(refusal)


def _longer(spec: CompositionSpec) -> Delta:
    return SetDuration(spec.duration_seconds + _SECONDS_PER_STEP)


def _shorter(spec: CompositionSpec) -> Delta:
    return SetDuration(spec.duration_seconds - _SECONDS_PER_STEP)


def _another_take(spec: CompositionSpec) -> Delta:
    """The next seed, which is the same piece's material drawn again.

    `seed=None` and `seed=0` compose identically — the spec's own docstring says
    so, and it is why the fallback reads a missing seed as zero rather than
    refusing: the request is for *a* reroll, and the first one is seed 1.
    """
    current = spec.seed if spec.seed is not None else 0
    return ReRoll(current + 1)


KEYWORD_TABLE: Final[tuple[Phrase, ...]] = (
    # The three moods, in the words a listener uses for them.
    Phrase(
        ("calm", "calmer", "calming", "relax", "relaxing", "soothe", "soothing"),
        _fixed(SetMood(Mood.CALMING)),
    ),
    Phrase(("sleep", "sleepy", "sleepier", "lullaby", "bedtime"), _fixed(SetMood(Mood.SLEEP))),
    Phrase(
        ("electrifying", "energetic", "excited", "exciting", "upbeat"),
        _fixed(SetMood(Mood.ELECTRIFYING)),
    ),
    # The performance layer. "less expressive" is a phrase of the lighter entry
    # and "expressive" a phrase of the louder one, which is the pair the longest
    # match is read in order to get right.
    Phrase(
        ("expressive", "human", "humanise", "humanize", "looser", "loose"),
        _fixed(SetHumanization(level="expressive")),
    ),
    Phrase(("less expressive", "less human", "subtle"), _fixed(SetHumanization(level="light"))),
    Phrase(
        ("mechanical", "robotic", "stiff", "quantised", "quantized", "metronomic"),
        _fixed(SetHumanization(level="none")),
    ),
    # What the accompaniment states.
    Phrase(
        ("broken chord", "broken chords", "arpeggio", "arpeggios", "arpeggiate", "rolled chords"),
        _fixed(SetHarmonyTexture(broken_chord=True)),
    ),
    Phrase(
        ("block chords", "held chords", "sustained chords", "pads"),
        _fixed(SetHarmonyTexture(broken_chord=False)),
    ),
    # What the bass does, in the vocabulary's own names for it.
    Phrase(
        ("pedal", "pedal point", "drone", "hold the bass", "sustained bass"),
        _fixed(SetBassMotion(BassMotion.PEDAL)),
    ),
    Phrase(("driving bass", "driving", "pushing bass"), _fixed(SetBassMotion(BassMotion.DRIVING))),
    Phrase(("arched bass", "arched"), _fixed(SetBassMotion(BassMotion.ARCHED))),
    Phrase(("sparse bass", "sparse", "fewer bass notes"), _fixed(SetBassMotion(BassMotion.SPARSE))),
    Phrase(
        ("root notes", "root bass", "plain bass", "simple bass"),
        _fixed(SetBassMotion(BassMotion.ROOT)),
    ),
    # The kit, one way: taking it away is absolute, putting it back is not — the
    # mood's own style is the plan's to name, and this reader has only the spec.
    Phrase(
        ("no drums", "without drums", "no percussion", "drop the drums", "take the drums out"),
        _fixed(SetDrumStyle(None)),
    ),
    # How the sections end. "cadence" on its own is not a phrase here: half and
    # full are the two a listener can mean, and the table cannot pick between
    # them, so the words that name one are the words that fire.
    Phrase(
        (
            "half cadence",
            "half cadences",
            "end on the dominant",
            "end each section on the dominant",
            "stop on the dominant",
        ),
        _fixed(SetSectionClose(close="half")),
    ),
    Phrase(
        (
            "full cadence",
            "full cadences",
            "end on the tonic",
            "end each section on the tonic",
            "resolve each section",
        ),
        _fixed(SetSectionClose(close="full")),
    ),
    Phrase(
        (
            "no cadence",
            "no cadences",
            "without cadences",
            "sections run on",
            "let the sections run on",
            "don't close the sections",
        ),
        _fixed(SetSectionClose(close="hold")),
    ),
    # Length. "longer section" is the unbuilt request; "longer" is the piece.
    Phrase(
        ("longer", "make it longer", "a bit longer", "play for longer", "go on for longer"),
        _longer,
    ),
    Phrase(
        ("shorter", "make it shorter", "a bit shorter", "cut it short", "play for less"),
        _shorter,
    ),
    Phrase(
        (
            "again",
            "reroll",
            "re-roll",
            "another one",
            "another take",
            "different take",
            "try again",
            "one more time",
        ),
        _another_take,
    ),
    # And the five requests the design named that plan v1 has no knob for. Each
    # is understood, refused by name, and answered with what to ask for instead.
    Phrase(
        ("up an octave", "down an octave", "an octave higher", "an octave lower", "register"),
        _unbuilt("SetRegister"),
    ),
    Phrase(
        ("harmonic rhythm", "chords change faster", "chords change slower", "change chords faster"),
        _unbuilt("SetHarmonicRhythm"),
    ),
    Phrase(("swing", "swung", "swing feel"), _unbuilt("SetSwing")),
    Phrase(
        ("drums in at", "drum entry", "drums enter", "drums come in", "bring the drums in"),
        _unbuilt("SetDrumEntry"),
    ),
    Phrase(
        (
            "longer section",
            "longer chorus",
            "extend the section",
            "extend the chorus",
            "add a section",
        ),
        _unbuilt("ExtendSection"),
    ),
)
"""Every phrase the fallback reads, and the request each one asks for.

A table of literals rather than a formula, which means **one test case per
entry**: nothing derives the mapping from anything else, so a mis-typed phrase or
a request assigned to the wrong words is invisible to every other test in the
suite. `tests/unit/test_session_translator.py` walks this table and asserts each
phrase of each entry resolves to that entry's request, so an entry that is never
exercised is a failing test rather than a hole.

The order is the tiebreak and not the matching rule — a phrase anywhere in the
sentence fires, and the words are what decide — but it is the order the *table*
is written in that breaks a genuine tie, and the table is written most
fundamental first so a tie resolves to the broader reading.
"""

_WORD: Final[re.Pattern[str]] = re.compile(r"[a-z0-9']+")
"""What a word is, for both sides of a match.

Digits are in because a phrase may contain one day; the apostrophe is in so a
contraction stays one token. Neither is a choice about grammar — matching on
whole words rather than on substrings is: "calm" is not a word of "calmer", and a
substring match would read "calmer" as a request for calm and then report the
"er" as unread.
"""


def _words(text: str) -> tuple[str, ...]:
    return tuple(_WORD.findall(text.lower()))


def _occurrences(words: tuple[str, ...], phrase: tuple[str, ...]) -> list[int]:
    """Every position at which `phrase` occurs in `words`, in order."""
    if not phrase or len(phrase) > len(words):
        return []
    last = len(words) - len(phrase)
    return [start for start in range(last + 1) if words[start : start + len(phrase)] == phrase]


def _readings(words: tuple[str, ...]) -> tuple[list[tuple[int, int]], set[int]]:
    """Which entries the sentence fires, and which of its words they account for.

    Greedy longest-first, then the table's own order, then the earlier position:
    every phrase occurrence is a candidate, candidates are sorted longest first,
    and one that overlaps words an earlier candidate already claimed is dropped.
    That is the whole of the rule, and it is what makes "less expressive" the
    request while "expressive" — a phrase of another entry, and a word inside it
    — does not also fire, and what makes "longer chorus" read as a section rather
    than as thirty seconds.
    """
    candidates: list[tuple[int, int, int]] = []
    for index, entry in enumerate(KEYWORD_TABLE):
        for phrase in entry.phrases:
            tokens = _words(phrase)
            candidates.extend((start, len(tokens), index) for start in _occurrences(words, tokens))
    candidates.sort(key=lambda match: (-match[1], match[2], match[0]))

    claimed: set[int] = set()
    firing: list[tuple[int, int]] = []
    for start, length, index in candidates:
        span = range(start, start + length)
        if claimed.intersection(span):
            continue
        claimed.update(span)
        firing.append((start, index))
    firing.sort()
    return firing, claimed


def _from_keywords(text: str, spec: CompositionSpec, *, note: str) -> Translation:
    """Read `text` with the table, and report what the table did not read."""
    words = _words(text)
    firing, claimed = _readings(words)

    deltas: list[Delta] = []
    refusals: list[DeltaRefusal] = []
    for _start, index in firing:
        reading = KEYWORD_TABLE[index].read(spec)
        if isinstance(reading, DeltaRefusal):
            refusals.append(reading)
        else:
            deltas.append(reading)

    leftover = " ".join(word for position, word in enumerate(words) if position not in claimed)
    return Translation(
        text=text,
        source="keywords",
        deltas=tuple(deltas),
        refusals=tuple(refusals),
        unread=(leftover,) if leftover else (),
        note=note,
    )


def _spelling(call: ToolCall) -> str:
    """A call the reader could not turn into a request, written as it was made."""
    arguments = ", ".join(f"{name}={value!r}" for name, value in call.arguments.items())
    return f"{call.name}({arguments})"


def _unknown(request: str) -> DeltaRefusal:
    """The refusal for a request this engine has no knob for at all.

    The second builder of `unknown_knob` beside `refuse_uncarried` — a model can
    name a knob that does not exist, which is the same situation as a user asking
    for one the design never built — and the sentence is short because there is
    nothing to offer instead: no nearby request would answer it.
    """
    return DeltaRefusal(
        request=request,
        reason="unknown_knob",
        message=(
            f"there is no request called {request!r}, so nothing was changed for it. This "
            f"engine carries {len(DELTA_TYPES)} requests, and the schema you were given names "
            "every one of them."
        ),
    )


def _from_calls(
    calls: Sequence[ToolCall],
) -> tuple[list[Delta], list[DeltaRefusal], list[str]]:
    """Read a model's calls, keeping every one of them visible somewhere.

    Three places a call can land and none of them is a drop: a request to apply,
    a refusal carrying the vocabulary's own reason, or the spelling of a call
    that was not a request at all — an argument the knob's constructor refused —
    which is reported as unread rather than refused, because there is no request
    to refuse. A model that invents a knob gets the refusal, since a name this
    engine does not carry is exactly what `unknown_knob` means.
    """
    deltas: list[Delta] = []
    refusals: list[DeltaRefusal] = []
    unread: list[str] = []
    for call in calls:
        unbuilt = refuse_uncarried(call.name)
        if unbuilt is not None:
            refusals.append(unbuilt)
        elif call.name not in DELTA_TYPES:
            refusals.append(_unknown(call.name))
        else:
            try:
                deltas.append(delta_from_dict({**call.arguments, "knob": call.name}))
            except (TypeError, ValueError):
                unread.append(_spelling(call))
    return deltas, refusals, unread


def _request(text: str, spec: CompositionSpec, *, request_id: str) -> ChatRequest:
    """The one model call: the sentence, the piece, and the vocabulary.

    The piece is described with the conductor's own `spec_line` rather than
    rendered again here — a model reading feedback has to know what the piece is
    to choose a value for it, and a second rendering of the spec is a second
    place a field can be forgotten.
    """
    return ChatRequest(
        messages=(
            Message(role="system", content=SYSTEM_PROMPT),
            Message(
                role="user",
                content=f"feedback: {text}\nthe piece as it stands: {spec_line(spec)}",
            ),
        ),
        tools=(
            ToolSpec(
                name=REQUEST_TOOL,
                description=REQUEST_TOOL_DESCRIPTION,
                parameters=request_schema(),
            ),
        ),
        request_id=request_id,
    )


async def translate(
    text: str,
    *,
    client: ChatClient | None,
    spec: CompositionSpec,
    request_id: str = "",
) -> Translation:
    """Read one piece of feedback into the requests it asks for.

    A model is consulted when there is one, and the table reads the words when
    there is not — or when the model's own call failed, because "the model is
    down" is not a reason for the feedback box to stop working. A model that
    answers in prose is *not* second-guessed by the table: its sentence is the
    note, and the words it declined to read are the ones it was asked about.

    `spec` is required rather than optional, and that is a statement about the
    product: this reads feedback *about a piece*, so there is always one. The
    relative readers need it for their base, every request is folded against it
    afterwards, and a caller holding no spec has nothing to be given a
    translation of — which is the state that would need a branch, and the
    branch would be about a situation that does not exist.
    """
    if client is None:
        return _from_keywords(text, spec, note=_NO_MODEL)

    result = await client.chat(_request(text, spec, request_id=request_id))
    if result.error is not None:
        return _from_keywords(
            text,
            spec,
            note=f"the model call failed ({result.error.error_code}): {_READ_BY_KEYWORD}",
        )
    if not result.tool_calls:
        if not result.content.strip():
            return _from_keywords(
                text, spec, note=f"the model answered nothing: {_READ_BY_KEYWORD}"
            )
        return Translation(text=text, source="model", note=result.content.strip())

    deltas, refusals, unread = _from_calls(result.tool_calls)
    return Translation(
        text=text,
        source="model",
        deltas=tuple(deltas),
        refusals=tuple(refusals),
        unread=tuple(unread),
        note=result.content.strip(),
    )


__all__ = [
    "KEYWORD_TABLE",
    "REQUEST_TOOL",
    "REQUEST_TOOL_DESCRIPTION",
    "SYSTEM_PROMPT",
    "Phrase",
    "Translation",
    "translate",
]
