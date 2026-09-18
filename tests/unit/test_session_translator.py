"""The words a listener types, read into the requests the engine carries.

Four properties, and the tests are grouped to witness them:

- **The table is the fallback, and it is complete.** Every phrase of every
  entry is read, and read *as that entry's request* — a literal table has one
  coverage hole per entry, so the loop that walks it is the test that catches a
  mis-typed phrase or a request matched to the wrong words.
- **The longest match wins.** "less expressive" is lighter playing and
  "expressive" is not; "longer chorus" is a section and "longer" is the piece.
  Both are decided by one ordering rule, and both are asserted.
- **What was not read is reported.** A sentence asking for two things, one of
  which the table does not carry, comes back with a request *and* the words that
  named nothing. A partial reading that says nothing about its missing half is
  the silent ignore the vocabulary forbids.
- **The model is one call, and its failures degrade rather than break.** A
  model's calls become requests, its prose becomes the note, its invented knobs
  are refused, and an unreachable model hands the sentence to the table with a
  note naming why. That last one is the plan's "feedback degrades" bullet.

The model is scripted; nothing here composes, which is what keeps this file
fast — the translator reads words, and `deltas.py` is where folding is tested.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from saimc.compose.motif import BassMotion
from saimc.llm.base import ChatRequest, ChatResult, LLMError, ToolCall
from saimc.session import translator
from saimc.session.deltas import (
    DELTA_TYPES,
    UNCARRIED,
    Delta,
    DeltaRefusal,
    ReRoll,
    SetBassMotion,
    SetDrumEntry,
    SetDrumStyle,
    SetDuration,
    SetHarmonicRhythm,
    SetHarmonyTexture,
    SetHumanization,
    SetMood,
    SetSectionClose,
    apply_deltas,
)
from saimc.session.translator import KEYWORD_TABLE, REQUEST_TOOL, Phrase, translate
from saimc.spec import CompositionSpec, Mood

_SPEC = CompositionSpec(mood=Mood.CALMING, duration_seconds=180, seed=5)
"""A piece with everything a relative reader needs: a length and a seed."""


class _Model:
    """A scripted model that records the request it was sent."""

    def __init__(self, *replies: ChatResult) -> None:
        self.replies = list(replies)
        self.requests: list[ChatRequest] = []

    async def chat(self, request: ChatRequest) -> ChatResult:
        self.requests.append(request)
        if self.replies:
            return self.replies.pop(0)
        return ChatResult(content="Nothing to change.")


def _read(text: str, *, client: Any = None, spec: CompositionSpec = _SPEC) -> Any:
    return asyncio.run(translate(text, client=client, spec=spec))


def _calls(*calls: tuple[str, dict[str, Any]], content: str = "") -> ChatResult:
    return ChatResult(
        content=content,
        tool_calls=tuple(ToolCall(name=name, arguments=args) for name, args in calls),
    )


class TestTheKeywordTableReadsWhatItClaims:
    """One case per entry, which is what a table of literals costs."""

    @pytest.mark.parametrize(
        ("phrase", "expected"),
        [
            ("calm", SetMood(Mood.CALMING)),
            ("relaxing", SetMood(Mood.CALMING)),
            ("sleepier", SetMood(Mood.SLEEP)),
            ("upbeat", SetMood(Mood.ELECTRIFYING)),
            ("expressive", SetHumanization(level="expressive")),
            ("subtle", SetHumanization(level="light")),
            ("robotic", SetHumanization(level="none")),
            ("arpeggios", SetHarmonyTexture(broken_chord=True)),
            ("pads", SetHarmonyTexture(broken_chord=False)),
            ("drone", SetBassMotion(BassMotion.PEDAL)),
            ("pushing bass", SetBassMotion(BassMotion.DRIVING)),
            ("arched", SetBassMotion(BassMotion.ARCHED)),
            ("sparse", SetBassMotion(BassMotion.SPARSE)),
            ("root notes", SetBassMotion(BassMotion.ROOT)),
            ("without drums", SetDrumStyle(None)),
            ("half cadence", SetSectionClose(close="half")),
            ("full cadence", SetSectionClose(close="full")),
            ("no cadence", SetSectionClose(close="hold")),
            ("make it longer", SetDuration(210)),
            ("cut it short", SetDuration(150)),
            ("another take", ReRoll(6)),
            ("chords change faster", SetHarmonicRhythm((2, 1, 1))),
            ("slower harmonic rhythm", SetHarmonicRhythm((4,))),
        ],
    )
    def test_a_phrase_becomes_the_request_it_names(self, phrase: str, expected: Delta) -> None:
        """The intent, asserted literally.

        The loop below proves each entry's phrases *reach* it; this proves the
        entry says what it was meant to say. Both are needed: a table swapped
        wholesale between two entries passes the loop and fails here.
        """
        translation = _read(phrase)
        assert translation.deltas == (expected,)
        assert translation.refusals == ()
        assert translation.unread == ()

    @pytest.mark.parametrize(("index", "entry"), list(enumerate(KEYWORD_TABLE)))
    def test_every_phrase_of_every_entry_fires_that_entry(self, index: int, entry: Phrase) -> None:
        """The ratchet on a literal table: no entry is a hole.

        Each phrase is read *alone*, so the assertion is that it fires exactly
        one entry and that entry is its own. A mis-typed phrase matches nothing
        and comes back unread; a phrase shared with another entry fires the
        wrong one. Neither is visible to any other test in the suite.
        """
        for phrase in entry.phrases:
            translation = _read(phrase)
            reading = entry.read(_SPEC)
            if isinstance(reading, DeltaRefusal):
                assert translation.refusals == (reading,), phrase
                assert translation.deltas == ()
            else:
                assert translation.deltas == (reading,), phrase
                assert translation.refusals == ()
            assert translation.unread == (), phrase

    def test_the_table_is_the_order_it_is_written_as(self) -> None:
        """The tiebreak is the table's order, so the order is a declared edit.

        Not evidence that the order is right — nothing can be, it is a
        judgement — but the ratchet that makes re-ordering a judgement rather
        than a side effect of moving an entry. The same shape as the arbiter's
        tier pin.
        """
        assert [entry.phrases[0] for entry in KEYWORD_TABLE] == [
            "calm",
            "sleep",
            "electrifying",
            "expressive",
            "less expressive",
            "mechanical",
            "broken chord",
            "block chords",
            "pedal",
            "driving bass",
            "arched bass",
            "sparse bass",
            "root notes",
            "no drums",
            "half cadence",
            "full cadence",
            "no cadence",
            "longer",
            "shorter",
            "again",
            "up an octave",
            "harmonic rhythm",
            "chords change slower",
            "chords change faster",
            "swing",
            "drums in at",
            "longer section",
        ]

    def test_every_unbuilt_request_has_a_phrase(self) -> None:
        """`UNCARRIED` is the design's list, and the table has to cover it.

        The premise rather than the assertion, and the reason it is here: an
        unbuilt request added to the vocabulary would otherwise be readable by
        the model and unread by the fallback, which is a difference between the
        two paths that nothing else would notice. The directionless phrase is
        in the set beside them because it is the *other* way a phrase refuses —
        a knob this engine carries whose words name no direction — and the two
        are read the same way here for the same reason: both are sentences the
        fallback owes a listener, and neither may be an unread word.
        """
        readable = {
            phrase
            for entry in KEYWORD_TABLE
            for phrase in entry.phrases
            if isinstance(entry.read(_SPEC), DeltaRefusal)
        }
        assert readable == {
            "up an octave",
            "down an octave",
            "an octave higher",
            "an octave lower",
            "register",
            "harmonic rhythm",
            "swing",
            "swung",
            "swing feel",
            "drums in at",
            "drum entry",
            "drums enter",
            "drums come in",
            "bring the drums in",
            "longer section",
            "longer chorus",
            "extend the section",
            "extend the chorus",
            "add a section",
        }
        assert {entry.request for entry in UNCARRIED} == {
            "SetRegister",
            "SetSwing",
            "ExtendSection",
        }

    def test_a_phrase_named_for_a_request_the_vocabulary_lacks_is_refused_at_import(self) -> None:
        """The guard on `_unbuilt`, fired rather than trusted.

        A phrase for a request that is *not* unbuilt would otherwise become a
        sentence the fallback can never read — an entry that looks like a
        refusal and is silently an unread word.
        """
        with pytest.raises(ValueError, match="not in UNCARRIED"):
            translator._unbuilt("SetTempo")


class TestAnUnbuiltRequestIsRefusedByIdent:
    """The distinction `UNCARRIED` exists for: understood, and not built."""

    def test_swing_refuses_and_says_what_to_ask_for_instead(self) -> None:
        translation = _read("can you add some swing please")
        assert translation.deltas == ()
        assert [refusal.request for refusal in translation.refusals] == ["SetSwing"]
        refusal = translation.refusals[0]
        assert refusal.reason == "unknown_knob"
        assert "triplet grid" in refusal.message
        assert refusal.nearest is not None
        assert "SetDrumStyle" in refusal.nearest
        assert translation.unread == ("can you add some please",)

    @pytest.mark.parametrize(
        ("text", "knob"),
        [
            ("up an octave", "SetRegister"),
            ("swing feel", "SetSwing"),
            ("extend the chorus", "ExtendSection"),
        ],
    )
    def test_each_unbuilt_request_is_refused_by_its_own_name(self, text: str, knob: str) -> None:
        """The name *and* the vocabulary's own sentence.

        The sentence is what says "understood and unbuilt" rather than "a word I
        do not know", and it is the table's `why` rather than anything written
        here — so an unbuilt request refused through the unknown-knob path, which
        names the request and explains nothing, fails this.
        """
        entry = next(each for each in UNCARRIED if each.request == knob)
        translation = _read(text)
        assert [refusal.request for refusal in translation.refusals] == [knob]
        assert entry.why in translation.refusals[0].message
        assert translation.refusals[0].nearest == entry.instead


class TestADirectionTheWordsDidNotName:
    """The knob this engine *has*, refused for the words rather than the knob.

    The distinction is the whole reason `direction_unnamed` exists: "swing" is a
    request no knob carries, and "harmonic rhythm" is a request the plan carries
    whose words do not say which way to move it. Answering the second with the
    first's sentence would tell a listener the engine cannot do something it can.

    `SetDrumEntry` joined this class in F5a, when the plan grew
    `percussion_entry_bar` and the phrase stopped being an unbuilt request. Its
    missing piece is the *bar* rather than a direction, which is the same
    situation: the knob is there, the words name it, and the value is not in
    them.
    """

    @pytest.mark.parametrize(
        ("text", "knob", "missing"),
        [
            ("harmonic rhythm", "SetHarmonicRhythm", "faster"),
            ("bring the drums in", "SetDrumEntry", "drums in at bar 8"),
        ],
    )
    def test_a_bare_knob_is_refused_for_the_direction_and_not_the_knob(
        self, text: str, knob: str, missing: str
    ) -> None:
        translation = _read(text)
        assert translation.deltas == ()
        assert [refusal.request for refusal in translation.refusals] == [knob]
        refusal = translation.refusals[0]
        assert refusal.reason == "direction_unnamed"
        assert missing in refusal.message

    def test_a_bare_knob_leaves_only_what_it_did_not_read(self) -> None:
        translation = _read("more harmonic rhythm please")
        (refusal,) = translation.refusals
        assert "faster" in refusal.message
        assert "slower" in refusal.message
        assert translation.unread == ("more please",)

    def test_the_drum_entry_knob_is_one_the_engine_takes(self) -> None:
        """The premise, for the same reason the harmonic rhythm one has it:
        the phrase is refused for the words, so the request itself has to be
        honoured when the words do carry the value."""
        assert "SetDrumEntry" in DELTA_TYPES
        application = apply_deltas(_SPEC, [SetDrumEntry(bar=6)])
        assert application.ok, application.refused
        assert application.plan.percussion_entry_bar == 6

    def test_it_is_not_the_unbuilt_knob_and_says_nothing_about_not_carrying(self) -> None:
        """The false sentence, asserted absent.

        `refuse_uncarried` answers with "is not a knob this engine carries yet",
        which is the one thing that is untrue here — so the message is asserted
        against it rather than merely asserted to be non-empty.
        """
        (refusal,) = _read("harmonic rhythm").refusals
        assert "not a knob this engine carries yet" not in refusal.message
        assert refusal.nearest is None, "the message names both directions already"

    def test_the_knob_is_one_the_engine_takes_and_the_refusal_is_not_about_that(self) -> None:
        """The premise: the same request with a direction is honoured.

        Without this the refusal above would pass for a knob that is genuinely
        unbuilt. `SetHarmonicRhythm` is in `DELTA_TYPES` and folds into the plan,
        which is what makes the refusal a reading of the *words*.
        """
        assert "SetHarmonicRhythm" in DELTA_TYPES
        translation = _read("harmonic rhythm slower")
        assert translation.deltas == (SetHarmonicRhythm((4,)),)
        assert translation.refusals == ()

    @pytest.mark.parametrize(
        ("text", "pattern"),
        [
            ("chords change faster", (2, 1, 1)),
            ("change chords faster", (2, 1, 1)),
            ("faster harmonic rhythm", (2, 1, 1)),
            ("harmonic rhythm faster", (2, 1, 1)),
            ("chords change slower", (4,)),
            ("change chords slower", (4,)),
            ("slower harmonic rhythm", (4,)),
            ("harmonic rhythm slower", (4,)),
        ],
    )
    def test_a_direction_reads_whether_it_comes_before_or_after_the_knob(
        self, text: str, pattern: tuple[int, ...]
    ) -> None:
        """The words said out of order still read, and the longest match is why.

        "harmonic rhythm faster" contains the directionless entry's two words as
        a subspan; a three-word phrase beating a two-word one is the ordering
        rule, so the direction is read rather than reported as a trailing unread
        word — which is what would happen if the shorter phrase were tried first.
        """
        translation = _read(text)
        assert translation.deltas == (SetHarmonicRhythm(pattern),)
        assert translation.refusals == ()
        assert translation.unread == ()

    def test_a_uniform_fast_pattern_is_not_what_a_listener_asking_for_faster_gets(self) -> None:
        """`(2, 1, 1)` and not `(1,)`, which is a musical finding and not taste.

        A section's close is written as two one-bar chords, so a one-bar pattern
        makes the cadence indistinguishable from the progression and the piece
        reads **0.0** against `harmonic_rhythm_variety`'s 0.10 floor — measured
        across three moods and three lengths. Every revision made with `(1,)`
        would be rejected by the ratchet, so the reading a listener gets for
        "faster" is the varied one.
        """
        assert _read("chords change faster").deltas == (SetHarmonicRhythm((2, 1, 1)),)
        assert _read("chords change faster").deltas != (SetHarmonicRhythm((1,)),)


class TestTheLongestMatchWins:
    """One ordering rule, and the three cases that would be wrong without it."""

    def test_less_expressive_is_lighter_and_expressive_is_not_also_read(self) -> None:
        """The pair the rule exists for: a phrase and a word inside it.

        Without the rule both entries fire, and the user who asked for less
        gets one request for less and one for more — with the order between them
        deciding which one the piece ends up as.
        """
        translation = _read("this is less expressive than I wanted")
        assert translation.deltas == (SetHumanization(level="light"),)

    def test_longer_chorus_is_a_section_and_not_a_longer_piece(self) -> None:
        translation = _read("make the longer chorus happen twice")
        assert [refusal.request for refusal in translation.refusals] == ["ExtendSection"]
        assert translation.deltas == ()

    def test_a_bare_relative_word_still_reads_on_its_own(self) -> None:
        translation = _read("longer")
        assert translation.deltas == (SetDuration(210),)

    def test_a_word_is_a_word_and_not_a_prefix(self) -> None:
        """Token matching rather than substring, and "calmer" is the case.

        A substring match reads "calmer" as a request for calm and then reports
        the "er" as unread — a request the user made, plus a word the reader
        did not understand, out of one adjective.
        """
        translation = _read("calmer")
        assert translation.deltas == (SetMood(Mood.CALMING),)
        assert translation.unread == ()

    def test_a_contraction_stays_one_word_and_so_does_an_apostrophe(self) -> None:
        """The tokeniser's other half: "doesn't" is not "doesn" and "t"."""
        translation = _read("it doesn't feel robotic")
        assert translation.deltas == (SetHumanization(level="none"),)
        assert translation.unread == ("it doesn't feel",)


class TestWhatWasNotReadIsReported:
    """A partial reading that says nothing about its missing half is a silent ignore."""

    def test_a_sentence_asking_for_two_things_reports_the_half_it_could_not_read(self) -> None:
        translation = _read("calmer please, and put a saxophone in it")
        assert translation.deltas == (SetMood(Mood.CALMING),)
        assert translation.unread == ("please and put a saxophone in it",)

    def test_nothing_read_at_all_is_not_an_error(self) -> None:
        """A sentence the table does not know is still a reading: of nothing."""
        translation = _read("what do you think of this")
        assert translation.deltas == ()
        assert translation.refusals == ()
        assert not translation.read
        assert translation.unread == ("what do you think of this",)

    def test_a_fully_read_sentence_reports_nothing_unread(self) -> None:
        translation = _read("calmer, no drums")
        assert translation.deltas == (SetMood(Mood.CALMING), SetDrumStyle(None))
        assert translation.unread == ()
        assert translation.read


class TestTheFallbackNeedsNoModel:
    """The plan's degradation bullet: the feedback box works with no LLM."""

    def test_no_client_reads_the_keywords_and_says_so(self) -> None:
        translation = _read("calmer", client=None)
        assert translation.source == "keywords"
        assert translation.deltas == (SetMood(Mood.CALMING),)
        assert "no language model is configured" in translation.note
        assert "keyword" in translation.note

    def test_an_unreachable_model_falls_back_with_the_code_in_the_note(self) -> None:
        client = _Model(
            ChatResult(error=LLMError(error_code="llm_unreachable", message="connection refused"))
        )
        translation = _read("calmer", client=client)
        assert translation.source == "keywords"
        assert translation.deltas == (SetMood(Mood.CALMING),)
        assert "llm_unreachable" in translation.note

    def test_a_model_that_answers_nothing_falls_back(self) -> None:
        translation = _read("calmer", client=_Model(ChatResult(content="  ")))
        assert translation.source == "keywords"
        assert "answered nothing" in translation.note

    def test_the_fallback_is_not_consulted_when_the_model_answered_in_prose(self) -> None:
        """A model that declines to read something is not second-guessed.

        Its sentence is the answer, and it was asked *because* the table is
        narrower. Reading the words anyway would turn "the engine cannot do
        that" into a request the model chose not to make.
        """
        client = _Model(ChatResult(content="the engine has no knob for reverb"))
        translation = _read("calmer with reverb", client=client)
        assert translation.source == "model"
        assert translation.deltas == ()
        assert translation.note == "the engine has no knob for reverb"

    def test_a_relative_read_is_anchored_on_the_piece_the_user_is_looking_at(self) -> None:
        """Both relative readers read the spec they were handed, not a default.

        The base of "longer" is the piece's own length and the base of "again"
        is its own seed, so a spec change moves what the same words mean — which
        is what makes these two requests rather than constants in disguise.
        """
        other = CompositionSpec(mood=Mood.CALMING, duration_seconds=300, seed=41)
        assert _read("longer", spec=other).deltas == (SetDuration(330),)
        assert _read("again", spec=other).deltas == (ReRoll(42),)

    def test_a_seedless_piece_rerolls_to_the_first_seed(self) -> None:
        """`seed=None` and `seed=0` compose identically, so the next draw is 1."""
        seedless = CompositionSpec(mood=Mood.CALMING, duration_seconds=180)
        assert _read("try again", spec=seedless).deltas == (ReRoll(1),)


class TestTheRequestsTheTableMakesAreOnesTheEngineWillTake:
    """The table proposes; the applier is the judge, and it is never a crash."""

    @pytest.mark.parametrize(("index", "entry"), list(enumerate(KEYWORD_TABLE)))
    def test_every_entry_folds_into_the_engine_without_a_crash(
        self, index: int, entry: Phrase
    ) -> None:
        """Every request the fallback can make, folded.

        Not "accepted" — an unbuilt request's refusal is the answer it is meant
        to give — but *answered*: the fold returns, and whatever it said about
        the request, it said it in the vocabulary's own words.
        """
        translation = _read(entry.phrases[0])
        requests: tuple[Delta, ...] = translation.deltas
        application = apply_deltas(_SPEC, requests)
        assert len(application.applied) + len(application.refused) == len(requests)

    def test_the_length_control_refuses_at_the_bound_rather_than_clamping(self) -> None:
        """The one entry whose request the spec can be too long for.

        A table that clamped would be writing the user's request for them: the
        honest answer to "longer" at the six-hundred-second cap is that the
        length cannot go up, with the bound named — which is the vocabulary's
        rule, and the applier is where it is enforced.
        """
        at_the_cap = CompositionSpec(mood=Mood.CALMING, duration_seconds=600, seed=5)
        translation = _read("make it longer", spec=at_the_cap)
        assert translation.deltas == (SetDuration(630),)
        application = apply_deltas(at_the_cap, translation.deltas)
        assert application.applied == ()
        assert [refusal.reason for refusal in application.refused] == ["violates_the_spec"]
        assert "600" in application.refused[0].message
        assert application.refused[0].nearest == "SetDuration(600)"

    def test_the_shortening_control_refuses_at_the_floor_the_same_way(self) -> None:
        shortest = CompositionSpec(mood=Mood.CALMING, duration_seconds=30, seed=5)
        application = apply_deltas(shortest, _read("shorter", spec=shortest).deltas)
        assert application.applied == ()
        assert application.refused[0].nearest == "SetDuration(30)"


class TestTheModelPath:
    """One call, a typed tool, and every call it makes landing somewhere visible."""

    def test_the_request_offers_the_vocabulary_the_engine_carries(self) -> None:
        """The schema is the same one `revise` offers, not a second copy.

        A hand-written list here would be a second place the vocabulary is
        written down, and the copy in a schema is the one that drifts.
        """
        client = _Model(_calls(("SetMood", {"mood": "sleep"})))
        request = client_request(client, "make it sleepier please")
        assert [spec.name for spec in request.tools] == [REQUEST_TOOL]
        assert request.tools[0].parameters["properties"]["knob"]["enum"] == sorted(DELTA_TYPES)
        assert request.tools[0].parameters["required"] == ["knob"]

    def test_the_request_carries_the_feedback_and_the_piece(self) -> None:
        """A model choosing a value for this piece has to be told what it is."""
        client = _Model(_calls(("SetMood", {"mood": "sleep"})))
        request = client_request(client, "make it sleepier")
        user = request.messages[-1].content
        assert "make it sleepier" in user
        assert "mood=calming" in user
        assert "duration_seconds=180s" in user
        assert "seed=5" in user
        assert request.messages[0].content == translator.SYSTEM_PROMPT

    def test_calls_become_the_requests_in_the_order_they_were_made(self) -> None:
        client = _Model(
            _calls(
                ("SetMood", {"mood": "sleep"}),
                ("SetDrumStyle", {"name": None}),
                content="doing both",
            )
        )
        translation = _read("sleepier and no drums please", client=client)
        assert translation.source == "model"
        assert translation.deltas == (SetMood(Mood.SLEEP), SetDrumStyle(None))
        assert translation.note == "doing both"
        assert translation.unread == ()

    def test_a_model_that_asks_for_an_unbuilt_knob_gets_the_vocabularys_refusal(self) -> None:
        """The two paths refuse the same thing in the same words.

        A model naming `SetSwing` is asking for exactly what the user asking for
        swing is asking for, and the sentence that comes back is `UNCARRIED`'s
        rather than one written here.
        """
        client = _Model(_calls(("SetSwing", {"ratio": 0.6})))
        translation = _read("add some swing", client=client)
        assert [refusal.request for refusal in translation.refusals] == ["SetSwing"]
        assert "triplet grid" in translation.refusals[0].message
        assert translation.unread == ()

    def test_a_model_that_invents_a_knob_is_refused_by_name(self) -> None:
        """`unknown_knob`'s second builder: a name nothing ever designed."""
        client = _Model(_calls(("SetReverb", {"amount": 0.4})))
        translation = _read("more reverb", client=client)
        assert [refusal.request for refusal in translation.refusals] == ["SetReverb"]
        refusal = translation.refusals[0]
        assert refusal.reason == "unknown_knob"
        assert "SetReverb" in refusal.message
        assert str(len(DELTA_TYPES)) in refusal.message
        assert refusal.nearest is None

    def test_a_call_with_no_arguments_is_reported_rather_than_dropped(self) -> None:
        """The call is spelled back, because there is no request to refuse.

        A malformed call is not a request the engine declined — it is a thing
        the model said that could not be read as one — and the two are told
        apart everywhere else in this module, so they are here too.
        """
        client = _Model(_calls(("SetTempo", {})))
        translation = _read("a bit faster", client=client)
        assert translation.deltas == ()
        assert translation.refusals == ()
        assert translation.unread == ("SetTempo()",)

    def test_a_value_the_enumeration_does_not_have_is_reported_rather_than_dropped(self) -> None:
        client = _Model(_calls(("SetHumanization", {"level": "loud"})))
        translation = _read("more human", client=client)
        assert translation.unread == ("SetHumanization(level='loud')",)

    def test_a_call_arguing_a_knob_of_its_own_is_not_able_to_rename_itself(self) -> None:
        """The call's own name is the knob, whatever its arguments also say."""
        client = _Model(_calls(("SetMood", {"knob": "SetReverb", "mood": "sleep"})))
        translation = _read("sleepier", client=client)
        assert translation.deltas == (SetMood(Mood.SLEEP),)


def client_request(client: _Model, text: str) -> ChatRequest:
    """Read `text` through `client` and hand back the request it was sent."""
    _read(text, client=client)
    assert len(client.requests) == 1
    return client.requests[0]


class TestTheModuleExports:
    def test_the_module_exports_what_it_claims(self) -> None:
        assert set(translator.__all__) == {
            "KEYWORD_TABLE",
            "REQUEST_TOOL",
            "REQUEST_TOOL_DESCRIPTION",
            "SYSTEM_PROMPT",
            "Phrase",
            "Translation",
            "translate",
        }
