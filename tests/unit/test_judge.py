"""The listening judge: what it is shown, what it is not, and what it is counted as.

The module's claim is that it produces an opinion the scorecard did not
produce. Two things have to hold for that to be worth anything, and both are
tested here: the judge cannot see a measurement, and an answer that moved with
the order is not counted as an answer.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest

from saimc.compose.engine import compose
from saimc.compose.percussion import DRUM_KICK, DRUM_SNARE
from saimc.compose.score import (
    PPQ,
    VOICE_BASS,
    VOICE_HARMONY,
    VOICE_MELODY,
    VOICE_PERCUSSION,
    KeySignature,
    Measure,
    NotationScore,
    NoteEvent,
)
from saimc.judge import (
    JudgedPiece,
    JudgeVerdict,
    Pairing,
    PairJudgement,
    describe_piece,
    duration_name,
    judge_pair,
    note_name,
    pairs_from,
    read_verdict,
    score_judgements,
)
from saimc.llm.base import ChatRequest, ChatResult, LLMError
from saimc.quality import score_piece
from saimc.spec import CompositionSpec, Mood


def _score(*notes: NoteEvent, bars: int = 1) -> NotationScore:
    bar = 4 * PPQ
    return NotationScore.make(
        ppq=PPQ,
        key=KeySignature(root="C", mode="major"),
        time_signature="4/4",
        tempo_bpm=100,
        measures=[
            Measure(index=i, start_tick=i * bar, end_tick=(i + 1) * bar, time_signature="4/4")
            for i in range(bars)
        ],
        notes=list(notes),
    )


def _piece(label: str, **quality: Any) -> JudgedPiece:
    out = compose(CompositionSpec(mood=Mood.CALMING, duration_seconds=30, seed=1))
    return JudgedPiece(
        label=label,
        notation=out.notation_score,
        quality=score_piece(out.notation_score),
    )


class ScriptedJudge:
    """A judge with a fixed answer, recording what it was shown."""

    def __init__(self, *replies: ChatResult) -> None:
        self.replies = list(replies)
        self.requests: list[ChatRequest] = []

    async def chat(self, request: ChatRequest) -> ChatResult:
        self.requests.append(request)
        if not self.replies:
            return ChatResult(error=LLMError("llm_unreachable", "the script is empty"))
        return self.replies.pop(0)

    async def aclose(self) -> None:
        return None


def _says(prefers: str, reason: str = "the second has a line that goes somewhere") -> ChatResult:
    return ChatResult(content=f'{{"prefers": "{prefers}", "reason": "{reason}"}}')


class TestSpelling:
    @pytest.mark.parametrize(
        ("midi", "expected"), [(60, "C4"), (21, "A0"), (61, "C#4"), (127, "G9")]
    )
    def test_note_names(self, midi: int, expected: str) -> None:
        assert note_name(midi) == expected

    @pytest.mark.parametrize(
        ("ticks", "expected"),
        [
            (PPQ, "quarter"),
            (PPQ * 2, "half"),
            (PPQ // 2, "eighth"),
            (PPQ * 3 // 2, "dotted-quarter"),
            (PPQ * 4, "whole"),
        ],
    )
    def test_duration_names(self, ticks: int, expected: str) -> None:
        assert duration_name(ticks) == expected

    def test_a_length_with_no_name_is_reported_in_beats(self) -> None:
        assert duration_name(PPQ * 5 // 3) == "1.67-beat"


class TestTheJudgeIsBlind:
    def test_a_description_carries_no_measurement(self) -> None:
        """The blindness is the whole design, so it is asserted over real output.

        Every metric name, checked against the text of a described piece. A
        prompt that leaked one would make the agreement rate partly a measure
        of the scorecard agreeing with itself.
        """
        out = compose(CompositionSpec(mood=Mood.ELECTRIFYING, duration_seconds=120, seed=3))
        text = describe_piece(out.notation_score).lower()
        for metric in score_piece(out.notation_score).as_dict():
            assert metric.replace("_", " ") not in text
            assert metric not in text

    def test_it_cannot_be_handed_one_even_by_accident(self) -> None:
        """`describe_piece` takes a score and a bar count. That is the blindness.

        A prompt that promises not to mention the metrics is a promise; a
        function with no parameter to receive one cannot break it. The
        signature is the guard, so the signature is what is asserted.
        """
        assert set(inspect.signature(describe_piece).parameters) == {"score", "bars"}
        with pytest.raises(TypeError):
            describe_piece(_score(), quality=object())  # type: ignore[call-arg]


class TestWhatTheJudgeReads:
    def test_the_voices_are_named_the_way_the_engine_numbers_them(self) -> None:
        """The melody line holds the melody, and the bass line the bass.

        The first draft of `judge.py` hard-coded the four voice ids in the
        wrong order, which labelled the bass as the melody. It read as music
        either way round, which is exactly why this is a test and not a glance.
        """
        text = describe_piece(
            _score(
                NoteEvent(voice_id=VOICE_MELODY, pitch_midi=84, tick=0, duration_ticks=PPQ),
                NoteEvent(voice_id=VOICE_HARMONY, pitch_midi=64, tick=0, duration_ticks=PPQ),
                NoteEvent(voice_id=VOICE_BASS, pitch_midi=36, tick=0, duration_ticks=PPQ),
            )
        )
        assert "Melody: C6 (quarter)" in text
        assert "Harmony: E4 (quarter)" in text
        assert "Bass: C2 (quarter)" in text

    def test_the_voices_are_written_top_down(self) -> None:
        text = describe_piece(
            _score(
                NoteEvent(voice_id=VOICE_BASS, pitch_midi=36, tick=0, duration_ticks=PPQ),
                NoteEvent(voice_id=VOICE_MELODY, pitch_midi=84, tick=0, duration_ticks=PPQ),
            )
        )
        assert text.index("Melody:") < text.index("Bass:")

    def test_the_kit_is_named_rather_than_pitched(self) -> None:
        """On channel 10 the pitch is the drum, so `C2 (eighth)` would be a lie."""
        text = describe_piece(
            _score(
                NoteEvent(
                    voice_id=VOICE_PERCUSSION, pitch_midi=DRUM_KICK, tick=0, duration_ticks=60
                ),
                NoteEvent(
                    voice_id=VOICE_PERCUSSION,
                    pitch_midi=DRUM_SNARE,
                    tick=PPQ * 2,
                    duration_ticks=60,
                ),
            )
        )
        assert "Drums: 1 kick, 3 snare" in text
        assert note_name(DRUM_KICK) not in text

    def test_simultaneous_notes_are_written_as_a_chord(self) -> None:
        text = describe_piece(
            _score(
                NoteEvent(voice_id=VOICE_HARMONY, pitch_midi=60, tick=0, duration_ticks=PPQ),
                NoteEvent(voice_id=VOICE_HARMONY, pitch_midi=64, tick=0, duration_ticks=PPQ),
            )
        )
        assert "Harmony: C4+E4 (quarter)" in text

    def test_a_truncated_piece_says_how_much_it_left_out(self) -> None:
        out = compose(CompositionSpec(mood=Mood.CALMING, duration_seconds=300, seed=2))
        text = describe_piece(out.notation_score, bars=4)
        assert "bars 1-4 of" in text
        assert text.count("\nBar ") == 4

    def test_a_short_piece_claims_no_truncation(self) -> None:
        assert "of" not in describe_piece(_score(), bars=8).split("\n")[4]


class TestTheScorecardSOwnCall:
    def test_the_better_measured_piece_is_preferred(self) -> None:
        clean = _piece("clean")
        # Same piece, but refused by the linter — the first element of the
        # musical order, and the one nothing below it can overturn.
        illegal = JudgedPiece(
            label="illegal",
            notation=clean.notation,
            quality=clean.quality,
            lint_passed=False,
        )
        assert Pairing("p", left=clean, right=illegal).scorecard_prefers() == "left"
        assert Pairing("p", left=illegal, right=clean).scorecard_prefers() == "right"

    def test_two_pieces_the_music_cannot_separate_are_neither(self) -> None:
        """A tie is reported as a tie rather than settled by a plan hash.

        `arbiter_order` breaks such a tie on the plan hash and the seed, which
        is deliberately arbitrary. Asking a listener to agree with a hash would
        be asking them to lose, so the pair is incomparable instead.
        """
        piece = _piece("a")
        same = JudgedPiece(label="b", notation=piece.notation, quality=piece.quality)
        assert Pairing("p", left=piece, right=same).scorecard_prefers() == "neither"


class TestReadingTheAnswer:
    def test_plain_json(self) -> None:
        verdict = read_verdict('{"prefers": "second", "reason": "it breathes"}')
        assert verdict.prefers == "second"
        assert verdict.reason == "it breathes"
        assert verdict.usable

    def test_a_fenced_block_is_unwrapped(self) -> None:
        assert read_verdict('```json\n{"prefers": "first", "reason": "x"}\n```').prefers == "first"

    def test_prose_is_an_error_rather_than_a_guess(self) -> None:
        """A judge that has to be interpreted is one whose rate is the reader's."""
        verdict = read_verdict("I think the first one, probably.")
        assert not verdict.usable
        assert verdict.error == "the reply was not JSON"

    def test_an_unknown_preference_is_named(self) -> None:
        assert read_verdict('{"prefers": "the left one"}').error == (
            "unreadable preference 'the left one'"
        )


class TestBothWaysRound:
    @pytest.mark.asyncio
    async def test_the_pair_is_asked_twice_with_the_sides_swapped(self) -> None:
        left, right = _piece("left"), _piece("right")
        # Two different pieces, so the two prompts are distinguishable.
        right = JudgedPiece(
            label="right",
            notation=compose(
                CompositionSpec(mood=Mood.SLEEP, duration_seconds=30, seed=9)
            ).notation_score,
            quality=right.quality,
        )
        judge = ScriptedJudge(_says("first"), _says("second"))
        await judge_pair(judge, Pairing("p", left=left, right=right))

        assert len(judge.requests) == 2
        first_prompt = judge.requests[0].messages[1].content
        second_prompt = judge.requests[1].messages[1].content
        assert first_prompt != second_prompt
        # What was PIECE ONE the first time is PIECE TWO the second.
        written_left = describe_piece(left.notation)
        assert written_left in first_prompt.split("PIECE TWO")[0]
        assert written_left in second_prompt.split("PIECE TWO")[1]

    @pytest.mark.asyncio
    async def test_an_answer_that_survives_the_swap_is_the_judge_s_answer(self) -> None:
        judge = ScriptedJudge(_says("first"), _says("second"))
        judged = await judge_pair(judge, Pairing("p", left=_piece("a"), right=_piece("b")))
        assert judged.consistent
        assert judged.prefers == "left"

    @pytest.mark.asyncio
    async def test_a_judge_that_always_says_first_has_told_you_about_itself(self) -> None:
        """Position bias, caught by construction rather than hoped against."""
        judge = ScriptedJudge(_says("first"), _says("first"))
        judged = await judge_pair(judge, Pairing("p", left=_piece("a"), right=_piece("b")))
        assert not judged.consistent
        assert judged.prefers is None
        assert judged.agrees is None

    @pytest.mark.asyncio
    async def test_a_failed_call_is_not_a_verdict(self) -> None:
        judge = ScriptedJudge(_says("first"))  # the second call runs out of script
        judged = await judge_pair(judge, Pairing("p", left=_piece("a"), right=_piece("b")))
        assert not judged.consistent
        assert judged.reverse.error == "llm_unreachable"


def _judgement(pair_id: str, forward: str, reverse: str, scorecard: str) -> PairJudgement:
    return PairJudgement(
        pair_id=pair_id,
        forward=JudgeVerdict(forward, "because"),  # type: ignore[arg-type]
        reverse=JudgeVerdict(reverse, "because"),  # type: ignore[arg-type]
        scorecard=scorecard,  # type: ignore[arg-type]
    )


class TestWhatASweepComesTo:
    def test_the_rates_count_what_they_say_they_count(self) -> None:
        report = score_judgements(
            [
                _judgement("agree", "first", "second", "left"),
                _judgement("disagree", "first", "second", "right"),
                _judgement("inconsistent", "first", "first", "left"),
                _judgement("listener-tied", "neither", "neither", "left"),
                _judgement("scorecard-tied", "first", "second", "neither"),
            ],
            label="x",
            model="m",
        )
        assert report.pairs == 5
        # The tied-both-ways pair is consistent; it just cannot be compared.
        assert report.consistent_pairs == 4
        assert report.comparable_pairs == 2
        assert report.agreeing_pairs == 1
        assert report.agreement_rate == 0.5
        assert report.consistency_rate == 0.8

    def test_an_empty_sweep_rates_zero_rather_than_dividing_by_it(self) -> None:
        report = score_judgements([], label="x", model="m")
        assert report.agreement_rate == 0.0
        assert report.consistency_rate == 0.0

    def test_the_disagreements_carry_the_listener_s_own_sentence(self) -> None:
        report = score_judgements(
            [_judgement("d", "first", "second", "right")], label="x", model="m"
        )
        (only,) = report.disagreements()
        assert only.pair_id == "d"
        payload = report.to_dict()["disagreements"][0]
        assert payload["listener_preferred"] == "left"
        assert payload["scorecard_preferred"] == "right"
        assert payload["reason"] == "because"

    def test_the_report_carries_no_verdict(self) -> None:
        """There is no bar yet, so there is nothing for a `passed` to mean."""
        assert not hasattr(score_judgements([], label="x", model="m"), "passed")


class TestPairing:
    def test_every_piece_is_compared_once(self) -> None:
        pieces = [_piece(f"p{i}") for i in range(6)]
        pairs = pairs_from(pieces, seed=3)
        assert len(pairs) == 3
        labels = [p.left.label for p in pairs] + [p.right.label for p in pairs]
        assert sorted(labels) == sorted(p.label for p in pieces)

    def test_an_odd_piece_out_is_dropped_rather_than_paired_with_itself(self) -> None:
        pairs = pairs_from([_piece(f"p{i}") for i in range(5)], seed=0)
        assert len(pairs) == 2

    def test_the_shuffle_is_seeded(self) -> None:
        pieces = [_piece(f"p{i}") for i in range(8)]
        assert [p.pair_id for p in pairs_from(pieces, seed=7)] == [
            p.pair_id for p in pairs_from(pieces, seed=7)
        ]
        assert [p.pair_id for p in pairs_from(pieces, seed=7)] != [
            p.pair_id for p in pairs_from(pieces, seed=8)
        ]
