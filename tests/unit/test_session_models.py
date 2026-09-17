"""The session records: their invariants, and their document round trips.

Two properties carry the weight here, and both are about what a *stored*
record promises:

- A draft is a record, not a copy. It holds the pair `compose` is a function
  of, so `test_a_draft_recomposes_to_the_performance_it_recorded` recomposes
  and compares — which is the only thing that makes "the draft describes the
  music the user heard" a checkable claim rather than a convention.
- Every read goes through a constructor, so the invariants hold on load as
  well as on construction. The refusals are therefore tested twice where it
  matters: once by building the bad value, once by hand-editing the document
  and reading it back.

The fixtures compose for real. A stub `PieceQuality` or a hand-written
`LintReport` would let the round trip pass while the composition it claims to
describe was impossible.
"""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime
from typing import get_args

import pytest

from saimc.compose.engine import EngineOutput, compose
from saimc.compose.linter import LintReport, lint
from saimc.compose.motif import BassMotion
from saimc.compose.plan import (
    PLAN_FORMAT,
    UnsupportedPlanVersionError,
    default_plan,
)
from saimc.llm.base import LLMError
from saimc.quality import score_piece
from saimc.session.deltas import (
    Delta,
    RequestSource,
    SetBassMotion,
    SetTempo,
    SetTimeSignature,
    apply_deltas,
)
from saimc.session.models import (
    _REQUEST_SOURCES,
    SESSION_FORMAT,
    SESSION_SCHEMA_VERSION,
    Draft,
    Session,
    SketchRecord,
    ToolInvocation,
    Turn,
    UnsupportedSessionVersionError,
    Verdict,
    require_id_segment,
)
from saimc.session.tools import _draft_from
from saimc.spec import (
    SPEC_SCHEMA_VERSION,
    CompositionSpec,
    Mood,
    TimeSignature,
    UnsupportedSpecVersionError,
)

_NOW = datetime(2026, 9, 16, 12, 0, tzinfo=UTC)
_SPEC = CompositionSpec(mood=Mood.CALMING, duration_seconds=30, seed=5)
_PLAN = default_plan(_SPEC)
_OUTPUT = compose(_SPEC, plan=_PLAN)

_SKETCH = SketchRecord(
    wav_path="sketches/d1/audio.wav",
    ogg_path="sketches/d1/audio.ogg",
    ogg_sha256="0" * 64,
    ogg_size_bytes=4096,
)


def _lint_report(output: EngineOutput = _OUTPUT) -> LintReport:
    return lint(
        output.notation_score,
        chord_bars=output.chord_bars or None,
        bar_keys=output.bar_keys or None,
        voice_instruments={v.voice_id: v.instrument for v in output.voice_instruments},
    )


def _draft(
    draft_id: str = "draft-one",
    *,
    sketch: SketchRecord | None = None,
    parent_id: str | None = None,
    deltas: tuple[Delta, ...] = (),
    requests_source: RequestSource | None = None,
) -> Draft:
    """A real draft: composed, linted and scored, not assembled by hand."""
    return Draft(
        draft_id=draft_id,
        created_at=_NOW,
        spec=_SPEC,
        plan=_PLAN,
        performance_plan_hash=_OUTPUT.performance_plan.compute_hash(),
        quality=score_piece(_OUTPUT.notation_score, piece=draft_id),
        lint=_lint_report(),
        parent_id=parent_id,
        deltas=deltas,
        requests_source=requests_source,
        sketch=sketch,
    )


def _revision(deltas: tuple[Delta, ...], *, parent_id: str = "draft-one") -> Draft:
    """A real revision: the chain folded from the root, then composed under it.

    Built through the fold and the tool's own recorder rather than posed,
    because the record's two halves are not independent: the plan *is* the
    fold's plan, and a fixture carrying one chain and another chain's plan would
    make every round-trip case below a case about a document no tool can write.
    """
    application = apply_deltas(_SPEC, deltas)
    output = compose(application.spec, plan=application.plan)
    return _draft_from(
        "draft-two",
        application.spec,
        output,
        parent_id=parent_id,
        deltas=application.applied,
        requests_source="conductor",
    )


def _session(
    *,
    session_id: str = "session-one",
    brief: str = "something for a rainy day",
    turns: list[Turn] | None = None,
    drafts: list[Draft] | None = None,
    verdicts: list[Verdict] | None = None,
    finalized_job_id: str | None = None,
    spec: CompositionSpec | None = None,
) -> Session:
    return Session(
        session_id=session_id,
        created_at=_NOW,
        updated_at=_NOW,
        brief=brief,
        spec=spec,
        turns=[] if turns is None else turns,
        drafts=[] if drafts is None else drafts,
        verdicts=[] if verdicts is None else verdicts,
        finalized_job_id=finalized_job_id,
    )


class TestAnIdIsOnePathSegment:
    """Ids become directory names, so anything else is refused at the model.

    Refusing here rather than at the join is what makes the rule hold on
    load too: a document naming `../../etc` never becomes a `Session`, so no
    later caller has to remember to check.
    """

    @pytest.mark.parametrize("bad", ["", ".", "..", "a/b", "a\\b", "x\x00y"])
    def test_a_traversing_or_empty_id_is_refused(self, bad: str) -> None:
        with pytest.raises(ValueError):
            require_id_segment(bad, label="session_id")

    def test_an_ordinary_id_passes(self) -> None:
        require_id_segment("0f8a4c2b", label="session_id")

    def test_a_session_refuses_a_traversing_id(self) -> None:
        with pytest.raises(ValueError, match="session_id"):
            _session(session_id="../escape")

    def test_a_draft_refuses_a_traversing_id(self) -> None:
        with pytest.raises(ValueError, match="draft_id"):
            _draft("../escape")


class TestSketchRecord:
    def test_a_recorded_path_stays_inside_the_session(self) -> None:
        with pytest.raises(ValueError, match="climb out"):
            replace(_SKETCH, ogg_path="sketches/../../audio.ogg")

    def test_an_absolute_path_is_refused(self) -> None:
        with pytest.raises(ValueError, match="relative to the session"):
            replace(_SKETCH, wav_path="/tmp/audio.wav")

    def test_an_empty_digest_is_refused(self) -> None:
        with pytest.raises(ValueError, match="digest"):
            replace(_SKETCH, ogg_sha256="")

    def test_a_negative_size_is_refused(self) -> None:
        with pytest.raises(ValueError, match="negative"):
            replace(_SKETCH, ogg_size_bytes=-1)


class TestToolInvocation:
    def test_a_refusal_must_name_its_reason(self) -> None:
        with pytest.raises(ValueError, match="must name its reason"):
            ToolInvocation(name="apply_delta", outcome="refused")

    def test_an_error_must_name_its_reason(self) -> None:
        with pytest.raises(ValueError, match="must name its reason"):
            ToolInvocation(name="draft", outcome="error")

    def test_a_success_may_not_also_carry_a_reason(self) -> None:
        with pytest.raises(ValueError, match="succeeded"):
            ToolInvocation(name="draft", outcome="ok", error_code="delta_refused")

    def test_a_refusal_is_recorded_rather_than_raised(self) -> None:
        """A refusal is an answer. The record keeps it as one."""
        call = ToolInvocation(
            name="apply_delta",
            arguments={"voice": "melody", "register": 12},
            result="out of the celesta's compass; nearest legal is 7",
            outcome="refused",
            error_code="delta_out_of_range",
        )
        assert call.ok is False
        assert call.result.startswith("out of the celesta")

    def test_a_tool_with_no_arguments_round_trips(self) -> None:
        call = ToolInvocation(name="compare", result="ranked 3 drafts")
        assert ToolInvocation.from_document(call.to_document()) == call

    def test_an_unknown_outcome_is_refused_on_load(self) -> None:
        """A `Literal` is a promise by the writer; a document is not the writer."""
        document = ToolInvocation(name="draft").to_document()
        document["outcome"] = "shrugged"
        with pytest.raises(ValueError, match="must be one of"):
            ToolInvocation.from_document(document)


class TestTurn:
    def test_a_failed_conductor_call_carries_no_tool_calls(self) -> None:
        with pytest.raises(ValueError, match="ran no tools"):
            Turn(
                created_at=_NOW,
                trigger="brief",
                calls=(ToolInvocation(name="draft"),),
                llm_error=LLMError("llm_unreachable", "no host"),
            )

    def test_a_silent_turn_is_allowed_and_recorded(self) -> None:
        """A model that answers nothing is a real outcome, not an error.

        Refusing the record would leave the loop with nothing to show and
        nothing to bound; the budget bounds it instead.
        """
        turn = Turn(created_at=_NOW, trigger="message")
        assert turn.narration == ""
        assert turn.calls == ()
        assert turn.failed is False

    def test_an_unknown_trigger_is_refused_on_load(self) -> None:
        document = Turn(created_at=_NOW, trigger="brief").to_document()
        document["trigger"] = "telepathy"
        with pytest.raises(ValueError, match="must be one of"):
            Turn.from_document(document)

    def test_a_turn_round_trips_with_its_failure(self) -> None:
        turn = Turn(
            created_at=_NOW,
            trigger="auto",
            llm_error=LLMError("llm_not_configured", "no valid LLM configuration"),
            model="a-model",
            latency_ms=12,
        )
        loaded = Turn.from_document(turn.to_document())
        assert loaded == turn
        assert loaded.failed is True
        assert loaded.llm_error is not None
        assert loaded.llm_error.error_code == "llm_not_configured"

    def test_a_turn_round_trips_with_its_calls_in_order(self) -> None:
        turn = Turn(
            created_at=_NOW,
            trigger="brief",
            narration="Drafting four candidates.",
            calls=(
                ToolInvocation(name="parse_brief", arguments={"text": "rainy"}, result="{}"),
                ToolInvocation(name="draft", arguments={"n": 4}, result='["draft-two"]'),
            ),
            model="a-model",
            latency_ms=800,
        )
        loaded = Turn.from_document(turn.to_document())
        assert loaded == turn
        assert [call.name for call in loaded.calls] == ["parse_brief", "draft"]


class TestVerdict:
    def test_a_verdict_with_nothing_in_it_is_refused(self) -> None:
        with pytest.raises(ValueError, match="like, a dislike, or feedback"):
            Verdict(draft_id="draft-one", at=_NOW)

    def test_feedback_alone_is_a_verdict(self) -> None:
        verdict = Verdict(draft_id="draft-one", at=_NOW, feedback="warmer, please")
        assert verdict.value is None

    def test_a_like_alone_is_a_verdict(self) -> None:
        assert Verdict(draft_id="draft-one", at=_NOW, value="like").value == "like"

    def test_whitespace_is_not_feedback(self) -> None:
        with pytest.raises(ValueError, match="like, a dislike, or feedback"):
            Verdict(draft_id="draft-one", at=_NOW, feedback="   ")

    def test_an_unknown_value_is_refused_on_load(self) -> None:
        document = Verdict(draft_id="draft-one", at=_NOW, value="like").to_document()
        document["value"] = "love"
        with pytest.raises(ValueError, match="must be one of"):
            Verdict.from_document(document)


class TestDraft:
    def test_a_draft_recomposes_to_the_performance_it_recorded(self) -> None:
        """The load-bearing claim: a draft is a record, not a copy.

        If this fails, `performance_plan_hash` is a number that describes
        nothing and a stored draft has stopped being the music the user
        heard.
        """
        draft = _draft()
        again = compose(draft.spec, plan=draft.plan)
        assert again.performance_plan.compute_hash() == draft.performance_plan_hash
        assert again.notation_score.compute_hash() == draft.score_hash

    def test_the_score_hash_is_the_lint_reports_own(self) -> None:
        """One owner for one value: two copies can disagree, and this is where."""
        draft = _draft()
        assert draft.score_hash == draft.lint.score_hash

    def test_the_plan_hash_is_the_plans_own(self) -> None:
        assert _draft().plan_hash == _PLAN.compute_hash()

    def test_a_draft_without_a_sketch_is_a_real_state(self) -> None:
        assert _draft().sketch is None

    def test_a_draft_round_trips(self) -> None:
        draft = _draft(sketch=_SKETCH, parent_id=None)
        loaded = Draft.from_document(draft.to_document())
        assert loaded == draft
        assert loaded.quality == draft.quality
        assert loaded.lint.issues == draft.lint.issues
        assert loaded.plan.compute_hash() == draft.plan_hash

    def test_a_draft_keeps_its_lint_findings_whole(self) -> None:
        """A pass/fail flag would drop the reason a piece is illegal."""
        document = _draft().to_document()
        assert document["lint"]["passed"] is True
        assert document["lint"]["score_hash"] == _OUTPUT.notation_score.compute_hash()

    def test_a_newer_spec_schema_is_refused(self) -> None:
        document = _draft().to_document()
        document["spec"]["schema_version"] = SPEC_SCHEMA_VERSION + 1
        with pytest.raises(UnsupportedSpecVersionError, match="schema version"):
            Draft.from_document(document)

    def test_a_plan_this_build_cannot_read_is_refused(self) -> None:
        """A plan is stored materialized, so both directions are refused."""
        document = _draft().to_document()
        document["plan"]["format"] = "CompositionPlan:99"
        with pytest.raises(UnsupportedPlanVersionError, match="cannot read"):
            Draft.from_document(document)

    def test_the_plan_is_stored_as_the_document_its_hash_covers(self) -> None:
        document = _draft().to_document()
        assert document["plan"]["format"] == PLAN_FORMAT
        assert document["plan"] == _PLAN.to_canonical_dict()


class TestTheChainOfDeltas:
    """`Draft.deltas` is the accumulated chain, and the record is where that is kept.

    The whole difficulty of a revision is that a plan is *derived* from the spec,
    and `default_plan` reads only a mood and a meter — so a second revision
    cannot patch the first one's plan, and folding only its own deltas from its
    own spec would quietly discard them. What makes that recoverable is that the
    chain is on the record rather than in a step, and these are the properties
    the fold depends on.
    """

    _CHAIN = (
        SetBassMotion(motion=BassMotion.SPARSE),
        SetTempo(tempo_bpm=90),
    )

    def test_a_revision_round_trips_with_its_whole_chain(self) -> None:
        """The deltas have to survive a save and a load for a fold to be a replay.

        Equality is asserted on the tuple rather than on its length, because the
        types are frozen dataclasses and two deltas that are equal are the same
        request — which is the property the fold reads.
        """
        draft = _revision(self._CHAIN)
        loaded = Draft.from_document(draft.to_document())

        assert loaded == draft
        assert loaded.deltas == self._CHAIN

    def test_the_chain_is_written_as_plain_values(self) -> None:
        """No type names in the document, so a log of these is readable on its own.

        The knob is the key a reader dispatches on and every field is its plain
        value, which is what `delta_to_dict` promises and what the preference
        log will be read through.
        """
        document = _revision(self._CHAIN).to_document()

        assert document["deltas"] == [
            {"knob": "SetBassMotion", "motion": "sparse"},
            {"knob": "SetTempo", "tempo_bpm": 90},
        ]

    def test_a_revision_records_the_chain_it_folded(self) -> None:
        """The premise the round trip rests on: the fixture is a real revision.

        A chain that did not actually move the plan would make the case above
        about two identical documents, so both halves of the fold are asserted —
        the tempo it wrote and the figure it moved to the front.
        """
        draft = _revision(self._CHAIN)

        assert draft.spec.tempo_bpm == 90
        assert draft.plan.bass_figures[0] != _PLAN.bass_figures[0]

    def test_a_meter_delta_round_trips_as_its_member(self) -> None:
        """The enums are held as members rather than as their names.

        `delta_to_dict` writes the plain value and `delta_from_dict` hands it
        back, so a name outside the enum fails at the constructor instead of
        becoming a string nothing else in the module expects. The premise is
        asserted — the document really does carry the bare name — because a
        serializer that wrote the `TimeSignature` repr would pass the round trip
        while being useless to a reader.
        """
        delta = SetTimeSignature(time_signature=TimeSignature.SIX_EIGHT)
        document = _revision((delta,)).to_document()

        assert document["deltas"] == [{"knob": "SetTimeSignature", "time_signature": "6/8"}]
        assert Draft.from_document(document).deltas == (delta,)

    def test_a_draft_with_no_chain_writes_an_empty_list(self) -> None:
        """A root draft is a draft with an empty chain, not one with no key."""
        draft = _draft()

        assert draft.deltas == ()
        assert draft.to_document()["deltas"] == []

    def test_a_document_from_before_the_chain_is_read_as_empty(self) -> None:
        """`.get()` tolerance, the same one every other sidecar field has.

        A draft written by the build before this one has no `deltas` key at all,
        and it means the same thing as an empty one: the draft was not a
        revision.
        """
        document = _draft().to_document()
        del document["deltas"]

        assert Draft.from_document(document).deltas == ()

    def test_deltas_with_no_parent_are_refused(self) -> None:
        """A contradiction rather than a preference: a revision of nothing.

        The chain is folded from the *root's* spec, and the root is found by
        walking `parent_id` — a draft carrying deltas and naming no parent has
        no root to fold from, so the record refuses to exist rather than
        deferring the failure to the first revision that tries to use it.

        The record here is malformed on purpose: it is the shape under test, and
        `_draft` is the only way to reach it, since `_revision` goes through the
        tool that cannot build one.
        """
        with pytest.raises(ValueError, match="deltas with no parent"):
            _draft("draft-two", deltas=self._CHAIN)

    def test_a_document_that_contradicts_itself_is_refused_on_load(self) -> None:
        """The load path, not the constructor — C3's rule for closed vocabularies.

        A hand-edited document is not the writer, so the invariant has to hold
        where the record is *read* and not merely where it is built.
        """
        document = _draft().to_document()
        document["deltas"] = [{"knob": "SetTempo", "tempo_bpm": 90}]

        with pytest.raises(ValueError, match="deltas with no parent"):
            Draft.from_document(document)

    def test_a_document_naming_a_knob_this_build_lacks_is_refused(self) -> None:
        """The vocabulary is closed, and an unknown knob is named rather than skipped.

        A dropped delta would be a revision that silently did not happen, which
        is the failure the whole delta module refuses once — here on the load
        path, where a document is not the writer.
        """
        document = _revision(self._CHAIN).to_document()
        document["deltas"][0] = {"knob": "SetSwingRatio", "ratio": 0.5}

        with pytest.raises(ValueError, match="unknown knob"):
            Draft.from_document(document)


class TestHowTheRequestsWereAskedFor:
    """One value per draft: which of the four paths produced this step's requests.

    It is what makes the preference log a log of *preferences* rather than of
    edits — a like attached to a delta the user named and one attached to a
    delta a model chose weigh differently when the log is read as a dataset, and
    a record that could not tell them apart would have thrown that away at the
    moment it was written. So the field is part of the document, and these are
    the properties a document has to have.
    """

    _CHAIN = (SetBassMotion(motion=BassMotion.SPARSE),)

    def test_the_source_is_written_and_read_back(self) -> None:
        draft = _revision(self._CHAIN)

        assert draft.requests_source == "conductor"
        assert draft.to_document()["requests_source"] == "conductor"
        assert Draft.from_document(draft.to_document()).requests_source == "conductor"

    def test_a_draft_drafted_from_the_brief_alone_records_no_source(self) -> None:
        """There is no step to name: nothing asked for this draft in particular."""
        assert _draft().requests_source is None
        assert _draft().to_document()["requests_source"] is None

    def test_a_document_from_before_the_source_is_read_as_absent(self) -> None:
        """`.get()` tolerance, as every other field of this record has.

        A draft written by the build before `requests_source` is a draft whose
        step has no record of who asked for it — which is the honest reading and
        not the same as "typed".
        """
        document = _revision(self._CHAIN).to_document()
        del document["requests_source"]

        assert Draft.from_document(document).requests_source is None

    def test_a_source_outside_the_vocabulary_is_refused_on_load(self) -> None:
        """A `Literal` field is a promise by the writer, and a document is not the writer."""
        document = _revision(self._CHAIN).to_document()
        document["requests_source"] = "whispered"

        with pytest.raises(ValueError, match="requests_source"):
            Draft.from_document(document)

    def test_the_sources_the_record_accepts_are_the_ones_the_type_names(self) -> None:
        """The premise the refusal above rests on, asserted rather than believed.

        A value added to `RequestSource` and forgotten here would make the
        load-time refusal reject a source this build writes, which is the kind of
        disagreement a ratchet is for.
        """
        assert set(_REQUEST_SOURCES) == set(get_args(RequestSource))

    def test_a_chain_with_no_recorded_source_is_read_as_absent(self) -> None:
        """The tolerance, and the reason the invariant is one-directional.

        A strict "the pair arrives together" would have made this field the one
        additive field on the record that a document cannot be missing, which is
        the *opposite* of what every other default-bearing field here does — and
        `test_a_document_from_before_the_chain_is_read_as_empty` is the
        precedent. What is refused is the shape nothing can produce: a source
        for a step that asked for nothing. The complete pair is the writer's
        business, and `revise_draft` is the only writer.
        """
        draft = _draft("draft-two", parent_id="draft-one", deltas=self._CHAIN)

        assert draft.requests_source is None

    def test_a_source_with_no_requests_is_refused(self) -> None:
        """A root draft has no step to place, so a source on one is provenance for nothing."""
        with pytest.raises(ValueError, match="no step to place"):
            _draft("draft-two", requests_source="typed")

    def test_a_document_that_records_a_source_for_no_requests_is_refused_on_load(self) -> None:
        """The load path, not the constructor: a document on disk is not the writer."""
        document = _revision(self._CHAIN).to_document()
        document["deltas"] = []

        with pytest.raises(ValueError, match="no step to place"):
            Draft.from_document(document)


class TestSession:
    def test_a_session_needs_a_brief(self) -> None:
        with pytest.raises(ValueError, match="needs a brief"):
            _session(brief="   ")

    def test_two_drafts_may_not_share_an_id(self) -> None:
        with pytest.raises(ValueError, match="share an id"):
            _session(drafts=[_draft("same"), _draft("same")])

    def test_a_verdict_must_name_a_draft_the_session_has(self) -> None:
        with pytest.raises(ValueError, match="does not have"):
            _session(
                drafts=[_draft("draft-one")],
                verdicts=[Verdict(draft_id="draft-two", at=_NOW, value="like")],
            )

    def test_the_check_is_reachable_after_a_mutation(self) -> None:
        """The lists a constructor validated can be appended to afterwards.

        That is why `check()` is public: the store calls it before writing,
        so a session mutated into an inconsistent state is refused by the
        writer rather than by the reader that finds it later.
        """
        draft = _draft("same")
        session = _session(drafts=[draft])
        session.drafts.append(draft)
        with pytest.raises(ValueError, match="share an id"):
            session.check()

    def test_a_finalized_session_says_so(self) -> None:
        assert _session().is_finalized is False
        assert _session(finalized_job_id="abc123").is_finalized is True

    def test_the_lookup_by_id_raises_for_an_absent_draft(self) -> None:
        session = _session(drafts=[_draft("draft-one")])
        assert session.draft("draft-one").draft_id == "draft-one"
        with pytest.raises(KeyError):
            session.draft("draft-two")

    def test_a_session_round_trips_whole(self) -> None:
        """Turns, drafts and verdicts, in order, through one document."""
        session = _session(
            turns=[
                Turn(
                    created_at=_NOW,
                    trigger="brief",
                    narration="Drafting two candidates.",
                    calls=(ToolInvocation(name="draft", arguments={"n": 2}, result='["a"]'),),
                    model="a-model",
                    latency_ms=640,
                ),
                Turn(
                    created_at=_NOW,
                    trigger="message",
                    narration="Widening the bass.",
                    calls=(ToolInvocation(name="revise", arguments={"draft": "a"}, result="ok"),),
                ),
            ],
            drafts=[_draft("a", sketch=_SKETCH), _draft("b")],
            verdicts=[Verdict(draft_id="a", at=_NOW, value="like", feedback="keep the intro")],
            finalized_job_id="job-1",
            spec=_SPEC,
        )
        loaded = Session.from_document(json.loads(json.dumps(session.to_document())))
        assert loaded == session
        assert [draft.draft_id for draft in loaded.drafts] == ["a", "b"]
        assert [turn.trigger for turn in loaded.turns] == ["brief", "message"]
        assert loaded.verdicts[0].feedback == "keep the intro"
        assert loaded.draft("a").sketch == _SKETCH
        assert loaded.spec == _SPEC

    def test_a_session_with_no_parsed_brief_yet_round_trips(self) -> None:
        """`spec` is None until `parse_brief` runs, and that is a state a
        document has to be able to hold — omitting the key would read as
        "a spec was considered here", which is a different claim."""
        session = _session()
        assert session.spec is None
        loaded = Session.from_document(json.loads(json.dumps(session.to_document())))
        assert loaded.spec is None

    def test_a_spec_from_a_newer_build_is_refused_on_the_session(self) -> None:
        """The session's own spec is guarded by the reader the drafts use."""
        document = _session(spec=_SPEC).to_document()
        document["spec"]["schema_version"] = SPEC_SCHEMA_VERSION + 1
        with pytest.raises(UnsupportedSpecVersionError, match="this session"):
            Session.from_document(document)

    def test_the_document_names_its_own_shape(self) -> None:
        assert _session().to_document()["format"] == SESSION_FORMAT

    def test_a_foreign_session_document_is_refused(self) -> None:
        """Derived, not spelled out: a literal here is a trap for the next
        bump — it passed until the constant caught up with it."""
        document = _session().to_document()
        document["format"] = f"Session:{SESSION_SCHEMA_VERSION + 1}"
        with pytest.raises(UnsupportedSessionVersionError, match="format"):
            Session.from_document(document)

    def test_a_document_with_no_format_is_refused(self) -> None:
        """A pre-history document is not one this build can complete."""
        document = _session().to_document()
        del document["format"]
        with pytest.raises(UnsupportedSessionVersionError):
            Session.from_document(document)
