"""The preference log: the rows, and the file they live in.

The rows used to be a field of the session document, and these cases were
written about them there. They are here now because the *subject* moved: a row
has to outlive the session it judges, and a field of the session is deleted with
it at `SESSION_RETENTION_DAYS`. So the row was never really a part of that
document — it was a dataset row that happened to be stored inside a record with
a shorter life than the datum.

Two sets of cases, and the difference matters:

- **The rows follow the chain.** A verdict writes one row per request the judged
  draft's chain holds, each naming the *step's* reader rather than the judged
  draft's, because one chain can be built by two of them. Ten cases, rewritten
  against `preference_rows` rather than re-thought: the walk is the same walk.
- **The log is its own document.** A row round-trips, an absent file reads as
  empty, a line that is not a row refuses the whole read naming the line. Seven
  cases that were claims about a session document's `preferences` key, and are
  now claims about a JSONL file — a reshape rather than a move, since the claim
  is about a container the session no longer has.

The fixtures compose for real, like the models' own: a row carrying a plan the
engine could not have produced would make the chain cases assert about a
document no tool can write.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from saimc.canonical import canonical_dumps
from saimc.compose.engine import EngineOutput, compose
from saimc.compose.linter import LintReport, lint
from saimc.compose.motif import BassMotion
from saimc.compose.plan import default_plan
from saimc.quality import score_piece
from saimc.session.deltas import (
    Delta,
    RequestSource,
    SetBassMotion,
    SetTimeSignature,
    apply_deltas,
    delta_to_dict,
)
from saimc.session.models import Draft, Session, Verdict, VerdictValue
from saimc.session.preferences import (
    PREFERENCES_FILENAME,
    Preference,
    PreferenceLog,
    preference_rows,
)
from saimc.session.store import SESSION_RETENTION_DAYS, SessionStorage
from saimc.session.tools import _draft_from
from saimc.spec import CompositionSpec, Mood, TimeSignature

_NOW = datetime(2026, 9, 16, 12, 0, tzinfo=UTC)
_SPEC = CompositionSpec(mood=Mood.CALMING, duration_seconds=30, seed=5)
_PLAN = default_plan(_SPEC)
_OUTPUT = compose(_SPEC, plan=_PLAN)


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
    )


def _revision(
    deltas: tuple[Delta, ...],
    *,
    parent_id: str = "draft-one",
    draft_id: str = "draft-two",
    source: RequestSource | None = "conductor",
) -> Draft:
    """A real revision: the chain folded from the root, then composed under it."""
    application = apply_deltas(_SPEC, deltas)
    output = compose(application.spec, plan=application.plan)
    return _draft_from(
        draft_id,
        application.spec,
        output,
        parent_id=parent_id,
        deltas=application.applied,
        requests_source=source,
    )


def _session(*, drafts: list[Draft] | None = None) -> Session:
    return Session(
        session_id="session-one",
        created_at=_NOW,
        updated_at=_NOW,
        brief="something for a rainy day",
        drafts=[] if drafts is None else drafts,
    )


def _judged(
    session: Session,
    draft_id: str,
    *,
    value: VerdictValue | None = "like",
    feedback: str | None = None,
) -> tuple[Preference, ...]:
    """Record a verdict on `session` and return the rows it implies.

    The two calls the endpoint makes, in the endpoint's order, so a case here
    cannot pass while the pair it stands for is written some other way.
    """
    verdict = Verdict(draft_id=draft_id, at=_NOW, value=value, feedback=feedback)
    session.record_verdict(verdict)
    return preference_rows(session, verdict)


def _rewrite_first_row(log: PreferenceLog, **changes: object) -> None:
    """Hand-edit the first line of the log, the way a torn file reaches a reader.

    Written back with `json.dumps` rather than the canonical writer on purpose:
    the reader parses a file an operator may have edited, so a case about what
    it refuses must not depend on how the editor spelled the bytes.
    """
    lines = log.path.read_text(encoding="utf-8").splitlines()
    document = json.loads(lines[0])
    document.update(changes)
    lines[0] = json.dumps(document)
    log.path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def log(tmp_path: Path) -> PreferenceLog:
    return PreferenceLog(tmp_path / "preferences")


class TestTheRowsFollowTheChain:
    """Every verdict, read against the chain of the draft it judged.

    A row is `(plan_hash, delta, verdict)` — what was asked for, in which piece,
    and how it was received — and the last of the three is the only part no
    threshold can supply. These cases are about the row being the *whole* datum,
    which means naming the step that asked for each request rather than the
    reader that touched it last.
    """

    _FIRST = (SetBassMotion(motion=BassMotion.SPARSE),)
    _SECOND = (SetTimeSignature(TimeSignature.THREE_FOUR),)

    def _chain(self) -> list[Draft]:
        """A root, a typed revision of it, and a model-read revision of that.

        Two different readers on one chain on purpose: it is the case a log
        labelling every row with the last reader would get wrong, and the step
        each request belongs to is the only place the right answer lives.
        """
        return [
            _draft("draft-one"),
            _revision(self._FIRST, source="typed"),
            _revision(
                (*self._FIRST, *self._SECOND),
                parent_id="draft-two",
                draft_id="draft-three",
                source="model",
            ),
        ]

    def test_one_row_is_written_for_every_request_in_the_chain(self) -> None:
        rows = _judged(_session(drafts=self._chain()), "draft-three")

        assert [(row.delta, row.requests_source) for row in rows] == [
            (self._FIRST[0], "typed"),
            (self._SECOND[0], "model"),
        ]

    def test_the_case_can_tell_the_two_readers_apart(self) -> None:
        """The premise the ordering above rests on, asserted rather than believed.

        Why the pairing is a walk: one chain, two readers, two rows. A fixture
        whose steps shared a reader would let the case above pass while saying
        nothing about the walk at all — so the judged draft's own source is
        asserted to be the one *not* on the first row.
        """
        chain = self._chain()
        rows = _judged(_session(drafts=chain), "draft-three")

        assert chain[-1].requests_source == "model"
        assert rows[0].requests_source != chain[-1].requests_source

    def test_every_row_names_the_piece_the_verdict_was_cast_on(self) -> None:
        """A chain's requests are all changes to the one piece the user heard."""
        chain = self._chain()
        rows = _judged(_session(drafts=chain), "draft-three")

        assert {row.plan_hash for row in rows} == {chain[-1].plan_hash}
        assert {row.draft_id for row in rows} == {"draft-three"}
        assert {row.verdict for row in rows} == {"like"}
        assert {row.at for row in rows} == {_NOW}

    def test_the_rows_are_in_the_order_the_requests_were_added(self) -> None:
        """The chain's order, so a log read in file order is a log in time order."""
        rows = _judged(_session(drafts=self._chain()), "draft-three")

        assert [row.delta for row in rows] == [*self._FIRST, *self._SECOND]

    def test_the_verdict_is_recorded_beside_the_rows(self) -> None:
        """Two records of one judgement, written by two calls in one order.

        The session keeps the verdict and the log keeps the rows, because only
        one of the two has to survive the session — which is why this is two
        calls rather than one method owning both.
        """
        session = _session(drafts=self._chain())
        _judged(session, "draft-three")

        assert [(verdict.draft_id, verdict.value) for verdict in session.verdicts] == [
            ("draft-three", "like")
        ]

    def test_a_piece_drafted_from_the_brief_has_no_request_to_weigh(self) -> None:
        """Liking it is a judgement about the spec; there is no change to attribute."""
        session = _session(drafts=self._chain())
        rows = _judged(session, "draft-one")

        assert rows == ()
        assert len(session.verdicts) == 1

    def test_a_verdict_in_words_alone_writes_no_row(self) -> None:
        """Words are the most useful thing the product gets and are not yet a preference.

        Nothing here knows whether "the bass is muddy" is praise, and a row
        claiming one would be this module inventing a judgement the user made in
        a vocabulary it cannot read.
        """
        session = _session(drafts=self._chain())
        rows = _judged(session, "draft-three", value=None, feedback="the bass is muddy")

        assert rows == ()
        assert session.verdicts[0].feedback == "the bass is muddy"

    def test_a_chain_that_does_not_line_up_places_what_it_can(self) -> None:
        """A posed draft, not a written one: the constructor cannot see a prefix.

        `deltas` and `parent_id` are checked together on one record, and whether
        a parent's chain is a prefix of its child's is a fact about *two*. So a
        session can hold a line that does not line up, and it is reached by
        posing one — the same way a hand-edited document reaches it in the wild.
        The requests the walk cannot place carry no source rather than a guess.
        """
        session = _session(
            drafts=[
                _draft("draft-one"),
                _draft("draft-two", parent_id="draft-one", deltas=self._SECOND),
                _draft(
                    "draft-three",
                    parent_id="draft-two",
                    deltas=self._FIRST,
                    requests_source="typed",
                ),
            ]
        )
        rows = _judged(session, "draft-three")

        assert [(row.delta, row.requests_source) for row in rows] == [(self._FIRST[0], None)]

    def test_a_cycle_in_the_lineage_reports_what_it_cannot_place(self) -> None:
        """Two drafts naming each other, which no tool writes and a file can.

        The walk terminates, reports each request once, and pairs the request it
        cannot place with no source rather than with a guess. The terminator here
        is the prefix check rather than the record of what has been seen — this
        chain's second step is not an extension of its parent — so the guard is
        witnessed by the case below, and this one is about the answer being
        honest rather than about which line produced it.
        """
        session = _session(
            drafts=[
                _draft("draft-one"),
                _draft("draft-two", parent_id="draft-three", deltas=self._FIRST),
                _draft(
                    "draft-three",
                    parent_id="draft-two",
                    deltas=(*self._FIRST, *self._SECOND),
                    requests_source="typed",
                ),
            ]
        )
        rows = _judged(session, "draft-three")

        assert [(row.delta, row.requests_source) for row in rows] == [
            (self._FIRST[0], None),
            (self._SECOND[0], "typed"),
        ]

    def test_the_walk_is_ended_by_the_drafts_it_has_seen_and_not_by_their_shape(self) -> None:
        """The cycle the prefix check cannot catch, which is why the guard is there.

        A loop whose chains each extend their parent's evenly has every prefix
        check pass on the way round — the closure of a loop of extensions is
        equality — so nothing but the set of what the walk has visited ends it.
        Fired as a sabotage: deleting that guard hangs this case, and the case
        above cannot see the deletion at all, which is the whole reason the two
        are separate.
        """
        chain = (*self._FIRST, *self._SECOND)
        session = _session(
            drafts=[
                _draft("draft-one"),
                _draft("draft-two", parent_id="draft-three", deltas=chain),
                _draft(
                    "draft-three",
                    parent_id="draft-two",
                    deltas=chain,
                    requests_source="typed",
                ),
            ]
        )
        rows = _judged(session, "draft-three")

        assert [(row.delta, row.requests_source) for row in rows] == [
            (self._FIRST[0], None),
            (self._SECOND[0], None),
        ]


class TestTheLogIsItsOwnDocument:
    """The rows' container, now a JSONL file at its own root.

    These were claims about `document["preferences"]` — a key of the session.
    The session has no such key any more, and the claims are not the less true
    for it: a row round-trips, an absent container reads as empty, a malformed
    row is refused at load. What changed is which document is making them, which
    is why each is rewritten rather than repointed.
    """

    _CHAIN = (SetBassMotion(motion=BassMotion.SPARSE),)

    def _written(self, log: PreferenceLog) -> tuple[Preference, ...]:
        """Two rows on one chain, in the log, returned as written."""
        session = _session(
            drafts=[
                _draft("draft-one"),
                _revision(self._CHAIN, source="typed"),
            ]
        )
        rows = _judged(session, "draft-two")
        assert log.append(rows) == len(rows)
        return rows

    def test_a_row_is_written_down_and_read_back(self, log: PreferenceLog) -> None:
        rows = self._written(log)

        assert log.path.read_text(encoding="utf-8").splitlines() == [
            canonical_dumps(row.to_document()) for row in rows
        ]
        assert log.rows() == rows

    def test_the_file_is_named_and_the_root_is_a_directory_of_its_own(
        self, log: PreferenceLog, tmp_path: Path
    ) -> None:
        """One file, addressed by name, at a root that is nobody else's."""
        assert log.root == tmp_path / "preferences"
        assert log.path == log.root / PREFERENCES_FILENAME

    def test_an_absent_log_reads_as_empty(self, log: PreferenceLog) -> None:
        """Nothing has been recorded, which is the honest reading of a file
        that was never created — and not the same as a log that is there and
        empty, which is why neither is created by a read."""
        assert log.rows() == ()
        assert not log.path.exists()

    def test_appending_nothing_creates_no_file(self, log: PreferenceLog) -> None:
        """The ordinary case of a piece drafted from the brief, which has no
        request to weigh: a run that recorded no preference leaves no trace to
        notice, rather than an empty file that reads as "something happened"."""
        assert log.append(()) == 0
        assert log.rows() == ()
        assert not log.root.exists()

    def test_a_row_outliving_the_draft_it_judges_is_read_back(self, log: PreferenceLog) -> None:
        """The asymmetry with `verdicts`, and the whole reason the row is stored.

        A verdict naming a draft the session does not have is a broken record,
        and `Session.check()` refuses it. A row naming one is the judgement kept
        after the pruning that would take the draft away — and it now reads back
        *because the log never consults a session at all*. That is the stronger
        form of the old claim: the reader cannot refuse it even by mistake,
        since there is no session in reach for it to check against.
        """
        rows = self._written(log)

        assert {row.draft_id for row in rows} == {"draft-two"}
        assert log.rows() == rows

    def test_a_row_whose_delta_is_stored_as_it_is_written(self, log: PreferenceLog) -> None:
        """The delta goes in through `delta_to_dict`, the form a draft's chain uses."""
        rows = self._written(log)

        assert json.loads(log.path.read_text(encoding="utf-8").splitlines()[0])["delta"] == (
            delta_to_dict(self._CHAIN[0])
        )
        assert rows[0].delta == self._CHAIN[0]

    def test_a_row_whose_verdict_is_not_a_verdict_is_refused_on_load(
        self, log: PreferenceLog
    ) -> None:
        self._written(log)
        _rewrite_first_row(log, verdict="shrug")

        with pytest.raises(ValueError, match="preference verdict"):
            log.rows()

    def test_a_row_whose_source_is_outside_the_vocabulary_is_refused_on_load(
        self, log: PreferenceLog
    ) -> None:
        self._written(log)
        _rewrite_first_row(log, requests_source="whispered")

        with pytest.raises(ValueError, match="requests_source"):
            log.rows()

    def test_a_row_whose_delta_is_not_a_request_is_refused_on_load(self, log: PreferenceLog) -> None:
        self._written(log)
        _rewrite_first_row(log, delta={"knob": "SetSaxophone"})

        with pytest.raises(ValueError, match="SetSaxophone"):
            log.rows()

    def test_a_line_that_is_not_a_row_names_the_line_it_is_on(self, log: PreferenceLog) -> None:
        """A torn line is a thing an operator can look at, so the refusal says
        where — and it refuses the *whole* read rather than skipping the line,
        because a preference rate over the rows a reader happened to understand
        is a number describing nobody."""
        self._written(log)
        lines = log.path.read_text(encoding="utf-8").splitlines()
        lines.insert(1, '{"draft_id": "draft-two"')
        log.path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        with pytest.raises(ValueError, match=r"line 2 is not a preference row"):
            log.rows()

    def test_a_line_holding_something_that_is_not_an_object_is_refused(
        self, log: PreferenceLog
    ) -> None:
        """A JSON array is a well-formed line and not a row, and the difference
        between "unparseable" and "the wrong shape" is a sentence, not a crash."""
        self._written(log)
        _rewrite_first_row(log, delta=[])
        with pytest.raises(ValueError, match="not a preference row"):
            log.rows()

        log.path.write_text('["a row"]\n', encoding="utf-8")
        with pytest.raises(ValueError, match="expected a JSON object"):
            log.rows()

    def test_the_log_takes_its_root_from_the_environment(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The shape the other two stores use, so a deployment has one place per
        store that says where its state lives."""
        elsewhere = tmp_path / "elsewhere"
        monkeypatch.setenv("SAIMC_PREFERENCES_DIR", str(elsewhere))

        assert PreferenceLog().root == elsewhere
        assert PreferenceLog(tmp_path / "explicit").root == tmp_path / "explicit"


class TestTheLogOutlivesTheSession:
    """The reason the module exists, as behaviour rather than as a docstring.

    A session directory is deleted at `SESSION_RETENTION_DAYS` and its drafts go
    with it. Every row describes a chain that lives on exactly those drafts, so
    a log kept inside the session dies with the thing it is a record of. These
    two cases are the twin of `test_a_session_prune_leaves_a_job_alone`: a
    retention walker in one root, and a record in another that does not move.
    """

    def _pruned_away(self, tmp_path: Path, log: PreferenceLog) -> tuple[bytes, Path]:
        """Write a judged session, age it past the horizon, prune, and hand back
        the log's bytes as they were before the prune with the directory that
        the prune took."""
        store = SessionStorage(tmp_path / "sessions")
        session = store.create("something for a rainy day")
        session.drafts = [
            _draft("draft-one"),
            _revision((SetBassMotion(motion=BassMotion.SPARSE),)),
        ]
        rows = _judged(session, "draft-two")
        store.save(session, at=datetime.now(UTC) - timedelta(days=SESSION_RETENTION_DAYS + 1))
        assert log.append(rows) == len(rows)

        before = log.path.read_bytes()
        assert store.prune() == 1
        assert list(store.list_all()) == []
        return before, store.session_dir(session.session_id)

    def test_a_session_prune_leaves_the_log_byte_identical(
        self, tmp_path: Path, log: PreferenceLog
    ) -> None:
        before, _ = self._pruned_away(tmp_path, log)

        assert log.path.read_bytes() == before

    def test_the_rows_survive_the_session_that_wrote_them(
        self, tmp_path: Path, log: PreferenceLog
    ) -> None:
        """The docstring's claim, finally true: the drafts are gone, so the
        chain no longer replays and the verdict no longer resolves — and the
        rows are still readable, which is the whole of what they are for."""
        _, session_dir = self._pruned_away(tmp_path, log)

        assert not session_dir.exists()
        rows = log.rows()
        assert [row.draft_id for row in rows] == ["draft-two"]
        assert [row.verdict for row in rows] == ["like"]


def test_the_module_exports_what_it_claims() -> None:
    """A ratchet on the log's public surface, in every other module's shape."""
    from saimc.session import preferences

    assert set(preferences.__all__) == {
        "DEFAULT_PREFERENCES_DIR",
        "PREFERENCES_FILENAME",
        "Preference",
        "PreferenceLog",
        "preference_rows",
    }


def test_a_row_refuses_a_traversing_draft_id() -> None:
    """A draft id is a path segment everywhere else in this package, and this
    record is read back through its constructor — so the refusal lands on load
    as well as on construction, without a reader having to remember it."""
    with pytest.raises(ValueError, match="draft_id"):
        Preference(
            draft_id="../escape",
            at=_NOW,
            plan_hash="0" * 64,
            delta=SetBassMotion(motion=BassMotion.SPARSE),
            verdict="like",
        )
