"""The session store: what it refuses, what it skips, and what it leaves alone.

Three properties here are the reason the module exists rather than a
one-line `json.dump`:

- A session written to disk is never one this store's own `get` would
  refuse. That is tested by *trying* it — mutate a session into an
  inconsistent state and require `save` to refuse — rather than by trusting
  the check to be called.
- Reading is skipped, not fatal, for a document this build cannot read. One
  corrupt file must not make every other session unreachable.
- Pruning never deletes what it cannot read, which is a deliberate leak and
  is pinned as one so it is not "fixed" into a delete-on-a-guess.

The last class pins the reason the root is separate: the sessions walker
knows nothing but sessions, and the jobs walker knows nothing but jobs, so
neither store's retention governs the other's files even when they happen to
share a directory.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from saimc.compose.engine import compose
from saimc.compose.linter import lint
from saimc.compose.motif import BassMotion
from saimc.compose.plan import default_plan
from saimc.jobs.storage import JobStorage
from saimc.quality import score_piece
from saimc.session.deltas import SetBassMotion
from saimc.session.models import (
    SESSION_FORMAT,
    Draft,
    Publication,
    Session,
    ToolInvocation,
    Turn,
    UnsupportedSessionVersionError,
)
from saimc.session.preferences import Preference, PreferenceLog
from saimc.session.store import (
    HISTORY_DIRNAME,
    SESSION_RETENTION_DAYS,
    UNDO_DEPTH,
    SessionStorage,
    UndoUnavailable,
)
from saimc.spec import CompositionSpec, Mood

_NOW = datetime(2026, 9, 16, 12, 0, tzinfo=UTC)
_SPEC = CompositionSpec(mood=Mood.CALMING, duration_seconds=30, seed=5)
_PLAN = default_plan(_SPEC)
_OUTPUT = compose(_SPEC, plan=_PLAN)


@pytest.fixture
def store(tmp_path):
    return SessionStorage(tmp_path / "sessions")


def _draft(draft_id: str = "draft-one") -> Draft:
    return Draft(
        draft_id=draft_id,
        created_at=_NOW,
        spec=_SPEC,
        plan=_PLAN,
        performance_plan_hash=_OUTPUT.performance_plan.compute_hash(),
        quality=score_piece(_OUTPUT.notation_score, piece=draft_id),
        lint=lint(
            _OUTPUT.notation_score,
            chord_bars=_OUTPUT.chord_bars or None,
            bar_keys=_OUTPUT.bar_keys or None,
            voice_instruments={v.voice_id: v.instrument for v in _OUTPUT.voice_instruments},
        ),
    )


def _turn(narration: str = "Drafting one candidate.") -> Turn:
    return Turn(
        created_at=_NOW,
        trigger="brief",
        narration=narration,
        calls=(ToolInvocation(name="draft", arguments={"n": 1}, result='["draft-one"]'),),
        model="a-model",
        latency_ms=420,
    )


def _document_path(store: SessionStorage, session_id: str):
    return store.session_dir(session_id) / "session.json"


def _corrupt(store: SessionStorage, session_id: str, payload: str) -> None:
    _document_path(store, session_id).write_text(payload)


def _snapshots(store: SessionStorage, session_id: str) -> list[str]:
    """The undo steps a session keeps, oldest first."""
    return sorted(
        path.name for path in (store.session_dir(session_id) / HISTORY_DIRNAME).glob("*.json")
    )


def _snapshot_path(store: SessionStorage, session_id: str, step: int):
    return store.session_dir(session_id) / HISTORY_DIRNAME / f"{step:08d}.json"


class TestCreateAndGet:
    def test_create_persists_immediately(self, store: SessionStorage) -> None:
        session = store.create("something for a rainy day")
        reloaded = store.get(session.session_id)
        assert reloaded.brief == "something for a rainy day"
        assert reloaded.turns == []
        assert reloaded.is_finalized is False

    def test_a_new_session_id_is_an_opaque_hex_token(self, store: SessionStorage) -> None:
        """An opaque token, not a path — the Phase 1 rule the artifacts follow."""
        first, second = store.new_session_id(), store.new_session_id()
        assert first != second
        assert first.isalnum()
        assert "/" not in first

    def test_the_document_carries_the_sessions_own_format_tag(self, store: SessionStorage) -> None:
        session = store.create("p")
        payload = json.loads(_document_path(store, session.session_id).read_text())
        assert payload["format"] == SESSION_FORMAT

    def test_get_unknown_raises_keyerror(self, store: SessionStorage) -> None:
        with pytest.raises(KeyError):
            store.get("does-not-exist")

    def test_a_malformed_id_is_a_different_refusal_from_a_missing_one(
        self, store: SessionStorage
    ) -> None:
        """`ValueError` for an id this store could not have issued, `KeyError`
        for one it could have and did not — the API maps them to different
        statuses, so collapsing them would be a real loss of information."""
        with pytest.raises(ValueError):
            store.get("../escape")
        with pytest.raises(KeyError):
            store.get("0f8a4c2b")

    def test_the_root_is_created_on_construction(self, tmp_path) -> None:
        root = tmp_path / "not-yet"
        assert not root.exists()
        SessionStorage(root)
        assert root.is_dir()

    def test_the_environment_names_the_root(self, tmp_path, monkeypatch) -> None:
        root = tmp_path / "from-env"
        monkeypatch.setenv("SAIMC_SESSIONS_DIR", str(root))
        assert SessionStorage().root == root

    def test_an_explicit_root_beats_the_environment(self, tmp_path, monkeypatch) -> None:
        monkeypatch.setenv("SAIMC_SESSIONS_DIR", str(tmp_path / "from-env"))
        assert SessionStorage(tmp_path / "explicit").root == tmp_path / "explicit"


class TestSave:
    def test_a_whole_session_round_trips_through_the_disk(self, store: SessionStorage) -> None:
        session = store.create("p")
        session.turns.append(_turn())
        session.drafts.append(_draft())
        store.save(session)
        assert store.get(session.session_id) == session

    def test_save_bumps_the_clock(self, store: SessionStorage) -> None:
        """`at` pins it — which is what makes every prune test below possible —
        and the default moves it forward, which is what retention reads."""
        session = store.create("p")
        store.save(session, at=_NOW)
        assert store.get(session.session_id).updated_at == _NOW
        store.save(session)
        assert session.updated_at > _NOW
        assert store.get(session.session_id).updated_at == session.updated_at

    def test_a_refused_save_leaves_the_record_as_it_was(self, store: SessionStorage) -> None:
        """The check runs before the clock moves, on purpose.

        A save that bumped `updated_at` and *then* refused would leave an
        in-memory session claiming an update that never reached the disk.
        """
        session = store.create("p")
        original = session.updated_at
        session.drafts.append(_draft("same"))
        session.drafts.append(_draft("same"))
        with pytest.raises(ValueError, match="share an id"):
            store.save(session, at=_NOW)
        assert session.updated_at == original

    def test_a_refused_save_writes_nothing(self, store: SessionStorage) -> None:
        session = store.create("p")
        session.drafts.append(_draft("same"))
        session.drafts.append(_draft("same"))
        with pytest.raises(ValueError):
            store.save(session)
        assert store.get(session.session_id).drafts == []

    def test_repeated_saves_never_leave_a_torn_document(self, store: SessionStorage) -> None:
        """The tmp+rename idiom: a reader sees the previous document or the
        new one, never half of either."""
        session = store.create("p")
        for index in range(10):
            session.turns.append(_turn(f"turn {index}"))
            store.save(session)
            assert len(store.get(session.session_id).turns) == index + 1


class TestUndo:
    """Undo is a mechanism of the store, not a habit of its callers.

    Every save keeps the document it replaced, so `undo` reaches a state
    nothing had to remember to record. The tests below are about that
    history: that it exists, that it is finite, that walking it does not
    walk *forward*, and that it will not cross a publish.
    """

    def test_undo_restores_the_session_before_the_last_save(self, store: SessionStorage) -> None:
        session = store.create("p")
        session.turns.append(_turn("kept"))
        store.save(session)
        session.turns.append(_turn("undone"))
        store.save(session)
        restored = store.undo(session.session_id)
        assert [turn.narration for turn in restored.turns] == ["kept"]
        assert [turn.narration for turn in store.get(session.session_id).turns] == ["kept"]

    def test_undo_walks_back_one_step_at_a_time(self, store: SessionStorage) -> None:
        session = store.create("p")
        for index in range(3):
            session.turns.append(_turn(f"turn {index}"))
            store.save(session)
        assert [len(store.undo(session.session_id).turns) for _ in range(3)] == [2, 1, 0]
        with pytest.raises(UndoUnavailable, match="no earlier state"):
            store.undo(session.session_id)

    def test_a_second_undo_goes_further_back_rather_than_forward(
        self, store: SessionStorage
    ) -> None:
        """There is no redo, and this is what says so.

        The state an undo leaves is discarded, so undoing twice is two steps
        into the past — not a toggle that returns what it took, which is what
        re-running the rotation would accidentally build.
        """
        session = store.create("p")
        for narration in ("first", "second", "third"):
            session.turns.append(_turn(narration))
            store.save(session)
        assert [turn.narration for turn in store.undo(session.session_id).turns] == [
            "first",
            "second",
        ]
        assert [turn.narration for turn in store.undo(session.session_id).turns] == ["first"]

    def test_undo_is_refused_when_there_is_nothing_behind_the_session(
        self, store: SessionStorage
    ) -> None:
        session = store.create("p")
        with pytest.raises(UndoUnavailable, match="no earlier state"):
            store.undo(session.session_id)

    def test_undo_is_refused_once_the_session_has_published(self, store: SessionStorage) -> None:
        """A queued render cannot be recalled, so the publish is not undoable.

        Restoring the document from before the publish would leave a session
        saying it had never published while the piece it published renders
        on — and the user's next publish would start a second render of the
        same draft. The refusal is checked against the document on disk, and
        it changes nothing.
        """
        session = store.create("p")
        session.turns.append(_turn())
        session.drafts.append(_draft())
        store.save(session)
        session.publication = Publication(job_id="publishedjob", draft_id="draft-one")
        store.save(session)
        with pytest.raises(UndoUnavailable, match="publishedjob"):
            store.undo(session.session_id)
        assert store.get(session.session_id).finalized_job_id == "publishedjob"

    def test_undo_moves_the_clock_so_the_workspace_hears_about_it(
        self, store: SessionStorage
    ) -> None:
        """The restored document is old; the change is now.

        The SSE stream emits on `updated_at`, so an undo that left the old
        timestamp in place would be invisible until the next unrelated save.

        Both saves are pinned, and that is the whole test. An unpinned
        snapshot carries a *recent* timestamp of its own — it was written
        moments ago — so an assertion that the restored clock is "later than
        some past instant" is satisfied by the snapshot rather than by the
        bump, and stays green with the bump deleted.
        """
        session = store.create("p")
        yesterday = _NOW - timedelta(days=1)
        store.save(session, at=yesterday)
        session.turns.append(_turn())
        store.save(session, at=_NOW)
        restored = store.undo(session.session_id)
        assert restored.turns == []
        assert restored.updated_at > _NOW
        assert store.get(session.session_id).updated_at == restored.updated_at

    def test_the_history_is_bounded_by_the_depth(self, store: SessionStorage) -> None:
        session = store.create("p")
        for index in range(UNDO_DEPTH + 5):
            session.turns.append(_turn(f"turn {index}"))
            store.save(session)
        assert len(_snapshots(store, session.session_id)) == UNDO_DEPTH

    def test_the_states_dropped_are_the_oldest_ones(self, store: SessionStorage) -> None:
        """The bound has to keep the near past, which is the one a user
        reaches for. A ring that dropped the *newest* would hold ten steps
        and none of them the one just taken.

        Run past the depth rather than exactly to it, which is also what
        holds the step numbers to the maximum already on disk: a counter
        taken as the *count* would start colliding with a kept snapshot the
        moment pruning removed one, and the walk back would repeat a state
        instead of descending through it.
        """
        session = store.create("p")
        for index in range(UNDO_DEPTH + 3):
            session.turns.append(_turn(f"turn {index}"))
            store.save(session)
        walked = [len(store.undo(session.session_id).turns) for _ in range(UNDO_DEPTH)]
        assert walked == list(range(UNDO_DEPTH + 2, 2, -1))
        with pytest.raises(UndoUnavailable, match="no earlier state"):
            store.undo(session.session_id)

    def test_a_save_that_dies_after_the_rotation_cannot_lose_the_session(
        self, store: SessionStorage, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The rotation copies the document rather than moving it aside.

        `_write` is atomic, so the worst a crash between the two leaves is a
        snapshot of a duplicate. Moving `session.json` into `history/` first
        would leave it missing — an outcome `get` cannot tell apart from a
        session that never existed, which is the one failure this store's
        whole write path is arranged to avoid.
        """
        session = store.create("p")
        session.turns.append(_turn())

        def _die(*args: object, **kwargs: object) -> None:
            raise OSError("the disk filled up")

        monkeypatch.setattr(store, "_write", _die)
        with pytest.raises(OSError):
            store.save(session)
        assert store.get(session.session_id).turns == []

    def test_a_snapshot_is_rebuilt_through_the_constructors(self, store: SessionStorage) -> None:
        """Same rule as `get`: a history file is a document from disk."""
        session = store.create("p")
        session.turns.append(_turn())
        store.save(session)
        _snapshot_path(store, session.session_id, 1).write_text("{}")
        with pytest.raises(UnsupportedSessionVersionError):
            store.undo(session.session_id)

    def test_a_refused_save_leaves_no_snapshot(self, store: SessionStorage) -> None:
        """The rotation is after the check, for the same reason the clock is.

        A save that refused after rotating would offer an undo step into a
        state the save never reached.
        """
        session = store.create("p")
        session.drafts.append(_draft("same"))
        session.drafts.append(_draft("same"))
        with pytest.raises(ValueError):
            store.save(session)
        assert _snapshots(store, session.session_id) == []

    def test_saving_a_session_that_was_never_written_has_nothing_to_rotate(
        self, store: SessionStorage
    ) -> None:
        """`save` takes any `Session`, including one built rather than created.

        There is no document to keep, so the rotation does nothing — and it
        must not make that a failure, which is what reading a file that is
        not there would turn it into.
        """
        session = Session(
            session_id="handbuilt01",
            created_at=_NOW,
            updated_at=_NOW,
            brief="something for a rainy day",
        )
        store.save(session)
        assert store.get("handbuilt01").brief == "something for a rainy day"
        assert _snapshots(store, "handbuilt01") == []

    def test_the_history_is_not_mistaken_for_a_session(self, store: SessionStorage) -> None:
        """`list_all` globs `*/session.json`; a history file is one directory
        deeper and must not appear as a session of its own."""
        session = store.create("p")
        session.turns.append(_turn())
        store.save(session)
        assert [seen.session_id for seen in store.list_all()] == [session.session_id]

    def test_prune_takes_the_history_with_the_session(self, store: SessionStorage) -> None:
        session = store.create("p")
        session.turns.append(_turn())
        store.save(session)
        store.save(session, at=datetime.now(UTC) - timedelta(days=SESSION_RETENTION_DAYS + 1))
        assert store.prune() == 1
        assert not store.session_dir(session.session_id).exists()


class TestListAll:
    def test_list_all_yields_every_session(self, store: SessionStorage) -> None:
        created = {store.create(f"brief {i}").session_id for i in range(3)}
        assert {session.session_id for session in store.list_all()} == created

    def test_an_unreadable_session_does_not_break_the_listing(self, store: SessionStorage) -> None:
        good = store.create("good")
        bad = store.create("bad")
        _corrupt(store, bad.session_id, "{ not json")
        assert [session.session_id for session in store.list_all()] == [good.session_id]

    def test_a_document_from_another_shape_is_skipped_not_raised(
        self, store: SessionStorage
    ) -> None:
        good = store.create("good")
        future = store.create("future")
        payload = json.loads(_document_path(store, future.session_id).read_text())
        payload["format"] = "Session:99"
        _document_path(store, future.session_id).write_text(json.dumps(payload))
        assert [session.session_id for session in store.list_all()] == [good.session_id]

    def test_a_session_a_newer_build_wrote_is_refused_directly(self, store: SessionStorage) -> None:
        """`list_all` skips it, but asking for it by id must say why."""
        session = store.create("p")
        payload = json.loads(_document_path(store, session.session_id).read_text())
        payload["format"] = "Session:99"
        _document_path(store, session.session_id).write_text(json.dumps(payload))
        with pytest.raises(UnsupportedSessionVersionError, match="Session:99"):
            store.get(session.session_id)


class TestPrune:
    def test_prune_removes_a_session_past_its_horizon(self, store: SessionStorage) -> None:
        session = store.create("p")
        store.save(session, at=datetime.now(UTC) - timedelta(days=SESSION_RETENTION_DAYS + 1))
        assert store.prune() == 1
        with pytest.raises(KeyError):
            store.get(session.session_id)

    def test_prune_keeps_a_recent_session(self, store: SessionStorage) -> None:
        session = store.create("p")
        store.save(session)
        assert store.prune() == 0
        store.get(session.session_id)

    def test_prune_removes_the_sketches_with_the_session(self, store: SessionStorage) -> None:
        session = store.create("p")
        sketch = store.sketch_dir(session.session_id, "draft-one")
        (sketch / "audio.wav").write_bytes(b"RIFF")
        store.save(session, at=datetime.now(UTC) - timedelta(days=SESSION_RETENTION_DAYS + 1))
        store.prune()
        assert not store.session_dir(session.session_id).exists()

    def test_prune_never_deletes_what_it_cannot_read(self, store: SessionStorage) -> None:
        """A deliberate, visible leak: the directory stays where a human can
        look at it, and `list_all` skips it for the same reason. Deleting a
        document on the strength of a guess about its age is the worse
        failure of the two."""
        session = store.create("p")
        _corrupt(store, session.session_id, "{ not json")
        assert store.prune(now=datetime.now(UTC) + timedelta(days=3650)) == 0
        assert _document_path(store, session.session_id).exists()


class TestSketchPaths:
    def test_the_sketch_directory_is_created_per_draft(self, store: SessionStorage) -> None:
        """One directory per draft, because a fan-out writes several sketches
        in one turn and they must not overwrite each other's `audio.wav`."""
        session = store.create("p")
        first = store.sketch_dir(session.session_id, "draft-one")
        second = store.sketch_dir(session.session_id, "draft-two")
        assert first.is_dir()
        assert second.is_dir()
        assert first != second
        assert first.parent == second.parent

    def test_a_draft_id_that_is_not_a_segment_is_refused(self, store: SessionStorage) -> None:
        session = store.create("p")
        with pytest.raises(ValueError, match="draft_id"):
            store.sketch_dir(session.session_id, "../escape")

    def test_a_recorded_path_resolves_against_its_session(self, store: SessionStorage) -> None:
        session = store.create("p")
        resolved = store.sketch_path(session.session_id, "sketches/draft-one/audio.ogg")
        assert resolved == store.session_dir(session.session_id) / "sketches/draft-one/audio.ogg"

    def test_a_climbing_path_is_refused_at_the_join(self, store: SessionStorage) -> None:
        """The `SketchRecord` refuses one at construction, so this is the
        second line of defence on the one place a stored string becomes a
        file handle."""
        session = store.create("p")
        with pytest.raises(ValueError, match="outside the session directory"):
            store.sketch_path(session.session_id, "../../etc/passwd")

    def test_a_malformed_session_id_is_refused(self, store: SessionStorage) -> None:
        with pytest.raises(ValueError, match="session_id"):
            store.sketch_path("../escape", "audio.ogg")


class TestTheTwoRootsKnowNothingOfEachOther:
    """Sharing a root is the failure the separate default root avoids.

    `JobStorage.list_all` globs `*/job.json` and `JobStorage.prune` iterates
    only what it lists, so a session under the jobs root would be silently
    never pruned — no error, just files that outlive every retention
    horizon. Pointing both stores at one directory here pins the behaviour
    that makes the separate root worth having: each walker sees only its own
    document shape, whatever else happens to be in the directory.
    """

    def test_a_job_directory_is_not_a_session(self, tmp_path) -> None:
        root = tmp_path / "shared"
        jobs = JobStorage(root)
        sessions = SessionStorage(root)
        jobs.create("a one-shot request")
        session = sessions.create("an interactive one")
        assert [s.session_id for s in sessions.list_all()] == [session.session_id]

    def test_a_session_prune_leaves_a_job_alone(self, tmp_path) -> None:
        root = tmp_path / "shared"
        jobs = JobStorage(root)
        sessions = SessionStorage(root)
        job = jobs.create("p")
        session = sessions.create("p")
        sessions.save(session, at=datetime.now(UTC) - timedelta(days=SESSION_RETENTION_DAYS + 1))
        assert sessions.prune() == 1
        assert jobs.get(job.job_id).job_id == job.job_id

    def test_a_job_prune_leaves_a_session_alone(self, tmp_path) -> None:
        root = tmp_path / "shared"
        jobs = JobStorage(root)
        sessions = SessionStorage(root)
        job = jobs.create("p")
        session = sessions.create("p")
        jobs.save(job, at=datetime.now(UTC) - timedelta(days=30))
        jobs.prune()
        assert sessions.get(session.session_id).session_id == session.session_id

    def test_a_preference_log_in_a_shared_root_is_not_a_session(self, tmp_path) -> None:
        """The third member of the family, and the one retention must never reach.

        The log has a root of its own precisely because it is not pruned — but a
        deployment that ignored that and pointed it at the sessions root must
        still not have its judgements read as sessions or deleted with them. Both
        halves are asserted because they fail differently: the listing is about a
        glob, and the prune is about the bytes.
        """
        root = tmp_path / "shared"
        sessions = SessionStorage(root)
        log = PreferenceLog(root)
        session = sessions.create("p")
        assert log.append(
            [
                Preference(
                    draft_id="draft-one",
                    at=_NOW,
                    plan_hash="0" * 64,
                    delta=SetBassMotion(motion=BassMotion.SPARSE),
                    verdict="like",
                )
            ]
        ) == 1

        assert [s.session_id for s in sessions.list_all()] == [session.session_id]
        before = log.path.read_bytes()
        sessions.save(session, at=datetime.now(UTC) - timedelta(days=SESSION_RETENTION_DAYS + 1))
        assert sessions.prune() == 1
        assert log.path.read_bytes() == before
        assert len(log.rows()) == 1
