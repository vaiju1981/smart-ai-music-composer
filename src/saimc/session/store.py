"""Session storage — one JSON document per session, under its own root.

Root: `SAIMC_SESSIONS_DIR`, default `./var/sessions`. Deliberately **not**
under the jobs root, for a reason the job storage states in its own code:
`JobStorage.list_all` globs `*/job.json` (`storage.py:174`) and `prune`
iterates only `list_all` (`:190`), so a session directory written under the
jobs root would be *silently never pruned* — no error, no warning, just
files that outlive every retention horizon — and `_sweep_interrupted_jobs`
(`worker.py:110`), the only crash recovery the worker has, knows nothing but
jobs. A separate root makes the two lifecycles independent by construction
rather than by a special case in each walker.

Layout, mirroring `JobStorage`::

    {root}/
        {session_id}/
            session.json
            sketches/
                {draft_id}/
                    {draft_id}.mid
                    audio.wav
                    audio.ogg

The session document is written atomically (tmp + rename), so a crash
mid-write leaves the previous document rather than half of a new one. Every
read goes through the models' constructors, so a `Session` loaded from disk
has had the same invariants applied as one built in memory — uniqueness of
draft ids, verdicts that name a draft the session has, ids that are single
path segments. A hand-edited or truncated document is refused there rather
than silently loaded with a hole in it.

Retention is per session, and a sketch is pruned with the session that owns
it. Sessions are kept longer than jobs are: a job is a render, which its
artifacts outlive, while a session holds the user's own work and the
`(plan, verdict)` pairs the slow loop will learn from. Deleting that after a
week would discard input that cannot be regenerated.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from saimc.canonical import canonical_dumps
from saimc.session.models import Session, require_id_segment

logger = logging.getLogger(__name__)

DEFAULT_SESSIONS_DIR = Path("./var/sessions")
"""Default location for session documents. Override via `SAIMC_SESSIONS_DIR`."""

SESSION_RETENTION_DAYS = 30
"""How long a session survives without being touched.

Longer than a job's 7 days on purpose: a job is a render whose artifacts
outlive it, while a session is the user's working state and the raw material
of the slow loop.
"""

SKETCHES_DIRNAME = "sketches"


class SessionStorage:
    """Synchronous JSON-file backend for Session records.

    Single-user and local, like `JobStorage`: no locking, no owner, no
    multi-tenant addressing. A session is addressed by its id, which is an
    opaque token — the same Phase 1 rule the job artifacts follow (§7).
    """

    def __init__(self, root: Path | str | None = None) -> None:
        env_root = os.environ.get("SAIMC_SESSIONS_DIR")
        if root is None:
            root = env_root or DEFAULT_SESSIONS_DIR
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def new_session_id(self) -> str:
        return uuid.uuid4().hex

    def create(self, brief: str) -> Session:
        """Create a fresh session for `brief`. Persists immediately."""
        now = datetime.now(UTC)
        session = Session(
            session_id=self.new_session_id(),
            created_at=now,
            updated_at=now,
            brief=brief,
        )
        self._write(session)
        return session

    def get(self, session_id: str) -> Session:
        """Load a session by id.

        Raises `KeyError` when there is no such session, and `ValueError`
        when the id is not one this store could have issued — a missing
        session and a malformed id are different situations, and the caller
        that maps them to a status code needs to tell them apart.
        """
        require_id_segment(session_id, label="session_id")
        path = self._session_path(session_id)
        if not path.exists():
            raise KeyError(session_id)
        return self._read(path)

    def save(self, session: Session, *, at: datetime | None = None) -> None:
        """Persist the session, bumping `updated_at` unless `at` is given.

        The consistency check comes first, and it is here rather than in the
        writer because this is the only path that persists a session which
        has been mutated since it was constructed: a session that gained a
        duplicate draft id would otherwise be written happily and then
        refused by this store's own `get`, because every read rebuilds
        through the constructors. Checking before the clock moves also means
        a refused save leaves the in-memory record exactly as it was.
        """
        session.check()
        session.updated_at = at or datetime.now(UTC)
        self._write(session)

    def list_all(self) -> Iterator[Session]:
        """Iterate over every persisted session, oldest path first.

        A session that fails to deserialize (a newer format, a truncated
        document, a draft naming a plan this build cannot read) is skipped
        rather than breaking the whole listing — the same tolerance
        `JobStorage.list_all` shows, and for the same reason: one bad file
        must not make the rest unreachable.
        """
        for path in sorted(self._root.glob("*/session.json")):
            try:
                yield self._read(path)
            except Exception:
                logger.warning("skipping unreadable session file: %s", path, exc_info=True)

    def prune(self, now: datetime | None = None) -> int:
        """Delete sessions past their retention horizon. Returns count deleted.

        A session that cannot be deserialized is left alone: pruning never
        deletes what it cannot read. That is a leak, deliberate and visible
        — the directory stays where a human can inspect it, and it is
        skipped for exactly the reason `list_all` skips it. Deleting a
        document on the strength of a guess about its age would be the
        worse failure of the two.
        """
        now = now or datetime.now(UTC)
        deleted = 0
        for session in list(self.list_all()):
            age = (now - session.updated_at).total_seconds()
            if age > SESSION_RETENTION_DAYS * 24 * 3600:
                self._delete(session.session_id)
                deleted += 1
        return deleted

    def session_dir(self, session_id: str) -> Path:
        require_id_segment(session_id, label="session_id")
        return self._root / session_id

    def sketch_dir(self, session_id: str, draft_id: str) -> Path:
        """The per-draft directory a sketch is written into, created if absent.

        The renderer is handed this as its `out_dir` and the draft id as its
        `job_id`, which is what gives each draft's sketch a directory of its
        own: a fan-out writes several in one turn, and they must not
        overwrite each other's `audio.wav`.
        """
        require_id_segment(draft_id, label="draft_id")
        path = self.session_dir(session_id) / SKETCHES_DIRNAME / draft_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def sketch_path(self, session_id: str, relative_path: str) -> Path:
        """Resolve a sketch path recorded on a draft against its session.

        The path was validated when the `SketchRecord` was constructed (and
        therefore on load), so this is a join rather than a check — but it
        re-checks containment anyway, because this is the one place a stored
        string becomes a file handle and the cost of being wrong is reading
        a file the session does not own.
        """
        base = self.session_dir(session_id)
        resolved = (base / relative_path).resolve()
        if not resolved.is_relative_to(base.resolve()):
            raise ValueError(
                f"sketch path {relative_path!r} resolves outside the session directory"
            )
        return resolved

    def _session_path(self, session_id: str) -> Path:
        return self._root / session_id / "session.json"

    def _write(self, session: Session) -> None:
        path = self._session_path(session.session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = session.to_document()
        # Atomic write: tmp + rename, so a crash mid-write doesn't leave a half-file.
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=str(path.parent),
            delete=False,
            prefix=".session-",
            suffix=".tmp",
        ) as fh:
            tmp_name = fh.name
            fh.write(canonical_dumps(payload))
            fh.write("\n")
        os.replace(tmp_name, path)

    def _read(self, path: Path) -> Session:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.loads(fh.read())
        return Session.from_document(payload)

    def _delete(self, session_id: str) -> None:
        import shutil

        shutil.rmtree(self._root / session_id, ignore_errors=True)


__all__ = [
    "DEFAULT_SESSIONS_DIR",
    "SESSION_RETENTION_DAYS",
    "SKETCHES_DIRNAME",
    "SessionStorage",
]
