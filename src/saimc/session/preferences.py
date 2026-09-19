"""The preference log — an append-only file of `(plan_hash, delta, verdict)` rows.

Each row is one accepted request, read against the chain of the draft it
produced, together with the verdict the user gave that draft. It is the row of
a preference dataset, and the verdict is the only part of it no threshold can
supply.

**It lives outside the session, and that is the whole point.** A session
directory is deleted at `SESSION_RETENTION_DAYS` and its drafts go with it —
while a row describes a chain that lives on exactly those drafts. A log that
has to outlive the session it judges cannot be *inside* it, so it has a root of
its own (`SAIMC_PREFERENCES_DIR`, default `./var/preferences`) for the reason
`store.py` gives for the sessions root not being under the jobs root: a
lifecycle that is independent by construction beats a special case in each
walker. Nothing prunes this file, ever — it is the one thing here that cannot be
regenerated from anything else.

The file is JSONL, append-only, one row per line and flushed as it is written,
so an interrupted run keeps what it wrote and a reader can read a prefix. Rows
are never rewritten: `undo` rewinds a session, and the log is not part of what
it rewinds, because an opinion the user gave is not unmade by taking back the
edit they gave it about.

A line that is not a row is refused rather than skipped, and the refusal names
the line. That is the opposite of `SessionStorage.list_all`'s tolerance, and
deliberately so: a listing that skips a document it cannot read is still a
listing, while a preference rate computed over the rows a reader happened to
understand is a number describing nobody. The refusal is a `ValueError` naming
the path and the line number, because a torn line is a thing an operator can
look at and fix.

`Session` no longer holds these rows. That is the point of the module: one home
for the log means there is no second copy of a judgement that can disagree with
it, which is the rule every other stored value here follows.

Known limit, stated and not guarded: `at` is generated server-side, so a
duplicate `POST /verdict` is indistinguishable from a second intentional
verdict. The log is therefore at-least-once. An idempotency key would be new
surface for a hazard no caller has.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Final

from saimc.canonical import canonical_dumps
from saimc.session.deltas import Delta, RequestSource, delta_from_dict, delta_to_dict
from saimc.session.models import (
    REQUEST_SOURCES,
    VERDICT_VALUES,
    Draft,
    Session,
    Verdict,
    VerdictValue,
    one_of,
    require_id_segment,
)

DEFAULT_PREFERENCES_DIR = Path("./var/preferences")
"""Default directory for the log. Override via `SAIMC_PREFERENCES_DIR`."""

PREFERENCES_FILENAME: Final[str] = "preferences.jsonl"
"""The one file. A single append-only file rather than a file per session,
because the log is read as a corpus and a corpus is one thing to hand a
builder — and because the rows are meant to outlive whatever wrote them."""


@dataclass(frozen=True)
class Preference:
    """One accepted request, and what the listener thought of the piece it made.

    The row of a preference dataset: `(plan_hash, delta, verdict)` says what was
    asked for, in which piece, and how the result was received — and the last of
    those is the only part a threshold cannot supply. The chain it describes
    lives on drafts, and drafts are what a session prunes, which is why the rows
    are written down rather than derived. `retune.proposal` reads them and
    `saimc-preferences` prints what it finds; nothing edits the repair table from
    them, deliberately, so a row is a measurement rather than a setting.

    `requests_source` is the *step's* reader rather than the judged draft's,
    because one chain can be built by two of them — a slider dragged and then a
    sentence typed — and a row labelling both with the last one would record a
    preference the user never expressed. `None` means the step's reader was not
    recorded, which is what it means on a draft and covers both a document
    written before that field existed and a chain this build cannot place.

    No check requires the draft to exist, unlike a verdict: a row outliving the
    piece it judges is the point of writing it down.
    """

    draft_id: str
    at: datetime
    plan_hash: str
    delta: Delta
    verdict: VerdictValue
    requests_source: RequestSource | None = None

    def __post_init__(self) -> None:
        require_id_segment(self.draft_id, label="draft_id")

    def to_document(self) -> dict[str, Any]:
        return {
            "draft_id": self.draft_id,
            "at": self.at.isoformat(),
            "plan_hash": self.plan_hash,
            "delta": delta_to_dict(self.delta),
            "verdict": self.verdict,
            "requests_source": self.requests_source,
        }

    @classmethod
    def from_document(cls, payload: dict[str, Any]) -> Preference:
        raw_source = payload.get("requests_source")
        return cls(
            draft_id=payload["draft_id"],
            at=datetime.fromisoformat(payload["at"]),
            plan_hash=payload["plan_hash"],
            delta=delta_from_dict(_delta_document(payload["delta"])),
            verdict=one_of(payload["verdict"], VERDICT_VALUES, field_name="preference verdict"),
            requests_source=(
                None
                if raw_source is None
                else one_of(raw_source, REQUEST_SOURCES, field_name="requests_source")
            ),
        )


def _delta_document(raw: Any) -> dict[str, Any]:
    """The `delta` field, which a hand-edited line can hold anything in.

    `delta_from_dict` reads a mapping, so a line holding a list or a string
    there would fail inside it with an `AttributeError` — a traceback where this
    reader's whole promise is a `ValueError` naming the line. Refusing the shape
    here rather than catching the attribute error below keeps the refusal
    specific: nothing else in `from_document` is allowed to raise one.
    """
    if not isinstance(raw, dict):
        raise ValueError(f"delta must be a request object, got {type(raw).__name__}")
    return raw


def preference_rows(session: Session, verdict: Verdict) -> tuple[Preference, ...]:
    """The rows `verdict` writes, read against the draft it judged.

    A verdict carrying no like and no dislike writes no row. Words are the most
    useful thing the product gets and they are not by themselves a preference —
    nothing here knows whether they are praise — so they are recorded on the
    verdict for the reader that can tell, and the log keeps the judgements that
    are already unambiguous.

    The draft's own `plan_hash` is the piece every row names: a chain's requests
    are all changes to the one piece the user heard, and the requests are what
    tell one row from another.

    Raises `KeyError` for a draft the session does not have, like every other
    reader of a draft id — the verdict names something that has to be there.

    Separate from `Session.record_verdict`, which records the verdict and
    nothing else, because this is the one half of that act which needs the
    session's *drafts* rather than its log: the two are recorded in different
    places now, so no single method can own both without this module's import
    running back into `models`.
    """
    if verdict.value is None:
        return ()
    draft = session.draft(verdict.draft_id)
    return tuple(
        Preference(
            draft_id=draft.draft_id,
            at=verdict.at,
            plan_hash=draft.plan_hash,
            delta=delta,
            verdict=verdict.value,
            requests_source=source,
        )
        for delta, source in _request_origins(session, draft)
    )


def _request_origins(
    session: Session, draft: Draft
) -> tuple[tuple[Delta, RequestSource | None], ...]:
    """Every request in `draft`'s chain, paired with the step that asked for it.

    A chain is its steps folded in order, so a request's step is the draft in
    the `parent_id` walk whose own `deltas` first held it. That is what makes
    the pairing a walk rather than a copy of something already stored — a draft
    stores its chain and its own step's reader, and nothing else — and it is the
    same walk `_lineage` makes in `tools.py`, in the one direction that module
    cannot provide: it imports this one.

    The walk is bounded by the drafts the session has and cycle-checked, because
    a lineage is read off a document rather than built in memory, and a document
    is not the writer. A record that does not line up — a parent whose chain is
    not a prefix of the child's, which only a hand-edited file produces — has
    the requests it cannot place paired with no source rather than with a guess:
    inventing one would write a preference the record does not have.

    A draft with no parent carries no deltas, so a piece drafted from the brief
    has nothing to log: liking it is a judgement about the spec, and there is no
    request to weigh. That is why the caller writes no row for one.
    """
    by_id = {known.draft_id: known for known in session.drafts}
    steps: list[tuple[tuple[Delta, ...], RequestSource | None]] = []
    current = draft
    seen: set[str] = set()
    while True:
        seen.add(current.draft_id)
        parent = None if current.parent_id is None else by_id.get(current.parent_id)
        if (
            parent is None
            or parent.draft_id in seen
            or current.deltas[: len(parent.deltas)] != parent.deltas
        ):
            steps.append((current.deltas, None))
            break
        steps.append((current.deltas[len(parent.deltas) :], current.requests_source))
        current = parent
    return tuple(
        (delta, source) for step_deltas, source in reversed(steps) for delta in step_deltas
    )


class PreferenceLog:
    """Append-only JSONL of `Preference` rows, at its own root.

    Single-user and local, like `SessionStorage` and `JobStorage`: no locking,
    no owner, no multi-tenant addressing. The root is read from
    `SAIMC_PREFERENCES_DIR` when none is passed, the same shape the other two
    stores use, so a deployment has one place per store that says where its
    state lives.

    Nothing is created until there is a row to write. A session store mkdirs
    eagerly because its root is a directory of directories that later calls
    address by id; this log is one file, so an empty log is the absence of that
    file and a run that recorded no preference leaves no trace to notice. That
    also makes `append(())` — the ordinary case of a piece drafted from the
    brief, which has no request to weigh — a genuine no-op rather than an empty
    file that reads as "something happened here".
    """

    def __init__(self, root: Path | str | None = None) -> None:
        env_root = os.environ.get("SAIMC_PREFERENCES_DIR")
        if root is None:
            root = env_root or DEFAULT_PREFERENCES_DIR
        self._root = Path(root)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def path(self) -> Path:
        return self._root / PREFERENCES_FILENAME

    def append(self, rows: Iterable[Preference]) -> int:
        """Append `rows`, one line each, flushed before returning.

        Returns how many were written, which is what a caller that wants to
        report "N judgements logged" needs and what `0` means without a second
        call: nothing was written and, in particular, no file was created.

        Flushed rather than closed-and-reopened per row: one `open` per verdict
        is the same number of file handles for a burst of judgements, and the
        flush is what the durability actually needs — a row that reached the
        kernel survives the process that wrote it.
        """
        materialized = list(rows)
        if not materialized:
            return 0
        self._root.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            for row in materialized:
                handle.write(canonical_dumps(row.to_document()) + "\n")
            handle.flush()
        return len(materialized)

    def rows(self) -> tuple[Preference, ...]:
        """Every row in the log, in the order it was written.

        An absent log is an empty log: nothing has been recorded, which is the
        honest reading of a file that was never created. A line that is not a
        row refuses the whole read, naming the line — see the module docstring
        for why this is stricter than `list_all`.
        """
        if not self.path.exists():
            return ()
        rows: list[Preference] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                try:
                    payload = json.loads(line)
                    if not isinstance(payload, dict):
                        raise ValueError(f"expected a JSON object, got {type(payload).__name__}")
                    rows.append(Preference.from_document(payload))
                except (ValueError, KeyError, TypeError) as exc:
                    raise ValueError(
                        f"{self.path} line {number} is not a preference row: {exc}"
                    ) from exc
        return tuple(rows)


__all__ = [
    "DEFAULT_PREFERENCES_DIR",
    "PREFERENCES_FILENAME",
    "Preference",
    "PreferenceLog",
    "preference_rows",
]
