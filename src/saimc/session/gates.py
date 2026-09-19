"""Gates over a session: what the harness shipped, judged as a release.

`saimc.release.gates` judges the *engine*: it composes a supplied matrix and
asks whether those notes clear the bars. That gate cannot see the harness at
all, and the hole it leaves is the one this module closes. A session drafts
several candidates, an arbiter ranks them, the user picks one — and the
quality of the piece that ships is then a property of *that search*, not of
the generator. Compose a hundred candidates and keep the one that happens to
clear every threshold, and a report of "the shipped piece is clean" says
nothing about a generator that misses half the time.

So the two numbers are read together, and both come from the session's own
log rather than from a fresh composition: the published draft's scorecard,
and how many of the session's candidates cleared the same bars. The session
keeps every candidate it made — `Draft` carries its plan hash and its
`PieceQuality` — so the search is visible after the fact and this gate can
count it instead of inferring it from the winner.

**Nothing in the publish path consults this gate, deliberately.** Open
Question 4 of the plan asks who wins when the user's verdict contradicts a
threshold, and the answer settled there is that correctness (the linter) is
non-negotiable while taste is the user's — so a user may ship a piece the
scorecard dislikes. A gate that refused the publish would reverse that
decision silently, one layer down. It is a release bar over the harness: it
reports, and a person reads it.
"""

from __future__ import annotations

from saimc.quality import QUALITY_THRESHOLDS
from saimc.release.gates import GateResult
from saimc.session.models import Session

__all__ = ["gate_session_publication"]


def gate_session_publication(session: Session) -> GateResult:
    """The published draft clears every threshold, and the search is reported.

    An unpublished session fails rather than passing vacuously, for
    `gate_musical_quality`'s reason: a gate that has nothing to judge and
    says so with a green light is a bar that cannot be measured.

    The failure detail names the misses with the finding's own line
    (`QualityFinding.message()`, which carries the measured value, the bar,
    and the hint a critic would follow) rather than with a count, because
    "two thresholds missed" is not something a reader can act on. When a
    sibling candidate would have cleared every bar, it is named too — that
    is the sentence the fan-out exists to make possible, and a report of a
    failing piece that stays silent about a passing one beside it is
    describing only half of what the session knows.
    """
    if session.publication is None:
        return GateResult(
            "session_publication",
            False,
            "this session has not published a piece; there is nothing to judge",
        )
    chosen = session.draft(session.publication.draft_id)
    findings = chosen.quality.findings()
    total = len(session.drafts)
    clean = [draft for draft in session.drafts if not draft.quality.findings()]
    rate = f"{len(clean)}/{total} of this session's candidates cleared every threshold"

    if not findings:
        return GateResult(
            "session_publication",
            True,
            f"the published draft clears all {len(QUALITY_THRESHOLDS)} thresholds; {rate}",
        )

    missed = "; ".join(finding.message() for finding in findings)
    detail = (
        f"the published draft misses {len(findings)} of {len(QUALITY_THRESHOLDS)} thresholds: "
        f"{missed}. {rate}"
    )
    # `clean` cannot hold the chosen draft on this path — it has findings —
    # so the first entry is a sibling, and there is one only when some other
    # candidate really did clear every bar. First in the session's own order,
    # which is the order the candidates were drafted in, so the sentence is
    # stable and two reports of one session name the same draft.
    alternative = clean[0] if clean else None
    if alternative is not None:
        detail += f", including {alternative.draft_id!r}, which was not published"
    return GateResult("session_publication", False, detail)
