"""Job state machine — the contract for what state a job may be in and
which transitions are legal.

Per `docs/roadmap.md` §7:

    queued -> parsing -> composing -> validating
          -> rendering_audio -> rendering_sheet -> rendering_animation
          -> complete
                                                                  v
                                                              failed (terminal)

Every non-terminal state may transition to `failed` (per §7, "Failed is
reachable from every state"). `complete` and `failed` are the only two
terminal states. Cancel is allowed only in `queued`, `parsing`, and
`composing`; once rendering has started, cancel returns 409 Conflict.

This module contains the pure state-machine logic only — it does NOT
touch RQ, Valkey, the filesystem, or the network. The worker (slice 3.2)
and the API (slice 3.3) call into `JobState.transition()` and rely on
the returned `TransitionResult` to decide what to do next.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import ClassVar


class JobState(StrEnum):
    """The set of legal job states (per the §7 state machine)."""

    QUEUED = "queued"
    PARSING = "parsing"
    COMPOSING = "composing"
    VALIDATING = "validating"
    RENDERING_AUDIO = "rendering_audio"
    RENDERING_SHEET = "rendering_sheet"
    RENDERING_ANIMATION = "rendering_animation"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


TERMINAL_STATES: frozenset[JobState] = frozenset(
    {JobState.COMPLETE, JobState.FAILED, JobState.CANCELLED}
)

CANCELLABLE_STATES: frozenset[JobState] = frozenset(
    {JobState.QUEUED, JobState.PARSING, JobState.COMPOSING}
)

RENDERING_STATES: frozenset[JobState] = frozenset(
    {JobState.RENDERING_AUDIO, JobState.RENDERING_SHEET, JobState.RENDERING_ANIMATION}
)


class IllegalTransitionError(Exception):
    """Raised when a requested transition is not in the legal set."""

    def __init__(self, from_state: JobState, to_state: JobState, reason: str) -> None:
        super().__init__(f"Illegal transition {from_state.value} -> {to_state.value}: {reason}")
        self.from_state = from_state
        self.to_state = to_state
        self.reason = reason


@dataclass(frozen=True)
class TransitionResult:
    """Returned from `JobStateMachine.transition` on success.

    Attributes mirror the §7 status fields:
    - `progress` is a float in [0.0, 1.0]; it is computed deterministically
      from the current state so the API can surface it.
    - `current_stage` is the human-readable label of the current state.
    """

    state: JobState
    progress: float
    current_stage: str


_PROGRESS_TABLE: dict[JobState, float] = {
    JobState.QUEUED: 0.00,
    JobState.PARSING: 0.05,
    JobState.COMPOSING: 0.15,
    JobState.VALIDATING: 0.25,
    JobState.RENDERING_AUDIO: 0.45,
    JobState.RENDERING_SHEET: 0.65,
    JobState.RENDERING_ANIMATION: 0.85,
    JobState.COMPLETE: 1.00,
    JobState.FAILED: 1.00,
    JobState.CANCELLED: 1.00,
}


class JobStateMachine:
    """Stateless state-machine object.

    Holds the legal-transition table; the *current state* lives on the
    persisted Job. The API/worker call `machine.transition(current, next)`
    to validate a transition before persisting it.
    """

    LEGAL_NEXT: ClassVar[dict[JobState, frozenset[JobState]]] = {
        JobState.QUEUED: frozenset({JobState.PARSING, JobState.FAILED, JobState.CANCELLED}),
        JobState.PARSING: frozenset({JobState.COMPOSING, JobState.FAILED, JobState.CANCELLED}),
        JobState.COMPOSING: frozenset({JobState.VALIDATING, JobState.FAILED, JobState.CANCELLED}),
        JobState.VALIDATING: frozenset({JobState.RENDERING_AUDIO, JobState.FAILED}),
        JobState.RENDERING_AUDIO: frozenset({JobState.RENDERING_SHEET, JobState.FAILED}),
        JobState.RENDERING_SHEET: frozenset({JobState.RENDERING_ANIMATION, JobState.FAILED}),
        JobState.RENDERING_ANIMATION: frozenset({JobState.COMPLETE, JobState.FAILED}),
        JobState.COMPLETE: frozenset(),
        JobState.FAILED: frozenset(),
        JobState.CANCELLED: frozenset(),
    }

    def transition(self, current: JobState, target: JobState) -> TransitionResult:
        """Validate and return the transition.

        Raises `IllegalTransitionError` if `target` is not reachable from
        `current`. The `failed` transition is permitted from every
        non-terminal state, which §7 requires.
        """
        if current in TERMINAL_STATES:
            raise IllegalTransitionError(current, target, f"{current.value} is terminal.")
        allowed = self.LEGAL_NEXT[current]
        if target not in allowed:
            raise IllegalTransitionError(
                current, target, f"allowed from {current.value}: {sorted(s.value for s in allowed)}"
            )
        return TransitionResult(
            state=target,
            progress=_PROGRESS_TABLE[target],
            current_stage=target.value,
        )

    def can_cancel(self, current: JobState) -> bool:
        """True iff the job can be cancelled from `current` (per §7)."""
        return current in CANCELLABLE_STATES

    def is_terminal(self, state: JobState) -> bool:
        return state in TERMINAL_STATES

    def all_states(self) -> Iterable[JobState]:
        """Every state — useful for tests, documentation, and UI dropdowns."""
        yield from JobState


__all__ = [
    "CANCELLABLE_STATES",
    "RENDERING_STATES",
    "TERMINAL_STATES",
    "IllegalTransitionError",
    "JobState",
    "JobStateMachine",
    "TransitionResult",
]
