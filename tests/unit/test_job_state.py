"""Unit tests for the job state machine."""

from __future__ import annotations

import itertools

import pytest

from saimc.jobs.state import (
    CANCELLABLE_STATES,
    RENDERING_STATES,
    TERMINAL_STATES,
    IllegalTransitionError,
    JobState,
    JobStateMachine,
)


class TestLegalHappyPath:
    def test_full_happy_path(self) -> None:
        sm = JobStateMachine()
        path = [
            JobState.QUEUED,
            JobState.PARSING,
            JobState.COMPOSING,
            JobState.VALIDATING,
            JobState.RENDERING_AUDIO,
            JobState.RENDERING_SHEET,
            JobState.RENDERING_ANIMATION,
            JobState.COMPLETE,
        ]
        for current, target in itertools.pairwise(path):
            result = sm.transition(current, target)
            assert result.state == target
            assert 0.0 <= result.progress <= 1.0


class TestFailedReachableFromEveryState:
    @pytest.mark.parametrize(
        "current",
        [
            JobState.QUEUED,
            JobState.PARSING,
            JobState.COMPOSING,
            JobState.VALIDATING,
            JobState.RENDERING_AUDIO,
            JobState.RENDERING_SHEET,
            JobState.RENDERING_ANIMATION,
        ],
    )
    def test_failed_reachable(self, current: JobState) -> None:
        sm = JobStateMachine()
        result = sm.transition(current, JobState.FAILED)
        assert result.state == JobState.FAILED


class TestIllegalTransitions:
    @pytest.mark.parametrize(
        ("current", "target"),
        [
            (JobState.QUEUED, JobState.COMPLETE),
            (JobState.QUEUED, JobState.RENDERING_AUDIO),
            (JobState.PARSING, JobState.COMPLETE),
            (JobState.VALIDATING, JobState.PARSING),
            (JobState.RENDERING_AUDIO, JobState.QUEUED),
            (JobState.COMPLETE, JobState.FAILED),
            (JobState.FAILED, JobState.COMPLETE),
            (JobState.CANCELLED, JobState.PARSING),
        ],
    )
    def test_illegal_rejected(self, current: JobState, target: JobState) -> None:
        sm = JobStateMachine()
        with pytest.raises(IllegalTransitionError):
            sm.transition(current, target)


class TestCancelSemantics:
    @pytest.mark.parametrize("state", list(CANCELLABLE_STATES))
    def test_can_cancel_from_cancellable(self, state: JobState) -> None:
        sm = JobStateMachine()
        assert sm.can_cancel(state)

    @pytest.mark.parametrize(
        "state",
        [s for s in JobState if s not in CANCELLABLE_STATES],
    )
    def test_cannot_cancel_from_other_states(self, state: JobState) -> None:
        sm = JobStateMachine()
        assert not sm.can_cancel(state)

    def test_cancel_from_queued_legal(self) -> None:
        sm = JobStateMachine()
        result = sm.transition(JobState.QUEUED, JobState.CANCELLED)
        assert result.state == JobState.CANCELLED

    def test_cancel_from_parsing_legal(self) -> None:
        sm = JobStateMachine()
        result = sm.transition(JobState.PARSING, JobState.CANCELLED)
        assert result.state == JobState.CANCELLED

    def test_cancel_from_composing_legal(self) -> None:
        sm = JobStateMachine()
        result = sm.transition(JobState.COMPOSING, JobState.CANCELLED)
        assert result.state == JobState.CANCELLED

    def test_cancel_from_validating_illegal(self) -> None:
        sm = JobStateMachine()
        with pytest.raises(IllegalTransitionError):
            sm.transition(JobState.VALIDATING, JobState.CANCELLED)

    def test_cancel_from_rendering_illegal(self) -> None:
        sm = JobStateMachine()
        for state in RENDERING_STATES:
            with pytest.raises(IllegalTransitionError):
                sm.transition(state, JobState.CANCELLED)


class TestTerminalStates:
    def test_terminal_set(self) -> None:
        assert (
            frozenset({JobState.COMPLETE, JobState.FAILED, JobState.CANCELLED}) == TERMINAL_STATES
        )

    @pytest.mark.parametrize("state", list(TERMINAL_STATES))
    def test_cannot_transition_from_terminal(self, state: JobState) -> None:
        sm = JobStateMachine()
        for target in JobState:
            with pytest.raises(IllegalTransitionError):
                sm.transition(state, target)


class TestProgressMonotonic:
    def test_progress_advances(self) -> None:
        sm = JobStateMachine()
        last_progress = 0.0
        path = [
            JobState.QUEUED,
            JobState.PARSING,
            JobState.COMPOSING,
            JobState.VALIDATING,
            JobState.RENDERING_AUDIO,
            JobState.RENDERING_SHEET,
            JobState.RENDERING_ANIMATION,
            JobState.COMPLETE,
        ]
        for current, target in itertools.pairwise(path):
            r = sm.transition(current, target)
            assert r.progress > last_progress
            last_progress = r.progress


class TestAllStates:
    def test_all_states_yields_every_value(self) -> None:
        sm = JobStateMachine()
        assert set(sm.all_states()) == set(JobState)
