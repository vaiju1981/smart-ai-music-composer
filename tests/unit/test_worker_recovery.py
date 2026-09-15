"""Worker recovery tests: startup sweep, stage-crash catch-all, cancel race."""

from __future__ import annotations

from pathlib import Path

import pytest

from saimc.jobs import worker as worker_mod
from saimc.jobs.stages import StageResult
from saimc.jobs.state import JobState
from saimc.jobs.storage import JobStorage
from saimc.jobs.worker import _sweep_interrupted_jobs, _walk_stages, run_job


@pytest.fixture
def storage(tmp_path: Path) -> JobStorage:
    return JobStorage(tmp_path)


class TestStartupSweep:
    def test_started_jobs_are_failed_but_queued_jobs_survive(self, storage: JobStorage) -> None:
        stuck = storage.create("a prompt")
        stuck.state = JobState.RENDERING_ANIMATION
        stuck.current_stage = "rendering_animation"
        storage.save(stuck)
        queued = storage.create("another prompt")

        swept = _sweep_interrupted_jobs(storage)

        assert swept == 1
        stuck_after = storage.get(stuck.job_id)
        assert stuck_after.state == JobState.FAILED
        assert stuck_after.error is not None
        assert stuck_after.error.error_code == "worker_interrupted"
        assert stuck_after.error.stage == "rendering_animation"
        queued_after = storage.get(queued.job_id)
        assert queued_after.state == JobState.QUEUED
        assert queued_after.error is None

    def test_terminal_jobs_are_untouched(self, storage: JobStorage) -> None:
        done = storage.create("done")
        done.state = JobState.COMPLETE
        storage.save(done)
        failed = storage.create("failed")
        failed.state = JobState.FAILED
        storage.save(failed)

        assert _sweep_interrupted_jobs(storage) == 0
        assert storage.get(done.job_id).state == JobState.COMPLETE
        assert storage.get(failed.job_id).state == JobState.FAILED


class TestStageCrashCatchAll:
    def test_unexpected_stage_exception_marks_job_failed(
        self, storage: JobStorage, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        job = storage.create("a prompt")

        def boom(*args: object, **kwargs: object) -> None:
            raise RuntimeError("corrupt sidecar")

        monkeypatch.setattr(worker_mod, "parse_stage", boom)
        result = run_job(job.job_id, jobs_root=str(storage.root))

        assert result["error"] == "stage_crash"
        failed = storage.get(job.job_id)
        assert failed.state == JobState.FAILED
        assert failed.error is not None
        assert failed.error.error_code == "stage_crash"
        assert "corrupt sidecar" in failed.error.message


class TestMissingJob:
    def test_pruned_broker_entry_returns_instead_of_crashing(
        self, storage: JobStorage
    ) -> None:
        """A broker entry can outlive its job directory: `run_job` must
        report it and return rather than crash on the unbound `job` the
        crash handler would otherwise reach for.
        """
        assert run_job("no-such-job-id", jobs_root=str(storage.root)) == {
            "job_id": "no-such-job-id",
            "state": "missing",
            "error": "job_not_found",
        }


class TestCancelRace:
    def test_cancelled_persisted_job_is_not_clobbered(
        self, storage: JobStorage, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A cancel landing mid-stage must win over the in-memory walk."""
        job = storage.create("a prompt")
        job.state = JobState.RENDERING_AUDIO
        job.current_stage = "rendering_audio"
        storage.save(job)

        # The API process cancels the job while the worker is mid-stage:
        # the persisted copy is terminal, the walk's in-memory copy is not.
        persisted_cancelled = storage.get(job.job_id)
        persisted_cancelled.state = JobState.CANCELLED
        storage.save(persisted_cancelled)

        def succeed_stage(job: object, storage: object) -> StageResult:
            return StageResult(job=job, next_state=JobState.RENDERING_SHEET)  # type: ignore[arg-type]

        monkeypatch.setattr(worker_mod, "render_audio_stage", succeed_stage)

        in_memory = storage.get(job.job_id)
        in_memory.state = JobState.RENDERING_AUDIO  # the walk's stale copy
        _walk_stages(in_memory, storage, worker_mod.JobStateMachine())  # type: ignore[arg-type]

        assert storage.get(job.job_id).state == JobState.CANCELLED
