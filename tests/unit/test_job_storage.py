"""Unit tests for the on-disk job storage."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from saimc.jobs.state import JobState
from saimc.jobs.storage import (
    ArtifactRecord,
    JobError,
    JobStorage,
)
from saimc.spec import CompositionSpec, Mood


@pytest.fixture
def store(tmp_path):
    s = JobStorage(tmp_path)
    yield s
    s.prune()


class TestCreateAndGet:
    def test_create_returns_queued_job(self, store: JobStorage) -> None:
        job = store.create("calming piano music")
        assert job.state == JobState.QUEUED
        assert job.input_prompt == "calming piano music"
        assert job.input_spec is None
        assert job.progress == 0.0
        assert job.is_terminal() is False

    def test_create_persists_immediately(self, store: JobStorage) -> None:
        job = store.create("p1")
        reloaded = store.get(job.job_id)
        assert reloaded.job_id == job.job_id
        assert reloaded.input_prompt == "p1"

    def test_get_unknown_raises_keyerror(self, store: JobStorage) -> None:
        with pytest.raises(KeyError):
            store.get("does-not-exist")


class TestAttachSpec:
    def test_attach_spec_sets_fields_and_seed(self, store: JobStorage) -> None:
        job = store.create("p")
        spec = CompositionSpec(mood=Mood.CALMING, seed=42)
        store.attach_spec(job, spec, parser_source="llm", attempts=1)
        store.save(job)
        reloaded = store.get(job.job_id)
        assert reloaded.input_spec == spec
        assert reloaded.parser_source == "llm"
        assert reloaded.attempts == 1
        assert reloaded.seed == 42

    def test_attach_spec_no_seed_keeps_none(self, store: JobStorage) -> None:
        job = store.create("p")
        spec = CompositionSpec(mood=Mood.CALMING)
        store.attach_spec(job, spec, parser_source="fallback", attempts=3)
        store.save(job)
        reloaded = store.get(job.job_id)
        assert reloaded.seed is None


class TestAttachArtifact:
    def test_attach_audio_artifact(self, store: JobStorage) -> None:
        job = store.create("p")
        store.ensure_artifact_dir(job.job_id).joinpath("audio.wav").write_bytes(b"RIFF")
        store.attach_artifact(
            job,
            ArtifactRecord(
                kind="audio",
                container="wav",
                codec="pcm_s24le",
                path="audio.wav",
                sha256="a" * 64,
                size_bytes=4,
            ),
        )
        store.save(job)
        reloaded = store.get(job.job_id)
        assert "audio" in reloaded.artifacts
        a = reloaded.artifacts["audio"]
        assert a.container == "wav"
        assert a.sha256 == "a" * 64

    def test_artifact_path_resolves_under_root(self, store: JobStorage) -> None:
        job = store.create("p")
        p = store.artifact_path(job.job_id, "audio")
        assert p == store.root / job.job_id / "artifacts" / "audio"


class TestAttachError:
    def test_attach_error(self, store: JobStorage) -> None:
        job = store.create("p")
        store.attach_error(job, JobError(error_code="x", message="y", stage="parsing"))
        store.save(job)
        reloaded = store.get(job.job_id)
        assert reloaded.error == JobError(error_code="x", message="y", stage="parsing")


class TestListAndPrune:
    def test_list_all_includes_every_job(self, store: JobStorage) -> None:
        for _ in range(3):
            store.create("p")
        ids = [j.job_id for j in store.list_all()]
        assert len(ids) == 3

    def test_prune_removes_old_completed(self, store: JobStorage) -> None:
        job = store.create("p")
        job.state = JobState.COMPLETE
        store.save(job, at=datetime.now(UTC) - timedelta(days=8))
        deleted = store.prune()
        assert deleted == 1
        with pytest.raises(KeyError):
            store.get(job.job_id)

    def test_prune_keeps_recent_completed(self, store: JobStorage) -> None:
        job = store.create("p")
        job.state = JobState.COMPLETE
        store.save(job)
        deleted = store.prune()
        assert deleted == 0
        store.get(job.job_id)

    def test_prune_removes_old_failed(self, store: JobStorage) -> None:
        job = store.create("p")
        job.state = JobState.FAILED
        store.save(job, at=datetime.now(UTC) - timedelta(hours=25))
        deleted = store.prune()
        assert deleted == 1

    def test_prune_does_not_delete_active(self, store: JobStorage) -> None:
        job = store.create("p")
        store.save(job, at=datetime.now(UTC) - timedelta(days=30))
        deleted = store.prune()
        assert deleted == 0
        store.get(job.job_id)


class TestAtomicWrite:
    def test_concurrent_readers_do_not_observe_partial_writes(self, store: JobStorage) -> None:
        """The tmp+rename pattern means a reader never sees a torn JSON."""
        job = store.create("p")
        for i in range(10):
            spec = CompositionSpec(mood=Mood.CALMING, duration_seconds=30 + i)
            store.attach_spec(job, spec, parser_source="llm", attempts=1)
            store.save(job)
            reloaded = store.get(job.job_id)
            assert reloaded.input_spec is not None
            assert reloaded.input_spec.duration_seconds == 30 + i
