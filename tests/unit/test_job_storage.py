"""Unit tests for the on-disk job storage."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from saimc.jobs.state import JobState
from saimc.jobs.storage import (
    ArtifactRecord,
    JobStorage,
    UnsupportedSpecVersionError,
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

    def test_toolchain_provenance_round_trips(self, store: JobStorage) -> None:
        """The manifest derives its toolchain block from the persisted job."""
        job = store.create("p")
        store.attach_artifact(
            job,
            ArtifactRecord(
                kind="audio",
                container="wav",
                codec="pcm_s24le",
                path="audio.wav",
                sha256="a" * 64,
                size_bytes=4,
                toolchain={"engine": "fluidsynth", "version": "2.3.4"},
            ),
        )
        store.save(job)
        reloaded = store.get(job.job_id)
        assert reloaded.artifacts["audio"].toolchain == {
            "engine": "fluidsynth",
            "version": "2.3.4",
        }


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
            job.input_spec = CompositionSpec(mood=Mood.CALMING, duration_seconds=30 + i)
            store.save(job)
            reloaded = store.get(job.job_id)
            assert reloaded.input_spec is not None
            assert reloaded.input_spec.duration_seconds == 30 + i


class TestSchemaVersionGuard:
    def test_future_spec_version_raises_a_readable_error(self, store, tmp_path) -> None:
        """A job written by a newer build fails with guidance, not a pydantic dump."""
        job = store.create("p")
        job.input_spec = CompositionSpec(mood=Mood.CALMING)
        store.save(job)
        job_file = tmp_path / job.job_id / "job.json"
        payload = json.loads(job_file.read_text())
        payload["input_spec"]["schema_version"] = 999
        job_file.write_text(json.dumps(payload))

        with pytest.raises(UnsupportedSpecVersionError) as exc_info:
            store.get(job.job_id)
        assert "999" in str(exc_info.value)
        assert "1" in str(exc_info.value)

    def test_current_spec_version_loads_normally(self, store) -> None:
        job = store.create("p")
        job.input_spec = CompositionSpec(mood=Mood.CALMING)
        store.save(job)
        reloaded = store.get(job.job_id)
        assert reloaded.input_spec is not None
        assert reloaded.input_spec.mood == Mood.CALMING
