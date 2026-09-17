"""Unit tests for the on-disk job storage."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from saimc.compose.plan import PLAN_FORMAT, default_plan
from saimc.jobs.state import JobState
from saimc.jobs.storage import (
    ArtifactRecord,
    JobStorage,
    UnsupportedPlanVersionError,
    UnsupportedSpecVersionError,
)
from saimc.spec import SPEC_SCHEMA_VERSION, CompositionSpec, Mood


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
        assert str(SPEC_SCHEMA_VERSION) in str(exc_info.value)

    def test_current_spec_version_loads_normally(self, store) -> None:
        job = store.create("p")
        job.input_spec = CompositionSpec(mood=Mood.CALMING)
        store.save(job)
        reloaded = store.get(job.job_id)
        assert reloaded.input_spec is not None
        assert reloaded.input_spec.mood == Mood.CALMING


class TestPlanPersistence:
    """The job records the plan it composes under, materialized.

    Stored rather than re-derived, for the reason `plan.py` gives: a plan
    rebuilt from a module table at read time would make this job's
    reproducibility a claim about the build rather than about the job.
    """

    def test_a_job_with_no_plan_stores_and_loads_none(self, store, tmp_path) -> None:
        job = store.create("p")
        job.input_spec = CompositionSpec(mood=Mood.CALMING)
        store.save(job)
        assert store.get(job.job_id).input_plan is None
        payload = json.loads((tmp_path / job.job_id / "job.json").read_text())
        assert payload["input_plan"] is None

    def test_a_jobs_plan_round_trips_unchanged(self, store) -> None:
        spec = CompositionSpec(mood=Mood.ELECTRIFYING, duration_seconds=45, seed=3)
        job = store.create("p")
        job.input_spec = spec
        job.input_plan = replace(default_plan(spec), harmony_pad_velocity=42)
        store.save(job)

        reloaded = store.get(job.job_id)
        assert reloaded.input_plan is not None
        assert reloaded.input_plan == job.input_plan
        assert reloaded.input_plan.compute_hash() == job.input_plan.compute_hash()

    def test_a_plan_from_the_future_is_refused(self, store, tmp_path) -> None:
        """Refused rather than coerced, unlike the spec.

        A spec is widened additively and Pydantic fills what it is
        missing; a plan is stored *materialized*, so a version this build
        cannot read is one whose fields it cannot know. Loading it anyway
        would make the job's reproducibility a claim the build cannot
        honour while still reporting success.
        """
        job = store.create("p")
        job.input_spec = CompositionSpec(mood=Mood.CALMING)
        job.input_plan = default_plan(job.input_spec)
        store.save(job)
        job_file = tmp_path / job.job_id / "job.json"
        payload = json.loads(job_file.read_text())
        payload["input_plan"]["format"] = "CompositionPlan:999"
        job_file.write_text(json.dumps(payload))

        with pytest.raises(UnsupportedPlanVersionError) as exc_info:
            store.get(job.job_id)
        assert "CompositionPlan:999" in str(exc_info.value)
        assert PLAN_FORMAT in str(exc_info.value)

    def test_a_plan_from_the_past_is_refused_too(self, store, tmp_path) -> None:
        """The other direction, which the spec guard does not have to care
        about: an older plan is missing fields, so it is not a plan this
        build can honour either."""
        job = store.create("p")
        job.input_spec = CompositionSpec(mood=Mood.CALMING)
        job.input_plan = default_plan(job.input_spec)
        store.save(job)
        job_file = tmp_path / job.job_id / "job.json"
        payload = json.loads(job_file.read_text())
        payload["input_plan"]["format"] = "CompositionPlan:0"
        job_file.write_text(json.dumps(payload))

        with pytest.raises(UnsupportedPlanVersionError):
            store.get(job.job_id)

    def test_a_job_written_before_plans_loads_with_none(self, store, tmp_path) -> None:
        """The `.get()` read, which is what every job on disk today needs."""
        job = store.create("p")
        job.input_spec = CompositionSpec(mood=Mood.CALMING)
        store.save(job)
        job_file = tmp_path / job.job_id / "job.json"
        payload = json.loads(job_file.read_text())
        del payload["input_plan"]
        job_file.write_text(json.dumps(payload))
        assert store.get(job.job_id).input_plan is None

    def test_an_unreadable_plan_does_not_break_a_listing(self, store, tmp_path) -> None:
        """`list_all` skips what it cannot read, and a refused plan is one
        of those — the same treatment the future spec version gets."""
        good = store.create("good")
        good.input_spec = CompositionSpec(mood=Mood.CALMING)
        store.save(good)
        bad = store.create("bad")
        bad.input_spec = CompositionSpec(mood=Mood.CALMING)
        bad.input_plan = default_plan(bad.input_spec)
        store.save(bad)
        bad_file = tmp_path / bad.job_id / "job.json"
        payload = json.loads(bad_file.read_text())
        payload["input_plan"]["format"] = "CompositionPlan:999"
        bad_file.write_text(json.dumps(payload))

        listed = {job.job_id for job in store.list_all()}
        assert good.job_id in listed
        assert bad.job_id not in listed
