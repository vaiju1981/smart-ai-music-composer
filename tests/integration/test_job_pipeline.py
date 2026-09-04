"""Integration tests for the Phase 1 job pipeline.

These tests exercise the FastAPI routes, the on-disk storage, and the
worker stage walk end-to-end against real objects (no RQ, no Valkey).
The composition engine is stubbed — it lands in slice 4 — so this
suite verifies the contract between slices 1-3 only.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from saimc.jobs.api import create_app
from saimc.jobs.manifest import (
    AssetRecord,
    ManifestInputs,
    ToolchainRecord,
    build_manifest,
    write_manifest,
)
from saimc.jobs.state import JobState
from saimc.jobs.storage import ArtifactRecord, JobStorage
from saimc.jobs.worker import run_job
from saimc.spec import CompositionSpec, Mood


@pytest.fixture
def storage(tmp_path: Path) -> JobStorage:
    return JobStorage(tmp_path)


@pytest.fixture
def client(storage: JobStorage) -> TestClient:
    app = create_app(jobs_root=storage.root)
    return TestClient(app)


def _stub_engine_returns(spec: CompositionSpec) -> tuple[object, object]:
    """Stand-in for the slice-4 composition engine.

    Returns two opaque objects representing (NotationScore, PerformancePlan).
    The integration test does not inspect them — slice 4 is responsible
    for making them meaningful.
    """
    return (object(), object())


class TestEndToEndJobWalk:
    def test_without_engine_transitions_to_failed(
        self, client: TestClient, storage: JobStorage
    ) -> None:
        """Until slice 4 lands the composition engine, the worker must
        end in `failed` with a structured error rather than silently
        claim success."""
        spec = CompositionSpec(mood=Mood.CALMING, seed=42, duration_seconds=180)
        create = client.post(
            "/jobs/from-spec",
            json={"spec": spec.model_dump(mode="json")},
        ).json()
        job_id = create["job_id"]
        result = run_job(job_id, jobs_root=str(storage.root))
        assert result["state"] == "failed"
        job = storage.get(job_id)
        assert job.state == JobState.FAILED
        assert job.error is not None
        assert job.error.error_code == "engine_not_implemented"

    def test_with_engine_stub_walks_to_complete_and_emits_manifest(
        self, client: TestClient, storage: JobStorage, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Inject a stub composition engine so the worker walks to COMPLETE.

        `compose_stage` in `saimc.jobs.stages` calls the engine; we
        monkeypatch the engine argument via a wrapper so the worker
        receives a real returned NotationScore/PerformancePlan stand-in.
        """
        # We monkeypatch `compose_stage` itself to return success regardless
        # of the engine argument. This is the cleanest seam for an integration
        # test; the unit-level test of `compose_stage` already covers the
        # engine-not-None path.
        from saimc.jobs import stages

        original_compose_stage = stages.compose_stage

        def _always_succeed(job, storage, engine=None):  # type: ignore[no-untyped-def]
            return original_compose_stage(job, storage, engine=_stub_engine_returns)

        monkeypatch.setattr(stages, "compose_stage", _always_succeed)
        # The worker imports compose_stage into its own module namespace, so
        # also patch the reference it sees.
        from saimc.jobs import worker as w

        monkeypatch.setattr(w, "compose_stage", _always_succeed)

        spec = CompositionSpec(mood=Mood.CALMING, seed=42, duration_seconds=180)
        create = client.post(
            "/jobs/from-spec",
            json={"spec": spec.model_dump(mode="json")},
        ).json()
        job_id = create["job_id"]
        result = run_job(job_id, jobs_root=str(storage.root))
        assert result["state"] == "complete"

        job = storage.get(job_id)
        assert job.state == JobState.COMPLETE
        assert job.progress == 1.0
        assert job.parser_source == "from-spec"
        assert job.seed == 42
        assert job.input_spec == spec

        # Manifest requires at least one artifact; add a stub.
        storage.ensure_artifact_dir(job_id).joinpath("audio.wav").write_bytes(b"RIFF")
        storage.attach_artifact(
            job,
            ArtifactRecord(
                kind="audio",
                container="wav",
                codec="pcm_s24le",
                path="audio.wav",
                sha256=hashlib.sha256(b"RIFF").hexdigest(),
                size_bytes=4,
            ),
        )
        storage.save(job)

        manifest_inputs = ManifestInputs(
            notation_score_sha256="0" * 64,
            performance_plan_sha256="1" * 64,
            completed_at=datetime.now(UTC),
            toolchain_by_kind={
                "audio": ToolchainRecord(
                    engine="fluidsynth",
                    version="2.3.4",
                    build_sha="x" * 40,
                    config={},
                )
            },
            assets={
                "soundfont": AssetRecord(
                    name="Salamander Grand Piano",
                    version="2023",
                    sha256="0" * 64,
                    license="CC BY 3.0",
                    source_url="https://salamanderan.com/",
                    notice_path="LICENSES/Salamander-Grand-Piano.txt",
                )
            },
            dependencies={"music21": "9.5.0"},
            license_obligations={"fluidsynth": "LICENSES/fluidsynth.txt"},
        )
        manifest = build_manifest(job, manifest_inputs)
        assert manifest["job_id"] == job_id
        assert manifest["artifacts"]["audio"]["sha256"] == hashlib.sha256(b"RIFF").hexdigest()

        manifest_path = write_manifest(job, manifest_inputs, storage.root)
        assert manifest_path == storage.root / job_id / "manifest.json"
        assert manifest_path.exists()


class TestCancelSemantics:
    def test_cancel_queued_is_persisted(self, client: TestClient, storage: JobStorage) -> None:
        create = client.post("/jobs", json={"prompt": "x"}).json()
        job_id = create["job_id"]
        resp = client.post(f"/jobs/{job_id}/cancel")
        assert resp.status_code == 200
        assert resp.json()["state"] == "cancelled"
        job = storage.get(job_id)
        assert job.state == JobState.CANCELLED


class TestPruneIntegration:
    def test_prune_keeps_active_removes_old(self, storage: JobStorage, tmp_path: Path) -> None:
        # An old completed job.
        old_job = storage.create("old")
        old_job.state = JobState.COMPLETE
        storage.save(old_job, at=datetime.now(UTC).replace(year=2020, month=1, day=1))
        # A recent queued job.
        new_job = storage.create("new")

        deleted = storage.prune()
        assert deleted == 1

        with pytest.raises(KeyError):
            storage.get(old_job.job_id)
        storage.get(new_job.job_id)
