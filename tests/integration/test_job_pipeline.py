"""Integration tests for the Phase 1 job pipeline.

These tests exercise the FastAPI routes, the on-disk storage, and the
worker stage walk end-to-end against real objects (no RQ, no Valkey).
The composition engine is stubbed — it lands in slice 4 — so this
suite verifies the contract between slices 1-3 only.
"""

from __future__ import annotations

import hashlib
import json
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
from saimc.jobs.storage import JobStorage
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

    Returns a real `EngineOutput` so downstream stages can read its
    fields (`notation_score`, `performance_plan`). The integration test
    does not inspect the musical content; it only needs the worker to
    advance through every stage.
    """
    from saimc.compose.duration import DurationArrangement
    from saimc.compose.engine import EngineOutput
    from saimc.compose.forms import ChordTemplate
    from saimc.compose.score import (
        KeySignature,
        Measure,
        NotationScore,
        NoteEvent,
        PerformanceNoteEvent,
        PerformancePlan,
    )

    note = NoteEvent(voice_id=0, pitch_midi=60, tick=0, duration_ticks=480, velocity=64)
    measure = Measure(index=0, start_tick=0, end_tick=1920, time_signature="4/4")
    notation = NotationScore.make(
        ppq=480,
        key=KeySignature(root="C", mode="major"),
        time_signature="4/4",
        tempo_bpm=80.0,
        measures=[measure],
        notes=[note],
    )
    perf_note = PerformanceNoteEvent(
        voice_id=0,
        pitch_midi=60,
        start_us=0,
        duration_us=500_000,
        velocity=64,
    )
    performance = PerformancePlan.make(sample_rate=44100, notes=[perf_note])
    arrangement = DurationArrangement(
        form_bars=1,
        template=ChordTemplate(name="stub_1bar", bars=1, chords=((0, 1),)),
        repetition_count=1,
        total_bars=1,
        tempo_bpm=80.0,
    )
    return EngineOutput(
        notation_score=notation,
        performance_plan=performance,
        arrangement=arrangement,
        key=KeySignature(root="C", mode="major"),
        time_signature="4/4",
    )


def _stub_audio_stage_advances(job, storage, **_kwargs):  # type: ignore[no-untyped-def]
    """Stand-in for `render_audio_stage` that writes a tiny stub WAV.

    The test environment has no FluidSynth; this stub lets the worker
    walk to COMPLETE while still exercising the artifact-attach
    contract that the real stage uses.
    """
    from saimc.jobs.stages import StageResult
    from saimc.jobs.state import JobState
    from saimc.jobs.storage import ArtifactRecord

    audio_dir = storage.ensure_artifact_dir(job.job_id)
    audio_path = audio_dir / "audio.wav"
    audio_path.write_bytes(b"RIFF")
    storage.attach_artifact(
        job,
        ArtifactRecord(
            kind="audio",
            container="wav",
            codec="pcm_s24le",
            path=audio_path.name,
            sha256=hashlib.sha256(b"RIFF").hexdigest(),
            size_bytes=4,
        ),
    )
    return StageResult(job=job, next_state=JobState.RENDERING_SHEET)


def _stub_sheet_stage_advances(job, storage, **_kwargs):  # type: ignore[no-untyped-def]
    """Stand-in for `render_sheet_stage` that writes a tiny stub SVG.

    The test environment has no Node render-service build; this stub
    lets the worker walk to COMPLETE while exercising the same
    artifact-attach contract the real stage uses.
    """
    from saimc.jobs.stages import StageResult
    from saimc.jobs.state import JobState
    from saimc.jobs.storage import ArtifactRecord

    sheet_path = storage.ensure_artifact_dir(job.job_id) / "sheet.svg"
    sheet_path.write_bytes(b"<svg></svg>")
    storage.attach_artifact(
        job,
        ArtifactRecord(
            kind="sheet",
            container="svg",
            codec="svg",
            path=sheet_path.name,
            sha256=hashlib.sha256(b"<svg></svg>").hexdigest(),
            size_bytes=11,
        ),
    )
    return StageResult(job=job, next_state=JobState.RENDERING_ANIMATION)


def _stub_animation_stage_advances(job, storage, **_kwargs):  # type: ignore[no-untyped-def]
    """Stand-in for `render_animation_stage` that writes a tiny stub WebM.

    The test environment has no audited ffmpeg build; this stub lets
    the worker walk to COMPLETE while exercising the same
    artifact-attach contract the real stage uses.
    """
    from saimc.jobs.stages import StageResult
    from saimc.jobs.state import JobState
    from saimc.jobs.storage import ArtifactRecord

    animation_path = storage.ensure_artifact_dir(job.job_id) / "animation.webm"
    animation_path.write_bytes(b"\x1aE\xdf\xa3")
    storage.attach_artifact(
        job,
        ArtifactRecord(
            kind="animation",
            container="webm",
            codec="vp9",
            path=animation_path.name,
            sha256=hashlib.sha256(b"\x1aE\xdf\xa3").hexdigest(),
            size_bytes=4,
        ),
    )
    return StageResult(job=job, next_state=JobState.COMPLETE)


class TestEndToEndJobWalk:
    def test_default_engine_walks_to_complete_and_emits_manifest(
        self,
        client: TestClient,
        storage: JobStorage,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Default engine (= the real Phase 1 composer) takes the job to COMPLETE.

        The renderers are stubbed because the test env has no
        FluidSynth/Node/ffmpeg toolchain; the worker still walks
        through every state.
        """
        from saimc.jobs import stages
        from saimc.jobs import worker as w

        monkeypatch.setattr(stages, "render_audio_stage", _stub_audio_stage_advances)
        monkeypatch.setattr(w, "render_audio_stage", _stub_audio_stage_advances)
        monkeypatch.setattr(stages, "render_sheet_stage", _stub_sheet_stage_advances)
        monkeypatch.setattr(w, "render_sheet_stage", _stub_sheet_stage_advances)
        monkeypatch.setattr(stages, "render_animation_stage", _stub_animation_stage_advances)
        monkeypatch.setattr(w, "render_animation_stage", _stub_animation_stage_advances)

        spec = CompositionSpec(mood=Mood.CALMING, seed=42, duration_seconds=180)
        create = client.post(
            "/jobs/from-spec",
            json={"spec": spec.model_dump(mode="json")},
        ).json()
        job_id = create["job_id"]

        # Plant a long-expired completed job so this run also proves
        # the retention prune fires on every job run.
        expired = storage.create("expired")
        expired.state = JobState.COMPLETE
        storage.save(expired, at=datetime.now(UTC).replace(year=2020, month=1, day=1))

        result = run_job(job_id, jobs_root=str(storage.root))
        assert result["state"] == "complete"
        job = storage.get(job_id)
        assert job.state == JobState.COMPLETE
        assert job.progress == 1.0
        assert job.parser_source == "from-spec"
        assert job.seed == 42
        assert job.input_spec == spec

        # Audio artifact should have been attached by the stubbed stage.
        assert "audio" in job.artifacts
        audio = job.artifacts["audio"]
        assert audio.sha256 == hashlib.sha256(b"RIFF").hexdigest()

        # The worker emits manifest.json the moment the job completes
        # (roadmap §9); with the toolchain-free stubs it still records
        # the spec hash, artifacts, and license obligations.
        manifest_path = storage.root / job_id / "manifest.json"
        assert manifest_path.is_file()
        emitted = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert emitted["job_id"] == job_id
        assert emitted["notation_score_sha256"]
        assert emitted["license_obligations"]["salamander-grand-piano"]

        # A job run also applies retention (roadmap §7): the expired
        # completed job planted above is pruned before the walk starts.
        assert "expired" not in [j.job_id for j in storage.list_all()]

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

    def test_with_engine_stub_walks_to_complete_and_emits_manifest(
        self, client: TestClient, storage: JobStorage, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Inject a stub composition engine so the worker walks to COMPLETE.

        `compose_stage` in `saimc.jobs.stages` calls the engine; we
        monkeypatch the engine argument via a wrapper so the worker
        receives a real returned NotationScore/PerformancePlan stand-in.
        The audio stage is also stubbed because the test env has no
        FluidSynth.
        """
        from saimc.jobs import stages

        original_compose_stage = stages.compose_stage

        def _always_succeed(job, storage, engine=None):  # type: ignore[no-untyped-def]
            return original_compose_stage(job, storage, engine=_stub_engine_returns)

        monkeypatch.setattr(stages, "compose_stage", _always_succeed)
        # The worker imports compose_stage into its own module namespace, so
        # also patch the reference it sees.
        from saimc.jobs import worker as w

        monkeypatch.setattr(w, "compose_stage", _always_succeed)
        monkeypatch.setattr(stages, "render_audio_stage", _stub_audio_stage_advances)
        monkeypatch.setattr(w, "render_audio_stage", _stub_audio_stage_advances)
        monkeypatch.setattr(stages, "render_sheet_stage", _stub_sheet_stage_advances)
        monkeypatch.setattr(w, "render_sheet_stage", _stub_sheet_stage_advances)
        monkeypatch.setattr(stages, "render_animation_stage", _stub_animation_stage_advances)
        monkeypatch.setattr(w, "render_animation_stage", _stub_animation_stage_advances)

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

        assert "audio" in job.artifacts
        audio = job.artifacts["audio"]
        assert audio.sha256 == hashlib.sha256(b"RIFF").hexdigest()

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
