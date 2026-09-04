"""Unit tests for the artifact manifest builder."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from saimc.jobs.manifest import (
    AssetRecord,
    ManifestInputs,
    ToolchainRecord,
    build_manifest,
    write_manifest,
)
from saimc.jobs.state import JobState
from saimc.jobs.storage import ArtifactRecord, Job, JobStorage
from saimc.spec import Mood


@pytest.fixture
def completed_job(tmp_path: Path) -> Job:
    storage = JobStorage(tmp_path)
    job = storage.create("calming piano music")
    job.input_spec = type(job.input_spec) if job.input_spec else None  # placeholder
    from saimc.spec import CompositionSpec

    spec = CompositionSpec(mood=Mood.CALMING, seed=7)
    job.input_spec = spec
    job.seed = spec.seed
    job.state = JobState.COMPLETE
    storage.ensure_artifact_dir(job.job_id).joinpath("audio.wav").write_bytes(b"RIFF")
    storage.ensure_artifact_dir(job.job_id).joinpath("sheet.svg").write_bytes(b"<svg/>")
    storage.ensure_artifact_dir(job.job_id).joinpath("animation.webm").write_bytes(b"WEB")
    storage.attach_artifact(
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
    storage.attach_artifact(
        job,
        ArtifactRecord(
            kind="sheet",
            container="svg",
            codec="osmd",
            path="sheet.svg",
            sha256="b" * 64,
            size_bytes=7,
        ),
    )
    storage.attach_artifact(
        job,
        ArtifactRecord(
            kind="animation",
            container="webm",
            codec="vp9+opus",
            path="animation.webm",
            sha256="c" * 64,
            size_bytes=3,
        ),
    )
    storage.save(job)
    return job


def _inputs() -> ManifestInputs:
    return ManifestInputs(
        notation_score_sha256="d" * 64,
        performance_plan_sha256="e" * 64,
        completed_at=datetime.now(UTC),
        toolchain_by_kind={
            "audio": ToolchainRecord(
                engine="fluidsynth",
                version="2.3.4",
                build_sha="f" * 40,
                config={"soundfont": "salamander.sf2"},
            ),
            "sheet": ToolchainRecord(
                engine="osmd",
                version="1.8.9",
                build_sha="abc123",
                config={"renderer": "headless-chromium-130"},
            ),
            "animation": ToolchainRecord(
                engine="ffmpeg",
                version="8.1.2",
                build_sha="464beb5e7bf0c311e68b45ae2f04e9cc2af88851abb4082231742a74d97b524c",
                config={"video": "libvpx-vp9", "audio": "libopus"},
            ),
        },
        assets={
            "soundfont": AssetRecord(
                name="Salamander Grand Piano",
                version="2023-08-15",
                sha256="0" * 64,
                license="CC BY 3.0",
                source_url="https://salamanderan.com/",
                notice_path="LICENSES/Salamander-Grand-Piano.txt",
            ),
            "notation_font": AssetRecord(
                name="Bravura",
                version="1.3",
                sha256="1" * 64,
                license="SIL OFL 1.1",
                source_url="https://github.com/steinbergmedia/bravura",
                notice_path="LICENSES/Bravura.txt",
            ),
        },
        dependencies={
            "music21": "9.5.0",
            "pydantic": "2.9.2",
            "fastapi": "0.115.5",
        },
        license_obligations={
            "ffmpeg": "LICENSES/ffmpeg.txt",
            "fluidsynth": "LICENSES/fluidsynth.txt",
        },
    )


class TestBuildManifest:
    def test_basic_shape(self, completed_job: Job) -> None:
        m = build_manifest(completed_job, _inputs())
        assert m["job_id"] == completed_job.job_id
        assert m["engine_version"] == completed_job.engine_version
        assert m["seed"] == 7
        assert set(m["artifacts"].keys()) == {"audio", "sheet", "animation"}
        assert m["notation_score_sha256"] == "d" * 64
        assert m["performance_plan_sha256"] == "e" * 64

    def test_input_spec_has_sha256(self, completed_job: Job) -> None:
        m = build_manifest(completed_job, _inputs())
        spec_block = m["input_spec"]
        assert "spec" in spec_block
        assert "sha256" in spec_block
        assert spec_block["spec"]["mood"] == "calming"
        assert len(spec_block["sha256"]) == 64

    def test_toolchain_records_per_kind(self, completed_job: Job) -> None:
        m = build_manifest(completed_job, _inputs())
        assert m["toolchain"]["audio"]["engine"] == "fluidsynth"
        assert m["toolchain"]["sheet"]["engine"] == "osmd"
        assert m["toolchain"]["animation"]["engine"] == "ffmpeg"
        assert m["toolchain"]["animation"]["build_sha"] == (
            "464beb5e7bf0c311e68b45ae2f04e9cc2af88851abb4082231742a74d97b524c"
        )

    def test_assets_record_attribution(self, completed_job: Job) -> None:
        m = build_manifest(completed_job, _inputs())
        assert m["assets"]["soundfont"]["license"] == "CC BY 3.0"
        assert m["assets"]["notation_font"]["notice_path"] == "LICENSES/Bravura.txt"

    def test_dependencies_are_pinned(self, completed_job: Job) -> None:
        m = build_manifest(completed_job, _inputs())
        assert m["dependencies"]["music21"] == "9.5.0"

    def test_license_obligations_point_at_l(self, completed_job: Job) -> None:
        m = build_manifest(completed_job, _inputs())
        assert m["license_obligations"]["ffmpeg"] == "LICENSES/ffmpeg.txt"

    def test_rejects_job_without_spec(self, tmp_path: Path) -> None:
        storage = JobStorage(tmp_path)
        job = storage.create("p")
        job.state = JobState.COMPLETE
        storage.ensure_artifact_dir(job.job_id).joinpath("audio.wav").write_bytes(b"x")
        storage.attach_artifact(
            job,
            ArtifactRecord(
                kind="audio",
                container="wav",
                codec="pcm",
                path="audio.wav",
                sha256="0" * 64,
                size_bytes=1,
            ),
        )
        storage.save(job)
        with pytest.raises(ValueError, match="no parsed spec"):
            build_manifest(job, _inputs())

    def test_rejects_non_complete_job(self, tmp_path: Path) -> None:
        storage = JobStorage(tmp_path)
        job = storage.create("p")
        from saimc.spec import CompositionSpec

        job.input_spec = CompositionSpec(mood=Mood.CALMING)
        storage.save(job)
        with pytest.raises(ValueError, match="not complete"):
            build_manifest(job, _inputs())

    def test_rejects_job_with_no_artifacts(self, tmp_path: Path) -> None:
        storage = JobStorage(tmp_path)
        job = storage.create("p")
        from saimc.spec import CompositionSpec

        job.input_spec = CompositionSpec(mood=Mood.CALMING)
        job.state = JobState.COMPLETE
        storage.save(job)
        with pytest.raises(ValueError, match="no artifacts"):
            build_manifest(job, _inputs())


class TestWriteManifest:
    def test_writes_canonical_json(self, completed_job: Job, tmp_path: Path) -> None:
        path = write_manifest(completed_job, _inputs(), tmp_path)
        text = path.read_text(encoding="utf-8")
        assert text.endswith("\n")
        # Canonical JSON: sorted keys (so the first key is "artifacts" and
        # the second is "assets" — alphabetical — and there is no
        # whitespace separator between consecutive tokens).
        assert text.startswith('{"artifacts":')
        assert '", "' not in text  # no `", "` between key-value pairs
        # Parseable, round-tripable.
        payload = json.loads(text)
        assert payload["job_id"] == completed_job.job_id

    def test_writes_to_correct_path(self, completed_job: Job, tmp_path: Path) -> None:
        path = write_manifest(completed_job, _inputs(), tmp_path)
        assert path == tmp_path / completed_job.job_id / "manifest.json"
