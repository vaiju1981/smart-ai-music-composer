"""Unit tests for the artifact manifest builder."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from saimc.compose.plan import PLAN_FORMAT
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

    def test_records_the_model_when_the_llm_served_the_parse(self, completed_job: Job) -> None:
        """§9: `parser_source` is accompanied by the model identifier.

        The adapter has always reported it in `ParseResult.extra`; before
        this field existed nothing carried it as far as the manifest, so
        the provenance record named a parser but not the model behind it.
        """
        completed_job.parser_source = "llm"
        completed_job.model = "gemma4:31b-cloud"
        m = build_manifest(completed_job, _inputs())
        assert m["parser_source"] == "llm"
        assert m["model"] == "gemma4:31b-cloud"

    def test_omits_the_model_for_a_fallback_only_parse(self, completed_job: Job) -> None:
        """Absence has to mean "no model was involved", not "unknown".

        A null here would be indistinguishable from a failed lookup, so
        the key is dropped rather than set when no LLM contributed.
        """
        completed_job.parser_source = "fallback"
        completed_job.model = None
        m = build_manifest(completed_job, _inputs())
        assert m["parser_source"] == "fallback"
        assert "model" not in m

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


class TestTheManifestRecordsThePlan:
    """§9: the manifest names the plan the piece was composed under.

    Without it the manifest records the *inputs* and the *outputs* and
    nothing about the decisions in between — and those decisions are what
    §6's determinism claim is made about, since `(plan, seed) -> notes` is
    the reproducible pair and `(spec, seed) -> notes` only holds while the
    module tables hold still.
    """

    def test_a_job_with_no_plan_omits_the_block(self, completed_job: Job) -> None:
        """Omitted, not nulled — the `model` rule. Its absence reads as
        "the engine's defaults composed this", which is what every job
        written before the field existed also means."""
        m = build_manifest(completed_job, _inputs())
        assert "input_plan" not in m

    def test_the_block_carries_the_document_and_its_digest(self, completed_job: Job) -> None:
        from saimc.canonical import canonical_dumps
        from saimc.compose.plan import default_plan

        completed_job.input_plan = default_plan(completed_job.input_spec)
        m = build_manifest(completed_job, _inputs())
        block = m["input_plan"]
        assert set(block) == {"plan", "sha256"}
        assert block["plan"]["format"] == PLAN_FORMAT
        # The digest is of exactly the document in the same block, so a
        # reader who has the manifest has everything needed to verify it.
        document = canonical_dumps(block["plan"])
        assert block["sha256"] == hashlib.sha256(document.encode("utf-8")).hexdigest()

    def test_the_recorded_plan_reads_back_as_the_plan(self, completed_job: Job) -> None:
        """The document is the canonical form, so it is reloadable — a
        manifest that recorded a plan no reader could rebuild would be a
        provenance record in name only."""
        from saimc.compose.plan import CompositionPlan, default_plan

        completed_job.input_plan = default_plan(completed_job.input_spec)
        block = build_manifest(completed_job, _inputs())["input_plan"]
        assert CompositionPlan.from_canonical_dict(block["plan"]) == completed_job.input_plan

    def test_the_block_survives_the_manifest_write(self, completed_job: Job, tmp_path: Path) -> None:
        """Through canonical JSON on disk, which is how it is read."""
        from saimc.compose.plan import default_plan

        completed_job.input_plan = default_plan(completed_job.input_spec)
        path = write_manifest(completed_job, _inputs(), tmp_path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["input_plan"]["plan"]["format"] == PLAN_FORMAT


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
