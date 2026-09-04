"""Tests for `emit_manifest`: manifest emission from the persisted job alone."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from saimc.compose.engine import compose
from saimc.compose.serialization import write_engine_output
from saimc.jobs.manifest import emit_manifest
from saimc.jobs.state import JobState
from saimc.jobs.storage import ArtifactRecord, JobStorage
from saimc.render.attribution import SOUNDFONT_NOTICE_PATH
from saimc.spec import CompositionSpec, Mood


def _spec() -> CompositionSpec:
    return CompositionSpec(mood=Mood.CALMING, seed=7)


def _completed_job(tmp_path: Path) -> tuple[JobStorage, Any]:
    """A completed job with a real sidecar and toolchain-bearing artifacts."""
    store = JobStorage(tmp_path)
    job = store.create("calming piano")
    job.input_spec = _spec()
    output = compose(job.input_spec)
    write_engine_output(store.job_dir(job.job_id) / "engine_output.json", output)

    artifacts_dir = store.ensure_artifact_dir(job.job_id)
    (artifacts_dir / "audio.wav").write_bytes(b"RIFF")
    (artifacts_dir / "sheet.svg").write_bytes(b"<svg/>")
    (artifacts_dir / "animation.webm").write_bytes(b"WEB")

    audio_toolchain = {
        "engine": "fluidsynth",
        "version": "2.3.4",
        "build_sha": "f" * 40,
        "soundfont": "Salamander.sf2",
        "soundfont_sha256": "9" * 64,
        "ffmpeg_version": "8.1.2",
        "ffmpeg_build_sha": "e" * 64,
        "ffmpeg_configuration": "--enable-libvpx --enable-libopus",
    }
    store.attach_artifact(
        job,
        ArtifactRecord(
            kind="audio",
            container="wav",
            codec="pcm_s24le",
            path="audio.wav",
            sha256="a" * 64,
            size_bytes=4,
            toolchain=audio_toolchain,
        ),
    )
    store.attach_artifact(
        job,
        ArtifactRecord(
            kind="sheet",
            container="svg",
            codec="svg",
            path="sheet.svg",
            sha256="b" * 64,
            size_bytes=6,
            toolchain={"engine": "opensheetmusicdisplay", "version": "2.1.2"},
        ),
    )
    store.attach_artifact(
        job,
        ArtifactRecord(
            kind="animation",
            container="webm",
            codec="vp9",
            path="animation.webm",
            sha256="c" * 64,
            size_bytes=3,
            toolchain={"engine": "ffmpeg", "version": "8.1.2", "build_sha": "e" * 40},
        ),
    )

    job.state = JobState.COMPLETE
    store.save(job)
    return store, store.get(job.job_id)


def test_emit_manifest_from_persisted_job(tmp_path: Path) -> None:
    store, job = _completed_job(tmp_path)
    path = emit_manifest(job, store.root)

    assert path == store.root / job.job_id / "manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))

    # Canonical hashes came from the sidecar, not hardcoded.
    assert payload["notation_score_sha256"]
    assert payload["performance_plan_sha256"]

    # Toolchain derived from the artifacts' provenance.
    assert payload["toolchain"]["audio"]["engine"] == "fluidsynth"
    assert payload["toolchain"]["audio"]["version"] == "2.3.4"
    assert (
        payload["toolchain"]["audio"]["config"]["ffmpeg_configuration"]
        == "--enable-libvpx --enable-libopus"
    )
    assert payload["toolchain"]["sheet"]["version"] == "2.1.2"
    assert payload["toolchain"]["animation"]["engine"] == "ffmpeg"

    # The soundfont asset carries the Salamander attribution (§10 #12).
    assert payload["assets"]["soundfont"]["sha256"] == "9" * 64
    assert payload["assets"]["soundfont"]["license"] == "CC BY 3.0"
    assert payload["assets"]["soundfont"]["notice_path"] == SOUNDFONT_NOTICE_PATH

    # License obligations point at the notice files.
    assert (
        payload["license_obligations"]["salamander-grand-piano"]
        == "LICENSES/Salamander-Grand-Piano.txt"
    )


def test_emit_manifest_attributes_the_font_that_rendered(tmp_path: Path) -> None:
    """A dedicated-font render carries that font's license, not Salamander's."""
    store, job = _completed_job(tmp_path)
    job.artifacts["audio"].toolchain["soundfont"] = "MFA_Boston_1.sf2"
    store.save(job)

    emit_manifest(store.get(job.job_id), store.root)
    payload = json.loads((store.root / job.job_id / "manifest.json").read_text(encoding="utf-8"))
    soundfont = payload["assets"]["soundfont"]
    assert soundfont["name"] == "MFA Boston 1"
    assert soundfont["license"] == "CC BY 3.0"
    assert soundfont["source_url"] == "https://musical-artifacts.com/artifacts/3593"
    assert soundfont["notice_path"] == "THIRD_PARTY_NOTICES.md#mfa-boston-1-soundfont"


def test_emit_manifest_never_inherits_attribution_for_unknown_fonts(tmp_path: Path) -> None:
    """A font outside the registry shows as unverified, not as another font."""
    store, job = _completed_job(tmp_path)
    job.artifacts["audio"].toolchain["soundfont"] = "Mystery_Kit.sf2"
    store.save(job)

    emit_manifest(store.get(job.job_id), store.root)
    payload = json.loads((store.root / job.job_id / "manifest.json").read_text(encoding="utf-8"))
    soundfont = payload["assets"]["soundfont"]
    assert soundfont["name"] == "Mystery_Kit.sf2"
    assert soundfont["license"] == "unverified"
    assert soundfont["source_url"] == ""


def test_emit_manifest_records_installed_dependency_versions(tmp_path: Path) -> None:
    store, job = _completed_job(tmp_path)
    emit_manifest(job, store.root)
    payload = json.loads((store.root / job.job_id / "manifest.json").read_text(encoding="utf-8"))
    deps = payload["dependencies"]
    assert deps["music21"] != "unknown"  # music21 is a hard dependency


def test_emit_manifest_without_sidecar_leaves_hashes_empty(tmp_path: Path) -> None:
    store = JobStorage(tmp_path)
    job = store.create("p")
    job.input_spec = _spec()
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
            toolchain={"engine": "fluidsynth", "version": "2.3.4"},
        ),
    )
    job.state = JobState.COMPLETE
    store.save(job)
    job = store.get(job.job_id)

    path = emit_manifest(job, store.root)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["notation_score_sha256"] == ""
    assert payload["assets"]["soundfont"]["sha256"] == ""


def test_emit_manifest_rejects_incomplete_job(tmp_path: Path) -> None:
    store = JobStorage(tmp_path)
    job = store.create("p")
    job.input_spec = _spec()
    with pytest.raises(ValueError, match="not complete"):
        emit_manifest(job, store.root)
