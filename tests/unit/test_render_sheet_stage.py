"""Tests for the `render_sheet_stage` worker helper."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest

from saimc.jobs.stages import render_sheet_stage, transition_to
from saimc.jobs.state import JobState
from saimc.jobs.storage import JobStorage
from saimc.render.sheet import SheetRenderError, SheetRenderErrorCode
from saimc.spec import CompositionSpec, Mood


def _spec() -> CompositionSpec:
    return CompositionSpec(mood=Mood.CALMING, seed=42, duration_seconds=180)


def _prepared_job(tmp_path: Path) -> tuple[JobStorage, Any]:
    """A job with a spec and engine-output sidecar on disk, in VALIDATING."""
    from saimc.compose.engine import compose
    from saimc.compose.serialization import write_engine_output

    store = JobStorage(tmp_path)
    job = store.create("job1")
    job.input_spec = _spec()
    output = compose(job.input_spec)
    write_engine_output(store.job_dir(job.job_id) / "engine_output.json", output)

    for state in (JobState.PARSING, JobState.COMPOSING, JobState.VALIDATING):
        transition_to(job, state)
    store.save(job)
    return store, store.get(job.job_id)


def test_no_spec_returns_failed(tmp_path: Path) -> None:
    store = JobStorage(tmp_path)
    job = store.create("job1")
    result = render_sheet_stage(job, store)
    assert result.next_state == JobState.FAILED
    assert result.error is not None
    assert result.error.error_code == "no_spec"
    assert result.error.stage == "rendering_sheet"


def test_missing_sidecar_returns_failed(tmp_path: Path) -> None:
    store = JobStorage(tmp_path)
    job = store.create("job1")
    job.input_spec = _spec()
    result = render_sheet_stage(job, store)
    assert result.next_state == JobState.FAILED
    assert result.error is not None
    assert result.error.error_code == "engine_output_missing"


def test_advances_and_attaches_sheet(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Happy path: sidecar on disk, mocked render_sheet, artifact attached."""
    store, job = _prepared_job(tmp_path)

    def _fake_render_sheet(musicxml_path: Path, out_svg_path: Path, **_kwargs: Any) -> Any:
        from saimc.render.sheet import SheetArtifact

        out_svg_path.write_bytes(b"<svg>ok</svg>")
        return SheetArtifact(
            sheet_path=out_svg_path,
            container="svg",
            codec="svg",
            sha256=hashlib.sha256(b"<svg>ok</svg>").hexdigest(),
            size_bytes=12,
            osmd_version="stub",
        )

    monkeypatch.setattr("saimc.render.sheet.render_sheet", _fake_render_sheet)

    result = render_sheet_stage(job, store)
    assert result.error is None
    assert result.next_state == JobState.RENDERING_ANIMATION

    # The worker persists the job after the stage returns; mirror that
    # here so the reload below sees the attached artifact.
    store.save(job)
    job = store.get(job.job_id)
    sheet = job.artifacts["sheet"]
    assert sheet.container == "svg"
    assert sheet.sha256 == hashlib.sha256(b"<svg>ok</svg>").hexdigest()

    # The MusicXML export lands next to the SVG for provenance.
    musicxml = store.ensure_artifact_dir(job.job_id) / "score.musicxml"
    assert musicxml.is_file()
    assert "<score-partwise" in musicxml.read_text(encoding="utf-8")


def test_render_error_returns_failed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    store, job = _prepared_job(tmp_path)

    def _fake_fail(musicxml_path: Path, out_svg_path: Path, **_kwargs: Any) -> Any:
        raise SheetRenderError(
            code=SheetRenderErrorCode.RENDER_FAILED,
            message="OSMD exploded",
        )

    monkeypatch.setattr("saimc.render.sheet.render_sheet", _fake_fail)

    result = render_sheet_stage(job, store)
    assert result.next_state == JobState.FAILED
    assert result.error is not None
    assert result.error.error_code == SheetRenderErrorCode.RENDER_FAILED
    assert result.error.stage == "rendering_sheet"
