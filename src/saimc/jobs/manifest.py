"""Artifact manifest emission per `docs/roadmap.md` §9.

Every completed job emits a machine-readable `manifest.json` alongside
the artifacts. It is the reproducibility and provenance record for that
specific render.

Required fields (per §9):

- `job_id`, `created_at`, `completed_at`
- `input_spec`: the full CompositionSpec (schema_version included) and its sha256
- `seed`, `engine_version`
- `parser_source`: `llm` | `fallback`, plus the model identifier when `llm`
- `notation_score_sha256`, `performance_plan_sha256`
- `artifacts`: per-artifact `{kind, container, codec, path, sha256, size_bytes}`
- `toolchain`: per-artifact `{engine, version, build_sha, config}`
- `assets`: `{soundfont: {…}, notation_font: {…}}`
- `dependencies`: pinned versions of every library involved in the render
- `license_obligations`: pointers into the distribution's `LICENSES/`
  directory for every LGPL/LGPL-conditional component, plus the
  FFmpeg `--configure` flag set used to build the bundled FFmpeg.

Phase 1 emits a JSON file at `{jobs_dir}/{job_id}/manifest.json`. The
Phase 4 §9 invariant is that the canonical artifacts (spec, NotationScore,
PerformancePlan) are reproducible hash-for-hash against the manifest's
recorded `toolchain` + `dependencies` + `seed`, while media artifacts
are verified semantically per §8.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from saimc.canonical import canonical_dumps
from saimc.jobs.storage import Job

MANIFEST_FILENAME = "manifest.json"


@dataclass(frozen=True)
class ToolchainRecord:
    """How a specific artifact was produced."""

    engine: str  # "fluidsynth" | "osmd" | "ffmpeg" | "music21" | …
    version: str
    build_sha: str
    config: Mapping[str, str]


@dataclass(frozen=True)
class AssetRecord:
    """Provenance for one shipped asset (soundfont, notation font, etc.)."""

    name: str
    version: str
    sha256: str
    license: str
    source_url: str
    notice_path: str


@dataclass(frozen=True)
class ManifestInputs:
    """Everything needed to render one job's manifest.

    `toolchain_by_kind` and `assets` are passed in by the renderer so
    the manifest module itself does not need to know about FluidSynth,
    OSMD, FFmpeg, etc. — it just records what it's told.
    """

    notation_score_sha256: str
    performance_plan_sha256: str
    completed_at: datetime
    toolchain_by_kind: Mapping[str, ToolchainRecord] = field(default_factory=dict)
    assets: Mapping[str, AssetRecord] = field(default_factory=dict)
    dependencies: Mapping[str, str] = field(default_factory=dict)
    license_obligations: Mapping[str, str] = field(default_factory=dict)


def build_manifest(job: Job, inputs: ManifestInputs) -> dict[str, Any]:
    """Construct the manifest payload as a plain dict.

    Raises `ValueError` if the job is not in a state from which a
    manifest can be emitted (must have a parsed spec + at least one
    artifact, and the job must be `complete`).
    """
    if job.input_spec is None:
        raise ValueError(f"Job {job.job_id} has no parsed spec; cannot emit manifest.")
    if job.state.name != "COMPLETE":
        raise ValueError(f"Job {job.job_id} is in state {job.state.value}, not complete.")
    if not job.artifacts:
        raise ValueError(f"Job {job.job_id} has no artifacts; nothing to manifest.")

    spec_payload = job.input_spec.model_dump(mode="json")
    spec_sha = _sha256_str(spec_payload)

    artifacts_section: dict[str, Any] = {}
    for kind, art in job.artifacts.items():
        artifacts_section[kind] = {
            "kind": art.kind,
            "container": art.container,
            "codec": art.codec,
            "path": art.path,
            "sha256": art.sha256,
            "size_bytes": art.size_bytes,
        }

    toolchain_section = {
        kind: {
            "engine": tc.engine,
            "version": tc.version,
            "build_sha": tc.build_sha,
            "config": dict(tc.config),
        }
        for kind, tc in inputs.toolchain_by_kind.items()
    }

    assets_section = {
        kind: {
            "name": a.name,
            "version": a.version,
            "sha256": a.sha256,
            "license": a.license,
            "source_url": a.source_url,
            "notice_path": a.notice_path,
        }
        for kind, a in inputs.assets.items()
    }

    return {
        "job_id": job.job_id,
        "created_at": job.created_at.isoformat(),
        "completed_at": inputs.completed_at.isoformat(),
        "input_spec": {"spec": spec_payload, "sha256": spec_sha},
        "seed": job.seed,
        "engine_version": job.engine_version,
        "parser_source": job.parser_source,
        "notation_score_sha256": inputs.notation_score_sha256,
        "performance_plan_sha256": inputs.performance_plan_sha256,
        "artifacts": artifacts_section,
        "toolchain": toolchain_section,
        "assets": assets_section,
        "dependencies": dict(inputs.dependencies),
        "license_obligations": dict(inputs.license_obligations),
    }


def write_manifest(job: Job, inputs: ManifestInputs, jobs_root: Path) -> Path:
    """Build the manifest and write it under `{jobs_root}/{job_id}/manifest.json`.

    Returns the path written.
    """
    payload = build_manifest(job, inputs)
    out_path = jobs_root / job.job_id / MANIFEST_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(canonical_dumps(payload) + "\n", encoding="utf-8")
    return out_path


def _sha256_str(payload: Any) -> str:
    """Stable SHA-256 over the canonical serialization of `payload`."""
    from saimc.canonical import canonical_sha256

    return canonical_sha256(payload)


__all__ = [
    "MANIFEST_FILENAME",
    "AssetRecord",
    "ManifestInputs",
    "ToolchainRecord",
    "build_manifest",
    "write_manifest",
]
