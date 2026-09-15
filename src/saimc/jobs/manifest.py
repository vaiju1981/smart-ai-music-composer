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
    quality: Mapping[str, Any] | None = None
    """The musical-quality scorecard for this render (`saimc.quality`).

    Optional so a manifest can still be emitted when the engine-output
    sidecar is unavailable; the field is omitted rather than filled
    with a placeholder that would read as a measurement.
    """


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

    payload: dict[str, Any] = {
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
    if inputs.quality is not None:
        payload["quality"] = dict(inputs.quality)
    return payload


def write_manifest(job: Job, inputs: ManifestInputs, jobs_root: Path) -> Path:
    """Build the manifest and write it under `{jobs_root}/{job_id}/manifest.json`.

    Returns the path written.
    """
    payload = build_manifest(job, inputs)
    out_path = jobs_root / job.job_id / MANIFEST_FILENAME
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(canonical_dumps(payload) + "\n", encoding="utf-8")
    return out_path


def emit_manifest(job: Job, jobs_root: Path) -> Path:
    """Emit a completed job's manifest from its persisted records alone.

    The worker calls this the moment a job reaches `complete`. Every
    input is derived from data already on disk: the canonical hashes
    come from the engine-output sidecar, the toolchain block from the
    artifacts' `toolchain` dicts (written by the render stages), the
    soundfont asset from the attribution module, and the dependency
    versions from the installed distributions.

    Raises the same `ValueError`s as `build_manifest` if the job is
    not a completed job with artifacts.
    """
    from importlib.metadata import PackageNotFoundError
    from importlib.metadata import version as dist_version

    from saimc.compose.serialization import read_engine_output
    from saimc.quality import score_piece
    from saimc.render.attribution import (
        attribution_for_soundfont,
        license_obligations,
    )

    notation_score_sha = ""
    performance_plan_sha = ""
    quality: dict[str, Any] | None = None
    if job.input_spec is not None:
        sidecar = jobs_root / job.job_id / "engine_output.json"
        if sidecar.exists():
            output = read_engine_output(sidecar)
            notation_score_sha = output.notation_score.compute_hash()
            performance_plan_sha = output.performance_plan.compute_hash()
            # The scorecard is recorded per render so quality can be
            # trended across a corpus without re-composing anything.
            measured = score_piece(output.notation_score, piece=job.job_id)
            quality = measured.entry()

    toolchain_by_kind: dict[str, ToolchainRecord] = {}
    for kind, art in job.artifacts.items():
        tc = dict(art.toolchain)
        if not tc:
            continue
        if kind == "audio":
            toolchain_by_kind[kind] = ToolchainRecord(
                engine=tc.get("engine", "fluidsynth"),
                version=tc.get("version", "unknown"),
                build_sha=tc.get("build_sha", ""),
                config={
                    "soundfont": tc.get("soundfont", ""),
                    "soundfont_sha256": tc.get("soundfont_sha256", ""),
                    "ffmpeg_version": tc.get("ffmpeg_version", ""),
                    "ffmpeg_build_sha": tc.get("ffmpeg_build_sha", ""),
                    "ffmpeg_configuration": tc.get("ffmpeg_configuration", ""),
                },
            )
        elif kind == "sheet":
            toolchain_by_kind[kind] = ToolchainRecord(
                engine=tc.get("engine", "opensheetmusicdisplay"),
                version=tc.get("version", "unknown"),
                build_sha="",
                config={"renderer": tc.get("renderer", "")},
            )
        elif kind == "animation":
            toolchain_by_kind[kind] = ToolchainRecord(
                engine=tc.get("engine", "ffmpeg"),
                version=tc.get("version", "unknown"),
                build_sha=tc.get("build_sha", ""),
                config={
                    "configuration": tc.get("configuration", ""),
                    "video_codec": tc.get("video_codec", ""),
                    "audio_codec": tc.get("audio_codec", ""),
                    "dimensions": tc.get("dimensions", ""),
                    "fps": tc.get("fps", ""),
                },
            )

    assets: dict[str, AssetRecord] = {}
    audio_record = job.artifacts.get("audio")
    audio_tc = dict(audio_record.toolchain) if audio_record is not None else {}
    if audio_tc:
        # Attribute the font that actually rendered, not a default —
        # dedicated fonts (MFA Boston 1, 105-Sitar, Wetthasinghe's
        # Harmonium) carry their own licenses.
        sf = attribution_for_soundfont(str(audio_tc.get("soundfont", "")))
        assets["soundfont"] = AssetRecord(
            name=sf.name,
            version=sf.version,
            sha256=audio_tc.get("soundfont_sha256", ""),
            license=sf.license,
            source_url=sf.source_url,
            notice_path=sf.notice_path,
        )

    dependencies: dict[str, str] = {}
    for dist in ("music21", "mido", "Pillow"):
        try:
            dependencies[dist] = dist_version(dist)
        except PackageNotFoundError:  # pragma: no cover - render extra absent
            dependencies[dist] = "unknown"

    return write_manifest(
        job,
        ManifestInputs(
            notation_score_sha256=notation_score_sha,
            performance_plan_sha256=performance_plan_sha,
            completed_at=job.updated_at,
            toolchain_by_kind=toolchain_by_kind,
            assets=assets,
            dependencies=dependencies,
            license_obligations=license_obligations(),
            quality=quality,
        ),
        jobs_root,
    )


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
    "emit_manifest",
    "write_manifest",
]
