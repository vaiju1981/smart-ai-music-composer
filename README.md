# Smart AI Music Composer

Prompt-driven music composition with auditable, license-clean output. The architecture is documented in [`docs/roadmap.md`](docs/roadmap.md); this README only covers running the code.

## Status

Phase 1 architecture approved (see line 3 of `docs/roadmap.md`). Implementation is in progress.

## Quick start (development)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
ruff check src tests
ruff format --check src tests
mypy src
```

## FFmpeg

Phase 1 requires an LGPL FFmpeg shared build with `--enable-libvpx` and `--enable-libopus` (WebM/VP9+Opus per §4). The release-gate path is `scripts/build_ffmpeg.sh`, which pins FFmpeg 8.1.2 with sha256 `464beb5e7bf0c311e68b45ae2f04e9cc2af88851abb4082231742a74d97b524c`, configures it with the §10 #7 flags, captures the full configure + build logs, and runs `saimc-audit-ffmpeg` against the resulting binary.

A user-pointed-at binary is allowed **only if it passes the audit**. For example, Homebrew's stock `ffmpeg` formula is configured with `--enable-gpl` and so fails:

```
$ saimc-audit-ffmpeg
{
  "binary_path": "/opt/homebrew/bin/ffmpeg",
  "version": "8.1.2",
  "license": "gpl",
  "ok": false,
  "forbidden_flags_present": ["--enable-gpl"],
  ...
}
```

`scripts/audit_ffmpeg.py` is the same code as the `saimc-audit-ffmpeg` entry point.

## License

MIT for our code. Third-party dependencies and assets are tracked in §4 of [`docs/roadmap.md`](docs/roadmap.md); every shippable artifact must satisfy the LGPL packaging checklist there before distribution. The bundled Salamander Grand Piano sample pack is CC BY 3.0 with attribution in `LICENSES/Salamander-Grand-Piano.txt` (added during the audio-render slice).

## Repo layout

- `src/saimc/` — Python package
  - `spec.py` — `CompositionSpec` Pydantic schema
  - `canonical.py` — canonical JSON serializer for reproducibility (§6)
  - `llm/` — `LLMClient` interface, `OllamaAdapter`, fallback parser
  - `parser.py` — top-level prompt → spec orchestration
  - `compose/` — composition engine (NotationScore + PerformancePlan)
  - `jobs/` — RQ worker + state machine + manifest emission
  - `render/` — audio / sheet / animation renderers
  - `scripts/` — FFmpeg build + other ops scripts
- `tests/fixtures/parser_benchmark.jsonl` — the 100-prompt benchmark corpus (§8, `docs/parser-benchmark.md`)
- `tests/unit/` — unit tests
- `tests/acceptance/` — §8 acceptance suite (release-gate run)
- `render-service/` — Node + Playwright + OSMD/VexFlow headless notation renderer
- `scripts/` — repo-level ops scripts (benchmark runner, etc.)
- `docs/roadmap.md` — architecture and release gates
- `docs/parser-benchmark.md` — benchmark corpus process
- `MODELS.md` — model registry (release gate per §10 #3)

## Environment variables

- `OLLAMA_BASE_URL` — Ollama endpoint (Cloud or self-hosted).
- `OLLAMA_MODEL` — environment-specific model identifier.
- `OLLAMA_API_KEY` — Bearer key for Cloud; optional/absent for self-hosted.
- `SAIMC_RENDER_FFMPEG` — path to the locally-built LGPL FFmpeg (release-gate binary; required by the audio/animation renderers).
- `SAIMC_RENDER_FLUIDSYNTH` — path to the FluidSynth binary.
- `SAIMC_RENDER_SERVICE_DIR` — path to the local Node notation renderer project.
- `SAIMC_NODE_BIN` — path to Node.js (auto-detected from `PATH` or nvm by `run.sh`).
- `SAIMC_VALKEY_URL` — local Valkey URL (default `valkey://127.0.0.1:6379/0`).
- `SAIMC_JOBS_DIR` — local job-artifact directory (default `./var/jobs`).
