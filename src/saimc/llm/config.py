"""LLM configuration loading — config file, environment, or defaults.

Per `docs/roadmap.md` §3, Ollama is the Phase 1 LLM provider. This module
resolves the adapter settings so the app works with zero setup:

- `[llm]` section of a `saimc.toml` file (repo root by default; a
  different file can be selected with `SAIMC_CONFIG`).
- `OLLAMA_BASE_URL` / `OLLAMA_MODEL` / `OLLAMA_API_KEY` environment
  variables, which override the file per key.
- Built-in defaults: local Ollama, the Phase 1 model, no API key.

The API key is optional in every path — only attached when present.
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path

CONFIG_ENV_VAR = "SAIMC_CONFIG"
CONFIG_FILENAME = "saimc.toml"

DEFAULT_BASE_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "gemma4:31b-cloud"


@dataclass(frozen=True)
class LLMConfig:
    """Resolved LLM adapter settings."""

    base_url: str
    model: str
    api_key: str | None = None


def _section_from_file(path: Path) -> dict[str, str]:
    with path.open("rb") as fh:
        payload = tomllib.load(fh)
    section = payload.get("llm", {})
    if not isinstance(section, dict):
        raise ValueError(f"[llm] section in {path} must be a table")
    return {str(k): str(v) for k, v in section.items() if v != ""}


def _config_path() -> Path:
    """Resolve the config file path: SAIMC_CONFIG, else ./saimc.toml."""
    override = os.environ.get(CONFIG_ENV_VAR)
    if override:
        path = Path(override)
        if not path.is_file():
            raise ValueError(f"{CONFIG_ENV_VAR} points at a missing config file: {override}")
        return path
    return Path(CONFIG_FILENAME)


def load_llm_config() -> LLMConfig:
    """Resolve LLM settings: env var > config file > built-in default."""
    file_values: dict[str, str] = {}
    path = _config_path()
    if path.is_file():
        file_values = _section_from_file(path)

    base_url = os.environ.get("OLLAMA_BASE_URL") or file_values.get("base_url") or DEFAULT_BASE_URL
    model = os.environ.get("OLLAMA_MODEL") or file_values.get("model") or DEFAULT_MODEL
    api_key = os.environ.get("OLLAMA_API_KEY") or file_values.get("api_key") or None
    return LLMConfig(base_url=base_url.rstrip("/"), model=model, api_key=api_key)


__all__ = [
    "CONFIG_ENV_VAR",
    "CONFIG_FILENAME",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "LLMConfig",
    "load_llm_config",
]
