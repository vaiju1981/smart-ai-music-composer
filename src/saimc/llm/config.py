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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from saimc.llm.ollama import OllamaAdapter

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


def build_ollama_adapter() -> OllamaAdapter | None:
    """The configured adapter, or `None` when there is no configuration to build one from.

    `None` rather than a stub client, because the two callers do different
    things with the absence and both have to tell it apart from a model that
    answered nothing: the worker wraps it in a client whose every call refuses
    with a named reason, and the session API leaves the slot empty, where every
    deterministic tool still runs and only the conductor's own turn needs a
    model. A stub here would be a second source of `llm_not_configured`, and
    two places deciding what "not configured" means is one too many.

    The adapter import stays deferred, as it always has been in the worker that
    used to hold this: `saimc.llm.ollama` pulls in the provider's transport, and
    a process with no configuration should not pay for it.
    """
    try:
        config = load_llm_config()
    except ValueError:
        return None
    from saimc.llm.ollama import OllamaAdapter

    return OllamaAdapter(base_url=config.base_url, model=config.model, api_key=config.api_key)


__all__ = [
    "CONFIG_ENV_VAR",
    "CONFIG_FILENAME",
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "LLMConfig",
    "build_ollama_adapter",
    "load_llm_config",
]
