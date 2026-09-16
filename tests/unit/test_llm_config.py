"""Unit tests for LLM configuration loading (saimc.toml + env + defaults)."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from saimc.llm.config import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    load_llm_config,
)

OLLAMA_ENV_VARS = ("OLLAMA_BASE_URL", "OLLAMA_MODEL", "OLLAMA_API_KEY", "SAIMC_CONFIG")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Hermetic: no OLLAMA_* env, no repo-root saimc.toml in play."""
    for var in OLLAMA_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)


class TestDefaults:
    def test_defaults_with_no_file_and_no_env(self) -> None:
        cfg = load_llm_config()
        assert cfg.base_url == DEFAULT_BASE_URL
        assert cfg.model == DEFAULT_MODEL
        assert cfg.api_key is None


class TestConfigFile:
    def test_reads_llm_section_from_repo_config(self, tmp_path: Path) -> None:
        (tmp_path / "saimc.toml").write_text(
            textwrap.dedent(
                """
                [llm]
                base_url = "https://ollama.example.invalid"
                model = "test-model"
                """
            ),
            encoding="utf-8",
        )
        cfg = load_llm_config()
        assert cfg.base_url == "https://ollama.example.invalid"
        assert cfg.model == "test-model"
        assert cfg.api_key is None

    def test_optional_api_key_in_file(self, tmp_path: Path) -> None:
        (tmp_path / "saimc.toml").write_text(
            '[llm]\nmodel = "m"\napi_key = "test-key"\n', encoding="utf-8"
        )
        assert load_llm_config().api_key == "test-key"

    def test_saimc_config_env_selects_a_different_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        other = tmp_path / "other.toml"
        other.write_text('[llm]\nmodel = "other-model"\n', encoding="utf-8")
        monkeypatch.setenv("SAIMC_CONFIG", str(other))
        assert load_llm_config().model == "other-model"

    def test_missing_saimc_config_env_is_an_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("SAIMC_CONFIG", "/nonexistent/saimc.toml")
        with pytest.raises(ValueError, match="SAIMC_CONFIG"):
            load_llm_config()

    def test_empty_values_in_file_fall_through_to_defaults(self, tmp_path: Path) -> None:
        (tmp_path / "saimc.toml").write_text('[llm]\nmodel = ""\n', encoding="utf-8")
        assert load_llm_config().model == DEFAULT_MODEL


class TestPrecedence:
    def test_env_overrides_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        (tmp_path / "saimc.toml").write_text(
            '[llm]\nbase_url = "https://file.example.invalid"\nmodel = "file-model"\n',
            encoding="utf-8",
        )
        monkeypatch.setenv("OLLAMA_BASE_URL", "https://env.example.invalid")
        cfg = load_llm_config()
        assert cfg.base_url == "https://env.example.invalid"
        assert cfg.model == "file-model"  # env only overrides its own key


class TestWorkerWiring:
    def test_worker_builds_adapter_from_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from saimc.jobs.worker import _build_default_llm_client
        from saimc.llm.ollama import OllamaAdapter

        (tmp_path / "saimc.toml").write_text('[llm]\nmodel = "wired-model"\n', encoding="utf-8")
        client = _build_default_llm_client()
        assert isinstance(client, OllamaAdapter)
        assert client.model_identifier == "wired-model"

    def test_invalid_config_falls_back_to_noop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from saimc.jobs import worker
        from saimc.spec import SpecError

        def boom() -> None:
            raise ValueError("bad config")

        monkeypatch.setattr("saimc.llm.config.load_llm_config", boom)
        client = worker._build_default_llm_client()
        import asyncio

        result = asyncio.run(client.parse(None))  # type: ignore[arg-type]
        assert isinstance(result.error, SpecError)
        assert result.error.error_code == "llm_not_configured"
