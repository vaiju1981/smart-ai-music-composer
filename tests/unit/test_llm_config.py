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


class TestTheModelOverride:
    """`build_ollama_adapter(model=…)`, which is how a sweep picks a candidate.

    §10 #3 step 3 runs the corpus against several models, and the host, the key
    and the timeout stay the configured ones for all of them — so the override
    is one keyword on the existing factory rather than a second construction
    site that would have to re-read the configuration and decide for itself
    what "not configured" means.
    """

    def test_an_explicit_model_wins_over_the_configured_one(self, tmp_path: Path) -> None:
        from saimc.llm.config import build_ollama_adapter

        (tmp_path / "saimc.toml").write_text('[llm]\nmodel = "configured-model"\n', encoding="utf-8")
        client = build_ollama_adapter(model="sweep-candidate")
        assert client is not None
        assert client.model_identifier == "sweep-candidate"

    def test_everything_else_still_comes_from_the_configuration(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The override moves the tag and nothing else: not the host, not the key.

        Asserted against the construction call rather than against the built
        adapter, because the host is not part of the adapter's public surface
        and reaching into `_base_url` would read a spelling rather than the
        contract. This *is* the contract: three keywords, one of them
        overridden, so a version that rebuilt the adapter from the override
        alone fails here.
        """
        import saimc.llm.ollama as ollama
        from saimc.llm.config import build_ollama_adapter

        (tmp_path / "saimc.toml").write_text(
            '[llm]\nmodel = "configured-model"\n'
            'base_url = "https://host.example.invalid"\n'
            'api_key = "test-key"\n',
            encoding="utf-8",
        )
        calls: list[dict[str, object]] = []

        class _Recorder:
            def __init__(self, **kwargs: object) -> None:
                calls.append(dict(kwargs))

        monkeypatch.setattr(ollama, "OllamaAdapter", _Recorder)
        assert build_ollama_adapter(model="sweep-candidate") is not None
        assert calls == [
            {
                "base_url": "https://host.example.invalid",
                "model": "sweep-candidate",
                "api_key": "test-key",
            }
        ]

    def test_no_override_keeps_the_configured_model(self, tmp_path: Path) -> None:
        from saimc.llm.config import build_ollama_adapter

        (tmp_path / "saimc.toml").write_text('[llm]\nmodel = "configured-model"\n', encoding="utf-8")
        client = build_ollama_adapter()
        assert client is not None
        assert client.model_identifier == "configured-model"

    def test_an_empty_override_is_not_an_override(self, tmp_path: Path) -> None:
        """An empty string would otherwise become an adapter with no model."""
        from saimc.llm.config import build_ollama_adapter

        (tmp_path / "saimc.toml").write_text('[llm]\nmodel = "configured-model"\n', encoding="utf-8")
        client = build_ollama_adapter(model="")
        assert client is not None
        assert client.model_identifier == "configured-model"

    def test_an_unreadable_config_still_yields_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The override does not smuggle in a second source of "configured"."""
        from saimc.llm.config import build_ollama_adapter

        def boom() -> None:
            raise ValueError("bad config")

        monkeypatch.setattr("saimc.llm.config.load_llm_config", boom)
        assert build_ollama_adapter(model="sweep-candidate") is None


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
