"""Tests for ``common.config``."""

from __future__ import annotations

from pathlib import Path

import pytest

from common.config import Settings, get_settings


def test_defaults_are_sensible(isolated_env: Path) -> None:
    """Defaults should point at a real open-weight model, not a mock/test name.

    The `isolated_env` fixture strips `LLMEVAL_MODEL_NAME` from the parent
    shell so this test sees the code-committed default, not whatever a Colab
    / GPU-box notebook happened to export upstream.

    The accepted prefix list covers the open-weight orgs we actually ship a
    sane default for. It's intentionally broad because the problem statement
    says "any open-weight model" and bumping the default to, say, Qwen or
    Gemma shouldn't require a test change.
    """
    s = Settings()
    accepted_prefixes = (
        "meta-llama/",
        "mistralai/",
        "microsoft/",     # Phi family
        "Qwen/",
        "HuggingFaceTB/", # SmolLM
        "google/",        # Gemma
    )
    assert s.model_name.startswith(accepted_prefixes), (
        f"Default model {s.model_name!r} isn't one of the known open-weight "
        f"orgs {accepted_prefixes}. If you're intentionally pointing at a "
        f"new provider, add its HF-hub prefix to this list."
    )
    assert "/" in s.model_name, "model_name must be an HF-hub-shaped id (org/name)"
    assert s.vllm_host == "127.0.0.1"
    assert s.vllm_port == 8000
    assert s.vllm_timeout_s > 0
    assert s.vllm_max_retries >= 0
    assert s.log_format in {"json", "console"}
    assert s.seed == 42
    assert s.mock_backend is False


def test_env_overrides_are_applied(monkeypatch: pytest.MonkeyPatch, isolated_env: Path) -> None:
    monkeypatch.setenv("LLMEVAL_MODEL_NAME", "test/model")
    monkeypatch.setenv("LLMEVAL_VLLM_PORT", "1234")
    monkeypatch.setenv("LLMEVAL_MOCK_BACKEND", "true")
    s = Settings()
    assert s.model_name == "test/model"
    assert s.vllm_port == 1234
    assert s.mock_backend is True


def test_vllm_base_url_composes_host_and_port(isolated_env: Path) -> None:
    s = Settings(vllm_host="localhost", vllm_port=9000)
    assert s.vllm_base_url == "http://localhost:9000/v1"


def test_get_settings_is_cached(isolated_env: Path) -> None:
    first = get_settings()
    second = get_settings()
    assert first is second


def test_get_settings_cache_clear_returns_fresh_instance(
    monkeypatch: pytest.MonkeyPatch, isolated_env: Path
) -> None:
    first = get_settings()
    get_settings.cache_clear()
    monkeypatch.setenv("LLMEVAL_VLLM_PORT", "4321")
    second = get_settings()
    assert first is not second
    assert second.vllm_port == 4321
