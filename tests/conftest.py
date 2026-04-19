"""Shared pytest fixtures."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from common.config import get_settings


@pytest.fixture(autouse=True)
def _reset_settings_cache() -> Iterator[None]:
    """Clear the cached :class:`Settings` before and after each test.

    Prevents tests that tweak environment variables from leaking into each other.
    """
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def isolated_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Path]:
    """Point ``cache_dir`` and ``results_dir`` at a tmp dir for test isolation.

    Also disables the mock-backend flag so tests that mock the client behave
    predictably, and strips any ``LLMEVAL_*`` overrides the *parent* shell may
    have exported (real GPU runs commonly export ``LLMEVAL_MODEL_NAME`` and
    ``LLMEVAL_GEN_*``; we don't want those leaking into unit tests that are
    supposed to exercise the code's built-in defaults).
    """
    monkeypatch.setenv("LLMEVAL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LLMEVAL_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("LLMEVAL_MOCK_BACKEND", "false")

    # Make the test environment hermetic w.r.t. any LLMEVAL_* that the
    # contributor / notebook / CI runner may have pre-exported.
    for leaky in (
        "LLMEVAL_MODEL_NAME",
        "LLMEVAL_GEN_TEMPERATURE",
        "LLMEVAL_GEN_TOP_P",
        "LLMEVAL_GEN_TOP_K",
        "LLMEVAL_CACHE_VERSION",
    ):
        monkeypatch.delenv(leaky, raising=False)

    # Avoid touching any real .env on the contributor's machine.
    monkeypatch.chdir(tmp_path)
    os.environ.setdefault("LLMEVAL_LOG_FORMAT", "console")
    yield tmp_path
