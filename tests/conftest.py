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
    predictably.
    """
    monkeypatch.setenv("LLMEVAL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LLMEVAL_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("LLMEVAL_MOCK_BACKEND", "false")
    # Avoid touching any real .env on the contributor's machine.
    monkeypatch.chdir(tmp_path)
    os.environ.setdefault("LLMEVAL_LOG_FORMAT", "console")
    yield tmp_path
