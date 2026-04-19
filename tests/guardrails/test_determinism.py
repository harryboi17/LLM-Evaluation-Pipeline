"""Tests for ``guardrails.validate.verify_determinism``."""

from __future__ import annotations

from pathlib import Path

import pytest

from common.config import get_settings
from common.errors import DeterminismError
from guardrails.validate import verify_determinism


@pytest.fixture
def _mock_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("LLMEVAL_MOCK_BACKEND", "true")
    monkeypatch.setenv("LLMEVAL_CACHE_DIR", str(tmp_path / "cache"))
    get_settings.cache_clear()
    return tmp_path


async def test_verify_determinism_mock_reports_identical(_mock_env: Path) -> None:
    """Mock backend returns a deterministic canned string, so 5 runs must match."""
    report = await verify_determinism("What is 2+2?", n_runs=5, max_tokens=8)
    assert report.n_runs == 5
    assert report.identical is True
    assert report.first_divergence_at is None
    assert len(set(report.completions)) == 1


async def test_verify_determinism_requires_at_least_two_runs(_mock_env: Path) -> None:
    with pytest.raises(DeterminismError):
        await verify_determinism("prompt", n_runs=1)


async def test_verify_determinism_detects_injected_drift(
    _mock_env: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Patch the mock generator so its output depends on a call counter."""
    calls = {"n": 0}

    def flaky_mock(prompt: str, max_tokens: int, *, echo: bool = False, logprobs: int | None = None):  # type: ignore[no-untyped-def]
        from common.types import GenerationResult

        calls["n"] += 1
        text = f"deterministic-{prompt}" if calls["n"] != 3 else f"DIFFERENT-{prompt}"
        return GenerationResult(
            text=text,
            prompt_tokens=1,
            completion_tokens=1,
            finish_reason="stop",
        )

    monkeypatch.setattr("common.vllm_client._mock_generation", flaky_mock)

    report = await verify_determinism("x", n_runs=5, max_tokens=4)
    assert report.identical is False
    assert report.first_divergence_at == 2  # 0-based index of 3rd run
    # Completions were collected in order even when drift was detected.
    assert len(report.completions) == 5
