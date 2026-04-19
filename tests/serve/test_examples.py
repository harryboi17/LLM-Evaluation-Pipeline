"""Tests for ``serve.examples.demo`` and ``serve.examples.concurrent_demo``."""

from __future__ import annotations

import json

import pytest

from common.config import get_settings
from serve.examples.concurrent_demo import main as concurrent_main
from serve.examples.demo import main as demo_main


@pytest.fixture
def _mock_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMEVAL_MOCK_BACKEND", "true")
    get_settings.cache_clear()


def test_demo_main_runs_three_prompts_against_mock(
    _mock_backend: None, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = demo_main()
    assert rc == 0
    captured = capsys.readouterr()
    assert "--- short_greedy" in captured.out
    assert "--- long_with_stops" in captured.out
    assert "--- stream_demo" in captured.out

    summary_line = captured.out.strip().splitlines()[-1]
    summary = json.loads(summary_line)
    assert summary["demo"] == "serve.examples.demo"
    labels = [r["label"] for r in summary["results"]]
    assert labels == ["short_greedy", "long_with_stops", "stream_demo"]
    # stream entry records TTFT, non-stream entries record token counts.
    stream_entry = next(r for r in summary["results"] if r["label"] == "stream_demo")
    assert stream_entry["mode"] == "stream"
    assert stream_entry["ttft_s"] is not None


def test_concurrent_demo_main_reports_speedup_fields(
    _mock_backend: None, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = concurrent_main(["--concurrency", "4", "--max-tokens", "8"])
    assert rc == 0
    captured = capsys.readouterr()
    summary = json.loads(captured.out)
    assert summary["concurrency"] == 4
    assert summary["max_tokens"] == 8
    assert "sequential_wall_s" in summary
    assert "concurrent_wall_s" in summary
    assert summary["sequential_wall_s"] >= 0
    assert summary["concurrent_wall_s"] >= 0
