"""Tests for ``serve.client``."""

from __future__ import annotations

import json

import pytest

from common.config import get_settings
from serve.client import main


@pytest.fixture
def _mock_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMEVAL_MOCK_BACKEND", "true")
    get_settings.cache_clear()


def test_client_main_generate_prints_completion_and_summary(
    _mock_backend: None, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(["hello world", "--max-tokens", "8"])
    assert rc == 0
    captured = capsys.readouterr()
    # stdout: the completion text
    assert "[mock completion for prompt" in captured.out
    # stderr (last line): a JSON summary with mode=generate
    summary_line = captured.err.strip().splitlines()[-1]
    summary = json.loads(summary_line)
    assert summary["mode"] == "generate"
    assert summary["finish_reason"] == "stop"
    assert "wall_s" in summary


def test_client_main_stream_emits_tokens_and_summary(
    _mock_backend: None, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(["hello", "--stream", "--max-tokens", "8"])
    assert rc == 0
    captured = capsys.readouterr()
    assert "[mock completion for prompt" in captured.out
    summary_line = captured.err.strip().splitlines()[-1]
    summary = json.loads(summary_line)
    assert summary["mode"] == "stream"
    assert summary["prompt_chars"] == len("hello")
    assert summary["completion_chars"] > 0


def test_client_main_rejects_no_prompt() -> None:
    with pytest.raises(SystemExit):
        main([])
