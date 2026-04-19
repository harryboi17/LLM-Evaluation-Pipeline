"""Tests for the ``guardrails.validate`` CLI entry point."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from common.config import get_settings
from guardrails.validate import main


@pytest.fixture
def _mock_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("LLMEVAL_MOCK_BACKEND", "true")
    monkeypatch.setenv("LLMEVAL_CACHE_DIR", str(tmp_path / "cache"))
    get_settings.cache_clear()
    return tmp_path


def test_validate_cli_all_valid_exits_zero(
    _mock_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    jsonl = _mock_env / "outputs.jsonl"
    jsonl.write_text("\n".join(json.dumps({"output": v}) for v in ["Paris", "Tokyo", "56"]))
    report = _mock_env / "report.json"

    rc = main(["validate", str(jsonl), "--short-answer", "--report", str(report)])

    assert rc == 0
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary == {"valid": 3, "invalid": 0}
    assert report.exists()


def test_validate_cli_any_invalid_exits_one(
    _mock_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    jsonl = _mock_env / "outputs.jsonl"
    jsonl.write_text("\n".join(json.dumps({"output": v}) for v in ["Paris", "a b c d"]))

    rc = main(["validate", str(jsonl), "--short-answer"])

    assert rc == 1
    summary = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert summary == {"valid": 1, "invalid": 1}


def test_determinism_cli_mock_backend_identical(
    _mock_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    report = _mock_env / "det.json"
    rc = main(
        [
            "determinism",
            "What is 2+2?",
            "--n-runs",
            "3",
            "--max-tokens",
            "8",
            "--report",
            str(report),
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert payload["identical"] is True
    assert payload["first_divergence_at"] is None
    assert report.exists()
    full = json.loads(report.read_text())
    assert len(full["completions"]) == 3
