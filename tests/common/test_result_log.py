"""Tests for ``common.result_log``."""

from __future__ import annotations

from pathlib import Path

import pytest

from common.config import get_settings
from common.result_log import ResultLogEntry, log_result, read_results


@pytest.fixture
def _results_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("LLMEVAL_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("LLMEVAL_CACHE_DIR", str(tmp_path / "cache"))
    get_settings.cache_clear()
    return tmp_path


def test_log_result_writes_csv_with_header(_results_env: Path) -> None:
    entry = ResultLogEntry(
        model="test/model",
        task="mmlu",
        method="baseline",
        metric="acc",
        value=0.42,
        seed=42,
        limit=100,
        stderr=0.01,
        wall_s=12.5,
        mock_backend=True,
        notes="first smoke run",
    )
    out = log_result(entry)
    assert out.run_id == 1
    assert out.timestamp  # ISO string
    assert (_results_env / "results" / "result-log.csv").exists()
    assert (_results_env / "results" / "result-log.md").exists()

    rows = read_results()
    assert len(rows) == 1
    assert rows[0]["model"] == "test/model"
    assert rows[0]["task"] == "mmlu"
    assert float(rows[0]["value"]) == pytest.approx(0.42)
    assert rows[0]["mock_backend"] == "true"


def test_log_result_appends_and_increments_run_id(_results_env: Path) -> None:
    for i in range(3):
        log_result(
            ResultLogEntry(
                model="test/model",
                task="mmlu",
                method=f"variant_{i}",
                metric="acc",
                value=0.40 + 0.01 * i,
                seed=42,
            )
        )
    rows = read_results()
    assert len(rows) == 3
    assert [int(r["run_id"]) for r in rows] == [1, 2, 3]


def test_markdown_regenerated_on_every_append(_results_env: Path) -> None:
    log_result(
        ResultLogEntry(
            model="m", task="t", method="baseline", metric="acc", value=0.5, seed=1
        )
    )
    md = (_results_env / "results" / "result-log.md").read_text()
    assert "| 1 " in md
    assert "baseline" in md
    assert "0.5000" in md
    log_result(
        ResultLogEntry(
            model="m",
            task="t",
            method="better",
            metric="acc",
            value=0.55,
            seed=1,
            delta=0.05,
        )
    )
    md = (_results_env / "results" / "result-log.md").read_text()
    # Second run should be rendered as a separate row with delta filled in.
    assert "| 2 " in md
    assert "+0.0500" in md


def test_read_results_returns_empty_when_no_runs(_results_env: Path) -> None:
    assert read_results() == []
