"""Tests for ``eval_runner.run_eval``.

Skips gracefully if ``lm-eval`` isn't installed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from common.config import get_settings

pytest.importorskip("lm_eval", reason="lm-eval extra not installed")

from eval_runner.run_eval import main


@pytest.fixture
def _mock_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("LLMEVAL_MOCK_BACKEND", "true")
    monkeypatch.setenv("LLMEVAL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LLMEVAL_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("LLMEVAL_MODEL_NAME", "mock/test-model")
    get_settings.cache_clear()
    return tmp_path


def test_main_writes_outputs_when_harness_succeeds(
    _mock_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Stub out ``simple_evaluate`` so we don't actually download datasets."""
    fake_results: dict[str, Any] = {
        "results": {
            "custom_qa": {
                "exact_match,none": 0.42,
                "exact_match_stderr,none": 0.05,
                "alias": "custom_qa",
            }
        },
    }

    with patch("eval_runner.run_eval._run_harness", return_value=fake_results):
        rc = main(["--task", "custom_qa", "--limit", "5", "--method", "pipeline_smoke"])

    assert rc == 0
    results_dir = _mock_env / "results"
    assert (results_dir / "custom_qa.json").exists()
    assert (results_dir / "run_meta.json").exists()
    summary_md = (results_dir / "summary.md").read_text()
    assert "custom_qa" in summary_md
    assert "exact_match" in summary_md

    run_meta = json.loads((results_dir / "run_meta.json").read_text())
    assert run_meta["model"] == "mock/test-model"
    assert run_meta["tasks"] == ["custom_qa"]
    assert run_meta["limit"] == 5
    assert run_meta["mock_backend"] is True
    assert run_meta["method"] == "pipeline_smoke"
    # stderr should also carry the run summary for log aggregation (rule 28).
    assert "eval_run" in capsys.readouterr().err

    # New: the result-log must have grown by one row with the method label.
    from common.result_log import read_results

    rows = read_results()
    assert len(rows) == 1
    assert rows[0]["method"] == "pipeline_smoke"
    assert rows[0]["task"] == "custom_qa"
    assert float(rows[0]["value"]) == pytest.approx(0.42)


def test_main_rejects_empty_task_list(_mock_env: Path) -> None:
    from common.errors import EvalError

    with pytest.raises(EvalError, match="at least one task"):
        main(["--task", ","])
