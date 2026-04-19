"""Tests for ``improve.sweep`` -- decoding-params sweep orchestrator."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from common.config import get_settings
from common.errors import EvalError
from improve.sweep import SweepCell, build_grid, main


def test_build_grid_cartesian_product() -> None:
    cells = build_grid([0.0, 0.5], [0.9, 1.0], [0, 50])
    assert len(cells) == 2 * 2 * 2
    # All eight combinations must be unique.
    assert len({(c.temperature, c.top_p, c.top_k) for c in cells}) == 8


def test_sweep_cell_method_label_is_filesystem_safe() -> None:
    cell = SweepCell(temperature=0.7, top_p=0.95, top_k=40)
    label = cell.method_label
    assert "sweep_" in label
    # Must be safe for CSV (no commas) and for shell (no spaces).
    assert "," not in label
    assert " " not in label


def test_main_rejects_loglikelihood_only_tasks_without_force() -> None:
    with pytest.raises(EvalError, match="log-likelihood"):
        main(["--task", "hellaswag", "--dry-run"])


def test_main_rejects_mmlu_subject_groups_without_force() -> None:
    with pytest.raises(EvalError, match="log-likelihood"):
        main(["--task", "mmlu_stem", "--dry-run"])


def test_main_dry_run_on_generative_task_prints_grid(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import json

    rc = main(
        [
            "--task",
            "custom_qa",
            "--temperature",
            "0.0,0.5",
            "--top-p",
            "1.0",
            "--top-k",
            "0",
            "--dry-run",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    parsed = json.loads(captured.out)
    assert len(parsed) == 2
    assert {c["temperature"] for c in parsed} == {0.0, 0.5}


def test_main_invokes_run_eval_subprocess_per_cell(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Patch subprocess.run so we can count cells without touching lm-eval."""
    monkeypatch.setenv("LLMEVAL_RESULTS_DIR", str(tmp_path))
    get_settings.cache_clear()

    calls: list[list[str]] = []

    def fake_run(cmd: list[str], check: bool = False, env: dict[str, str] | None = None) -> Any:
        _ = (check, env)
        calls.append(list(cmd))
        return MagicMock(returncode=0)

    with patch("improve.sweep.subprocess.run", side_effect=fake_run):
        rc = main(
            [
                "--task",
                "custom_qa",
                "--temperature",
                "0.0,0.7",
                "--top-p",
                "1.0",
                "--top-k",
                "0",
                "--limit",
                "5",
            ]
        )

    assert rc == 0
    assert len(calls) == 2  # one process per (t, p, k) cell
    # Every call must pipe through run_eval with a --method label.
    for argv in calls:
        assert "-m" in argv
        i = argv.index("-m")
        assert argv[i + 1] == "eval_runner.run_eval"
        assert "--method" in argv


def test_main_forwards_nonzero_exit_when_any_cell_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.setenv("LLMEVAL_RESULTS_DIR", str(tmp_path))
    get_settings.cache_clear()
    counter = {"n": 0}

    def fake_run(cmd: list[str], check: bool = False, env: dict[str, str] | None = None) -> Any:
        _ = (cmd, check, env)
        counter["n"] += 1
        return MagicMock(returncode=1 if counter["n"] == 2 else 0)

    with patch("improve.sweep.subprocess.run", side_effect=fake_run):
        rc = main(
            [
                "--task",
                "custom_qa",
                "--temperature",
                "0.0,0.2,0.4",
                "--top-p",
                "1.0",
                "--top-k",
                "0",
            ]
        )

    assert rc == 1  # at least one cell failed


def test_force_flag_allows_loglikelihood_task(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    monkeypatch.setenv("LLMEVAL_RESULTS_DIR", str(tmp_path))
    get_settings.cache_clear()

    # Dry-run still returns 0 even on a loglikelihood task when --force is set.
    rc = main(
        [
            "--task",
            "hellaswag",
            "--force",
            "--temperature",
            "0.0",
            "--top-p",
            "1.0",
            "--top-k",
            "0",
            "--dry-run",
        ]
    )
    assert rc == 0
