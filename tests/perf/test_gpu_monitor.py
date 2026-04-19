"""Tests for ``perf.gpu_monitor``."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from perf.gpu_monitor import _parse_nvidia_smi, main, record, sample_once


def test_parse_nvidia_smi_handles_multi_gpu() -> None:
    raw = (
        "0, 85, 20, 7800, 16000, 72, 220.4\n"
        "1, 10, 5, 2000, 16000, 55, 90.1\n"
    )
    rows = _parse_nvidia_smi(raw)
    assert len(rows) == 2
    assert rows[0][0] == "0"
    assert rows[1][0] == "1"


def test_parse_nvidia_smi_handles_blank_lines() -> None:
    raw = "\n0, 85, 20, 7800, 16000, 72, 220.4\n\n"
    assert _parse_nvidia_smi(raw) == [
        ["0", "85", "20", "7800", "16000", "72", "220.4"],
    ]


def test_sample_once_raises_without_nvidia_smi() -> None:
    with (
        patch("perf.gpu_monitor.shutil.which", return_value=None),
        pytest.raises(FileNotFoundError),
    ):
        sample_once()


def test_sample_once_surfaces_nonzero_exit(tmp_path: Path) -> None:
    import subprocess

    fake = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="boom")
    with (
        patch("perf.gpu_monitor.shutil.which", return_value="/usr/bin/nvidia-smi"),
        patch("perf.gpu_monitor.subprocess.run", return_value=fake),
        pytest.raises(RuntimeError, match="boom"),
    ):
        sample_once()


def test_record_writes_csv_with_header_and_rows(tmp_path: Path) -> None:
    path = tmp_path / "gpu.csv"

    rows_per_call = [["0", "80", "15", "7000", "16000", "70", "200.0"]]

    # record() loops until the deadline; we patch sample_once to return one row
    # per GPU per call, and we cap total calls by using a tiny duration.
    with (
        patch("perf.gpu_monitor.sample_once", return_value=rows_per_call),
        patch("perf.gpu_monitor.time.sleep", return_value=None),
    ):
        n = record(path, interval_s=0.01, duration_s=0.05)

    assert path.exists()
    content = path.read_text().strip().splitlines()
    assert content[0].startswith("timestamp,")
    assert n >= 1


def test_main_without_nvidia_smi_returns_2(capsys: pytest.CaptureFixture[str]) -> None:
    with patch("perf.gpu_monitor.shutil.which", return_value=None):
        rc = main([])
    assert rc == 2
    assert "not found" in capsys.readouterr().err
