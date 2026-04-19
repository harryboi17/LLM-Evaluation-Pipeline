"""Tests for ``perf.load_test``."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from common.config import get_settings
from perf.load_test import RequestMetric, _build_specs, main, run_load, write_csv


@pytest.fixture
def _mock_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    monkeypatch.setenv("LLMEVAL_MOCK_BACKEND", "true")
    monkeypatch.setenv("LLMEVAL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LLMEVAL_RESULTS_DIR", str(tmp_path / "results"))
    get_settings.cache_clear()
    return tmp_path


def test_build_specs_mixes_short_and_long_deterministically() -> None:
    import random

    rng = random.Random(42)
    specs = _build_specs(
        num_requests=50,
        short_frac=0.5,
        short_max_tokens=32,
        long_max_tokens=256,
        rng=rng,
    )
    assert len(specs) == 50
    kinds = {s.prompt_kind for s in specs}
    assert kinds == {"short", "long"}
    # With the same seed, second run yields the same split.
    rng2 = random.Random(42)
    specs2 = _build_specs(50, 0.5, 32, 256, rng2)
    assert [s.prompt_kind for s in specs2] == [s.prompt_kind for s in specs]


async def test_run_load_against_mock_backend_produces_ok_rows(_mock_env: Path) -> None:
    rows = await run_load(
        num_requests=8,
        concurrency=4,
        short_frac=0.5,
        short_max_tokens=8,
        long_max_tokens=16,
        mode="generate",
        seed=7,
    )
    assert len(rows) == 8
    assert all(isinstance(r, RequestMetric) for r in rows)
    assert all(r.status == "ok" for r in rows)
    assert all(r.wall_s >= 0 for r in rows)


async def test_run_load_stream_mode_records_ttft(_mock_env: Path) -> None:
    rows = await run_load(
        num_requests=4,
        concurrency=2,
        short_frac=0.5,
        short_max_tokens=8,
        long_max_tokens=16,
        mode="stream",
        seed=7,
    )
    assert len(rows) == 4
    assert all(r.mode == "stream" for r in rows)
    # Mock stream emits non-empty deltas, so TTFT should be set.
    assert all(r.ttft_s is not None for r in rows)


def test_write_csv_round_trip(tmp_path: Path) -> None:
    rows = [
        RequestMetric(
            idx=0,
            prompt_kind="short",
            prompt_chars=10,
            max_tokens=16,
            mode="generate",
            wall_s=0.5,
            ttft_s=None,
            output_tokens=7,
            status="ok",
            started_at=100.0,
            ended_at=100.5,
        ),
        RequestMetric(
            idx=1,
            prompt_kind="long",
            prompt_chars=200,
            max_tokens=128,
            mode="stream",
            wall_s=2.0,
            ttft_s=0.4,
            output_tokens=80,
            status="ok",
            started_at=100.1,
            ended_at=102.1,
        ),
    ]
    path = tmp_path / "metrics.csv"
    write_csv(rows, path)

    with path.open() as f:
        reader = csv.DictReader(f)
        readback = list(reader)
    assert len(readback) == 2
    assert readback[0]["prompt_kind"] == "short"
    assert readback[0]["ttft_s"] == ""  # stored as empty for None
    assert readback[1]["ttft_s"] == "0.4"


def test_main_writes_csv_and_prints_json(
    _mock_env: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    out = _mock_env / "metrics.csv"
    rc = main(
        [
            "--num-requests",
            "6",
            "--concurrency",
            "3",
            "--short-frac",
            "1.0",
            "--short-max-tokens",
            "4",
            "--output",
            str(out),
        ]
    )
    assert rc == 0
    assert out.exists()
    summary_line = capsys.readouterr().out.strip().splitlines()[-1]
    import json

    summary = json.loads(summary_line)
    assert summary["ok"] == 6
    assert summary["errors"] == 0
    assert summary["output_csv"] == str(out)
