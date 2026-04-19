"""Tests for ``perf.metrics``."""

from __future__ import annotations

import pandas as pd
import pytest

from perf.metrics import Percentiles, compute_tpot, percentiles, summarize


def test_percentiles_basic() -> None:
    p = percentiles([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
    assert p.p50 == pytest.approx(5.5)
    assert p.p95 == pytest.approx(9.55, rel=0.1)
    assert p.p99 == pytest.approx(9.91, rel=0.1)


def test_percentiles_handles_empty() -> None:
    p = percentiles([])
    assert p == Percentiles(0.0, 0.0, 0.0)


def test_percentiles_ignores_nan() -> None:
    p = percentiles([1.0, float("nan"), 2.0, 3.0])
    assert p.p50 == pytest.approx(2.0)


def test_compute_tpot_stream_subtracts_ttft() -> None:
    # wall=2.0, ttft=0.4, tokens=80 -> 80 / 1.6 = 50 t/s
    assert compute_tpot(2.0, 0.4, 80) == pytest.approx(50.0)


def test_compute_tpot_generate_uses_full_wall() -> None:
    # No ttft -> 80 / 2.0 = 40 t/s
    assert compute_tpot(2.0, None, 80) == pytest.approx(40.0)


def test_compute_tpot_degenerate_inputs_return_zero() -> None:
    assert compute_tpot(0.0, None, 10) == 0.0
    assert compute_tpot(1.0, None, 0) == 0.0


def test_summarize_groups_by_prompt_kind_and_mode() -> None:
    rows = [
        {
            "prompt_kind": "short",
            "mode": "generate",
            "wall_s": 0.5,
            "ttft_s": None,
            "output_tokens": 8,
            "status": "ok",
            "started_at": 100.0,
            "ended_at": 100.5,
        },
        {
            "prompt_kind": "short",
            "mode": "generate",
            "wall_s": 0.7,
            "ttft_s": None,
            "output_tokens": 8,
            "status": "ok",
            "started_at": 100.1,
            "ended_at": 100.8,
        },
        {
            "prompt_kind": "long",
            "mode": "stream",
            "wall_s": 3.0,
            "ttft_s": 0.4,
            "output_tokens": 200,
            "status": "ok",
            "started_at": 100.0,
            "ended_at": 103.0,
        },
        {
            "prompt_kind": "long",
            "mode": "stream",
            "wall_s": 3.2,
            "ttft_s": 0.5,
            "output_tokens": 210,
            "status": "ok",
            "started_at": 100.2,
            "ended_at": 103.4,
        },
    ]
    df = summarize(rows)
    assert set(df["prompt_kind"]) == {"short", "long"}
    short_row = df[(df["prompt_kind"] == "short") & (df["mode"] == "generate")].iloc[0]
    assert short_row["n"] == 2
    assert short_row["ok"] == 2
    assert short_row["error_rate"] == 0.0
    assert short_row["wall_s_p50"] == pytest.approx(0.6, abs=0.01)

    long_row = df[(df["prompt_kind"] == "long") & (df["mode"] == "stream")].iloc[0]
    assert long_row["ttft_s_p50"] == pytest.approx(0.45, abs=0.01)
    assert long_row["tpot_mean"] > 0


def test_summarize_empty_returns_empty_df() -> None:
    df = summarize([])
    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_summarize_error_rows_counted_but_not_in_latency() -> None:
    rows = [
        {
            "prompt_kind": "short",
            "mode": "generate",
            "wall_s": 0.5,
            "ttft_s": None,
            "output_tokens": 8,
            "status": "ok",
            "started_at": 100.0,
            "ended_at": 100.5,
        },
        {
            "prompt_kind": "short",
            "mode": "generate",
            "wall_s": 0.1,
            "ttft_s": None,
            "output_tokens": 0,
            "status": "error",
            "started_at": 100.0,
            "ended_at": 100.1,
        },
    ]
    df = summarize(rows)
    row = df.iloc[0]
    assert row["n"] == 2
    assert row["ok"] == 1
    assert row["errors"] == 1
    assert row["error_rate"] == pytest.approx(0.5)
    # Latency p50 counts only ok rows.
    assert row["wall_s_p50"] == pytest.approx(0.5)
