"""Post-processing for ``metrics.csv`` → per-configuration aggregates.

Reads rows emitted by :mod:`perf.load_test`, groups by configuration dimensions
(``prompt_kind`` by ``mode``), and computes:

* **TTFT** (time to first token) — stream mode only
* **TPOT** (tokens per output second) — ``output_tokens / (wall_s - ttft_s)``
  where ``ttft_s`` is known, else ``output_tokens / wall_s``
* **P50 / P95 / P99** end-to-end latency
* **Error rate** and **throughput** (requests per wall second)

Outputs a compact summary dataframe. Percentiles are computed with numpy so
pandas is only needed for the I/O layer.

Usage::

    python -m perf.metrics --input metrics.csv --summary summary.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from common.logging import get_logger

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class Percentiles:
    """Basic latency percentiles."""

    p50: float
    p95: float
    p99: float

    def as_dict(self) -> dict[str, float]:
        return {"p50": self.p50, "p95": self.p95, "p99": self.p99}


def percentiles(values: Sequence[float]) -> Percentiles:
    """Return P50/P95/P99 of ``values``; empty input returns zeros.

    Uses linear interpolation (numpy's default) which is standard for latency
    reporting.
    """
    arr = np.asarray([v for v in values if v is not None and not np.isnan(v)])
    if arr.size == 0:
        return Percentiles(0.0, 0.0, 0.0)
    return Percentiles(
        p50=float(np.percentile(arr, 50)),
        p95=float(np.percentile(arr, 95)),
        p99=float(np.percentile(arr, 99)),
    )


def compute_tpot(wall_s: float, ttft_s: float | None, output_tokens: int) -> float:
    """Return tokens per output second.

    ``wall_s - ttft_s`` isolates the decode phase when TTFT is known (stream
    mode). Otherwise we fall back to ``output_tokens / wall_s`` which over-counts
    prefill time as decode time but is still a useful comparator.
    """
    if output_tokens <= 0 or wall_s <= 0:
        return 0.0
    if ttft_s is not None and ttft_s > 0 and ttft_s < wall_s:
        return output_tokens / (wall_s - ttft_s)
    return output_tokens / wall_s


def summarize(rows: Iterable[dict[str, object]]) -> pd.DataFrame:
    """Aggregate per-request metrics into a summary ``DataFrame``.

    Each output row covers one ``(prompt_kind, mode)`` slice and contains:

    * ``n``, ``ok``, ``errors``, ``error_rate``
    * ``wall_s_p50``, ``wall_s_p95``, ``wall_s_p99``
    * ``ttft_s_p50``, ``ttft_s_p95``, ``ttft_s_p99`` (NaN outside stream mode)
    * ``tpot_mean``, ``tpot_p50``
    * ``throughput_rps`` — computed over the whole run window, not per-row
    """
    df = pd.DataFrame(list(rows))
    if df.empty:
        return pd.DataFrame()

    # Coerce optionals.
    df["ttft_s"] = pd.to_numeric(df.get("ttft_s"), errors="coerce")
    df["wall_s"] = pd.to_numeric(df["wall_s"], errors="coerce")
    df["output_tokens"] = pd.to_numeric(df["output_tokens"], errors="coerce").fillna(0).astype(int)

    # Derived per-row metric.
    df["tpot"] = df.apply(
        lambda r: compute_tpot(float(r["wall_s"]), r["ttft_s"], int(r["output_tokens"])),
        axis=1,
    )

    out_rows: list[dict[str, object]] = []
    for (prompt_kind, mode), g in df.groupby(["prompt_kind", "mode"], sort=True):
        ok_mask = g["status"] == "ok"
        ok = g[ok_mask]
        wall_p = percentiles(ok["wall_s"].tolist())
        ttft_p = percentiles(ok["ttft_s"].dropna().tolist())
        tpot_values = ok["tpot"].replace([np.inf, -np.inf], np.nan).dropna().tolist()

        # Throughput over the run window: request count / (latest end - earliest start)
        started = pd.to_numeric(ok.get("started_at"), errors="coerce").dropna()
        ended = pd.to_numeric(ok.get("ended_at"), errors="coerce").dropna()
        if len(started) > 0 and len(ended) > 0:
            window_s = float(ended.max() - started.min())
            throughput_rps = (len(ok) / window_s) if window_s > 0 else float("nan")
        else:
            throughput_rps = float("nan")

        out_rows.append(
            {
                "prompt_kind": prompt_kind,
                "mode": mode,
                "n": len(g),
                "ok": int(ok_mask.sum()),
                "errors": int((~ok_mask).sum()),
                "error_rate": float((~ok_mask).sum() / len(g)) if len(g) else 0.0,
                "wall_s_p50": wall_p.p50,
                "wall_s_p95": wall_p.p95,
                "wall_s_p99": wall_p.p99,
                "ttft_s_p50": ttft_p.p50 if len(ttft_p.as_dict()) else float("nan"),
                "ttft_s_p95": ttft_p.p95,
                "ttft_s_p99": ttft_p.p99,
                "tpot_mean": float(np.mean(tpot_values)) if tpot_values else 0.0,
                "tpot_p50": float(np.percentile(tpot_values, 50)) if tpot_values else 0.0,
                "throughput_rps": throughput_rps,
            }
        )
    return pd.DataFrame(out_rows)


def load_metrics(path: Path) -> pd.DataFrame:
    """Read ``metrics.csv`` into a DataFrame with proper dtypes."""
    df = pd.read_csv(path)
    return df


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI argv."""
    p = argparse.ArgumentParser(description="Summarise perf metrics.csv.")
    p.add_argument("--input", default="metrics.csv", help="Path to raw metrics.csv.")
    p.add_argument(
        "--summary",
        default=None,
        help="Optional summary CSV output. If omitted, prints to stdout.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m perf.metrics``."""
    args = _parse_args(argv)
    df = load_metrics(Path(args.input))
    summary = summarize(df.to_dict(orient="records"))
    if args.summary:
        summary.to_csv(args.summary, index=False)
        log.info("perf_summary_written", path=args.summary, rows=len(summary))
    else:
        sys.stdout.write(summary.to_csv(index=False))
    sys.stdout.write(
        json.dumps(
            {
                "input_rows": len(df),
                "summary_rows": len(summary),
            }
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "Percentiles",
    "compute_tpot",
    "load_metrics",
    "main",
    "percentiles",
    "summarize",
]
