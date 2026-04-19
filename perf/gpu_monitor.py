"""Optional GPU utilisation sampler built on ``nvidia-smi``.

Spawns ``nvidia-smi --query-gpu=...`` in CSV mode, samples at a fixed interval,
and writes one row per sample to a CSV so the series can be joined with
:mod:`perf.load_test` output by timestamp.

Usage::

    # In one terminal (background sampler):
    python -m perf.gpu_monitor --output gpu.csv --interval 0.5 --duration 60

    # In another (load gen):
    python -m perf.load_test --output metrics.csv ...

Both CSVs embed wall-clock ``timestamp`` so analyses can merge on timestamp bins.
Degrades gracefully when ``nvidia-smi`` is not on ``PATH``: exits with a
non-zero code and an informative message rather than raising.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

from common.logging import get_logger

log = get_logger(__name__)

_NVIDIA_SMI = "nvidia-smi"
_QUERY_FIELDS: tuple[str, ...] = (
    "index",
    "utilization.gpu",
    "utilization.memory",
    "memory.used",
    "memory.total",
    "temperature.gpu",
    "power.draw",
)
_CSV_COLUMNS: tuple[str, ...] = ("timestamp", *_QUERY_FIELDS)


def _build_query_cmd() -> list[str]:
    """Build the ``nvidia-smi`` argv emitting a single CSV line per GPU."""
    return [
        _NVIDIA_SMI,
        f"--query-gpu={','.join(_QUERY_FIELDS)}",
        "--format=csv,noheader,nounits",
    ]


def _parse_nvidia_smi(raw: str) -> list[list[str]]:
    """Parse the raw ``nvidia-smi`` output (CSV, one row per GPU)."""
    return [line.strip().split(", ") for line in raw.strip().splitlines() if line.strip()]


def sample_once() -> list[list[str]]:
    """Execute ``nvidia-smi`` once and return a list of per-GPU field lists.

    Raises :class:`FileNotFoundError` if ``nvidia-smi`` is not installed, or
    :class:`RuntimeError` if it exits non-zero.
    """
    if shutil.which(_NVIDIA_SMI) is None:
        raise FileNotFoundError(f"{_NVIDIA_SMI} not on PATH")
    result = subprocess.run(
        _build_query_cmd(),
        check=False,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
    return _parse_nvidia_smi(result.stdout)


def record(path: Path, interval_s: float, duration_s: float) -> int:
    """Sample ``nvidia-smi`` every ``interval_s`` for ``duration_s`` seconds.

    Writes one CSV row per GPU per sample. Returns the number of samples taken
    (``1`` row per GPU per sample).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    n_samples = 0
    deadline = time.monotonic() + duration_s
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(_CSV_COLUMNS)
        while time.monotonic() < deadline:
            ts = time.time()
            try:
                rows = sample_once()
            except (FileNotFoundError, RuntimeError) as exc:
                log.warning("gpu_sample_failed", reason=str(exc))
                break
            for r in rows:
                writer.writerow([ts, *r])
                n_samples += 1
            time.sleep(interval_s)
    log.info("gpu_monitor_done", samples=n_samples, path=str(path))
    return n_samples


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI argv."""
    p = argparse.ArgumentParser(description="Periodic nvidia-smi sampler.")
    p.add_argument("--output", default="gpu.csv")
    p.add_argument("--interval", type=float, default=0.5, help="Seconds between samples.")
    p.add_argument("--duration", type=float, default=60.0, help="Seconds to sample for.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m perf.gpu_monitor``."""
    if shutil.which(_NVIDIA_SMI) is None:
        sys.stderr.write(
            f"{_NVIDIA_SMI} not found on PATH — GPU monitoring unavailable on this host. "
            "Perf runs will still produce latency/TTFT/TPOT without GPU metrics.\n"
        )
        return 2
    args = _parse_args(argv)
    record(Path(args.output), args.interval, args.duration)
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main", "record", "sample_once"]
