"""Decoding-params sweep for generative tasks.

Part E's target is HellaSwag (loglikelihood), where temperature / top_p /
top_k don't change anything — the model's raw logits are scored directly.
But the assignment explicitly lists decoding-knob tuning under Part E's
"Allowed levers" for cases where the target benchmark is **generative**
(GSM8K, the ``custom_qa`` task, self-consistency-style majority voting, …).

This module is the thin orchestrator for that case. It takes a grid of
``(temperature, top_p, top_k)`` combinations and, for each cell, invokes
``eval_runner.run_eval`` in a subprocess with a ``--method`` label that
encodes the cell. Every run lands in ``results/result-log.csv`` the usual
way, so afterwards the results can be sorted / plotted through
``results/analysis.ipynb``.

Why subprocess-per-cell instead of an in-process loop? Two reasons:

1. ``lm_eval.simple_evaluate`` holds non-trivial state (task cache, seed
   restoration, global numpy/torch RNG hooks). One fresh process per cell
   is the safest way to prevent cross-cell leakage.
2. If a single cell OOMs or hangs, it doesn't take the whole sweep down.

Usage::

    # Default grid (9 cells): t in {0.0, 0.3, 0.7} x top_p in {0.9, 0.95, 1.0}
    uv run python -m improve.sweep --task custom_qa --limit 30

    # Custom grid via CLI:
    uv run python -m improve.sweep --task custom_qa \\
        --temperature 0.0,0.2,0.5,0.9 --top-p 0.95 --top-k 1,20,50

    # Against a running vLLM server (real numbers):
    LLMEVAL_MOCK_BACKEND=false uv run python -m improve.sweep \\
        --task gsm8k --limit 200

The sweep does **not** apply to loglikelihood tasks. If you point it at
HellaSwag / MMLU / ARC you'll get identical numbers across cells (as
expected), and a warning to that effect.
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from common.config import get_settings
from common.errors import EvalError
from common.logging import get_logger

log = get_logger(__name__)

# Tasks we know are purely loglikelihood; decoding knobs are no-ops here.
_LOGLIKELIHOOD_ONLY: frozenset[str] = frozenset(
    {
        "hellaswag",
        "arc_challenge",
        "arc_easy",
        "winogrande",
        "piqa",
    }
) | frozenset({f"mmlu_{suffix}" for suffix in ("stem", "humanities", "social_sciences", "other")})


@dataclass(frozen=True, slots=True)
class SweepCell:
    """One point in the decoding-params grid."""

    temperature: float
    top_p: float
    top_k: int

    @property
    def method_label(self) -> str:
        """Human-friendly label for the result-log ``method`` column."""
        return f"sweep_t{self.temperature:g}_p{self.top_p:g}_k{self.top_k}"


def _parse_floats(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def build_grid(
    temperatures: list[float],
    top_ps: list[float],
    top_ks: list[int],
) -> list[SweepCell]:
    """Cartesian product of the three axes."""
    return [
        SweepCell(temperature=t, top_p=p, top_k=k)
        for t, p, k in itertools.product(temperatures, top_ps, top_ks)
    ]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Decoding-params sweep over a generative task.")
    p.add_argument(
        "--task",
        required=True,
        help="Task name (e.g. custom_qa, gsm8k). Loglikelihood-only tasks are rejected with a warning.",
    )
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--num-fewshot", type=int, default=None, dest="num_fewshot")
    p.add_argument(
        "--temperature",
        default="0.0,0.3,0.7",
        help="Comma-separated temperatures to sweep.",
    )
    p.add_argument(
        "--top-p",
        default="0.9,0.95,1.0",
        dest="top_p",
        help="Comma-separated nucleus-sampling thresholds.",
    )
    p.add_argument(
        "--top-k",
        default="0",
        dest="top_k",
        help="Comma-separated top-k values (0 disables).",
    )
    p.add_argument(
        "--max-concurrency",
        type=int,
        default=8,
        dest="max_concurrency",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Run even if the task is known to be loglikelihood-only.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the grid and exit without firing subprocesses.",
    )
    return p.parse_args(argv)


def _is_loglikelihood_only(task: str) -> bool:
    return task in _LOGLIKELIHOOD_ONLY or task.startswith("mmlu_")


def _invoke_run_eval(cell: SweepCell, args: argparse.Namespace) -> int:
    """Invoke ``run_eval.py`` as a subprocess with the cell's decoding params.

    The sweep pipes its knobs in via environment variables. ``run_eval.py``
    doesn't currently expose ``--temperature`` etc. directly because for
    loglikelihood tasks (its primary use) they're irrelevant; generative
    tasks' decoding kwargs come from the task YAML. For a sweep, we override
    the task YAML's defaults by injecting ``LLMEVAL_GEN_*`` env vars that
    ``eval_runner.vllm_model`` respects.
    """
    cmd = [
        "uv",
        "run",
        "python",
        "-m",
        "eval_runner.run_eval",
        "--task",
        args.task,
        "--method",
        cell.method_label,
        "--max-concurrency",
        str(args.max_concurrency),
        "--notes",
        f"sweep cell t={cell.temperature} p={cell.top_p} k={cell.top_k}",
    ]
    if args.limit is not None:
        cmd.extend(["--limit", str(args.limit)])
    if args.num_fewshot is not None:
        cmd.extend(["--num-fewshot", str(args.num_fewshot)])

    env: dict[str, str] = {
        "LLMEVAL_GEN_TEMPERATURE": str(cell.temperature),
        "LLMEVAL_GEN_TOP_P": str(cell.top_p),
        "LLMEVAL_GEN_TOP_K": str(cell.top_k),
    }
    log.info(
        "sweep_cell_start",
        method=cell.method_label,
        temperature=cell.temperature,
        top_p=cell.top_p,
        top_k=cell.top_k,
    )
    result = subprocess.run(cmd, check=False, env={**_os_env(), **env})
    return result.returncode


def _os_env() -> dict[str, str]:
    """Snapshot of the current environment, decoupled for test patching."""
    return dict(os.environ)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = get_settings()
    _ = settings  # ensures Settings load; env-vars propagate to subprocesses

    if _is_loglikelihood_only(args.task) and not args.force:
        raise EvalError(
            f"task '{args.task}' is scored by log-likelihood; decoding "
            "knobs have no effect on its accuracy. Pass --force if you "
            "really want to run the sweep anyway (e.g. for wall-time "
            "comparison)."
        )

    temperatures = _parse_floats(args.temperature)
    top_ps = _parse_floats(args.top_p)
    top_ks = _parse_ints(args.top_k)
    grid = build_grid(temperatures, top_ps, top_ks)

    log.info(
        "sweep_start",
        task=args.task,
        cells=len(grid),
        temperatures=temperatures,
        top_ps=top_ps,
        top_ks=top_ks,
    )

    if args.dry_run:
        sys.stdout.write(
            json.dumps([dataclasses.asdict(cell) for cell in grid], indent=2) + "\n"
        )
        return 0

    started = time.monotonic()
    ok, fail = 0, 0
    for cell in grid:
        rc = _invoke_run_eval(cell, args)
        if rc == 0:
            ok += 1
        else:
            fail += 1
            log.warning("sweep_cell_failed", method=cell.method_label, rc=rc)
    wall = time.monotonic() - started

    summary = {
        "task": args.task,
        "cells": len(grid),
        "ok": ok,
        "failed": fail,
        "wall_s": round(wall, 2),
    }
    sys.stdout.write(json.dumps(summary) + "\n")
    log.info("sweep_complete", **summary)
    return 0 if fail == 0 else 1


def _sweep_results_path() -> Path:
    """Sweep results all land in the central result-log; no separate file."""
    return get_settings().results_dir / "result-log.csv"


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["SweepCell", "build_grid", "main"]
