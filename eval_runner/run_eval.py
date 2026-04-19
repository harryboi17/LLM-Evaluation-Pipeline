"""CLI entry point for running the lm-evaluation-harness against our vLLM model.

Usage::

    # Run all three tasks (defaults to the `mmlu_high_school_computer_science`
    # subset + `hellaswag` + our custom JSON task), limited to 50 examples each:
    python -m eval_runner.run_eval --limit 50

    # One specific task:
    python -m eval_runner.run_eval --task custom --limit 20

    # Against the mock backend (no GPU / no HF auth required):
    LLMEVAL_MOCK_BACKEND=true python -m eval_runner.run_eval --task custom

Outputs:

* ``results/<task>.json`` — per-task raw metrics from lm-eval.
* ``results/summary.md`` — a compact Markdown summary table across all tasks
  in this run.
* ``results/run_meta.json`` — run-level metadata (model, seed, limits, duration).

Every output path is sourced from :attr:`common.config.Settings.results_dir` —
no hardcoded paths. Caching lives under :attr:`Settings.cache_dir` via
:class:`common.cache.PromptCache`, so repeat runs skip already-scored examples.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import yaml

from common.config import get_settings
from common.errors import EvalError
from common.logging import get_logger
from common.result_log import ResultLogEntry, log_result

log = get_logger(__name__)

# Default task list mirrors PLAN.md. MMLU's full suite is ~15k examples across
# 57 subjects; the harness also exposes per-subject slugs which are much cheaper
# to run. We default to one representative subject; override with --task to run
# the whole MMLU aggregate.
_DEFAULT_TASKS: tuple[str, ...] = (
    "mmlu_high_school_computer_science",
    "hellaswag",
    "custom_qa",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI argv."""
    p = argparse.ArgumentParser(
        description="Run lm-evaluation-harness against the vLLM-backed model.",
    )
    p.add_argument(
        "--task",
        default=",".join(_DEFAULT_TASKS),
        help="Comma-separated list of task names.",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap examples per task (default: run full dataset).",
    )
    p.add_argument(
        "--num-fewshot",
        type=int,
        default=None,
        dest="num_fewshot",
        help="Override few-shot count for all tasks.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        dest="output_dir",
        help="Override Settings.results_dir.",
    )
    p.add_argument(
        "--max-concurrency",
        type=int,
        default=8,
        dest="max_concurrency",
        help="Max concurrent in-flight vLLM requests.",
    )
    p.add_argument(
        "--method",
        default="baseline",
        help="Free-text label for this run in result-log.csv (e.g. 'baseline', 'clean_prompt').",
    )
    p.add_argument(
        "--notes",
        default="",
        help="Free-text notes attached to every result-log row in this run.",
    )
    return p.parse_args(argv)


def _materialise_custom_task(src_dir: Path) -> Path:
    """Rewrite ``custom_task.yaml`` with an absolute ``data_files`` path.

    ``datasets.load_dataset('json', data_files=...)`` resolves paths relative to
    the current working directory, not the YAML file's directory. Committing an
    absolute path in the repo isn't portable, so we materialise a resolved copy
    of the YAML into a temp directory and point :class:`TaskManager` at that.

    Args:
        src_dir: Directory containing ``custom_task.yaml`` and
            ``custom_data.jsonl`` (typically ``eval_runner/tasks/``).

    Returns:
        Path to a temp directory containing a resolved ``custom_task.yaml``
        whose ``data_files`` points at the original JSONL by absolute path.
    """
    src_yaml = src_dir / "custom_task.yaml"
    src_data = (src_dir / "custom_data.jsonl").resolve()
    if not src_yaml.exists():
        raise EvalError(f"missing task yaml: {src_yaml}")
    if not src_data.exists():
        raise EvalError(f"missing task data: {src_data}")

    config: dict[str, Any] = yaml.safe_load(src_yaml.read_text(encoding="utf-8"))
    config.setdefault("dataset_kwargs", {})
    config["dataset_kwargs"]["data_files"] = {"test": str(src_data)}

    tmp_dir = Path(tempfile.mkdtemp(prefix="llmeval_tasks_"))
    (tmp_dir / "custom_task.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return tmp_dir


def _run_harness(
    tasks: list[str],
    limit: int | None,
    num_fewshot: int | None,
    max_concurrency: int,
) -> dict[str, Any]:
    """Invoke ``lm_eval.simple_evaluate`` with our model and custom tasks dir."""
    try:
        from lm_eval import simple_evaluate
        from lm_eval.tasks import TaskManager
    except ImportError as exc:
        raise EvalError(
            "lm-eval is not installed. Run `uv sync --extra eval`."
        ) from exc

    from eval_runner.vllm_model import get_vllm_eval_model_class

    settings = get_settings()
    model_cls = get_vllm_eval_model_class()
    model = model_cls(
        model=settings.model_name,
        max_concurrency=max_concurrency,
    )

    custom_task_src = Path(__file__).parent / "tasks"
    resolved_task_dir = _materialise_custom_task(custom_task_src)
    try:
        task_manager = TaskManager(include_path=str(resolved_task_dir))

        log.info(
            "harness_start",
            tasks=tasks,
            limit=limit,
            num_fewshot=num_fewshot,
            model=settings.model_name,
        )
        results: dict[str, Any] = simple_evaluate(
            model=model,
            tasks=tasks,
            num_fewshot=num_fewshot,
            limit=limit,
            task_manager=task_manager,
            random_seed=settings.seed,
            numpy_random_seed=settings.seed,
            torch_random_seed=settings.seed,
            fewshot_random_seed=settings.seed,
        )
    finally:
        shutil.rmtree(resolved_task_dir, ignore_errors=True)
    return results


def _write_outputs(
    results: dict[str, Any],
    tasks: list[str],
    output_dir: Path,
    run_meta: dict[str, Any],
) -> None:
    """Persist per-task JSON, run metadata, and the Markdown summary."""
    output_dir.mkdir(parents=True, exist_ok=True)

    per_task: dict[str, dict[str, Any]] = results.get("results", {})
    for task in tasks:
        if task not in per_task:
            log.warning("task_missing_from_results", task=task)
            continue
        (output_dir / f"{task}.json").write_text(
            json.dumps(per_task[task], indent=2, sort_keys=True, default=str)
        )

    (output_dir / "run_meta.json").write_text(
        json.dumps(run_meta, indent=2, sort_keys=True, default=str)
    )

    summary_lines = [
        "# Evaluation Summary",
        "",
        f"- **Model:** `{run_meta['model']}`",
        f"- **Seed:** {run_meta['seed']}",
        f"- **Limit per task:** {run_meta.get('limit', 'full')}",
        f"- **Duration:** {run_meta['duration_s']:.1f}s",
        "",
        "| Task | Metric | Value |",
        "| --- | --- | --- |",
    ]
    for task in tasks:
        task_metrics = per_task.get(task, {})
        for metric_name, value in task_metrics.items():
            if not isinstance(value, int | float):
                continue
            summary_lines.append(f"| `{task}` | `{metric_name}` | {value:.4f} |")
    (output_dir / "summary.md").write_text("\n".join(summary_lines) + "\n")
    log.info("eval_outputs_written", dir=str(output_dir))


_DEFAULT_METRIC_PREFERENCE: tuple[str, ...] = (
    "acc_norm,none",
    "acc,none",
    "exact_match,exact_match",
    "exact_match,none",
    "acc_norm",
    "acc",
    "exact_match",
)
_DEFAULT_STDERR_PREFERENCE: tuple[str, ...] = (
    "acc_norm_stderr,none",
    "acc_stderr,none",
    "exact_match_stderr,exact_match",
    "exact_match_stderr,none",
    "acc_norm_stderr",
    "acc_stderr",
    "exact_match_stderr",
)


def _pick_primary_metric(metrics: dict[str, Any]) -> tuple[str, float, float | None]:
    """Return ``(metric_name, value, stderr)`` for a task's metrics dict.

    Tries a fixed preference list (matches lm-eval's naming quirks) and falls
    back to the first numeric metric. Returns ``(metric, value, stderr or None)``.
    """
    for name in _DEFAULT_METRIC_PREFERENCE:
        if name in metrics and isinstance(metrics[name], int | float):
            stderr: float | None = None
            for sname in _DEFAULT_STDERR_PREFERENCE:
                if sname in metrics and isinstance(metrics[sname], int | float):
                    stderr = float(metrics[sname])
                    break
            return (name, float(metrics[name]), stderr)
    for name, v in metrics.items():
        if isinstance(v, int | float):
            return (name, float(v), None)
    raise EvalError(f"no numeric metric in {metrics!r}")


def _append_to_result_log(
    tasks: list[str],
    per_task: dict[str, dict[str, Any]],
    run_meta: dict[str, Any],
    method_label: str,
    notes: str,
) -> None:
    """Append one row per task to ``result-log.csv`` / ``result-log.md``."""
    for task in tasks:
        if task not in per_task:
            continue
        try:
            metric_name, value, stderr = _pick_primary_metric(per_task[task])
        except EvalError as exc:
            log.warning("result_log_skip", task=task, reason=str(exc))
            continue
        log_result(
            ResultLogEntry(
                model=str(run_meta["model"]),
                task=task,
                method=method_label,
                metric=metric_name,
                value=value,
                seed=int(run_meta["seed"]),
                limit=run_meta.get("limit"),
                num_fewshot=run_meta.get("num_fewshot"),
                stderr=stderr,
                wall_s=float(run_meta["duration_s"]),
                mock_backend=bool(run_meta["mock_backend"]),
                notes=notes,
            )
        )


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m eval_runner.run_eval``.

    Returns:
        ``0`` on success, non-zero if a requested task failed to produce results.
    """
    args = _parse_args(argv)
    settings = get_settings()

    tasks = [t.strip() for t in args.task.split(",") if t.strip()]
    if not tasks:
        raise EvalError("--task must specify at least one task name")

    output_dir = Path(args.output_dir) if args.output_dir else settings.results_dir

    started = time.monotonic()
    results = _run_harness(
        tasks=tasks,
        limit=args.limit,
        num_fewshot=args.num_fewshot,
        max_concurrency=args.max_concurrency,
    )
    duration_s = time.monotonic() - started

    run_meta: dict[str, Any] = {
        "model": settings.model_name,
        "seed": settings.seed,
        "tasks": tasks,
        "limit": args.limit,
        "num_fewshot": args.num_fewshot,
        "max_concurrency": args.max_concurrency,
        "mock_backend": settings.mock_backend,
        "duration_s": duration_s,
        "method": args.method,
        "notes": args.notes,
    }
    _write_outputs(results, tasks, output_dir, run_meta)
    _append_to_result_log(
        tasks,
        results.get("results", {}),
        run_meta,
        method_label=args.method,
        notes=args.notes,
    )

    sys.stderr.write(json.dumps({"eval_run": run_meta}, default=str) + "\n")
    log.info("eval_complete", **run_meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main"]
