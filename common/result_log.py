"""Central result-log utility.

Shared by :mod:`eval_runner.run_eval` and :mod:`improve.infer` so every
evaluation run lands in the same place and the before/after story is easy to
skim.

The log is stored as two files under ``Settings.results_dir``:

* ``result-log.csv`` — append-only, machine-parseable source of truth.
* ``result-log.md`` — rendered Markdown table, regenerated on every append.

Schema (CSV column order):

1. ``run_id``        — ordinal, auto-incremented.
2. ``timestamp``     — ISO-8601 UTC.
3. ``commit``        — short git SHA of HEAD at log time (empty if no repo).
4. ``model``         — model identifier.
5. ``task``          — task name.
6. ``method``        — free-text label ("baseline", "clean_prompt", "+fewshot5", ...).
7. ``limit``         — examples evaluated (empty = full dataset).
8. ``num_fewshot``   — few-shot count, empty if N/A.
9. ``seed``          — RNG seed.
10. ``metric``       — primary metric name ("acc", "exact_match", "acc_norm").
11. ``value``        — primary metric value (float).
12. ``stderr``       — harness-reported stderr (float, empty if not reported).
13. ``delta``        — value - baseline_value (float, empty if baseline).
14. ``ci95_low`` / ``ci95_high`` — paired-bootstrap CI (float, empty if not computed).
15. ``wall_s``       — run wall time.
16. ``mock_backend`` — ``true`` / ``false``.
17. ``notes``        — free text, comma-escaped.
"""

from __future__ import annotations

import csv
import datetime as dt
import os
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

from common.config import get_settings
from common.logging import get_logger

log = get_logger(__name__)

_CSV_COLUMNS: tuple[str, ...] = (
    "run_id",
    "timestamp",
    "commit",
    "model",
    "task",
    "method",
    "limit",
    "num_fewshot",
    "seed",
    "metric",
    "value",
    "stderr",
    "delta",
    "ci95_low",
    "ci95_high",
    "wall_s",
    "mock_backend",
    "notes",
)


@dataclass(slots=True)
class ResultLogEntry:
    """One result row. Floats may be ``None`` when not applicable."""

    model: str
    task: str
    method: str
    metric: str
    value: float
    seed: int
    limit: int | None = None
    num_fewshot: int | None = None
    stderr: float | None = None
    delta: float | None = None
    ci95_low: float | None = None
    ci95_high: float | None = None
    wall_s: float | None = None
    mock_backend: bool = False
    notes: str = ""
    run_id: int = field(default=0, init=False)
    timestamp: str = field(default="", init=False)
    commit: str = field(default="", init=False)


def _git_short_sha() -> str:
    """Return the short git SHA of HEAD, or empty string if unavailable."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path.cwd(),
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        )
        return out.decode("utf-8").strip()
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return ""


def _log_paths() -> tuple[Path, Path]:
    """Return ``(csv_path, md_path)`` under ``Settings.results_dir``."""
    results_dir = get_settings().results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    return (results_dir / "result-log.csv", results_dir / "result-log.md")


def _next_run_id(csv_path: Path) -> int:
    """Return one past the highest ``run_id`` in the CSV (1 if empty)."""
    if not csv_path.exists():
        return 1
    try:
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            ids = [int(row.get("run_id", "0") or 0) for row in reader]
        return (max(ids) + 1) if ids else 1
    except (OSError, ValueError):
        return 1


def _format_float(value: float | None, digits: int = 4) -> str:
    """Format a float for CSV / MD; empty when ``None`` / NaN."""
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def log_result(entry: ResultLogEntry) -> ResultLogEntry:
    """Append ``entry`` to ``result-log.csv`` and regenerate ``result-log.md``.

    Mutates ``entry`` in place to fill in ``run_id``, ``timestamp``, ``commit``.

    Args:
        entry: The populated entry (everything except ``run_id`` / ``timestamp`` / ``commit``).

    Returns:
        The entry with auto-populated fields filled in.
    """
    csv_path, md_path = _log_paths()
    entry.run_id = _next_run_id(csv_path)
    entry.timestamp = dt.datetime.now(tz=dt.UTC).isoformat(timespec="seconds")
    entry.commit = _git_short_sha()

    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(_CSV_COLUMNS))
        if write_header:
            writer.writeheader()
        writer.writerow(_entry_to_csv_row(entry))

    _regenerate_markdown(csv_path, md_path)

    log.info(
        "result_logged",
        run_id=entry.run_id,
        task=entry.task,
        method=entry.method,
        metric=entry.metric,
        value=round(entry.value, 4),
        delta=entry.delta,
    )
    return entry


def _entry_to_csv_row(entry: ResultLogEntry) -> dict[str, str]:
    """Convert a :class:`ResultLogEntry` to a string-valued row dict."""
    data = asdict(entry)
    # Normalise numeric fields and replace ``None`` with empty strings so the
    # CSV is human-friendly.
    row: dict[str, str] = {}
    for col in _CSV_COLUMNS:
        raw = data.get(col)
        if raw is None:
            row[col] = ""
        elif isinstance(raw, bool):
            row[col] = "true" if raw else "false"
        elif isinstance(raw, float):
            # Use enough precision for significance testing.
            row[col] = f"{raw:.6f}"
        else:
            # Escape embedded commas via the csv writer itself; just str() here.
            s = str(raw).replace("\n", " ").strip()
            row[col] = s
    return row


def read_results() -> list[dict[str, str]]:
    """Return every row in ``result-log.csv`` as a list of dicts."""
    csv_path, _ = _log_paths()
    if not csv_path.exists():
        return []
    with csv_path.open() as f:
        return list(csv.DictReader(f))


def _regenerate_markdown(csv_path: Path, md_path: Path) -> None:
    """Rewrite ``result-log.md`` from the authoritative CSV."""
    rows: list[dict[str, str]] = []
    if csv_path.exists():
        with csv_path.open() as f:
            rows = list(csv.DictReader(f))

    lines = [
        "# Result Log",
        "",
        "All evaluation runs in one place. Newest runs at the bottom.",
        "Source of truth: [`result-log.csv`](result-log.csv).",
        "",
        "| # | When | Commit | Task | Method | Limit | Metric | Value | Δ | 95% CI | Wall | Mock |",
        "|---|------|--------|------|--------|-------|--------|-------|----|--------|------|------|",
    ]
    for row in rows:
        ci_low = row.get("ci95_low", "")
        ci_high = row.get("ci95_high", "")
        ci = f"[{ci_low}, {ci_high}]" if ci_low and ci_high else ""
        wall = row.get("wall_s", "")
        wall_fmt = f"{float(wall):.1f}s" if wall else ""
        mock = "yes" if row.get("mock_backend", "") == "true" else "no"
        lines.append(
            "| {run_id} | {timestamp} | `{commit}` | {task} | `{method}` | {limit} | "
            "{metric} | {value} | {delta} | {ci} | {wall} | {mock} |".format(
                run_id=row.get("run_id", ""),
                timestamp=row.get("timestamp", "")[:19],
                commit=row.get("commit", ""),
                task=row.get("task", ""),
                method=row.get("method", ""),
                limit=row.get("limit", "") or "full",
                metric=row.get("metric", ""),
                value=_pretty_float(row.get("value", "")),
                delta=_delta_cell(row.get("delta", "")),
                ci=ci,
                wall=wall_fmt,
                mock=mock,
            )
        )
    if not rows:
        lines.append("| _no runs yet_ | | | | | | | | | | | |")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _pretty_float(s: str, digits: int = 4) -> str:
    try:
        return f"{float(s):.{digits}f}"
    except (TypeError, ValueError):
        return s


def _delta_cell(s: str) -> str:
    """Format delta with explicit sign, empty when unavailable."""
    if not s:
        return ""
    try:
        v = float(s)
    except ValueError:
        return s
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.4f}"


def effective_results_dir() -> Path:
    """Expose the resolved results directory (honours ``LLMEVAL_RESULTS_DIR``)."""
    settings = get_settings()
    return Path(os.fspath(settings.results_dir))


__all__ = [
    "ResultLogEntry",
    "effective_results_dir",
    "log_result",
    "read_results",
]
