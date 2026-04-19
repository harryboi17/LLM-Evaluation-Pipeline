"""Prepare HellaSwag data for Part E.

Downloads the HellaSwag validation split via ``datasets.load_dataset`` and
writes two JSONL files under ``Settings.results_dir / "improve" /``:

* ``hellaswag_eval.jsonl`` — the evaluation subset the ablation runs on.
* ``hellaswag_fewshot_pool.jsonl`` — the pool Semantic few-shot retrieves from.

Both files are keyed on the ``ind`` field of the original dataset, deterministic
under the same ``--seed`` / ``--n-eval`` / ``--n-pool`` arguments.

Usage::

    python -m improve.prepare_data --n-eval 200 --n-pool 200 --seed 42

Against ``LLMEVAL_MOCK_BACKEND=true`` this still downloads the real HellaSwag
JSON via ``datasets`` (the data is public). If you're fully offline, use
``--source local`` to point at a pre-downloaded JSONL instead.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

from common.config import get_settings
from common.errors import EvalError
from common.logging import get_logger

log = get_logger(__name__)

_HF_DATASET = "hellaswag"


def _preprocess_example(example: dict[str, Any]) -> dict[str, Any]:
    """Normalise a HellaSwag example into our internal schema.

    Internal schema (matches what ``improve.infer`` consumes):

    * ``ind``: original dataset index (str)
    * ``ctx``: the context / story prefix
    * ``activity_label``: the activity tag (so we can strip it when desired)
    * ``endings``: ``list[str]`` of length 4
    * ``label``: int in [0, 3]
    """
    ctx = example.get("ctx") or example.get("ctx_a", "")
    ctx_b = example.get("ctx_b", "")
    if ctx_b:
        ctx = f"{ctx} {ctx_b}" if ctx else ctx_b
    return {
        "ind": str(example.get("ind", "")),
        "ctx": ctx,
        "activity_label": example.get("activity_label", ""),
        "endings": list(example.get("endings", [])),
        "label": int(example.get("label", 0)) if example.get("label", "") != "" else -1,
    }


def _load_via_datasets() -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise EvalError("datasets not installed; run `uv sync --extra eval`.") from exc
    log.info("hellaswag_download_start", split="validation")
    ds = load_dataset(_HF_DATASET, split="validation")
    rows = [_preprocess_example(dict(r)) for r in ds]
    log.info("hellaswag_download_complete", n=len(rows))
    return rows


def _load_local(path: Path) -> list[dict[str, Any]]:
    return [
        _preprocess_example(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def prepare(
    n_eval: int,
    n_pool: int,
    seed: int,
    source: str,
    local_path: Path | None,
    out_dir: Path,
) -> tuple[Path, Path]:
    """Split HellaSwag into ``(eval, fewshot_pool)`` JSONL files.

    Args:
        n_eval: Number of evaluation examples.
        n_pool: Number of few-shot pool examples (must be disjoint from eval).
        seed: RNG seed for the shuffle.
        source: ``"hf"`` (default, uses :mod:`datasets`) or ``"local"``.
        local_path: When ``source == "local"``, path to a JSONL in our schema.
        out_dir: Output directory.

    Returns:
        ``(eval_path, pool_path)`` after writing.
    """
    if source == "hf":
        all_rows = _load_via_datasets()
    elif source == "local":
        if not local_path or not local_path.exists():
            raise EvalError(f"--source local requires --local-path; got {local_path!r}")
        all_rows = _load_local(local_path)
    else:
        raise EvalError(f"unknown --source: {source}")

    # Drop examples without gold labels (the HellaSwag test split has these).
    labelled = [r for r in all_rows if r["label"] >= 0 and r["endings"]]
    if len(labelled) < n_eval + n_pool:
        raise EvalError(
            f"only {len(labelled)} labelled rows available; "
            f"need n_eval ({n_eval}) + n_pool ({n_pool}) = {n_eval + n_pool}"
        )

    rng = random.Random(seed)
    shuffled = labelled[:]
    rng.shuffle(shuffled)

    eval_rows = shuffled[:n_eval]
    pool_rows = shuffled[n_eval : n_eval + n_pool]

    out_dir.mkdir(parents=True, exist_ok=True)
    eval_path = out_dir / "hellaswag_eval.jsonl"
    pool_path = out_dir / "hellaswag_fewshot_pool.jsonl"
    _write_jsonl(eval_rows, eval_path)
    _write_jsonl(pool_rows, pool_path)
    log.info(
        "hellaswag_split_written",
        eval_path=str(eval_path),
        pool_path=str(pool_path),
        n_eval=len(eval_rows),
        n_pool=len(pool_rows),
    )
    return (eval_path, pool_path)


def _write_jsonl(rows: list[dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare HellaSwag data for Part E.")
    p.add_argument("--n-eval", type=int, default=200, dest="n_eval")
    p.add_argument("--n-pool", type=int, default=200, dest="n_pool")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--source", choices=["hf", "local"], default="hf")
    p.add_argument("--local-path", default=None, dest="local_path")
    p.add_argument("--out-dir", default=None, dest="out_dir")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = get_settings()
    seed = args.seed if args.seed is not None else settings.seed
    out_dir = (
        Path(args.out_dir) if args.out_dir else (settings.results_dir / "improve")
    )
    eval_path, pool_path = prepare(
        n_eval=args.n_eval,
        n_pool=args.n_pool,
        seed=seed,
        source=args.source,
        local_path=Path(args.local_path) if args.local_path else None,
        out_dir=out_dir,
    )
    sys.stdout.write(
        json.dumps(
            {
                "eval_path": str(eval_path),
                "pool_path": str(pool_path),
                "n_eval": args.n_eval,
                "n_pool": args.n_pool,
                "seed": seed,
            }
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main", "prepare"]
