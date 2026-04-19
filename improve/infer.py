"""Part E — custom HellaSwag eval loop with per-variant ablation.

Flow:

1. Load ``hellaswag_eval.jsonl`` + ``hellaswag_fewshot_pool.jsonl`` (written by
   :mod:`improve.prepare_data`).
2. For each example and each candidate ending, build a ``(prompt, continuation)``
   pair from the chosen variant and score the continuation log-likelihood via
   :class:`common.vllm_client.VLLMClient` (``echo=True``, ``max_tokens=0``,
   ``logprobs=5``). Every request goes through :class:`common.cache.PromptCache`
   keyed on (model, prompt+continuation, scoring params) so repeated ablations
   hit the cache.
3. Select the argmax option under the variant's scoring rule
   (``sum`` / ``length_norm`` / ``byte_norm``).
4. Record per-example correctness, aggregate to accuracy, and (if this variant
   is not the baseline) compute a 95% paired-bootstrap CI against the cached
   baseline run.
5. Append one row to ``results/result-log.csv`` via
   :mod:`common.result_log` and one entry to ``docs/improvement-log.md``.

All heavy work is async and batched through an :class:`asyncio.Semaphore` so
concurrency is bounded by ``--max-concurrency``.

Run a single variant::

    LLMEVAL_MOCK_BACKEND=true python -m improve.infer \\
        --variant baseline --n-eval 30

Run the full ablation via :file:`improve/eval.sh`.
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common.cache import PromptCache
from common.config import get_settings
from common.errors import EvalError
from common.logging import get_logger
from common.result_log import ResultLogEntry, log_result
from common.stats import paired_bootstrap
from common.vllm_client import VLLMClient
from improve.optimize_prompt import (
    FewshotExample,
    PromptPair,
    ScoringMode,
    SemanticRetriever,
    build_pairs_baseline,
    build_pairs_clean,
    build_pairs_fewshot,
    score_logprob,
)

log = get_logger(__name__)

_LOGPROBS_TOP_K = 5


@dataclass(frozen=True, slots=True)
class VariantConfig:
    """One ablation cell: prompt builder + scoring mode + few-shot settings."""

    name: str
    kind: str  # "baseline" | "clean" | "fewshot_random" | "fewshot_semantic"
    scoring: ScoringMode
    k: int = 0


_VARIANTS: dict[str, VariantConfig] = {
    "baseline": VariantConfig("baseline", "baseline", "sum"),
    "length_norm": VariantConfig("length_norm", "baseline", "length_norm"),
    "byte_norm": VariantConfig("byte_norm", "baseline", "byte_norm"),
    "clean_prompt": VariantConfig("clean_prompt", "clean", "sum"),
    "clean_length_norm": VariantConfig("clean_length_norm", "clean", "length_norm"),
    "fewshot_random_5": VariantConfig(
        "fewshot_random_5", "fewshot_random", "length_norm", k=5
    ),
    "fewshot_semantic_5": VariantConfig(
        "fewshot_semantic_5", "fewshot_semantic", "length_norm", k=5
    ),
}


@dataclass(frozen=True, slots=True)
class ExampleResult:
    """Per-example outcome recorded by ``evaluate()``."""

    ind: str
    label: int
    predicted: int
    correct: bool
    scores: list[float]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file into a list of dicts."""
    if not path.exists():
        raise EvalError(f"missing file: {path}")
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _to_fewshot_examples(rows: list[dict[str, Any]]) -> list[FewshotExample]:
    return [
        FewshotExample(
            ctx=str(r.get("ctx", "")),
            activity=str(r.get("activity_label", "")),
            endings=list(r.get("endings", [])),
            label=int(r.get("label", 0)),
        )
        for r in rows
    ]


def _pick_fewshot(
    variant: VariantConfig,
    query_ctx: str,
    pool: list[FewshotExample],
    retriever: SemanticRetriever | None,
    rng: random.Random,
) -> list[FewshotExample]:
    """Resolve the few-shot list for this query under the given variant."""
    if variant.k <= 0 or not pool:
        return []
    if variant.kind == "fewshot_random":
        k = min(variant.k, len(pool))
        return rng.sample(pool, k)
    if variant.kind == "fewshot_semantic":
        assert retriever is not None
        return retriever.topk(query_ctx, variant.k)
    return []


def _build_pairs(
    variant: VariantConfig,
    ctx: str,
    endings: list[str],
    activity: str,
    fewshot: list[FewshotExample],
) -> list[PromptPair]:
    """Dispatch to the variant's prompt builder."""
    if variant.kind == "baseline":
        return build_pairs_baseline(ctx, endings, activity, fewshot)
    if variant.kind == "clean":
        return build_pairs_clean(ctx, endings, activity, fewshot)
    if variant.kind in {"fewshot_random", "fewshot_semantic"}:
        return build_pairs_fewshot(ctx, endings, activity, fewshot, clean=True)
    raise EvalError(f"unknown variant.kind: {variant.kind}")


async def _score_pair(
    client: VLLMClient,
    sem: asyncio.Semaphore,
    cache: PromptCache,
    model: str,
    pair: PromptPair,
    scoring: ScoringMode,
) -> float:
    """Score a single ``PromptPair`` — cached, async, returns a scalar score.

    Cache key pins the full request shape: prompt text (``prompt + continuation``),
    echo, logprobs, max_tokens, temperature. Changing the scoring mode alone
    does not require re-hitting the model — we cache the raw token logprobs and
    derive scores locally.
    """
    full_prompt = pair.prompt + pair.continuation
    cache_params: dict[str, Any] = {
        "echo": True,
        "logprobs": _LOGPROBS_TOP_K,
        "max_tokens": 0,
        "temperature": 0.0,
    }
    raw = cache.get(model, full_prompt, cache_params)
    if raw is None:
        async with sem:
            result = await client.generate(
                full_prompt,
                max_tokens=0,
                temperature=0.0,
                echo=True,
                logprobs=_LOGPROBS_TOP_K,
            )
        raw = dict(result.raw) if result.raw else {}
        cache.put(model, full_prompt, cache_params, raw)

    logprobs_block = raw.get("choices", [{}])[0].get("logprobs") or {}
    tokens: list[str] = logprobs_block.get("tokens", [])
    token_logprobs: list[float | None] = logprobs_block.get("token_logprobs", [])

    # Figure out how many tokens the continuation contributed. We look at the
    # number of tokens whose text-offset (or token index) corresponds to the
    # continuation. vLLM / OpenAI doesn't always provide text_offset, so we
    # fall back to "anything beyond the prompt's whitespace-token count":
    prompt_toks = max(0, len(pair.prompt.split()))
    # Mock backend emits N whitespace tokens for the full prompt; real vLLM
    # emits BPE tokens. Both converge on "continuation is the tail of tokens".
    n_cont = max(1, len(tokens) - prompt_toks)
    cont_logprobs = token_logprobs[-n_cont:] if token_logprobs else []
    cont_logprobs_clean: list[float] = [lp for lp in cont_logprobs if lp is not None]
    return score_logprob(
        cont_logprobs_clean,
        ending=pair.continuation,
        mode=scoring,
    )


async def _evaluate_example(
    client: VLLMClient,
    sem: asyncio.Semaphore,
    cache: PromptCache,
    model: str,
    variant: VariantConfig,
    row: dict[str, Any],
    pool: list[FewshotExample],
    retriever: SemanticRetriever | None,
    rng: random.Random,
) -> ExampleResult:
    """Score all endings for one example and return the per-example outcome."""
    ctx = str(row.get("ctx", ""))
    activity = str(row.get("activity_label", ""))
    endings = [str(e) for e in row.get("endings", [])]
    label = int(row.get("label", -1))

    fewshot = _pick_fewshot(variant, ctx, pool, retriever, rng)
    pairs = _build_pairs(variant, ctx, endings, activity, fewshot)

    scores = await asyncio.gather(
        *(_score_pair(client, sem, cache, model, p, variant.scoring) for p in pairs)
    )
    score_list: list[float] = list(scores)
    predicted = max(range(len(score_list)), key=lambda i: score_list[i])
    return ExampleResult(
        ind=str(row.get("ind", "")),
        label=label,
        predicted=predicted,
        correct=(predicted == label),
        scores=score_list,
    )


async def evaluate(
    variant_name: str,
    eval_rows: list[dict[str, Any]],
    pool_rows: list[dict[str, Any]],
    max_concurrency: int,
    seed: int,
    model: str,
) -> list[ExampleResult]:
    """Run ``variant_name`` over ``eval_rows`` and return per-example results."""
    if variant_name not in _VARIANTS:
        raise EvalError(f"unknown variant: {variant_name}")
    variant = _VARIANTS[variant_name]

    pool = _to_fewshot_examples(pool_rows)
    retriever: SemanticRetriever | None = None
    if variant.kind == "fewshot_semantic":
        retriever = SemanticRetriever(pool)

    cache = PromptCache()
    rng = random.Random(seed)
    sem = asyncio.Semaphore(max_concurrency)
    async with VLLMClient() as client:
        return list(
            await asyncio.gather(
                *(
                    _evaluate_example(
                        client,
                        sem,
                        cache,
                        model,
                        variant,
                        row,
                        pool,
                        retriever,
                        rng,
                    )
                    for row in eval_rows
                )
            )
        )


def _accuracy(results: list[ExampleResult]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r.correct) / len(results)


def _load_baseline_correctness(out_dir: Path) -> list[int] | None:
    """Read the baseline run's per-example correctness vector, if present."""
    path = out_dir / "baseline.json"
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [int(c) for c in payload.get("correct_per_example", [])]


def _save_variant_results(
    out_dir: Path,
    variant_name: str,
    results: list[ExampleResult],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{variant_name}.json"
    path.write_text(
        json.dumps(
            {
                "variant": variant_name,
                "n": len(results),
                "accuracy": _accuracy(results),
                "correct_per_example": [int(r.correct) for r in results],
                "per_example": [
                    {
                        "ind": r.ind,
                        "label": r.label,
                        "predicted": r.predicted,
                        "correct": r.correct,
                        "scores": r.scores,
                    }
                    for r in results
                ],
            },
            indent=2,
        )
    )
    return path


def _append_improvement_log(
    improve_log: Path,
    variant_name: str,
    notes: str,
    acc: float,
    delta: float | None,
    ci95: tuple[float, float] | None,
    p_value: float | None,
    n: int,
    wall_s: float,
    commit: str,
    model: str,
    seed: int,
    mock: bool,
) -> None:
    """Append a per-variant markdown entry to the improvement log."""
    improve_log.parent.mkdir(parents=True, exist_ok=True)
    if not improve_log.exists():
        improve_log.write_text("# Improvement Log\n\n")

    when = dt.datetime.now(tz=dt.UTC).isoformat(timespec="minutes")
    delta_str = "n/a (baseline)" if delta is None else f"{delta:+.4f}"
    ci_str = "n/a (baseline)" if ci95 is None else f"[{ci95[0]:+.4f}, {ci95[1]:+.4f}]"
    if ci95 is None:
        sig_str = "n/a (baseline)"
    elif ci95[0] > 0.0:
        sig_str = "significantly better (95% CI > 0)"
    elif ci95[1] < 0.0:
        sig_str = "significantly worse (95% CI < 0)"
    else:
        sig_str = "not significant (95% CI includes 0)"
    p_str = "n/a (baseline)" if p_value is None else f"{p_value:.4f}"

    block = (
        f"\n## {variant_name} — hellaswag\n"
        f"- **When:** {when}\n"
        f"- **Commit:** `{commit}`\n"
        f"- **Task / limit / seed:** hellaswag / {n} / {seed}\n"
        f"- **Model:** `{model}` (mock_backend={mock})\n"
        f"- **Accuracy:** {acc:.4f}\n"
        f"- **Delta vs baseline:** {delta_str}\n"
        f"- **95% CI (paired bootstrap, 10k):** {ci_str}\n"
        f"- **Two-sided p-value:** {p_str}\n"
        f"- **Significance:** {sig_str}\n"
        f"- **Wall time:** {wall_s:.1f}s\n"
        f"- **Notes:** {notes or '(none)'}\n"
    )
    with improve_log.open("a", encoding="utf-8") as f:
        f.write(block)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one Part-E variant end-to-end.")
    p.add_argument("--variant", default="baseline", choices=sorted(_VARIANTS))
    p.add_argument("--n-eval", type=int, default=None, dest="n_eval")
    p.add_argument("--eval-path", default=None, dest="eval_path")
    p.add_argument("--pool-path", default=None, dest="pool_path")
    p.add_argument("--out-dir", default=None, dest="out_dir")
    p.add_argument("--max-concurrency", type=int, default=8, dest="max_concurrency")
    p.add_argument("--notes", default="", help="Free-text notes for the logs.")
    p.add_argument(
        "--bootstrap-iters",
        type=int,
        default=10_000,
        dest="bootstrap_iters",
        help="Paired-bootstrap resamples for the 95% CI vs baseline.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = get_settings()

    out_dir = Path(args.out_dir) if args.out_dir else (settings.results_dir / "improve")
    eval_path = Path(args.eval_path) if args.eval_path else (out_dir / "hellaswag_eval.jsonl")
    pool_path = Path(args.pool_path) if args.pool_path else (out_dir / "hellaswag_fewshot_pool.jsonl")

    eval_rows = load_jsonl(eval_path)
    pool_rows = load_jsonl(pool_path)
    if args.n_eval is not None:
        eval_rows = eval_rows[: args.n_eval]

    log.info(
        "improve_variant_start",
        variant=args.variant,
        n_eval=len(eval_rows),
        n_pool=len(pool_rows),
        model=settings.model_name,
    )
    t0 = time.monotonic()
    results = asyncio.run(
        evaluate(
            variant_name=args.variant,
            eval_rows=eval_rows,
            pool_rows=pool_rows,
            max_concurrency=args.max_concurrency,
            seed=settings.seed,
            model=settings.model_name,
        )
    )
    wall_s = time.monotonic() - t0

    variant_path = _save_variant_results(out_dir, args.variant, results)
    acc = _accuracy(results)

    delta: float | None = None
    ci95: tuple[float, float] | None = None
    p_value: float | None = None
    baseline_correct = _load_baseline_correctness(out_dir)
    variant_correct = [int(r.correct) for r in results]
    if args.variant != "baseline" and baseline_correct is not None:
        n_match = min(len(baseline_correct), len(variant_correct))
        if n_match > 0:
            bootstrap = paired_bootstrap(
                baseline=baseline_correct[:n_match],
                improved=variant_correct[:n_match],
                resamples=args.bootstrap_iters,
                seed=settings.seed,
            )
            delta = bootstrap.diff_acc
            ci95 = (bootstrap.ci_low, bootstrap.ci_high)
            p_value = bootstrap.p_value

    # --- result log -----------------------------------------------------------
    from common.result_log import _git_short_sha

    commit = _git_short_sha()
    log_result(
        ResultLogEntry(
            model=settings.model_name,
            task="hellaswag",
            method=args.variant,
            metric="acc",
            value=acc,
            seed=settings.seed,
            limit=len(results),
            stderr=None,
            delta=delta,
            ci95_low=ci95[0] if ci95 else None,
            ci95_high=ci95[1] if ci95 else None,
            wall_s=wall_s,
            mock_backend=settings.mock_backend,
            notes=args.notes,
        )
    )

    # --- improvement log ------------------------------------------------------
    improve_log_path = Path(__file__).parent.parent / "docs" / "improvement-log.md"
    _append_improvement_log(
        improve_log_path,
        args.variant,
        notes=args.notes,
        acc=acc,
        delta=delta,
        ci95=ci95,
        p_value=p_value,
        n=len(results),
        wall_s=wall_s,
        commit=commit,
        model=settings.model_name,
        seed=settings.seed,
        mock=settings.mock_backend,
    )

    summary = {
        "variant": args.variant,
        "accuracy": acc,
        "n": len(results),
        "delta_vs_baseline": delta,
        "ci95": list(ci95) if ci95 else None,
        "p_value": p_value,
        "wall_s": round(wall_s, 3),
        "variant_json": str(variant_path),
    }
    sys.stdout.write(json.dumps(summary) + "\n")
    log.info("improve_variant_complete", **summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["ExampleResult", "VariantConfig", "evaluate", "main"]
