"""Asyncio-based load generator for a vLLM OpenAI-compatible server.

Fires a configurable number of concurrent requests (``--concurrency``) across a
configurable total number of requests (``--num-requests``) drawn from a prompt
pool that mixes **short** and **long** prompts. For each request records:

* ``wall_s`` — end-to-end wall time
* ``ttft_s`` — time to first token (streaming only)
* ``output_tokens`` — completion token count (for TPOT derivation)
* ``status`` — ``ok`` / ``error`` / ``timeout``
* optional ``gpu_util_pct`` / ``gpu_mem_mb`` — if a :mod:`perf.gpu_monitor` run
  is attached

Output is a single ``metrics.csv`` with one row per request. Percentiles and
aggregates are computed by :mod:`perf.metrics` (kept separate so analysis is
re-runnable without re-firing load).

Usage::

    # Quick smoke (offline, mock backend):
    LLMEVAL_MOCK_BACKEND=true python -m perf.load_test \\
        --num-requests 32 --concurrency 8 --output metrics.csv

    # Real run (requires `make serve`):
    python -m perf.load_test \\
        --num-requests 200 --concurrency 16 --mode stream \\
        --output results/perf/metrics.csv
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from common.config import get_settings
from common.errors import VLLMClientError, VLLMTimeoutError
from common.logging import get_logger
from common.vllm_client import VLLMClient

log = get_logger(__name__)

_SHORT_PROMPTS: tuple[str, ...] = (
    "What is 2+2?",
    "Translate 'hello' to French.",
    "Name a large city.",
    "Complete: the sky is",
    "What color is a banana?",
    "Pick a prime number under 20.",
)

_LONG_PROMPT = (
    "You are a technical writer. Write a detailed, three-paragraph overview of "
    "how a modern GPU-backed LLM inference server achieves high throughput, "
    "covering continuous batching, paged attention memory management, and "
    "prefill versus decode scheduling. Use concrete numbers where appropriate "
    "and reference the tradeoffs between latency and throughput. Finish with a "
    "short summary sentence."
)


@dataclass(frozen=True, slots=True)
class _RequestSpec:
    """Immutable spec for one synthetic request."""

    idx: int
    prompt_kind: str  # "short" | "long"
    prompt: str
    max_tokens: int


@dataclass(slots=True)
class RequestMetric:
    """One row of ``metrics.csv`` — see module docstring."""

    idx: int
    prompt_kind: str
    prompt_chars: int
    max_tokens: int
    mode: str  # "generate" | "stream"
    wall_s: float
    ttft_s: float | None
    output_tokens: int
    status: str
    error: str = ""
    started_at: float = 0.0
    ended_at: float = 0.0
    extra: dict[str, float] = field(default_factory=dict)


def _build_specs(
    num_requests: int,
    short_frac: float,
    short_max_tokens: int,
    long_max_tokens: int,
    rng: random.Random,
) -> list[_RequestSpec]:
    """Deterministically build ``num_requests`` specs mixing short / long prompts."""
    specs: list[_RequestSpec] = []
    for i in range(num_requests):
        is_short = rng.random() < short_frac
        if is_short:
            prompt = rng.choice(_SHORT_PROMPTS)
            specs.append(
                _RequestSpec(
                    idx=i,
                    prompt_kind="short",
                    prompt=prompt,
                    max_tokens=short_max_tokens,
                )
            )
        else:
            specs.append(
                _RequestSpec(
                    idx=i,
                    prompt_kind="long",
                    prompt=_LONG_PROMPT,
                    max_tokens=long_max_tokens,
                )
            )
    return specs


async def _run_one(
    client: VLLMClient,
    sem: asyncio.Semaphore,
    spec: _RequestSpec,
    mode: str,
) -> RequestMetric:
    """Execute a single request and return its metric row."""
    async with sem:
        started = time.monotonic()
        started_wall = time.time()
        try:
            if mode == "stream":
                first_token_t: float | None = None
                out_tokens = 0
                text_parts: list[str] = []
                async for chunk in client.stream_generate(
                    spec.prompt,
                    max_tokens=spec.max_tokens,
                    temperature=0.0,
                ):
                    if chunk.delta:
                        if first_token_t is None:
                            first_token_t = time.monotonic()
                        # approximate; refined by metrics.py if a tokenizer is attached
                        out_tokens += max(1, len(chunk.delta.split()))
                        text_parts.append(chunk.delta)
                ended = time.monotonic()
                return RequestMetric(
                    idx=spec.idx,
                    prompt_kind=spec.prompt_kind,
                    prompt_chars=len(spec.prompt),
                    max_tokens=spec.max_tokens,
                    mode="stream",
                    wall_s=ended - started,
                    ttft_s=(first_token_t - started) if first_token_t else None,
                    output_tokens=out_tokens,
                    status="ok",
                    started_at=started_wall,
                    ended_at=started_wall + (ended - started),
                )
            result = await client.generate(
                spec.prompt,
                max_tokens=spec.max_tokens,
                temperature=0.0,
            )
            ended = time.monotonic()
            return RequestMetric(
                idx=spec.idx,
                prompt_kind=spec.prompt_kind,
                prompt_chars=len(spec.prompt),
                max_tokens=spec.max_tokens,
                mode="generate",
                wall_s=ended - started,
                ttft_s=None,
                output_tokens=result.completion_tokens,
                status="ok",
                started_at=started_wall,
                ended_at=started_wall + (ended - started),
            )
        except VLLMTimeoutError as exc:
            return RequestMetric(
                idx=spec.idx,
                prompt_kind=spec.prompt_kind,
                prompt_chars=len(spec.prompt),
                max_tokens=spec.max_tokens,
                mode=mode,
                wall_s=time.monotonic() - started,
                ttft_s=None,
                output_tokens=0,
                status="timeout",
                error=str(exc),
                started_at=started_wall,
                ended_at=started_wall + (time.monotonic() - started),
            )
        except VLLMClientError as exc:
            return RequestMetric(
                idx=spec.idx,
                prompt_kind=spec.prompt_kind,
                prompt_chars=len(spec.prompt),
                max_tokens=spec.max_tokens,
                mode=mode,
                wall_s=time.monotonic() - started,
                ttft_s=None,
                output_tokens=0,
                status="error",
                error=str(exc),
                started_at=started_wall,
                ended_at=started_wall + (time.monotonic() - started),
            )


async def run_load(
    num_requests: int,
    concurrency: int,
    short_frac: float,
    short_max_tokens: int,
    long_max_tokens: int,
    mode: str,
    seed: int,
) -> list[RequestMetric]:
    """Fire ``num_requests`` at bounded ``concurrency`` and return per-request rows."""
    rng = random.Random(seed)
    specs = _build_specs(
        num_requests=num_requests,
        short_frac=short_frac,
        short_max_tokens=short_max_tokens,
        long_max_tokens=long_max_tokens,
        rng=rng,
    )
    sem = asyncio.Semaphore(concurrency)
    async with VLLMClient() as client:
        metrics = await asyncio.gather(*(_run_one(client, sem, s, mode) for s in specs))
    return list(metrics)


_CSV_COLUMNS: tuple[str, ...] = (
    "idx",
    "prompt_kind",
    "prompt_chars",
    "max_tokens",
    "mode",
    "wall_s",
    "ttft_s",
    "output_tokens",
    "status",
    "error",
    "started_at",
    "ended_at",
)


def write_csv(metrics: list[RequestMetric], path: Path) -> None:
    """Write one row per metric to ``path`` with a stable column order."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(_CSV_COLUMNS))
        writer.writeheader()
        for m in metrics:
            row = {k: v for k, v in asdict(m).items() if k in _CSV_COLUMNS}
            # Normalise optional float for CSV friendliness.
            if row["ttft_s"] is None:
                row["ttft_s"] = ""
            writer.writerow(row)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI argv."""
    p = argparse.ArgumentParser(description="Async load test against vLLM.")
    p.add_argument("--num-requests", type=int, default=64, dest="num_requests")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--short-frac", type=float, default=0.6, dest="short_frac")
    p.add_argument("--short-max-tokens", type=int, default=32, dest="short_max_tokens")
    p.add_argument("--long-max-tokens", type=int, default=256, dest="long_max_tokens")
    p.add_argument("--mode", choices=["generate", "stream"], default="generate")
    p.add_argument("--output", default="metrics.csv", help="CSV output path.")
    p.add_argument("--seed", type=int, default=None, help="RNG seed; defaults to Settings.seed.")
    return p.parse_args(argv)


async def _amain(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = get_settings()
    seed = args.seed if args.seed is not None else settings.seed

    log.info(
        "load_test_start",
        num_requests=args.num_requests,
        concurrency=args.concurrency,
        short_frac=args.short_frac,
        mode=args.mode,
        seed=seed,
    )
    t0 = time.monotonic()
    metrics = await run_load(
        num_requests=args.num_requests,
        concurrency=args.concurrency,
        short_frac=args.short_frac,
        short_max_tokens=args.short_max_tokens,
        long_max_tokens=args.long_max_tokens,
        mode=args.mode,
        seed=seed,
    )
    wall = time.monotonic() - t0

    output_path = Path(args.output)
    write_csv(metrics, output_path)

    n_ok = sum(1 for m in metrics if m.status == "ok")
    summary = {
        "num_requests": args.num_requests,
        "concurrency": args.concurrency,
        "mode": args.mode,
        "ok": n_ok,
        "errors": args.num_requests - n_ok,
        "wall_s": round(wall, 4),
        "throughput_rps": round(args.num_requests / wall, 2) if wall > 0 else None,
        "output_csv": str(output_path),
    }
    sys.stdout.write(json.dumps(summary) + "\n")
    log.info("load_test_complete", **summary)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m perf.load_test``."""
    return asyncio.run(_amain(argv))


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["RequestMetric", "main", "run_load", "write_csv"]
