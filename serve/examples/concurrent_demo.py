"""Concurrency proof — measure throughput under parallel load.

Fires ``--concurrency`` identical prompts sequentially, then again using
``asyncio.gather``, and reports the speedup. On a real vLLM server the
concurrent run should benefit from request batching and be meaningfully
faster than sequential; against the mock backend both paths complete almost
instantly but the code still exercises.

Usage::

    python -m serve.examples.concurrent_demo --concurrency 8 --max-tokens 32
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from typing import Any

from common.logging import get_logger
from common.vllm_client import VLLMClient

log = get_logger(__name__)

_PROMPT_TEMPLATE = "Give a single-word synonym for the word: {word}\nSynonym:"
_WORDS: tuple[str, ...] = (
    "happy",
    "fast",
    "small",
    "bright",
    "strong",
    "quiet",
    "clever",
    "brave",
    "gentle",
    "ancient",
    "noisy",
    "warm",
    "cold",
    "sharp",
    "smooth",
    "rough",
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI argv."""
    p = argparse.ArgumentParser(description="Concurrent generation demo.")
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=32, dest="max_tokens")
    return p.parse_args(argv)


def _build_prompts(concurrency: int) -> list[str]:
    """Build ``concurrency`` prompts by cycling through the word pool."""
    return [_PROMPT_TEMPLATE.format(word=_WORDS[i % len(_WORDS)]) for i in range(concurrency)]


async def _one_request(client: VLLMClient, prompt: str, max_tokens: int) -> float:
    """Run a single request and return its per-request wall time."""
    t = time.monotonic()
    await client.generate(prompt, max_tokens=max_tokens, temperature=0.0)
    return time.monotonic() - t


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute both sequential and concurrent passes; return a summary dict."""
    prompts = _build_prompts(args.concurrency)

    async with VLLMClient() as client:
        log.info("concurrent_demo_sequential", n=len(prompts))
        seq_start = time.monotonic()
        seq_per_req: list[float] = []
        for prompt in prompts:
            seq_per_req.append(await _one_request(client, prompt, args.max_tokens))
        seq_wall = time.monotonic() - seq_start

        log.info("concurrent_demo_concurrent", n=len(prompts))
        con_start = time.monotonic()
        con_per_req = list(
            await asyncio.gather(
                *(_one_request(client, p, args.max_tokens) for p in prompts)
            )
        )
        con_wall = time.monotonic() - con_start

    speedup = round(seq_wall / con_wall, 2) if con_wall > 0 else None
    return {
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "sequential_wall_s": round(seq_wall, 4),
        "concurrent_wall_s": round(con_wall, 4),
        "speedup_x": speedup,
        "sequential_per_req_mean_s": round(sum(seq_per_req) / len(seq_per_req), 4),
        "concurrent_per_req_mean_s": round(sum(con_per_req) / len(con_per_req), 4),
        "sequential_per_req_max_s": round(max(seq_per_req), 4),
        "concurrent_per_req_max_s": round(max(con_per_req), 4),
    }


async def _amain(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    log.info("concurrent_demo_start", concurrency=args.concurrency, max_tokens=args.max_tokens)
    summary = await _run(args)
    sys.stdout.write(json.dumps(summary, indent=2) + "\n")
    log.info("concurrent_demo_complete", **summary)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m serve.examples.concurrent_demo``."""
    return asyncio.run(_amain(argv))


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main"]
