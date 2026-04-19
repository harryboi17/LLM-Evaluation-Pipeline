"""CLI runner for :class:`common.vllm_client.VLLMClient`.

Sends a single prompt to the vLLM server and prints the completion to ``stdout``.
At the end, emits a small JSON summary (duration, token counts) to ``stderr`` per
Rule 28 (long-running scripts emit a JSON summary).

Usage::

    # Non-streaming (default):
    python -m serve.client "What is the capital of France?"

    # Streaming:
    python -m serve.client "Write a haiku." --stream --max-tokens 64

    # Custom decoding:
    python -m serve.client "..." --temperature 0.7 --top-p 0.9 --seed 7

``VLLMClient`` is re-exported for the small number of callers that prefer
``from serve.client import VLLMClient``.
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


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI argv into a namespace."""
    p = argparse.ArgumentParser(description="Send a single prompt to the vLLM server.")
    p.add_argument("prompt", help="Prompt text (quote it in the shell).")
    p.add_argument("--max-tokens", type=int, default=128, dest="max_tokens")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top-p", type=float, default=1.0, dest="top_p")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--stop",
        action="append",
        default=None,
        help="Stop sequence; repeatable.",
    )
    p.add_argument("--stream", action="store_true", help="Use the streaming endpoint.")
    return p.parse_args(argv)


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    """Execute the request and return the JSON summary."""
    start = time.monotonic()
    async with VLLMClient() as client:
        if args.stream:
            return await _run_stream(client, args, start)
        return await _run_generate(client, args, start)


async def _run_stream(
    client: VLLMClient,
    args: argparse.Namespace,
    start: float,
) -> dict[str, Any]:
    """Stream tokens to stdout as they arrive and return a timing summary."""
    first_token_t: float | None = None
    text_parts: list[str] = []
    async for chunk in client.stream_generate(
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=args.stop,
        seed=args.seed,
    ):
        if chunk.delta:
            if first_token_t is None:
                first_token_t = time.monotonic()
            text_parts.append(chunk.delta)
            sys.stdout.write(chunk.delta)
            sys.stdout.flush()
    sys.stdout.write("\n")
    ttft = (first_token_t - start) if first_token_t is not None else None
    return {
        "mode": "stream",
        "prompt_chars": len(args.prompt),
        "completion_chars": sum(len(part) for part in text_parts),
        "ttft_s": round(ttft, 4) if ttft is not None else None,
        "wall_s": round(time.monotonic() - start, 4),
    }


async def _run_generate(
    client: VLLMClient,
    args: argparse.Namespace,
    start: float,
) -> dict[str, Any]:
    """Run a single non-streaming completion and return a timing summary."""
    result = await client.generate(
        args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stop=args.stop,
        seed=args.seed,
    )
    sys.stdout.write(result.text + "\n")
    return {
        "mode": "generate",
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "finish_reason": result.finish_reason,
        "wall_s": round(time.monotonic() - start, 4),
    }


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``python -m serve.client``.

    Args:
        argv: Optional argv for testing; defaults to ``sys.argv[1:]``.

    Returns:
        ``0`` on success, non-zero on unhandled client errors.
    """
    args = _parse_args(argv)
    summary = asyncio.run(_run(args))
    log.info("client_run_complete", **summary)
    sys.stderr.write(json.dumps(summary) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["VLLMClient", "main"]
