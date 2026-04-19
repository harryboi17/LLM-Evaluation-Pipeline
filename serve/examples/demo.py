"""Three-prompt serving demo (short greedy, long with stops, streaming).

Runs against a live vLLM server by default, or against the canned mock backend
when ``LLMEVAL_MOCK_BACKEND=true``. Prints each completion to ``stdout`` and a
JSON summary of all three runs to ``stderr`` at the end.

Usage::

    # Against the mock backend (no GPU needed):
    LLMEVAL_MOCK_BACKEND=true python -m serve.examples.demo

    # Against a running vLLM server (after `make serve`):
    python -m serve.examples.demo
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any

from common.logging import get_logger
from common.vllm_client import VLLMClient

log = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class _DemoPrompt:
    label: str
    prompt: str
    max_tokens: int
    temperature: float = 0.0
    stop: list[str] | None = None
    stream: bool = False
    extra: dict[str, Any] = field(default_factory=dict)


_PROMPTS: tuple[_DemoPrompt, ...] = (
    _DemoPrompt(
        label="short_greedy",
        prompt="What is the capital of France? Answer in one word.",
        max_tokens=16,
    ),
    _DemoPrompt(
        label="long_with_stops",
        prompt=(
            "List three benefits of using Python for data science. "
            "Number each item on a new line starting with the number.\n1."
        ),
        max_tokens=128,
        stop=["\n4.", "\n\n"],
    ),
    _DemoPrompt(
        label="stream_demo",
        prompt="Write a two-line haiku about distributed systems.",
        max_tokens=64,
        stream=True,
    ),
)


async def _run_non_stream(client: VLLMClient, spec: _DemoPrompt) -> dict[str, Any]:
    """Execute a single non-streaming prompt and return a result dict."""
    t0 = time.monotonic()
    result = await client.generate(
        spec.prompt,
        max_tokens=spec.max_tokens,
        temperature=spec.temperature,
        stop=spec.stop,
    )
    return {
        "label": spec.label,
        "mode": "generate",
        "prompt": spec.prompt,
        "completion": result.text,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "finish_reason": result.finish_reason,
        "wall_s": round(time.monotonic() - t0, 4),
    }


async def _run_stream(client: VLLMClient, spec: _DemoPrompt) -> dict[str, Any]:
    """Execute a streaming prompt, printing tokens live, and return timing info."""
    t0 = time.monotonic()
    first_token_t: float | None = None
    text_parts: list[str] = []
    async for chunk in client.stream_generate(
        spec.prompt,
        max_tokens=spec.max_tokens,
        temperature=spec.temperature,
        stop=spec.stop,
    ):
        if chunk.delta:
            if first_token_t is None:
                first_token_t = time.monotonic()
            text_parts.append(chunk.delta)
    text = "".join(text_parts)
    ttft_s = (first_token_t - t0) if first_token_t is not None else None
    return {
        "label": spec.label,
        "mode": "stream",
        "prompt": spec.prompt,
        "completion": text,
        "ttft_s": round(ttft_s, 4) if ttft_s is not None else None,
        "wall_s": round(time.monotonic() - t0, 4),
    }


async def _amain() -> int:
    """Run all three demo prompts sequentially and emit a JSON summary."""
    log.info("demo_start", count=len(_PROMPTS))
    results: list[dict[str, Any]] = []
    async with VLLMClient() as client:
        for spec in _PROMPTS:
            out = await (_run_stream(client, spec) if spec.stream else _run_non_stream(client, spec))
            results.append(out)
            sys.stdout.write(f"--- {out['label']} ({out['mode']}) ---\n")
            sys.stdout.write(out["completion"].rstrip() + "\n\n")
            sys.stdout.flush()
    summary = {"demo": "serve.examples.demo", "results": results}
    # Single-line JSON on its own terminal line so the test (and any downstream
    # consumer) can parse it with `tail -n1 | jq`.
    sys.stdout.write(json.dumps(summary) + "\n")
    log.info("demo_complete", count=len(results))
    return 0


def main() -> int:
    """Entry point for ``python -m serve.examples.demo``."""
    return asyncio.run(_amain())


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["main"]
