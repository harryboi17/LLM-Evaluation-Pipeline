#!/usr/bin/env bash
# End-to-end sanity check. Uses LLMEVAL_MOCK_BACKEND=true so no GPU is required.
# Fails fast (set -e) on any stage that regressed.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[smoke] lint"
uv run ruff check .

echo "[smoke] typecheck"
uv run mypy common serve

echo "[smoke] unit tests (common + serve)"
uv run pytest tests/common tests/serve -q

echo "[smoke] mock-backend generation via VLLMClient"
LLMEVAL_MOCK_BACKEND=true uv run python - <<'PY'
import asyncio
from common.vllm_client import VLLMClient

async def main() -> None:
    async with VLLMClient() as client:
        r = await client.generate("smoke test", max_tokens=16)
        assert r.text, "empty completion"
        print(f"ok: {r.text[:60]!r} finish_reason={r.finish_reason}")

asyncio.run(main())
PY

echo "[smoke] serve.client CLI (generate)"
LLMEVAL_MOCK_BACKEND=true uv run python -m serve.client "smoke" --max-tokens 8 >/dev/null

echo "[smoke] serve.examples.demo"
LLMEVAL_MOCK_BACKEND=true uv run python -m serve.examples.demo >/dev/null

echo "[smoke] serve.examples.concurrent_demo"
LLMEVAL_MOCK_BACKEND=true uv run python -m serve.examples.concurrent_demo \
    --concurrency 4 --max-tokens 8 >/dev/null

echo "[smoke] all good"
