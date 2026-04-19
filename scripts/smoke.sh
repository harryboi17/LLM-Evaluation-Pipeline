#!/usr/bin/env bash
# End-to-end sanity check. Uses LLMEVAL_MOCK_BACKEND=true so no GPU is required.
# Fails fast (set -e) on any stage that regressed.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[smoke] lint"
uv run ruff check .

echo "[smoke] typecheck"
uv run mypy common serve eval_runner

echo "[smoke] unit tests (common + serve + eval_runner)"
uv run pytest tests/common tests/serve tests/eval_runner -q

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

echo "[smoke] eval_runner.run_eval (custom_qa, limit=3) if lm-eval available"
if uv run python -c "import lm_eval" 2>/dev/null; then
    SMOKE_OUT=$(mktemp -d)
    LLMEVAL_MOCK_BACKEND=true \
    LLMEVAL_MODEL_NAME=mock/test \
    LLMEVAL_CACHE_DIR="$SMOKE_OUT/cache" \
    LLMEVAL_RESULTS_DIR="$SMOKE_OUT/results" \
        uv run python -m eval_runner.run_eval --task custom_qa --limit 3 >/dev/null
    test -f "$SMOKE_OUT/results/custom_qa.json"
    test -f "$SMOKE_OUT/results/summary.md"
    test -f "$SMOKE_OUT/results/run_meta.json"
    rm -rf "$SMOKE_OUT"
else
    echo "  (skipped — lm-eval not installed; run 'uv sync --extra eval' to enable)"
fi

echo "[smoke] all good"
