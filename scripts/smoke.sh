#!/usr/bin/env bash
# End-to-end sanity check. Uses LLMEVAL_MOCK_BACKEND=true so no GPU is required.
# Fails fast (set -e) on any stage that regressed.
set -euo pipefail

cd "$(dirname "$0")/.."

echo "[smoke] lint"
uv run ruff check .

echo "[smoke] typecheck"
uv run mypy common serve eval_runner perf guardrails improve

echo "[smoke] unit tests (all)"
uv run pytest tests -q

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
        uv run python -m eval_runner.run_eval --task custom_qa --limit 3 --method pipeline_smoke >/dev/null
    test -f "$SMOKE_OUT/results/custom_qa.json"
    test -f "$SMOKE_OUT/results/summary.md"
    test -f "$SMOKE_OUT/results/run_meta.json"
    test -f "$SMOKE_OUT/results/result-log.csv"
    rm -rf "$SMOKE_OUT"
else
    echo "  (skipped - lm-eval not installed; run 'uv sync --extra eval' to enable)"
fi

echo "[smoke] perf.load_test + perf.metrics"
PERF_OUT=$(mktemp -d)
LLMEVAL_MOCK_BACKEND=true uv run python -m perf.load_test \
    --num-requests 12 --concurrency 4 --mode generate \
    --output "$PERF_OUT/metrics.csv" >/dev/null
LLMEVAL_MOCK_BACKEND=true uv run python -m perf.metrics \
    --input "$PERF_OUT/metrics.csv" --summary "$PERF_OUT/summary.csv" >/dev/null
test -f "$PERF_OUT/metrics.csv"
test -f "$PERF_OUT/summary.csv"
rm -rf "$PERF_OUT"

echo "[smoke] guardrails.validate (determinism + batch validate)"
GUARD_OUT=$(mktemp -d)
printf '%s\n' '{"output":"Paris"}' '{"output":"Tokyo"}' '{"output":"56"}' > "$GUARD_OUT/out.jsonl"
uv run python -m guardrails.validate validate "$GUARD_OUT/out.jsonl" --short-answer >/dev/null
LLMEVAL_MOCK_BACKEND=true uv run python -m guardrails.validate determinism \
    "What is 2+2?" --n-runs 3 --max-tokens 8 >/dev/null
rm -rf "$GUARD_OUT"

echo "[smoke] improve.infer (baseline + one variant, tiny synthetic set)"
IMP_OUT=$(mktemp -d)
uv run python - <<PY >/dev/null
import json, random
from pathlib import Path
out = Path("$IMP_OUT/improve")
out.mkdir(parents=True, exist_ok=True)
rng = random.Random(42)
rows = []
for i in range(6):
    endings = [
        " answer A.", " answer B.", " answer C.", " answer D.",
    ]
    rng.shuffle(endings)
    rows.append({
        "ind": f"smoke-{i}",
        "ctx": f"[header] smoke Story {i}: the thing happens and then",
        "activity_label": "smoke",
        "endings": endings,
        "label": i % 4,
    })
(out / "hellaswag_eval.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
(out / "hellaswag_fewshot_pool.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
PY
LLMEVAL_MOCK_BACKEND=true LLMEVAL_MODEL_NAME=mock/smoke \
LLMEVAL_CACHE_DIR="$IMP_OUT/cache" LLMEVAL_RESULTS_DIR="$IMP_OUT" \
    uv run python -m improve.infer --variant baseline --n-eval 6 \
    --bootstrap-iters 200 \
    --eval-path "$IMP_OUT/improve/hellaswag_eval.jsonl" \
    --pool-path "$IMP_OUT/improve/hellaswag_fewshot_pool.jsonl" \
    --out-dir "$IMP_OUT/improve" >/dev/null
LLMEVAL_MOCK_BACKEND=true LLMEVAL_MODEL_NAME=mock/smoke \
LLMEVAL_CACHE_DIR="$IMP_OUT/cache" LLMEVAL_RESULTS_DIR="$IMP_OUT" \
    uv run python -m improve.infer --variant length_norm --n-eval 6 \
    --bootstrap-iters 200 \
    --eval-path "$IMP_OUT/improve/hellaswag_eval.jsonl" \
    --pool-path "$IMP_OUT/improve/hellaswag_fewshot_pool.jsonl" \
    --out-dir "$IMP_OUT/improve" >/dev/null
test -f "$IMP_OUT/improve/baseline.json"
test -f "$IMP_OUT/improve/length_norm.json"
test -f "$IMP_OUT/result-log.csv"
rm -rf "$IMP_OUT"

echo "[smoke] all good"
