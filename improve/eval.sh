#!/usr/bin/env bash
# Part E orchestrator -- runs the full ablation ladder end-to-end.
#
# Uses the mock backend by default so the pipeline runs anywhere. Point at a
# real vLLM by unsetting LLMEVAL_MOCK_BACKEND and setting LLMEVAL_MODEL_NAME.
#
# Usage:
#   bash improve/eval.sh               # defaults: n_eval=30 against mock
#   N_EVAL=200 bash improve/eval.sh    # bigger run
#   LLMEVAL_MOCK_BACKEND=false LLMEVAL_MODEL_NAME=meta-llama/Llama-3.2-1B-Instruct \
#       N_EVAL=200 bash improve/eval.sh
#
# The script writes:
#   - results/improve/hellaswag_eval.jsonl  (via prepare_data)
#   - results/improve/hellaswag_fewshot_pool.jsonl
#   - results/improve/<variant>.json for every variant
#   - appends one row per variant to results/result-log.csv / .md
#   - appends one entry per variant to docs/improvement-log.md
set -euo pipefail

cd "$(dirname "$0")/.."

: "${N_EVAL:=30}"
: "${N_POOL:=30}"
: "${MAX_CONCURRENCY:=8}"
: "${LLMEVAL_MOCK_BACKEND:=true}"
export LLMEVAL_MOCK_BACKEND

VARIANTS=(
  "baseline"
  "length_norm"
  "byte_norm"
  "clean_prompt"
  "clean_length_norm"
  "fewshot_random_5"
  "fewshot_semantic_5"
)

echo "[improve] preparing data (n_eval=$N_EVAL, n_pool=$N_POOL)"
if [ ! -f results/improve/hellaswag_eval.jsonl ] || [ ! -f results/improve/hellaswag_fewshot_pool.jsonl ]; then
    uv run python -m improve.prepare_data --n-eval "$N_EVAL" --n-pool "$N_POOL" \
        || {
            echo "[improve] dataset prep failed - is 'datasets' installed? Run: uv sync --extra eval --extra improve"
            exit 2
        }
else
    echo "[improve] reusing existing results/improve/hellaswag_{eval,fewshot_pool}.jsonl"
fi

for variant in "${VARIANTS[@]}"; do
    echo "[improve] running variant: $variant"
    uv run python -m improve.infer \
        --variant "$variant" \
        --n-eval "$N_EVAL" \
        --max-concurrency "$MAX_CONCURRENCY" \
        --notes "ablation run via improve/eval.sh" \
        || {
            echo "[improve] variant $variant FAILED (non-fatal, continuing)"
            continue
        }
done

echo "[improve] done. see:"
echo "  - results/result-log.md"
echo "  - docs/improvement-log.md"
echo "  - improve/report.md"
