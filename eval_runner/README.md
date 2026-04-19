# Part B — Evaluation

Custom lm-evaluation-harness model wrapper that routes all calls through our vLLM server, with prompt-level caching and a hand-authored custom task.

## What's here

| File | Purpose |
|---|---|
| `vllm_model.py` | `LM` subclass implementing `loglikelihood`, `loglikelihood_rolling`, and `generate_until`. Fan-out via `asyncio.gather` + bounded `Semaphore`; every request memoised in `common.cache.PromptCache`. |
| `run_eval.py` | CLI runner calling `lm_eval.simple_evaluate`. Writes `results/<task>.json`, `results/run_meta.json`, and `results/summary.md`. |
| `tasks/custom_task.yaml` | Custom task registered with the harness (`output_type: generate_until`, metric: exact-match, normalised for case/whitespace). |
| `tasks/custom_data.jsonl` | 50 hand-authored short-answer examples covering general knowledge, arithmetic, science, programming, and simple reasoning. |

## Quickstart

```bash
# Offline (mock backend, no GPU):
LLMEVAL_MOCK_BACKEND=true make eval

# Run all three tasks (MMLU-HS-CS subset + HellaSwag + custom):
make eval

# One task with a small cap:
uv run python -m eval_runner.run_eval --task custom_qa --limit 10

# Custom limits, few-shot, concurrency:
uv run python -m eval_runner.run_eval \
    --task mmlu_high_school_computer_science,hellaswag \
    --limit 100 \
    --num-fewshot 5 \
    --max-concurrency 16
```

## How loglikelihood is computed

For each `(context, continuation)` pair:

1. Tokenise `context` → `n_ctx`; tokenise `context + continuation` → `n_full`.
2. `n_cont = n_full - n_ctx`.
3. Send `prompt = context + continuation` to vLLM with `echo=True`, `max_tokens=0`, `logprobs=5`.
4. The server returns token-level log-probabilities including for echoed prompt tokens.
5. Sum `token_logprobs[-n_cont:]` → continuation log-likelihood.
6. `is_greedy` is true iff every continuation token is the argmax in its position's top-K.

In `LLMEVAL_MOCK_BACKEND=true` mode the tokenizer falls back to whitespace splitting (matching the mock generator), so the same code path exercises end-to-end without an HF tokenizer load.

## Caching

Every (prompt, decoding-params) pair is keyed via SHA-256 in a SQLite file under `LLMEVAL_CACHE_DIR`. That means:

- Re-running `make eval` with identical inputs is ~free (no tokens paid twice).
- Part E ablations that only change prompt templates hit the cache for the unchanged portion of the run.
- Determinism holds across restarts (same prompt + same params → byte-identical response from cache).

## Outputs

After a run, `LLMEVAL_RESULTS_DIR` contains:

```
results/
├── <task>.json         # raw harness metrics per task
├── run_meta.json       # model, seed, limit, duration, tasks
└── summary.md          # compact Markdown table across tasks
```

## Tests

`tests/eval_runner/` (all run against the mock backend; no network, no GPU):

- `test_vllm_model.py` — end-to-end coverage of the three LM methods plus focused unit tests of `_score_continuation` logic.
- `test_run_eval.py` — stubs `simple_evaluate` and verifies the CLI writes the expected output files and meta JSON.
- `test_custom_data.py` — asserts the JSONL has exactly 50 valid, short-answer examples (contract for the custom task).

Tests that need `lm-eval` installed skip gracefully via `pytest.importorskip` when only the base dev group is present.
