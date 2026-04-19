# Part B — Evaluation

`lm-evaluation-harness` wrapper that routes through the vLLM endpoint.

- `vllm_model.py` — subclasses `lm_eval.api.model.LM`, implements
  `loglikelihood`, `loglikelihood_rolling`, and `generate_until`.
- `run_eval.py` — CLI runner; writes `results/<task>.json` + `results/summary.md`.
- `tasks/custom_task.yaml` + `tasks/custom_data.jsonl` — hand-authored benchmark.

Uses `common.cache.PromptCache` for deterministic reruns. Implemented in Part B.
