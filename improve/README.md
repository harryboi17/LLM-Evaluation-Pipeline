# Part E — Benchmark Improvement

Inference-time improvement of a single benchmark (HellaSwag / MMLU / ARC-Challenge).

- `prepare_data.py` — dev/test splits + example embeddings for few-shot selection.
- `optimize_prompt.py` — template rewriting + semantic few-shot selection.
- `infer.py` — runs inference with template + self-consistency + regex normalization.
- `eval.sh` — orchestrates baseline + improved runs.
- `report.md` — 400–700 words with CI, ablation, examples, cost/latency trade-offs.

Statistical significance via `common.stats.paired_bootstrap` (p < 0.05).

Implemented in Part E.
