# Part E — Benchmark Improvement

Raise HellaSwag accuracy without touching model weights, with real
statistical significance and end-to-end reproducibility.

## What's here

| File | Purpose |
|---|---|
| `prepare_data.py` | Download HellaSwag validation split via `datasets`, shuffle on seed, split into an eval subset and a disjoint few-shot pool. |
| `optimize_prompt.py` | Prompt variants (`baseline`, `clean`, `fewshot_random`, `fewshot_semantic`) + scoring modes (`sum`, `length_norm`, `byte_norm`) + `SemanticRetriever` over `sentence-transformers/all-MiniLM-L6-v2`. |
| `infer.py` | The eval loop. Scores each option via `VLLMClient.generate(echo=True, logprobs=5, max_tokens=0)`, picks argmax under the variant's scoring mode, computes paired-bootstrap CI vs baseline, and appends to `results/result-log.*` and `docs/improvement-log.md`. |
| `eval.sh` | Sequential orchestrator — prepares data + runs every variant. |
| `report.md` | Final Part E write-up: methodology, results table, ablation interpretation, 10+ before/after examples, cost/latency tradeoffs, exact config. |

## Running the full ablation

```bash
# Offline (mock backend, no GPU required):
bash improve/eval.sh

# Real run (requires `make serve` running + `uv sync --extra eval --extra improve`):
LLMEVAL_MOCK_BACKEND=false N_EVAL=500 bash improve/eval.sh
```

This populates:

- `results/improve/<variant>.json` per variant (per-example correctness + scores)
- `results/result-log.csv` / `.md` — one row per variant
- `docs/improvement-log.md` — one entry per variant with hypothesis, CI, p-value, and decision

## Statistical test

Paired bootstrap on per-example correctness (baseline vs variant), 10,000
resamples, 95% CI from the 2.5/97.5 percentiles, two-sided p-value from
the proportion of resamples with the opposite sign (doubled, capped).
Math lives in `common.stats.paired_bootstrap`.

Significance rule: 95% CI strictly excludes zero. A variant with `ci_low > 0`
is significantly better; `ci_high < 0` is significantly worse; otherwise not
significant.

## Tests

`tests/improve/`:

- `test_optimize_prompt.py` — prompt variants produce the expected
  `(prompt, continuation)` pairs; scoring modes (`sum` / `length_norm` /
  `byte_norm`) compute the right scalar for a known token-logprob vector;
  `_as_continuation` guarantees a single-space prefix regardless of input.
- `test_infer.py` — end-to-end `evaluate()` against the mock backend for
  baseline, `clean_length_norm`, and `fewshot_random_5`; full CLI `main()`
  run through baseline → variant including result-log assertions.

## See also

- [`report.md`](report.md) — the report itself, with numbers, CIs, and the
  ablation interpretation.
- [`../docs/improvement-log.md`](../docs/improvement-log.md) — chronological
  journal of every run.
- [`../results/result-log.md`](../results/result-log.md) — result table.
