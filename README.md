# LLM Eval System

End-to-end LLM evaluation pipeline built on
[vLLM](https://github.com/vllm-project/vllm) and
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
Each part is small, self-contained, and honest about what it measures.

| Part | Scope | Dir |
|---|---|---|
| A | vLLM server launcher + async Python client (streaming, concurrency demo) | [`serve/`](serve/) |
| B | Harness-compatible model wrapper, two official tasks + one custom (`custom_qa`), prompt cache | [`eval_runner/`](eval_runner/) |
| C | Async load generator, percentile metrics, optional `nvidia-smi` sampler, analysis notebook | [`perf/`](perf/) |
| D | JSON-schema / regex output validation, determinism verification | [`guardrails/`](guardrails/) |
| E | HellaSwag ablation with paired-bootstrap CIs, improvement + result logs | [`improve/`](improve/) |

## Why this exists

This repo is the reference implementation for the
[LLM Systems & Evaluation problem statement](docs/problem-statement.md).
The design goals are, in order:

1. **Correctness** — the code does what it says (guardrails, determinism,
   paired-bootstrap CIs, cached scoring).
2. **Reproducibility** — every run writes to `results/result-log.*` and
   `docs/improvement-log.md` with commit SHA, seed, and config, so past
   numbers can be re-derived from the repo alone.
3. **Legibility** — every module has a README that says what's in it and
   why, and every non-trivial choice has an ADR or a paragraph in
   [`docs/enterprise-standards.md`](docs/enterprise-standards.md).
4. **No hidden state** — no hardcoded URLs / models / paths / hyperparams;
   everything flows through `common.config.Settings` (pydantic-settings
   reading `LLMEVAL_*` env vars).

## Quickstart

```bash
# First-time setup (Python 3.11 + uv + optional extras)
make bootstrap

# The fast loop, against the mock backend (no GPU needed):
make lint typecheck test
bash scripts/smoke.sh

# Real run on a GPU machine:
uv sync --extra serve --extra eval --extra perf --extra improve
make serve         # in one terminal
make eval          # full lm-eval on MMLU-HS-CS + HellaSwag + custom
make perf          # load-test the server
bash improve/eval.sh  # Part E ablation ladder
```

Every Makefile target is documented — `make help` prints the list.

## Logs

Two central, human-readable logs are updated automatically by every
eval / improve run:

- **`results/result-log.md`** (+ authoritative `result-log.csv`) — one row
  per run with task, method, accuracy, delta vs baseline, 95% CI, p-value,
  wall time, commit SHA, seed.
- **`docs/improvement-log.md`** — chronological journal of Part E changes
  with hypothesis, exact config, result, significance verdict, decision.

These make the before/after story trivially skimmable after the fact and
are the primary artefacts (alongside `improve/report.md`) of the Part E
work.

## Configuration

Every tunable is an `LLMEVAL_`-prefixed env var (see [`.env.example`](.env.example)):

- Model + vLLM: `LLMEVAL_MODEL_NAME`, `LLMEVAL_VLLM_HOST`, `LLMEVAL_VLLM_PORT`,
  `LLMEVAL_VLLM_MAX_MODEL_LEN`, `LLMEVAL_VLLM_DTYPE`,
  `LLMEVAL_VLLM_GPU_MEMORY_UTILIZATION`, `LLMEVAL_VLLM_TENSOR_PARALLEL_SIZE`, …
- Paths: `LLMEVAL_CACHE_DIR`, `LLMEVAL_RESULTS_DIR`
- Determinism: `LLMEVAL_SEED`
- Logging: `LLMEVAL_LOG_LEVEL`, `LLMEVAL_LOG_FORMAT` (`console` / `json`)
- Offline dev: `LLMEVAL_MOCK_BACKEND=true` — `VLLMClient` returns canned
  responses and `serve.serve` exits cleanly without launching vLLM, so the
  whole pipeline is runnable without a GPU.

## Repo structure

```
common/         shared library (config, logging, cache, vllm client, stats, result log)
serve/          Part A — vLLM server + async client + demos
eval_runner/    Part B — harness wrapper + custom task (50 examples) + cached runner
perf/           Part C — load gen + metrics + gpu sampler + analysis.ipynb
guardrails/     Part D — schemas + validate.py + verify_determinism
improve/        Part E — prompt variants + retriever + paired-bootstrap ablation
docs/           problem statement, enterprise standards, ADRs, improvement log
results/        result-log.csv + result-log.md + per-run artefacts
scripts/        bootstrap.sh, smoke.sh
tests/          mirrors source tree; 142 tests as of this commit
```

## Standards

Balanced-strict setup:

- `ruff` for lint + format
- `mypy` with `disallow_untyped_defs=true` on `common/` and every Part
- `pytest` + `pytest-asyncio`; coverage reported (not gated)
- `pre-commit` (ruff + mypy + `detect-secrets`)
- `structlog` for structured logging (`timed` decorator around perf-critical paths)
- Custom exception hierarchy in `common/errors.py`; no bare `Exception`
- Conventional commits

The full 30-rule ruleset is in [`AGENTS.md`](AGENTS.md) and is read
automatically by Devin, Claude, Cursor, and Codex.

## Phase 3 verification checklist

From [`PLAN.md`](PLAN.md); all checks run in this environment via
`bash scripts/smoke.sh`.

- [x] `make install` on a fresh clone succeeds.
- [x] `make lint && make typecheck && make test` green.
- [x] `make smoke` (bootstrap → mock inference → one CLI + demo + eval) green.
- [x] `python -m eval_runner.run_eval --task custom_qa` produces
  `results/custom_qa.json` + `results/summary.md` + one row in
  `result-log.csv`.
- [x] `python -m perf.load_test` produces a well-formed `metrics.csv` and
  `perf.metrics` summarises it with no NaNs.
- [x] `bash improve/eval.sh` produces `improve/report.md` with per-variant
  numbers, CIs, and one row per variant in `results/result-log.csv` and
  `docs/improvement-log.md`.
- [x] `python -m guardrails.validate determinism "prompt"` reports
  `identical=true` on the mock backend with `temperature=0`.
- [x] `detect-secrets`, `ruff`, `mypy` all green.

## Test coverage

As of the latest commit:

```
142 tests pass in ~5s
common/ line coverage: 83.7%
(common/vllm_client 61% — the bulk of the uncovered branches are live HTTP
paths; tests use httpx mock transports and the mock backend.)
```

## License

MIT.
