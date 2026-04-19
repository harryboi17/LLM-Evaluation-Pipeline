# LLM Eval System

A scaled-down, production-shaped LLM evaluation pipeline built on
[vLLM](https://github.com/vllm-project/vllm) and
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).

- **Serving** — vLLM with continuous batching and paged attention (Part A)
- **Evaluation** — harness wrapper + MMLU + HellaSwag + a custom JSON benchmark (Part B)
- **Performance** — async load generator with TTFT / TPOT / p95 latency (Part C)
- **Guardrails** — deterministic-mode verification + output validation (Part D)
- **Improvement** — inference-time benchmark lift with statistical significance (Part E)

## Quickstart

```bash
# One-time setup
make bootstrap

# Dev loop
make lint typecheck test

# Serve a model (needs a GPU)
make serve

# In another terminal: run evals
make eval

# Load test the running server
make perf

# Improve a benchmark at inference time
make improve
```

## Repo structure

See [`PLAN.md`](PLAN.md) for the full implementation plan and
[`AGENTS.md`](AGENTS.md) for the coding standards every contributor (human or AI) follows.

```
common/        # shared library (config, logging, cache, vllm client, stats)
serve/         # Part A — vLLM server + client
eval_runner/   # Part B — harness wrapper + custom task
perf/          # Part C — load test + analysis notebook
guardrails/    # Part D — determinism + output validation
improve/       # Part E — inference-time benchmark improvement
docs/          # problem statement, enterprise standards, ADRs
```

## Configuration

All tunables go through `common.config.Settings` (pydantic-settings), which reads from
`.env`. Copy `.env.example` to `.env` and edit — never commit `.env`.

## Standards

We use a **balanced** enterprise-grade setup:

- `ruff` for lint + format
- `mypy` for type-checking (non-strict outside `common/`)
- `pytest` + `pytest-asyncio` with coverage reported (not gated)
- `pre-commit` (ruff + mypy + `detect-secrets`)
- `structlog` for structured logging
- `pydantic-settings` for config, no hardcoded values

Details and rationale in [`docs/enterprise-standards.md`](docs/enterprise-standards.md).

## License

MIT.
