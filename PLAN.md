# Implementation Plan — LLM Systems & Evaluation Interview

> Companion to [`docs/problem-statement.md`](docs/problem-statement.md). This plan covers
> how we'll build the project and the enterprise-grade pre-build setup we apply before
> any Part A–E code is written.

**Decisions locked in:**

- **Standards rigor:** Balanced (pragmatic). Full ruff + mypy + pytest + pre-commit + structured
  logging, but no CI or Docker. Coverage reported, not gated.
- **Hardware target:** GPU available. vLLM serves a real open-weight model
  (default: `meta-llama/Llama-3.2-1B-Instruct`). A `MOCK_BACKEND=1` fallback keeps the
  stack runnable without a GPU for dev.
- **AI guidance scope:** Multi-vendor. `AGENTS.md` is canonical; mirrors in `.claude/`,
  `.cursor/rules/`, and `.devin/` (including reusable skills).

---

## Phase 0 — Pre-build setup (no Part code yet)

This is the scaffolding an AI collaborator needs to do useful, safe, consistent work on
this project. Everything in Phase 0 is done **before** Part A.

### 0.1 Repository layout

```
LLMEvalSystem/
├── PLAN.md                          # this file
├── README.md                        # human quickstart
├── AGENTS.md                        # canonical ruleset (AI + human)
├── Makefile                         # single entry point for every operation
├── pyproject.toml                   # deps + ruff + mypy + pytest config
├── uv.lock                          # pinned deps (committed)
├── .python-version                  # 3.11
├── .env.example                     # all configurable keys, no secrets
├── .gitignore
├── .editorconfig
├── .pre-commit-config.yaml
│
├── .devin/
│   ├── config.json                  # enable cursor/claude rule imports
│   └── skills/
│       ├── start-vllm/SKILL.md
│       ├── run-eval/SKILL.md
│       ├── perf-test/SKILL.md
│       ├── improve-bench/SKILL.md
│       └── check-standards/SKILL.md
├── .cursor/rules/
│   ├── python-standards.md          # alwaysApply: true
│   ├── logging-and-errors.md
│   └── testing.md
├── .claude/
│   └── CLAUDE.md                    # mirrors AGENTS.md
│
├── docs/
│   ├── problem-statement.md
│   ├── enterprise-standards.md      # deep-dive + rationale for the rules
│   └── adr/
│       └── 0001-vllm-as-serving-backend.md
│
├── common/                          # shared library used by all parts
│   ├── __init__.py
│   ├── config.py                    # pydantic-settings Settings
│   ├── logging.py                   # structlog config + @timed decorator
│   ├── errors.py                    # custom exception hierarchy
│   ├── types.py                     # shared dataclasses / TypedDicts
│   ├── cache.py                     # SQLite prompt cache
│   ├── vllm_client.py               # async httpx client w/ retries + mock
│   └── stats.py                     # paired-bootstrap CI / p-value helpers
│
├── serve/                           # Part A (stub only in Phase 0)
├── eval_runner/                     # Part B (stub only in Phase 0)
├── perf/                            # Part C (stub only in Phase 0)
├── guardrails/                      # Part D (stub only in Phase 0)
├── improve/                         # Part E (stub only in Phase 0)
│
├── scripts/
│   ├── bootstrap.sh                 # first-time setup
│   └── smoke.sh                     # end-to-end sanity check
└── tests/
    ├── conftest.py
    └── common/                      # one test file per common/ module
```

### 0.2 Python environment

- `uv` managed, Python 3.11 pinned via `.python-version`.
- Dependency tiers in `pyproject.toml`:
  - **Base:** `httpx`, `pydantic`, `pydantic-settings`, `structlog`, `numpy`.
  - **Optional extras** (installed only when needed):
    `serve` (vllm), `eval` (lm-eval, datasets), `perf` (pandas, matplotlib, seaborn, jupyter),
    `improve` (sentence-transformers, scipy).
  - **Dev group:** `ruff`, `mypy`, `pytest`, `pytest-asyncio`, `pytest-cov`, `pre-commit`,
    `detect-secrets`.
- `uv.lock` is committed.

### 0.3 Tooling configuration

| Tool | Setting | Value |
|------|---------|-------|
| ruff | `line-length` | 100 |
| ruff | `target-version` | `py311` |
| ruff | `select` | `E, F, I, N, UP, B, SIM, RUF, PL` |
| ruff format | (replaces black) | enabled |
| mypy | `strict` | **false** |
| mypy | `disallow_untyped_defs` | `true` for `common/` + public module APIs |
| mypy | `ignore_missing_imports` | `true` |
| pytest | `asyncio_mode` | `auto` |
| pytest-cov | gate | **none** (report only) |

### 0.4 Pre-commit hooks

ruff (lint + format), mypy (changed files), `detect-secrets`, `end-of-file-fixer`,
`trailing-whitespace`, `check-added-large-files` (500 KB cap), `check-yaml`, `check-toml`.

### 0.5 Makefile targets

`install`, `serve`, `client-demo`, `eval`, `perf`, `perf-analyze`, `improve`, `lint`,
`format`, `typecheck`, `test`, `test-cov`, `smoke`, `clean`, `all`.

### 0.6 Config management

Single `Settings` class in `common/config.py` built on `pydantic-settings`, reading from
`.env`. **No hardcoded hosts, model names, paths, or hyperparameters anywhere in code.**

### 0.7 AI guidance files

- **`AGENTS.md`** — canonical ruleset (read by Devin, Claude Code, Cursor, Codex).
- **`.claude/CLAUDE.md`** — mirrors AGENTS.md.
- **`.cursor/rules/*.md`** — scoped rules with Cursor frontmatter.
- **`.devin/config.json`** — `{ "read_config_from": { "cursor": true, "claude": true } }`.
- **`.devin/skills/*/SKILL.md`** — 5 reusable workflows:
  - `start-vllm` — launch the server via `make serve`.
  - `run-eval` — subagent that runs `make eval` and summarizes results.
  - `perf-test` — runs the load test and prints key percentiles.
  - `improve-bench` — drives Part E iterations.
  - `check-standards` — runs lint + typecheck + tests, reports failures.

---

## Phase 1 — Enterprise coding standards (content of `AGENTS.md`)

~30 rules, grouped. Every rule is concrete and automatable where possible.

### Code
1. Python 3.11; no 3.12+-only syntax.
2. Line length 100. `ruff format` owns formatting; never hand-format.
3. Type hints on every public function and class attribute.
4. Google-style docstrings on every public API (`Args` / `Returns` / `Raises`).
5. No wildcard imports. Imports grouped stdlib / third-party / first-party.
6. `pathlib.Path` over `os.path`.
7. No magic numbers. Extract to `UPPER_SNAKE` constants or config.
8. Dataclasses / pydantic models across module boundaries, not raw dicts.

### Errors & logging
9. Custom exception hierarchy in `common/errors.py`. Never raise bare `Exception`.
10. Never catch bare `Exception` except at process entry points; always log with context.
11. Structured logging only (`structlog`). No `print` in non-script code.
12. Every external I/O call has a timeout and retry policy (exp backoff, capped, idempotent only).

### Config & secrets
13. No hardcoded URLs / models / paths / hyperparameters. All from `Settings`.
14. No secrets in the repo. `.env` gitignored; `.env.example` lists keys with placeholders.
15. `detect-secrets` pre-commit hook must pass.

### Tests
16. Every `common/` module has a matching `tests/common/test_<module>.py`.
17. Async tests use `pytest-asyncio`; no `asyncio.run` inside tests.
18. Tests are deterministic (seeded, no network unless marked `@pytest.mark.integration`).
19. Coverage reported but not gated. Aim ≥70% on `common/`.

### Concurrency & determinism
20. `asyncio` for I/O concurrency; threads only for CPU-bound pre/post-processing.
21. Determinism-relevant paths set seeds explicitly; residual nondeterminism is documented.

### Dependencies
22. Never edit `pyproject.toml` deps by hand — use `uv add` / `uv remove`. `uv.lock` is committed.
23. Pin direct deps to minor version. Transitive pins come from the lock.

### Git & process
24. Conventional commits (`feat:` / `fix:` / `docs:` / `perf:` / `test:` / `refactor:` / `chore:`).
25. Each PR/commit touches one logical concern.
26. Never commit generated artifacts (`results/raw/`, `.cache/`, notebook outputs).
    Exceptions: `metrics.csv`, `results/summary.md`.
27. No `git push --force` on shared branches.

### Observability & performance
28. Long-running scripts emit a JSON summary at the end (duration, counts, errors).
29. Perf-critical paths have a `@timed` decorator from `common/logging.py`.

### AI-specific
30. AIs must read `AGENTS.md` before editing. If a rule conflicts with an instruction,
    flag it rather than silently break it.

---

## Phase 2 — Part A–E implementation plan

Implementation order: **`common/` (Phase 0) → A → B → C → D → E.**

### Part A — Serving
- `serve/serve.py` — wraps `vllm serve` with flags from `Settings`.
- `serve/client.py` — thin re-export of `common.vllm_client.VLLMClient` with a CLI demo.
- `serve/examples/{demo,concurrent_demo}.py` — streaming demo + concurrency proof.
- Tests with mocked httpx.

### Part B — Evaluation
- `eval_runner/vllm_model.py` — `LM` subclass implementing `loglikelihood`,
  `loglikelihood_rolling`, `generate_until`; reads from `common/cache.py` first.
- `eval_runner/run_eval.py` — CLI runner writing `results/<task>.json` + summary.
- `eval_runner/tasks/custom_task.yaml` + `custom_data.jsonl` — 50 hand-authored examples.

### Part C — Performance & Scaling
- `perf/load_test.py` — asyncio load gen (short+long prompts, configurable concurrency).
- `perf/metrics.py` — post-process → `metrics.csv` with TTFT, TPOT, P50/P95/P99.
- `perf/analysis.ipynb` — plots + 200-word commentary.
- `perf/gpu_monitor.py` — optional `nvidia-smi` sampler.

### Part D — Guardrails & Determinism
- `guardrails/validate.py` — regex/JSON-schema validators + `verify_determinism()`.
- `guardrails/schemas/` — per-task output schemas.
- `guardrails/README.md` — what was tested, where nondeterminism persists.

### Part E — Benchmark Improvement
- Target benchmark: chosen after Part B baseline exists (HellaSwag/MMLU/ARC-C).
- `improve/prepare_data.py`, `improve/optimize_prompt.py`, `improve/infer.py`, `improve/eval.sh`.
- `improve/report.md` (400–700 words): baseline vs improved with **95% CI via paired bootstrap**
  (from `common/stats.py`), ablation table, 10+ before/after examples, cost/latency trade-offs,
  exact seeds + decoding settings.
- Statistical test: paired bootstrap on per-example correctness, 10k resamples, p < 0.05.

---

## Phase 3 — Verification

- [ ] `make install` on a fresh clone succeeds.
- [ ] `make lint && make typecheck && make test` green.
- [ ] `make smoke` (bootstrap → serve → one generation → one eval → one perf request) green.
- [ ] `make eval` produces `results/{mmlu,hellaswag,custom}.json` + `results/summary.md`.
- [ ] `make perf` produces `metrics.csv` with no NaNs.
- [ ] `make improve` produces `improve/report.md` with CI numbers filled in.
- [ ] `guardrails/validate.py` demonstrates identical output across 5 runs of the same prompt.
- [ ] `detect-secrets`, ruff, mypy all green.

---

## Risks / trade-offs

- **vLLM GPU requirement.** If the GPU turns out unavailable, `MOCK_BACKEND=1` keeps all
  eval / perf / improve code runnable against a deterministic mock.
- **Balanced ≠ strict.** Coverage and mypy strictness are not gated. Flipping either on
  later is a one-line change.
- **4-hour scope vs scaffold depth.** Phase 0 is upfront investment; it pays back by making
  A–E small, focused increments. If time is tight we trim Devin skills and the ADR.
- **Multi-vendor AI configs** risk drift between `AGENTS.md`, `CLAUDE.md`, `.cursor/rules/`.
  Mitigation: `AGENTS.md` is source; others mirror or get regenerated by a sync script.
- **Part E target benchmark** is not locked. Decided after Part B produces a baseline.
