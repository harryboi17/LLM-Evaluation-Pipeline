# Project Rules — LLM Eval System

These are always-on rules for any contributor (human or AI) working on this repo.
They are the **canonical source**; `.claude/CLAUDE.md` and `.cursor/rules/*.md` mirror
subsets of this file. Rationale and examples live in [`docs/enterprise-standards.md`](docs/enterprise-standards.md).

**Before editing anything, read this file.** If an instruction conflicts with a rule here,
flag it rather than silently break it.

---

## 1. Code

1. Python **3.11**. No 3.12+-only syntax (e.g., PEP 695 `type` statements).
2. **Line length 100**. `ruff format` owns formatting — never hand-format.
3. **Type hints on every public function and class attribute.** Private helpers may omit them
   when the types are obvious from one line above.
4. **Google-style docstrings on every public API** (module, class, function). Include
   `Args`, `Returns`, `Raises` where applicable.
5. **No wildcard imports.** Imports grouped stdlib → third-party → first-party, separated
   by a blank line (ruff/isort enforces this).
6. **`pathlib.Path` over `os.path`.** Convert to `str` only at I/O boundaries that require it.
7. **No magic numbers** in logic code. Extract to `UPPER_SNAKE` module constants or to
   `common.config.Settings`.
8. **Dataclasses or pydantic models** across module boundaries. Do not pass raw dicts of
   arbitrary shape between packages.

## 2. Errors & logging

9. **Custom exception hierarchy** in `common/errors.py` (`LLMEvalError` → domain errors).
   Never raise bare `Exception`.
10. **Never catch bare `Exception`** except at process entry points, and always log with
    full context (including the exception via `log.exception(...)` or `structlog`'s
    `exc_info=True`).
11. **Structured logging only** via `structlog`. No `print` in non-script code. Bind
    context keys (`run_id`, `model`, `task`) at the start of a logical operation.
12. **Every external I/O call has a timeout and a retry policy.** Exponential backoff,
    capped retries, idempotent operations only. 4xx responses are never retried.

## 3. Config & secrets

13. **No hardcoded URLs, model names, paths, or hyperparameters.** Everything goes through
    `common.config.Settings` (env-driven via `pydantic-settings`).
14. **No secrets in the repo.** `.env` is gitignored; `.env.example` lists every key with
    placeholder values.
15. **`detect-secrets` pre-commit hook must pass.** If it flags a false positive, update
    `.secrets.baseline`; don't suppress the hook.

## 4. Tests

16. **Every module in `common/` has a matching `tests/common/test_<module>.py`.**
17. **Async tests use `pytest-asyncio`.** No `asyncio.run` inside tests; let the plugin
    handle the loop.
18. **Tests are deterministic** — seeded RNG, no network calls unless marked
    `@pytest.mark.integration`, no reliance on wall-clock timing.
19. Coverage is **reported but not gated**. Aim for ≥70% on `common/`.

## 5. Concurrency & determinism

20. Use **`asyncio` for I/O concurrency**. Use threads only for CPU-bound pre/post-processing.
    Do not mix the two in a single operation.
21. **Determinism-relevant paths set seeds explicitly** (`Settings.seed`). Document any
    residual nondeterminism (e.g., CUDA nondeterministic kernels, fp16 reduction order)
    rather than pretending it doesn't exist.

## 6. Dependencies

22. **Never edit `pyproject.toml` dependencies by hand.** Use `uv add <pkg>` / `uv remove <pkg>`.
    `uv.lock` is committed.
23. **Pin direct deps to minor version** (e.g., `vllm>=0.6,<1.0`). Transitive pins come from
    the lock file.

## 7. Git & process

24. **Conventional Commits**: `feat:`, `fix:`, `docs:`, `perf:`, `test:`, `refactor:`, `chore:`.
    First line ≤72 chars, imperative mood.
25. **Each PR/commit touches one logical concern.** No drive-by reformatting in feature commits.
26. **Never commit generated artifacts** — `results/raw/`, `.cache/`, notebook outputs.
    Exceptions (kept under version control): `metrics.csv`, `results/summary.md`,
    `results/<task>.json`.
27. **No `git push --force`** on shared branches.

## 8. Observability & performance

28. **Long-running scripts emit a JSON summary at the end** (duration, counts, errors). Makes
    log grepping and CI parsing trivial.
29. **Perf-critical paths have a `@timed` decorator** from `common/logging.py`.

## 9. AI-specific

30. **AIs must read this file before editing.** If a rule conflicts with a user instruction,
    flag it and ask — don't silently break the rule.

---

## Commands to know

| Command | Purpose |
|---------|---------|
| `make install` | Install base + dev deps |
| `make lint` | Ruff lint |
| `make format` | Ruff format + auto-fix |
| `make typecheck` | Mypy |
| `make test` | Pytest |
| `make all` | Lint + typecheck + test |
| `make serve` | Launch vLLM server |
| `make eval` | Run the benchmark suite |
| `make perf` | Load test → metrics.csv |
| `make improve` | Part E improvement pipeline |
| `make smoke` | End-to-end sanity |
| `make clean` | Drop caches |

## Key files

- `PLAN.md` — implementation plan.
- `common/config.py` — the one place hyperparameters live.
- `common/vllm_client.py` — every network call goes through this.
- `common/errors.py` — the exception hierarchy.
- `docs/enterprise-standards.md` — the rationale behind each rule.
