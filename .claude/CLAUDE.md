# LLM Eval System — Claude Code guidance

The canonical ruleset lives in [`AGENTS.md`](../AGENTS.md). Claude Code will read this
file automatically and should treat `AGENTS.md` as authoritative.

**Before editing:** read `AGENTS.md` and `PLAN.md`. If an instruction conflicts with a
rule, flag it rather than silently break it.

## TL;DR guardrails

- Python 3.11, line length 100, ruff + mypy.
- Type hints + Google-style docstrings on every public API.
- No bare `Exception` (use `common/errors.py`). No `print` (use `structlog`).
- No hardcoded URLs / models / paths / secrets — go through `common/config.py:Settings`.
- All network I/O has timeouts + retries and goes through `common/vllm_client.py`.
- Tests in `tests/common/test_<module>.py` for every module in `common/`.
- `make all` (lint + typecheck + test) must pass before you stop.
- Conventional Commits. One logical concern per commit.

See `AGENTS.md` for the full 30-rule ruleset and `docs/enterprise-standards.md` for rationale.
