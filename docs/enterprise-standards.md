# Enterprise coding standards — rationale

This is the companion to [`AGENTS.md`](../AGENTS.md). `AGENTS.md` lists the rules; this
document explains **why** each rule exists and what problem it prevents. When a rule
feels like it's getting in the way, read the "why" here before asking to relax it.

---

## 1. Code style

### Rule 2 — Line length 100, ruff format owns formatting

Hand-formatting wastes review cycles and produces bikeshedding. A deterministic formatter
removes the argument entirely. 100 cols is a pragmatic compromise: wide enough for modern
type hints and dataclass fields, narrow enough for side-by-side diffs.

### Rule 3 — Type hints on every public function

Without types, refactoring is archaeology. With types + mypy, a rename across 30 files
takes 30 seconds. We only require hints on *public* APIs to keep local helpers terse.

### Rule 4 — Google-style docstrings on every public API

`Args` / `Returns` / `Raises` is enough for IDE hover, auto-generated docs, and for an AI
collaborator to understand a function's contract without reading its body.

### Rule 7 — No magic numbers

A hardcoded `timeout=120` in one file and `TIMEOUT = 60` in another is a bug in waiting.
Hoist every tunable to `common.config.Settings`; a reviewer can then verify "these are all
the knobs" without grepping.

### Rule 8 — Dataclasses / pydantic over raw dicts across module boundaries

Raw dicts have no schema. A dataclass says "these are the fields, this is the shape" and
mypy enforces it. Breaking changes become compile-time errors instead of prod stack traces.

---

## 2. Errors & logging

### Rule 9-10 — Custom exception hierarchy

A caller who wants to tolerate serving errors should be able to write
`except ServeError` without also catching unrelated `ValidationError`s. A flat
`raise Exception` kills that option forever. The hierarchy takes one file and pays for
itself the first time you need to handle "retry vs abort".

### Rule 11 — Structured logging only

`print` goes to stdout, mixes with generated text, and loses context. `structlog` events
are key-value pairs you can grep, aggregate, and pipe into observability tools. JSON
format also means `jq` and a single command extract any metric from a run.

### Rule 12 — Every I/O has timeout + retry

No timeout = a hung test or a hung prod worker. No retry = one blip equals one failed
benchmark run. Retries are scoped to transport errors and 5xx; we never retry 4xx because
4xx is our bug, not a transient fault.

---

## 3. Config & secrets

### Rule 13 — All tunables via Settings

Testing config changes becomes trivial: set an env var, rerun. No code edit, no recompile,
no rebuild. This is also how "deterministic mode" flows through the stack — flip
`LLMEVAL_SEED`, everything downstream sees it.

### Rule 14-15 — No secrets in the repo, `detect-secrets` in pre-commit

A leaked secret is a very bad day. Even in a demo repo, habits matter — professionals
don't commit keys. `detect-secrets` is a cheap safety net; false positives get added to the
baseline rather than silenced per-line.

---

## 4. Tests

### Rule 16 — A test file per module

If a module is worth writing, it's worth a smoke test. This also prevents the classic
failure mode "we have 10k LOC and no tests for the critical 500 LOC".

### Rule 17-18 — Async tests, no network, no wall-clock

Flaky tests erode trust faster than anything else. A test that hangs, or passes Monday and
fails Tuesday, will eventually be ignored. Determinism is a hard gate.

### Rule 19 — Coverage reported, not gated

Coverage gates produce "test the setter" antipatterns. Reporting keeps awareness high
without forcing bad tests. For Part E-like research code, a gate would be actively
counterproductive.

---

## 5. Concurrency & determinism

### Rule 20 — asyncio for I/O, threads for CPU

Mixing asyncio and threads in a single code path produces deadlocks and starvation.
Pick one per operation. vLLM calls are I/O → asyncio. Tokenizer pre-processing is CPU →
a process pool, not threads (because of the GIL).

### Rule 21 — Seed + documented residual nondeterminism

Pretending determinism is fully achievable on GPU is a lie — CUDA nondeterministic
reductions exist. We set every seed we control, document what remains, and choose
evaluations (like `loglikelihood`) that tolerate it.

---

## 6. Dependencies

### Rule 22-23 — uv managed, pinned minor versions, lock committed

The lock file is the definition of "reproducible". Hand-edited `pyproject.toml` drifts.
`uv add` updates both files atomically. Minor-version pins balance "don't break on a patch
release" with "don't silently consume a major version bump".

---

## 7. Git & process

### Rule 24 — Conventional Commits

`git log --oneline` is much more useful when each line starts with `feat|fix|perf|...`.
It also drives changelog generation and semantic versioning if we ever need them.

### Rule 25 — One logical concern per commit

Bisect only works if each commit is a self-contained change. Mixing a refactor + a bug
fix in one commit hides the fix from `git bisect` and makes revert risky.

### Rule 26 — No generated artifacts committed

Notebooks with outputs produce gigantic diffs and leak accidental secrets (API responses).
`.cache/` is machine-local. `results/raw/` is disposable. `results/summary.md` is the
human-readable summary and *is* committed.

---

## 8. Observability

### Rule 28 — JSON summary at end of long runs

Parsing a wall of structured log events is fine; parsing a wall of `print` is not. A final
`summary = {...}; log.info("summary", **summary)` line makes every run self-describing.

### Rule 29 — @timed on perf-critical paths

You cannot optimize what you don't measure. The decorator is one line, logs are searchable,
and regressions show up in `grep "elapsed_ms"` across runs.

---

## 9. AI-specific

### Rule 30 — AIs must read AGENTS.md

When an AI silently breaks a rule, the human reviewer has to find the break *and* explain
the rule *again*. Reading the file first is cheap. Flagging a conflict is always better
than silently working around it.

---

## Where these rules are enforced

| Rule | Enforcement |
|------|-------------|
| 2 (line length) | `ruff format` + pre-commit hook |
| 3 (type hints) | `mypy` on `common/` and Part modules |
| 4 (docstrings) | **Manual review** (no ruff rule is worth the noise) |
| 5 (imports) | `ruff` `I` rules |
| 7 (magic numbers) | `pylint` `PLR2004` (relaxed in tests) |
| 9, 10 (exceptions) | `ruff` `B` rules + review |
| 11 (logging) | Grep for `print(` in non-script code during review |
| 13, 14 (config, secrets) | `detect-secrets` + review |
| 16 (test per module) | Review checklist |
| 18 (no network in tests) | `pytest.mark.integration` marker required |
| 22, 23 (deps) | Review of `pyproject.toml` changes |
| 24 (conventional commits) | Manual review, can be linted with `commitlint` later |
| 26 (no artifacts) | `.gitignore` + pre-commit `check-added-large-files` |
| 28 (JSON summary) | Review checklist |
