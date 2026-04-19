---
name: check-standards
description: Enforce the project's enterprise coding standards
allowed-tools:
  - read
  - exec
permissions:
  allow:
    - Exec(make lint)
    - Exec(make typecheck)
    - Exec(make test)
    - Exec(make all)
    - Exec(uv run pre-commit run --all-files)
---

Verify the repo satisfies the rules in `AGENTS.md`.

1. Run `make lint` — ruff check. Report any findings.
2. Run `make typecheck` — mypy on `common/ serve/ eval_runner/ guardrails/ improve/`.
3. Run `make test` — pytest with coverage report.
4. Run `uv run pre-commit run --all-files` — catches `detect-secrets`, whitespace,
   large files, etc.
5. Produce a markdown checklist:
   - [ ] ruff clean
   - [ ] mypy clean
   - [ ] tests passing
   - [ ] pre-commit clean
   - [ ] coverage on `common/` ≥ 70%
6. If anything fails, list the top 5 failures by file path and first error message.

Do NOT attempt to fix failures automatically unless the user asks — some findings need
judgment (e.g., refactoring vs. `# noqa`).
