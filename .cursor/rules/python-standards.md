---
description: "Python coding standards for the LLM Eval System"
globs: "**/*.py"
alwaysApply: true
---

# Python standards

Canonical ruleset: [`AGENTS.md`](../../AGENTS.md). Read it first.

## Code style

- **Python 3.11.** No 3.12+-only syntax.
- **Line length 100.** `ruff format` is the source of truth; never hand-format.
- **Type hints on every public function and class attribute.** Use
  `from __future__ import annotations` at the top of each module.
- **Google-style docstrings** on every public API with `Args`, `Returns`, `Raises`.
- **No wildcard imports.** Group stdlib → third-party → first-party.
- **`pathlib.Path` over `os.path`.**
- **No magic numbers.** Extract to `UPPER_SNAKE` constants or `common/config.py:Settings`.
- **Dataclasses / pydantic models** across module boundaries, never raw `dict`s.

## Configuration

- Read every tunable from `common.config.get_settings()`. No hardcoded URLs, paths,
  model names, or hyperparameters.
- Never read environment variables directly — add the key to `Settings` first.

## Imports

- `common` is first-party. Example ordering:

```python
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import httpx
import structlog

from common.config import get_settings
from common.errors import VLLMClientError
```

## Async

- Use `asyncio` for I/O concurrency.
- No `asyncio.run` inside library code; callers manage the loop.
- Every external I/O call has a timeout.

## Do / don't examples

```python
# DO
from common.config import get_settings
settings = get_settings()
timeout = settings.vllm_timeout_s

# DON'T
timeout = 120  # magic number, hardcoded
```

```python
# DO
from common.errors import VLLMClientError
raise VLLMClientError(f"unexpected status={resp.status_code}")

# DON'T
raise Exception("error")  # bare Exception, no context
```
