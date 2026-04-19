---
description: "Testing conventions for pytest + pytest-asyncio"
globs: "tests/**/*.py"
alwaysApply: true
---

# Testing

## Layout

- Mirror source layout: `common/cache.py` → `tests/common/test_cache.py`.
- One test module per source module.
- Shared fixtures live in `tests/conftest.py`.

## Conventions

- **Deterministic.** Seed every RNG; no wall-clock dependencies; no network.
- Tests that require a running vLLM server are marked `@pytest.mark.integration`
  and skipped by default in the CI-equivalent `make test` target.
- **`pytest-asyncio`** handles the loop. Test functions are `async def` where
  needed; no `asyncio.run`.
- Use `monkeypatch` / `tmp_path` fixtures for isolation.
- Mock external HTTP with `httpx.MockTransport`, not by monkey-patching `httpx.post`.

## Assertions

- Prefer exact assertions over `assert x`. Examples:

```python
assert result.prompt_tokens == 12
assert result.finish_reason == "stop"
```

## Do / don't examples

```python
# DO
import pytest
from common.stats import paired_bootstrap

def test_paired_bootstrap_detects_uniform_improvement():
    baseline = [0] * 100
    improved = [1] * 100
    result = paired_bootstrap(baseline, improved, resamples=1000, seed=0)
    assert result.diff_acc == 1.0
    assert result.ci_low > 0.9
    assert result.p_value < 0.05
```

```python
# DON'T
def test_something():
    import time
    time.sleep(0.1)  # wall-clock dependency
    assert 1 == 1    # trivial, no signal
```
