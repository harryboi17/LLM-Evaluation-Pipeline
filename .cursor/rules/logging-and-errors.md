---
description: "Logging and error-handling rules"
globs: "**/*.py"
alwaysApply: true
---

# Logging and errors

## Logging

- **`structlog` only.** No `print` in non-script code. Script entry points under
  `scripts/` may `print` for terminal output.
- Use `from common.logging import get_logger` and `log = get_logger(__name__)` at module
  top. Never call `structlog.get_logger()` directly.
- Bind context keys at the start of a logical operation:

```python
log = get_logger(__name__).bind(run_id=run_id, task=task, model=settings.model_name)
log.info("eval_started", n_examples=len(examples))
```

- Log events are **snake_case nouns / verbs**: `eval_started`, `vllm_retry`,
  `cache_hit`. Do not put dynamic values in the event name.
- Long-running scripts emit a JSON summary at the end (duration, counts, errors).

## Errors

- Use the hierarchy in `common/errors.py`:
  - `LLMEvalError` — base.
  - `ServeError` / `VLLMClientError` / `VLLMTimeoutError` — serving layer.
  - `EvalError` — evaluation layer.
  - `CacheError` — prompt cache.
  - `GuardrailError` / `ValidationError` / `DeterminismError` — guardrails.
- Never raise bare `Exception`. Never catch bare `Exception` except at process entry points,
  and always log with `exc_info=True`.
- Chain exceptions: `raise VLLMClientError(...) from exc`.

## Timeouts & retries

- Every external I/O call has a timeout and retry policy.
  - Exponential backoff, capped retries (from `Settings.vllm_max_retries`).
  - Retry only on transport errors (`httpx.TimeoutException`, `httpx.HTTPError`)
    and 5xx. **Never retry 4xx.**
  - Operation must be idempotent.

## Do / don't examples

```python
# DO
try:
    resp = await client.post(url, json=body)
except httpx.TimeoutException as exc:
    log.warning("vllm_timeout", url=url, exc_info=True)
    raise VLLMTimeoutError(f"timeout on {url}") from exc

# DON'T
try:
    resp = requests.post(url, json=body)  # no timeout, sync, no retry
except Exception as e:
    print(f"error: {e}")  # print, bare Exception, silent swallow
```
