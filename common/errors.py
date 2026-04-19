"""Custom exception hierarchy for the LLM Evaluation System.

All project-specific exceptions inherit from :class:`LLMEvalError`. Catch this base
class at process entry points to centralize error handling; never catch bare
``Exception``.

Each domain has its own subclass so callers can pick the granularity they need:

- Serving: :class:`ServeError`, :class:`VLLMClientError`, :class:`VLLMTimeoutError`
- Evaluation: :class:`EvalError`
- Config: :class:`ConfigError`
- Cache: :class:`CacheError`
- Guardrails: :class:`GuardrailError`, :class:`ValidationError`, :class:`DeterminismError`
"""

from __future__ import annotations


class LLMEvalError(Exception):
    """Base exception for every error raised by this project."""


class ConfigError(LLMEvalError):
    """Raised when configuration is missing or invalid."""


class ServeError(LLMEvalError):
    """Raised by the serving layer (Part A)."""


class VLLMClientError(ServeError):
    """Raised when the vLLM client cannot complete a request."""


class VLLMTimeoutError(VLLMClientError):
    """Raised when a vLLM request exceeds its timeout."""


class EvalError(LLMEvalError):
    """Raised by the evaluation layer (Part B)."""


class CacheError(LLMEvalError):
    """Raised by the prompt cache when a stored entry is corrupt or unwritable."""


class GuardrailError(LLMEvalError):
    """Raised by the guardrails layer (Part D)."""


class ValidationError(GuardrailError):
    """Raised when an output fails schema or regex validation."""


class DeterminismError(GuardrailError):
    """Raised when a determinism check detects variance."""


__all__ = [
    "CacheError",
    "ConfigError",
    "DeterminismError",
    "EvalError",
    "GuardrailError",
    "LLMEvalError",
    "ServeError",
    "VLLMClientError",
    "VLLMTimeoutError",
    "ValidationError",
]
