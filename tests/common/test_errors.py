"""Tests for ``common.errors``."""

from __future__ import annotations

import pytest

from common.errors import (
    CacheError,
    ConfigError,
    DeterminismError,
    EvalError,
    GuardrailError,
    LLMEvalError,
    ServeError,
    ValidationError,
    VLLMClientError,
    VLLMTimeoutError,
)


@pytest.mark.parametrize(
    "cls",
    [
        ConfigError,
        ServeError,
        VLLMClientError,
        VLLMTimeoutError,
        EvalError,
        CacheError,
        GuardrailError,
        ValidationError,
        DeterminismError,
    ],
)
def test_all_errors_inherit_from_base(cls: type[Exception]) -> None:
    assert issubclass(cls, LLMEvalError)


def test_vllm_timeout_is_a_vllm_client_error() -> None:
    assert issubclass(VLLMTimeoutError, VLLMClientError)
    assert issubclass(VLLMClientError, ServeError)


def test_validation_and_determinism_are_guardrails() -> None:
    assert issubclass(ValidationError, GuardrailError)
    assert issubclass(DeterminismError, GuardrailError)


def test_exception_chaining_preserves_cause() -> None:
    cause = ValueError("boom")
    try:
        try:
            raise cause
        except ValueError as exc:
            raise VLLMClientError("wrapped") from exc
    except VLLMClientError as wrapped:
        assert wrapped.__cause__ is cause
        assert str(wrapped) == "wrapped"
