"""Shared type definitions used across the project."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict

FinishReason = Literal["stop", "length", "error", "content_filter", "tool_calls"]


class DecodingParams(TypedDict, total=False):
    """OpenAI-compatible decoding parameters for a generation request.

    All keys are optional; callers pass only what they need. Values that are
    ``None`` should be omitted rather than sent as ``null``.
    """

    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    stop: list[str]
    seed: int
    n: int
    logprobs: int


@dataclass(frozen=True, slots=True)
class GenerationResult:
    """Result of a single non-streaming generation call.

    Attributes:
        text: The generated completion text.
        prompt_tokens: Tokens consumed by the prompt.
        completion_tokens: Tokens produced in the completion.
        finish_reason: Why generation stopped (``stop`` / ``length`` / ...).
        logprobs: Optional per-token logprobs aligned with ``text``.
        raw: The raw provider response; kept out of ``repr`` for log hygiene.
    """

    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: FinishReason | str
    logprobs: list[dict[str, float]] | None = None
    raw: dict[str, Any] | None = field(default=None, repr=False)

    @property
    def total_tokens(self) -> int:
        """Return ``prompt_tokens + completion_tokens``."""
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True, slots=True)
class StreamChunk:
    """A single chunk emitted by a streaming generation."""

    delta: str
    finish_reason: str | None = None


@dataclass(frozen=True, slots=True)
class EvalExample:
    """A single evaluation example (benchmark item)."""

    id: str
    prompt: str
    target: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EvalResult:
    """Result of evaluating a single example."""

    example_id: str
    prediction: str
    target: str
    correct: bool
    metadata: dict[str, Any] = field(default_factory=dict)


__all__ = [
    "DecodingParams",
    "EvalExample",
    "EvalResult",
    "FinishReason",
    "GenerationResult",
    "StreamChunk",
]
