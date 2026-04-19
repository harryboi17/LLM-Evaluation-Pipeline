"""Tests for ``common.types``."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from common.types import EvalExample, EvalResult, GenerationResult, StreamChunk


def test_generation_result_total_tokens() -> None:
    gr = GenerationResult(text="hi", prompt_tokens=3, completion_tokens=4, finish_reason="stop")
    assert gr.total_tokens == 7


def test_generation_result_is_frozen() -> None:
    gr = GenerationResult(text="hi", prompt_tokens=1, completion_tokens=1, finish_reason="stop")
    with pytest.raises(FrozenInstanceError):
        gr.text = "bye"  # type: ignore[misc]


def test_stream_chunk_minimal() -> None:
    c = StreamChunk(delta="token")
    assert c.delta == "token"
    assert c.finish_reason is None


def test_eval_example_and_result_are_frozen() -> None:
    ex = EvalExample(id="x1", prompt="Q?", target="A")
    res = EvalResult(example_id="x1", prediction="A", target="A", correct=True)
    with pytest.raises(FrozenInstanceError):
        ex.prompt = "new"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        res.correct = False  # type: ignore[misc]


def test_metadata_defaults_are_independent_instances() -> None:
    a = EvalExample(id="a", prompt="", target="")
    b = EvalExample(id="b", prompt="", target="")
    assert a.metadata == {}
    assert b.metadata == {}
    assert a.metadata is not b.metadata
