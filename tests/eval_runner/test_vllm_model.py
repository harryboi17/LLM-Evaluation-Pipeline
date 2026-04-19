"""Tests for ``eval_runner.vllm_model``.

These tests skip gracefully if ``lm-eval`` isn't installed (which is the case
when only the base dependency group is present). They rely exclusively on the
mock backend; no real model, network, or GPU is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from common.config import get_settings

lm_eval = pytest.importorskip("lm_eval", reason="lm-eval extra not installed")

from eval_runner.vllm_model import (  # noqa: E402
    _score_continuation,
    get_vllm_eval_model_class,
)


@dataclass
class _FakeInstance:
    """Minimal stand-in for :class:`lm_eval.api.instance.Instance`."""

    args: tuple[Any, ...]


@pytest.fixture
def _mock_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("LLMEVAL_MOCK_BACKEND", "true")
    monkeypatch.setenv("LLMEVAL_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("LLMEVAL_RESULTS_DIR", str(tmp_path / "results"))
    monkeypatch.setenv("LLMEVAL_MODEL_NAME", "mock/test-model")
    get_settings.cache_clear()


def test_score_continuation_sums_tail_and_checks_greedy() -> None:
    raw = {
        "choices": [
            {
                "logprobs": {
                    "tokens": ["the", "quick", "brown", "fox"],
                    "token_logprobs": [None, -0.5, -1.0, -0.25],
                    "top_logprobs": [
                        None,
                        {"quick": -0.5, "slow": -2.0},
                        {"brown": -1.0, "black": -1.5},
                        {"fox": -0.25, "dog": -3.0},
                    ],
                }
            }
        ]
    }
    # Score last 2 tokens ("brown", "fox"): -1.0 + -0.25 = -1.25, greedy.
    total, is_greedy = _score_continuation(raw, n_cont=2)
    assert total == pytest.approx(-1.25)
    assert is_greedy is True


def test_score_continuation_detects_non_greedy_continuation() -> None:
    raw = {
        "choices": [
            {
                "logprobs": {
                    "tokens": ["cat"],
                    "token_logprobs": [-2.0],
                    "top_logprobs": [{"cat": -2.0, "dog": -0.5}],
                }
            }
        ]
    }
    total, is_greedy = _score_continuation(raw, n_cont=1)
    assert total == pytest.approx(-2.0)
    # "dog" had a higher logprob than the actual token "cat" -> not greedy.
    assert is_greedy is False


def test_score_continuation_handles_empty_logprobs() -> None:
    raw = {"choices": [{"logprobs": {}}]}
    total, is_greedy = _score_continuation(raw, n_cont=3)
    assert total == 0.0
    assert is_greedy is True


def test_loglikelihood_against_mock_backend(_mock_env: None) -> None:
    model_cls = get_vllm_eval_model_class()
    model = model_cls()
    requests = [
        _FakeInstance(("The capital of France is", " Paris")),
        _FakeInstance(("Two plus two is", " four")),
    ]
    results = model.loglikelihood(requests)
    assert len(results) == 2
    for logp, is_greedy in results:
        assert isinstance(logp, float)
        assert isinstance(is_greedy, bool)


def test_loglikelihood_rolling_against_mock_backend(_mock_env: None) -> None:
    model_cls = get_vllm_eval_model_class()
    model = model_cls()
    requests = [
        _FakeInstance(("A short piece of text to score.",)),
        _FakeInstance(("Another longer piece of text with multiple words.",)),
    ]
    results = model.loglikelihood_rolling(requests)
    assert len(results) == 2
    for lp in results:
        assert isinstance(lp, float)
        # Mock backend emits logprobs of -1.0 per echoed token (except the first
        # which is None), so the sum must be strictly negative.
        assert lp < 0.0


def test_loglikelihood_rolling_math_matches_mock_emission(_mock_env: None) -> None:
    """Verify the rolling sum equals what the mock generator actually emits.

    The mock generator in ``common.vllm_client._mock_generation`` assigns
    ``logprob=None`` to the first echoed token and ``-1.0`` to every
    subsequent token. So for a text with N whitespace tokens, the rolling
    log-likelihood should be exactly ``-(N - 1)``. This catches regressions
    in the slice math or ``None`` filtering that purely "is it negative?"
    assertions would miss.
    """
    model_cls = get_vllm_eval_model_class()
    model = model_cls()
    inputs = [
        ("one two three four five", -4.0),  # 5 tokens -> -(5-1) = -4
        ("alpha beta", -1.0),  # 2 tokens -> -1
        ("solo", 0.0),  # 1 token -> first is None -> sum = 0
    ]
    requests = [_FakeInstance((text,)) for text, _ in inputs]
    results = model.loglikelihood_rolling(requests)
    assert len(results) == len(inputs)
    for (_, expected), actual in zip(inputs, results, strict=True):
        assert actual == pytest.approx(expected), (
            f"rolling sum drift: expected {expected}, got {actual}"
        )


def test_generate_until_against_mock_backend(_mock_env: None) -> None:
    model_cls = get_vllm_eval_model_class()
    model = model_cls()
    requests = [
        _FakeInstance(("What is 2+2?", {"until": ["\n"], "max_gen_toks": 8})),
        _FakeInstance(("Describe Python.", {"until": ["."], "max_gen_toks": 32})),
    ]
    results = model.generate_until(requests)
    assert len(results) == 2
    for text in results:
        assert isinstance(text, str)


def test_generate_until_cache_hit_skips_network(_mock_env: None) -> None:
    """A second identical call should be served entirely from the cache."""
    model_cls = get_vllm_eval_model_class()
    model = model_cls()
    req = _FakeInstance(("Tell me a fact.", {"until": ["\n"], "max_gen_toks": 8}))

    first = model.generate_until([req])
    # Subsequent identical call hits the cache.
    second = model.generate_until([req])
    assert first == second


def test_gen_overrides_from_settings_change_cache_key(
    monkeypatch: pytest.MonkeyPatch, _mock_env: None
) -> None:
    """Different temperature overrides must not hit each other's cache rows.

    Exercises the ``improve.sweep`` -> Settings -> vllm_model plumbing: the
    sweep injects ``LLMEVAL_GEN_TEMPERATURE`` (and friends) as env vars per
    subprocess; downstream, ``_one_generate`` reads them from Settings and
    mixes them into the cache key so each sweep cell records its own row.
    """
    from common.config import get_settings
    from common.vllm_client import _mock_generation

    req = _FakeInstance(("Pick a number.", {"until": ["\n"], "max_gen_toks": 4}))
    call_count = {"n": 0}

    orig_mock = _mock_generation

    def counting_mock(*args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        call_count["n"] += 1
        return orig_mock(*args, **kwargs)

    monkeypatch.setattr("common.vllm_client._mock_generation", counting_mock)

    # Cell 1: temperature=0.0 (default).
    model_cls = get_vllm_eval_model_class()
    model = model_cls()
    model.generate_until([req])
    calls_after_first = call_count["n"]

    # Cell 2: temperature=0.7 via env; different cache key so it must re-fire.
    monkeypatch.setenv("LLMEVAL_GEN_TEMPERATURE", "0.7")
    get_settings.cache_clear()
    model2 = model_cls()
    model2.generate_until([req])
    assert call_count["n"] > calls_after_first

    # Cell 3: re-enter cell 2's config; must be a cache hit (no new mock call).
    current = call_count["n"]
    model3 = model_cls()
    model3.generate_until([req])
    assert call_count["n"] == current


def test_create_from_arg_string_parses_key_value_pairs(_mock_env: None) -> None:
    model_cls = get_vllm_eval_model_class()
    m = model_cls.create_from_arg_string("max_concurrency=4,default_max_gen_tokens=64")
    assert m._max_concurrency == 4
    assert m._default_max_gen_tokens == 64
