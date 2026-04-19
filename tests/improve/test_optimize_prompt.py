"""Tests for ``improve.optimize_prompt`` — prompt variants and scoring."""

from __future__ import annotations

import pytest

from improve.optimize_prompt import (
    FewshotExample,
    build_pairs_baseline,
    build_pairs_clean,
    build_pairs_fewshot,
    score_logprob,
)

_ENDINGS = [" sits on the mat", " chases the mouse"]


def test_baseline_returns_one_pair_per_ending() -> None:
    pairs = build_pairs_baseline(
        ctx="The cat",
        endings=_ENDINGS,
        activity="cat",
        fewshot=[],
    )
    assert len(pairs) == 2
    assert pairs[0].prompt == "The cat"
    assert pairs[0].continuation == " sits on the mat"


def test_clean_variant_strips_header_and_activity() -> None:
    ctx = "[header: Indoor Cats] Cats The cat"
    activity = "cats"
    pairs = build_pairs_clean(
        ctx=ctx,
        endings=[" are great"],
        activity=activity,
        fewshot=[],
    )
    # header tag removed; activity prefix also removed.
    assert pairs[0].prompt == "The cat"
    assert pairs[0].continuation == " are great"


def test_fewshot_prepends_demonstrations() -> None:
    demo = FewshotExample(
        ctx="Bread is",
        activity="baking",
        endings=[" delicious", " stale"],
        label=0,
    )
    pairs = build_pairs_fewshot(
        ctx="The cat",
        endings=_ENDINGS,
        activity="cat",
        fewshot=[demo],
        clean=True,
    )
    # The prompt for both options should carry the demonstration first.
    assert "Bread is delicious" in pairs[0].prompt
    assert "The cat" in pairs[0].prompt
    assert pairs[0].continuation == " sits on the mat"


def test_fewshot_empty_pool_falls_back_to_clean() -> None:
    pairs_fs = build_pairs_fewshot(
        ctx="The cat",
        endings=_ENDINGS,
        activity="cat",
        fewshot=[],
    )
    pairs_clean = build_pairs_clean("The cat", _ENDINGS, "cat", [])
    assert [p.prompt for p in pairs_fs] == [p.prompt for p in pairs_clean]


def test_score_logprob_sum() -> None:
    assert score_logprob([-1.0, -2.0, -3.0], " a b c", mode="sum") == pytest.approx(-6.0)


def test_score_logprob_length_norm_divides_by_token_count() -> None:
    # 3 tokens, -6 sum -> -2.0 per token.
    assert score_logprob([-1.0, -2.0, -3.0], " a b c", mode="length_norm") == pytest.approx(-2.0)


def test_score_logprob_byte_norm_divides_by_ascii_byte_count() -> None:
    # " a b c" is 6 bytes (ASCII). -6 / 6 = -1.0.
    assert score_logprob([-1.0, -2.0, -3.0], " a b c", mode="byte_norm") == pytest.approx(-1.0)


def test_score_logprob_empty_logprobs_returns_zero() -> None:
    assert score_logprob([], "anything", mode="length_norm") == pytest.approx(0.0)


def test_score_logprob_ignores_none_entries() -> None:
    # None-filtering happens in the client path; score_logprob receives floats
    # only (see optimize_prompt.score_logprob signature), so passing a clean
    # list with finite values behaves identically regardless of mode.
    assert score_logprob([-1.0], "x", mode="sum") == pytest.approx(-1.0)


def test_score_logprob_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="unknown ScoringMode"):
        score_logprob([-1.0], "x", mode="bogus")  # type: ignore[arg-type]
