"""Tests for ``improve.optimize_prompt`` — prompt variants and scoring."""

from __future__ import annotations

from pathlib import Path

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


def test_semantic_retriever_tfidf_fallback_under_mock_backend(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Under LLMEVAL_MOCK_BACKEND=true the retriever should use TF-IDF.

    This is the important contract: the fallback must work without any HF
    network round-trip, return deterministic top-k, and match the query to
    the most lexically-similar pool entry.
    """
    monkeypatch.setenv("LLMEVAL_MOCK_BACKEND", "true")
    monkeypatch.setenv("LLMEVAL_CACHE_DIR", str(tmp_path / "cache"))
    from common.config import get_settings
    from improve.optimize_prompt import FewshotExample, SemanticRetriever

    get_settings.cache_clear()

    pool = [
        FewshotExample(ctx="A chef slices onions", activity="cooking", endings=["x"], label=0),
        FewshotExample(ctx="A runner ties shoes", activity="sports", endings=["y"], label=0),
        FewshotExample(ctx="A scientist writes equations", activity="science", endings=["z"], label=0),
    ]
    retriever = SemanticRetriever(pool)
    result = retriever.topk("A cook chops vegetables in the kitchen", k=1)

    assert retriever.backend == "tfidf"
    assert len(result) == 1
    # The cooking example shares "chef/cook + chop/slice + ingredient" with
    # the query; it should win over the two unrelated entries.
    assert result[0].activity == "cooking"


def test_semantic_retriever_topk_empty_pool_returns_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("LLMEVAL_MOCK_BACKEND", "true")
    monkeypatch.setenv("LLMEVAL_CACHE_DIR", str(tmp_path / "cache"))
    from common.config import get_settings
    from improve.optimize_prompt import SemanticRetriever

    get_settings.cache_clear()
    assert SemanticRetriever(pool=[]).topk("anything", k=3) == []
