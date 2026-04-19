"""Prompt variants + semantic few-shot retrieval for Part E.

Each variant is a function ``(ctx, endings, activity, fewshot_examples) -> list[(prompt, continuation)]``
returning one pair per HellaSwag option. The pairs feed straight into
``common.vllm_client.VLLMClient.generate(prompt=prompt+continuation, echo=True,
logprobs=K, max_tokens=0)`` to compute per-option log-likelihoods.

Variants:

* ``baseline`` — lm-eval-style context + bare ending. No normalisation hook.
* ``clean_prompt`` — strip ``[header]`` tags and the ``activity_label`` prefix
  from the context (HellaSwag ships these artifacts in the raw data).
* ``fewshot_random`` — prepend ``k`` random exemplars from a pool.
* ``fewshot_semantic`` — prepend ``k`` exemplars chosen by sentence-transformer
  cosine similarity between the query context and each pool context.

The ``score_logprob`` function turns the raw per-token log-probabilities into a
score, with length / byte normalisation options.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from common.logging import get_logger

if TYPE_CHECKING:
    from collections.abc import Sequence

log = get_logger(__name__)

# HellaSwag context artefacts we want to strip in the "clean" variant.
_HEADER_PATTERN = re.compile(r"\[[^\]]+\]\s*")


def _strip_artifacts(ctx: str, activity: str) -> str:
    """Remove ``[header]`` tags and a leading ``activity_label`` prefix."""
    cleaned = _HEADER_PATTERN.sub("", ctx).strip()
    if activity and cleaned.lower().startswith(activity.lower()):
        cleaned = cleaned[len(activity) :].lstrip(" :-")
    return cleaned


@dataclass(frozen=True, slots=True)
class FewshotExample:
    """One demonstration used by few-shot variants."""

    ctx: str
    activity: str
    endings: list[str]
    label: int

    @property
    def target_ending(self) -> str:
        return self.endings[self.label]


def _format_fewshot(examples: Sequence[FewshotExample], clean: bool) -> str:
    """Format a list of exemplars into a single newline-separated block."""
    blocks: list[str] = []
    for ex in examples:
        ctx = _strip_artifacts(ex.ctx, ex.activity) if clean else ex.ctx
        # Use the same single-space convention as PromptPair continuations.
        blocks.append(f"{ctx} {ex.target_ending.lstrip()}")
    return "\n\n".join(blocks)


ScoringMode = Literal["sum", "length_norm", "byte_norm"]


def score_logprob(
    token_logprobs: Sequence[float],
    ending: str,
    mode: ScoringMode,
) -> float:
    """Turn a list of per-token log-probabilities into a scalar score.

    Args:
        token_logprobs: Sequence of per-token log-probabilities for the
            continuation (ending). Non-numeric entries are ignored.
        ending: The raw continuation string; used for byte-norm.
        mode: ``sum`` (raw), ``length_norm`` (divide by token count),
            ``byte_norm`` (divide by UTF-8 byte length).

    Returns:
        The scalar score; higher means more likely under the model.
    """
    clean_lps = [lp for lp in token_logprobs if lp is not None]
    total = sum(clean_lps)
    if mode == "sum":
        return total
    if mode == "length_norm":
        n = max(1, len(clean_lps))
        return total / n
    if mode == "byte_norm":
        n_bytes = max(1, len(ending.encode("utf-8")))
        return total / n_bytes
    raise ValueError(f"unknown ScoringMode: {mode}")


@dataclass(frozen=True, slots=True)
class PromptPair:
    """A (prompt, continuation) pair fed to VLLMClient for scoring."""

    prompt: str
    continuation: str


def _as_continuation(ending: str) -> str:
    """Return the ending prefixed with exactly one leading space.

    Real vLLM tokenisers produce different token counts for ``"foo"`` vs
    ``" foo"`` (a leading space is usually its own token). lm-eval's HellaSwag
    formatter normalises continuations with a single leading space; we
    replicate that here so the scoring is apples-to-apples regardless of the
    user's input convention.
    """
    return f" {ending.lstrip()}"


def build_pairs_baseline(
    ctx: str,
    endings: Sequence[str],
    activity: str,  # kept for interface symmetry with the other variants
    fewshot: Sequence[FewshotExample],  # baseline ignores fewshot
) -> list[PromptPair]:
    """Default lm-eval-style HellaSwag prompt: raw context + bare ending."""
    _ = (activity, fewshot)  # intentionally unused
    return [PromptPair(prompt=ctx, continuation=_as_continuation(e)) for e in endings]


def build_pairs_clean(
    ctx: str,
    endings: Sequence[str],
    activity: str,
    fewshot: Sequence[FewshotExample],
) -> list[PromptPair]:
    """Same as baseline but with header tags and activity label stripped."""
    _ = fewshot  # intentionally unused
    cleaned_ctx = _strip_artifacts(ctx, activity)
    return [PromptPair(prompt=cleaned_ctx, continuation=_as_continuation(e)) for e in endings]


def build_pairs_fewshot(
    ctx: str,
    endings: Sequence[str],
    activity: str,
    fewshot: Sequence[FewshotExample],
    *,
    clean: bool = True,
) -> list[PromptPair]:
    """Few-shot variant: prepend ``len(fewshot)`` demonstrations."""
    if not fewshot:
        return build_pairs_clean(ctx, endings, activity, fewshot)
    demo_block = _format_fewshot(fewshot, clean=clean)
    ctx_piece = _strip_artifacts(ctx, activity) if clean else ctx
    prefix = f"{demo_block}\n\n{ctx_piece}"
    return [PromptPair(prompt=prefix, continuation=_as_continuation(e)) for e in endings]


class SemanticRetriever:
    """Sentence-transformer-backed nearest-neighbour retriever.

    Builds a dense index over a pool of :class:`FewshotExample` contexts and,
    given a query context, returns the top-``k`` nearest exemplars by cosine
    similarity.

    Lazily imports ``sentence_transformers`` so this module is still importable
    without the ``improve`` extra installed.
    """

    def __init__(self, pool: Sequence[FewshotExample], model: str | None = None) -> None:
        self._pool: list[FewshotExample] = list(pool)
        self._embedder: Any | None = None
        self._pool_emb: Any | None = None
        self._model_name: str = model or "sentence-transformers/all-MiniLM-L6-v2"

    def _ensure_embedder(self) -> None:
        if self._embedder is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers not installed; run `uv sync --extra improve`."
            ) from exc
        log.info("semantic_retriever_load", model=self._model_name)
        self._embedder = SentenceTransformer(self._model_name)
        self._pool_emb = self._embedder.encode(
            [ex.ctx for ex in self._pool],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def topk(self, query: str, k: int) -> list[FewshotExample]:
        """Return the top-``k`` pool examples most similar to ``query``."""
        if k <= 0 or not self._pool:
            return []
        k = min(k, len(self._pool))
        self._ensure_embedder()
        assert self._embedder is not None and self._pool_emb is not None
        import numpy as np

        q_emb = self._embedder.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0]
        sims = np.asarray(self._pool_emb) @ np.asarray(q_emb)
        order = np.argsort(-sims)[:k]
        return [self._pool[int(i)] for i in order]


__all__ = [
    "FewshotExample",
    "PromptPair",
    "ScoringMode",
    "SemanticRetriever",
    "build_pairs_baseline",
    "build_pairs_clean",
    "build_pairs_fewshot",
    "score_logprob",
]
