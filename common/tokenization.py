"""Tokenizer helper shared by :mod:`eval_runner` and :mod:`improve`.

We need an exact-count tokeniser at two layers of the pipeline:

* Part B's ``VLLMEvalModel.loglikelihood`` needs ``n_ctx`` and ``n_full`` to
  know which slice of the echoed token log-probs belongs to the continuation.
* Part E's ``improve.infer`` needs ``n_cont`` for the same reason, plus it
  optionally length-normalises by continuation token count.

Having one implementation guarantees both layers agree on what a "token" is,
which matters whenever we compare a Part B run (via lm-eval) with a Part E
run (custom eval loop) on the same model.

Two modes:

* **HF** (default) — lazy-loads :class:`transformers.AutoTokenizer` and
  reports the real BPE count. Matches what vLLM's server-side tokeniser sees
  token-for-token.
* **Whitespace fallback** — used when ``Settings.mock_backend`` is ``True``
  or when the ``transformers`` extra isn't installed. Matches the
  whitespace-based ``_mock_generation`` in :mod:`common.vllm_client`, so the
  whole offline pipeline stays self-consistent.
"""

from __future__ import annotations

from typing import Any

from common.config import get_settings
from common.errors import EvalError
from common.logging import get_logger

log = get_logger(__name__)


class _WhitespaceTokenizer:
    """Whitespace-splitting stand-in used in mock mode."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        _ = add_special_tokens  # parity with HF tokeniser signature
        return list(range(len(text.split())))


class TokenCounter:
    """Count tokens using either an HF tokeniser or the whitespace fallback.

    Lazy-loads the HF tokeniser so the heavy ``transformers`` import and
    the HF-hub round-trip only happen when the caller actually counts tokens
    in non-mock mode.

    Args:
        model: HuggingFace model id for ``AutoTokenizer.from_pretrained``.
        use_mock: If ``True``, skip the HF tokeniser and use whitespace
            splitting. Normally fed from ``Settings.mock_backend``.
    """

    def __init__(self, model: str, use_mock: bool) -> None:
        self._model = model
        self._use_mock = use_mock
        self._hf_tokenizer: Any | None = None

    @classmethod
    def from_settings(cls, model: str | None = None) -> TokenCounter:
        """Factory that picks the right mode from :class:`Settings`."""
        settings = get_settings()
        return cls(model or settings.model_name, use_mock=settings.mock_backend)

    def _ensure(self) -> Any:
        if self._use_mock:
            return _WhitespaceTokenizer()
        if self._hf_tokenizer is None:
            try:
                from transformers import AutoTokenizer
            except ImportError as exc:
                raise EvalError(
                    "transformers is not installed. Run `uv sync --extra eval`."
                ) from exc
            log.info("loading_hf_tokenizer", model=self._model)
            self._hf_tokenizer = AutoTokenizer.from_pretrained(self._model)
        return self._hf_tokenizer

    def count(self, text: str) -> int:
        """Return the number of tokens in ``text`` (no special tokens)."""
        return len(self._ensure().encode(text, add_special_tokens=False))


__all__ = ["TokenCounter"]
