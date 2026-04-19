"""Custom lm-evaluation-harness model wrapper backed by vLLM + prompt cache.

Implements the three methods lm-eval-harness calls on an :class:`~lm_eval.api.model.LM`:

* :meth:`loglikelihood` — score ``(context, continuation)`` pairs by summing the
  log-probabilities of the continuation tokens given the context. Used by
  multiple-choice tasks (MMLU, HellaSwag, ARC, Winogrande, …).
* :meth:`loglikelihood_rolling` — total log-likelihood of a long string for
  perplexity-style tasks (wikitext, lambada). Same mechanism as above with
  the whole string treated as continuation.
* :meth:`generate_until` — free-form generation with a list of stop strings.
  Used by generative tasks including our custom short-answer benchmark.

All three paths share a common substrate:

1. Requests are fanned out via ``asyncio.gather`` with a bounded
   :class:`asyncio.Semaphore` so the client doesn't fire a flood at vLLM;
   vLLM's continuous batching takes care of efficient scheduling on the server.
2. Every (prompt, decoding-params) pair is memoised via
   :class:`common.cache.PromptCache`, so Part E ablations and repeat runs are
   cheap and deterministic.
3. In ``LLMEVAL_MOCK_BACKEND=true`` mode, :class:`common.vllm_client.VLLMClient`
   returns canned responses (including synthetic logprobs) so the full
   evaluation pipeline is runnable offline, and a whitespace tokenizer stands
   in for the HF tokenizer.

Log-likelihood math (per OpenAI-completions convention used by vLLM):

1. Tokenize ``context`` alone → ``n_ctx``.
2. Tokenize ``context + continuation`` → ``n_full``.
3. ``n_cont = n_full - n_ctx``.
4. Send ``prompt = context + continuation`` with ``echo=True``,
   ``max_tokens=0``, ``logprobs=K`` — the server returns token-level
   log-probabilities including for echoed prompt tokens.
5. Sum ``token_logprobs[-n_cont:]``; that's the loglikelihood.
6. ``is_greedy`` is ``True`` iff every continuation token matches the top-1
   prediction at its position.
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from common.cache import PromptCache
from common.config import get_settings
from common.errors import EvalError
from common.logging import get_logger

if TYPE_CHECKING:
    from lm_eval.api.instance import Instance

log = get_logger(__name__)

_LOGPROBS_TOP_K = 5
_LOGLIK_CACHE_PARAMS_BASE: dict[str, Any] = {
    "echo": True,
    "logprobs": _LOGPROBS_TOP_K,
    "max_tokens": 0,
    "temperature": 0.0,
}


def _lm_eval_base() -> Any:
    """Return the ``lm_eval.api.model.LM`` base class.

    Lazy-imported so the module is importable without ``lm-eval`` installed
    (e.g., for isolated unit tests of helper functions). The harness itself
    must be installed whenever :meth:`VLLMEvalModel.run` is actually called.
    """
    try:
        from lm_eval.api.model import LM as _LM
    except ImportError as exc:
        raise EvalError(
            "lm-eval is not installed. Run `uv sync --extra eval` first."
        ) from exc
    return _LM


class _WhitespaceTokenizer:
    """Whitespace-splitting stand-in used when the real HF tokenizer is unavailable.

    Matches the tokenisation used by :func:`common.vllm_client._mock_generation`
    so the mock backend and the eval layer count tokens the same way.
    """

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return list(range(len(text.split())))


class _TokenCounter:
    """Adapter that produces token counts using either HF or the whitespace fallback.

    Lazy-loads the HF tokenizer on first real call so the heavy import and
    the network round-trip to the HF hub only happen when actually needed.
    """

    def __init__(self, model: str, use_mock: bool) -> None:
        self._model = model
        self._use_mock = use_mock
        self._hf_tokenizer: Any | None = None

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
        return len(self._ensure().encode(text, add_special_tokens=False))


def _build_vllm_eval_class() -> type:
    """Build :class:`VLLMEvalModel` subclassing the lm-eval ``LM`` at call time."""
    base = _lm_eval_base()

    class VLLMEvalModel(base):  # type: ignore[misc, valid-type]
        """lm-evaluation-harness :class:`LM` delegating to :class:`VLLMClient`.

        Args:
            model: HuggingFace model id. Defaults to ``Settings.model_name``.
            max_concurrency: Upper bound on concurrent in-flight requests to vLLM.
            default_max_gen_tokens: Fallback ``max_gen_toks`` when a harness
                task doesn't supply one.
        """

        def __init__(
            self,
            model: str | None = None,
            max_concurrency: int = 8,
            default_max_gen_tokens: int = 256,
        ) -> None:
            super().__init__()
            settings = get_settings()
            self._model: str = model or settings.model_name
            self._max_concurrency: int = max_concurrency
            self._default_max_gen_tokens: int = default_max_gen_tokens
            self._cache: PromptCache = PromptCache()
            self._token_counter: _TokenCounter = _TokenCounter(
                self._model, use_mock=settings.mock_backend
            )

        # --- lm-eval registration glue ---------------------------------------

        @classmethod
        def create_from_arg_string(
            cls,
            arg_string: str,
            additional_config: dict[str, Any] | None = None,
        ) -> VLLMEvalModel:
            """Parse ``"k1=v1,k2=v2"`` into kwargs and construct the model."""
            kwargs: dict[str, Any] = {}
            for raw in (arg_string or "").split(","):
                part = raw.strip()
                if not part:
                    continue
                key, _, value = part.partition("=")
                key = key.strip()
                value = value.strip()
                if value.isdigit():
                    kwargs[key] = int(value)
                else:
                    try:
                        kwargs[key] = float(value)
                    except ValueError:
                        kwargs[key] = value
            return cls(**kwargs)

        # --- Core harness API -----------------------------------------------

        def loglikelihood(
            self, requests: list[Instance]
        ) -> list[tuple[float, bool]]:
            """Score ``(context, continuation)`` pairs."""
            pairs: list[tuple[str, str]] = [
                (str(req.args[0]), str(req.args[1])) for req in requests
            ]
            log.info("loglikelihood_batch", n=len(pairs))
            return asyncio.run(self._loglikelihood_batch(pairs))

        def loglikelihood_rolling(
            self, requests: list[Instance]
        ) -> list[float]:
            """Return total log-likelihood for each input string."""
            texts: list[str] = [str(req.args[0]) for req in requests]
            log.info("loglikelihood_rolling_batch", n=len(texts))
            return asyncio.run(self._rolling_batch(texts))

        def generate_until(
            self, requests: list[Instance]
        ) -> list[str]:
            """Free-form generation with stop sequences."""
            items: list[tuple[str, dict[str, Any]]] = []
            for req in requests:
                context = str(req.args[0])
                gen_kwargs: dict[str, Any] = (
                    dict(req.args[1]) if len(req.args) > 1 and req.args[1] else {}
                )
                items.append((context, gen_kwargs))
            log.info("generate_until_batch", n=len(items))
            return asyncio.run(self._generate_batch(items))

        # --- Async workers --------------------------------------------------

        async def _loglikelihood_batch(
            self, pairs: list[tuple[str, str]]
        ) -> list[tuple[float, bool]]:
            from common.vllm_client import VLLMClient

            sem = asyncio.Semaphore(self._max_concurrency)
            async with VLLMClient() as client:
                return list(
                    await asyncio.gather(
                        *(self._one_loglikelihood(client, sem, c, cc) for c, cc in pairs)
                    )
                )

        async def _rolling_batch(self, texts: list[str]) -> list[float]:
            from common.vllm_client import VLLMClient

            sem = asyncio.Semaphore(self._max_concurrency)
            async with VLLMClient() as client:
                return list(
                    await asyncio.gather(
                        *(self._one_rolling(client, sem, t) for t in texts)
                    )
                )

        async def _generate_batch(
            self, items: list[tuple[str, dict[str, Any]]]
        ) -> list[str]:
            from common.vllm_client import VLLMClient

            sem = asyncio.Semaphore(self._max_concurrency)
            async with VLLMClient() as client:
                return list(
                    await asyncio.gather(
                        *(self._one_generate(client, sem, c, kw) for c, kw in items)
                    )
                )

        # --- Per-request implementations ------------------------------------

        async def _one_loglikelihood(
            self,
            client: Any,
            sem: asyncio.Semaphore,
            context: str,
            continuation: str,
        ) -> tuple[float, bool]:
            n_ctx = self._token_counter.count(context)
            n_full = self._token_counter.count(context + continuation)
            n_cont = max(0, n_full - n_ctx)
            if n_cont == 0:
                # Continuation adds no tokens (e.g., whitespace-only); trivially greedy.
                return (0.0, True)

            prompt = context + continuation
            raw = await self._get_or_fetch(
                client,
                sem,
                prompt,
                _LOGLIK_CACHE_PARAMS_BASE,
                _fetch_with_echo,
            )
            return _score_continuation(raw, n_cont)

        async def _one_rolling(
            self,
            client: Any,
            sem: asyncio.Semaphore,
            text: str,
        ) -> float:
            params = {**_LOGLIK_CACHE_PARAMS_BASE, "logprobs": 1}
            raw = await self._get_or_fetch(
                client,
                sem,
                text,
                params,
                _fetch_with_echo_logprobs1,
            )
            lp_info: dict[str, Any] = raw["choices"][0].get("logprobs") or {}
            token_logprobs: list[float | None] = lp_info.get("token_logprobs", [])
            return sum(lp for lp in token_logprobs if lp is not None)

        async def _one_generate(
            self,
            client: Any,
            sem: asyncio.Semaphore,
            context: str,
            gen_kwargs: dict[str, Any],
        ) -> str:
            max_tokens = int(gen_kwargs.get("max_gen_toks", self._default_max_gen_tokens))
            temperature = float(gen_kwargs.get("temperature", 0.0))
            top_p = float(gen_kwargs.get("top_p", 1.0))
            until_raw = gen_kwargs.get("until")
            stop: list[str] | None = list(until_raw) if until_raw else None
            seed = gen_kwargs.get("seed")

            cache_params: dict[str, Any] = {
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "stop": stop or [],
                "seed": seed,
            }
            cached = self._cache.get(self._model, context, cache_params)
            if cached is not None:
                return str(cached.get("text", ""))

            async with sem:
                result = await client.generate(
                    context,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    stop=stop,
                    seed=seed,
                )
            text: str = str(result.text)
            # Defensive trim: if the server went past a stop string, truncate it here.
            for s in stop or []:
                idx = text.find(s)
                if idx >= 0:
                    text = text[:idx]
            self._cache.put(self._model, context, cache_params, {"text": text})
            return text

        async def _get_or_fetch(
            self,
            client: Any,
            sem: asyncio.Semaphore,
            prompt: str,
            params: dict[str, Any],
            fetch: Any,
        ) -> dict[str, Any]:
            """Look up or populate the prompt cache for a logprob-style request."""
            cached = self._cache.get(self._model, prompt, params)
            if cached is not None:
                return cached
            async with sem:
                raw: dict[str, Any] = await fetch(client, prompt, params)
            self._cache.put(self._model, prompt, params, raw)
            return raw

    return VLLMEvalModel


async def _fetch_with_echo(
    client: Any, prompt: str, params: dict[str, Any]
) -> dict[str, Any]:
    result = await client.generate(
        prompt,
        max_tokens=params["max_tokens"],
        temperature=params["temperature"],
        echo=params["echo"],
        logprobs=params["logprobs"],
    )
    return dict(result.raw)


async def _fetch_with_echo_logprobs1(
    client: Any, prompt: str, params: dict[str, Any]
) -> dict[str, Any]:
    return await _fetch_with_echo(client, prompt, params)


def _score_continuation(raw: dict[str, Any], n_cont: int) -> tuple[float, bool]:
    """Extract ``(sum_logprob, is_greedy)`` over the last ``n_cont`` tokens.

    Args:
        raw: The OpenAI-style completion payload (as stored in
            :attr:`GenerationResult.raw`).
        n_cont: Number of continuation tokens at the tail of the echoed prompt.

    Returns:
        Tuple of total continuation log-probability and a boolean indicating
        whether each continuation token was the argmax of its position's top-K.
    """
    lp_info: dict[str, Any] = raw["choices"][0].get("logprobs") or {}
    tokens: list[str] = lp_info.get("tokens", [])
    token_logprobs: list[float | None] = lp_info.get("token_logprobs", [])
    top_logprobs: list[dict[str, float] | None] = lp_info.get("top_logprobs", [])

    cont_tokens = tokens[-n_cont:] if tokens else []
    cont_lps = token_logprobs[-n_cont:] if token_logprobs else []
    cont_top = top_logprobs[-n_cont:] if top_logprobs else []

    total = sum(lp for lp in cont_lps if lp is not None)

    is_greedy = True
    for tok, top in zip(cont_tokens, cont_top, strict=False):
        if not top:
            continue
        best = max(top.items(), key=lambda kv: kv[1])[0]
        if best != tok:
            is_greedy = False
            break
    return (total, is_greedy)


def get_vllm_eval_model_class() -> type:
    """Return the :class:`VLLMEvalModel` class (lazy because lm-eval is optional)."""
    return _build_vllm_eval_class()


__all__ = [
    "get_vllm_eval_model_class",
]
