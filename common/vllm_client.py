"""Async HTTP client for a vLLM OpenAI-compatible endpoint.

This module is the single network boundary of the project. Every module that
generates text goes through :class:`VLLMClient`. Having one client means:

- Timeouts and retries are defined once and applied everywhere.
- A single ``LLMEVAL_MOCK_BACKEND=true`` switch lets the whole stack run
  offline against deterministic canned responses.
- Log events carry consistent context.

The client supports both non-streaming (:meth:`VLLMClient.generate`) and
streaming (:meth:`VLLMClient.stream_generate`) completions. For chat-style
interfaces, extend this module rather than creating a parallel client.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import TracebackType
from typing import Any

import httpx

from common.config import get_settings
from common.errors import VLLMClientError, VLLMTimeoutError
from common.logging import get_logger
from common.types import GenerationResult, StreamChunk

log = get_logger(__name__)

_DEFAULT_MAX_TOKENS = 256
_BACKOFF_BASE_S = 0.5


class VLLMClient:
    """Async client for a vLLM OpenAI-compatible endpoint.

    Use as an async context manager::

        async with VLLMClient() as client:
            result = await client.generate("Hello")

    Each instance owns an :class:`httpx.AsyncClient` and must be closed. The
    client retries transport and 5xx errors with exponential backoff; 4xx
    responses are never retried.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_s: float | None = None,
        max_retries: int | None = None,
        model: str | None = None,
    ) -> None:
        """Construct the client.

        Args:
            base_url: Override the OpenAI-compatible base URL
                (defaults to ``Settings.vllm_base_url``).
            api_key: Bearer token (defaults to ``Settings.vllm_api_key``).
            timeout_s: Per-request timeout (defaults to ``Settings.vllm_timeout_s``).
            max_retries: Max retry attempts (defaults to ``Settings.vllm_max_retries``).
            model: Model name (defaults to ``Settings.model_name``).
        """
        settings = get_settings()
        self._base_url = (base_url or settings.vllm_base_url).rstrip("/")
        self._api_key = api_key or settings.vllm_api_key
        self._timeout = timeout_s if timeout_s is not None else settings.vllm_timeout_s
        self._max_retries = max_retries if max_retries is not None else settings.vllm_max_retries
        self._model = model or settings.model_name
        self._mock = settings.mock_backend
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> VLLMClient:
        self._client = httpx.AsyncClient(
            timeout=self._timeout,
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise VLLMClientError(
                "VLLMClient must be used as an async context manager "
                "(`async with VLLMClient() as client:`)"
            )
        return self._client

    async def _post_with_retry(
        self,
        path: str,
        json_body: dict[str, Any],
    ) -> httpx.Response:
        """POST ``json_body`` to ``path`` with retry on transport / 5xx errors."""
        client = self._ensure_client()
        url = f"{self._base_url}{path}"
        last_exc: BaseException | None = None
        for attempt in range(self._max_retries + 1):
            try:
                resp = await client.post(url, json=json_body)
                if 400 <= resp.status_code < 500:
                    raise VLLMClientError(f"{resp.status_code} from {path}: {resp.text[:200]}")
                if resp.status_code >= 500:
                    raise VLLMClientError(f"{resp.status_code} from {path}: {resp.text[:200]}")
                return resp
            except httpx.TimeoutException as exc:
                err = VLLMTimeoutError(f"timeout on {path}")
                err.__cause__ = exc
                last_exc = err
            except httpx.HTTPError as exc:
                err_http = VLLMClientError(f"http error on {path}: {exc}")
                err_http.__cause__ = exc
                last_exc = err_http
            except VLLMClientError as exc:
                if path.endswith("/completions") and "4" in str(exc)[:3]:
                    # 4xx: do not retry
                    raise
                last_exc = exc
            if attempt < self._max_retries:
                backoff = _BACKOFF_BASE_S * (2**attempt)
                log.warning(
                    "vllm_retry",
                    path=path,
                    attempt=attempt + 1,
                    backoff_s=backoff,
                )
                await asyncio.sleep(backoff)
        assert last_exc is not None  # guaranteed by loop structure
        raise last_exc

    @asynccontextmanager
    async def _stream_with_retry(
        self,
        path: str,
        json_body: dict[str, Any],
    ) -> AsyncIterator[httpx.Response]:
        """Open a streaming POST; yields the response for the caller to iterate."""
        client = self._ensure_client()
        url = f"{self._base_url}{path}"
        last_exc: BaseException | None = None
        for attempt in range(self._max_retries + 1):
            try:
                req = client.build_request("POST", url, json=json_body)
                resp = await client.send(req, stream=True)
                if 400 <= resp.status_code < 500:
                    body = (await resp.aread()).decode("utf-8", errors="replace")
                    await resp.aclose()
                    raise VLLMClientError(f"{resp.status_code} from {path}: {body[:200]}")
                if resp.status_code >= 500:
                    await resp.aclose()
                    raise VLLMClientError(f"{resp.status_code} from {path}")
                try:
                    yield resp
                finally:
                    await resp.aclose()
                return
            except httpx.TimeoutException as exc:
                err = VLLMTimeoutError(f"timeout on {path}")
                err.__cause__ = exc
                last_exc = err
            except httpx.HTTPError as exc:
                err_http = VLLMClientError(f"http error on {path}: {exc}")
                err_http.__cause__ = exc
                last_exc = err_http
            if attempt < self._max_retries:
                backoff = _BACKOFF_BASE_S * (2**attempt)
                log.warning(
                    "vllm_stream_retry",
                    path=path,
                    attempt=attempt + 1,
                    backoff_s=backoff,
                )
                await asyncio.sleep(backoff)
        assert last_exc is not None
        raise last_exc

    async def generate(
        self,
        prompt: str,
        *,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        seed: int | None = None,
        n: int = 1,
    ) -> GenerationResult:
        """Run a single non-streaming completion.

        Args:
            prompt: The raw prompt string.
            max_tokens: Maximum completion tokens.
            temperature: Sampling temperature (``0.0`` for deterministic).
            top_p: Nucleus sampling probability mass.
            stop: Optional stop sequences.
            seed: Optional RNG seed passed to the backend.
            n: Number of completions to request (only the first is returned).

        Returns:
            A :class:`GenerationResult` with the completion text and usage info.

        Raises:
            VLLMClientError: On 4xx / 5xx / transport errors after retries.
            VLLMTimeoutError: On timeout after retries.
        """
        if self._mock:
            return _mock_generation(prompt, max_tokens)

        body: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "stream": False,
        }
        if stop:
            body["stop"] = stop
        if seed is not None:
            body["seed"] = seed

        resp = await self._post_with_retry("/completions", body)
        data = resp.json()
        choice = data["choices"][0]
        usage = data.get("usage", {})
        return GenerationResult(
            text=choice["text"],
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            finish_reason=choice.get("finish_reason", "stop"),
            raw=data,
        )

    async def stream_generate(
        self,
        prompt: str,
        *,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        temperature: float = 0.0,
        top_p: float = 1.0,
        stop: list[str] | None = None,
        seed: int | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Yield :class:`StreamChunk` objects as the server emits tokens.

        Args:
            prompt: The raw prompt string.
            max_tokens: Maximum completion tokens.
            temperature: Sampling temperature.
            top_p: Nucleus sampling probability mass.
            stop: Optional stop sequences.
            seed: Optional RNG seed.

        Yields:
            :class:`StreamChunk` objects; the final chunk has ``finish_reason`` set.

        Raises:
            VLLMClientError: On 4xx / 5xx / transport errors after retries.
            VLLMTimeoutError: On timeout after retries.
        """
        if self._mock:
            for chunk in _mock_stream(prompt, max_tokens):
                yield chunk
            return

        body: dict[str, Any] = {
            "model": self._model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "stream": True,
        }
        if stop:
            body["stop"] = stop
        if seed is not None:
            body["seed"] = seed

        async with self._stream_with_retry("/completions", body) as resp:
            async for line in resp.aiter_lines():
                if not line or not line.startswith("data: "):
                    continue
                payload = line[len("data: ") :].strip()
                if payload == "[DONE]":
                    return
                data = json.loads(payload)
                choice = data["choices"][0]
                yield StreamChunk(
                    delta=choice.get("text", ""),
                    finish_reason=choice.get("finish_reason"),
                )


def _mock_generation(prompt: str, max_tokens: int) -> GenerationResult:
    """Return a deterministic canned response for offline dev."""
    text = f"[mock completion for prompt of length {len(prompt)}]"
    return GenerationResult(
        text=text,
        prompt_tokens=max(1, len(prompt.split())),
        completion_tokens=max(1, min(max_tokens, len(text.split()))),
        finish_reason="stop",
    )


def _mock_stream(prompt: str, max_tokens: int) -> list[StreamChunk]:
    """Return a deterministic stream split into word-sized chunks."""
    text = _mock_generation(prompt, max_tokens).text
    chunks: list[StreamChunk] = [StreamChunk(delta=word + " ") for word in text.split()]
    chunks.append(StreamChunk(delta="", finish_reason="stop"))
    return chunks


__all__ = ["VLLMClient"]
