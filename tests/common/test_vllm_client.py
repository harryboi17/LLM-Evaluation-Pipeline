"""Tests for ``common.vllm_client``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from common.config import get_settings
from common.errors import VLLMClientError, VLLMTimeoutError
from common.vllm_client import VLLMClient


def _completion_payload(text: str = "hello") -> dict[str, Any]:
    return {
        "id": "cmpl-1",
        "object": "text_completion",
        "choices": [
            {
                "index": 0,
                "text": text,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    }


@pytest.fixture
def transport_ok() -> httpx.MockTransport:
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["prompt"] == "hi"
        return httpx.Response(200, json=_completion_payload("world"))

    return httpx.MockTransport(handler)


async def _make_client(transport: httpx.MockTransport) -> VLLMClient:
    client = VLLMClient(base_url="http://mock/v1", max_retries=1, timeout_s=5.0)
    client._client = httpx.AsyncClient(transport=transport, timeout=5.0)  # type: ignore[assignment]
    return client


async def test_generate_parses_completion(
    transport_ok: httpx.MockTransport, isolated_env: Path
) -> None:
    client = await _make_client(transport_ok)
    try:
        result = await client.generate("hi", max_tokens=16)
        assert result.text == "world"
        assert result.prompt_tokens == 5
        assert result.completion_tokens == 2
        assert result.finish_reason == "stop"
        assert result.total_tokens == 7
    finally:
        await client._client.aclose()  # type: ignore[union-attr]


async def test_generate_without_context_manager_raises(isolated_env: Path) -> None:
    client = VLLMClient(base_url="http://mock/v1")
    with pytest.raises(VLLMClientError):
        await client.generate("hi")


async def test_generate_retries_on_5xx_then_succeeds(isolated_env: Path) -> None:
    attempts: list[int] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        if len(attempts) < 2:
            return httpx.Response(503, text="busy")
        return httpx.Response(200, json=_completion_payload("ok"))

    transport = httpx.MockTransport(handler)
    client = await _make_client(transport)
    try:
        r = await client.generate("hi")
        assert r.text == "ok"
        assert len(attempts) == 2
    finally:
        await client._client.aclose()  # type: ignore[union-attr]


async def test_generate_does_not_retry_on_4xx(isolated_env: Path) -> None:
    attempts: list[int] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(1)
        return httpx.Response(400, text="bad prompt")

    transport = httpx.MockTransport(handler)
    client = await _make_client(transport)
    try:
        with pytest.raises(VLLMClientError):
            await client.generate("hi")
        assert len(attempts) == 1
    finally:
        await client._client.aclose()  # type: ignore[union-attr]


async def test_generate_raises_timeout_after_retries(isolated_env: Path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    transport = httpx.MockTransport(handler)
    client = VLLMClient(base_url="http://mock/v1", max_retries=1, timeout_s=0.01)
    client._client = httpx.AsyncClient(transport=transport, timeout=0.01)  # type: ignore[assignment]
    try:
        with pytest.raises(VLLMTimeoutError):
            await client.generate("hi")
    finally:
        await client._client.aclose()  # type: ignore[union-attr]


async def test_mock_backend_returns_canned_response(
    monkeypatch: pytest.MonkeyPatch, isolated_env: Path
) -> None:
    monkeypatch.setenv("LLMEVAL_MOCK_BACKEND", "true")
    get_settings.cache_clear()
    async with VLLMClient() as client:
        result = await client.generate("hello world", max_tokens=5)
    assert "[mock completion for prompt" in result.text
    assert result.finish_reason == "stop"


async def test_stream_generate_mock_yields_chunks(
    monkeypatch: pytest.MonkeyPatch, isolated_env: Path
) -> None:
    monkeypatch.setenv("LLMEVAL_MOCK_BACKEND", "true")
    get_settings.cache_clear()
    async with VLLMClient() as client:
        deltas: list[str] = []
        last_reason: str | None = None
        async for chunk in client.stream_generate("hello", max_tokens=5):
            deltas.append(chunk.delta)
            last_reason = chunk.finish_reason
    assert len(deltas) >= 1
    assert last_reason == "stop"
