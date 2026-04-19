"""Application settings loaded from the environment via ``pydantic-settings``.

All runtime tunables live on :class:`Settings`. Code must read values via
:func:`get_settings` — never read environment variables directly.

Environment variables are read with the ``LLMEVAL_`` prefix. A ``.env`` file at the
project root is loaded automatically (copy :file:`.env.example` to get started).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for the project.

    Attributes:
        model_name: HuggingFace-style model identifier served by vLLM.
        vllm_host: Host on which the vLLM server listens.
        vllm_port: Port on which the vLLM server listens.
        vllm_timeout_s: Per-request timeout in seconds.
        vllm_max_retries: Max retries for transient network errors (capped).
        vllm_api_key: Bearer token for the vLLM OpenAI endpoint (``EMPTY`` by default).
        cache_dir: Directory for the SQLite prompt cache and other on-disk state.
        results_dir: Directory for benchmark outputs.
        log_level: ``structlog`` / stdlib log level (``INFO`` by default).
        log_format: ``console`` for dev, ``json`` for machine-readable logs.
        seed: RNG seed used across all deterministic code paths.
        mock_backend: When ``True``, :class:`common.vllm_client.VLLMClient` returns
            deterministic canned responses without contacting the network.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="LLMEVAL_",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Model / serving ---
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    vllm_host: str = "127.0.0.1"
    vllm_port: int = 8000
    vllm_timeout_s: float = 120.0
    vllm_max_retries: int = 3
    vllm_api_key: str = "EMPTY"

    # --- Paths ---
    cache_dir: Path = Field(default_factory=lambda: Path(".cache"))
    results_dir: Path = Field(default_factory=lambda: Path("results"))

    # --- Logging ---
    log_level: str = "INFO"
    log_format: Literal["json", "console"] = "console"

    # --- Determinism ---
    seed: int = 42

    # --- Mock backend (offline dev) ---
    mock_backend: bool = False

    @property
    def vllm_base_url(self) -> str:
        """Return the OpenAI-compatible base URL (``.../v1``) for the vLLM server."""
        return f"http://{self.vllm_host}:{self.vllm_port}/v1"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached :class:`Settings` singleton.

    The cache ensures the ``.env`` file is parsed exactly once per process.
    Tests can reset the cache via ``get_settings.cache_clear()``.
    """
    return Settings()


__all__ = ["Settings", "get_settings"]
