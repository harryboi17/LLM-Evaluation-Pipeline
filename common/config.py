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
        vllm_max_model_len: Override the model's max sequence length. ``None``
            uses the model's own config value.
        vllm_dtype: Precision for model weights / activations. ``auto`` lets vLLM
            pick; ``float16`` / ``bfloat16`` / ``float32`` force a specific dtype.
        vllm_gpu_memory_utilization: Fraction of free GPU memory vLLM is allowed
            to allocate (0.0-1.0).
        vllm_tensor_parallel_size: Number of GPUs for tensor-parallel sharding.
        vllm_max_num_seqs: Max concurrent sequences per step (``None`` = vLLM default).
        vllm_trust_remote_code: Pass ``--trust-remote-code`` to vLLM; only enable
            for models whose HF repo you have audited.
        vllm_download_dir: Override the HuggingFace download cache directory.
        cache_dir: Directory for the SQLite prompt cache and other on-disk state.
        results_dir: Directory for benchmark outputs.
        cache_version: Opaque version string mixed into every prompt-cache key.
            Bump it via ``LLMEVAL_CACHE_VERSION`` to invalidate all cached
            completions when the serving config (dtype, max_model_len, vLLM
            version) changes in a way the model field alone doesn't capture.
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

    # --- vLLM server tunables ---
    vllm_max_model_len: int | None = None
    vllm_dtype: Literal["auto", "float16", "bfloat16", "float32"] = "auto"
    vllm_gpu_memory_utilization: float = 0.9
    vllm_tensor_parallel_size: int = 1
    vllm_max_num_seqs: int | None = None
    vllm_trust_remote_code: bool = False
    vllm_download_dir: Path | None = None

    # --- Paths ---
    cache_dir: Path = Field(default_factory=lambda: Path(".cache"))
    results_dir: Path = Field(default_factory=lambda: Path("results"))

    # --- Cache tuning ---
    cache_version: str = "v1"

    # --- Generation-time overrides (used by improve.sweep) ---
    # When set, the eval_runner vLLM wrapper replaces the task YAML's
    # corresponding gen_kwargs for generate_until calls. Loglikelihood-only
    # tasks ignore these (decoding params don't affect scoring). All three are
    # None by default so task YAMLs keep driving behaviour in normal runs.
    # Env vars: LLMEVAL_GEN_TEMPERATURE, LLMEVAL_GEN_TOP_P, LLMEVAL_GEN_TOP_K.
    gen_temperature: float | None = None
    gen_top_p: float | None = None
    gen_top_k: int | None = None

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
