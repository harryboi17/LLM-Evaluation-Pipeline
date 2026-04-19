"""Shared library for the LLM Evaluation System.

Modules here are imported by every Part (A-E). They provide:

- :mod:`common.config` — application settings via pydantic-settings
- :mod:`common.logging` — structured logging configuration and helpers
- :mod:`common.errors` — custom exception hierarchy
- :mod:`common.types` — cross-package dataclasses and TypedDicts
- :mod:`common.cache` — SQLite-backed prompt cache
- :mod:`common.vllm_client` — async httpx client for the vLLM OpenAI endpoint
- :mod:`common.stats` — paired-bootstrap helpers for benchmark comparisons
"""

from __future__ import annotations

__all__ = [
    "__version__",
]

__version__ = "0.1.0"
