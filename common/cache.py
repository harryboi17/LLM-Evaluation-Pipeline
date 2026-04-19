"""SQLite-backed prompt cache for deterministic, cheap reruns.

Cache keys are ``sha256(json({model, prompt, params}))`` so identical requests —
regardless of call order or timing — share a single entry. The store is a single
SQLite file (no external server), safe for concurrent reads and serialized
writes via a per-instance lock.

Callers should instantiate :class:`PromptCache` once per process and share it
across requests.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from common.config import get_settings
from common.errors import CacheError
from common.logging import get_logger

log = get_logger(__name__)


_SCHEMA: str = """
CREATE TABLE IF NOT EXISTS prompt_cache (
    key         TEXT PRIMARY KEY,
    model       TEXT NOT NULL,
    prompt      TEXT NOT NULL,
    params      TEXT NOT NULL,
    response    TEXT NOT NULL,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cache_model ON prompt_cache (model);
"""


def _compute_key(model: str, prompt: str, params: dict[str, Any]) -> str:
    """Return a stable hex digest for ``(cache_version, model, prompt, params)``.

    The active :attr:`common.config.Settings.cache_version` is mixed into the
    key so bumping ``LLMEVAL_CACHE_VERSION`` invalidates every stored entry
    without needing to delete the SQLite file. This is the escape hatch when
    the serving config (dtype, max_model_len, vLLM release) changes in a way
    the ``model`` field alone doesn't capture.

    Args:
        model: Model identifier (e.g., ``meta-llama/Llama-3.2-1B-Instruct``).
        prompt: The exact prompt string sent to the model.
        params: Decoding parameters dict. Keys are sorted for stability.

    Returns:
        A hex-encoded SHA-256 digest.
    """
    payload = json.dumps(
        {
            "cache_version": get_settings().cache_version,
            "model": model,
            "prompt": prompt,
            "params": params,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class PromptCache:
    """Thread-safe SQLite cache keyed by ``(model, prompt, decoding params)``.

    The underlying file is created on first use; callers do not need to
    pre-provision it.
    """

    def __init__(self, path: Path | None = None) -> None:
        """Initialize the cache.

        Args:
            path: Explicit cache file path. When ``None`` (default), uses
                ``Settings.cache_dir / prompt_cache.sqlite``.
        """
        self._path = path or (get_settings().cache_dir / "prompt_cache.sqlite")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Yield an autocommit SQLite connection."""
        conn = sqlite3.connect(self._path, isolation_level=None)
        try:
            yield conn
        finally:
            conn.close()

    def get(
        self,
        model: str,
        prompt: str,
        params: dict[str, Any],
    ) -> dict[str, Any] | None:
        """Return a cached response or ``None``.

        Args:
            model: Model identifier.
            prompt: The exact prompt.
            params: Decoding parameters.

        Returns:
            The cached response dict, or ``None`` on miss.

        Raises:
            CacheError: If the cached entry is corrupt JSON.
        """
        key = _compute_key(model, prompt, params)
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT response FROM prompt_cache WHERE key = ?", (key,)).fetchone()
        if row is None:
            log.debug("cache_miss", key=key[:12])
            return None
        try:
            result: dict[str, Any] = json.loads(row[0])
        except json.JSONDecodeError as exc:
            raise CacheError(f"corrupt cache entry for key={key}") from exc
        log.debug("cache_hit", key=key[:12])
        return result

    def put(
        self,
        model: str,
        prompt: str,
        params: dict[str, Any],
        response: dict[str, Any],
    ) -> None:
        """Insert or replace a cache entry.

        Args:
            model: Model identifier.
            prompt: The exact prompt.
            params: Decoding parameters.
            response: The response dict to cache (must be JSON-serializable).
        """
        key = _compute_key(model, prompt, params)
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO prompt_cache "
                "(key, model, prompt, params, response, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    key,
                    model,
                    prompt,
                    json.dumps(params, sort_keys=True),
                    json.dumps(response),
                    time.time(),
                ),
            )
        log.debug("cache_put", key=key[:12])

    def size(self) -> int:
        """Return the number of cached entries."""
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT COUNT(*) FROM prompt_cache").fetchone()
        return int(row[0])

    def clear(self) -> None:
        """Remove every cache entry (intended for tests)."""
        with self._lock, self._connect() as conn:
            conn.execute("DELETE FROM prompt_cache")


__all__ = ["PromptCache"]
