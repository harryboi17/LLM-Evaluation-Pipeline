"""Tests for ``common.cache``."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from common.cache import PromptCache, _compute_key
from common.errors import CacheError


@pytest.fixture
def cache(tmp_path: Path) -> PromptCache:
    return PromptCache(path=tmp_path / "cache.sqlite")


def test_compute_key_is_stable() -> None:
    k1 = _compute_key("m", "p", {"a": 1, "b": 2})
    k2 = _compute_key("m", "p", {"b": 2, "a": 1})  # same content, different order
    assert k1 == k2


def test_compute_key_changes_with_input() -> None:
    k1 = _compute_key("m", "p", {"a": 1})
    k2 = _compute_key("m", "p", {"a": 2})
    k3 = _compute_key("m", "q", {"a": 1})
    k4 = _compute_key("m2", "p", {"a": 1})
    assert len({k1, k2, k3, k4}) == 4


def test_put_then_get_roundtrip(cache: PromptCache) -> None:
    cache.put("m", "hello", {"t": 0.0}, {"text": "world"})
    got = cache.get("m", "hello", {"t": 0.0})
    assert got == {"text": "world"}


def test_miss_returns_none(cache: PromptCache) -> None:
    assert cache.get("m", "never", {"t": 0.0}) is None


def test_replace_existing_entry(cache: PromptCache) -> None:
    cache.put("m", "p", {"t": 0.0}, {"text": "v1"})
    cache.put("m", "p", {"t": 0.0}, {"text": "v2"})
    assert cache.get("m", "p", {"t": 0.0}) == {"text": "v2"}
    assert cache.size() == 1


def test_size_and_clear(cache: PromptCache) -> None:
    assert cache.size() == 0
    for i in range(3):
        cache.put("m", f"p{i}", {}, {"text": str(i)})
    assert cache.size() == 3
    cache.clear()
    assert cache.size() == 0


def test_corrupt_entry_raises_cache_error(cache: PromptCache, tmp_path: Path) -> None:
    cache.put("m", "p", {}, {"text": "ok"})
    # Manually corrupt the stored JSON.
    with sqlite3.connect(tmp_path / "cache.sqlite") as conn:
        conn.execute("UPDATE prompt_cache SET response = '{not json'")
        conn.commit()
    with pytest.raises(CacheError):
        cache.get("m", "p", {})
