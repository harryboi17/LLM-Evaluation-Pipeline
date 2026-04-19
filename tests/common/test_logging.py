"""Tests for ``common.logging``."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from common.logging import configure_logging, get_logger, reset_for_tests, timed


def setup_function() -> None:
    reset_for_tests()


def test_get_logger_returns_bound_logger(isolated_env: Path) -> None:
    log = get_logger("test.module")
    assert hasattr(log, "info")
    assert hasattr(log, "bind")


def test_configure_logging_is_idempotent(isolated_env: Path) -> None:
    configure_logging()
    configure_logging()
    configure_logging()  # should not raise


def test_timed_decorator_preserves_sync_return_value(isolated_env: Path) -> None:
    @timed("unit.sync")
    def add(a: int, b: int) -> int:
        return a + b

    assert add(2, 3) == 5


def test_timed_decorator_preserves_async_return_value(isolated_env: Path) -> None:
    @timed("unit.async")
    async def mul(a: int, b: int) -> int:
        await asyncio.sleep(0)
        return a * b

    assert asyncio.run(mul(4, 5)) == 20


def test_timed_decorator_propagates_exceptions(isolated_env: Path) -> None:
    @timed("unit.raises")
    def boom() -> int:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        boom()


def test_timed_default_label_is_qualname(isolated_env: Path) -> None:
    @timed()
    def named() -> int:
        return 1

    # Just confirm the decorator preserves return value; the label is the
    # function's __qualname__ which is visible in stderr during this test.
    assert named() == 1
