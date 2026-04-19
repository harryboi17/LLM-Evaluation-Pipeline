"""Structured logging for the project.

Every module uses :func:`get_logger` and never calls :mod:`structlog` or stdlib
:mod:`logging` directly. Configuration is applied exactly once on first call.

The log format is controlled by :attr:`common.config.Settings.log_format`:

* ``console`` — human-friendly, coloured output (default for dev).
* ``json`` — one-line JSON per event (for CI, log aggregation, grep).

A small :func:`timed` decorator emits a ``timing`` event with ``elapsed_ms`` around any
function (sync or async). Apply it to perf-critical code paths.
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, ParamSpec, TypeVar, cast

import structlog

from common.config import get_settings

P = ParamSpec("P")
R = TypeVar("R")

_CONFIGURED: bool = False


def configure_logging() -> None:
    """Configure ``structlog`` and stdlib logging. Idempotent.

    Reads :class:`common.config.Settings` for level and format. Subsequent calls
    after the first are no-ops.
    """
    global _CONFIGURED  # noqa: PLW0603 — module-level one-shot flag
    if _CONFIGURED:
        return

    settings = get_settings()
    level_name = settings.log_level.upper()
    level = getattr(logging, level_name, logging.INFO)

    # Route stdlib logging to stderr so third-party libraries (httpx, vllm) share
    # the project's log stream.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=level,
        force=True,
    )

    processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    if settings.log_format == "json":
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()))

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )
    _CONFIGURED = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound ``structlog`` logger, configuring logging on first call.

    Args:
        name: Optional logger name; typically ``__name__`` of the caller.

    Returns:
        A structlog ``BoundLogger`` ready for use.
    """
    if not _CONFIGURED:
        configure_logging()
    if name is None:
        return cast(structlog.stdlib.BoundLogger, structlog.get_logger())
    return cast(structlog.stdlib.BoundLogger, structlog.get_logger(name))


def timed(label: str | None = None) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator that logs wall-clock duration of a function.

    Works for both sync and ``async`` functions. The emitted event is ``timing``
    with keys ``event`` (the label) and ``elapsed_ms``.

    Args:
        label: Event label. Defaults to the wrapped function's ``__qualname__``.

    Returns:
        A decorator preserving the wrapped callable's signature.
    """

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        event = label or fn.__qualname__

        if asyncio.iscoroutinefunction(fn):

            @wraps(fn)
            async def awrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                log = get_logger("common.timed")
                start = time.perf_counter_ns()
                try:
                    return await fn(*args, **kwargs)  # type: ignore[no-any-return]
                finally:
                    elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
                    log.info("timing", label=event, elapsed_ms=round(elapsed_ms, 3))

            return cast(Callable[P, R], awrapper)

        @wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            log = get_logger("common.timed")
            start = time.perf_counter_ns()
            try:
                return fn(*args, **kwargs)
            finally:
                elapsed_ms = (time.perf_counter_ns() - start) / 1_000_000
                log.info("timing", label=event, elapsed_ms=round(elapsed_ms, 3))

        return wrapper

    return decorator


def reset_for_tests() -> None:
    """Reset the one-shot configuration flag. Intended for use from tests only."""
    global _CONFIGURED  # noqa: PLW0603 — module-level one-shot flag
    _CONFIGURED = False


__all__: list[str] = [
    "configure_logging",
    "get_logger",
    "reset_for_tests",
    "timed",
]


# Type alias for ``typing.Any`` — not used at runtime but kept for mypy completeness.
_ = Any
