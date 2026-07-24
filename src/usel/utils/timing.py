"""Timing utilities: a context-manager/decorator ``Timer`` and ``timeit`` helper."""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from types import TracebackType
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


class Timer:
    """A context manager and decorator for measuring elapsed wall-clock time.

    Examples
    --------
    >>> with Timer() as t:
    ...     _ = sum(range(1000))
    >>> t.elapsed >= 0
    True
    """

    def __init__(self, label: str | None = None) -> None:
        self.label = label
        self.start_time: float | None = None
        self.elapsed: float = 0.0

    def __enter__(self) -> Timer:
        self.start_time = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        assert self.start_time is not None
        self.elapsed = time.perf_counter() - self.start_time

    def __call__(self, func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with Timer(self.label or func.__name__) as t:
                result = func(*args, **kwargs)
            wrapper.last_elapsed = t.elapsed  # type: ignore[attr-defined]
            return result

        wrapper.last_elapsed = 0.0  # type: ignore[attr-defined]
        return wrapper  # type: ignore[return-value]


def timeit(func: F) -> F:
    """Decorator that measures and returns elapsed time alongside the result.

    The wrapped function returns a tuple ``(result, elapsed_seconds)``.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> tuple[Any, float]:
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed

    return wrapper  # type: ignore[return-value]
