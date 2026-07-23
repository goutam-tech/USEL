"""Lightweight profiling helpers: function profiling and memory usage."""

from __future__ import annotations

import cProfile
import io
import os
import pstats
from collections.abc import Callable
from typing import Any, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def profile(func: F, *args: Any, **kwargs: Any) -> tuple[Any, str]:
    """Run ``func(*args, **kwargs)`` under cProfile.

    Returns a tuple of ``(result, report_text)`` where ``report_text`` is a
    human-readable profiling report sorted by cumulative time.
    """
    profiler = cProfile.Profile()
    profiler.enable()
    result = func(*args, **kwargs)
    profiler.disable()

    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream).sort_stats("cumulative")
    stats.print_stats(20)
    return result, stream.getvalue()


def memory_usage_mb() -> float:
    """Return the current process's resident memory usage in megabytes.

    Falls back to ``resource`` (POSIX) if ``psutil`` is unavailable, and
    returns ``0.0`` if neither is available (e.g. unsupported platform).
    """
    try:
        import psutil

        process = psutil.Process(os.getpid())
        return process.memory_info().rss / (1024 * 1024)
    except ImportError:
        pass

    try:
        import resource
        import sys

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # Linux reports KB; macOS reports bytes.
        if sys.platform == "darwin":
            return usage / (1024 * 1024)
        return usage / 1024
    except (ImportError, ModuleNotFoundError):
        return 0.0
