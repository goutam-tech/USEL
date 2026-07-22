"""Console and file loggers, debug mode toggling, and a performance logger."""

from __future__ import annotations

import logging
import time
from pathlib import Path

_DEFAULT_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"


def get_console_logger(name: str = "usel", level: int = logging.INFO) -> logging.Logger:
    """Return a logger configured to write formatted messages to the console."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        logger.addHandler(handler)
    return logger


def get_file_logger(
    name: str = "usel", filepath: str | Path = "usel.log", level: int = logging.INFO
) -> logging.Logger:
    """Return a logger configured to write formatted messages to a file."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    path = Path(filepath)
    if not any(
        isinstance(h, logging.FileHandler) and h.baseFilename == str(path.resolve())
        for h in logger.handlers
    ):
        handler = logging.FileHandler(path)
        handler.setFormatter(logging.Formatter(_DEFAULT_FORMAT))
        logger.addHandler(handler)
    return logger


def set_debug_mode(logger: logging.Logger, enabled: bool = True) -> None:
    """Toggle debug-level verbosity on the given logger."""
    logger.setLevel(logging.DEBUG if enabled else logging.INFO)


class PerformanceLogger:
    """A simple named-checkpoint performance logger.

    Examples
    --------
    >>> perf = PerformanceLogger()
    >>> perf.checkpoint("start")
    >>> perf.checkpoint("end")
    >>> perf.report()['start'] >= 0
    True
    """

    def __init__(self) -> None:
        self._checkpoints: dict[str, float] = {}
        self._start = time.perf_counter()

    def checkpoint(self, label: str) -> float:
        """Record a checkpoint and return elapsed seconds since construction."""
        elapsed = time.perf_counter() - self._start
        self._checkpoints[label] = elapsed
        return elapsed

    def report(self) -> dict[str, float]:
        """Return a mapping of checkpoint labels to elapsed seconds."""
        return dict(self._checkpoints)

    def reset(self) -> None:
        """Reset the performance logger's start time and checkpoints."""
        self._checkpoints.clear()
        self._start = time.perf_counter()
