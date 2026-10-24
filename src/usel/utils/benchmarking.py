"""Reproducible CPU-time and memory benchmarking helpers.

The helpers in this module make benchmark runs comparable between machines
and between runs:

* inputs are built by a ``setup`` callable *outside* the timed region, using
  fixed seeds supplied by the caller;
* every benchmark performs warm-up runs, then several timed repeats, and
  reports min / mean / median / standard deviation;
* peak Python-level memory is measured in a separate run with
  :mod:`tracemalloc`, so memory tracking never skews the timings;
* the execution environment (Python, NumPy, SciPy, platform, CPU count) is
  captured next to the results so that numbers can be interpreted later.

Only the standard library and NumPy/SciPy (already USEL dependencies) are
required.
"""

from __future__ import annotations

import json
import os
import platform
import statistics
import time
import tracemalloc
from collections.abc import Callable, Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from usel.exceptions import ValidationError
from usel.utils.profiling import memory_usage_mb

__all__ = [
    "BenchmarkResult",
    "environment_info",
    "run_benchmark",
    "format_table",
    "results_to_dict",
    "write_json",
]

_MB = 1024 * 1024


@dataclass(frozen=True)
class BenchmarkResult:
    """Outcome of a single benchmark.

    Attributes
    ----------
    name:
        Benchmark name, e.g. ``"matmul"``.
    params:
        Parameters that identify the case, e.g. ``{"size": 200}``.
    repeats, warmup:
        Number of timed repeats and untimed warm-up runs that were performed.
    times_s:
        Wall-clock time of every timed repeat, in seconds.
    min_s, mean_s, median_s, std_s:
        Summary statistics of ``times_s`` (``std_s`` is ``0.0`` for one repeat).
    peak_memory_mb:
        Peak Python-allocated memory (including NumPy buffers) during one
        extra run, in megabytes. ``None`` when memory tracking was disabled.
    rss_mb:
        Resident set size of the process after the benchmark, in megabytes.
        ``0.0`` if the platform does not expose it.
    """

    name: str
    params: dict[str, Any] = field(default_factory=dict)
    repeats: int = 1
    warmup: int = 0
    times_s: tuple[float, ...] = ()
    min_s: float = 0.0
    mean_s: float = 0.0
    median_s: float = 0.0
    std_s: float = 0.0
    peak_memory_mb: float | None = None
    rss_mb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        data = asdict(self)
        data["times_s"] = list(self.times_s)
        return data


def environment_info() -> dict[str, Any]:
    """Describe the machine and software stack the benchmarks ran on."""
    import numpy as np
    import scipy

    import usel

    return {
        "usel_version": usel.__version__,
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "cpu_count": os.cpu_count(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    }


def run_benchmark(
    name: str,
    func: Callable[..., Any],
    *,
    setup: Callable[[], Sequence[Any]] | None = None,
    params: dict[str, Any] | None = None,
    repeats: int = 5,
    warmup: int = 1,
    track_memory: bool = True,
) -> BenchmarkResult:
    """Benchmark ``func`` and return a :class:`BenchmarkResult`.

    Parameters
    ----------
    name:
        Benchmark name.
    func:
        The callable to measure. It is called as ``func(*args)`` where
        ``args`` is whatever ``setup`` returned (nothing if ``setup`` is
        ``None``).
    setup:
        Optional callable that builds the inputs. It is invoked before every
        run and is **not** timed, so input generation never pollutes results.
        Use fixed seeds inside it to keep runs reproducible.
    params:
        Parameters recorded with the result (they do not affect the run).
    repeats:
        Number of timed runs (must be >= 1).
    warmup:
        Number of untimed warm-up runs (must be >= 0).
    track_memory:
        If true, run ``func`` once more under :mod:`tracemalloc` to record
        peak memory.

    Raises
    ------
    ValidationError
        If ``repeats`` < 1 or ``warmup`` < 0.
    """
    if repeats < 1:
        raise ValidationError(f"repeats must be >= 1, got {repeats}")
    if warmup < 0:
        raise ValidationError(f"warmup must be >= 0, got {warmup}")

    def _args() -> Sequence[Any]:
        return setup() if setup is not None else ()

    for _ in range(warmup):
        func(*_args())

    times: list[float] = []
    for _ in range(repeats):
        args = _args()
        start = time.perf_counter()
        func(*args)
        times.append(time.perf_counter() - start)

    peak_mb: float | None = None
    if track_memory:
        args = _args()
        already_tracing = tracemalloc.is_tracing()
        if not already_tracing:
            tracemalloc.start()
        tracemalloc.reset_peak()
        baseline, _ = tracemalloc.get_traced_memory()
        try:
            func(*args)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            if not already_tracing:
                tracemalloc.stop()
        peak_mb = max(peak - baseline, 0) / _MB

    return BenchmarkResult(
        name=name,
        params=dict(params or {}),
        repeats=repeats,
        warmup=warmup,
        times_s=tuple(times),
        min_s=min(times),
        mean_s=statistics.fmean(times),
        median_s=statistics.median(times),
        std_s=statistics.stdev(times) if len(times) > 1 else 0.0,
        peak_memory_mb=peak_mb,
        rss_mb=memory_usage_mb(),
    )


def results_to_dict(results: Iterable[BenchmarkResult]) -> dict[str, Any]:
    """Bundle results and environment info into one JSON-serialisable dict."""
    return {
        "environment": environment_info(),
        "results": [r.to_dict() for r in results],
    }


def write_json(results: Iterable[BenchmarkResult], path: str | Path) -> Path:
    """Write results (plus environment info) to ``path`` as JSON.

    Parent directories are created when missing. Returns the written path.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = results_to_dict(results)
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return target


def format_table(results: Iterable[BenchmarkResult]) -> str:
    """Render results as a fixed-width text table."""
    header = (
        f"{'Benchmark':<28} | {'min (s)':>10} | {'median (s)':>10} | "
        f"{'std (s)':>10} | {'peak mem (MB)':>13}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        label = r.name
        if r.params:
            label += "[" + ",".join(f"{k}={v}" for k, v in r.params.items()) + "]"
        mem = f"{r.peak_memory_mb:13.3f}" if r.peak_memory_mb is not None else f"{'n/a':>13}"
        lines.append(
            f"{label:<28} | {r.min_s:>10.6f} | {r.median_s:>10.6f} | {r.std_s:>10.6f} | {mem}"
        )
    return "\n".join(lines)
