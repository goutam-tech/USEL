"""Smoke tests for the USEL benchmark suite.

These confirm the benchmark scripts complete, emit valid JSON/CSV with the
required metadata fields, and tolerate run-to-run timing variation without
asserting identical timings. Numerical correctness assertions live in the
unit-test suite; benchmark execution is not treated as proof of correctness.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

# ``benchmarks/`` lives at the repo root and is not an installed package, so
# add the repo root to ``sys.path`` to import the benchmark suite directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pytest

from benchmarks.bench_suite import WORKLOADS, environment, main, measure, run_benchmark

WORKLOAD_NAMES = [name for name, _ in WORKLOADS]


@pytest.mark.parametrize("name", WORKLOAD_NAMES)
def test_each_workload_completes_and_emits_valid_output(tmp_path, name):
    out_dir = tmp_path / "results"
    argv = ["--out", str(out_dir), "--repeats", "2", "--workload", name]
    assert main(argv) == 0

    data = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))

    # Required top-level metadata.
    assert "environment" in data
    assert "benchmarks" in data
    assert len(data["benchmarks"]) == 1

    bench = data["benchmarks"][0]
    assert bench["name"] == name
    assert isinstance(bench["parameters"], dict)
    assert bench["repeats"] == 2
    assert isinstance(bench["median_seconds"], float)
    assert isinstance(bench["stdev_seconds"], float)
    assert isinstance(bench["peak_memory_mb"], float)

    env = data["environment"]
    for key in ("python", "platform", "numpy", "scipy", "utc_timestamp"):
        assert key in env, f"missing environment field {key}"

    # CSV mirrors JSON with required columns.
    rows = list(csv.DictReader((out_dir / "results.csv").open(encoding="utf-8")))
    assert len(rows) == 1
    for col in ("name", "median_seconds", "stdev_seconds", "peak_memory_mb"):
        assert col in rows[0]


def test_same_workload_twice_tolerates_variation(tmp_path):
    first = run_benchmark(name="fft", repeats=3)
    second = run_benchmark(name="fft", repeats=3)

    assert len(first) == len(second) == 1
    # Both complete successfully; timings need not be identical (noise).
    assert first[0]["median_seconds"] > 0
    assert second[0]["median_seconds"] > 0
    assert first[0]["peak_memory_mb"] >= 0


def test_environment_collects_metadata():
    env = environment()
    assert env["python"]
    assert env["platform"]
    assert env["numpy"]
    assert env["scipy"]


def test_measure_reports_median_and_spread():
    def fn():
        total = 0.0
        for i in range(1000):
            total += i * i
        return total

    median, stdev, peak = measure(fn, repeats=4, warmup=1)
    assert median > 0
    assert stdev >= 0
    assert peak >= 0
