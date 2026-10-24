import json

import pytest

from usel.exceptions import ValidationError
from usel.utils.benchmarking import (
    BenchmarkResult,
    environment_info,
    format_table,
    results_to_dict,
    run_benchmark,
    write_json,
)


def _work(n):
    return sum(range(n))


def test_run_benchmark_basic_statistics():
    result = run_benchmark("sum", _work, setup=lambda: (1000,), repeats=4, warmup=1)

    assert isinstance(result, BenchmarkResult)
    assert result.name == "sum"
    assert len(result.times_s) == 4
    assert result.min_s <= result.median_s
    assert result.min_s <= result.mean_s
    assert result.std_s >= 0.0


def test_setup_is_called_for_every_run_and_not_timed():
    calls = []

    def setup():
        calls.append(1)
        return (10,)

    run_benchmark("sum", _work, setup=setup, repeats=3, warmup=2, track_memory=True)

    # 2 warm-ups + 3 timed repeats + 1 memory run
    assert len(calls) == 6


def test_no_setup_calls_function_without_args():
    result = run_benchmark("noop", lambda: None, repeats=2, warmup=0, track_memory=False)

    assert result.peak_memory_mb is None
    assert result.warmup == 0


def test_single_repeat_has_zero_std():
    result = run_benchmark("noop", lambda: None, repeats=1, warmup=0)

    assert result.std_s == 0.0


def test_peak_memory_detects_allocation():
    result = run_benchmark("alloc", lambda: bytearray(5 * 1024 * 1024), repeats=1, warmup=0)

    assert result.peak_memory_mb is not None
    assert result.peak_memory_mb >= 4.0


def test_tracemalloc_state_restored():
    import tracemalloc

    assert not tracemalloc.is_tracing()
    run_benchmark("noop", lambda: None, repeats=1, warmup=0)
    assert not tracemalloc.is_tracing()


@pytest.mark.parametrize("repeats", [0, -1])
def test_invalid_repeats(repeats):
    with pytest.raises(ValidationError):
        run_benchmark("x", lambda: None, repeats=repeats)


def test_invalid_warmup():
    with pytest.raises(ValidationError):
        run_benchmark("x", lambda: None, warmup=-1)


def test_environment_info_keys():
    env = environment_info()

    for key in ("usel_version", "python", "platform", "cpu_count", "numpy", "scipy"):
        assert key in env


def test_params_are_recorded_and_copied():
    params = {"size": 5}
    result = run_benchmark("x", lambda: None, params=params, repeats=1, warmup=0)
    params["size"] = 99

    assert result.params == {"size": 5}


def test_json_roundtrip(tmp_path):
    result = run_benchmark("x", lambda: None, params={"n": 1}, repeats=2, warmup=0)
    path = write_json([result], tmp_path / "nested" / "bench.json")

    data = json.loads(path.read_text(encoding="utf-8"))

    assert data == json.loads(json.dumps(results_to_dict([result])))
    assert data["results"][0]["name"] == "x"
    assert len(data["results"][0]["times_s"]) == 2
    assert "environment" in data


def test_format_table_contains_name_and_params():
    with_mem = run_benchmark("matmul", lambda: None, params={"size": 7}, repeats=1, warmup=0)
    no_mem = run_benchmark("plain", lambda: None, repeats=1, warmup=0, track_memory=False)

    table = format_table([with_mem, no_mem])

    assert "matmul[size=7]" in table
    assert "plain" in table
    assert "n/a" in table
