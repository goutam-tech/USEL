import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_benchmarks", ROOT / "benchmarks" / "run_benchmarks.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_quick_suite_runs_and_writes_json(tmp_path, capsys):
    runner = _load_runner()
    out = tmp_path / "bench.json"

    code = runner.main(["--quick", "--repeats", "2", "--warmup", "0", "--json", str(out)])

    assert code == 0
    assert "matmul[size=50]" in capsys.readouterr().out
    data = json.loads(out.read_text(encoding="utf-8"))
    names = {r["name"] for r in data["results"]}
    assert {"matmul", "fft", "ode_rk4"} <= names


def test_no_memory_flag(tmp_path):
    runner = _load_runner()
    results = runner.build_suite(quick=True, repeats=1, warmup=0, track_memory=False)

    assert all(r.peak_memory_mb is None for r in results)
