"""usel command line interface.

Commands
--------
usel version     Print the installed usel version.
usel doctor       Check environment health (Python version, dependencies).
usel benchmark    Run a quick matrix-multiplication benchmark.
usel info         Print platform and package information.
usel config       Show or validate a configuration file.
usel test         Run the test suite via pytest.
"""

from __future__ import annotations

import argparse
import importlib
import platform
import subprocess
import sys
import time
from collections.abc import Sequence

import usel
from usel.math.matrix import Matrix

_REQUIRED_PACKAGES = ("numpy", "scipy", "sympy")


def _cmd_version(_: argparse.Namespace) -> int:
    print(f"usel version {usel.__version__}")
    return 0


def _cmd_info(_: argparse.Namespace) -> int:
    print("usel Info")
    print("-" * 40)
    print(f"Version:        {usel.__version__}")
    print(f"License:        {usel.__license__}")
    print(f"Python:         {platform.python_version()}")
    print(f"Platform:       {platform.platform()}")
    print(f"Processor:      {platform.processor() or 'unknown'}")
    return 0


def _cmd_doctor(_: argparse.Namespace) -> int:
    print("Running usel environment check...")
    ok = True

    major, minor = sys.version_info[:2]
    if (major, minor) >= (3, 11):
        print(f"[OK]   Python version {major}.{minor} >= 3.11")
    else:
        print(f"[FAIL] Python version {major}.{minor} < 3.11")
        ok = False

    for package in _REQUIRED_PACKAGES:
        try:
            importlib.import_module(package)
            print(f"[OK]   Dependency '{package}' is importable")
        except ImportError:
            print(f"[FAIL] Dependency '{package}' is missing")
            ok = False

    if ok:
        print("\nAll checks passed. usel is ready to use.")
    else:
        print("\nSome checks failed. Please review the output above.")
    return 0 if ok else 1


def _cmd_benchmark(args: argparse.Namespace) -> int:
    size = args.size
    print(f"Benchmarking {size}x{size} matrix multiplication...")
    a = Matrix.random(size, size, seed=42)
    b = Matrix.random(size, size, seed=43)

    start = time.perf_counter()
    _ = a @ b
    elapsed = time.perf_counter() - start

    print(f"Completed in {elapsed:.6f} seconds")
    return 0


def _cmd_config(args: argparse.Namespace) -> int:
    from usel.config.loader import load_config

    try:
        data = load_config(args.path)
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] Could not load configuration: {exc}")
        return 1

    print(f"[OK] Configuration file '{args.path}' loaded successfully")
    print(f"Keys: {sorted(data.keys())}")
    return 0


def _cmd_test(args: argparse.Namespace) -> int:
    cmd = [sys.executable, "-m", "pytest"]
    if args.path:
        cmd.append(args.path)
    result = subprocess.run(cmd, check=False)  # noqa: S603
    return result.returncode


def build_parser() -> argparse.ArgumentParser:
    """Build and return the usel argument parser."""
    parser = argparse.ArgumentParser(
        prog="usel", description="usel: CPU-only scientific computing framework"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("version", help="Print the installed usel version").set_defaults(
        func=_cmd_version
    )
    subparsers.add_parser("info", help="Print platform and package information").set_defaults(
        func=_cmd_info
    )
    subparsers.add_parser(
        "doctor", help="Check environment health (Python version, dependencies)"
    ).set_defaults(func=_cmd_doctor)

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="Run a quick matrix-multiplication benchmark"
    )
    benchmark_parser.add_argument(
        "--size", type=int, default=200, help="Matrix dimension for the benchmark (default: 200)"
    )
    benchmark_parser.set_defaults(func=_cmd_benchmark)

    config_parser = subparsers.add_parser("config", help="Load and validate a configuration file")
    config_parser.add_argument("path", help="Path to a .yaml, .json, or .toml configuration file")
    config_parser.set_defaults(func=_cmd_config)

    test_parser = subparsers.add_parser("test", help="Run the test suite via pytest")
    test_parser.add_argument("path", nargs="?", default=None, help="Optional test path filter")
    test_parser.set_defaults(func=_cmd_test)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """usel CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
