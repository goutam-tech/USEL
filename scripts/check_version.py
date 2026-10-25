
#!/usr/bin/env python3
"""Check that the version is consistent across the project.

Compares pyproject.toml, src/usel/__init__.py and CHANGELOG.md.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEMVER = r"\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.\-]+)?"


def pyproject_version(root: Path = ROOT) -> str:
    text = (root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(rf'^version\s*=\s*"({SEMVER})"', text, re.MULTILINE)
    if not match:
        raise ValueError("no version found in pyproject.toml")
    return match.group(1)


def package_version(root: Path = ROOT) -> str:
    text = (root / "src" / "usel" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(rf'^__version__\s*=\s*"({SEMVER})"', text, re.MULTILINE)
    if not match:
        raise ValueError("no __version__ found in src/usel/__init__.py")
    return match.group(1)


def changelog_versions(root: Path = ROOT) -> list[str]:
    text = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    return re.findall(rf"^## \[({SEMVER})\]", text, re.MULTILINE)


def check(root: Path = ROOT) -> list[str]:
    """Return a list of problems (empty when everything is consistent)."""
    problems: list[str] = []
    py_ver = pyproject_version(root)
    pkg_ver = package_version(root)
    versions = changelog_versions(root)

    if py_ver != pkg_ver:
        problems.append(
            f"pyproject.toml ({py_ver}) != usel.__version__ ({pkg_ver})"
        )
    else:
        if not versions:
            problems.append("CHANGELOG.md has no released version section")
        elif pkg_ver not in versions:
            problems.append(f"CHANGELOG.md has no '## [{pkg_ver}]' section")

    return problems


def main() -> int:
    try:
        problems = check()
    except (OSError, ValueError) as exc:
        print(f"[FAIL] {exc}")
        return 1

    if problems:
        for problem in problems:
            print(f"[FAIL] {problem}")
        return 1

    print(f"[OK] version {package_version()} is consistent")
    return 0


if __name__ == "__main__":
    sys.exit(main())
