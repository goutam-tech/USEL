import importlib.util
from pathlib import Path

import usel

ROOT = Path(__file__).resolve().parents[3]


def _load_checker():
    spec = importlib.util.spec_from_file_location(
        "check_version", ROOT / "scripts" / "check_version.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_repository_versions_are_consistent():
    checker = _load_checker()

    assert checker.check(ROOT) == []
    assert checker.package_version(ROOT) == usel.__version__


def _make_project(tmp_path, pyproject="1.2.0", package="1.2.0", changelog="1.2.0"):
    (tmp_path / "src" / "usel").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(f'[project]\nversion = "{pyproject}"\n')
    (tmp_path / "src" / "usel" / "__init__.py").write_text(f'__version__ = "{package}"\n')
    (tmp_path / "CHANGELOG.md").write_text(f"# Changelog\n\n## [{changelog}]\n")
    return tmp_path


def test_detects_pyproject_mismatch(tmp_path):
    checker = _load_checker()
    root = _make_project(tmp_path, pyproject="1.0.9")

    problems = checker.check(root)

    assert len(problems) == 1
    assert "pyproject.toml" in problems[0]


def test_detects_missing_changelog_entry(tmp_path):
    checker = _load_checker()
    root = _make_project(tmp_path, changelog="1.1.0")

    problems = checker.check(root)

    assert any("CHANGELOG.md" in p for p in problems)


def test_changelog_has_unreleased_section():
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")

    assert "## [Unreleased]" in text
