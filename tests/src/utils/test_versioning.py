import importlib.util
import sys
from pathlib import Path

import pytest

import usel

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "check_version.py"
CHANGELOG = ROOT / "CHANGELOG.md"
PYPROJECT = ROOT / "pyproject.toml"

# These tests inspect repository files, so they only make sense in a source checkout.
pytestmark = pytest.mark.skipif(
    not (SCRIPT.exists() and CHANGELOG.exists() and PYPROJECT.exists()),
    reason="repository files not available (not a source checkout)",
)


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_version", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["check_version"] = module
    spec.loader.exec_module(module)
    return module


def _make_project(tmp_path, pyproject="1.2.0", package="1.2.0", changelog=("1.2.0",)):
    (tmp_path / "src" / "usel").mkdir(parents=True)
    (tmp_path / "pyproject.toml").write_text(f'[project]\nversion = "{pyproject}"\n')
    (tmp_path / "src" / "usel" / "__init__.py").write_text(f'__version__ = "{package}"\n')
    headings = "".join(f"## [{v}] - 2026-01-01\n\n" for v in changelog)
    (tmp_path / "CHANGELOG.md").write_text(f"# Changelog\n\n## [Unreleased]\n\n{headings}")
    return tmp_path


def test_repository_versions_are_consistent():
    checker = _load_checker()

    assert checker.check(ROOT) == []
    assert checker.package_version(ROOT) == checker.pyproject_version(ROOT)


def test_imported_package_matches_declared_version():
    # Only meaningful when the imported ``usel`` comes from this checkout; a stale
    # install elsewhere on the machine must not make the test fail.
    checker = _load_checker()
    if not Path(usel.__file__).resolve().is_relative_to(ROOT):
        pytest.skip("usel is imported from outside this checkout")

    assert usel.__version__ == checker.pyproject_version(ROOT)


def test_changelog_has_unreleased_section():
    assert "## [Unreleased]" in CHANGELOG.read_text(encoding="utf-8")


def test_consistent_project_has_no_problems(tmp_path):
    checker = _load_checker()

    assert checker.check(_make_project(tmp_path)) == []


def test_detects_package_version_mismatch(tmp_path):
    checker = _load_checker()
    root = _make_project(tmp_path, package="1.1.0", pyproject="1.0.9", changelog=("1.0.9",))

    problems = checker.check(root)

    assert len(problems) == 1
    assert "usel.__version__" in problems[0]


def test_detects_changelog_mismatch(tmp_path):
    checker = _load_checker()
    root = _make_project(tmp_path, changelog=("1.1.0", "1.0.9"))

    problems = checker.check(root)

    assert any("CHANGELOG.md" in p for p in problems)


def test_older_release_in_changelog_does_not_count(tmp_path):
    checker = _load_checker()
    root = _make_project(tmp_path, pyproject="1.2.0", package="1.2.0", changelog=("1.1.0",))

    assert any("CHANGELOG.md" in p for p in checker.check(root))


def test_changelog_without_release_section(tmp_path):
    checker = _load_checker()
    root = _make_project(tmp_path, changelog=())

    assert any("no released" in p for p in checker.check(root))
