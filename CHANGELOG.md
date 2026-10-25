# Changelog

All notable changes to USEL are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [1.1.2] - 2026-10-25

### Added
- Added improvements to package version consistency validation.

### Changed
- Updated the release workflow to run after `USEL CI` succeeds on `main`.
- Pinned third-party GitHub Actions to full commit SHAs.

### Fixed
- Fixed package version consistency between `pyproject.toml` and `usel.__version__`.

## [1.1.1] - 2026-10-25

### Fixed
- Synchronized the package version with `pyproject.toml`.
- Improved package version consistency checks.

## [1.1.0] - 2026-10-25

### Added
- Added version consistency checks for project metadata, package version, and changelog sections.

### Changed
- Updated the project version to 1.1.0.

## [1.0.9]