"""Load and save configuration files in YAML, JSON, or TOML format.

The format is inferred from the file extension: ``.yaml``/``.yml``,
``.json``, or ``.toml``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from usel.exceptions import ConfigError

_SUPPORTED_SUFFIXES = {".yaml", ".yml", ".json", ".toml"}


def load_config(filepath: str | Path) -> dict[str, Any]:
    """Load a configuration file, inferring the format from its extension."""
    path = Path(filepath)
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")
    if path.suffix not in _SUPPORTED_SUFFIXES:
        raise ConfigError(
            f"Unsupported configuration format '{path.suffix}'. "
            f"Supported formats: {sorted(_SUPPORTED_SUFFIXES)}"
        )

    try:
        text = path.read_text(encoding="utf-8")
        if path.suffix in (".yaml", ".yml"):
            import yaml

            data = yaml.safe_load(text) or {}
        elif path.suffix == ".json":
            data = json.loads(text) if text.strip() else {}
        else:  # .toml
            import toml

            data = toml.loads(text)
    except Exception as exc:  # noqa: BLE001 - re-raised as ConfigError
        raise ConfigError(f"Failed to parse configuration file {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ConfigError(f"Configuration root must be a mapping/object, got {type(data).__name__}")
    return data


def save_config(data: dict[str, Any], filepath: str | Path) -> None:
    """Save a configuration dictionary, inferring the format from its extension."""
    path = Path(filepath)
    if path.suffix not in _SUPPORTED_SUFFIXES:
        raise ConfigError(
            f"Unsupported configuration format '{path.suffix}'. "
            f"Supported formats: {sorted(_SUPPORTED_SUFFIXES)}"
        )

    try:
        if path.suffix in (".yaml", ".yml"):
            import yaml

            path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        elif path.suffix == ".json":
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        else:  # .toml
            import toml

            path.write_text(toml.dumps(data), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - re-raised as ConfigError
        raise ConfigError(f"Failed to write configuration file {path}: {exc}") from exc
