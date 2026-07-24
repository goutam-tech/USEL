import pytest

from usel.config.loader import load_config, save_config
from usel.exceptions import ConfigError


def test_load_json_config(tmp_path):

    file = tmp_path / "config.json"

    file.write_text('{"name":"usel","version":1}')

    data = load_config(file)

    assert data["name"] == "usel"
    assert data["version"] == 1


def test_save_json_config(tmp_path):

    file = tmp_path / "output.json"

    save_config({"test": True}, file)

    data = load_config(file)

    assert data["test"] is True


def test_load_yaml_config(tmp_path):

    file = tmp_path / "config.yaml"

    file.write_text("""
        name: usel
        enabled: true
        """)

    data = load_config(file)

    assert data["name"] == "usel"


def test_load_toml_config(tmp_path):

    file = tmp_path / "config.toml"

    file.write_text("""
        name="usel"
        """)

    data = load_config(file)

    assert data["name"] == "usel"


def test_missing_config_file():

    with pytest.raises(ConfigError):
        load_config("missing.json")


def test_invalid_extension(tmp_path):

    file = tmp_path / "config.txt"

    file.write_text("hello")

    with pytest.raises(ConfigError):
        load_config(file)
