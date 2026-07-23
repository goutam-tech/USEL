import pytest

from usel.cli.main import build_parser, main


def test_version_command(capsys):

    result = main(["version"])

    captured = capsys.readouterr()

    assert result == 0
    assert "usel version" in captured.out


def test_info_command(capsys):

    result = main(["info"])

    captured = capsys.readouterr()

    assert result == 0
    assert "usel Info" in captured.out


def test_parser_creation():

    parser = build_parser()

    assert parser is not None


def test_invalid_command():

    with pytest.raises(SystemExit):
        main(["invalid"])
