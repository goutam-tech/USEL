import pytest

from usel.utils.precision import (
    get_global_precision,
    round_to_precision,
    set_global_precision,
)


def test_default_precision():

    precision = get_global_precision()

    assert isinstance(precision, int)


def test_set_precision():

    set_global_precision(4)

    assert get_global_precision() == 4


def test_round_with_custom_precision():

    value = round_to_precision(3.141592, 3)

    assert value == 3.142


def test_round_with_global_precision():

    set_global_precision(2)

    value = round_to_precision(3.14159)

    assert value == 3.14


def test_negative_precision():

    with pytest.raises(ValueError):
        set_global_precision(-1)
