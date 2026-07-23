import pytest

from usel.exceptions import ValidationError
from usel.utils.validation import (
    validate_positive,
    validate_range,
    validate_shape,
    validate_type,
)


def test_validate_positive_success():

    validate_positive(10)


def test_validate_positive_failure():

    with pytest.raises(ValidationError):
        validate_positive(-1)


def test_validate_range_success():

    validate_range(5, 1, 10)


def test_validate_range_failure():

    with pytest.raises(ValidationError):
        validate_range(20, 1, 10)


def test_validate_shape_success():

    validate_shape((2, 3), (2, 3))


def test_validate_shape_failure():

    with pytest.raises(ValidationError):
        validate_shape((2, 3), (3, 3))


def test_validate_type_success():

    validate_type(10, int)


def test_validate_type_failure():

    with pytest.raises(ValidationError):
        validate_type("hello", int)
