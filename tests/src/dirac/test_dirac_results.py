import dataclasses

import numpy as np
import pytest

from usel.dirac.results import DiracResult


def create_result():
    spinor = np.ones((5, 4, 10), dtype=complex)

    probability = np.ones((5, 10))

    t = np.arange(5)

    x = np.linspace(0, 1, 10)

    return DiracResult(spinor=spinor, probability=probability, t=t, x=x)


def test_result_creation():
    result = create_result()

    assert result.spinor.shape == (5, 4, 10)


def test_total_probability():
    result = create_result()

    total = result.total_probability

    assert total.shape == (5,)


def test_upper_component():
    result = create_result()

    upper = result.upper_component

    assert upper.shape == (5, 10)


def test_lower_component():
    result = create_result()

    lower = result.lower_component

    assert lower.shape == (5, 10)


def test_result_is_frozen():
    result = create_result()

    with pytest.raises(dataclasses.FrozenInstanceError):
        result.t = np.array([1])
