import math

import numpy as np
import pytest

from usel.solvers import (
    euler,
    improved_euler,
    rk2,
    rk4,
)


def decay(t, y):
    return -y


solvers = [
    euler,
    improved_euler,
    rk2,
    rk4,
]


def get_final_value(result):
    """
    Handles different solver output formats.
    """

    result = np.asarray(result)

    if result.ndim == 2:
        return result[-1, -1]

    return result[-1]


@pytest.mark.parametrize("solver", solvers)
def test_solver_runs(solver):

    result = solver(decay, t0=0.0, y0=1.0, t_end=1.0, h=0.01)

    assert result is not None


@pytest.mark.parametrize("solver", solvers)
def test_decay_solution_accuracy(solver):

    result = solver(decay, t0=0.0, y0=1.0, t_end=1.0, h=0.001)

    numerical = get_final_value(result)

    expected = math.exp(-1)

    error = abs(numerical - expected)

    assert float(error) < 0.01


def test_rk4_accuracy():

    euler_result = euler(decay, t0=0, y0=1, t_end=1, h=0.01)

    rk4_result = rk4(decay, t0=0, y0=1, t_end=1, h=0.01)

    expected = math.exp(-1)

    euler_error = abs(get_final_value(euler_result) - expected)

    rk4_error = abs(get_final_value(rk4_result) - expected)

    assert float(rk4_error) < float(euler_error)
