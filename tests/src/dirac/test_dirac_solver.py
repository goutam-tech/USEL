import numpy as np
import pytest

from usel.dirac.solver import DiracSolver
from usel.exceptions import ValidationError


def create_grid():

    return np.linspace(-1, 1, 10)


def create_spinor(n):

    psi = np.zeros((4, n), dtype=complex)

    psi[0, n // 2] = 1

    return psi


def test_solver_initialization():

    x = create_grid()

    solver = DiracSolver(x)

    assert solver.x.shape == (10,)

    assert solver.mass == 1


def test_invalid_grid_dimension():

    with pytest.raises(ValidationError):
        DiracSolver(np.ones((2, 2)))


def test_build_hamiltonian_shape():

    x = create_grid()

    solver = DiracSolver(x)

    H = solver.build_hamiltonian()

    assert H.shape == (40, 40)


def test_custom_potential():

    x = create_grid()

    V = np.ones(len(x))

    solver = DiracSolver(x, potential=V)

    assert np.allclose(solver.potential, V)


def test_energy_relation():

    solver = DiracSolver(create_grid())

    E = solver.energy(momentum=2)

    expected = np.sqrt(2**2 + 1)

    assert np.isclose(E, expected)


def test_solver_output_shape():

    x = create_grid()

    solver = DiracSolver(x)

    psi = create_spinor(len(x))

    result = solver.solve(psi, t_end=0.1, dt=0.05)

    assert result.spinor.shape[1:] == (4, len(x))

    assert len(result.t) == 3
