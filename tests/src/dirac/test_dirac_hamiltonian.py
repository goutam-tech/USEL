import numpy as np

from usel.dirac.hamiltonian import DiracHamiltonian


def test_free_hamiltonian_shape():

    x = np.linspace(-1, 1, 10)

    H = DiracHamiltonian(x)

    matrix = H.matrix()

    assert matrix.shape == (40, 40)


def test_hamiltonian_contains_mass_term():

    x = np.linspace(0, 1, 5)

    mass = 2

    H = DiracHamiltonian(x, mass=mass)

    matrix = H.matrix()

    assert matrix.shape == (20, 20)


def test_hamiltonian_with_potential():

    x = np.linspace(0, 1, 5)

    V = np.ones(5)

    H = DiracHamiltonian(x, potential=V)

    matrix = H.matrix()

    assert matrix.shape == (20, 20)


def test_hamiltonian_matrix_complex():

    x = np.linspace(0, 1, 6)

    H = DiracHamiltonian(x)

    matrix = H.matrix()

    assert np.iscomplexobj(matrix)
