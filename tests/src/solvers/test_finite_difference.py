import numpy as np
from usel.solvers.finite_difference import solve


def test_finite_difference_returns_tuple(monkeypatch):
    class DummyMatrix:
        def __init__(self):

            self.matrix = np.eye(5)

    def fake_build(grid, potential, mass, hbar):
        return DummyMatrix()

    monkeypatch.setattr("usel.solvers.finite_difference.build_hamiltonian", fake_build)

    def fake_eigsh(matrix, k, which):
        energies = np.array([3, 1, 2])
        states = np.eye(5, 3)

        return energies, states

    monkeypatch.setattr("usel.solvers.finite_difference.eigsh", fake_eigsh)

    energies, states = solve(grid=np.linspace(0, 1, 5), potential=lambda x: 0, levels=3)

    assert len(energies) == 3
    assert states.shape == (5, 3)


def test_energy_ordering(monkeypatch):
    class Dummy:
        matrix = np.eye(4)

    monkeypatch.setattr("usel.solvers.finite_difference.build_hamiltonian", lambda *args: Dummy())

    monkeypatch.setattr(
        "usel.solvers.finite_difference.eigsh",
        lambda *args, **kwargs: (np.array([5, 1, 3]), np.eye(4, 3)),
    )

    energies, _ = solve(np.arange(4), lambda x: 0, levels=3)

    assert list(energies) == [1, 3, 5]
