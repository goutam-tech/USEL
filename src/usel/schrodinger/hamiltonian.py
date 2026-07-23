"""
Hamiltonian construction.

H = T + V

Used for solving stationary Schrödinger equation.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import eigsh

from .grids import SpatialGrid
from .operators import QuantumOperator, kinetic_energy_operator, potential_operator


class Hamiltonian(QuantumOperator):
    """
    Quantum Hamiltonian operator.

    H = kinetic + potential
    """

    def __init__(self, kinetic: QuantumOperator, potential: QuantumOperator):

        matrix = kinetic.matrix + potential.matrix

        super().__init__(matrix)


def build_hamiltonian(
    grid: SpatialGrid, potential: np.ndarray, mass: float = 1.0, hbar: float = 1.0
) -> Hamiltonian:
    """
    Create Hamiltonian.

    Parameters
    ----------
    grid:
        Spatial grid

    potential:
        Potential array

    mass:
        Particle mass

    hbar:
        Reduced Planck constant
    """

    kinetic = kinetic_energy_operator(grid, mass, hbar)

    potential_op = potential_operator(potential)

    return Hamiltonian(kinetic, potential_op)


def solve_energy_levels(hamiltonian: Hamiltonian, levels: int = 5):
    """
    Solve:

    Hψ = Eψ

    Returns energy eigenvalues
    and eigenstates.
    """

    energies, states = eigsh(hamiltonian.matrix, k=levels, which="SA")

    index = np.argsort(energies)

    return (energies[index], states[:, index])
