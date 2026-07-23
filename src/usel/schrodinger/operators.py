"""
Quantum mechanical operators.

Implements:
- Position operator
- Momentum operator
- Kinetic energy operator
- Potential operator

Using finite difference methods.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix, diags

from .grids import SpatialGrid


class QuantumOperator:
    """
    Base quantum operator.

    Represents an operator matrix acting on wavefunctions.
    """

    def __init__(self, matrix: csr_matrix):
        self.matrix = matrix

    def apply(self, wavefunction: np.ndarray) -> np.ndarray:
        """
        Apply operator to state.
        """

        return self.matrix @ wavefunction

    @property
    def shape(self):
        return self.matrix.shape


def position_operator(grid: SpatialGrid) -> QuantumOperator:
    """
    Position operator.

    x̂ = x
    """

    matrix = diags(grid.x, format="csr")

    return QuantumOperator(matrix)


def momentum_operator(grid: SpatialGrid, hbar: float = 1.0) -> QuantumOperator:
    """
    Momentum operator.

    p̂ = -iħ d/dx

    Central finite difference.
    """

    n = grid.points
    dx = grid.dx

    upper = np.ones(n - 1)
    lower = -np.ones(n - 1)

    derivative = diags([lower, upper], [-1, 1], shape=(n, n), format="csr")

    matrix = -1j * hbar / (2 * dx) * derivative

    return QuantumOperator(matrix)


def kinetic_energy_operator(
    grid: SpatialGrid, mass: float = 1.0, hbar: float = 1.0
) -> QuantumOperator:
    """
    Kinetic energy operator.

    T̂ = -(ħ²/2m)d²/dx²
    """

    n = grid.points
    dx = grid.dx

    diagonal = -2 * np.ones(n)

    off_diagonal = np.ones(n - 1)

    laplacian = diags(
        [off_diagonal, diagonal, off_diagonal], [-1, 0, 1], shape=(n, n), format="csr"
    )

    matrix = -(hbar**2) / (2 * mass) * laplacian / dx**2

    return QuantumOperator(matrix)


def potential_operator(potential: np.ndarray) -> QuantumOperator:
    """
    Potential energy operator.

    V̂ = V(x)
    """

    matrix = diags(potential, format="csr")

    return QuantumOperator(matrix)
