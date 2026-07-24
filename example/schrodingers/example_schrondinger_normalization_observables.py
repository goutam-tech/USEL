"""
USEL Schrödinger Observables Example
"""

import numpy as np

from usel.schrodinger.grids import SpatialGrid
from usel.schrodinger.hamiltonian import build_hamiltonian, solve_energy_levels
from usel.schrodinger.normalization import is_normalized, normalize, probability_density
from usel.schrodinger.observables import expectation, fidelity, uncertainty
from usel.schrodinger.operators import momentum_operator, position_operator


def main():

    grid = SpatialGrid(0, 1, 1000)

    # Infinite well

    V = np.zeros_like(grid.x)

    H = build_hamiltonian(grid, V)

    energies, states = solve_energy_levels(H, 3)

    psi = states[:, 0]

    # Normalize

    psi = normalize(psi, grid.dx)

    print("Normalization:", is_normalized(psi, grid.dx))

    # Operators

    X = position_operator(grid)

    P = momentum_operator(grid)

    # Expectation values

    x_value = expectation(psi, X, grid.dx)

    p_value = expectation(psi, P, grid.dx)

    energy = expectation(psi, H, grid.dx)

    print("\nObservables")

    print("<x> =", np.real(x_value))

    print("<p> =", p_value)

    print("<E> =", np.real(energy))

    # Uncertainty

    dx = uncertainty(psi, X, grid.dx)

    dp = uncertainty(psi, P, grid.dx)

    print("\nUncertainty")

    print("Δx =", dx)

    print("Δp =", dp)

    print("ΔxΔp =", dx * dp)

    # Probability

    density = probability_density(psi)

    print("\nProbability sum:", np.sum(density) * grid.dx)

    # Fidelity self check

    print("\nFidelity:", fidelity(psi, psi, grid.dx))


if __name__ == "__main__":
    main()
