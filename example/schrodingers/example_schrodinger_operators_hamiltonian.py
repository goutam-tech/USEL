"""
USEL Schrödinger Example

Demonstrates:
- Quantum operators
- Hamiltonian construction
- Energy eigenvalue solving
- Infinite square well validation
"""

import numpy as np

from usel.schrodinger.grids import SpatialGrid
from usel.schrodinger.hamiltonian import build_hamiltonian, solve_energy_levels
from usel.schrodinger.operators import kinetic_energy_operator, momentum_operator, position_operator


def infinite_square_well(x):
    """
    Infinite square well potential.

    Inside box:
        V(x)=0

    Boundary conditions are handled
    by the finite difference grid.
    """

    return np.zeros_like(x)


def normalize(psi, dx):
    """
    Normalize wavefunction.

    ∫|ψ|² dx = 1
    """

    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)

    return psi / norm


def main():

    print("USEL Schrödinger Operator + Hamiltonian Test\n")

    # ---------------------------
    # Create spatial grid
    # ---------------------------

    grid = SpatialGrid(xmin=0.0, xmax=1.0, points=1000)

    print("Grid points:", grid.points)

    print("Grid spacing:", grid.dx)

    # ---------------------------
    # Operators
    # ---------------------------

    x_operator = position_operator(grid)

    p_operator = momentum_operator(grid)

    kinetic_operator = kinetic_energy_operator(grid)

    print("\nOperators created:")

    print("Position:", x_operator.shape)

    print("Momentum:", p_operator.shape)

    print("Kinetic:", kinetic_operator.shape)

    # ---------------------------
    # Potential
    # ---------------------------

    V = infinite_square_well(grid.x)

    # ---------------------------
    # Hamiltonian
    # ---------------------------

    H = build_hamiltonian(grid, V)

    print("\nHamiltonian:")

    print(H.shape)

    # ---------------------------
    # Solve eigenvalues
    # ---------------------------

    energies, states = solve_energy_levels(H, levels=5)

    print("\nEnergy Eigenvalues")

    print(" n   Computed       Analytical")

    for i, energy in enumerate(energies, start=1):
        analytical = i**2 * np.pi**2 / 2

        print(f"{i:2d}{energy:14.6f}{analytical:14.6f}")

    # ---------------------------
    # Ground state normalization
    # ---------------------------

    psi0 = states[:, 0]

    psi0 = normalize(psi0, grid.dx)

    probability = np.sum(np.abs(psi0) ** 2) * grid.dx

    print("\nGround state normalization:")

    print(f"{probability:.6f}")


if __name__ == "__main__":
    main()
