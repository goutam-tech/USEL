"""
USEL Schrödinger Time Evolution Example
"""

import numpy as np

from usel.schrodinger.boundary_conditions import DirichletBoundary
from usel.schrodinger.evolution import evolve
from usel.schrodinger.grids import SpatialGrid
from usel.schrodinger.hamiltonian import build_hamiltonian, solve_energy_levels
from usel.schrodinger.validation import check_probability_conservation, max_probability_error
from usel.schrodinger.visualization import plot_probability


def main():

    grid = SpatialGrid(0, 1, 400)

    V = np.zeros_like(grid.x)

    H = build_hamiltonian(grid, V)

    energies, states = solve_energy_levels(H, 1)

    psi0 = states[:, 0]

    evolved_states = evolve(psi0, H, dt=0.001, steps=100, dx=grid.dx, boundary=DirichletBoundary())

    print("Evolution shape:", evolved_states.shape)

    print("Probability conserved:", check_probability_conservation(evolved_states, grid.dx))

    print("Maximum error:", max_probability_error(evolved_states, grid.dx))

    plot_probability(grid.x, evolved_states[-1])


if __name__ == "__main__":
    main()
