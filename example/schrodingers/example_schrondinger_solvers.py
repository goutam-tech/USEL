import numpy as np

from usel.schrodinger.grids import SpatialGrid
from usel.solvers.finite_difference import solve


def main():

    grid = SpatialGrid(0, 1, 300)

    V = np.zeros_like(grid.x)

    energies, states = solve(grid, V)

    print("Energy:", energies[0])

    print("Finite difference OK")


if __name__ == "__main__":
    main()
