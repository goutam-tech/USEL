from usel.dirac.grids import DiracGrid
from usel.dirac.potentials import barrier_potential, free_potential, harmonic_potential

grid = DiracGrid(-10, 10, 500)


print("Grid size:", grid.size)


print("dx:", grid.dx)


psi = grid.spinor_zeros()


print("Spinor shape:", psi.shape)


V1 = free_potential(grid.x)


V2 = harmonic_potential(grid.x, 0.5)


V3 = barrier_potential(grid.x, 10, -1, 1)


print("Free potential:", V1.shape)


print("Harmonic:", V2.shape)


print("Barrier:", V3.max())
