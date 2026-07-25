from usel.dirac.grids import DiracGrid
from usel.dirac.hamiltonian import DiracHamiltonian
from usel.dirac.operators import DiracOperators
from usel.dirac.potentials import harmonic_potential

grid = DiracGrid(-5, 5, 100)


V = harmonic_potential(grid.x, 0.2)


operator = DiracOperators(grid.x)


p = operator.momentum()


print("Momentum:", p.shape)


H = DiracHamiltonian(grid.x, mass=1, potential=V)


matrix = H.matrix()


print("Hamiltonian:", matrix.shape)
