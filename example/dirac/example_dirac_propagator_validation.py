from usel.dirac.grids import DiracGrid
from usel.dirac.hamiltonian import DiracHamiltonian
from usel.dirac.propagators import apply_propagator, unitary_propagator
from usel.dirac.validation import norm, probability_density

grid = DiracGrid(-5, 5, 100)


psi = grid.spinor_zeros()


psi[0, 50] = 1


H = DiracHamiltonian(grid.x).matrix()


U = unitary_propagator(H, 0.01)


initial = norm(psi)


psi2 = apply_propagator(U, psi)


final = norm(psi2)


print("Initial norm:", initial)


print("Final norm:", final)


print("Probability shape:", probability_density(psi2).shape)
