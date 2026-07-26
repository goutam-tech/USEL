import numpy as np

from usel.dirac.grids import DiracGrid
from usel.dirac.visualization import (
    plot_probability,
    plot_probability_evolution,
    plot_spinor_components,
)

grid = DiracGrid(-10, 10, 300)


psi = grid.spinor_zeros()


packet = np.exp(-(grid.x**2))


psi[0] = packet


probability = np.sum(np.abs(psi) ** 2, axis=0)


plot_probability(grid.x, probability)


plot_spinor_components(grid.x, psi)


history = []


for t in range(50):
    shifted = np.roll(probability, t)

    history.append(shifted)


history = np.array(history)


plot_probability_evolution(grid.x, history)
