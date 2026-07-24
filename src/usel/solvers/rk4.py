"""
Fourth order Runge-Kutta Schrödinger solver.
"""

import numpy as np


def evolve(psi, hamiltonian, dt, steps, hbar=1.0):

    def derivative(state):

        return -1j / hbar * hamiltonian.apply(state)

    states = []

    current = psi.copy()

    for _ in range(steps):
        k1 = derivative(current)

        k2 = derivative(current + dt * k1 / 2)

        k3 = derivative(current + dt * k2 / 2)

        k4 = derivative(current + dt * k3)

        current += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        states.append(current.copy())

    return np.array(states)
