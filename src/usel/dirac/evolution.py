"""
Time evolution for Dirac equation.

Uses:

ψ(t+dt)=Uψ(t)

U=exp(-iHDt/hbar)
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from usel.dirac.boundary_conditions import dirichlet


class DiracEvolution:
    def __init__(self, hamiltonian, hbar=1.0):
        self.H = hamiltonian

        self.hbar = hbar

    def propagator(self, dt):
        return expm(-1j * self.H * dt / self.hbar)

    def step(self, psi, dt):
        U = self.propagator(dt)

        n = psi.shape[1]

        state = psi.reshape(4 * n)

        state = U @ state

        psi = state.reshape(4, n)

        return dirichlet(psi)

    def evolve(self, psi0, dt, steps):
        psi = psi0.copy()

        history = [psi.copy()]

        for _ in range(steps):
            psi = self.step(psi, dt)

            history.append(psi.copy())

        return np.array(history)
