"""
Boundary conditions for quantum simulations.

Supports:
- Dirichlet
- Neumann
- Periodic
"""

from __future__ import annotations

import numpy as np


class BoundaryCondition:
    """
    Base boundary condition.
    """

    def apply(self, psi: np.ndarray):
        raise NotImplementedError


class DirichletBoundary(BoundaryCondition):
    """
    Fixed wavefunction boundaries.

    Example:
    Infinite square well

    ψ(0)=0
    ψ(L)=0
    """

    def apply(self, psi):

        psi = psi.copy()

        psi[0] = 0
        psi[-1] = 0

        return psi


class NeumannBoundary(BoundaryCondition):
    """
    Zero derivative boundary.

    dψ/dx = 0
    """

    def apply(self, psi):

        psi = psi.copy()

        psi[0] = psi[1]
        psi[-1] = psi[-2]

        return psi


class PeriodicBoundary(BoundaryCondition):
    """
    Periodic boundary.

    ψ(0)=ψ(L)
    """

    def apply(self, psi):

        psi = psi.copy()

        psi[-1] = psi[0]

        return psi
