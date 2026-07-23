"""Result containers for the Schrödinger equation solvers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class TimeIndependentResult:
    """Result of a time-independent Schrödinger equation solve.

    Attributes
    ----------
    energies:
        Sorted array of computed energy eigenvalues.
    eigenstates:
        Corresponding normalised eigenstates (columns of the matrix).
    x:
        Spatial grid points.
    """

    energies: np.ndarray
    eigenstates: np.ndarray
    x: np.ndarray


@dataclass(frozen=True)
class TimeDependentResult:
    """Result of a time-dependent Schrödinger equation solve.

    Attributes
    ----------
    psi:
        Wavefunction at each time step. Shape ``(n_steps, n_grid)``.
    probability:
        Probability density ``|psi|^2`` at each time step.
    t:
        Time values corresponding to each step.
    x:
        Spatial grid points.
    energies:
        Analytical energy values used for reference (may be empty).
    """

    psi: np.ndarray
    probability: np.ndarray
    t: np.ndarray
    x: np.ndarray
    energies: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def norm(self) -> np.ndarray:
        """Compute the total probability at each time step."""
        dx = self.x[1] - self.x[0] if len(self.x) > 1 else 1.0
        return np.sum(self.probability, axis=1) * dx

    def expectation_x(self, step: int = -1) -> float:
        """Compute the expectation value of position at the given time step."""
        dx = self.x[1] - self.x[0] if len(self.x) > 1 else 1.0
        return float(np.sum(self.x * self.probability[step]) * dx)

    def expectation_energy(self, hamiltonian: np.ndarray, step: int = -1) -> float:
        """Compute the expectation value of energy at the given time step."""
        psi = self.psi[step]
        return float(np.real(np.conj(psi) @ hamiltonian @ psi))
