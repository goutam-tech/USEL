"""Result containers for the Dirac equation solvers."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class DiracResult:
    """Result of a Dirac equation solve.

    Attributes
    ----------
    spinor:
        4-component (or 2-component) spinor at each time step.
        Shape ``(n_steps, n_components, n_grid)`` for spatial problems,
        or ``(n_steps, n_components)`` for plane-wave problems.
    probability:
        Total probability density at each time step.
    t:
        Time values.
    x:
        Spatial grid points (may be empty for plane-wave solutions).
    energy:
        Computed or reference energy spectrum.
    spin_expectation:
        Spin expectation values ``(⟨σ_x⟩, ⟨σ_y⟩, ⟨σ_z⟩)`` at each step.
    """

    spinor: np.ndarray
    probability: np.ndarray
    t: np.ndarray
    x: np.ndarray = field(default_factory=lambda: np.array([]))
    energy: np.ndarray = field(default_factory=lambda: np.array([]))
    spin_expectation: np.ndarray = field(default_factory=lambda: np.empty((0, 3)))

    @property
    def total_probability(self) -> np.ndarray:
        """Total integrated probability at each time step."""
        if self.spinor.ndim == 3 and self.spinor.shape[2] > 1:
            dx = self.x[1] - self.x[0] if len(self.x) > 1 else 1.0
            return np.sum(self.probability, axis=-1) * dx
        return self.probability

    @property
    def upper_component(self) -> np.ndarray:
        """The upper (large) component of the spinor."""
        return self.spinor[:, 0, :] if self.spinor.ndim == 3 else self.spinor[:, 0]

    @property
    def lower_component(self) -> np.ndarray:
        """The lower (small) component of the spinor."""
        return self.spinor[:, 1, :] if self.spinor.ndim == 3 else self.spinor[:, 1]
