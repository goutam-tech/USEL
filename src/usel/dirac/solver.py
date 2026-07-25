"""
Dirac Equation Solver.

Solves:

    iℏ ∂ψ/∂t = H_D ψ


Hamiltonian:

    H_D = c α.p + βmc² + V(x)


Uses:

- 4-component Dirac spinors
- Dirac alpha matrices
- Beta matrix
- Finite difference momentum
- Matrix exponential evolution

"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from usel.dirac.results import DiracResult
from usel.dirac.spinors import (
    dirac_alpha_matrices,
    dirac_beta,
    normalize_spinor,
    spin_expectation,
)
from usel.exceptions import ValidationError

HBAR = 1.0
C_LIGHT = 1.0
MASS = 1.0


class DiracSolver:
    """
    One dimensional Dirac equation solver.

    Wavefunction:

        ψ(x)=

        [ψ1]
        [ψ2]
        [ψ3]
        [ψ4]

    """

    def __init__(
        self,
        x: np.ndarray,
        potential=None,
        mass=MASS,
        c=C_LIGHT,
        hbar=HBAR,
    ):

        x = np.asarray(x)

        if x.ndim != 1:
            raise ValidationError("Grid must be one dimensional")

        if len(x) < 3:
            raise ValidationError("Grid requires at least 3 points")

        self.x = x

        self.mass = mass

        self.c = c

        self.hbar = hbar

        if potential is None:
            self.potential = np.zeros_like(x, dtype=float)

        else:
            potential = np.asarray(potential, dtype=float)

            if potential.shape != x.shape:
                raise ValidationError("Potential size mismatch")

            self.potential = potential

        # From spinors.py

        self.alpha_x, self.alpha_y, self.alpha_z = dirac_alpha_matrices()

        self.beta = dirac_beta()

    # ======================================================
    # Hamiltonian
    # ======================================================

    def build_hamiltonian(self):
        """
        Construct:

        H =
        c αx p + βmc² + V

        """

        n = len(self.x)

        dx = self.x[1] - self.x[0]

        # derivative operator

        derivative = np.diag(np.ones(n - 1), 1) - np.diag(np.ones(n - 1), -1)

        derivative /= 2 * dx

        momentum = -1j * self.hbar * derivative

        H = np.zeros((4 * n, 4 * n), dtype=complex)

        # kinetic term

        H += np.kron(self.c * self.alpha_x, momentum)

        # mass term

        H += np.kron(self.mass * self.c**2 * self.beta, np.eye(n))

        # scalar potential

        H += np.kron(np.eye(4), np.diag(self.potential))

        return H

    # ======================================================
    # Time evolution
    # ======================================================

    def solve(self, psi0, t_end, dt):
        """
        Evolve:

        ψ(t+dt)=Uψ(t)

        """

        n = len(self.x)

        if psi0.shape != (4, n):
            raise ValidationError("Initial spinor must have shape (4,N)")

        steps = int(round(t_end / dt))

        H = self.build_hamiltonian()

        U = expm(-1j * H * dt / self.hbar)

        psi = normalize_spinor(psi0)

        history = []

        probability = []

        spin = []

        for _ in range(steps + 1):
            history.append(psi.copy())

            probability.append(np.sum(np.abs(psi) ** 2, axis=0))

            spin.append(spin_expectation(psi[:, len(psi) // 2]))

            vector = psi.reshape(4 * n)

            vector = U @ vector

            psi = vector.reshape(4, n)

        return DiracResult(
            spinor=np.array(history),
            probability=np.array(probability),
            t=np.linspace(0, t_end, steps + 1),
            x=self.x,
            spin_expectation=np.array(spin),
        )

    # ======================================================
    # Relativistic Energy
    # ======================================================

    def energy(self, momentum):
        """
        Relativistic dispersion:

        E²=p²c²+m²c⁴

        """

        return np.sqrt((momentum * self.c) ** 2 + (self.mass * self.c**2) ** 2)
