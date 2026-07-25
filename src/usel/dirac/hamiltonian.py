from __future__ import annotations

import numpy as np

from usel.dirac.operators import DiracOperators
from usel.dirac.spinors import dirac_beta


class DiracHamiltonian:
    def __init__(self, x, mass=1.0, c=1.0, hbar=1.0, potential=None):

        self.x = x

        self.mass = mass

        self.c = c

        self.operator = DiracOperators(x, hbar, c)

        self.beta = dirac_beta()

        if potential is None:
            self.V = np.zeros(len(x))

        else:
            self.V = potential

    def matrix(self):

        n = len(self.x)

        H = self.operator.kinetic()

        H += np.kron(self.mass * self.c**2 * self.beta, np.eye(n))

        H += np.kron(np.eye(4), np.diag(self.V))

        return H
