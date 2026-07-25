from __future__ import annotations

import numpy as np

from usel.dirac.spinors import dirac_alpha_matrices


class DiracOperators:
    def __init__(self, x, hbar=1.0, c=1.0):

        self.x = x

        self.dx = x[1] - x[0]

        self.hbar = hbar

        self.c = c

        self.alpha = dirac_alpha_matrices()

    def derivative(self):

        n = len(self.x)

        D = np.diag(np.ones(n - 1), 1) - np.diag(np.ones(n - 1), -1)

        return D / (2 * self.dx)

    def momentum(self):

        return -1j * self.hbar * self.derivative()

    def kinetic(self):

        return np.kron(self.c * self.alpha[0], self.momentum())
