from __future__ import annotations

import numpy as np


def free_potential(x):

    return np.zeros_like(x, dtype=float)


def harmonic_potential(x, strength=1.0):

    return 0.5 * strength * x * x


def barrier_potential(x, height, start, end):

    V = np.zeros_like(x, dtype=float)

    mask = (x >= start) & (x <= end)

    V[mask] = height

    return V


class ScalarPotential:
    def __init__(self, values):

        self.values = np.asarray(values, dtype=float)

    def __call__(self, x):

        return self.values
