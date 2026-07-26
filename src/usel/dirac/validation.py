from __future__ import annotations

import numpy as np


def norm(psi, dx=1.0):
    return np.sum(np.abs(psi) ** 2) * dx


def check_normalization(psi, tolerance=1e-6):
    value = norm(psi)

    return abs(value - 1) < tolerance


def probability_density(psi):
    return np.sum(np.abs(psi) ** 2, axis=0)


def check_probability_conservation(before, after, tolerance=1e-6):
    return abs(before - after) < tolerance
