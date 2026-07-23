"""
Physics validation utilities.
"""

import numpy as np


def check_probability_conservation(states, dx, tolerance=1e-8):
    probabilities = []

    for psi in states:
        p = np.sum(np.abs(psi) ** 2) * dx

        probabilities.append(p)

    return np.allclose(probabilities, 1.0, atol=tolerance)


def energy_conservation(energies, tolerance=1e-8):

    return np.allclose(energies, energies[0], atol=tolerance)


def max_probability_error(states, dx):

    errors = []

    for psi in states:
        value = np.sum(np.abs(psi) ** 2) * dx

        errors.append(abs(value - 1))

    return max(errors)
