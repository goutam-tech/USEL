"""Analytical and initial wavefunctions for the Schrödinger equation."""

from __future__ import annotations

import math

import numpy as np


def particle_in_box_state(x: np.ndarray, n: int, x_min: float, x_max: float) -> np.ndarray:
    """Normalised eigenstate of the infinite square well.

    Parameters
    ----------
    x: Spatial grid.
    n: Quantum number (1, 2, 3, ...).
    x_min, x_max: Boundaries of the well.
    """
    L = x_max - x_min
    psi = np.zeros_like(x)
    mask = (x >= x_min) & (x <= x_max)
    psi[mask] = math.sqrt(2.0 / L) * np.sin(n * math.pi * (x[mask] - x_min) / L)
    return psi


def harmonic_oscillator_state(
    x: np.ndarray, n: int, omega: float = 1.0, mass: float = 1.0
) -> np.ndarray:
    """Normalised eigenstate of the quantum harmonic oscillator.

    Uses the recurrence relation for Hermite polynomials.
    """
    alpha = math.sqrt(mass * omega)
    xi = alpha * x
    psi = np.zeros_like(x, dtype=np.float64)

    # Hermite polynomials via recurrence: H_{n+1}(x) = 2x H_n(x) - 2n H_{n-1}(x)
    h_prev = np.ones_like(x, dtype=np.float64)  # H_0
    h_curr = 2.0 * xi.copy()  # H_1

    if n == 0:
        psi = h_prev
    elif n == 1:
        psi = h_curr
    else:
        for k in range(1, n):
            h_next = 2.0 * xi * h_curr - 2.0 * k * h_prev
            h_prev = h_curr
            h_curr = h_next
        psi = h_curr

    norm_factor = (mass * omega / math.pi) ** 0.25
    hermite_norm = math.sqrt(2.0**n * math.factorial(n))
    psi = norm_factor * np.exp(-0.5 * xi**2) * psi / hermite_norm
    return psi


def gaussian_wavepacket(
    x: np.ndarray,
    x0: float = 0.0,
    sigma: float = 1.0,
    k0: float = 0.0,
) -> np.ndarray:
    """Gaussian wave packet centred at ``x0`` with width ``sigma`` and momentum ``k0``.

    Parameters
    ----------
    x: Spatial grid.
    x0: Centre of the packet.
    sigma: Width (standard deviation).
    k0: Central wave number (determines average momentum).
    """
    envelope = np.exp(-0.5 * ((x - x0) / sigma) ** 2)
    oscillation = np.exp(1j * k0 * x)
    psi = envelope * oscillation
    # Normalise
    dx = x[1] - x[0] if len(x) > 1 else 1.0
    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    return psi / norm


def double_gaussian(
    x: np.ndarray,
    x1: float = -2.0,
    x2: float = 2.0,
    sigma: float = 0.5,
    k1: float = 5.0,
    k2: float = -5.0,
) -> np.ndarray:
    """Superposition of two Gaussian wave packets with opposite momenta."""
    psi1 = gaussian_wavepacket(x, x0=x1, sigma=sigma, k0=k1)
    psi2 = gaussian_wavepacket(x, x0=x2, sigma=sigma, k0=k2)
    psi = psi1 + psi2
    dx = x[1] - x[0] if len(x) > 1 else 1.0
    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    return psi / norm
