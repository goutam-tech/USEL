"""Schrödinger equation solvers for quantum mechanics problems.

This module provides numerical solvers for the time-independent and
time-dependent Schrödinger equation, along with standard potentials,
wavefunctions, and result containers.

Modules
-------
solver : Core solvers (finite difference, Crank-Nicolson, split operator).
potentials : Common potential energy functions.
wavefunctions : Analytical and initial-state wavefunctions.
results : Dataclass results for solver output.

Quick start
-----------
>>> from usel.schrodinger import SchrodingerSolver
>>> import numpy as np
>>> x = np.linspace(-5, 5, 500)
>>> solver = SchrodingerSolver(x, 0.5 * x**2)
>>> result = solver.eigenstates(n_states=5)
>>> result.energies
array([0.5, 1.5, 2.5, 3.5, 4.5])
"""

from __future__ import annotations

from usel.schrodinger.potentials import (
    barrier,
    coulomb,
    double_well,
    finite_square_well,
    harmonic_oscillator,
    infinite_square_well,
    kronig_penney,
)
from usel.schrodinger.results import TimeDependentResult, TimeIndependentResult
from usel.schrodinger.solver import SchrodingerSolver
from usel.schrodinger.wavefunctions import (
    double_gaussian,
    gaussian_wavepacket,
    harmonic_oscillator_state,
    particle_in_box_state,
)

__all__ = [
    "SchrodingerSolver",
    "TimeIndependentResult",
    "TimeDependentResult",
    "infinite_square_well",
    "finite_square_well",
    "harmonic_oscillator",
    "double_well",
    "barrier",
    "coulomb",
    "kronig_penney",
    "particle_in_box_state",
    "harmonic_oscillator_state",
    "gaussian_wavepacket",
    "double_gaussian",
]
