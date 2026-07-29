"""
maxwell: a symbolic + numerical toolkit for electromagnetism.

Currently implemented:
    maxwell.constants  - physical constants (epsilon0, mu0, c)
    maxwell.core       - the four Maxwell equations (differential form)
"""

from . import (
    boundary,
    constants,
    core,
    covariant,
    energy,
    integral,
    materials,
    potentials,
    special_cases,
    waves,
)

__all__ = [
    "constants",
    "core",
    "materials",
    "potentials",
    "integral",
    "waves",
    "energy",
    "boundary",
    "covariant",
    "special_cases",
]
__version__ = "0.3.0"
