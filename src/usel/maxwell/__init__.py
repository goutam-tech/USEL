"""
maxwell: a symbolic + numerical toolkit for electromagnetism.

Currently implemented:
    maxwell.constants  - physical constants (epsilon0, mu0, c)
    maxwell.core       - the four Maxwell equations (differential form)
"""

from . import constants
from . import core
from . import materials
from . import potentials
from . import integral
from . import waves
from . import energy
from . import boundary
from . import covariant
from . import special_cases

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
