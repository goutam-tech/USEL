"""usel: an open-source, CPU-only scientific computing framework.

This package provides the mathematical and computational foundation for
quantum mechanics, relativistic quantum mechanics, and computational
electromagnetics:

- Reusable matrix and linear algebra utilities
- Numerical solvers (root finding, ODEs, integration)
- FFT based frequency analysis
- Schrödinger equation solvers (bound states, wave-packet propagation)
- Dirac equation solvers (relativistic spinors, Pauli/Dirac matrices)
- Maxwell equation solvers (2D FDTD, electrostatics)
- Random number and matrix generation
- Configuration, IO, logging, and validation utilities
- A command line interface (``usel``)
"""

from __future__ import annotations

__version__ = "1.1.0"
__license__ = "MIT"
__all__ = ["__version__", "__license__"]
