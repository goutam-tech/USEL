"""
Physical constants for Dirac equation calculations.

Supports SI and natural units.

"""

from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicalConstants:
    """
    Fundamental physics constants.
    """

    # Speed of light
    c: float = 299792458.0

    # Reduced Planck constant
    hbar: float = 1.054571817e-34

    # Electron mass
    electron_mass: float = 9.1093837015e-31

    # Electron charge
    electron_charge: float = -1.602176634e-19

    # Vacuum permittivity
    epsilon0: float = 8.8541878128e-12


# Default constants

SI = PhysicalConstants()


# Natural units:

# hbar = c = 1


@dataclass(frozen=True)
class NaturalUnits:
    """
    Natural units used in relativistic physics.

    c = 1
    hbar = 1
    """

    c: float = 1.0

    hbar: float = 1.0


NATURAL = NaturalUnits()
