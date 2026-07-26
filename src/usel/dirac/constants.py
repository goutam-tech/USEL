from dataclasses import dataclass


@dataclass(frozen=True)
class PhysicalConstants:
    c: float = 299792458.0

    hbar: float = 1.054571817e-34

    electron_mass: float = 9.1093837015e-31

    electron_charge: float = -1.602176634e-19

    epsilon0: float = 8.8541878128e-12


SI = PhysicalConstants()


@dataclass(frozen=True)
class NaturalUnits:
    c: float = 1.0

    hbar: float = 1.0


NATURAL = NaturalUnits()
