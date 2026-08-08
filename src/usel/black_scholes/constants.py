import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DistributionConstants:
    SQRT_2: float = math.sqrt(2.0)
    SQRT_2PI: float = math.sqrt(2.0 * math.pi)
    INV_SQRT_2PI: float = 1.0 / math.sqrt(2.0 * math.pi)


@dataclass(frozen=True)
class ParaConstants:
    SQRT_2 = math.sqrt(2.0)


@dataclass(frozen=True)
class EpsilonConstants:
    EPSILON = np.finfo(np.float64).eps


@dataclass(frozen=True)
class ImpliedConstants:
    DEFAULT_LOW_VOL = 1e-6
    DEFAULT_HIGH_VOL = 5.0
    DEFAULT_INITIAL_GUESS = 0.5
    DEFAULT_TOLERANCE = 1e-8
    DEFAULT_MAX_ITERATIONS = 100
    ARBITRAGE_TOL = 1e-10


@dataclass(frozen=True)
class PdeConstants:
    DEFAULT_PDE_TOLERANCE = 1e-6


DISTRIBUTION_CONSTANTS = DistributionConstants()
PARA_CONSTANTS = ParaConstants()
EPSILON_CONSTANTS = EpsilonConstants()
IMPLIEDCONSTANTS = ImpliedConstants()
PDECONSTANTS = PdeConstants()
