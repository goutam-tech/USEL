"""
Grid utilities for Schrödinger simulations.

Provides spatial and temporal discretization tools.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class SpatialGrid:
    """
    One-dimensional spatial grid.

    Parameters
    ----------
    xmin : float
        Minimum coordinate.

    xmax : float
        Maximum coordinate.

    points : int
        Number of grid points.
    """

    xmin: float
    xmax: float
    points: int

    def __post_init__(self):
        if self.points < 2:
            raise ValueError("Grid requires at least two points")

        self.x = np.linspace(self.xmin, self.xmax, self.points)

        self.dx = self.x[1] - self.x[0]

    def size(self):
        return self.points

    def domain(self):
        return (self.xmin, self.xmax)

    def spacing(self):
        return self.dx


@dataclass
class TimeGrid:
    """
    Time discretization.

    Parameters
    ----------
    t0 : float
        Initial time.

    t_end : float
        Final time.

    steps : int
        Number of time steps.
    """

    t0: float
    t_end: float
    steps: int

    def __post_init__(self):

        if self.steps < 1:
            raise ValueError("Time steps must be positive")

        self.t = np.linspace(self.t0, self.t_end, self.steps)

        self.dt = self.t[1] - self.t[0] if self.steps > 1 else 0

    def size(self):
        return self.steps

    def timestep(self):
        return self.dt
