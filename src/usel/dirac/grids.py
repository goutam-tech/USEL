from __future__ import annotations

import numpy as np


class DiracGrid:
    def __init__(self, xmin: float, xmax: float, points: int):
        if points < 3:
            raise ValueError("Grid requires minimum 3 points")

        self.x = np.linspace(xmin, xmax, points)

    @property
    def dx(self):
        return self.x[1] - self.x[0]

    @property
    def size(self):
        return len(self.x)

    def index(self, value):
        return int(np.argmin(np.abs(self.x - value)))

    def zeros(self):
        return np.zeros(self.size)

    def spinor_zeros(self):
        return np.zeros((4, self.size), dtype=np.complex128)
