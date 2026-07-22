"""Example 8: Quantum tunnelling through a rectangular barrier."""

from __future__ import annotations

import numpy as np

from usel.schrodinger import SchrodingerSolver, gaussian_wavepacket


def main() -> None:
    x = np.linspace(-10, 10, 600)
    V = np.where((x >= -1) & (x <= 1), 2.0, 0.0)

    solver = SchrodingerSolver(x, V)

    psi0 = gaussian_wavepacket(x, x0=-5.0, sigma=0.5, k0=4.0)

    print("Quantum tunnelling — Gaussian packet encountering a barrier")
    td = solver.solve_time_dependent(psi0, t_end=3.0, dt=0.01)

    dx = x[1] - x[0]
    transmitted = np.sum(np.abs(td.psi[-1][x > 1]) ** 2) * dx
    reflected = np.sum(np.abs(td.psi[-1][x < -1]) ** 2) * dx
    barrier_region = np.sum(np.abs(td.psi[-1][(x >= -1) & (x <= 1)] ** 2)) * dx

    print(f"  Total probability:     {td.norm[-1]:.6f}")
    print(f"  Transmitted:           {transmitted:.6f}")
    print(f"  Reflected:             {reflected:.6f}")
    print(f"  In barrier:            {barrier_region:.6f}")
    print(f"  Transmission coeff:    {transmitted:.4f}")


if __name__ == "__main__":
    main()
