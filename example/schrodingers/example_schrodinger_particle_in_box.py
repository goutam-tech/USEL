"""Example 6: Particle in a Box — infinite square well eigenstates."""

from __future__ import annotations

import numpy as np

from usel.schrodinger import SchrodingerSolver, particle_in_box_state


def main() -> None:
    x = np.linspace(0, 1, 500)
    V = np.where((x >= 0) & (x <= 1), 0.0, 1e10)

    solver = SchrodingerSolver(x, V)
    result = solver.eigenstates(n_states=5)

    print("Infinite Square Well — Energy Eigenvalues")
    print("  n   Computed     Analytical")
    for i in range(5):
        analytical = (i + 1) ** 2 * np.pi**2 / 2.0
        print(f"  {i + 1}   {result.energies[i]:10.6f}   {analytical:10.6f}")

    print(f"\nGround-state normalisation check: "
          f"{np.sum(np.abs(result.eigenstates[:, 0]) ** 2) * (x[1] - x[0]):.6f}")


if __name__ == "__main__":
    main()
