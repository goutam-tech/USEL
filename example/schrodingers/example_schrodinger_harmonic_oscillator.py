"""Example 7: Quantum Harmonic Oscillator — eigenstates and time evolution."""

from __future__ import annotations

import numpy as np

from usel.schrodinger import SchrodingerSolver, gaussian_wavepacket


def main() -> None:
    x = np.linspace(-8, 8, 512)
    V = 0.5 * x**2

    solver = SchrodingerSolver(x, V)
    result = solver.eigenstates(n_states=6)

    print("Harmonic Oscillator — Energy Eigenvalues")
    print("  n   Computed     Analytical   (n + 0.5)")
    for i in range(6):
        print(f"  {i}   {result.energies[i]:10.6f}   {result.energies[i]:10.6f}")

    # Time evolution of a Gaussian wave packet
    psi0 = gaussian_wavepacket(x, x0=-3.0, sigma=0.5, k0=5.0)
    print("\nTime-evolving wave packet (dt=0.02, t_end=1.0)...")
    td = solver.solve_time_dependent(psi0, t_end=1.0, dt=0.02)

    print(f"  Initial probability: {td.norm[0]:.6f}")
    print(f"  Final probability:   {td.norm[-1]:.6f}")
    print(f"  ⟨x⟩ initial: {td.expectation_x(0):.4f}")
    print(f"  ⟨x⟩ final:   {td.expectation_x(-1):.4f}")


if __name__ == "__main__":
    main()
