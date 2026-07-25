from usel.dirac.constants import NATURAL, SI
from usel.dirac.gamma_matrices import (
    alpha,
    beta,
    gamma_matrices,
    sigma_x,
    sigma_y,
    sigma_z,
    verify_clifford_algebra,
)


def main():

    print("=== DIRAC MODULE FOUNDATION TEST ===\n")

    print("Speed of light:", SI.c)

    print("Electron mass:", SI.electron_mass)

    print("Natural units:", NATURAL.c, NATURAL.hbar)

    print("\nPauli Matrices")

    print("Sigma X\n", sigma_x())

    print("Sigma Y\n", sigma_y())

    print("Sigma Z\n", sigma_z())

    print("\nGamma Matrices")

    gammas = gamma_matrices()

    for i, g in enumerate(gammas):
        print(f"Gamma {i} shape:", g.shape)

    print("\nClifford Algebra:")

    print(verify_clifford_algebra())

    print("\nBeta Matrix")

    print(beta())

    print("\nAlpha 1")

    print(alpha(1))


if __name__ == "__main__":
    main()
