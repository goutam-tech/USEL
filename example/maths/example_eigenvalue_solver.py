"""Example 1: Eigenvalue Solver."""

from __future__ import annotations

from usel.linalg import eigenvalues, eigenvectors
from usel.math import Matrix


def main() -> None:
    m = Matrix([[4, 1], [2, 3]])
    print("Matrix:")
    print(m)

    values = eigenvalues(m)
    print(f"\nEigenvalues: {values}")

    values2, vectors = eigenvectors(m)
    print(f"\nEigenvectors:\n{vectors}")


if __name__ == "__main__":
    main()
