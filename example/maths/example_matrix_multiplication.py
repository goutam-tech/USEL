"""Example 3: Matrix Multiplication.

Demonstrates creating matrices and performing multiplication, transpose,
determinant, and inverse operations using ``usel.math.Matrix``.
"""

from __future__ import annotations
from usel.math import Matrix

def main() -> None:
    a = Matrix.random(4, 4, seed=1)
    b = Matrix.identity(4)

    print("Matrix A:")
    print(a)

    product = a @ b
    print("\nA @ I (should equal A):")
    print(product)

    print(f"\nDeterminant of A: {a.determinant():.6f}")
    print(f"Trace of A: {a.trace():.6f}")

    inverse = a.inverse()
    identity_check = a @ inverse
    print("\nA @ A^-1 (should be close to identity):")
    print(identity_check)


if __name__ == "__main__":
    main()