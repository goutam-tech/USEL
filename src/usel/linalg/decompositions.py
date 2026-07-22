"""Functional wrappers around common matrix decompositions.

These functions accept and return :class:`usel.math.Matrix` instances
(or raw NumPy arrays where noted), providing a functional API alongside the
object-oriented methods already available on ``Matrix``.
"""

from __future__ import annotations

import numpy as np

from usel.math.matrix import Matrix


def eigenvalues(matrix: Matrix) -> np.ndarray:
    """Return the eigenvalues of a square matrix."""
    return matrix.eigenvalues()


def eigenvectors(matrix: Matrix) -> tuple[np.ndarray, Matrix]:
    """Return (eigenvalues, eigenvectors) of a square matrix."""
    return matrix.eigenvectors()


def svd(matrix: Matrix) -> tuple[Matrix, np.ndarray, Matrix]:
    """Singular value decomposition. Returns (U, singular_values, Vt)."""
    return matrix.decompose_svd()


def qr(matrix: Matrix) -> tuple[Matrix, Matrix]:
    """QR decomposition. Returns (Q, R)."""
    return matrix.decompose_qr()


def lu(matrix: Matrix) -> tuple[Matrix, Matrix, Matrix]:
    """LU decomposition with partial pivoting. Returns (P, L, U)."""
    return matrix.decompose_lu()


def cholesky(matrix: Matrix) -> Matrix:
    """Cholesky decomposition for symmetric positive-definite matrices."""
    return matrix.decompose_cholesky()
