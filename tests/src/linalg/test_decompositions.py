import numpy as np
import pytest

from usel.exceptions import MatrixError
from usel.linalg.decompositions import cholesky, eigenvalues, eigenvectors, lu, qr, svd
from usel.math.matrix import Matrix


def test_eigenvalues():
    matrix = Matrix([[2, 0], [0, 3]])

    values = eigenvalues(matrix)

    assert sorted(values) == [2, 3]


def test_eigenvectors():
    matrix = Matrix([[2, 0], [0, 3]])

    values, vectors = eigenvectors(matrix)

    assert len(values) == 2
    assert vectors.shape == (2, 2)


def test_eigenvalue_eigenvector_residual():
    # A @ v = lambda * v must hold for every eigenpair.
    matrix = Matrix([[2, 1], [1, 2]])  # eigenvalues 3 and 1
    values, vectors = eigenvectors(matrix)
    for lam, vec in zip(values, vectors.data.T, strict=True):
        residual = matrix.data @ vec - lam * vec
        assert np.allclose(residual, 0.0, atol=1e-10)


def test_svd():
    matrix = Matrix([[1, 2], [3, 4]])

    u, s, vt = svd(matrix)

    assert u.shape == (2, 2)
    assert len(s) == 2
    assert vt.shape == (2, 2)


def test_svd_reconstruction():
    # U @ diag(s) @ Vt must reconstruct the original matrix.
    matrix = Matrix([[1, 2], [3, 4]])
    u, s, vt = svd(matrix)
    reconstructed = u.data @ np.diag(s) @ vt.data
    assert np.allclose(reconstructed, matrix.data, atol=1e-10)


def test_qr():
    matrix = Matrix([[1, 2], [3, 4]])

    q, r = qr(matrix)

    result = q @ r

    assert np.allclose(result.data, matrix.data)


def test_lu():
    matrix = Matrix([[4, 3], [6, 3]])

    p, L, u = lu(matrix)

    assert p.shape == (2, 2)
    assert L.shape == (2, 2)
    assert u.shape == (2, 2)


def test_cholesky():
    matrix = Matrix([[4, 2], [2, 3]])

    result = cholesky(matrix)

    assert result.shape == (2, 2)


def test_cholesky_reconstruction():
    # L @ L.T must reconstruct the original SPD matrix.
    matrix = Matrix([[4, 2], [2, 3]])
    L = cholesky(matrix)
    reconstructed = L.data @ L.data.T
    assert np.allclose(reconstructed, matrix.data, atol=1e-10)


def test_cholesky_non_spd_raises():
    with pytest.raises(MatrixError):
        cholesky(Matrix([[1, 2], [3, 4]]))
