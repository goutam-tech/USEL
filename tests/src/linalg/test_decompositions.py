import numpy as np
from usel.math.matrix import Matrix
from usel.linalg.decompositions import eigenvalues, eigenvectors, svd, qr, lu, cholesky


def test_eigenvalues():
    matrix = Matrix([[2, 0], [0, 3]])

    values = eigenvalues(matrix)

    assert sorted(values) == [2, 3]


def test_eigenvectors():
    matrix = Matrix([[2, 0], [0, 3]])

    values, vectors = eigenvectors(matrix)

    assert len(values) == 2
    assert vectors.shape == (2, 2)


def test_svd():
    matrix = Matrix([[1, 2], [3, 4]])

    u, s, vt = svd(matrix)

    assert u.shape == (2, 2)
    assert len(s) == 2
    assert vt.shape == (2, 2)


def test_qr():
    matrix = Matrix([[1, 2], [3, 4]])

    q, r = qr(matrix)

    result = q @ r

    assert np.allclose(result.data, matrix.data)


def test_lu():
    matrix = Matrix([[4, 3], [6, 3]])

    p, l, u = lu(matrix)

    assert p.shape == (2, 2)
    assert l.shape == (2, 2)
    assert u.shape == (2, 2)


def test_cholesky():
    matrix = Matrix([[4, 2], [2, 3]])

    result = cholesky(matrix)

    assert result.shape == (2, 2)
