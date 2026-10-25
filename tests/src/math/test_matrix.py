import numpy as np
import pytest

from usel.exceptions import MatrixError
from usel.math.matrix import Matrix


class TestMatrix:
    def test_matrix_creation(self):
        m = Matrix([[1, 2], [3, 4]])

        assert m.shape == (2, 2)

    def test_zero_matrix(self):
        m = Matrix.zeros(3, 3)

        assert np.all(m.data == 0)

    def test_one_matrix(self):
        m = Matrix.ones(2, 4)

        assert np.all(m.data == 1)

    def test_identity_matrix(self):
        m = Matrix.identity(3)

        assert np.array_equal(m.data, np.eye(3))

    def test_random_matrix(self):
        m1 = Matrix.random(3, 3, seed=1)
        m2 = Matrix.random(3, 3, seed=1)

        assert m1 == m2

    def test_matrix_addition(self):
        a = Matrix([[1, 2], [3, 4]])

        b = Matrix([[5, 6], [7, 8]])

        result = a + b

        expected = Matrix([[6, 8], [10, 12]])

        assert result == expected

    def test_matrix_subtraction(self):
        a = Matrix([[5, 6], [7, 8]])

        b = Matrix([[1, 2], [3, 4]])

        result = a - b

        expected = Matrix([[4, 4], [4, 4]])

        assert result == expected

    def test_matrix_multiplication(self):
        a = Matrix([[1, 2], [3, 4]])

        b = Matrix([[5, 6], [7, 8]])

        result = a @ b

        expected = Matrix([[19, 22], [43, 50]])

        assert result == expected

    def test_invalid_multiplication(self):
        a = Matrix.zeros(2, 3)

        b = Matrix.zeros(4, 2)

        with pytest.raises(MatrixError):
            a @ b

    def test_transpose(self):
        m = Matrix([[1, 2, 3], [4, 5, 6]])

        t = m.T

        assert t.shape == (3, 2)

    def test_determinant(self):
        m = Matrix([[1, 2], [3, 4]])

        assert m.determinant() == pytest.approx(-2)

    def test_trace(self):
        m = Matrix([[1, 2], [3, 4]])

        assert m.trace() == 5

    def test_inverse(self):
        m = Matrix([[4, 7], [2, 6]])

        inv = m.inverse()

        result = m @ inv

        assert np.allclose(result.data, np.eye(2))

    def test_norm(self):
        m = Matrix([[3, 4]])

        assert m.norm() == pytest.approx(5)

    def test_eigenvalues(self):
        m = Matrix([[2, 0], [0, 3]])

        values = m.eigenvalues()

        assert sorted(values) == [2, 3]

    def test_cholesky(self):
        m = Matrix([[4, 2], [2, 3]])

        result = m.decompose_cholesky()

        assert result.shape == (2, 2)

    @staticmethod
    def test_determinant_known_values():
        # 2x2: det = ad - bc = 1*4 - 2*3 = -2
        assert Matrix([[1, 2], [3, 4]]).determinant() == pytest.approx(-2)
        # 3x3 via cofactor expansion along the first row:
        # det = 1*(4*6-5*0) - 2*(0*6-5*1) + 3*(0*0-4*1) = 24 + 10 - 12 = 22
        m = Matrix([[1, 2, 3], [0, 4, 5], [1, 0, 6]])
        assert m.determinant() == pytest.approx(22)

    @staticmethod
    def test_determinant_singular_is_zero():
        # Rows are proportional, so the determinant must be exactly 0.
        m = Matrix([[1, 2], [2, 4]])
        assert m.determinant() == pytest.approx(0.0)

    @staticmethod
    def test_determinant_non_square_raises():
        with pytest.raises(MatrixError):
            Matrix.zeros(2, 3).determinant()

    @staticmethod
    def test_inverse_round_trip():
        for m in [
            Matrix([[4, 7], [2, 6]]),
            Matrix([[1, 2, 3], [0, 1, 4], [5, 6, 0]]),
            Matrix.identity(4),
        ]:
            inv = m.inverse()
            result = m @ inv
            assert np.allclose(result.data, np.eye(m.rows), atol=1e-10)

    @staticmethod
    def test_inverse_singular_raises():
        with pytest.raises(MatrixError):
            Matrix([[1, 2], [2, 4]]).inverse()

    @staticmethod
    def test_inverse_non_square_raises():
        with pytest.raises(MatrixError):
            Matrix.zeros(2, 3).inverse()

    @staticmethod
    def test_inverse_known_2x2():
        # A = [[4, 7], [2, 6]], det = 10, inv = 1/10 * [[6, -7], [-2, 4]]
        m = Matrix([[4, 7], [2, 6]])
        inv = m.inverse()
        expected = Matrix([[0.6, -0.7], [-0.2, 0.4]])
        assert np.allclose(inv.data, expected.data, atol=1e-10)

    @staticmethod
    def test_eigenvalue_eigenvector_residual():
        # A @ v = lambda * v must hold for every eigenpair.
        m = Matrix([[2, 1], [1, 2]])  # eigenvalues 3 and 1
        values, vectors = m.eigenvectors()
        for lam, vec in zip(values, vectors.data.T, strict=True):
            residual = m.data @ vec - lam * vec
            assert np.allclose(residual, 0.0, atol=1e-10)

    @staticmethod
    def test_svd_reconstruction():
        # U @ diag(s) @ Vt must reconstruct the original matrix.
        m = Matrix([[1, 2], [3, 4]])
        u, s, vt = m.decompose_svd()
        reconstructed = u.data @ np.diag(s) @ vt.data
        assert np.allclose(reconstructed, m.data, atol=1e-10)

    @staticmethod
    def test_lu_reconstruction():
        # P @ L @ U must reconstruct the original matrix.
        m = Matrix([[4, 3], [6, 3]])
        p, L, u = m.decompose_lu()
        reconstructed = p.data @ L.data @ u.data
        assert np.allclose(reconstructed, m.data, atol=1e-10)

    @staticmethod
    def test_cholesky_reconstruction():
        # L @ L.T must reconstruct the original SPD matrix.
        m = Matrix([[4, 2], [2, 3]])
        L = m.decompose_cholesky()
        reconstructed = L.data @ L.data.T
        assert np.allclose(reconstructed, m.data, atol=1e-10)

    @staticmethod
    def test_cholesky_non_spd_raises():
        with pytest.raises(MatrixError):
            Matrix([[1, 2], [3, 4]]).decompose_cholesky()

    @staticmethod
    def test_norm_known_values():
        # Frobenius norm of [[3, 4]] = sqrt(9 + 16) = 5
        assert Matrix([[3, 4]]).norm() == pytest.approx(5)
        # 1-norm (max column sum) of [[1, 2], [3, 4]] = max(1+3, 2+4) = 6
        assert Matrix([[1, 2], [3, 4]]).norm(1) == pytest.approx(6)
        # infinity-norm (max row sum) = max(1+2, 3+4) = 7
        assert Matrix([[1, 2], [3, 4]]).norm(np.inf) == pytest.approx(7)
