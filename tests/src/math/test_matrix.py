import pytest
import numpy as np

from usel.math.matrix import Matrix
from usel.exceptions import MatrixError


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
