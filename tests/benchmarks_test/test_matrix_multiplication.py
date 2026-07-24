import pytest

from usel.math import Matrix


class TestMatrixMultiplication:
    def test_random_matrix_creation(self):
        """Test random matrix generation."""

        matrix = Matrix.random(5, 5, seed=1)

        assert matrix is not None

    def test_matrix_dimensions(self):
        """Test generated matrix dimensions."""

        matrix = Matrix.random(10, 20, seed=1)

        assert matrix.rows == 10
        assert matrix.cols == 20

    def test_matrix_multiplication_shape(self):
        """Test multiplication output dimensions."""

        a = Matrix.random(5, 10, seed=1)
        b = Matrix.random(10, 8, seed=2)

        result = a @ b

        assert result.rows == 5
        assert result.cols == 8

    def test_matrix_multiplication_small_case(self):
        """Test multiplication with known values."""

        a = Matrix([[1, 2], [3, 4]])

        b = Matrix([[5, 6], [7, 8]])

        result = a @ b

        expected = Matrix([[19, 22], [43, 50]])

        assert result == expected

    @pytest.mark.parametrize("size", [2, 5, 10, 50])
    def test_multiple_matrix_sizes(self, size):
        """Test multiplication for different sizes."""

        a = Matrix.random(size, size, seed=1)
        b = Matrix.random(size, size, seed=2)

        result = a @ b

        assert result.rows == size
        assert result.cols == size
