"""A lightweight, NumPy-backed Matrix abstraction.

The :class:`Matrix` class wraps a 2D ``numpy.ndarray`` and exposes a clean,
object-oriented API for common linear-algebra operations, while keeping
the underlying data accessible via ``.data`` for interoperability with
NumPy and SciPy.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from usel.exceptions import MatrixError


class Matrix:
    """A 2D numerical matrix backed by a NumPy array.

    Parameters
    ----------
    data:
        A 2D array-like structure (list of lists, tuple of tuples, or
        ``numpy.ndarray``) representing the matrix entries.
    dtype:
        Optional NumPy dtype override. Defaults to ``float64``.
    """

    __slots__ = ("_data",)

    def __init__(
        self, data: Iterable[Iterable[float]] | np.ndarray, dtype: type = np.float64
    ) -> None:
        array = np.array(data, dtype=dtype)
        if array.ndim != 2:
            raise MatrixError(f"Matrix data must be 2-dimensional, got {array.ndim}D input")
        self._data = array

    @classmethod
    def zeros(cls, rows: int, cols: int) -> Matrix:
        """Create a ``rows x cols`` matrix of zeros."""
        return cls(np.zeros((rows, cols)))

    @classmethod
    def ones(cls, rows: int, cols: int) -> Matrix:
        """Create a ``rows x cols`` matrix of ones."""
        return cls(np.ones((rows, cols)))

    @classmethod
    def identity(cls, n: int) -> Matrix:
        """Create an ``n x n`` identity matrix."""
        return cls(np.eye(n))

    @classmethod
    def random(cls, rows: int, cols: int, seed: int | None = None) -> Matrix:
        """Create a ``rows x cols`` matrix of uniform random values in [0, 1)."""
        rng = np.random.default_rng(seed)
        return cls(rng.random((rows, cols)))

    @classmethod
    def from_sequence(cls, rows: Sequence[Sequence[float]]) -> Matrix:
        """Create a matrix from a sequence of row sequences."""
        return cls(rows)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def data(self) -> np.ndarray:
        """The underlying NumPy array (read/write view)."""
        return self._data

    @property
    def shape(self) -> tuple[int, int]:
        """The ``(rows, cols)`` shape of the matrix."""
        return self._data.shape  # type: ignore[return-value]

    @property
    def rows(self) -> int:
        return self._data.shape[0]

    @property
    def cols(self) -> int:
        return self._data.shape[1]

    @property
    def is_square(self) -> bool:
        return self.rows == self.cols

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------
    def transpose(self) -> Matrix:
        """Return the transpose of this matrix."""
        return Matrix(self._data.T)

    @property
    def T(self) -> Matrix:  # noqa: N802 - conventional alias
        return self.transpose()

    def multiply(self, other: Matrix) -> Matrix:
        """Matrix multiplication with another :class:`Matrix`."""
        if self.cols != other.rows:
            raise MatrixError(
                f"Cannot multiply matrices with shapes {self.shape} and {other.shape}"
            )
        return Matrix(self._data @ other.data)

    def inverse(self) -> Matrix:
        """Return the matrix inverse.

        Raises
        ------
        MatrixError
            If the matrix is not square or is singular.
        """
        if not self.is_square:
            raise MatrixError("Inverse is only defined for square matrices")
        try:
            return Matrix(np.linalg.inv(self._data))
        except np.linalg.LinAlgError as exc:
            raise MatrixError("Matrix is singular and cannot be inverted") from exc

    def determinant(self) -> float:
        """Return the determinant of a square matrix."""
        if not self.is_square:
            raise MatrixError("Determinant is only defined for square matrices")
        return float(np.linalg.det(self._data))

    def trace(self) -> float:
        """Return the sum of the diagonal elements."""
        return float(np.trace(self._data))

    def norm(self, ord: str | int | None = "fro") -> float:  # noqa: A002 - mirrors numpy API
        """Return a matrix norm. Defaults to the Frobenius norm."""
        return float(np.linalg.norm(self._data, ord=ord))

    def decompose_lu(self) -> tuple[Matrix, Matrix, Matrix]:
        """LU decomposition. Returns (P, L, U) such that P @ A = L @ U."""
        from scipy.linalg import lu

        p, w, u = lu(self._data)
        return Matrix(p), Matrix(w), Matrix(u)

    def decompose_qr(self) -> tuple[Matrix, Matrix]:
        """QR decomposition. Returns (Q, R)."""
        q, r = np.linalg.qr(self._data)
        return Matrix(q), Matrix(r)

    def decompose_svd(self) -> tuple[Matrix, np.ndarray, Matrix]:
        """Singular value decomposition. Returns (U, singular_values, Vt)."""
        u, s, vt = np.linalg.svd(self._data)
        return Matrix(u), s, Matrix(vt)

    def decompose_cholesky(self) -> Matrix:
        """Cholesky decomposition for symmetric positive-definite matrices."""
        if not self.is_square:
            raise MatrixError("Cholesky decomposition requires a square matrix")
        try:
            return Matrix(np.linalg.cholesky(self._data))
        except np.linalg.LinAlgError as exc:
            raise MatrixError(
                "Cholesky decomposition requires a symmetric positive-definite matrix"
            ) from exc

    def eigenvalues(self) -> np.ndarray:
        """Return the eigenvalues of a square matrix."""
        if not self.is_square:
            raise MatrixError("Eigenvalues are only defined for square matrices")
        return np.linalg.eigvals(self._data)

    def eigenvectors(self) -> tuple[np.ndarray, Matrix]:
        """Return (eigenvalues, eigenvectors) of a square matrix."""
        if not self.is_square:
            raise MatrixError("Eigenvectors are only defined for square matrices")
        values, vectors = np.linalg.eig(self._data)
        return values, Matrix(vectors)

    # ------------------------------------------------------------------
    # Operator overloads
    # ------------------------------------------------------------------
    def __matmul__(self, other: Matrix) -> Matrix:
        return self.multiply(other)

    def __add__(self, other: Matrix) -> Matrix:
        if self.shape != other.shape:
            raise MatrixError(f"Cannot add matrices with shapes {self.shape} and {other.shape}")
        return Matrix(self._data + other.data)

    def __sub__(self, other: Matrix) -> Matrix:
        if self.shape != other.shape:
            raise MatrixError(
                f"Cannot subtract matrices with shapes {self.shape} and {other.shape}"
            )
        return Matrix(self._data - other.data)

    def __mul__(self, scalar: float) -> Matrix:
        return Matrix(self._data * scalar)

    __rmul__ = __mul__

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Matrix):
            return NotImplemented
        return bool(np.array_equal(self._data, other.data))

    def __repr__(self) -> str:
        return f"Matrix(shape={self.shape})\n{self._data!r}"

    def __array__(self, dtype: type | None = None) -> np.ndarray:
        return self._data.astype(dtype) if dtype else self._data
