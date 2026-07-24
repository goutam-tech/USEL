"""Solvers for the time-independent and time-dependent Schrödinger equation.

Numerical methods include finite-difference matrix diagonalisation for
bound states and the Crank–Nicolson propagator for time evolution.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import eigsh

from usel.exceptions import SolverError, ValidationError
from usel.schrodinger.results import (
    TimeDependentResult,
    TimeIndependentResult,
)

PotentialFunc = Callable[[np.ndarray], np.ndarray]

HBAR = 1.0
MASS = 1.0


def _build_hamiltonian(
    x: np.ndarray,
    potential: np.ndarray,
    hbar: float = HBAR,
    mass: float = MASS,
) -> np.ndarray:
    """Construct the finite-difference Hamiltonian matrix.

    The kinetic energy operator is discretised using the three-point
    central-difference formula.  The potential is added on the diagonal.
    """
    n = len(x)
    dx = x[1] - x[0]
    coeff = hbar**2 / (2.0 * mass * dx**2)

    diag_main = 2.0 * coeff * np.ones(n) + potential
    diag_off = -coeff * np.ones(n - 1)

    H = np.diag(diag_main) + np.diag(diag_off, 1) + np.diag(diag_off, -1)
    return H


def _build_hamiltonian_sparse(
    x: np.ndarray,
    potential: np.ndarray,
    hbar: float = HBAR,
    mass: float = MASS,
) -> csr_matrix:
    """Construct a sparse finite-difference Hamiltonian matrix."""
    n = len(x)
    dx = x[1] - x[0]
    coeff = hbar**2 / (2.0 * mass * dx**2)

    main_diag = 2.0 * coeff * np.ones(n) + potential
    off_diag = -coeff * np.ones(n - 1)

    H = diags([off_diag, main_diag, off_diag], [-1, 0, 1], shape=(n, n), format="csr")
    return H


class SchrodingerSolver:
    """Unified solver for Schrödinger equation problems.

    Parameters
    ----------
    x:
        Spatial grid (1-D array).
    potential:
        Potential energy values on the grid, or a callable ``V(x)``.
    hbar:
        Reduced Planck constant (default 1.0 in natural units).
    mass:
        Particle mass (default 1.0).

    Examples
    --------
    >>> from usel.schrodinger import SchrodingerSolver
    >>> import numpy as np
    >>> x = np.linspace(-5, 5, 500)
    >>> V = 0.5 * x**2
    >>> solver = SchrodingerSolver(x, V)
    >>> result = solver.eigenstates(n_states=5)
    >>> result.energies[:3]
    array([0.5, 1.5, 2.5])
    """

    def __init__(
        self,
        x: np.ndarray,
        potential: np.ndarray | PotentialFunc,
        hbar: float = HBAR,
        mass: float = MASS,
    ) -> None:
        if x.ndim != 1:
            raise ValidationError("Grid x must be 1-dimensional")
        if len(x) < 3:
            raise ValidationError("Grid must contain at least 3 points")

        self._x = x
        self._hbar = hbar
        self._mass = mass

        if callable(potential):
            self._potential = potential(x)
        else:
            potential = np.asarray(potential, dtype=np.float64)
            if potential.shape != x.shape:
                raise ValidationError(
                    f"Potential shape {potential.shape} must match grid shape {x.shape}"
                )
            self._potential = potential

    @property
    def x(self) -> np.ndarray:
        """Spatial grid."""
        return self._x

    @property
    def potential(self) -> np.ndarray:
        """Potential energy array."""
        return self._potential

    @property
    def hamiltonian(self) -> np.ndarray:
        """The full Hamiltonian matrix."""
        return _build_hamiltonian(self._x, self._potential, self._hbar, self._mass)

    def eigenstates(self, n_states: int = 6, which: str = "SA") -> TimeIndependentResult:
        """Compute the lowest ``n_states`` energy eigenstates.

        Parameters
        ----------
        n_states:
            Number of eigenstates to compute.
        which:
            Eigenvalue selection criterion (``"SA"`` for smallest algebraic).

        Returns
        -------
        TimeIndependentResult
            Contains sorted energies, eigenstates, and the spatial grid.
        """
        n_grid = len(self._x)
        if n_states >= n_grid:
            raise SolverError("n_states must be smaller than the grid size")

        H_sparse = _build_hamiltonian_sparse(self._x, self._potential, self._hbar, self._mass)

        try:
            values, vectors = eigsh(H_sparse, k=n_states, which=which)
        except Exception as exc:
            raise SolverError(f"Eigenvalue computation failed: {exc}") from exc

        order = np.argsort(values)
        energies = values[order]
        eigenstates = vectors[:, order]

        # Normalise each eigenstate
        dx = self._x[1] - self._x[0]
        for i in range(n_states):
            norm = np.sqrt(np.sum(np.abs(eigenstates[:, i]) ** 2) * dx)
            if norm > 0:
                eigenstates[:, i] /= norm

        return TimeIndependentResult(energies=energies, eigenstates=eigenstates, x=self._x)

    def solve_time_dependent(
        self,
        psi0: np.ndarray,
        t_end: float,
        dt: float,
    ) -> TimeDependentResult:
        """Propagate an initial state using the Crank–Nicolson method.

        The propagator is:

            ψ(t+dt) = (I + i dt H / (2ℏ))⁻¹ (I − i dt H / (2ℏ)) ψ(t)

        Parameters
        ----------
        psi0:
            Initial wavefunction (complex-valued).
        t_end:
            Final simulation time.
        dt:
            Time step size.

        Returns
        -------
        TimeDependentResult
        """
        if dt <= 0:
            raise ValidationError("Time step dt must be positive")
        if t_end <= 0:
            raise ValidationError("t_end must be positive")
        if psi0.shape != self._x.shape:
            raise ValidationError("psi0 must have the same shape as the grid")

        n_steps = int(round(t_end / dt))
        n_grid = len(self._x)

        H = _build_hamiltonian(self._x, self._potential, self._hbar, self._mass)

        # Crank-Nicolson matrices
        A = np.eye(n_grid) + 0.5j * dt / self._hbar * H
        B = np.eye(n_grid) - 0.5j * dt / self._hbar * H

        # Pre-compute LU factorisation for efficiency
        from scipy.linalg import lu_factor, lu_solve

        A_lu = lu_factor(A)

        psi = psi0.astype(np.complex128).copy()

        psi_history = np.empty((n_steps + 1, n_grid), dtype=np.complex128)
        prob_history = np.empty((n_steps + 1, n_grid), dtype=np.float64)

        psi_history[0] = psi
        prob_history[0] = np.abs(psi) ** 2

        for step in range(n_steps):
            rhs = B @ psi
            psi = lu_solve(A_lu, rhs)
            psi_history[step + 1] = psi
            prob_history[step + 1] = np.abs(psi) ** 2

        t_values = np.linspace(0, t_end, n_steps + 1)

        return TimeDependentResult(
            psi=psi_history,
            probability=prob_history,
            t=t_values,
            x=self._x,
        )

    def solve_time_dependent_split_operator(
        self,
        psi0: np.ndarray,
        t_end: float,
        dt: float,
    ) -> TimeDependentResult:
        """Propagate using the split-operator (FFT) method.

        Splits the Hamiltonian into kinetic and potential parts and
        applies each half-step in their respective domains.

        Parameters
        ----------
        psi0:
            Initial wavefunction (complex-valued).
        t_end:
            Final simulation time.
        dt:
            Time step size.
        """
        if dt <= 0:
            raise ValidationError("Time step dt must be positive")
        if t_end <= 0:
            raise ValidationError("t_end must be positive")
        if psi0.shape != self._x.shape:
            raise ValidationError("psi0 must have the same shape as the grid")

        n_grid = len(self._x)
        n_steps = int(round(t_end / dt))
        dx = self._x[1] - self._x[0]

        k = np.fft.fftfreq(n_grid, d=dx / (2.0 * np.pi))

        # Kinetic energy in momentum space
        kinetic_k = self._hbar**2 * k**2 / (2.0 * self._mass)

        phase_half = np.exp(-1j * dt / 2.0 * kinetic_k / self._hbar)
        phase_pot = np.exp(-1j * dt / self._hbar * self._potential)

        psi = psi0.astype(np.complex128).copy()

        psi_history = np.empty((n_steps + 1, n_grid), dtype=np.complex128)
        prob_history = np.empty((n_steps + 1, n_grid), dtype=np.float64)

        psi_history[0] = psi
        prob_history[0] = np.abs(psi) ** 2

        for step in range(n_steps):
            # Half-step kinetic in k-space
            psi_k = np.fft.fft(psi)
            psi_k *= phase_half
            psi = np.fft.ifft(psi_k)

            # Full-step potential in x-space
            psi *= phase_pot

            # Half-step kinetic in k-space
            psi_k = np.fft.fft(psi)
            psi_k *= phase_half
            psi = np.fft.ifft(psi_k)

            psi_history[step + 1] = psi
            prob_history[step + 1] = np.abs(psi) ** 2

        t_values = np.linspace(0, t_end, n_steps + 1)

        return TimeDependentResult(
            psi=psi_history,
            probability=prob_history,
            t=t_values,
            x=self._x,
        )

    def expectation_value(self, psi: np.ndarray, operator: np.ndarray) -> complex:
        """Compute the expectation value ⟨ψ|O|ψ⟩."""
        dx = self._x[1] - self._x[0]
        return float(np.real(np.conj(psi) @ operator @ psi * dx))

    def probability_current(self, psi: np.ndarray) -> np.ndarray:
        """Compute the probability current j = (ℏ/2mi)(ψ* ∇ψ − ψ ∇ψ*)."""
        dx = self._x[1] - self._x[0]
        dpsi = np.gradient(psi, dx)
        current = (self._hbar / (2.0j * self._mass)) * (np.conj(psi) * dpsi - psi * np.conj(dpsi))
        return np.real(current)
