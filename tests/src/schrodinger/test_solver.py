from __future__ import annotations

import numpy as np
import pytest

from usel.exceptions import SolverError, ValidationError
from usel.schrodinger.solver import SchrodingerSolver


def make_harmonic_solver(n=400, extent=8.0, omega=1.0, mass=1.0):
    x = np.linspace(-extent, extent, n)
    V = 0.5 * mass * omega**2 * x**2
    return SchrodingerSolver(x, V, mass=mass)


def test_constructor_accepts_array_potential():
    x = np.linspace(-5, 5, 100)
    V = 0.5 * x**2
    solver = SchrodingerSolver(x, V)
    assert np.array_equal(solver.x, x)
    assert np.array_equal(solver.potential, V)


def test_constructor_accepts_callable_potential():
    x = np.linspace(-5, 5, 100)
    solver = SchrodingerSolver(x, lambda arr: 0.5 * arr**2)
    assert np.allclose(solver.potential, 0.5 * x**2)


def test_constructor_rejects_multidimensional_grid():
    x = np.zeros((10, 2))
    with pytest.raises(ValidationError):
        SchrodingerSolver(x, np.zeros((10, 2)))


def test_constructor_rejects_too_few_grid_points():
    x = np.linspace(-1, 1, 2)
    with pytest.raises(ValidationError):
        SchrodingerSolver(x, np.zeros(2))


def test_constructor_rejects_mismatched_potential_shape():
    x = np.linspace(-5, 5, 100)
    V = np.zeros(50)
    with pytest.raises(ValidationError):
        SchrodingerSolver(x, V)


def test_hamiltonian_property_is_hermitian():
    solver = make_harmonic_solver(n=50)
    H = solver.hamiltonian
    assert np.allclose(H, H.conj().T)


def test_eigenstates_matches_harmonic_oscillator_spectrum():
    solver = make_harmonic_solver(n=400, extent=8.0)
    result = solver.eigenstates(n_states=5)
    expected = np.array([0.5, 1.5, 2.5, 3.5, 4.5])
    assert np.allclose(result.energies, expected, atol=0.01)


def test_eigenstates_returns_normalized_states():
    solver = make_harmonic_solver(n=300)
    result = solver.eigenstates(n_states=3)
    dx = solver.x[1] - solver.x[0]
    for i in range(3):
        norm = np.sum(np.abs(result.eigenstates[:, i]) ** 2) * dx
        assert norm == pytest.approx(1.0, abs=1e-6)


def test_eigenstates_rejects_n_states_too_large():
    solver = make_harmonic_solver(n=20)
    with pytest.raises(SolverError):
        solver.eigenstates(n_states=25)


def test_eigenstates_result_contains_grid():
    solver = make_harmonic_solver(n=100)
    result = solver.eigenstates(n_states=2)
    assert np.array_equal(result.x, solver.x)


def test_crank_nicolson_conserves_norm():
    solver = make_harmonic_solver(n=200)
    eig = solver.eigenstates(n_states=1)
    psi0 = eig.eigenstates[:, 0].astype(complex)

    result = solver.solve_time_dependent(psi0, t_end=2.0, dt=0.01)
    assert np.allclose(result.norm, 1.0, atol=1e-8)


def test_crank_nicolson_eigenstate_only_accumulates_phase():
    """Propagating an energy eigenstate should leave |psi| unchanged."""
    solver = make_harmonic_solver(n=200)
    eig = solver.eigenstates(n_states=1)
    psi0 = eig.eigenstates[:, 0].astype(complex)

    result = solver.solve_time_dependent(psi0, t_end=1.0, dt=0.01)
    assert np.allclose(np.abs(result.psi[-1]), np.abs(psi0), atol=1e-4)


def test_solve_time_dependent_rejects_nonpositive_dt():
    solver = make_harmonic_solver(n=50)
    psi0 = np.ones(50, dtype=complex)
    with pytest.raises(ValidationError):
        solver.solve_time_dependent(psi0, t_end=1.0, dt=0.0)


def test_solve_time_dependent_rejects_nonpositive_t_end():
    solver = make_harmonic_solver(n=50)
    psi0 = np.ones(50, dtype=complex)
    with pytest.raises(ValidationError):
        solver.solve_time_dependent(psi0, t_end=0.0, dt=0.01)


def test_solve_time_dependent_rejects_mismatched_psi0_shape():
    solver = make_harmonic_solver(n=50)
    psi0 = np.ones(10, dtype=complex)
    with pytest.raises(ValidationError):
        solver.solve_time_dependent(psi0, t_end=1.0, dt=0.01)


def test_solve_time_dependent_output_shapes():
    solver = make_harmonic_solver(n=50)
    psi0 = np.zeros(50, dtype=complex)
    psi0[25] = 1.0
    result = solver.solve_time_dependent(psi0, t_end=0.1, dt=0.01)
    n_steps_expected = 11  # round(0.1/0.01) + 1
    assert result.psi.shape == (n_steps_expected, 50)
    assert result.t.shape == (n_steps_expected,)


def test_split_operator_conserves_norm():
    solver = make_harmonic_solver(n=256)
    eig = solver.eigenstates(n_states=1)
    psi0 = eig.eigenstates[:, 0].astype(complex)

    result = solver.solve_time_dependent_split_operator(psi0, t_end=2.0, dt=0.01)
    assert np.allclose(result.norm, 1.0, atol=1e-6)


def test_split_operator_and_crank_nicolson_agree_on_short_evolution():
    """Both propagators solve the same Schrodinger equation, so over a
    short time they should produce nearly identical dynamics."""
    solver = make_harmonic_solver(n=256)
    eig = solver.eigenstates(n_states=1)
    psi0 = eig.eigenstates[:, 0].astype(complex)

    cn = solver.solve_time_dependent(psi0, t_end=0.5, dt=0.005)
    split = solver.solve_time_dependent_split_operator(psi0, t_end=0.5, dt=0.005)

    assert np.allclose(np.abs(cn.psi[-1]), np.abs(split.psi[-1]), atol=1e-2)


def test_split_operator_rejects_nonpositive_dt():
    solver = make_harmonic_solver(n=50)
    psi0 = np.ones(50, dtype=complex)
    with pytest.raises(ValidationError):
        solver.solve_time_dependent_split_operator(psi0, t_end=1.0, dt=-0.01)


def test_split_operator_rejects_mismatched_psi0_shape():
    solver = make_harmonic_solver(n=50)
    psi0 = np.ones(30, dtype=complex)
    with pytest.raises(ValidationError):
        solver.solve_time_dependent_split_operator(psi0, t_end=1.0, dt=0.01)


def test_expectation_value_of_identity_is_norm_times_dx():
    solver = make_harmonic_solver(n=100)
    psi = np.ones(100, dtype=complex)
    identity = np.eye(100)
    dx = solver.x[1] - solver.x[0]
    result = solver.expectation_value(psi, identity)
    assert result == pytest.approx(np.sum(np.abs(psi) ** 2) * dx)


def test_expectation_value_of_hamiltonian_on_ground_state_matches_ground_energy():
    """<H> for a normalized eigenstate (Hpsi = E psi) should equal E
    directly, since expectation_value already includes the dx quadrature
    weight in its continuum inner product."""
    solver = make_harmonic_solver(n=400, extent=8.0)
    result = solver.eigenstates(n_states=1)
    psi0 = result.eigenstates[:, 0].astype(complex)
    H = solver.hamiltonian
    energy = solver.expectation_value(psi0, H)
    assert energy == pytest.approx(0.5, abs=0.01)


def test_probability_current_is_zero_for_real_wavefunction():
    solver = make_harmonic_solver(n=200)
    result = solver.eigenstates(n_states=1)
    psi_real = result.eigenstates[:, 0].astype(complex)  # real-valued ground state
    current = solver.probability_current(psi_real)
    assert np.allclose(current, 0.0, atol=1e-8)


def test_probability_current_nonzero_for_moving_wavepacket():
    x = np.linspace(-10, 10, 500)
    solver = SchrodingerSolver(x, np.zeros_like(x))
    k0 = 2.0
    psi = np.exp(-((x) ** 2)) * np.exp(1j * k0 * x)
    current = solver.probability_current(psi.astype(complex))
    core = slice(200, 300)
    assert np.mean(current[core]) > 0
