from __future__ import annotations

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from usel.schrodinger.results import TimeDependentResult, TimeIndependentResult


def test_time_independent_result_stores_fields():
    energies = np.array([0.5, 1.5, 2.5])
    eigenstates = np.eye(3)
    x = np.linspace(-1, 1, 3)
    result = TimeIndependentResult(energies=energies, eigenstates=eigenstates, x=x)
    assert np.array_equal(result.energies, energies)
    assert np.array_equal(result.eigenstates, eigenstates)
    assert np.array_equal(result.x, x)


def test_time_independent_result_is_frozen():
    result = TimeIndependentResult(
        energies=np.array([1.0]), eigenstates=np.eye(1), x=np.array([0.0])
    )
    with pytest.raises(FrozenInstanceError):
        result.energies = np.array([2.0])


def _build_time_dependent_result(x, psi):
    probability = np.abs(psi) ** 2
    t = np.linspace(0, 1, psi.shape[0])
    return TimeDependentResult(psi=psi, probability=probability, t=t, x=x)


def test_time_dependent_result_default_energies_is_empty():
    x = np.linspace(-1, 1, 5)
    psi = np.ones((3, 5), dtype=complex)
    result = _build_time_dependent_result(x, psi)
    assert result.energies.size == 0


def test_time_dependent_result_norm_property():
    x = np.linspace(0, 1, 5)
    dx = x[1] - x[0]
    psi = np.ones((2, 5), dtype=complex)
    result = _build_time_dependent_result(x, psi)
    expected_norm = np.sum(np.abs(psi) ** 2, axis=1) * dx
    assert np.allclose(result.norm, expected_norm)


def test_time_dependent_result_norm_single_point_grid_uses_dx_one():
    x = np.array([0.0])
    psi = np.array([[2.0]], dtype=complex)
    result = _build_time_dependent_result(x, psi)
    assert result.norm[0] == pytest.approx(4.0)


def test_expectation_x_matches_manual_computation():
    x = np.linspace(-2, 2, 200)
    dx = x[1] - x[0]
    psi = np.exp(-((x - 0.5) ** 2))
    psi_norm = psi / np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    psi_series = psi_norm[np.newaxis, :].astype(complex)
    result = _build_time_dependent_result(x, psi_series)

    expected = np.sum(x * np.abs(psi_norm) ** 2) * dx
    assert result.expectation_x(step=0) == pytest.approx(expected, rel=1e-6)


def test_expectation_x_defaults_to_last_step():
    x = np.linspace(-2, 2, 50)
    psi = np.stack([np.zeros(50, dtype=complex), np.ones(50, dtype=complex)])
    result = _build_time_dependent_result(x, psi)
    assert result.expectation_x() == result.expectation_x(step=-1)


def test_expectation_energy_matches_manual_computation():
    x = np.linspace(-2, 2, 4)
    psi_step = np.array([1.0, 0.0, 0.0, 0.0], dtype=complex)
    psi = psi_step[np.newaxis, :]
    result = _build_time_dependent_result(x, psi)

    hamiltonian = np.diag([3.0, 1.0, 1.0, 1.0])
    expected = np.real(np.conj(psi_step) @ hamiltonian @ psi_step)
    assert result.expectation_energy(hamiltonian, step=0) == pytest.approx(expected)
