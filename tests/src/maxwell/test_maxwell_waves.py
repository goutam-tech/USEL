import sympy as sp
import pytest

from usel.maxwell import core, waves
from usel.maxwell.constants import EPSILON_0, MU_0

N = core.N
x, y, z = N.x, N.y, N.z
t = sp.symbols("t", real=True)


class TestWaveEquation:
    def test_plane_wave_satisfies_scalar_wave_equation_per_component(self):
        E0, k, mu0, eps0 = sp.symbols("E0 k mu0 epsilon_0", positive=True)
        c = 1 / sp.sqrt(mu0 * eps0)
        w = c * k
        field = E0 * sp.cos(k * z - w * t)
        residual, ok = waves.check_wave_equation_scalar(field, t, mu0=mu0, epsilon0=eps0)
        assert ok
        assert residual == 0

    def test_plane_wave_satisfies_vector_wave_equation(self):
        E0, k, mu0, eps0 = sp.symbols("E0 k mu0 epsilon_0", positive=True)
        c = 1 / sp.sqrt(mu0 * eps0)
        w = c * k
        E = E0 * sp.cos(k * z - w * t) * N.i
        residual, ok = waves.check_wave_equation_vector(E, t, mu0=mu0, epsilon0=eps0)
        assert ok

    def test_wrong_dispersion_fails_wave_equation(self):
        E0, k, mu0, eps0 = sp.symbols("E0 k mu0 epsilon_0", positive=True)
        c = 1 / sp.sqrt(mu0 * eps0)
        w = 3 * c * k  # wrong
        E = E0 * sp.cos(k * z - w * t) * N.i
        residual, ok = waves.check_wave_equation_vector(E, t, mu0=mu0, epsilon0=eps0)
        assert not ok

    def test_static_field_trivially_satisfies_wave_equation(self):
        E0 = sp.symbols("E0", real=True)
        E = E0 * N.i  # uniform, time-independent: laplacian=0, d2/dt2=0
        residual, ok = waves.check_wave_equation_vector(E, t)
        assert ok
        assert core.is_zero_vector(residual)


class TestDispersionRelation:
    def test_matched_dispersion(self):
        k, mu0, eps0 = sp.symbols("k mu0 epsilon_0", positive=True)
        c = 1 / sp.sqrt(mu0 * eps0)
        w = c * k
        residual, ok = waves.check_dispersion_relation(k, w, mu0=mu0, epsilon0=eps0)
        assert ok
        assert residual == 0

    def test_mismatched_dispersion_fails(self):
        k, mu0, eps0 = sp.symbols("k mu0 epsilon_0", positive=True)
        c = 1 / sp.sqrt(mu0 * eps0)
        w = sp.Rational(3, 2) * c * k
        residual, ok = waves.check_dispersion_relation(k, w, mu0=mu0, epsilon0=eps0)
        assert not ok


class TestPlaneWaveConstruction:
    def _build(self, polarization):
        E0, k, mu0, eps0 = sp.symbols("E0 k mu0 epsilon_0", positive=True)
        c = 1 / sp.sqrt(mu0 * eps0)
        w = c * k
        E, B = waves.build_plane_wave_z(
            E0, k, w, t, polarization=polarization, mu0=mu0, epsilon0=eps0
        )
        return E, B, mu0, eps0

    def test_x_polarized_wave_satisfies_all_four_maxwell_equations(self):
        E, B, mu0, eps0 = self._build(N.i)
        J = 0 * N.i
        report = core.verify_all(E, B, rho=0, J=J, t=t, epsilon0=eps0, mu0=mu0)
        assert report.all_satisfied, report.summary()

    def test_y_polarized_wave_satisfies_all_four_maxwell_equations(self):
        E, B, mu0, eps0 = self._build(N.j)
        J = 0 * N.i
        report = core.verify_all(E, B, rho=0, J=J, t=t, epsilon0=eps0, mu0=mu0)
        assert report.all_satisfied, report.summary()

    def test_transversality_x_polarized(self):
        E, B, mu0, eps0 = self._build(N.i)
        results = waves.transverse_check(E, B, N.k)
        assert all(ok for (_, ok) in results.values())

    def test_transversality_y_polarized(self):
        E, B, mu0, eps0 = self._build(N.j)
        results = waves.transverse_check(E, B, N.k)
        assert all(ok for (_, ok) in results.values())

    def test_wave_equation_holds_for_constructed_wave(self):
        E, B, mu0, eps0 = self._build(N.i)
        _, ok_E = waves.check_wave_equation_vector(E, t, mu0=mu0, epsilon0=eps0)
        _, ok_B = waves.check_wave_equation_vector(B, t, mu0=mu0, epsilon0=eps0)
        assert ok_E
        assert ok_B
