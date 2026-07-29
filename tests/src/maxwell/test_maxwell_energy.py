import sympy as sp
import pytest

from usel.maxwell import core, energy, waves
from usel.maxwell.constants import EPSILON_0, MU_0, C_LIGHT

N = core.N
x, y, z = N.x, N.y, N.z
t = sp.symbols("t", real=True)


class TestPoyntingsTheorem:
    def test_vacuum_plane_wave_satisfies_poyntings_theorem(self):
        E0, k, mu0, eps0 = sp.symbols("E0 k mu0 epsilon_0", positive=True)
        c = 1 / sp.sqrt(mu0 * eps0)
        w = c * k
        E, B = waves.build_plane_wave_z(E0, k, w, t, polarization=N.i, mu0=mu0, epsilon0=eps0)
        J = 0 * N.i
        residual, ok = energy.check_poynting_theorem(E, B, J, t, epsilon0=eps0, mu0=mu0)
        assert ok

    def test_static_fields_trivially_satisfy_poyntings_theorem(self):
        E0, B0 = sp.symbols("E0 B0", real=True)
        E = E0 * N.i
        B = B0 * N.k
        J = 0 * N.i
        residual, ok = energy.check_poynting_theorem(E, B, J, t)
        assert ok
        assert residual == 0

    def test_ohmic_dissipation_case(self):
        E0, sigma = sp.symbols("E0 sigma", positive=True)
        E = E0 * N.i
        B = 0 * N.i
        J = sigma * E
        residual, ok = energy.check_poynting_theorem(E, B, J, t)
        assert not ok


class TestStressTensor:
    def test_trace_identity_generic_fields(self):
        Ex, Ey, Ez, Bx, By, Bz = sp.symbols("Ex Ey Ez Bx By Bz", real=True)
        E = Ex * N.i + Ey * N.j + Ez * N.k
        B = Bx * N.i + By * N.j + Bz * N.k
        residual, ok = energy.stress_tensor_trace_identity_check(E, B)
        assert ok
        assert residual == 0

    def test_tensor_is_symmetric(self):
        Ex, Ey, Bz = sp.symbols("Ex Ey Bz", real=True)
        E = Ex * N.i + Ey * N.j
        B = Bz * N.k
        T = energy.maxwell_stress_tensor(E, B)
        assert sp.simplify(T - T.T) == sp.zeros(3, 3)

    def test_pure_E_field_diagonal_structure(self):
        E0, eps0 = sp.symbols("E0 epsilon_0", positive=True)
        E = E0 * N.i
        B = 0 * N.i
        T = energy.maxwell_stress_tensor(E, B, epsilon0=eps0)
        assert sp.simplify(T[0, 0] - eps0 * E0**2 / 2) == 0
        assert sp.simplify(T[1, 1] + eps0 * E0**2 / 2) == 0
        assert sp.simplify(T[2, 2] + eps0 * E0**2 / 2) == 0


class TestLorentzForce:
    def test_perpendicular_E_and_v_cross_B(self):
        q, E0, B0, v0 = sp.symbols("q E0 B0 v0", positive=True)
        E = E0 * N.i
        B = B0 * N.k
        v = v0 * N.j
        F = energy.lorentz_force(q, E, B, v)
        expected = q * (E0 + v0 * B0) * N.i
        assert core.is_zero_vector(F - expected)

    def test_velocity_parallel_to_B_gives_pure_electric_force(self):
        q, E0, B0 = sp.symbols("q E0 B0", real=True)
        E = E0 * N.i
        B = B0 * N.k
        v = sp.symbols("v0", real=True) * N.k
        F = energy.lorentz_force(q, E, B, v)
        assert core.is_zero_vector(F - q * E0 * N.i)

    def test_zero_charge_gives_zero_force(self):
        E0, B0, v0 = sp.symbols("E0 B0 v0", real=True)
        E = E0 * N.i
        B = B0 * N.k
        v = v0 * N.j
        F = energy.lorentz_force(0, E, B, v)
        assert core.is_zero_vector(F)


class TestEnergyFluxSpeed:
    def test_poynting_over_energy_density_equals_c_for_plane_wave(self):
        E0, k, mu0, eps0 = sp.symbols("E0 k mu0 epsilon_0", positive=True)
        c = 1 / sp.sqrt(mu0 * eps0)
        w = c * k
        E, B = waves.build_plane_wave_z(E0, k, w, t, polarization=N.i, mu0=mu0, epsilon0=eps0)

        S = energy.poynting_vector(E, B, mu0=mu0)
        u = energy.energy_density(E, B, epsilon0=eps0, mu0=mu0)

        S_mag_sq = sp.simplify(S.dot(S))
        target_sq = sp.simplify((c * u) ** 2)
        assert sp.simplify(S_mag_sq - target_sq) == 0


class TestRadiationPressure:
    def test_reflecting_surface_is_double_absorbing(self):
        I0, c = sp.symbols("I0 c", positive=True)
        P_abs = energy.radiation_pressure(I0, c=c, reflecting=False)
        P_ref = energy.radiation_pressure(I0, c=c, reflecting=True)
        assert sp.simplify(P_ref - 2 * P_abs) == 0

    def test_absorbing_formula(self):
        I0, c = sp.symbols("I0 c", positive=True)
        P = energy.radiation_pressure(I0, c=c, reflecting=False)
        assert sp.simplify(P - I0 / c) == 0
