import sympy as sp
from usel.maxwell import core, special_cases
from usel.maxwell.constants import EPSILON_0, MU_0

N = core.N
x, y, z = N.x, N.y, N.z
t = sp.symbols("t", real=True)
r = sp.symbols("r", positive=True)


class TestCoulombsLaw:
    def test_matches_known_numeric_value(self):
        F = special_cases.coulombs_law(1, 1, 1, epsilon0=EPSILON_0)
        assert abs(float(F) - 8.9875e9) / 8.9875e9 < 1e-3

    def test_force_scales_inverse_square(self):
        q1, q2, eps0 = sp.symbols("q1 q2 epsilon_0", positive=True)
        F_r = special_cases.coulombs_law(q1, q2, r, epsilon0=eps0)
        F_2r = special_cases.coulombs_law(q1, q2, 2 * r, epsilon0=eps0)
        assert sp.simplify(F_r / F_2r - 4) == 0

    def test_matches_gauss_law_derivation(self):
        q, eps0 = sp.symbols("q epsilon_0", positive=True)
        residual, ok = special_cases.coulomb_field_matches_gauss(q, r, r, epsilon0=eps0)
        assert ok
        assert residual == 0


class TestBiotSavartLaw:
    def test_matches_ampere_law_derivation(self):
        I, mu0 = sp.symbols("I mu0", positive=True)
        residual, ok = special_cases.biot_savart_matches_ampere(I, r, r, mu0=mu0)
        assert ok
        assert residual == 0

    def test_field_scales_inverse_with_distance(self):
        I, mu0 = sp.symbols("I mu0", positive=True)
        B_r = special_cases.biot_savart_wire_field(I, r, mu0=mu0)
        B_2r = special_cases.biot_savart_wire_field(I, 2 * r, mu0=mu0)
        assert sp.simplify(B_r / B_2r - 2) == 0


class TestContinuityEquation1D:
    def test_matched_wave_pair_satisfies_continuity(self):
        rho0, k, w = sp.symbols("rho0 k omega", positive=True)
        xsym = sp.symbols("x", real=True)
        rho = rho0 * sp.cos(k * xsym - w * t)
        Jx = (rho0 * w / k) * sp.cos(k * xsym - w * t)
        residual, ok = special_cases.check_continuity_equation_1d(Jx, rho, xsym, t)
        assert ok
        assert residual == 0

    def test_mismatched_current_fails(self):
        rho0, k, w = sp.symbols("rho0 k omega", positive=True)
        xsym = sp.symbols("x", real=True)
        rho = rho0 * sp.cos(k * xsym - w * t)
        wrong_Jx = rho0 * sp.cos(k * xsym - w * t)
        residual, ok = special_cases.check_continuity_equation_1d(wrong_Jx, rho, xsym, t)
        assert not ok

    def test_static_charge_no_current_satisfies_continuity(self):
        rho0 = sp.symbols("rho0", real=True)
        xsym = sp.symbols("x", real=True)
        residual, ok = special_cases.check_continuity_equation_1d(0, rho0, xsym, t)
        assert ok


class TestContinuityEquation3D:
    def test_linearly_growing_uniform_charge_matched_current(self):
        rho0 = sp.symbols("rho0", positive=True)
        rho = rho0 * t
        J = -rho0 * x * N.i
        residual, ok = special_cases.check_continuity_equation(J, rho, t)
        assert ok
        assert residual == 0

    def test_mismatched_current_fails(self):
        rho0, wrong_rho0 = sp.symbols("rho0 wrong_rho0", positive=True)
        rho = rho0 * t
        J = -wrong_rho0 * x * N.i
        residual, ok = special_cases.check_continuity_equation(J, rho, t)
        assert not ok

    def test_steady_state_zero_current_zero_charge_rate(self):
        J = 0 * N.i
        rho = sp.symbols("rho0", real=True)
        residual, ok = special_cases.check_continuity_equation(J, rho, t)
        assert ok
        assert residual == 0


class TestSkinDepth:
    def test_copper_at_60hz_matches_known_value(self):
        omega = 2 * sp.pi * 60
        delta = special_cases.skin_depth(MU_0, sp.Float(5.96e7), omega)
        assert abs(float(delta) - 8.5e-3) / 8.5e-3 < 0.05

    def test_skin_depth_decreases_with_frequency(self):
        sigma = sp.Float(5.96e7)
        delta_low = special_cases.skin_depth(MU_0, sigma, 2 * sp.pi * 60)
        delta_high = special_cases.skin_depth(MU_0, sigma, 2 * sp.pi * 6e6)
        assert float(delta_high) < float(delta_low)
