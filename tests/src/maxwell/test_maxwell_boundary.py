import sympy as sp
from usel.maxwell import boundary, core

N = core.N
x, y, z = N.x, N.y, N.z


class TestDecomposeField:
    def test_perp_and_parallel_recombine_to_original(self):
        Fx, Fy, Fz = sp.symbols("Fx Fy Fz", real=True)
        F = Fx * N.i + Fy * N.j + Fz * N.k
        normal = N.k
        F_perp, F_par = boundary.decompose_field(F, normal)
        assert core.is_zero_vector(F_perp + F_par - F)

    def test_perp_component_is_along_normal(self):
        Fx, Fy, Fz = sp.symbols("Fx Fy Fz", real=True)
        F = Fx * N.i + Fy * N.j + Fz * N.k
        normal = N.k
        F_perp, _ = boundary.decompose_field(F, normal)
        assert core.is_zero_vector(F_perp - Fz * N.k)

    def test_parallel_component_is_orthogonal_to_normal(self):
        Fx, Fy, Fz = sp.symbols("Fx Fy Fz", real=True)
        F = Fx * N.i + Fy * N.j + Fz * N.k
        normal = N.k
        _, F_par = boundary.decompose_field(F, normal)
        assert sp.simplify(F_par.dot(normal)) == 0


class TestTangentialEContinuity:
    def test_identical_fields_are_continuous(self):
        Ex, Ey, Ez = sp.symbols("Ex Ey Ez", real=True)
        E = Ex * N.i + Ey * N.j + Ez * N.k
        residual, ok = boundary.check_tangential_E_continuity(E, E, N.k)
        assert ok

    def test_matched_tangential_mismatched_normal_still_continuous(self):
        """Only the tangential (x,y) component matters; the normal (z)
        component of E is allowed to differ across the interface."""
        Ex, Ey, Ez1, Ez2 = sp.symbols("Ex Ey Ez1 Ez2", real=True)
        E1 = Ex * N.i + Ey * N.j + Ez1 * N.k
        E2 = Ex * N.i + Ey * N.j + Ez2 * N.k
        residual, ok = boundary.check_tangential_E_continuity(E1, E2, N.k)
        assert ok

    def test_mismatched_tangential_component_fails(self):
        Ex1, Ex2 = sp.symbols("Ex1 Ex2", positive=True)
        E1 = Ex1 * N.i
        E2 = Ex2 * N.i
        residual, ok = boundary.check_tangential_E_continuity(E1, E2, N.k)
        assert not ok


class TestNormalBContinuity:
    def test_matched_normal_component_is_continuous(self):
        Bz = sp.symbols("Bz", real=True)
        B1 = Bz * N.k + sp.symbols("Bx1", real=True) * N.i
        B2 = Bz * N.k + sp.symbols("Bx2", real=True) * N.i
        residual, ok = boundary.check_normal_B_continuity(B1, B2, N.k)
        assert ok
        assert residual == 0

    def test_mismatched_normal_component_fails(self):
        Bz1, Bz2 = sp.symbols("Bz1 Bz2", positive=True)
        B1 = Bz1 * N.k
        B2 = Bz2 * N.k
        residual, ok = boundary.check_normal_B_continuity(B1, B2, N.k)
        assert not ok


class TestNormalDDiscontinuity:
    def test_matched_surface_charge(self):
        sigma_f = sp.symbols("sigma_f", real=True)
        D1 = sigma_f * N.k
        D2 = 0 * N.k
        residual, ok = boundary.check_normal_D_discontinuity(D1, D2, N.k, sigma_f)
        assert ok
        assert residual == 0

    def test_zero_surface_charge_requires_continuity(self):
        Dz = sp.symbols("Dz", real=True)
        D1 = Dz * N.k
        D2 = Dz * N.k
        residual, ok = boundary.check_normal_D_discontinuity(D1, D2, N.k, sigma_free=0)
        assert ok

    def test_wrong_surface_charge_fails(self):
        sigma_f, wrong_sigma = sp.symbols("sigma_f wrong_sigma", positive=True)
        D1 = sigma_f * N.k
        D2 = 0 * N.k
        residual, ok = boundary.check_normal_D_discontinuity(D1, D2, N.k, wrong_sigma)
        assert not ok


class TestTangentialHDiscontinuity:
    def test_matched_surface_current(self):
        Kx = sp.symbols("Kx", real=True)
        K = Kx * N.i
        H1 = -Kx * N.j
        H2 = 0 * N.j
        residual, ok = boundary.check_tangential_H_discontinuity(H1, H2, N.k, K)
        assert ok
        assert core.is_zero_vector(residual)

    def test_zero_surface_current_requires_continuity(self):
        Hx, Hy = sp.symbols("Hx Hy", real=True)
        H = Hx * N.i + Hy * N.j
        residual, ok = boundary.check_tangential_H_discontinuity(H, H, N.k, K_free=0 * N.i)
        assert ok

    def test_wrong_surface_current_fails(self):
        Kx, wrong_Kx = sp.symbols("Kx wrong_Kx", positive=True)
        K = Kx * N.i
        H1 = -Kx * N.j
        H2 = 0 * N.j
        wrong_K = wrong_Kx * N.i
        residual, ok = boundary.check_tangential_H_discontinuity(H1, H2, N.k, wrong_K)
        assert not ok


class TestFresnelNormalIncidence:
    def test_matched_media_no_reflection(self):
        n = sp.symbols("n", positive=True)
        r, tcoef = boundary.normal_incidence_fresnel(n, n)
        assert sp.simplify(r) == 0
        assert sp.simplify(tcoef - 1) == 0

    def test_air_to_glass(self):
        r, tcoef = boundary.normal_incidence_fresnel(1, sp.Rational(3, 2))
        assert sp.simplify(r - sp.Rational(-1, 5)) == 0
        assert sp.simplify(tcoef - sp.Rational(4, 5)) == 0

    def test_energy_conservation_generic_indices(self):
        n1, n2 = sp.symbols("n1 n2", positive=True)
        R, T, conserved = boundary.normal_incidence_power_coefficients(n1, n2)
        assert conserved

    def test_energy_conservation_air_to_glass(self):
        R, T, conserved = boundary.normal_incidence_power_coefficients(1, sp.Rational(3, 2))
        assert conserved
        assert sp.simplify(R + T - 1) == 0
