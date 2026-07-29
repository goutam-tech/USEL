import pytest
import sympy as sp
from usel.maxwell import core, materials
from usel.maxwell.constants import EPSILON_0, MU_0

N = core.N
x, y, z = N.x, N.y, N.z
t = sp.symbols("t", real=True)


class TestRoundTrips:
    def test_round_trip_D_to_E_symbolic_material(self):
        eps_r = sp.symbols("epsilon_r", positive=True)
        mat = materials.LinearIsotropicMaterial("generic", epsilon_r=eps_r, mu_r=1)
        E0 = sp.symbols("E0", real=True)
        E = E0 * N.i
        residual, ok = materials.round_trip_D_to_E(E, mat)
        assert ok
        assert core.is_zero_vector(residual)

    def test_round_trip_H_to_B_symbolic_material(self):
        mu_r = sp.symbols("mu_r", positive=True)
        mat = materials.LinearIsotropicMaterial("generic", epsilon_r=1, mu_r=mu_r)
        B0 = sp.symbols("B0", real=True)
        B = B0 * N.k
        residual, ok = materials.round_trip_H_to_B(B, mat)
        assert ok

    @pytest.mark.parametrize(
        "mat",
        [materials.VACUUM, materials.AIR, materials.WATER, materials.GLASS, materials.SILICON],
    )
    def test_round_trips_for_all_predefined_materials(self, mat):
        E0, B0 = sp.symbols("E0 B0", real=True)
        E = E0 * N.i + 2 * E0 * N.j
        B = B0 * N.k

        _, ok_de = materials.round_trip_D_to_E(E, mat)
        _, ok_hb = materials.round_trip_H_to_B(B, mat)
        assert ok_de
        assert ok_hb

    def test_vacuum_D_equals_epsilon0_E(self):
        E0 = sp.symbols("E0", real=True)
        E = E0 * N.i
        D = materials.D_from_E(E, materials.VACUUM)
        residual = core.simplify_vector(D - EPSILON_0 * E)
        assert core.is_zero_vector(residual)

    def test_vacuum_H_equals_B_over_mu0(self):
        B0 = sp.symbols("B0", real=True)
        B = B0 * N.k
        H = materials.H_from_B(B, materials.VACUUM)
        residual = core.simplify_vector(H - B / MU_0)
        assert core.is_zero_vector(residual)


class TestDecompositions:
    def test_D_decomposition_matches_direct_D(self):
        eps_r = sp.symbols("epsilon_r", positive=True)
        mat = materials.LinearIsotropicMaterial("dielectric", epsilon_r=eps_r, mu_r=1)
        E0 = sp.symbols("E0", real=True)
        E = E0 * N.j
        residual, ok = materials.verify_D_decomposition(E, mat)
        assert ok

    def test_H_decomposition_matches_direct_H(self):
        mu_r = sp.symbols("mu_r", positive=True)
        mat = materials.LinearIsotropicMaterial("magnetic", epsilon_r=1, mu_r=mu_r)
        B0 = sp.symbols("B0", real=True)
        B = B0 * N.i
        residual, ok = materials.verify_H_decomposition(B, mat)
        assert ok

    def test_susceptibilities_derived_correctly(self):
        mat = materials.dielectric(epsilon_r=4)
        assert mat.chi_e == 3
        magnet = materials.LinearIsotropicMaterial("m", epsilon_r=1, mu_r=5)
        assert magnet.chi_m == 4

    def test_refractive_index_glass(self):
        n = materials.GLASS.refractive_index
        assert sp.simplify(n - sp.sqrt(materials.GLASS.epsilon_r)) == 0
        assert abs(float(n) - 1.5) < 1e-6


class TestMaxwellInMatter:
    def test_gauss_law_in_matter_uniform_D_zero_rho(self):
        D0 = sp.symbols("D0", real=True)
        D = D0 * N.i
        residual, ok = materials.gauss_law_electric_in_matter(D, rho_free=0)
        assert ok
        assert residual == 0

    def test_gauss_law_in_matter_matches_free_charge(self):
        rho_f, eps = sp.symbols("rho_f epsilon", positive=True)
        r_vec = x * N.i + y * N.j + z * N.k
        D = (rho_f / 3) * r_vec
        residual, ok = materials.gauss_law_electric_in_matter(D, rho_free=rho_f)
        assert ok
        assert residual == 0

    def test_gauss_law_in_matter_fails_for_wrong_free_charge(self):
        rho_f, wrong_rho = sp.symbols("rho_f wrong_rho", positive=True)
        r_vec = x * N.i + y * N.j + z * N.k
        D = (rho_f / 3) * r_vec
        residual, ok = materials.gauss_law_electric_in_matter(D, rho_free=wrong_rho)
        assert not ok

    def test_ampere_maxwell_in_matter_displacement_current(self):
        H0, mu0, eps = sp.symbols("H0 mu0 epsilon", positive=True)
        D = H0 * t * N.i
        J_free = 0 * N.i
        C = H0
        H = C * y * N.k

        residual, ok = materials.ampere_maxwell_law_in_matter(H, J_free, D, t)
        assert ok
        assert core.is_zero_vector(residual)

    def test_ampere_maxwell_in_matter_fails_with_conduction_current_mismatch(self):
        H0 = sp.symbols("H0", positive=True)
        D = 0 * N.i
        H = 0 * N.i
        J_free = H0 * N.i
        residual, ok = materials.ampere_maxwell_law_in_matter(H, J_free, D, t)
        assert not ok


class TestOhmsLaw:
    def test_ohms_law_copper(self):
        E0 = sp.symbols("E0", real=True)
        E = E0 * N.i
        J = materials.ohms_law(E, materials.COPPER)
        residual = core.simplify_vector(J - materials.COPPER.sigma * E)
        assert core.is_zero_vector(residual)

    def test_ohms_law_insulator_gives_zero_current(self):
        E0 = sp.symbols("E0", real=True)
        E = E0 * N.i
        J = materials.ohms_law(E, materials.dielectric(epsilon_r=5))
        assert core.is_zero_vector(J)
