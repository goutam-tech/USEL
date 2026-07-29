import sympy as sp

from usel.maxwell import core, covariant, potentials

N = core.N
x, y, z = N.x, N.y, N.z
t = sp.symbols("t", real=True)


class TestFieldStrengthTensorStructure:
    def test_antisymmetric(self):
        Ex, Ey, Ez, Bx, By, Bz, c = sp.symbols("Ex Ey Ez Bx By Bz c", real=True, positive=True)
        E = Ex * N.i + Ey * N.j + Ez * N.k
        B = Bx * N.i + By * N.j + Bz * N.k
        F = covariant.field_strength_tensor(E, B, c=c)
        assert sp.simplify(F + F.T) == sp.zeros(4, 4)

    def test_diagonal_is_zero(self):
        Ex, c = sp.symbols("Ex c", real=True, positive=True)
        E = Ex * N.i
        B = 0 * N.i
        F = covariant.field_strength_tensor(E, B, c=c)
        for i in range(4):
            assert F[i, i] == 0


class TestInhomogeneousEquation:
    def test_vacuum_plane_wave_reduces_to_gauss_and_ampere_maxwell(self):
        E0, k, mu0, eps0 = sp.symbols("E0 k mu0 epsilon_0", positive=True)
        c = 1 / sp.sqrt(mu0 * eps0)
        w = c * k
        E = E0 * sp.cos(k * z - w * t) * N.i
        B = (E0 / c) * sp.cos(k * z - w * t) * N.j

        F = covariant.field_strength_tensor(E, B, c=c)
        J4 = covariant.four_current(0, 0 * N.i, c=c)
        residuals, ok = covariant.covariant_maxwell_check(F, J4, t, x, y, z, mu0=mu0, c=c)
        assert ok
        assert all(r == 0 for r in residuals)

    def test_static_point_charge_reduces_to_gauss_law(self):
        c, mu0 = sp.symbols("c mu0", positive=True)
        q, eps0 = sp.symbols("q epsilon_0", positive=True)
        r_vec = x * N.i + y * N.j + z * N.k
        r_mag = sp.sqrt(x**2 + y**2 + z**2)
        E = (q / (4 * sp.pi * eps0)) * r_vec / r_mag**3
        B = 0 * N.i
        mu0_val = 1 / (eps0 * c**2)

        F = covariant.field_strength_tensor(E, B, c=c)
        J4 = covariant.four_current(0, 0 * N.i, c=c)
        residuals, ok = covariant.covariant_maxwell_check(F, J4, t, x, y, z, mu0=mu0_val, c=c)
        assert ok

    def test_broken_dispersion_fails(self):
        E0, k, mu0, eps0 = sp.symbols("E0 k mu0 epsilon_0", positive=True)
        c = 1 / sp.sqrt(mu0 * eps0)
        w = 2 * c * k
        E = E0 * sp.cos(k * z - w * t) * N.i
        B = (E0 / c) * sp.cos(k * z - w * t) * N.j

        F = covariant.field_strength_tensor(E, B, c=c)
        J4 = covariant.four_current(0, 0 * N.i, c=c)
        residuals, ok = covariant.covariant_maxwell_check(F, J4, t, x, y, z, mu0=mu0, c=c)
        assert not ok


class TestHomogeneousPair:
    def test_uniform_static_B_and_curl_free_E(self):
        E0, B0 = sp.symbols("E0 B0", real=True)
        E = E0 * N.i
        B = B0 * N.k
        result = covariant.homogeneous_pair_check(E, B, t)
        assert result["all_satisfied"]

    def test_fields_from_a_four_potential_automatically_satisfy_it(self):
        phi = x**2 * sp.sin(y)
        A = sp.cos(t) * N.i + x * z * N.j
        E = potentials.E_from_potentials(phi, A, t)
        B = potentials.B_from_potentials(A)
        result = covariant.homogeneous_pair_check(E, B, t)
        assert result["all_satisfied"]

    def test_broken_pairing_fails(self):
        E0, B0 = sp.symbols("E0 B0", positive=True)
        E = E0 * N.i
        B = B0 * t * N.k
        result = covariant.homogeneous_pair_check(E, B, t)
        assert not result["all_satisfied"]
        assert result["gauss_magnetic"][1]
        assert not result["faraday"][1]


class TestLorentzInvariants:
    def test_invariants_preserved_under_generic_boost(self):
        Ex, Ey, Ez, Bx, By, Bz, v, c = sp.symbols("Ex Ey Ez Bx By Bz v c", real=True, positive=True)
        E = Ex * N.i + Ey * N.j + Ez * N.k
        B = Bx * N.i + By * N.j + Bz * N.k
        (res1, res2), ok = covariant.check_invariants_preserved_under_boost(E, B, v, c=c)
        assert ok
        assert res1 == 0
        assert res2 == 0

    def test_zero_velocity_boost_is_identity(self):
        Ex, Bz, c = sp.symbols("Ex Bz c", real=True, positive=True)
        E = Ex * N.i
        B = Bz * N.k
        E_p, B_p = covariant.lorentz_boost_fields(E, B, 0, c=c)
        assert core.is_zero_vector(E_p - E)
        assert core.is_zero_vector(B_p - B)

    def test_purely_electric_field_becomes_magnetic_under_boost(self):
        Ey, v, c = sp.symbols("Ey v c", positive=True)
        E = Ey * N.j
        B = 0 * N.i
        E_p, B_p = covariant.lorentz_boost_fields(E, B, v, c=c)
        Bz_p = B_p.dot(N.k)
        assert sp.simplify(Bz_p) != 0

    def test_field_invariants_formula(self):
        Ex, Bz, c = sp.symbols("Ex Bz c", real=True, positive=True)
        E = Ex * N.i
        B = Bz * N.k
        inv1, inv2 = covariant.field_invariants(E, B, c=c)
        assert sp.simplify(inv1 - (Ex**2 - c**2 * Bz**2)) == 0
        assert inv2 == 0
