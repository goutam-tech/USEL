import sympy as sp
from usel.maxwell import core
from usel.maxwell.constants import C_LIGHT, EPSILON_0, MU_0, speed_of_light_from_em_constants

N = core.N
x, y, z = N.x, N.y, N.z
t = sp.symbols("t", real=True)


class TestVectorCalculusHelpers:
    def test_time_derivative_of_polynomial_field(self):
        field = t**2 * N.i
        d_field = core.time_derivative(field, t)
        expected = 2 * t * N.i
        assert core.is_zero_vector(d_field - expected)

    def test_time_derivative_of_time_independent_field_is_zero(self):
        field = 3 * N.i + x * N.j
        d_field = core.time_derivative(field, t)
        assert core.is_zero_vector(d_field)

    def test_is_zero_vector_true_case(self):
        assert core.is_zero_vector(0 * N.i)

    def test_is_zero_vector_false_case(self):
        assert not core.is_zero_vector(N.i)

    def test_is_zero_scalar(self):
        assert core.is_zero_scalar(x - x)
        assert not core.is_zero_scalar(x)


class TestGaussLawElectric:
    def test_uniform_field_zero_charge_density(self):
        E0 = sp.symbols("E0", real=True)
        E = E0 * N.i
        residual, ok = core.gauss_law_electric(E, rho=0)
        assert ok
        assert residual == 0

    def test_point_charge_field_away_from_origin(self):
        q, eps0 = sp.symbols("q epsilon_0", positive=True)
        r_vec = x * N.i + y * N.j + z * N.k
        r_mag = sp.sqrt(x**2 + y**2 + z**2)
        E = (q / (4 * sp.pi * eps0)) * r_vec / r_mag**3

        residual, ok = core.gauss_law_electric(E, rho=0, epsilon0=eps0)
        assert ok
        assert residual == 0

    def test_uniform_field_with_wrong_charge_density_fails(self):
        E0, rho0, eps0 = sp.symbols("E0 rho0 epsilon_0", positive=True)
        E = E0 * N.i
        residual, ok = core.gauss_law_electric(E, rho=rho0, epsilon0=eps0)
        assert not ok
        assert residual == -rho0 / eps0

    def test_linear_field_matching_uniform_charge_density(self):
        rho0, eps0 = sp.symbols("rho0 epsilon_0", positive=True)
        r_vec = x * N.i + y * N.j + z * N.k
        E = (rho0 / (3 * eps0)) * r_vec
        residual, ok = core.gauss_law_electric(E, rho=rho0, epsilon0=eps0)
        assert ok
        assert residual == 0


class TestGaussLawMagnetic:
    def test_uniform_field_satisfies_law(self):
        B0 = sp.symbols("B0", real=True)
        B = B0 * N.k
        residual, ok = core.gauss_law_magnetic(B)
        assert ok
        assert residual == 0

    def test_solenoidal_field_satisfies_law(self):
        A = (y**2) * N.i + (x * z) * N.j
        B = core.curl_of(A)
        residual, ok = core.gauss_law_magnetic(B)
        assert ok
        assert residual == 0

    def test_field_with_nonzero_divergence_fails(self):
        B = x * N.i
        residual, ok = core.gauss_law_magnetic(B)
        assert not ok
        assert residual == 1


class TestFaradayLaw:
    def test_static_fields_trivially_satisfy_law(self):
        E0 = sp.symbols("E0", real=True)
        E = E0 * N.i
        B = 0 * N.k
        residual, ok = core.faraday_law(E, B, t)
        assert ok

    def test_time_varying_uniform_B_with_curl_free_E_fails(self):
        E0, B0 = sp.symbols("E0 B0", positive=True)
        E = E0 * N.i
        B = B0 * t * N.k
        residual, ok = core.faraday_law(E, B, t)
        assert not ok
        assert core.is_zero_vector(residual - B0 * N.k)

    def test_matching_time_varying_fields_satisfy_law(self):
        B0 = sp.symbols("B0", positive=True)
        B = B0 * t * N.k
        E = -B0 * x * N.j
        residual, ok = core.faraday_law(E, B, t)
        assert ok
        assert core.is_zero_vector(residual)


class TestAmpereMaxwellLaw:
    def test_static_uniform_fields_no_current_satisfy_law(self):
        B = 0 * N.i
        E = sp.symbols("E0", real=True) * N.i
        J = 0 * N.i
        residual, ok = core.ampere_maxwell_law(B, E, J, t)
        assert ok

    def test_displacement_current_alone_produces_consistent_B(self):
        E0, mu0, eps0 = sp.symbols("E0 mu0 epsilon_0", positive=True)
        E = E0 * t * N.i
        J = 0 * N.i
        C = mu0 * eps0 * E0
        B = C * y * N.k

        residual, ok = core.ampere_maxwell_law(B, E, J, t, mu0=mu0, epsilon0=eps0)
        assert ok
        assert core.is_zero_vector(residual)

    def test_wrong_coefficient_fails(self):
        E0, mu0, eps0 = sp.symbols("E0 mu0 epsilon_0", positive=True)
        E = E0 * t * N.i
        J = 0 * N.i
        wrong_C = 2 * mu0 * eps0 * E0
        B = wrong_C * y * N.k

        residual, ok = core.ampere_maxwell_law(B, E, J, t, mu0=mu0, epsilon0=eps0)
        assert not ok
        assert residual == mu0 * eps0 * E0 * N.i


class TestElectromagneticPlaneWave:
    def _build_wave(self):
        E0, k, mu0, eps0 = sp.symbols("E0 k mu0 epsilon_0", positive=True)
        c = 1 / sp.sqrt(mu0 * eps0)
        w = c * k
        phase = k * z - w * t

        E = E0 * sp.cos(phase) * N.i
        B = (E0 / c) * sp.cos(phase) * N.j
        return E, B, mu0, eps0

    def test_gauss_electric(self):
        E, B, mu0, eps0 = self._build_wave()
        residual, ok = core.gauss_law_electric(E, rho=0, epsilon0=eps0)
        assert ok

    def test_gauss_magnetic(self):
        E, B, mu0, eps0 = self._build_wave()
        residual, ok = core.gauss_law_magnetic(B)
        assert ok

    def test_faraday(self):
        E, B, mu0, eps0 = self._build_wave()
        residual, ok = core.faraday_law(E, B, t)
        assert ok

    def test_ampere_maxwell(self):
        E, B, mu0, eps0 = self._build_wave()
        J = 0 * N.i
        residual, ok = core.ampere_maxwell_law(B, E, J, t, mu0=mu0, epsilon0=eps0)
        assert ok

    def test_verify_all_reports_full_success(self):
        E, B, mu0, eps0 = self._build_wave()
        J = 0 * N.i
        report = core.verify_all(E, B, rho=0, J=J, t=t, epsilon0=eps0, mu0=mu0)
        assert report.all_satisfied
        assert "OK" in report.summary()
        assert "FAILS" not in report.summary()

    def test_verify_all_detects_broken_dispersion_relation(self):
        E0, k, mu0, eps0 = sp.symbols("E0 k mu0 epsilon_0", positive=True)
        c = 1 / sp.sqrt(mu0 * eps0)
        w = c * k
        phase = k * z - w * t

        E = E0 * sp.cos(phase) * N.i
        B = (E0 / (2 * c)) * sp.cos(phase) * N.j
        J = 0 * N.i

        report = core.verify_all(E, B, rho=0, J=J, t=t, epsilon0=eps0, mu0=mu0)
        assert report.gauss_electric_ok
        assert report.gauss_magnetic_ok
        assert not report.faraday_ok
        assert not report.ampere_maxwell_ok
        assert not report.all_satisfied


class TestConstants:
    def test_speed_of_light_matches_codata_within_tolerance(self):
        c_derived = float(speed_of_light_from_em_constants(EPSILON_0, MU_0))
        c_codata = float(C_LIGHT)
        relative_error = abs(c_derived - c_codata) / c_codata
        assert relative_error < 1e-3

    def test_constants_are_positive(self):
        assert float(EPSILON_0) > 0
        assert float(MU_0) > 0
        assert float(C_LIGHT) > 0
