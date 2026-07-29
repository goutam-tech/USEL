import sympy as sp
from usel.maxwell import core, potentials

N = core.N
x, y, z = N.x, N.y, N.z
t = sp.symbols("t", real=True)


class TestFaradayIsAutomatic:
    def test_static_arbitrary_potentials(self):
        phi = x**2 * y
        A = y * N.i + x * z * N.j
        residual, ok = potentials.faraday_law_is_automatic(phi, A, t)
        assert ok

    def test_time_dependent_arbitrary_potentials(self):
        phi = sp.cos(x) * t
        A = sp.sin(y * t) * N.i + z * t**2 * N.j + x * t * N.k
        residual, ok = potentials.faraday_law_is_automatic(phi, A, t)
        assert ok

    def test_plane_wave_style_potentials(self):
        A0, k, w = sp.symbols("A0 k omega", positive=True)
        phi = 0
        A = A0 * sp.sin(k * z - w * t) * N.i
        residual, ok = potentials.faraday_law_is_automatic(phi, A, t)
        assert ok


class TestElectrostaticPotential:
    def test_coulomb_potential_reproduces_point_charge_field(self):
        q, eps0 = sp.symbols("q epsilon_0", positive=True)
        r = sp.sqrt(x**2 + y**2 + z**2)
        phi = q / (4 * sp.pi * eps0 * r)
        A = 0 * N.i

        E = potentials.E_from_potentials(
            phi,
            A,
            t,
        )
        expected_E = (q / (4 * sp.pi * eps0)) * (x * N.i + y * N.j + z * N.k) / r**3

        residual = core.simplify_vector(E - expected_E)
        assert core.is_zero_vector(residual)

    def test_coulomb_potential_field_satisfies_gauss_law(self):
        q, eps0 = sp.symbols("q epsilon_0", positive=True)
        r = sp.sqrt(x**2 + y**2 + z**2)
        phi = q / (4 * sp.pi * eps0 * r)
        A = 0 * N.i

        E = potentials.E_from_potentials(phi, A, t)
        residual, ok = core.gauss_law_electric(E, rho=0, epsilon0=eps0)
        assert ok


class TestMagnetostaticPotential:
    def test_symmetric_gauge_A_gives_uniform_B(self):
        B0 = sp.symbols("B0", positive=True)
        A = (-B0 * y / 2) * N.i + (B0 * x / 2) * N.j

        B = potentials.B_from_potentials(A)
        residual = core.simplify_vector(B - B0 * N.k)
        assert core.is_zero_vector(residual)

    def test_symmetric_gauge_A_satisfies_coulomb_gauge(self):
        B0 = sp.symbols("B0", positive=True)
        A = (-B0 * y / 2) * N.i + (B0 * x / 2) * N.j
        residual, ok = potentials.check_coulomb_gauge(A)
        assert ok
        assert residual == 0

    def test_uniform_B_satisfies_gauss_law_magnetic(self):
        B0 = sp.symbols("B0", positive=True)
        A = (-B0 * y / 2) * N.i + (B0 * x / 2) * N.j
        B = potentials.B_from_potentials(A)
        residual, ok = core.gauss_law_magnetic(B)
        assert ok


class TestPlaneWaveFromPotentials:
    def _build(self):
        A0, k, mu0, eps0 = sp.symbols("A0 k mu0 epsilon_0", positive=True)
        c = 1 / sp.sqrt(mu0 * eps0)
        w = c * k
        phi = 0
        A = A0 * sp.sin(k * z - w * t) * N.i
        return phi, A, mu0, eps0, c

    def test_coulomb_gauge_holds(self):
        phi, A, mu0, eps0, c = self._build()
        residual, ok = potentials.check_coulomb_gauge(A)
        assert ok

    def test_derived_fields_satisfy_all_four_maxwell_equations(self):
        phi, A, mu0, eps0, c = self._build()
        E = potentials.E_from_potentials(phi, A, t)
        B = potentials.B_from_potentials(A)
        J = 0 * N.i

        report = core.verify_all(E, B, rho=0, J=J, t=t, epsilon0=eps0, mu0=mu0)
        assert report.all_satisfied, report.summary()


class TestLorenzGauge:
    def test_static_potentials_trivially_satisfy_lorenz_gauge(self):
        phi = x**2 - y**2
        A = y * N.i + x * N.j
        residual, ok = potentials.check_lorenz_gauge(A, phi, t)
        assert ok
        assert residual == 0

    def test_lorenz_gauge_holds_for_matched_time_dependent_potentials(self):
        C, c = sp.symbols("C c", positive=True)
        A = C * x * t * N.i
        phi = -C * c**2 * t**2 / 2
        residual, ok = potentials.check_lorenz_gauge(A, phi, t, c=c)
        assert ok
        assert residual == 0

    def test_lorenz_gauge_fails_for_mismatched_potentials(self):
        C, c = sp.symbols("C c", positive=True)
        A = C * x * t * N.i
        phi = -C * c**2 * t**2
        residual, ok = potentials.check_lorenz_gauge(A, phi, t, c=c)
        assert not ok

    def test_scalar_wave_equation_static_vacuum(self):
        residual, ok = potentials.check_lorenz_wave_equation_scalar(0, rho=0, t=t)
        assert ok
        assert residual == 0

    def test_scalar_wave_equation_static_point_charge_potential(self):
        q, eps0 = sp.symbols("q epsilon_0", positive=True)
        r = sp.sqrt(x**2 + y**2 + z**2)
        phi = q / (4 * sp.pi * eps0 * r)
        residual, ok = potentials.check_lorenz_wave_equation_scalar(phi, rho=0, t=t, epsilon0=eps0)
        assert ok

    def test_vector_wave_equation_plane_wave(self):
        A0, k, mu0, eps0 = sp.symbols("A0 k mu0 epsilon_0", positive=True)
        c = 1 / sp.sqrt(mu0 * eps0)
        w = c * k
        A = A0 * sp.sin(k * z - w * t) * N.i
        J = 0 * N.i

        residual, ok = potentials.check_lorenz_wave_equation_vector(A, J, t, mu0=mu0, c=c)
        assert ok

    def test_vector_wave_equation_fails_with_wrong_dispersion(self):
        A0, k, mu0, eps0 = sp.symbols("A0 k mu0 epsilon_0", positive=True)
        c = 1 / sp.sqrt(mu0 * eps0)
        w = 2 * c * k
        A = A0 * sp.sin(k * z - w * t) * N.i
        J = 0 * N.i

        residual, ok = potentials.check_lorenz_wave_equation_vector(A, J, t, mu0=mu0, c=c)
        assert not ok


class TestLaplacianHelper:
    def test_scalar_laplacian_of_quadratic(self):
        lap = potentials.laplacian_of(x**2 + y**2 + z**2)
        assert sp.simplify(lap - 6) == 0

    def test_scalar_laplacian_of_harmonic_function_is_zero(self):
        lap = potentials.laplacian_of(x**2 - y**2)
        assert sp.simplify(lap) == 0

    def test_vector_laplacian_componentwise(self):
        field = (x**2) * N.i + (y**3) * N.j
        lap = potentials.laplacian_of(field)
        expected = 2 * N.i + (6 * y) * N.j
        assert core.is_zero_vector(lap - expected)
