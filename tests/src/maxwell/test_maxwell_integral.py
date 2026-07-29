import numpy as np
import sympy as sp
from usel.maxwell import integral

t = sp.symbols("t", real=True)
r = sp.symbols("r", positive=True)


class TestGaussLawIntegralSphere:
    def test_point_charge_field_matches_enclosed_charge(self):
        q, eps0 = sp.symbols("q epsilon_0", positive=True)
        E_r = q / (4 * sp.pi * eps0 * r**2)
        residual, ok = integral.gauss_law_integral_sphere(E_r, r, r, q, epsilon0=eps0)
        assert ok
        assert residual == 0

    def test_uniformly_charged_sphere_interior_field(self):
        rho0, eps0 = sp.symbols("rho0 epsilon_0", positive=True)
        E_r = rho0 * r / (3 * eps0)
        Q_enc = rho0 * sp.Rational(4, 3) * sp.pi * r**3
        residual, ok = integral.gauss_law_integral_sphere(E_r, r, r, Q_enc, epsilon0=eps0)
        assert ok
        assert residual == 0

    def test_wrong_enclosed_charge_fails(self):
        q, wrong_q, eps0 = sp.symbols("q wrong_q epsilon_0", positive=True)
        E_r = q / (4 * sp.pi * eps0 * r**2)
        residual, ok = integral.gauss_law_integral_sphere(E_r, r, r, wrong_q, epsilon0=eps0)
        assert not ok


class TestAmpereLawIntegralCircle:
    def test_wire_field_matches_enclosed_current(self):
        I, mu0 = sp.symbols("I mu0", positive=True)
        B_phi = mu0 * I / (2 * sp.pi * r)
        residual, ok = integral.ampere_law_integral_circle(B_phi, r, r, I, mu0=mu0)
        assert ok
        assert residual == 0

    def test_uniform_current_density_interior_field(self):
        J0, mu0 = sp.symbols("J0 mu0", positive=True)
        B_phi = mu0 * J0 * r / 2
        I_enc = J0 * sp.pi * r**2
        residual, ok = integral.ampere_law_integral_circle(B_phi, r, r, I_enc, mu0=mu0)
        assert ok
        assert residual == 0

    def test_wrong_enclosed_current_fails(self):
        I, wrong_I, mu0 = sp.symbols("I wrong_I mu0", positive=True)
        B_phi = mu0 * I / (2 * sp.pi * r)
        residual, ok = integral.ampere_law_integral_circle(B_phi, r, r, wrong_I, mu0=mu0)
        assert not ok


class TestFaradayIntegral:
    def test_matched_emf_and_flux(self):
        B0, w, area = sp.symbols("B0 omega A", positive=True)
        flux_B = B0 * sp.sin(w * t) * area
        emf = -B0 * w * sp.cos(w * t) * area
        residual, ok = integral.faraday_law_integral(emf, flux_B, t)
        assert ok
        assert residual == 0

    def test_mismatched_emf_fails(self):
        B0, w, area = sp.symbols("B0 omega A", positive=True)
        flux_B = B0 * sp.sin(w * t) * area
        wrong_emf = B0 * w * sp.cos(w * t) * area  # sign error
        residual, ok = integral.faraday_law_integral(wrong_emf, flux_B, t)
        assert not ok


class TestAmpereMaxwellIntegral:
    def test_matched_displacement_current(self):
        E0, w, area, mu0, eps0 = sp.symbols("E0 omega A mu0 epsilon_0", positive=True)
        flux_E = E0 * sp.sin(w * t) * area
        I_enc = 0
        circulation_B = mu0 * eps0 * E0 * w * sp.cos(w * t) * area
        residual, ok = integral.ampere_maxwell_law_integral(
            circulation_B, I_enc, flux_E, t, mu0=mu0, epsilon0=eps0
        )
        assert ok
        assert residual == 0

    def test_mismatched_circulation_fails(self):
        E0, w, area, mu0, eps0 = sp.symbols("E0 omega A mu0 epsilon_0", positive=True)
        flux_E = E0 * sp.sin(w * t) * area
        I_enc = 0
        wrong_circulation_B = mu0 * eps0 * E0 * w * sp.sin(w * t) * area  # wrong trig fn
        residual, ok = integral.ampere_maxwell_law_integral(
            wrong_circulation_B, I_enc, flux_E, t, mu0=mu0, epsilon0=eps0
        )
        assert not ok


class TestNumericDifferentialOperators:
    def test_divergence_of_radial_field(self):
        field = lambda x, y, z: np.array([x, y, z])
        div = integral.numeric_divergence(field, (0.7, -0.3, 1.1))
        assert abs(div - 3.0) < 1e-4

    def test_divergence_of_uniform_field_is_zero(self):
        field = lambda x, y, z: np.array([1.0, 2.0, 3.0])
        div = integral.numeric_divergence(field, (0.1, 0.2, 0.3))
        assert abs(div) < 1e-8

    def test_curl_of_rotation_field(self):
        field = lambda x, y, z: np.array([-y, x, 0.0])
        curl_vec = integral.numeric_curl(field, (0.5, 0.5, 0.0))
        np.testing.assert_allclose(curl_vec, [0.0, 0.0, 2.0], atol=1e-4)

    def test_curl_of_irrotational_field_is_zero(self):
        field = lambda x, y, z: np.array([2 * x, 2 * y, 2 * z])
        curl_vec = integral.numeric_curl(field, (0.3, -0.2, 0.6))
        np.testing.assert_allclose(curl_vec, [0.0, 0.0, 0.0], atol=1e-4)


class TestNumericFluxAndCirculation:
    def test_radial_field_flux_through_sphere_matches_divergence_theorem(self):
        R = 1.0
        field = lambda x, y, z: np.array([x, y, z])
        param = lambda u, v: np.array(
            [R * np.sin(u) * np.cos(v), R * np.sin(u) * np.sin(v), R * np.cos(u)]
        )
        flux = integral.numeric_surface_flux(field, param, (1e-6, np.pi - 1e-6), (0, 2 * np.pi))
        expected = 4 * np.pi * R**3
        assert abs(flux - expected) / expected < 1e-3

    def test_rotation_field_circulation_around_circle(self):
        R = 1.5
        field = lambda x, y, z: np.array([-y, x, 0.0])
        curve = lambda tp: np.array([R * np.cos(tp), R * np.sin(tp), 0.0])
        circulation = integral.numeric_line_circulation(field, curve, (0, 2 * np.pi))
        expected = 2 * np.pi * R**2
        assert abs(circulation - expected) / expected < 1e-3


class TestDivergenceTheoremCheck:
    def test_radial_field_over_unit_box(self):
        field = lambda x, y, z: np.array([x, y, z])
        flux, vol_integral, rel_diff, agrees = integral.divergence_theorem_check(
            field, ((0, 1), (0, 1), (0, 1))
        )
        assert agrees
        assert abs(flux - 3.0) < 1e-2
        assert abs(vol_integral - 3.0) < 1e-2

    def test_uniform_field_has_zero_net_flux(self):
        field = lambda x, y, z: np.array([1.0, 0.0, 0.0])
        flux, vol_integral, rel_diff, agrees = integral.divergence_theorem_check(
            field, ((0, 1), (0, 1), (0, 1))
        )
        assert abs(flux) < 1e-6
        assert abs(vol_integral) < 1e-6


class TestStokesTheoremCheck:
    def test_rotation_field_over_unit_square(self):
        field = lambda x, y, z: np.array([-y, x, 0.0])
        circulation, curl_flux, rel_diff, agrees = integral.stokes_theorem_check(
            field, ((0, 1), (0, 1)), plane="xy", offset=0.0
        )
        assert agrees
        assert abs(circulation - 2.0) < 1e-2
        assert abs(curl_flux - 2.0) < 1e-2

    def test_irrotational_field_has_zero_circulation(self):
        field = lambda x, y, z: np.array([2 * x, 2 * y, 2 * z])
        circulation, curl_flux, rel_diff, agrees = integral.stokes_theorem_check(
            field, ((0, 1), (0, 1)), plane="xy", offset=0.0
        )
        assert abs(circulation) < 1e-6
        assert abs(curl_flux) < 1e-6
