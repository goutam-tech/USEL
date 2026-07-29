import sympy as sp

from .constants import EPSILON_0, MU_0
from .core import N, divergence_of, is_zero_scalar
from .integral import ampere_law_integral_circle, gauss_law_integral_sphere


def coulombs_law(q1, q2, r, epsilon0=EPSILON_0):
    return q1 * q2 / (4 * sp.pi * epsilon0 * r**2)


def coulomb_field_matches_gauss(q, r_symbol: sp.Symbol, r_val, epsilon0=EPSILON_0):
    E_r = q / (4 * sp.pi * epsilon0 * r_symbol**2)
    return gauss_law_integral_sphere(E_r, r_symbol, r_val, Q_enc=q, epsilon0=epsilon0)


def biot_savart_wire_field(current, r, mu0=MU_0):
    return mu0 * current / (2 * sp.pi * r)


def biot_savart_matches_ampere(current, r_symbol: sp.Symbol, r_val, mu0=MU_0):
    B_phi = mu0 * current / (2 * sp.pi * r_symbol)
    return ampere_law_integral_circle(B_phi, r_symbol, r_val, I_enc=current, mu0=mu0)


def continuity_equation_residual_1d(Jx, rho, x: sp.Symbol, t: sp.Symbol):
    return sp.diff(Jx, x) + sp.diff(rho, t)


def check_continuity_equation_1d(Jx, rho, x: sp.Symbol, t: sp.Symbol):
    residual = sp.simplify(continuity_equation_residual_1d(Jx, rho, x, t))
    return residual, is_zero_scalar(residual)


def continuity_equation_residual(J, rho, t: sp.Symbol, coord_sys=N):
    return divergence_of(J, coord_sys) + sp.diff(rho, t)


def check_continuity_equation(J, rho, t: sp.Symbol, coord_sys=N):
    residual = sp.simplify(continuity_equation_residual(J, rho, t, coord_sys))
    return residual, is_zero_scalar(residual)


def skin_depth(mu0, sigma, omega):
    return sp.sqrt(2 / (mu0 * sigma * omega))
