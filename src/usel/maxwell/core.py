from dataclasses import dataclass

import sympy as sp
from sympy.vector import CoordSys3D, Vector, curl, divergence, matrix_to_vector

from .constants import EPSILON_0, MU_0

N = CoordSys3D("N")


def divergence_of(field: Vector, coord_sys: CoordSys3D = N):
    return divergence(field)


def curl_of(field: Vector, coord_sys: CoordSys3D = N):
    return curl(field)


def time_derivative(field: Vector, t: sp.Symbol, coord_sys: CoordSys3D = N) -> Vector:
    matrix = field.to_matrix(coord_sys)
    d_matrix = matrix.diff(t)
    return matrix_to_vector(d_matrix, coord_sys)


def simplify_vector(field: Vector, coord_sys: CoordSys3D = N) -> Vector:
    matrix = sp.simplify(field.to_matrix(coord_sys))
    return matrix_to_vector(matrix, coord_sys)


def _is_close_to_zero(expr, tol: float = 1e-9) -> bool:
    simplified = sp.simplify(expr)
    if simplified == 0:
        return True
    if not simplified.free_symbols:
        try:
            return abs(complex(simplified)) < tol
        except (TypeError, ValueError):
            return False
    return False


def is_zero_vector(field: Vector, coord_sys: CoordSys3D = N) -> bool:
    matrix = sp.simplify(field.to_matrix(coord_sys))
    return all(_is_close_to_zero(component) for component in matrix)


def is_zero_scalar(expr) -> bool:
    return _is_close_to_zero(expr)


def gauss_law_electric(E: Vector, rho, epsilon0=EPSILON_0, coord_sys: CoordSys3D = N):
    residual = sp.simplify(divergence_of(E, coord_sys) - rho / epsilon0)
    return residual, residual == 0


def gauss_law_magnetic(B: Vector, coord_sys: CoordSys3D = N):
    residual = sp.simplify(divergence_of(B, coord_sys))
    return residual, residual == 0


def faraday_law(E: Vector, B: Vector, t: sp.Symbol, coord_sys: CoordSys3D = N):
    residual = simplify_vector(curl_of(E, coord_sys) + time_derivative(B, t, coord_sys), coord_sys)
    return residual, is_zero_vector(residual, coord_sys)


def ampere_maxwell_law(
    B: Vector,
    E: Vector,
    J: Vector,
    t: sp.Symbol,
    mu0=MU_0,
    epsilon0=EPSILON_0,
    coord_sys: CoordSys3D = N,
):
    residual = simplify_vector(
        curl_of(B, coord_sys) - mu0 * J - mu0 * epsilon0 * time_derivative(E, t, coord_sys),
        coord_sys,
    )
    return residual, is_zero_vector(residual, coord_sys)


@dataclass
class MaxwellReport:
    gauss_electric_residual: object
    gauss_electric_ok: bool
    gauss_magnetic_residual: object
    gauss_magnetic_ok: bool
    faraday_residual: object
    faraday_ok: bool
    ampere_maxwell_residual: object
    ampere_maxwell_ok: bool

    @property
    def all_satisfied(self) -> bool:
        return (
            self.gauss_electric_ok
            and self.gauss_magnetic_ok
            and self.faraday_ok
            and self.ampere_maxwell_ok
        )

    def summary(self) -> str:
        lines = [
            f"Gauss (electric):     {'OK' if self.gauss_electric_ok else 'FAILS'}  "
            f"residual = {self.gauss_electric_residual}",
            f"Gauss (magnetic):     {'OK' if self.gauss_magnetic_ok else 'FAILS'}  "
            f"residual = {self.gauss_magnetic_residual}",
            f"Faraday:              {'OK' if self.faraday_ok else 'FAILS'}  "
            f"residual = {self.faraday_residual}",
            f"Ampere-Maxwell:       {'OK' if self.ampere_maxwell_ok else 'FAILS'}  "
            f"residual = {self.ampere_maxwell_residual}",
        ]
        return "\n".join(lines)


def verify_all(
    E: Vector,
    B: Vector,
    rho,
    J: Vector,
    t: sp.Symbol,
    epsilon0=EPSILON_0,
    mu0=MU_0,
    coord_sys: CoordSys3D = N,
) -> MaxwellReport:
    ge_res, ge_ok = gauss_law_electric(E, rho, epsilon0, coord_sys)
    gm_res, gm_ok = gauss_law_magnetic(B, coord_sys)
    fa_res, fa_ok = faraday_law(E, B, t, coord_sys)
    am_res, am_ok = ampere_maxwell_law(B, E, J, t, mu0, epsilon0, coord_sys)

    return MaxwellReport(
        gauss_electric_residual=ge_res,
        gauss_electric_ok=ge_ok,
        gauss_magnetic_residual=gm_res,
        gauss_magnetic_ok=gm_ok,
        faraday_residual=fa_res,
        faraday_ok=fa_ok,
        ampere_maxwell_residual=am_res,
        ampere_maxwell_ok=am_ok,
    )
