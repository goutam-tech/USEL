import sympy as sp
from sympy.vector import CoordSys3D, Vector, gradient, matrix_to_vector

from .constants import C_LIGHT, EPSILON_0, MU_0
from .core import (
    N,
    curl_of,
    divergence_of,
    faraday_law,
    is_zero_scalar,
    is_zero_vector,
    simplify_vector,
    time_derivative,
)


def gradient_of(scalar_field, coord_sys: CoordSys3D = N) -> Vector:
    return gradient(scalar_field)


def laplacian_of(field, coord_sys: CoordSys3D = N):
    if isinstance(field, Vector):
        matrix = field.to_matrix(coord_sys)
        lap_matrix = matrix.applyfunc(
            lambda comp: divergence_of(gradient_of(comp, coord_sys), coord_sys)
        )
        return matrix_to_vector(lap_matrix, coord_sys)
    return divergence_of(gradient_of(field, coord_sys), coord_sys)


def E_from_potentials(phi, A: Vector, t: sp.Symbol, coord_sys: CoordSys3D = N) -> Vector:
    return -gradient_of(phi, coord_sys) - time_derivative(A, t, coord_sys)


def B_from_potentials(A: Vector, coord_sys: CoordSys3D = N) -> Vector:
    return curl_of(A, coord_sys)


def check_coulomb_gauge(A: Vector, coord_sys: CoordSys3D = N):
    residual = sp.simplify(divergence_of(A, coord_sys))
    return residual, is_zero_scalar(residual)


def check_lorenz_gauge(A: Vector, phi, t: sp.Symbol, c=C_LIGHT, coord_sys: CoordSys3D = N):
    residual = sp.simplify(divergence_of(A, coord_sys) + sp.diff(phi, t) / c**2)
    return residual, is_zero_scalar(residual)


def check_lorenz_wave_equation_scalar(
    phi, rho, t: sp.Symbol, epsilon0=EPSILON_0, c=C_LIGHT, coord_sys: CoordSys3D = N
):
    residual = sp.simplify(
        laplacian_of(phi, coord_sys) - sp.diff(phi, t, 2) / c**2 + rho / epsilon0
    )
    return residual, is_zero_scalar(residual)


def check_lorenz_wave_equation_vector(
    A: Vector, J: Vector, t: sp.Symbol, mu0=MU_0, c=C_LIGHT, coord_sys: CoordSys3D = N
):
    d2A_dt2 = time_derivative(time_derivative(A, t, coord_sys), t, coord_sys)
    residual = simplify_vector(laplacian_of(A, coord_sys) - d2A_dt2 / c**2 + mu0 * J, coord_sys)
    return residual, is_zero_vector(residual, coord_sys)


def faraday_law_is_automatic(phi, A: Vector, t: sp.Symbol, coord_sys: CoordSys3D = N):
    E = E_from_potentials(phi, A, t, coord_sys)
    B = B_from_potentials(A, coord_sys)
    return faraday_law(E, B, t, coord_sys)
