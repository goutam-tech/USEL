import sympy as sp
from sympy.vector import CoordSys3D, Vector

from .constants import EPSILON_0, MU_0
from .core import N, is_zero_scalar, is_zero_vector, simplify_vector, time_derivative
from .potentials import laplacian_of


def wave_equation_residual_scalar(
    field, t: sp.Symbol, mu0=MU_0, epsilon0=EPSILON_0, coord_sys: CoordSys3D = N
):
    return laplacian_of(field, coord_sys) - mu0 * epsilon0 * sp.diff(field, t, 2)


def wave_equation_residual_vector(
    field: Vector, t: sp.Symbol, mu0=MU_0, epsilon0=EPSILON_0, coord_sys: CoordSys3D = N
) -> Vector:
    d2_dt2 = time_derivative(time_derivative(field, t, coord_sys), t, coord_sys)
    return laplacian_of(field, coord_sys) - mu0 * epsilon0 * d2_dt2


def check_wave_equation_scalar(
    field, t: sp.Symbol, mu0=MU_0, epsilon0=EPSILON_0, coord_sys: CoordSys3D = N
):
    residual = sp.simplify(wave_equation_residual_scalar(field, t, mu0, epsilon0, coord_sys))
    return residual, is_zero_scalar(residual)


def check_wave_equation_vector(
    field: Vector, t: sp.Symbol, mu0=MU_0, epsilon0=EPSILON_0, coord_sys: CoordSys3D = N
):
    residual = simplify_vector(
        wave_equation_residual_vector(field, t, mu0, epsilon0, coord_sys), coord_sys
    )
    return residual, is_zero_vector(residual, coord_sys)


def speed_of_light_squared(mu0=MU_0, epsilon0=EPSILON_0):
    return 1 / (mu0 * epsilon0)


def dispersion_relation_residual(k, omega, mu0=MU_0, epsilon0=EPSILON_0):
    return sp.simplify(omega**2 - speed_of_light_squared(mu0, epsilon0) * k**2)


def check_dispersion_relation(k, omega, mu0=MU_0, epsilon0=EPSILON_0):
    residual = dispersion_relation_residual(k, omega, mu0, epsilon0)
    return residual, residual == 0


def build_plane_wave_z(
    E0,
    k,
    omega,
    t: sp.Symbol,
    polarization: Vector = None,
    phase=0,
    mu0=MU_0,
    epsilon0=EPSILON_0,
    coord_sys: CoordSys3D = N,
):
    if polarization is None:
        polarization = coord_sys.i

    z = coord_sys.z
    phase_arg = k * z - omega * t + phase
    E = E0 * sp.cos(phase_arg) * polarization

    k_hat = coord_sys.k
    B = (k / omega) * k_hat.cross(E)

    return E, B


def transverse_check(E: Vector, B: Vector, k_vec: Vector, coord_sys: CoordSys3D = N):
    results = {}
    for label, a, b in (("E_dot_B", E, B), ("E_dot_k", E, k_vec), ("B_dot_k", B, k_vec)):
        residual = sp.simplify(a.dot(b))
        results[label] = (residual, is_zero_scalar(residual))
    return results
