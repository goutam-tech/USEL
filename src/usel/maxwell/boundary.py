import sympy as sp
from .core import N, is_zero_scalar, is_zero_vector, simplify_vector
from sympy.vector import CoordSys3D, Vector


def decompose_field(F: Vector, normal: Vector, coord_sys: CoordSys3D = N):
    F_perp = F.dot(normal) * normal
    F_parallel = F - F_perp
    return F_perp, F_parallel


def check_tangential_E_continuity(
    E1: Vector, E2: Vector, normal: Vector, coord_sys: CoordSys3D = N
):
    diff = E1 - E2
    _, tangential_diff = decompose_field(diff, normal, coord_sys)
    residual = simplify_vector(tangential_diff, coord_sys)
    return residual, is_zero_vector(residual, coord_sys)


def check_normal_B_continuity(B1: Vector, B2: Vector, normal: Vector, coord_sys: CoordSys3D = N):
    residual = sp.simplify((B1 - B2).dot(normal))
    return residual, is_zero_scalar(residual)


def check_normal_D_discontinuity(
    D1: Vector, D2: Vector, normal: Vector, sigma_free, coord_sys: CoordSys3D = N
):
    residual = sp.simplify((D1 - D2).dot(normal) - sigma_free)
    return residual, is_zero_scalar(residual)


def check_tangential_H_discontinuity(
    H1: Vector, H2: Vector, normal: Vector, K_free: Vector, coord_sys: CoordSys3D = N
):
    residual = simplify_vector(normal.cross(H1 - H2) - K_free, coord_sys)
    return residual, is_zero_vector(residual, coord_sys)


def normal_incidence_fresnel(n1, n2):
    r = (n1 - n2) / (n1 + n2)
    t = 2 * n1 / (n1 + n2)
    return r, t


def normal_incidence_power_coefficients(n1, n2):
    r, t = normal_incidence_fresnel(n1, n2)
    R = sp.simplify(r**2)
    T = sp.simplify((n2 / n1) * t**2)
    conserved = is_zero_scalar(R + T - 1)
    return R, T, conserved
