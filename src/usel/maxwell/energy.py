import sympy as sp
from sympy.vector import CoordSys3D, Vector

from .constants import C_LIGHT, EPSILON_0, MU_0
from .core import N, divergence_of, is_zero_scalar


def poynting_vector(E: Vector, B: Vector, mu0=MU_0) -> Vector:
    return (E.cross(B)) / mu0


def energy_density(E: Vector, B: Vector, epsilon0=EPSILON_0, mu0=MU_0):
    return sp.Rational(1, 2) * (epsilon0 * E.dot(E) + B.dot(B) / mu0)


def check_poynting_theorem(
    E: Vector,
    B: Vector,
    J: Vector,
    t: sp.Symbol,
    epsilon0=EPSILON_0,
    mu0=MU_0,
    coord_sys: CoordSys3D = N,
):
    u = energy_density(E, B, epsilon0, mu0)
    S = poynting_vector(E, B, mu0)
    residual = sp.simplify(sp.diff(u, t) + divergence_of(S, coord_sys) + J.dot(E))
    return residual, is_zero_scalar(residual)


def maxwell_stress_tensor(
    E: Vector, B: Vector, epsilon0=EPSILON_0, mu0=MU_0, coord_sys: CoordSys3D = N
) -> sp.Matrix:
    E_comp = [E.dot(coord_sys.i), E.dot(coord_sys.j), E.dot(coord_sys.k)]
    B_comp = [B.dot(coord_sys.i), B.dot(coord_sys.j), B.dot(coord_sys.k)]
    E2 = E.dot(E)
    B2 = B.dot(B)

    T = sp.zeros(3, 3)
    for i in range(3):
        for j in range(3):
            delta_ij = 1 if i == j else 0
            T[i, j] = epsilon0 * (E_comp[i] * E_comp[j] - sp.Rational(1, 2) * delta_ij * E2) + (
                1 / mu0
            ) * (B_comp[i] * B_comp[j] - sp.Rational(1, 2) * delta_ij * B2)
    return T


def stress_tensor_trace_identity_check(
    E: Vector, B: Vector, epsilon0=EPSILON_0, mu0=MU_0, coord_sys: CoordSys3D = N
):
    T = maxwell_stress_tensor(E, B, epsilon0, mu0, coord_sys)
    u = energy_density(E, B, epsilon0, mu0)
    residual = sp.simplify(sp.trace(T) + u)
    return residual, is_zero_scalar(residual)


def lorentz_force(q, E: Vector, B: Vector, v: Vector) -> Vector:
    return q * (E + v.cross(B))


def radiation_pressure(intensity, c=C_LIGHT, reflecting: bool = False):
    factor = 2 if reflecting else 1
    return factor * intensity / c
