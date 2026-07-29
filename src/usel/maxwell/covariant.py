import sympy as sp
from sympy.vector import CoordSys3D, Vector

from .constants import C_LIGHT, MU_0
from .core import N, faraday_law, gauss_law_magnetic, is_zero_scalar


def field_strength_tensor(E: Vector, B: Vector, c=C_LIGHT, coord_sys: CoordSys3D = N) -> sp.Matrix:
    Ex, Ey, Ez = E.dot(coord_sys.i), E.dot(coord_sys.j), E.dot(coord_sys.k)
    Bx, By, Bz = B.dot(coord_sys.i), B.dot(coord_sys.j), B.dot(coord_sys.k)

    return sp.Matrix(
        [
            [0, Ex / c, Ey / c, Ez / c],
            [-Ex / c, 0, Bz, -By],
            [-Ey / c, -Bz, 0, Bx],
            [-Ez / c, By, -Bx, 0],
        ]
    )


def four_current(rho, J: Vector, c=C_LIGHT, coord_sys: CoordSys3D = N) -> sp.Matrix:
    Jx, Jy, Jz = J.dot(coord_sys.i), J.dot(coord_sys.j), J.dot(coord_sys.k)
    return sp.Matrix([c * rho, Jx, Jy, Jz])


def covariant_maxwell_check(
    F: sp.Matrix,
    J4: sp.Matrix,
    t: sp.Symbol,
    x: sp.Symbol,
    y: sp.Symbol,
    z: sp.Symbol,
    mu0=MU_0,
    c=C_LIGHT,
):
    coords = [t, x, y, z]

    def d_mu(expr, mu):
        if mu == 0:
            return sp.diff(expr, t) / c
        return sp.diff(expr, coords[mu])

    residuals = []
    for nu in range(4):
        lhs = sum(d_mu(F[mu, nu], mu) for mu in range(4))
        residual = sp.simplify(lhs + mu0 * J4[nu])
        residuals.append(residual)

    all_satisfied = all(is_zero_scalar(r) for r in residuals)
    return residuals, all_satisfied


def homogeneous_pair_check(E: Vector, B: Vector, t: sp.Symbol, coord_sys: CoordSys3D = N):
    gm_residual, gm_ok = gauss_law_magnetic(B, coord_sys)
    fa_residual, fa_ok = faraday_law(E, B, t, coord_sys)
    return {
        "gauss_magnetic": (gm_residual, gm_ok),
        "faraday": (fa_residual, fa_ok),
        "all_satisfied": gm_ok and fa_ok,
    }


def field_invariants(E: Vector, B: Vector, c=C_LIGHT) -> tuple[sp.Expr, sp.Expr]:
    return sp.simplify(E.dot(E) - c**2 * B.dot(B)), sp.simplify(E.dot(B))


def lorentz_boost_fields(
    E: Vector, B: Vector, v, c=C_LIGHT, coord_sys: CoordSys3D = N
) -> tuple[Vector, Vector]:
    gamma = 1 / sp.sqrt(1 - v**2 / c**2)

    Ex, Ey, Ez = E.dot(coord_sys.i), E.dot(coord_sys.j), E.dot(coord_sys.k)
    Bx, By, Bz = B.dot(coord_sys.i), B.dot(coord_sys.j), B.dot(coord_sys.k)

    Ex_p = Ex
    Ey_p = gamma * (Ey - v * Bz)
    Ez_p = gamma * (Ez + v * By)
    Bx_p = Bx
    By_p = gamma * (By + v * Ez / c**2)
    Bz_p = gamma * (Bz - v * Ey / c**2)

    E_prime = Ex_p * coord_sys.i + Ey_p * coord_sys.j + Ez_p * coord_sys.k
    B_prime = Bx_p * coord_sys.i + By_p * coord_sys.j + Bz_p * coord_sys.k
    return E_prime, B_prime


def check_invariants_preserved_under_boost(
    E: Vector, B: Vector, v, c=C_LIGHT, coord_sys: CoordSys3D = N
):
    inv1_before, inv2_before = field_invariants(E, B, c)
    E_p, B_p = lorentz_boost_fields(E, B, v, c, coord_sys)
    inv1_after, inv2_after = field_invariants(E_p, B_p, c)

    residual1 = sp.simplify(inv1_after - inv1_before)
    residual2 = sp.simplify(inv2_after - inv2_before)
    satisfied = is_zero_scalar(residual1) and is_zero_scalar(residual2)
    return (residual1, residual2), satisfied
