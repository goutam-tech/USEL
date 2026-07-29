from collections.abc import Callable, Sequence

import numpy as np
import sympy as sp
from scipy import integrate

from .constants import EPSILON_0, MU_0

Vec3 = np.ndarray
FieldFunc = Callable[[float, float, float], Vec3]


def gauss_law_integral_sphere(E_r_expr, r_symbol: sp.Symbol, r_val, Q_enc, epsilon0=EPSILON_0):
    flux = 4 * sp.pi * r_val**2 * E_r_expr.subs(r_symbol, r_val)
    residual = sp.simplify(flux - Q_enc / epsilon0)
    return residual, residual == 0


def ampere_law_integral_circle(B_phi_expr, r_symbol: sp.Symbol, r_val, I_enc, mu0=MU_0):
    circulation = 2 * sp.pi * r_val * B_phi_expr.subs(r_symbol, r_val)
    residual = sp.simplify(circulation - mu0 * I_enc)
    return residual, residual == 0


def faraday_law_integral(emf_expr, flux_B_expr, t: sp.Symbol):
    residual = sp.simplify(emf_expr + sp.diff(flux_B_expr, t))
    return residual, residual == 0


def ampere_maxwell_law_integral(
    circulation_B_expr, I_enc_expr, flux_E_expr, t: sp.Symbol, mu0=MU_0, epsilon0=EPSILON_0
):
    rhs = mu0 * I_enc_expr + mu0 * epsilon0 * sp.diff(flux_E_expr, t)
    residual = sp.simplify(circulation_B_expr - rhs)
    return residual, residual == 0


def numeric_surface_flux(
    field_func: FieldFunc,
    param_func: Callable[[float, float], Vec3],
    u_range: tuple[float, float],
    v_range: tuple[float, float],
    h: float = 1e-6,
    epsabs: float = 1e-8,
) -> float:

    def integrand(v, u):
        r = np.asarray(param_func(u, v), dtype=float)
        r_u = (np.asarray(param_func(u + h, v)) - np.asarray(param_func(u - h, v))) / (2 * h)
        r_v = (np.asarray(param_func(u, v + h)) - np.asarray(param_func(u, v - h))) / (2 * h)
        normal = np.cross(r_u, r_v)
        F = np.asarray(field_func(*r), dtype=float)
        return float(np.dot(F, normal))

    result, _ = integrate.dblquad(
        integrand, u_range[0], u_range[1], v_range[0], v_range[1], epsabs=epsabs
    )
    return result


def numeric_line_circulation(
    field_func: FieldFunc,
    curve_func: Callable[[float], Vec3],
    t_range: tuple[float, float],
    h: float = 1e-6,
    epsabs: float = 1e-8,
) -> float:

    def integrand(tp):
        r = np.asarray(curve_func(tp), dtype=float)
        r_prime = (np.asarray(curve_func(tp + h)) - np.asarray(curve_func(tp - h))) / (2 * h)
        F = np.asarray(field_func(*r), dtype=float)
        return float(np.dot(F, r_prime))

    result, _ = integrate.quad(integrand, t_range[0], t_range[1], epsabs=epsabs)
    return result


def numeric_divergence(field_func: FieldFunc, point: Sequence[float], h: float = 1e-5) -> float:
    point = np.asarray(point, dtype=float)
    total = 0.0
    for i in range(3):
        plus = point.copy()
        plus[i] += h
        minus = point.copy()
        minus[i] -= h
        total += (np.asarray(field_func(*plus))[i] - np.asarray(field_func(*minus))[i]) / (2 * h)
    return total


def numeric_curl(field_func: FieldFunc, point: Sequence[float], h: float = 1e-5) -> Vec3:
    point = np.asarray(point, dtype=float)

    def partial(component_idx, axis_idx):
        plus = point.copy()
        plus[axis_idx] += h
        minus = point.copy()
        minus[axis_idx] -= h
        return (
            np.asarray(field_func(*plus))[component_idx]
            - np.asarray(field_func(*minus))[component_idx]
        ) / (2 * h)

    curl_x = partial(2, 1) - partial(1, 2)
    curl_y = partial(0, 2) - partial(2, 0)
    curl_z = partial(1, 0) - partial(0, 1)
    return np.array([curl_x, curl_y, curl_z])


def divergence_theorem_check(
    field_func: FieldFunc,
    box_bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    rel_tol: float = 1e-3,
):

    (x0, x1), (y0, y1), (z0, z1) = box_bounds

    def face_flux(fixed_axis, fixed_val, sign, u_bounds, v_bounds, u_axis, v_axis):
        def integrand(v, u):
            point = [0.0, 0.0, 0.0]
            point[fixed_axis] = fixed_val
            point[u_axis] = u
            point[v_axis] = v
            F = np.asarray(field_func(*point), dtype=float)
            return sign * F[fixed_axis]

        result, _ = integrate.dblquad(integrand, u_bounds[0], u_bounds[1], v_bounds[0], v_bounds[1])
        return result

    flux = 0.0

    flux += face_flux(0, x1, +1, (y0, y1), (z0, z1), 1, 2)
    flux += face_flux(0, x0, -1, (y0, y1), (z0, z1), 1, 2)

    flux += face_flux(1, y1, +1, (x0, x1), (z0, z1), 0, 2)
    flux += face_flux(1, y0, -1, (x0, x1), (z0, z1), 0, 2)

    flux += face_flux(2, z1, +1, (x0, x1), (y0, y1), 0, 1)
    flux += face_flux(2, z0, -1, (x0, x1), (y0, y1), 0, 1)

    n = 25
    xs = np.linspace(x0, x1, n)
    ys = np.linspace(y0, y1, n)
    zs = np.linspace(z0, z1, n)
    div_grid = np.zeros((n, n, n))
    for i, xv in enumerate(xs):
        for j, yv in enumerate(ys):
            for k, zv in enumerate(zs):
                div_grid[i, j, k] = numeric_divergence(field_func, (xv, yv, zv))

    volume_integral = np.trapezoid(
        np.trapezoid(np.trapezoid(div_grid, zs, axis=2), ys, axis=1), xs, axis=0
    )

    denom = max(abs(flux), abs(volume_integral), 1e-12)
    rel_diff = abs(flux - volume_integral) / denom
    return flux, volume_integral, rel_diff, rel_diff < rel_tol


def stokes_theorem_check(
    field_func: FieldFunc,
    rect_bounds: tuple[tuple[float, float], tuple[float, float]],
    plane: str = "xy",
    offset: float = 0.0,
    rel_tol: float = 1e-3,
):

    (u0, u1), (v0, v1) = rect_bounds

    axes = {"xy": (0, 1, 2), "yz": (1, 2, 0), "zx": (2, 0, 1)}
    u_axis, v_axis, normal_axis = axes[plane]

    def to_point(u, v):
        point = [0.0, 0.0, 0.0]
        point[u_axis] = u
        point[v_axis] = v
        point[normal_axis] = offset
        return point

    def curve(t):
        perim = 2 * (u1 - u0) + 2 * (v1 - v0)
        s = t * perim
        if s < (u1 - u0):
            return to_point(u0 + s, v0)
        s -= u1 - u0
        if s < (v1 - v0):
            return to_point(u1, v0 + s)
        s -= v1 - v0
        if s < (u1 - u0):
            return to_point(u1 - s, v1)
        s -= u1 - u0
        return to_point(u0, v1 - s)

    circulation = numeric_line_circulation(field_func, curve, (0.0, 1.0))

    def curl_normal_component(u, v):
        point = to_point(u, v)
        curl_vec = numeric_curl(field_func, point)
        return curl_vec[normal_axis]

    curl_flux, _ = integrate.dblquad(lambda v, u: curl_normal_component(u, v), u0, u1, v0, v1)

    denom = max(abs(circulation), abs(curl_flux), 1e-12)
    rel_diff = abs(circulation - curl_flux) / denom
    return circulation, curl_flux, rel_diff, rel_diff < rel_tol
