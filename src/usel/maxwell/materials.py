from dataclasses import dataclass

import sympy as sp
from sympy.vector import CoordSys3D, Vector

from .constants import EPSILON_0, MU_0
from .core import (
    N,
    curl_of,
    divergence_of,
    is_zero_scalar,
    is_zero_vector,
    simplify_vector,
    time_derivative,
)

Number = float | int | sp.Expr


@dataclass(frozen=True)
class LinearIsotropicMaterial:
    name: str
    epsilon_r: Number = 1
    mu_r: Number = 1
    sigma: Number = 0

    @property
    def epsilon(self):
        return self.epsilon_r * EPSILON_0

    @property
    def mu(self):
        return self.mu_r * MU_0

    @property
    def chi_e(self):
        return self.epsilon_r - 1

    @property
    def chi_m(self):
        return self.mu_r - 1

    @property
    def refractive_index(self):
        return sp.sqrt(self.epsilon_r * self.mu_r)


VACUUM = LinearIsotropicMaterial("vacuum", epsilon_r=1, mu_r=1, sigma=0)
AIR = LinearIsotropicMaterial("air", epsilon_r=sp.Rational(10006, 10000), mu_r=1, sigma=0)
WATER = LinearIsotropicMaterial("water", epsilon_r=80, mu_r=1, sigma=0)
GLASS = LinearIsotropicMaterial("glass", epsilon_r=sp.Rational(225, 100), mu_r=1, sigma=0)
COPPER = LinearIsotropicMaterial("copper", epsilon_r=1, mu_r=1, sigma=sp.Float(5.96e7))
SILICON = LinearIsotropicMaterial(
    "silicon", epsilon_r=sp.Rational(117, 10), mu_r=1, sigma=sp.Rational(1, 100)
)


def dielectric(epsilon_r: Number, name: str = "dielectric") -> LinearIsotropicMaterial:
    return LinearIsotropicMaterial(name=name, epsilon_r=epsilon_r, mu_r=1, sigma=0)


def conductor(sigma: Number, name: str = "conductor") -> LinearIsotropicMaterial:
    return LinearIsotropicMaterial(name=name, epsilon_r=1, mu_r=1, sigma=sigma)


def D_from_E(E: Vector, material: LinearIsotropicMaterial) -> Vector:
    return material.epsilon * E


def E_from_D(D: Vector, material: LinearIsotropicMaterial) -> Vector:
    return D / material.epsilon


def B_from_H(H: Vector, material: LinearIsotropicMaterial) -> Vector:
    return material.mu * H


def H_from_B(B: Vector, material: LinearIsotropicMaterial) -> Vector:
    return B / material.mu


def P_from_E(E: Vector, material: LinearIsotropicMaterial) -> Vector:
    return EPSILON_0 * material.chi_e * E


def M_from_H(H: Vector, material: LinearIsotropicMaterial) -> Vector:
    return material.chi_m * H


def ohms_law(E: Vector, material: LinearIsotropicMaterial) -> Vector:
    return material.sigma * E


def verify_D_decomposition(E: Vector, material: LinearIsotropicMaterial, coord_sys: CoordSys3D = N):
    D_direct = D_from_E(E, material)
    D_decomposed = EPSILON_0 * E + P_from_E(E, material)
    residual = simplify_vector(D_direct - D_decomposed, coord_sys)
    return residual, is_zero_vector(residual, coord_sys)


def verify_H_decomposition(B: Vector, material: LinearIsotropicMaterial, coord_sys: CoordSys3D = N):
    H_direct = H_from_B(B, material)
    M = M_from_H(H_direct, material)
    H_decomposed = B / MU_0 - M
    residual = simplify_vector(H_direct - H_decomposed, coord_sys)
    return residual, is_zero_vector(residual, coord_sys)


def round_trip_D_to_E(E: Vector, material: LinearIsotropicMaterial, coord_sys: CoordSys3D = N):
    residual = simplify_vector(E_from_D(D_from_E(E, material), material) - E, coord_sys)
    return residual, is_zero_vector(residual, coord_sys)


def round_trip_H_to_B(B: Vector, material: LinearIsotropicMaterial, coord_sys: CoordSys3D = N):
    residual = simplify_vector(B_from_H(H_from_B(B, material), material) - B, coord_sys)
    return residual, is_zero_vector(residual, coord_sys)


def gauss_law_electric_in_matter(D: Vector, rho_free, coord_sys: CoordSys3D = N):
    residual = sp.simplify(divergence_of(D, coord_sys) - rho_free)
    return residual, is_zero_scalar(residual)


def ampere_maxwell_law_in_matter(
    H: Vector, J_free: Vector, D: Vector, t: sp.Symbol, coord_sys: CoordSys3D = N
):
    residual = simplify_vector(
        curl_of(H, coord_sys) - J_free - time_derivative(D, t, coord_sys), coord_sys
    )
    return residual, is_zero_vector(residual, coord_sys)
