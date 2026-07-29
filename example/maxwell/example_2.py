import numpy as np
import sympy as sp

from usel.maxwell import integral, special_cases
from usel.maxwell.constants import EPSILON_0, MU_0

r = sp.symbols("r", positive=True)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


section("1. Field near a current-carrying wire  [maxwell.special_cases + integral]")

I_wire = 5.0
r_probe = 0.02

B_formula = special_cases.biot_savart_wire_field(I_wire, r_probe, mu0=MU_0)
print(f"Wire carrying I = {I_wire} A, probe point r = {r_probe} m")
print(f"B (Biot-Savart formula) = {float(B_formula):.6e} T")

I_sym, mu0_sym = sp.symbols("I mu0", positive=True)
residual, ok = special_cases.biot_savart_matches_ampere(I_sym, r, r, mu0=mu0_sym)
print("\nSymbolically: does Biot-Savart's field satisfy Ampere's law exactly?")
print(f"circulation - mu0*I_enc = {residual}   (exact match: {ok})")


section("2. Field around a point charge  [maxwell.special_cases + integral]")

q_charge = 1e-9
r_probe2 = 0.05

F_formula = special_cases.coulombs_law(q_charge, q_charge, 2 * r_probe2, epsilon0=EPSILON_0)
print(f"Force between two {q_charge:.0e} C charges, {2 * r_probe2} m apart:")
print(f"F (Coulomb's law) = {float(F_formula):.6e} N")

q_sym, eps0_sym = sp.symbols("q epsilon_0", positive=True)
residual, ok = special_cases.coulomb_field_matches_gauss(q_sym, r, r, epsilon0=eps0_sym)
print("\nSymbolically: does Coulomb's field satisfy Gauss's law exactly?")
print(f"flux - Q_enc/epsilon0 = {residual}   (exact match: {ok})")


section("3. Divergence theorem, numerically  [maxwell.integral]")


def radial_field(x, y, z):
    return np.array([x, y, z])


R = 1.0
flux = integral.numeric_surface_flux(
    radial_field,
    lambda u, v: np.array([R * np.sin(u) * np.cos(v), R * np.sin(u) * np.sin(v), R * np.cos(u)]),
    (1e-6, np.pi - 1e-6),
    (0, 2 * np.pi),
)
print(f"F = r_vec (radial field), sphere of radius {R}")
print(f"Numeric flux through the closed sphere:  {flux:.6f}")
print(f"Expected (div(F)=3, volume=4/3*pi*R^3):  {3 * 4 / 3 * np.pi * R**3:.6f}")

flux2, vol_integral, rel_diff, agrees = integral.divergence_theorem_check(
    radial_field, ((0, 1), (0, 1), (0, 1))
)
print("\nSame check via divergence_theorem_check on a unit box:")
print(f"  closed-surface flux      = {flux2:.6f}")
print(f"  volume integral of div(F) = {vol_integral:.6f}")
print(f"  agree within tolerance: {agrees}")

section("4. Stokes' theorem, numerically  [maxwell.integral]")


def rotation_field(x, y, z):
    return np.array([-y, x, 0.0])


circulation, curl_flux, rel_diff, agrees = integral.stokes_theorem_check(
    rotation_field, ((0, 1), (0, 1)), plane="xy"
)
print("F = (-y, x, 0), unit square in the xy-plane")
print(f"  circulation around the loop = {circulation:.6f}")
print(f"  flux of curl(F) through it  = {curl_flux:.6f}")
print(f"  agree within tolerance: {agrees}")


section("5. Continuity equation & skin depth  [maxwell.special_cases]")

t = sp.symbols("t", real=True)
xsym = sp.symbols("x", real=True)
rho0, k, w = sp.symbols("rho0 k omega", positive=True)
rho = rho0 * sp.cos(k * xsym - w * t)
Jx = (rho0 * w / k) * sp.cos(k * xsym - w * t)
residual, ok = special_cases.check_continuity_equation_1d(Jx, rho, xsym, t)
print("Charge-density wave rho = rho0*cos(kx-wt), matched current Jx:")
print(f"div(J) + drho/dt = {residual}   (continuity holds: {ok})")

delta = special_cases.skin_depth(MU_0, sp.Float(5.96e7), 2 * sp.pi * 60)
print(f"\nSkin depth of copper at 60 Hz: {float(delta) * 1000:.2f} mm  (textbook value: ~8.5 mm)")
