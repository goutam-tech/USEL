import sympy as sp

from usel.maxwell import boundary, core, covariant, energy, waves
from usel.maxwell.constants import C_LIGHT

N = core.N
x, y, z = N.x, N.y, N.z
t = sp.symbols("t", real=True)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


section("1. Green light plane wave  [maxwell.waves + core]")

wavelength = 550e-9
k_val = 2 * sp.pi / wavelength
c_val = float(C_LIGHT)
w_val = c_val * float(k_val)
E0_val = 1.0

k_sym, w_sym, E0_sym = sp.symbols("k omega E0", positive=True)
mu0_sym, eps0_sym = sp.symbols("mu0 epsilon_0", positive=True)
E_wave, B_wave = waves.build_plane_wave_z(
    E0_sym, k_sym, w_sym, t, polarization=N.i, mu0=mu0_sym, epsilon0=eps0_sym
)

disp_residual, disp_ok = waves.check_dispersion_relation(
    k_sym, w_sym, mu0=mu0_sym, epsilon0=eps0_sym
)
print(f"lambda = {wavelength * 1e9:.0f} nm  ->  k = {float(k_val):.3e} rad/m")
print(f"Dispersion relation omega = c*k not yet imposed: satisfied = {disp_ok}")

c_sym = 1 / sp.sqrt(mu0_sym * eps0_sym)
E_wave, B_wave = waves.build_plane_wave_z(
    E0_sym, k_sym, c_sym * k_sym, t, polarization=N.i, mu0=mu0_sym, epsilon0=eps0_sym
)
report = core.verify_all(E_wave, B_wave, rho=0, J=0 * N.i, t=t, epsilon0=eps0_sym, mu0=mu0_sym)
print(f"With omega = c*k imposed, all four Maxwell equations satisfied: {report.all_satisfied}")

trans = waves.transverse_check(E_wave, B_wave, N.k)
print("Transversality (E, B, k mutually orthogonal):")
for label, (residual, ok) in trans.items():
    print(f"  {label}: {ok}")

section("2. Energy flow and radiation pressure  [maxwell.energy]")

poynting_residual, poynting_ok = energy.check_poynting_theorem(
    E_wave, B_wave, 0 * N.i, t, epsilon0=eps0_sym, mu0=mu0_sym
)
print(f"Poynting's theorem holds for this wave (no free current): {poynting_ok}")

S = energy.poynting_vector(E_wave, B_wave, mu0=mu0_sym)
u = energy.energy_density(E_wave, B_wave, epsilon0=eps0_sym, mu0=mu0_sym)
ratio_sq = sp.simplify(S.dot(S) / u**2)
print(f"|S|^2 / u^2 (should equal c^2): {sp.simplify(sp.sqrt(ratio_sq))}")

solar_constant = 1361.0
P_absorb = energy.radiation_pressure(solar_constant, c=c_val, reflecting=False)
P_reflect = energy.radiation_pressure(solar_constant, c=c_val, reflecting=True)
print(f"\nSolar sail example (solar constant = {solar_constant} W/m^2):")
print(f"  radiation pressure, absorbing sail  = {P_absorb:.4e} Pa")
print(f"  radiation pressure, reflecting sail = {P_reflect:.4e} Pa  (exactly 2x, as expected)")


section("3. Air-to-glass interface  [maxwell.boundary]")

n_air, n_glass = 1.0, 1.5
r_amp, t_amp = boundary.normal_incidence_fresnel(n_air, n_glass)
R_power, T_power, energy_conserved = boundary.normal_incidence_power_coefficients(n_air, n_glass)
print(f"n_air = {n_air}, n_glass = {n_glass}")
print(f"amplitude reflection r = {float(r_amp):.4f}, transmission t = {float(t_amp):.4f}")
print(f"power reflectance R = {float(R_power):.4f}, transmittance T = {float(T_power):.4f}")
print(f"R + T = 1 (energy conserved): {energy_conserved}")


section("4. Lorentz boost to a moving observer  [maxwell.covariant]")

Ey0, c_boost = sp.symbols("Ey0 c", positive=True)
E_snapshot = Ey0 * N.j
B_snapshot = (Ey0 / c_boost) * N.k

v = sp.symbols("v", positive=True)
E_boosted, B_boosted = covariant.lorentz_boost_fields(E_snapshot, B_snapshot, v, c=c_boost)

inv_before = covariant.field_invariants(E_snapshot, B_snapshot, c=c_boost)
inv_after = covariant.field_invariants(E_boosted, B_boosted, c=c_boost)
(res1, res2), invariants_ok = covariant.check_invariants_preserved_under_boost(
    E_snapshot, B_snapshot, v, c=c_boost
)

print(f"Original field snapshot: E = {E_snapshot}, B = {B_snapshot}")
print(f"Boosted (velocity v along x): E' = {E_boosted}")
print(f"                              B' = {B_boosted}")
print(
    f"Invariant E^2 - c^2*B^2  before: {sp.simplify(inv_before[0])}   after: {sp.simplify(inv_after[0])}"
)
print(f"Invariant E.B            before: {inv_before[1]}   after: {inv_after[1]}")
print(f"Both invariants preserved under boost: {invariants_ok}")
