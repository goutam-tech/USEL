import sympy as sp

from usel.maxwell import core, materials, potentials
from usel.maxwell.constants import EPSILON_0

N = core.N
x, y, z = N.x, N.y, N.z
t = sp.symbols("t", real=True)


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


section("1. Point charge field vs. Gauss's law  [maxwell.core]")

q = sp.symbols("q", positive=True)
r_vec = x * N.i + y * N.j + z * N.k
r_mag = sp.sqrt(x**2 + y**2 + z**2)
E_point = (q / (4 * sp.pi * EPSILON_0)) * r_vec / r_mag**3

residual, ok = core.gauss_law_electric(E_point, rho=0, epsilon0=EPSILON_0)
print("E(x,y,z) = q/(4*pi*eps0) * r_vec / r^3")
print(f"div(E) - rho/eps0  =  {residual}")
print(f"Gauss's law satisfied away from the origin: {ok}")


section("2. Uniform B field from the symmetric-gauge potential  [maxwell.potentials]")

B0 = sp.symbols("B0", positive=True)
A = (-B0 * y / 2) * N.i + (B0 * x / 2) * N.j  # symmetric gauge
B_uniform = potentials.B_from_potentials(A)

gauge_residual, gauge_ok = potentials.check_coulomb_gauge(A)
gauss_m_residual, gauss_m_ok = core.gauss_law_magnetic(B_uniform)

print("A = -B0*y/2 * x_hat + B0*x/2 * y_hat")
print(f"B = curl(A) = {B_uniform}")
print(f"Coulomb gauge div(A) = {gauge_residual}  (satisfied: {gauge_ok})")
print(f"Gauss's law for magnetism div(B) = {gauss_m_residual}  (satisfied: {gauss_m_ok})")


section("3. Faraday's law as a structural identity  [maxwell.potentials]")

phi_demo = x**2 * sp.sin(y)
A_demo = sp.cos(t) * N.i + x * z * N.j
residual, ok = potentials.faraday_law_is_automatic(phi_demo, A_demo, t)
print("Using arbitrary phi and A (no physical meaning):")
print(f"  phi = {phi_demo}")
print(f"  A   = {A_demo}")
print(f"curl(E) + dB/dt = {residual}   (always zero: {ok})")
print("This is why Faraday's law is never the equation that constrains")
print("phi, A -- Gauss's law and Ampere-Maxwell's law are.")


section("4. Point charge field inside water  [maxwell.materials]")

print(f"water: epsilon_r = {materials.WATER.epsilon_r}, mu_r = {materials.WATER.mu_r}")

D_in_water = materials.D_from_E(E_point, materials.WATER)
E_recovered, ok_roundtrip = materials.round_trip_D_to_E(E_point, materials.WATER)

print(f"D = epsilon_r * epsilon0 * E  (D is {materials.WATER.epsilon_r}x smaller-looking")
print("in the sense that E is what's screened by the water's polarization)")
print(f"Round trip E -> D -> E recovers the original field: {ok_roundtrip}")


gauss_matter_residual, gauss_matter_ok = materials.gauss_law_electric_in_matter(
    D_in_water, rho_free=0
)
print(
    "Gauss's law in matter, div(D) - rho_free = "
    f"{gauss_matter_residual}  (satisfied: {gauss_matter_ok})"
)
