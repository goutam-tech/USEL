# maxwell — API Reference

A Python library (SymPy for symbolic math, NumPy/SciPy for numerics) for
building electromagnetic fields and checking them against every form of
Maxwell's equations: differential, integral, potential, energetic,
relativistic, and in matter.

## How the library works, in general

Every "check" function follows the same pattern: you hand it a candidate
field or pair of fields, and it hands back a pair `(residual, satisfied)`.

- **`satisfied`** is a boolean — this is the one you check in your code.
- **`residual`** is the leftover symbolic expression after simplification.
  It equals exactly `0` when the equation truly holds. When it doesn't
  hold, `residual` tells you precisely what's left over, which is usually
  the fastest way to see *why* a field fails a given law.

To use any function, you first need two things already set up:

1. **A coordinate frame** — `maxwell.core.N`. This is shared across every
   single module in the library, so all fields you build (in any module)
   are automatically compatible with each other. Vector components are
   built along `N.i`, `N.j`, `N.k`, and position is referenced through
   `N.x`, `N.y`, `N.z`.
2. **A time symbol** — a plain SymPy symbol representing time (call it
   `t`), which you create once and pass into every function that needs a
   time derivative.

Everything else is just: build your field(s) as `sympy.vector.Vector`
objects using that frame, then pass them into whichever check function
matches the physics you're testing.

Units are SI throughout. Physical constants (epsilon0, mu0, c) are
importable from `maxwell.constants`, and every check function accepts
them as optional keyword overrides (`epsilon0=`, `mu0=`, `c=`) in case
you want to work with a different medium or keep them symbolic.

> **One important usage note:** for an *exact* zero-residual check, prefer
> passing symbolic placeholders for epsilon0/mu0 rather than the real
> numeric `EPSILON_0`/`MU_0` constants. The numeric constants are
> finite-precision floats (real CODATA measurements, not exact
> fractions), so an otherwise perfectly exact identity can leave a tiny
> (~1e-19) floating-point residual instead of a clean zero. Use the
> numeric constants when you want an actual physical number out (like a
> real wavelength or force in Newtons); use symbolic placeholders when
> you're verifying that an equation holds exactly.

---

## Module map

| Module | What it covers |
|---|---|
| `constants` | epsilon0, mu0, c (CODATA values) |
| `core` | The four Maxwell equations, differential form |
| `materials` | Constitutive relations (D, H, P, M), Maxwell's equations in matter |
| `potentials` | Scalar/vector potentials phi, A, gauges, wave equations |
| `integral` | Flux/circulation, symbolic canonical cases, numeric divergence/Stokes theorem |
| `waves` | Wave equation, dispersion relation, plane wave construction |
| `energy` | Poynting vector, energy density, stress tensor, Lorentz force |
| `boundary` | Interface boundary conditions, Fresnel coefficients |
| `covariant` | 4-tensor formulation, Lorentz boosts, field invariants |
| `special_cases` | Coulomb's law, Biot-Savart, continuity equation, skin depth |

Import path depends on your project layout — either `from maxwell import
core, ...` (standalone) or `from usel.maxwell.symbolic import core, ...`
(inside USEL).

---

## `maxwell.constants`

| Call | Returns | Meaning |
|---|---|---|
| `EPSILON_0` | 8.8541878128e-12 | vacuum permittivity (F/m) |
| `MU_0` | 1.25663706212e-6 | vacuum permeability (N/A^2) |
| `C_LIGHT` | 299792458 | speed of light (m/s) |
| `speed_of_light_from_em_constants(epsilon0, mu0)` | expression | 1/sqrt(epsilon0*mu0), for checking the two are consistent |

---

## `maxwell.core`

The four Maxwell equations, differential form. Every other module builds
on this one.

**Vector calculus helpers** — low-level building blocks, not usually
called directly unless you're constructing a new check yourself:
`divergence_of`, `curl_of`, `time_derivative`, `simplify_vector`,
`is_zero_vector`, `is_zero_scalar`.

**The four equations**, each taking the relevant fields plus optional
`epsilon0`/`mu0` overrides, each returning `(residual, satisfied)`:

| Call | Checks |
|---|---|
| `gauss_law_electric(E, rho, epsilon0)` | div(E) = rho/epsilon0 |
| `gauss_law_magnetic(B)` | div(B) = 0 |
| `faraday_law(E, B, t)` | curl(E) = -dB/dt |
| `ampere_maxwell_law(B, E, J, t, mu0, epsilon0)` | curl(B) = mu0*J + mu0*epsilon0*dE/dt |

**Check all four at once:** `verify_all(E, B, rho, J, t, epsilon0, mu0)`
returns a `MaxwellReport` object. Its `.all_satisfied` attribute is `True`
only if every one of the four laws holds; its `.summary()` method prints
a pass/fail line with the residual for each law, which is the quickest
way to see which specific equation a field is failing.

**How to use it:** build E and B as vectors, decide what charge density
(rho) and current density (J) you're claiming produced them, and call
`verify_all`. If a field is only meant to satisfy one particular law
(e.g. you're only testing Gauss's law), call that individual function
instead — it's faster and the residual is more directly interpretable.

---

## `maxwell.materials`

Constitutive relations for linear, isotropic media (D-E, H-B, Ohm's law),
plus Maxwell's equations written in terms of the macroscopic fields D
and H.

**Material objects:** `LinearIsotropicMaterial(name, epsilon_r, mu_r,
sigma)` holds a medium's relative permittivity, relative permeability,
and conductivity, and exposes `.epsilon`, `.mu`, `.chi_e`, `.chi_m`,
`.refractive_index` as derived properties. `dielectric(epsilon_r)` and
`conductor(sigma)` are shortcut constructors for the common cases.
Predefined materials ready to use: `VACUUM`, `AIR`, `WATER`, `GLASS`,
`COPPER`, `SILICON`.

**Conversions** (each takes a field and a material, returns the converted
field): `D_from_E`, `E_from_D`, `B_from_H`, `H_from_B`, `P_from_E`,
`M_from_H`, `ohms_law`.

**Consistency checks**, each returning `(residual, satisfied)`:

| Call | Checks |
|---|---|
| `verify_D_decomposition(E, material)` | D = eps0*E+P is the same as D = eps*E |
| `verify_H_decomposition(B, material)` | H = B/mu0-M is the same as H = B/mu |
| `round_trip_D_to_E(E, material)` | converting E to D to E returns the original field |
| `round_trip_H_to_B(B, material)` | converting B to H to B returns the original field |
| `gauss_law_electric_in_matter(D, rho_free)` | div(D) = rho_free |
| `ampere_maxwell_law_in_matter(H, J_free, D, t)` | curl(H) = J_free + dD/dt |

**How to use it:** pick or build a material, run a field through the
appropriate conversion function to get D or H, and — if you want to
double-check the physics rather than just trust the formula — run it
through the matching "in matter" check with the material's free charge
or current.

---

## `maxwell.potentials`

The scalar potential phi and vector potential A, from which E and B are
derived, plus the gauge conditions and wave equations they must satisfy.

| Call | Purpose |
|---|---|
| `gradient_of(scalar_field)` | gradient of a scalar |
| `laplacian_of(field)` | Laplacian (scalar or vector, applied componentwise for vectors) |
| `E_from_potentials(phi, A, t)` | builds E = -grad(phi) - dA/dt |
| `B_from_potentials(A)` | builds B = curl(A) |
| `check_coulomb_gauge(A)` | checks div(A) = 0 |
| `check_lorenz_gauge(A, phi, t, c)` | checks div(A) + (1/c^2)*dphi/dt = 0 |
| `check_lorenz_wave_equation_scalar(phi, rho, t, epsilon0, c)` | checks phi's wave equation |
| `check_lorenz_wave_equation_vector(A, J, t, mu0, c)` | checks A's wave equation |
| `faraday_law_is_automatic(phi, A, t)` | confirms Faraday's law holds for any phi, A |

**How to use it:** rather than picking E and B directly, pick a scalar
potential phi and vector potential A instead, then derive the fields with
`E_from_potentials`/`B_from_potentials`. This is the natural approach
whenever you're modeling something (like a radiating antenna or a
charging capacitor) where the potentials are the more physically natural
starting point. `faraday_law_is_automatic` is worth knowing about even if
you never call it directly: it demonstrates that Faraday's law is never
the equation constraining your choice of phi, A — Gauss's law and the
Ampere-Maxwell law (via the gauge/wave-equation checks above) are the
real constraints.

---

## `maxwell.integral`

Flux through closed surfaces and circulation around closed loops — the
integral form of Maxwell's equations. Two ways to work with it:

**Symbolic, for spherically/cylindrically symmetric setups** — you supply
the radial or azimuthal field component as a function of `r`, and the
function does the elementary geometry (surface area of a sphere,
circumference of a circle) for you:

| Call | Checks |
|---|---|
| `gauss_law_integral_sphere(E_r_expr, r_symbol, r_val, Q_enc, epsilon0)` | flux through a sphere = Q_enc/epsilon0 |
| `ampere_law_integral_circle(B_phi_expr, r_symbol, r_val, I_enc, mu0)` | circulation around a circle = mu0*I_enc |
| `faraday_law_integral(emf_expr, flux_B_expr, t)` | emf = -d(flux)/dt |
| `ampere_maxwell_law_integral(circulation_B_expr, I_enc_expr, flux_E_expr, t, mu0, epsilon0)` | circulation = mu0*I_enc + mu0*epsilon0*d(flux_E)/dt |

**Numeric, for an arbitrary field and an arbitrary surface or curve** —
you supply a field as a plain Python function of (x, y, z), and a
parametrization of the surface or curve; the integration itself is done
with scipy quadrature:

| Call | Purpose |
|---|---|
| `numeric_surface_flux(field_func, param_func, u_range, v_range)` | flux through any parametrized surface |
| `numeric_line_circulation(field_func, curve_func, t_range)` | circulation around any parametrized curve |
| `numeric_divergence(field_func, point)` | finite-difference divergence at a point |
| `numeric_curl(field_func, point)` | finite-difference curl at a point |
| `divergence_theorem_check(field_func, box_bounds)` | confirms closed-surface flux over a box equals the volume integral of divergence |
| `stokes_theorem_check(field_func, rect_bounds, plane)` | confirms loop circulation equals the flux of curl through the loop |

**How to use it:** use the symbolic functions when your field has the
right symmetry (a point charge, a long straight wire) — they're exact and
instant. Use the numeric functions when your field doesn't have a
convenient symmetry, or when you specifically want to demonstrate that
the divergence/Stokes theorem connects the integral and differential
forms for some arbitrary field you've made up.

---

## `maxwell.waves`

The homogeneous (source-free) wave equation, the dispersion relation, and
constructing plane-wave solutions.

| Call | Purpose |
|---|---|
| `check_wave_equation_scalar(field, t, mu0, epsilon0)` | checks the Laplacian of the field equals mu0*epsilon0 times its second time derivative |
| `check_wave_equation_vector(field, t, mu0, epsilon0)` | same, for a vector field |
| `check_dispersion_relation(k, omega, mu0, epsilon0)` | checks omega = c*k |
| `build_plane_wave_z(E0, k, omega, t, polarization, phase)` | builds a transverse EM plane wave traveling along +z |
| `transverse_check(E, B, k_vec)` | confirms E, B, and the propagation direction are mutually perpendicular |

**How to use it:** `build_plane_wave_z` gives you E and B for a wave of
whatever amplitude, wavenumber, and polarization you specify — but it
doesn't force the physically-correct relationship between the wavenumber
and frequency for you (you supply `omega` yourself). Pass `omega = c*k`
(with `c` from `1/sqrt(mu0*epsilon0)`) to get a genuinely physical wave,
and use `check_dispersion_relation` to confirm you did. Once built, the
wave can be fed straight into `core.verify_all` to confirm it satisfies
all four Maxwell equations.

---

## `maxwell.energy`

Energy, momentum, and force carried by the electromagnetic field.

| Call | Purpose |
|---|---|
| `poynting_vector(E, B, mu0)` | S = (1/mu0) * E cross B, the energy flux vector |
| `energy_density(E, B, epsilon0, mu0)` | u = half of (epsilon0*E^2 + B^2/mu0) |
| `check_poynting_theorem(E, B, J, t, epsilon0, mu0)` | checks du/dt + div(S) + J.E = 0 |
| `maxwell_stress_tensor(E, B, epsilon0, mu0)` | the 3x3 stress tensor T |
| `stress_tensor_trace_identity_check(E, B, epsilon0, mu0)` | checks trace(T) = -u |
| `lorentz_force(q, E, B, v)` | F = q*(E + v cross B) |
| `radiation_pressure(intensity, c, reflecting)` | pressure from EM radiation on a surface |

**How to use it:** `poynting_vector` and `energy_density` are the two
quantities everything else is built from — compute them first if you need
either directly. `check_poynting_theorem` is the most useful validation
tool here: it tells you whether a given E, B, J combination conserves
energy correctly (it will hold automatically for any fields that satisfy
Maxwell's equations with that source, and fail if there's an unbalanced
energy sink or source, e.g. dissipation with nothing supplying it).
`radiation_pressure` takes a plain intensity number (W/m^2) and a flag
for whether the surface absorbs or reflects — reflecting always gives
exactly double the pressure of absorbing.

---

## `maxwell.boundary`

Boundary conditions at the interface between two media, and
normal-incidence reflection/transmission.

| Call | Purpose |
|---|---|
| `decompose_field(F, normal)` | splits a field into components perpendicular and parallel to a surface normal |
| `check_tangential_E_continuity(E1, E2, normal)` | checks the tangential component of E is the same on both sides |
| `check_normal_B_continuity(B1, B2, normal)` | checks the normal component of B is the same on both sides |
| `check_normal_D_discontinuity(D1, D2, normal, sigma_free)` | checks the jump in normal D matches the free surface charge |
| `check_tangential_H_discontinuity(H1, H2, normal, K_free)` | checks the jump in tangential H matches the free surface current |
| `normal_incidence_fresnel(n1, n2)` | amplitude reflection/transmission coefficients (r, t) |
| `normal_incidence_power_coefficients(n1, n2)` | power reflectance/transmittance (R, T), and confirms R+T=1 |

**How to use it:** give it the fields on each side of an interface
(labeled 1 and 2) and the interface's unit normal vector; each check
function tells you whether the standard electromagnetic boundary
condition holds between them. The two Fresnel functions are standalone —
you don't need fields for those, just the refractive indices of the two
media — and are the quickest way to get reflection/transmission numbers
for a simple interface.

---

## `maxwell.covariant`

The relativistic 4-tensor formulation of Maxwell's equations.

**Sign convention note:** the field strength tensor is built with rows
and columns ordered (t, x, y, z), with the time coordinate treated as
c*t. With that convention, the inhomogeneous Maxwell equations take the
form d_mu F^{mu nu} = -mu0*J^nu — the minus sign is simply a bookkeeping
consequence of this particular indexing choice (rather than formally
raising/lowering indices with a metric tensor), and this convention has
been verified to reproduce Gauss's law and the Ampere-Maxwell law exactly,
component by component.

| Call | Purpose |
|---|---|
| `field_strength_tensor(E, B, c)` | builds the 4x4 tensor F from E and B |
| `four_current(rho, J, c)` | builds the 4-vector current (c*rho, Jx, Jy, Jz) |
| `covariant_maxwell_check(F, J4, t, x, y, z, mu0, c)` | checks the inhomogeneous equation, component by component |
| `homogeneous_pair_check(E, B, t)` | checks div(B)=0 and Faraday's law together (the physical content of the tensor's Bianchi identity) |
| `field_invariants(E, B, c)` | returns the two frame-independent quantities E^2-c^2*B^2 and E.B |
| `lorentz_boost_fields(E, B, v, c)` | transforms E, B into a frame moving at velocity v along the x-axis |
| `check_invariants_preserved_under_boost(E, B, v, c)` | confirms both invariants are unchanged after a boost |

**How to use it:** this module is for confirming a field genuinely obeys
relativistic electromagnetism, not just the non-relativistic form. Build
E and B as usual, then either package them into the tensor form with
`field_strength_tensor`/`four_current` and run `covariant_maxwell_check`,
or — often more useful in practice — use `lorentz_boost_fields` to see
what a moving observer would measure, and
`check_invariants_preserved_under_boost` to confirm the two physical
invariants of the field really are frame-independent for your fields.

---

## `maxwell.special_cases`

Classical formulas that fall out of Maxwell's equations as special or
limiting cases — deliberately cross-checked against `maxwell.integral`
rather than just defined as standalone formulas.

| Call | Purpose |
|---|---|
| `coulombs_law(q1, q2, r, epsilon0)` | force between two point charges |
| `coulomb_field_matches_gauss(q, r_symbol, r_val, epsilon0)` | confirms the Coulomb field is exactly what Gauss's law requires |
| `biot_savart_wire_field(I, r, mu0)` | field around a long straight current-carrying wire |
| `biot_savart_matches_ampere(I, r_symbol, r_val, mu0)` | confirms the Biot-Savart field is exactly what Ampere's law requires |
| `check_continuity_equation_1d(Jx, rho, x, t)` | checks charge conservation in 1D: dJx/dx + drho/dt = 0 |
| `check_continuity_equation(J, rho, t)` | same, full 3D |
| `skin_depth(mu0, sigma, omega)` | the field-penetration depth into a conductor |

**How to use it:** the Coulomb and Biot-Savart functions give you the
classical formula directly, if that's all you need. The "matches" checks
are there for when you want to demonstrate — rather than just assume —
that these textbook formulas are exactly what Gauss's law and Ampere's
law demand; they're a good sanity check if you're deriving a similar
formula for a new symmetric charge or current distribution and want to
verify it the same way.

---

## Running the test suite and examples

The test suite (152 tests across all 9 modules) can be run with `pytest`
from the project root. Every physics check in the library is tested with
both a genuinely valid solution and a deliberately broken one, so the
tests double as a specification of exactly what each function enforces.

Three runnable example scripts under `examples/` walk through the whole
library using real physical scenarios — visible light, a current-carrying
wire, a point charge, solar sail radiation pressure, an air-to-glass
interface, and a relativistic boost — organized roughly by which modules
they exercise: field fundamentals first, then integral-form and classical
formulas, then waves/energy/boundaries/relativity.