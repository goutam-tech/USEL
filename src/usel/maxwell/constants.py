import sympy as sp

EPSILON_0 = sp.Float(8.8541878128e-12)
MU_0 = sp.Float(1.25663706212e-6)
C_LIGHT = sp.Float(299792458)


def speed_of_light_from_em_constants(epsilon0=EPSILON_0, mu0=MU_0):
    return 1 / sp.sqrt(epsilon0 * mu0)
