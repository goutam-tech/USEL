from usel.dirac.spinors import np, positive_energy_spinor, spin_expectation, spinor_norm

p = np.array([0.1, 0, 0])

m = 1.0

E = np.sqrt(np.dot(p, p) + m * m)


psi = positive_energy_spinor(p, m, E, "up")


print(psi)

print("Norm:", spinor_norm(psi))

print("Spin:", spin_expectation(psi))
