"""
Visualization tools.
"""

import matplotlib.pyplot as plt


def plot_wavefunction(x, psi, title="Wavefunction"):
    plt.plot(x, psi.real, label="Real")

    plt.plot(x, psi.imag, label="Imaginary")

    plt.xlabel("Position")

    plt.ylabel("ψ(x)")

    plt.title(title)

    plt.legend()

    plt.grid()

    plt.show()


def plot_probability(x, psi):
    plt.plot(x, abs(psi) ** 2)

    plt.xlabel("Position")

    plt.ylabel("|ψ|²")

    plt.title("Probability Density")

    plt.grid()

    plt.show()


def animate_evolution(x, states):
    import matplotlib.animation as animation

    fig, ax = plt.subplots()
    (line,) = ax.plot(x, abs(states[0]) ** 2)

    def update(i):

        line.set_ydata(abs(states[i]) ** 2)

        return (line,)

    animation.FuncAnimation(fig, update, frames=len(states), interval=50)

    plt.show()
