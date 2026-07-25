from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def plot_probability(x, probability, title="Dirac Probability Density"):

    plt.figure(figsize=(8, 4))

    plt.plot(x, probability)

    plt.xlabel("Position x")

    plt.ylabel("Probability density")

    plt.title(title)

    plt.grid(True)

    plt.tight_layout()

    plt.show()


def plot_spinor_components(x, spinor, title="Dirac Spinor Components"):

    if spinor.shape[0] != 4:
        raise ValueError("Dirac spinor must have 4 components")

    plt.figure(figsize=(9, 5))

    labels = ["ψ1", "ψ2", "ψ3", "ψ4"]

    for i in range(4):
        plt.plot(x, np.real(spinor[i]), label=labels[i])

    plt.xlabel("Position x")

    plt.ylabel("Amplitude")

    plt.title(title)

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.show()


def plot_spin_expectation(t, spin, title="Spin Expectation"):

    plt.figure(figsize=(8, 4))

    labels = ["Sx", "Sy", "Sz"]

    for i in range(3):
        plt.plot(t, spin[:, i], label=labels[i])

    plt.xlabel("Time")

    plt.ylabel("Spin")

    plt.title(title)

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.show()


def plot_probability_evolution(x, probability_history, times=None):

    plt.figure(figsize=(8, 5))

    if times is None:
        times = np.arange(len(probability_history))

    extent = [x[0], x[-1], times[0], times[-1]]

    plt.imshow(probability_history, aspect="auto", origin="lower", extent=extent)

    plt.xlabel("Position x")

    plt.ylabel("Time")

    plt.title("Dirac Particle Evolution")

    plt.colorbar(label="Probability")

    plt.tight_layout()

    plt.show()
