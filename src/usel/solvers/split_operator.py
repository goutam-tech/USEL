"""
Split operator FFT solver.
"""

import numpy as np



def evolve(
    psi,
    potential,
    grid,
    dt,
    steps,
    mass=1.0,
    hbar=1.0
):


    x = grid.x

    dx = grid.dx


    n=len(x)


    k = (
        2*np.pi*
        np.fft.fftfreq(
            n,
            dx
        )
    )


    kinetic = np.exp(
        -1j*
        hbar*k*k*
        dt/
        (2*mass)
    )


    potential_half = np.exp(
        -1j*
        potential*
        dt/
        (2*hbar)
    )


    states=[]


    current=psi.copy()


    for _ in range(steps):


        current *= potential_half


        momentum = np.fft.fft(
            current
        )


        momentum *= kinetic


        current=np.fft.ifft(
            momentum
        )


        current *= potential_half


        states.append(
            current.copy()
        )


    return np.array(states)