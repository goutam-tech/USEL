"""Example 2: FFT Analysis — detecting a dominant frequency in a signal."""

from __future__ import annotations

import numpy as np

from usel.math import frequency_analysis


def main() -> None:
    sample_rate = 500.0
    t = np.arange(0, 1, 1 / sample_rate)
    signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)

    frequencies, magnitudes = frequency_analysis(signal, sample_rate)
    top_indices = np.argsort(magnitudes)[-2:][::-1]

    print("Dominant frequencies detected:")
    for idx in top_indices:
        print(f"  {frequencies[idx]:.1f} Hz (magnitude {magnitudes[idx]:.3f})")


if __name__ == "__main__":
    main()
