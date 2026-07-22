"""Fast Fourier Transform utilities and frequency analysis."""

from __future__ import annotations

import numpy as np

from usel.exceptions import ValidationError


def fft(signal: np.ndarray) -> np.ndarray:
    """Compute the Fast Fourier Transform of a 1D real or complex signal."""
    array = np.asarray(signal)
    if array.ndim != 1:
        raise ValidationError("fft expects a 1-dimensional signal")
    return np.fft.fft(array)


def ifft(spectrum: np.ndarray) -> np.ndarray:
    """Compute the Inverse Fast Fourier Transform of a spectrum."""
    array = np.asarray(spectrum)
    if array.ndim != 1:
        raise ValidationError("ifft expects a 1-dimensional spectrum")
    return np.fft.ifft(array)


def frequency_analysis(
    signal: np.ndarray, sample_rate: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return (frequencies, magnitudes) for the positive-frequency spectrum.

    Parameters
    ----------
    signal:
        The 1D time-domain signal.
    sample_rate:
        Number of samples per unit time (Hz).
    """
    array = np.asarray(signal)
    if array.ndim != 1:
        raise ValidationError("frequency_analysis expects a 1-dimensional signal")
    if sample_rate <= 0:
        raise ValidationError("sample_rate must be positive")

    n = array.shape[0]
    spectrum = np.fft.fft(array)
    frequencies = np.fft.fftfreq(n, d=1.0 / sample_rate)

    half = n // 2
    magnitudes = np.abs(spectrum[:half]) / n
    return frequencies[:half], magnitudes
