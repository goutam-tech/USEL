import numpy as np
import pytest

from usel.exceptions import ValidationError
from usel.math.transforms import fft, frequency_analysis, ifft


class TestTransforms:
    def test_fft_basic_signal(self):
        signal = np.array([1, 0, 0, 0])

        result = fft(signal)

        assert result.shape == (4,)

    def test_fft_inverse(self):
        signal = np.array([1, 2, 3, 4])

        spectrum = fft(signal)

        recovered = ifft(spectrum)

        assert np.allclose(recovered.real, signal)

    def test_fft_invalid_dimension(self):
        signal = np.array([[1, 2], [3, 4]])

        with pytest.raises(ValidationError):
            fft(signal)

    def test_ifft_invalid_dimension(self):
        spectrum = np.array([[1, 2], [3, 4]])

        with pytest.raises(ValidationError):
            ifft(spectrum)

    def test_frequency_analysis(self):
        signal = np.sin(np.linspace(0, 2 * np.pi, 100))

        freq, magnitude = frequency_analysis(signal, sample_rate=100)

        assert len(freq) == len(magnitude)

    def test_frequency_analysis_invalid_rate(self):
        signal = np.array([1, 2, 3, 4])

        with pytest.raises(ValidationError):
            frequency_analysis(signal, sample_rate=0)

    def test_frequency_analysis_invalid_signal(self):
        signal = np.array([[1, 2], [3, 4]])

        with pytest.raises(ValidationError):
            frequency_analysis(signal, sample_rate=10)

    @staticmethod
    def test_fft_matches_analytical_dft():
        # The DFT of a delta [1, 0, 0, 0] is the all-ones vector.
        signal = np.array([1.0, 0.0, 0.0, 0.0])
        result = fft(signal)
        assert np.allclose(result, np.array([1.0, 1.0, 1.0, 1.0]), atol=1e-12)

    @staticmethod
    def test_fft_of_sinusoid_peaks_at_correct_bin():
        # A sine at exactly one FFT bin should concentrate its energy there.
        n = 64
        sample_rate = 64.0
        t = np.arange(n) / sample_rate
        signal = np.sin(2 * np.pi * 5 * t)  # 5 Hz == bin 5
        spectrum = fft(signal)
        magnitudes = np.abs(spectrum[: n // 2])
        peak_bin = int(np.argmax(magnitudes))
        assert peak_bin == 5
        # DC bin of a pure sine is ~zero.
        assert magnitudes[0] < 1e-10

    @staticmethod
    def test_frequency_analysis_reports_peak_frequency():
        signal = np.sin(2 * np.pi * 7 * np.linspace(0, 1, 1000, endpoint=False))
        freq, magnitude = frequency_analysis(signal, sample_rate=1000)
        peak_freq = freq[int(np.argmax(magnitude))]
        assert peak_freq == pytest.approx(7.0, abs=1.0)
