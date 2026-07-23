import pytest
import numpy as np

from usel.math.transforms import fft, ifft, frequency_analysis

from usel.exceptions import ValidationError


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
