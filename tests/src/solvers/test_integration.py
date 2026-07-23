import pytest

from usel.solvers.integration import trapezoidal, simpson, gaussian_quadrature

from usel.exceptions import ValidationError


def test_trapezoidal():

    result = trapezoidal(lambda x: x * x, 0, 1)

    assert result == pytest.approx(1 / 3, rel=1e-3)


def test_simpson():

    result = simpson(lambda x: x * x, 0, 1)

    assert result == pytest.approx(1 / 3, rel=1e-6)


def test_gaussian_quadrature():

    result = gaussian_quadrature(lambda x: x * x, 0, 1)

    assert result == pytest.approx(1 / 3, rel=1e-10)


def test_trapezoidal_invalid_n():

    with pytest.raises(ValidationError):

        trapezoidal(lambda x: x, 0, 1, n=0)


def test_simpson_invalid_n():

    with pytest.raises(ValidationError):

        simpson(lambda x: x, 0, 1, n=1)


def test_gaussian_invalid_n():

    with pytest.raises(ValidationError):

        gaussian_quadrature(lambda x: x, 0, 1, n=0)
