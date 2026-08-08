from __future__ import annotations

import math

import numpy as np
import pytest

from usel.black_scholes.distributions import (
    inverse_normal_cdf,
    normal_cdf,
    normal_log_pdf,
    normal_pdf,
    normal_sf,
)


@pytest.mark.parametrize(
    ("x", "expected"),
    [
        (0.0, 0.3989422804014327),
        (1.0, 0.24197072451914337),
        (-1.0, 0.24197072451914337),
        (2.0, 0.05399096651318806),
        (-2.0, 0.05399096651318806),
    ],
)
def test_normal_pdf_scalar(x: float, expected: float) -> None:
    assert normal_pdf(x) == pytest.approx(expected, rel=1e-12)


def test_normal_pdf_array() -> None:
    x = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])

    expected = np.array(
        [
            0.05399096651318806,
            0.24197072451914337,
            0.3989422804014327,
            0.24197072451914337,
            0.05399096651318806,
        ]
    )

    result = normal_pdf(x)

    assert np.allclose(result, expected)


def test_normal_pdf_is_symmetric() -> None:
    x = np.linspace(-5.0, 5.0, 200)

    assert np.allclose(normal_pdf(x), normal_pdf(-x))


def test_normal_pdf_non_negative() -> None:
    x = np.linspace(-10.0, 10.0, 1000)

    assert np.all(normal_pdf(x) >= 0.0)


@pytest.mark.parametrize("x", [-3.0, -1.0, 0.0, 1.0, 3.0])
def test_normal_log_pdf_matches_pdf(x: float) -> None:
    expected = math.log(normal_pdf(x))

    assert normal_log_pdf(x) == pytest.approx(expected, rel=1e-12)


def test_normal_log_pdf_array() -> None:
    x = np.array([-1.0, 0.0, 1.0])

    result = normal_log_pdf(x)

    assert result.shape == x.shape


@pytest.mark.parametrize(
    ("x", "expected"),
    [
        (0.0, 0.5),
        (1.0, 0.8413447460685429),
        (-1.0, 0.15865525393145707),
        (2.0, 0.9772498680518208),
        (-2.0, 0.02275013194817921),
    ],
)
def test_normal_cdf_scalar(x: float, expected: float) -> None:
    assert normal_cdf(x) == pytest.approx(expected, rel=1e-10)


def test_normal_cdf_array() -> None:
    x = np.array([-1.0, 0.0, 1.0])

    expected = np.array(
        [
            0.15865525393145707,
            0.5,
            0.8413447460685429,
        ]
    )

    result = normal_cdf(x)

    assert np.allclose(result, expected)


def test_normal_cdf_limits() -> None:
    assert normal_cdf(-10.0) < 1e-20
    assert normal_cdf(10.0) > 0.999999999


def test_normal_cdf_symmetry() -> None:
    x = np.linspace(-5.0, 5.0, 100)

    lhs = normal_cdf(x)
    rhs = 1.0 - normal_cdf(-x)

    assert np.allclose(lhs, rhs)


@pytest.mark.parametrize("x", [-4.0, -2.0, -1.0, 0.0, 1.0, 3.0])
def test_normal_sf_relation_to_cdf(x: float) -> None:
    assert normal_sf(x) == pytest.approx(
        1.0 - normal_cdf(x),
        rel=1e-12,
    )


def test_normal_sf_array() -> None:
    x = np.array([-1.0, 0.0, 1.0])

    result = normal_sf(x)

    assert result.shape == x.shape


@pytest.mark.parametrize(
    ("p", "expected"),
    [
        (0.5, 0.0),
        (0.8413447460685429, 1.0),
        (0.15865525393145707, -1.0),
        (0.9772498680518208, 2.0),
        (0.02275013194817921, -2.0),
    ],
)
def test_inverse_normal_cdf_known_values(
    p: float,
    expected: float,
) -> None:
    assert inverse_normal_cdf(p) == pytest.approx(
        expected,
        abs=1e-6,
    )


@pytest.mark.parametrize(
    "probability",
    [
        -0.5,
        0.0,
        1.0,
        2.0,
    ],
)
def test_inverse_normal_cdf_invalid_probability(
    probability: float,
) -> None:
    with pytest.raises(ValueError):
        inverse_normal_cdf(probability)


def test_inverse_normal_cdf_array() -> None:
    probabilities = np.array([0.25, 0.5, 0.75])

    result = inverse_normal_cdf(probabilities)

    assert result.shape == probabilities.shape


@pytest.mark.parametrize(
    "probability",
    [
        0.01,
        0.05,
        0.25,
        0.5,
        0.75,
        0.95,
        0.99,
    ],
)
def test_inverse_is_inverse_of_cdf(
    probability: float,
) -> None:
    x = inverse_normal_cdf(probability)

    recovered = normal_cdf(x)

    assert recovered == pytest.approx(probability, abs=1e-6)


@pytest.mark.parametrize(
    "x",
    np.linspace(-4.0, 4.0, 25),
)
def test_pdf_positive(x: float) -> None:
    assert normal_pdf(x) > 0.0


@pytest.mark.parametrize(
    "x",
    np.linspace(-5.0, 5.0, 50),
)
def test_cdf_in_unit_interval(x: float) -> None:
    value = normal_cdf(x)

    assert 0.0 <= value <= 1.0


@pytest.mark.parametrize(
    "x",
    np.linspace(-5.0, 5.0, 50),
)
def test_sf_in_unit_interval(x: float) -> None:
    value = normal_sf(x)

    assert 0.0 <= value <= 1.0


def test_pdf_log_pdf_consistency() -> None:
    x = np.linspace(-4.0, 4.0, 100)

    assert np.allclose(
        np.exp(normal_log_pdf(x)),
        normal_pdf(x),
    )
