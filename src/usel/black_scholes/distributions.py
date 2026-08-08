"""
References
----------
Black, F., & Scholes, M. (1973).
The Pricing of Options and Corporate Liabilities.

Abramowitz, M., & Stegun, I.
Handbook of Mathematical Functions.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import DistributionConstants


def _asarray(x: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(x, dtype=np.float64)


def _return_scalar_or_array(
    original: ArrayLike,
    value: NDArray[np.float64],
):
    if np.isscalar(original):
        return float(value)
    return value


def normal_pdf(x: ArrayLike):
    original = x
    x = _asarray(x)

    pdf = DistributionConstants.INV_SQRT_2PI * np.exp(-0.5 * x * x)

    return _return_scalar_or_array(original, pdf)


def normal_log_pdf(x: ArrayLike):
    original = x
    x = _asarray(x)

    log_pdf = -0.5 * x * x - math.log(DistributionConstants.SQRT_2PI)

    return _return_scalar_or_array(original, log_pdf)


def normal_cdf(x: ArrayLike):
    original = x
    x = _asarray(x)

    erf = np.vectorize(math.erf)

    cdf = 0.5 * (1.0 + erf(x / DistributionConstants.SQRT_2))

    return _return_scalar_or_array(original, cdf)


def normal_sf(x: ArrayLike):
    original = x
    x = _asarray(x)

    erfc = np.vectorize(math.erfc)

    sf = 0.5 * erfc(x / DistributionConstants.SQRT_2)

    return _return_scalar_or_array(original, sf)


def inverse_normal_cdf(p: ArrayLike):
    original = p
    p = _asarray(p)

    if np.any((p <= 0.0) | (p >= 1.0)):
        raise ValueError("Probabilities must satisfy 0 < p < 1.")

    a = np.array(
        [
            -3.969683028665376e01,
            2.209460984245205e02,
            -2.759285104469687e02,
            1.383577518672690e02,
            -3.066479806614716e01,
            2.506628277459239e00,
        ]
    )

    b = np.array(
        [
            -5.447609879822406e01,
            1.615858368580409e02,
            -1.556989798598866e02,
            6.680131188771972e01,
            -1.328068155288572e01,
        ]
    )

    c = np.array(
        [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e00,
            -2.549732539343734e00,
            4.374664141464968e00,
            2.938163982698783e00,
        ]
    )

    d = np.array(
        [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e00,
            3.754408661907416e00,
        ]
    )

    plow = 0.02425
    phigh = 1.0 - plow

    x = np.empty_like(p)

    lower = p < plow
    upper = p > phigh
    middle = ~(lower | upper)

    if np.any(lower):
        q = np.sqrt(-2.0 * np.log(p[lower]))
        x[lower] = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )

    elif np.any(upper):
        q = np.sqrt(-2.0 * np.log(1.0 - p[upper]))
        x[upper] = -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )

    if np.any(middle):
        q = p[middle] - 0.5
        r = q * q

        x[middle] = ((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q) / (
            ((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0
        )

    return _return_scalar_or_array(original, x)
