import dataclasses

import pytest

from usel.dirac.constants import (
    NATURAL,
    SI,
    NaturalUnits,
    PhysicalConstants,
)


def test_si_constants_exist():
    assert isinstance(SI, PhysicalConstants)


def test_physical_constants_values():
    assert SI.c > 0
    assert SI.hbar > 0
    assert SI.electron_mass > 0
    assert SI.epsilon0 > 0


def test_electron_charge_is_negative():
    assert SI.electron_charge < 0


def test_natural_units():
    assert isinstance(NATURAL, NaturalUnits)

    assert NATURAL.c == 1.0
    assert NATURAL.hbar == 1.0


def test_constants_are_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        SI.c = 10
