import numpy as np

from usel.random.generators import (
    gaussian_numbers,
    random_matrix,
    random_state,
    random_vector,
    uniform_numbers,
)


def test_random_matrix_shape():
    matrix = random_matrix(3, 4, seed=1)

    assert matrix.shape == (3, 4)


def test_random_matrix_reproducibility():
    a = random_matrix(3, 3, seed=10)

    b = random_matrix(3, 3, seed=10)

    assert a == b


def test_gaussian_numbers():
    values = gaussian_numbers(100, seed=1)

    assert len(values) == 100


def test_uniform_numbers():
    values = uniform_numbers(50, low=5, high=10, seed=1)

    assert len(values) == 50

    assert np.all(values >= 5)

    assert np.all(values < 10)


def test_random_vector():
    vector = random_vector(10, seed=1)

    assert vector.shape == (10,)


def test_random_state():

    rng1 = random_state(5)

    rng2 = random_state(5)

    assert rng1.random() == rng2.random()
