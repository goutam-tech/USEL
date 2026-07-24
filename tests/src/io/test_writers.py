import numpy as np

from usel.io.readers import read_csv, read_json, read_npy, read_text
from usel.io.writers import write_csv, write_json, write_npy, write_text


def test_write_csv(tmp_path):

    file = tmp_path / "data.csv"

    data = np.array([[1, 2], [3, 4]])

    write_csv(data, file)

    result = read_csv(file)

    assert np.array_equal(result, data)


def test_write_json(tmp_path):

    file = tmp_path / "data.json"

    obj = {"name": "usel"}

    write_json(obj, file)

    result = read_json(file)

    assert result == obj


def test_write_npy(tmp_path):

    file = tmp_path / "array.npy"

    data = np.array([10, 20, 30])

    write_npy(data, file)

    result = read_npy(file)

    assert np.array_equal(result, data)


def test_write_text(tmp_path):

    file = tmp_path / "text.txt"

    write_text("hello usel", file)

    result = read_text(file)

    assert result == "hello usel"
