import numpy as np
import pytest

from usel.io.readers import read_csv, read_json, read_npy, read_text

from usel.exceptions import IOError_


def test_read_csv(tmp_path):

    file = tmp_path / "data.csv"

    file.write_text("1,2\n3,4")

    data = read_csv(file)

    assert data.shape == (2, 2)


def test_read_json(tmp_path):

    file = tmp_path / "data.json"

    file.write_text('{"value":10}')

    data = read_json(file)

    assert data["value"] == 10


def test_read_npy(tmp_path):

    file = tmp_path / "array.npy"

    array = np.array([1, 2, 3])

    np.save(file, array)

    result = read_npy(file)

    assert np.array_equal(result, array)


def test_read_text(tmp_path):

    file = tmp_path / "hello.txt"

    file.write_text("usel framework")

    result = read_text(file)

    assert result == "usel framework"


def test_missing_csv():

    with pytest.raises(IOError_):
        read_csv("missing.csv")


def test_missing_json():

    with pytest.raises(IOError_):
        read_json("missing.json")
