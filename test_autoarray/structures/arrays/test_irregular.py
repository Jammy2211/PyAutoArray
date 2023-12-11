import os
from os import path

import numpy as np
import pytest
import shutil

import autoarray as aa

test_values_dir = path.join("{}".format(path.dirname(path.realpath(__file__))), "files")


def test__input_as_list__convert_correctly():
    values = aa.ArrayIrregular(values=[1.0, -1.0])

    assert type(values) == aa.ArrayIrregular
    assert (values == np.array([1.0, -1.0])).all()
    assert values.in_list == [1.0, -1.0]


def test__values_from():
    values = aa.ArrayIrregular(values=[1.0, 2.0])

    values_from_1d = values.values_from(array_slim=np.array([1.0, 2.0]))

    assert values_from_1d.in_list == [1.0, 2.0]

    values = aa.ArrayIrregular(values=[1.0, 2.0, 3.0])

    values_from_1d = values.values_from(array_slim=np.array([1.0, 2.0, 3.0]))

    assert values_from_1d.in_list == [1.0, 2.0, 3.0]


def test__coordinates_from_grid_1d():
    values = aa.ArrayIrregular(values=[1.0, 2.0])

    coordinate_from_1d = values.grid_from(grid_slim=np.array([[1.0, 1.0], [2.0, 2.0]]))

    assert coordinate_from_1d.in_list == [(1.0, 1.0), (2.0, 2.0)]

    values = aa.ArrayIrregular(values=[[1.0, 2.0, 3.0]])

    coordinate_from_1d = values.grid_from(
        grid_slim=np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    )

    assert coordinate_from_1d.in_list == [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]


def test__output_values_to_json():
    values = aa.ArrayIrregular(values=[6.0, 7.0, 8.0])

    output_values_dir = path.join(
        "{}".format(path.dirname(path.realpath(__file__))),
        "files",
        "values",
        "output_test",
    )

    if path.exists(output_values_dir):
        shutil.rmtree(output_values_dir)

    os.makedirs(output_values_dir)

    file_path = path.join(output_values_dir, "values_test.dat")

    values.output_to_json(file_path=file_path)

    values = aa.ArrayIrregular.from_file(file_path=file_path)

    assert values.in_list == [6.0, 7.0, 8.0]

    with pytest.raises(FileExistsError):
        values.output_to_json(file_path=file_path)

    values.output_to_json(file_path=file_path, overwrite=True)
