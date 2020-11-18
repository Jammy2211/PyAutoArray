import os
from os import path

import numpy as np
import pytest
import shutil

import autoarray as aa
from autoarray.structures import arrays

test_values_dir = path.join(
    "{}".format(path.dirname(path.realpath(__file__))), "files", "values"
)


def test__indexes_give_entries_where_list_begin_and_end():

    values = aa.Values(values=[[0.0]])

    assert values.lower_indexes == [0]
    assert values.upper_indexes == [1]

    values = aa.Values(values=[[0.0, 0.0]])

    assert values.lower_indexes == [0]
    assert values.upper_indexes == [2]

    values = aa.Values(values=[[0.0, 0.0], [0.0]])

    assert values.lower_indexes == [0, 2]
    assert values.upper_indexes == [2, 3]

    values = aa.Values(values=[[0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0], [0.0]])

    assert values.lower_indexes == [0, 2, 5, 7]
    assert values.upper_indexes == [2, 5, 7, 8]


def test__input_as_list__convert_correctly():

    values = aa.Values(values=[1.0, -1.0])

    assert type(values) == arrays.Values
    assert (values == np.array([1.0, -1.0])).all()
    assert values.in_grouped_list == [[1.0, -1.0]]
    assert values.in_1d_list == [1.0, -1.0]

    values = aa.Values(values=[[1.0], [-1.0]])

    assert type(values) == arrays.Values
    assert (values == np.array([1.0, -1.0])).all()
    assert values.in_grouped_list == [[1.0], [-1.0]]
    assert values.in_1d_list == [1.0, -1.0]


def test__input_as_dict__retains_dict_and_converts_correctly():

    values = aa.Values(values=dict(source_0=1.0, source_1=-1.0))

    assert type(values) == arrays.Values
    assert (values == np.array([1.0, -1.0])).all()
    assert values.in_grouped_list == [[1.0, -1.0]]
    assert values.in_1d_list == [1.0, -1.0]
    assert values.as_dict["source_0"] == 1.0
    assert values.as_dict["source_1"] == -1.0

    values = aa.Values(values=dict(source_0=[1.0], source_1=[-1.0]))

    assert type(values) == arrays.Values
    assert (values == np.array([1.0, -1.0])).all()
    assert values.in_grouped_list == [[1.0], [-1.0]]
    assert values.in_1d_list == [1.0, -1.0]
    assert values.as_dict["source_0"] == [1.0]
    assert values.as_dict["source_1"] == [-1.0]


def test__values_from_arr_1d():

    values = aa.Values(values=[[1.0, 2.0]])

    values_from_1d = values.values_from_arr_1d(arr_1d=np.array([1.0, 2.0]))

    assert values_from_1d.in_grouped_list == [[1.0, 2.0]]

    values = aa.Values(values=[[1.0, 2.0], [3.0]])

    values_from_1d = values.values_from_arr_1d(arr_1d=np.array([1.0, 2.0, 3.0]))

    assert values_from_1d.in_grouped_list == [[1.0, 2.0], [3.0]]


def test__coordinates_from_grid_1d():

    values = aa.Values(values=[[1.0, 2.0]])

    coordinate_from_1d = values.grid_from_grid_1d(
        grid_1d=np.array([[1.0, 1.0], [2.0, 2.0]])
    )

    assert coordinate_from_1d.in_grouped_list == [[(1.0, 1.0), (2.0, 2.0)]]

    values = aa.Values(values=[[1.0, 2.0], [3.0]])

    coordinate_from_1d = values.grid_from_grid_1d(
        grid_1d=np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    )

    assert coordinate_from_1d.in_grouped_list == [
        [(1.0, 1.0), (2.0, 2.0)],
        [(3.0, 3.0)],
    ]


def test__load_values__retains_list_structure():
    values = aa.Values.from_file(
        file_path=path.join(test_values_dir, "values_test.dat")
    )

    assert values.in_grouped_list == [[1.0, 2.0], [3.0, 4.0, 5.0]]


def test__output_values_to_file():

    values = aa.Values([[4.0, 5.0], [6.0, 7.0, 8.0]])

    output_values_dir = "{}/files/values/output_test/".format(
        path.dirname(path.realpath(__file__))
    )
    if path.exists(output_values_dir):
        shutil.rmtree(output_values_dir)

    os.makedirs(output_values_dir)

    values.output_to_file(file_path=output_values_dir + "values_test.dat")

    values = aa.Values.from_file(file_path=output_values_dir + "values_test.dat")

    assert values.in_grouped_list == [[4.0, 5.0], [6.0, 7.0, 8.0]]

    with pytest.raises(FileExistsError):
        values.output_to_file(file_path=output_values_dir + "values_test.dat")

    values.output_to_file(
        file_path=output_values_dir + "values_test.dat", overwrite=True
    )
