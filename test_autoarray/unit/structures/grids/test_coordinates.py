import os
import shutil
import numpy as np
import pytest

import autoarray as aa
from autoarray.structures import grids

test_coordinates_dir = "{}/files/coordinates/".format(
    os.path.dirname(os.path.realpath(__file__))
)


def test__indexes_give_entries_where_list_begin_and_end():

    coordinates = aa.GridCoordinates(coordinates=[[(0.0, 0.0)]])

    assert coordinates.lower_indexes == [0]
    assert coordinates.upper_indexes == [1]

    coordinates = aa.GridCoordinates(coordinates=[[(0.0, 0.0), (0.0, 0.0)]])

    assert coordinates.lower_indexes == [0]
    assert coordinates.upper_indexes == [2]

    coordinates = aa.GridCoordinates(
        coordinates=[[(0.0, 0.0), (0.0, 0.0)], [(0.0, 0.0)]]
    )

    assert coordinates.lower_indexes == [0, 2]
    assert coordinates.upper_indexes == [2, 3]

    coordinates = aa.GridCoordinates(
        coordinates=[
            [(0.0, 0.0), (0.0, 0.0)],
            [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
            [(0.0, 0.0), (0.0, 0.0)],
            [(0.0, 0.0)],
        ]
    )

    assert coordinates.lower_indexes == [0, 2, 5, 7]
    assert coordinates.upper_indexes == [2, 5, 7, 8]


def test__input_as_list_or_list_of_other_types__all_convert_correctly():

    # Input tuples

    coordinates = aa.GridCoordinates(coordinates=[(1.0, -1.0), (1.0, 1.0)])

    assert type(coordinates) == grids.GridCoordinates
    assert (coordinates == np.array([[[1.0, -1.0], [1.0, 1.0]]])).all()
    assert coordinates.in_list == [[(1.0, -1.0), (1.0, 1.0)]]

    coordinates = aa.GridCoordinates(coordinates=[[(1.0, -1.0), (1.0, 1.0)]])

    assert type(coordinates) == grids.GridCoordinates
    assert (coordinates == np.array([[[1.0, -1.0], [1.0, 1.0]]])).all()
    assert coordinates.in_list == [[(1.0, -1.0), (1.0, 1.0)]]

    coordinates = aa.GridCoordinates(coordinates=[(1.0, -1.0), (1.0, 1.0)])

    assert type(coordinates) == grids.GridCoordinates
    assert (coordinates == np.array([[[1.0, -1.0], [1.0, 1.0]]])).all()
    assert coordinates.in_list == [[(1.0, -1.0), (1.0, 1.0)]]

    coordinates = aa.GridCoordinates(coordinates=[[(1.0, -1.0), (1.0, 1.0)]])

    assert type(coordinates) == grids.GridCoordinates
    assert (coordinates == np.array([[[1.0, -1.0], [1.0, 1.0]]])).all()
    assert coordinates.in_list == [[(1.0, -1.0), (1.0, 1.0)]]

    # Input np array

    coordinates = aa.GridCoordinates(
        coordinates=[np.array([1.0, -1.0]), np.array([1.0, 1.0])]
    )

    assert type(coordinates) == grids.GridCoordinates
    assert (coordinates == np.array([[[1.0, -1.0], [1.0, 1.0]]])).all()
    assert coordinates.in_list == [[(1.0, -1.0), (1.0, 1.0)]]

    coordinates = aa.GridCoordinates(
        coordinates=[[np.array([1.0, -1.0]), np.array([1.0, 1.0])]]
    )

    assert type(coordinates) == grids.GridCoordinates
    assert (coordinates == np.array([[[1.0, -1.0], [1.0, 1.0]]])).all()
    assert coordinates.in_list == [[(1.0, -1.0), (1.0, 1.0)]]

    coordinates = aa.GridCoordinates(coordinates=[np.array([[1.0, -1.0], [1.0, 1.0]])])

    assert type(coordinates) == grids.GridCoordinates
    assert (coordinates == np.array([[[1.0, -1.0], [1.0, 1.0]]])).all()
    assert coordinates.in_list == [[(1.0, -1.0), (1.0, 1.0)]]

    # Input list

    coordinates = aa.GridCoordinates(coordinates=[[1.0, -1.0], [1.0, 1.0]])

    assert type(coordinates) == grids.GridCoordinates
    assert (coordinates == np.array([[[1.0, -1.0], [1.0, 1.0]]])).all()
    assert coordinates.in_list == [[(1.0, -1.0), (1.0, 1.0)]]

    coordinates = aa.GridCoordinates(coordinates=[[[1.0, -1.0]], [[1.0, 1.0]]])

    assert type(coordinates) == grids.GridCoordinates
    assert (coordinates == np.array([[[1.0, -1.0], [1.0, 1.0]]])).all()
    assert coordinates.in_list == [[(1.0, -1.0)], [(1.0, 1.0)]]

    coordinates = aa.GridCoordinates(coordinates=[[[1.0, -1.0], [1.0, 1.0]]])

    assert type(coordinates) == grids.GridCoordinates
    assert (coordinates == np.array([[[1.0, -1.0], [1.0, 1.0]]])).all()
    assert coordinates.in_list == [[(1.0, -1.0), (1.0, 1.0)]]


def test__values_from_arr_1d():

    coordinates = aa.GridCoordinates(coordinates=[[(1.0, 1.0), (2.0, 2.0)]])

    values_from_1d = coordinates.values_from_arr_1d(arr_1d=np.array([1.0, 2.0]))

    assert values_from_1d.in_list == [[1.0, 2.0]]

    coordinates = aa.GridCoordinates(
        coordinates=[[(1.0, 1.0), (2.0, 2.0)], [(3.0, 3.0)]]
    )

    values_from_1d = coordinates.values_from_arr_1d(arr_1d=np.array([1.0, 2.0, 3.0]))

    assert values_from_1d.in_list == [[1.0, 2.0], [3.0]]


def test__coordinates_from_grid_1d():

    coordinates = aa.GridCoordinates(coordinates=[[(1.0, 1.0), (2.0, 2.0)]])

    coordinates_from_1d = coordinates.coordinates_from_grid_1d(
        grid_1d=np.array([[1.0, 1.0], [2.0, 2.0]])
    )

    assert type(coordinates_from_1d) == grids.GridCoordinates
    assert coordinates_from_1d.in_list == [[(1.0, 1.0), (2.0, 2.0)]]

    coordinates = aa.GridCoordinates(
        coordinates=[[(1.0, 1.0), (2.0, 2.0)], [(3.0, 3.0)]]
    )

    coordinates_from_1d = coordinates.coordinates_from_grid_1d(
        grid_1d=np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    )

    assert type(coordinates_from_1d) == grids.GridCoordinates
    assert coordinates_from_1d.in_list == [[(1.0, 1.0), (2.0, 2.0)], [(3.0, 3.0)]]

    coordinates_from_1d = coordinates.coordinates_from_grid_1d(
        grid_1d=np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    )

    assert type(coordinates_from_1d) == grids.GridCoordinates
    assert coordinates_from_1d.in_list == [[(1.0, 1.0), (2.0, 2.0)], [(3.0, 3.0)]]


def test__with_mask__converts_to_and_from_pixels():

    mask = aa.Mask.manual(
        mask=np.full(fill_value=False, shape=(2, 2)), pixel_scales=(2.0, 2.0)
    )

    coordinates = aa.GridCoordinates(coordinates=[[(1.0, -1.0), (1.0, 1.0)]])

    assert coordinates.in_list == [[(1.0, -1.0), (1.0, 1.0)]]

    coordinates = aa.GridCoordinates.from_pixels_and_mask(
        pixels=[[(0, 0), (0, 1)]], mask=mask
    )

    assert coordinates.in_list == [[(1.0, -1.0), (1.0, 1.0)]]


def test__from_yx_1d():

    coordinates = aa.GridCoordinates.from_yx_1d(y=[1.0, 1.0], x=[-1.0, 1.0])

    assert coordinates.in_list == [[(1.0, -1.0), (1.0, 1.0)]]

    coordinates = aa.GridCoordinates.from_yx_1d(
        y=[1.0, 1.0, 2.0, 4.0], x=[-1.0, 1.0, 3.0, 5.0]
    )

    assert coordinates.in_list == [[(1.0, -1.0), (1.0, 1.0), (2.0, 3.0), (4.0, 5.0)]]


def test__load_coordinates__retains_list_structure():
    coordinates = aa.GridCoordinates.from_file(
        file_path=test_coordinates_dir + "coordinates_test.dat"
    )

    assert coordinates.in_list == [
        [(1.0, 1.0), (2.0, 2.0)],
        [(3.0, 3.0), (4.0, 4.0), (5.0, 6.0)],
    ]


def test__output_coordinates_to_file():
    coordinates = aa.GridCoordinates(
        [[(4.0, 4.0), (5.0, 5.0)], [(6.0, 6.0), (7.0, 7.0), (8.0, 8.0)]]
    )

    output_coordinates_dir = "{}/files/coordinates/output_test/".format(
        os.path.dirname(os.path.realpath(__file__))
    )
    if os.path.exists(output_coordinates_dir):
        shutil.rmtree(output_coordinates_dir)

    os.makedirs(output_coordinates_dir)

    coordinates.output_to_file(
        file_path=output_coordinates_dir + "coordinates_test.dat"
    )

    coordinates = aa.GridCoordinates.from_file(
        file_path=output_coordinates_dir + "coordinates_test.dat"
    )

    assert coordinates.in_list == [
        [(4.0, 4.0), (5.0, 5.0)],
        [(6.0, 6.0), (7.0, 7.0), (8.0, 8.0)],
    ]

    with pytest.raises(FileExistsError):
        coordinates.output_to_file(
            file_path=output_coordinates_dir + "coordinates_test.dat"
        )

    coordinates.output_to_file(
        file_path=output_coordinates_dir + "coordinates_test.dat", overwrite=True
    )


def test__structure_from_result__maps_numpy_array_to__auto_array_or_grid():

    coordinates = aa.GridCoordinates(coordinates=[(1.0, -1.0), (1.0, 1.0)])

    result = coordinates.structure_from_result(result=np.array([1.0, 2.0]))

    assert isinstance(result, aa.Values)
    assert result.in_list == [[1.0, 2.0]]

    result = coordinates.structure_from_result(
        result=np.array([[1.0, 1.0], [2.0, 2.0]])
    )

    assert isinstance(result, aa.GridCoordinates)
    assert result.in_list == [[(1.0, 1.0), (2.0, 2.0)]]


def test__structure_list_from_result_list__maps_list_to_auto_arrays_or_grids():

    coordinates = aa.GridCoordinates(coordinates=[(1.0, -1.0), (1.0, 1.0)])

    result = coordinates.structure_from_result(result=[np.array([1.0, 2.0])])

    assert isinstance(result[0], aa.Values)
    assert result[0].in_list == [[1.0, 2.0]]

    result = coordinates.structure_from_result(
        result=[np.array([[1.0, 1.0], [2.0, 2.0]])]
    )

    assert isinstance(result[0], aa.GridCoordinates)
    assert result[0].in_list == [[(1.0, 1.0), (2.0, 2.0)]]
