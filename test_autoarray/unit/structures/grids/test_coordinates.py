import os
import shutil
import numpy as np
import pytest

import autoarray as aa
from autoarray.structures import grids

test_coordinates_dir = "{}/files/coordinates/".format(
    os.path.dirname(os.path.realpath(__file__))
)


class TestAbstractGridCoordinates:
    def test__indexes_give_entries_where_list_begin_and_end(self):

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

    def test__input_as_list_or_list_of_other_types__all_convert_correctly(self):

        # Input tuples

        coordinates = aa.GridCoordinates(coordinates=[(1.0, -1.0), (1.0, 1.0)])

        assert type(coordinates) == grids.GridCoordinates
        assert (coordinates == np.array([[[1.0, -1.0], [1.0, 1.0]]])).all()
        assert coordinates.in_list == [[(1.0, -1.0), (1.0, 1.0)]]
        assert coordinates.in_1d_list == [(1.0, -1.0), (1.0, 1.0)]

        coordinates = aa.GridCoordinates(coordinates=[[(1.0, -1.0), (1.0, 1.0)]])

        assert type(coordinates) == grids.GridCoordinates
        assert (coordinates == np.array([[[1.0, -1.0], [1.0, 1.0]]])).all()
        assert coordinates.in_list == [[(1.0, -1.0), (1.0, 1.0)]]
        assert coordinates.in_1d_list == [(1.0, -1.0), (1.0, 1.0)]

        coordinates = aa.GridCoordinates(coordinates=[(1.0, -1.0), (1.0, 1.0)])

        assert type(coordinates) == grids.GridCoordinates
        assert (coordinates == np.array([[[1.0, -1.0], [1.0, 1.0]]])).all()
        assert coordinates.in_list == [[(1.0, -1.0), (1.0, 1.0)]]
        assert coordinates.in_1d_list == [(1.0, -1.0), (1.0, 1.0)]

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
        assert coordinates.in_1d_list == [(1.0, -1.0), (1.0, 1.0)]

        coordinates = aa.GridCoordinates(
            coordinates=[[np.array([1.0, -1.0]), np.array([1.0, 1.0])]]
        )

        assert type(coordinates) == grids.GridCoordinates
        assert (coordinates == np.array([[[1.0, -1.0], [1.0, 1.0]]])).all()
        assert coordinates.in_list == [[(1.0, -1.0), (1.0, 1.0)]]
        assert coordinates.in_1d_list == [(1.0, -1.0), (1.0, 1.0)]

        coordinates = aa.GridCoordinates(
            coordinates=[np.array([[1.0, -1.0], [1.0, 1.0]])]
        )

        assert type(coordinates) == grids.GridCoordinates
        assert (coordinates == np.array([[[1.0, -1.0], [1.0, 1.0]]])).all()
        assert coordinates.in_list == [[(1.0, -1.0), (1.0, 1.0)]]
        assert coordinates.in_1d_list == [(1.0, -1.0), (1.0, 1.0)]

        # Input list

        coordinates = aa.GridCoordinates(coordinates=[[1.0, -1.0], [1.0, 1.0]])

        assert type(coordinates) == grids.GridCoordinates
        assert (coordinates == np.array([[[1.0, -1.0], [1.0, 1.0]]])).all()
        assert coordinates.in_list == [[(1.0, -1.0), (1.0, 1.0)]]
        assert coordinates.in_1d_list == [(1.0, -1.0), (1.0, 1.0)]

        coordinates = aa.GridCoordinates(coordinates=[[[1.0, -1.0]], [[1.0, 1.0]]])

        assert type(coordinates) == grids.GridCoordinates
        assert (coordinates == np.array([[[1.0, -1.0], [1.0, 1.0]]])).all()
        assert coordinates.in_list == [[(1.0, -1.0)], [(1.0, 1.0)]]
        assert coordinates.in_1d_list == [(1.0, -1.0), (1.0, 1.0)]

        coordinates = aa.GridCoordinates(coordinates=[[[1.0, -1.0], [1.0, 1.0]]])

        assert type(coordinates) == grids.GridCoordinates
        assert (coordinates == np.array([[[1.0, -1.0], [1.0, 1.0]]])).all()
        assert coordinates.in_list == [[(1.0, -1.0), (1.0, 1.0)]]
        assert coordinates.in_1d_list == [(1.0, -1.0), (1.0, 1.0)]

    def test__input_as_dict__retains_dict_and_convert_correctly(self):

        # Input tuples

        coordinates = aa.GridCoordinates(
            coordinates=dict(source_0=[(1.0, -1.0), (1.0, 1.0)], source_1=[(2.0, 2.0)])
        )

        assert type(coordinates) == grids.GridCoordinates
        assert (coordinates == np.array([[[1.0, -1.0], [1.0, 1.0], [2.0, 2.0]]])).all()
        assert coordinates.in_list == [[(1.0, -1.0), (1.0, 1.0)], [(2.0, 2.0)]]
        assert coordinates.in_1d_list == [(1.0, -1.0), (1.0, 1.0), (2.0, 2.0)]
        assert coordinates.as_dict["source_0"] == [(1.0, -1.0), (1.0, 1.0)]
        assert coordinates.as_dict["source_1"] == [(2.0, 2.0)]

    def test__values_from_arr_1d(self):

        coordinates = aa.GridCoordinates(coordinates=[[(1.0, 1.0), (2.0, 2.0)]])

        values_from_1d = coordinates.values_from_arr_1d(arr_1d=np.array([1.0, 2.0]))

        assert values_from_1d.in_list == [[1.0, 2.0]]

        coordinates = aa.GridCoordinates(
            coordinates=[[(1.0, 1.0), (2.0, 2.0)], [(3.0, 3.0)]]
        )

        values_from_1d = coordinates.values_from_arr_1d(
            arr_1d=np.array([1.0, 2.0, 3.0])
        )

        assert values_from_1d.in_list == [[1.0, 2.0], [3.0]]

    def test__with_mask__converts_to_and_from_pixels(self):

        mask = aa.Mask.manual(
            mask=np.full(fill_value=False, shape=(2, 2)), pixel_scales=(2.0, 2.0)
        )

        coordinates = aa.GridCoordinates(coordinates=[[(1.0, -1.0), (1.0, 1.0)]])

        assert coordinates.in_list == [[(1.0, -1.0), (1.0, 1.0)]]

        coordinates = aa.GridCoordinates.from_pixels_and_mask(
            pixels=[[(0, 0), (0, 1)]], mask=mask
        )

        assert coordinates.in_list == [[(1.0, -1.0), (1.0, 1.0)]]

    def test__load_coordinates__retains_list_structure(self):
        coordinates = aa.GridCoordinates.from_file(
            file_path=test_coordinates_dir + "coordinates_test.dat"
        )

        assert coordinates.in_list == [
            [(1.0, 1.0), (2.0, 2.0)],
            [(3.0, 3.0), (4.0, 4.0), (5.0, 6.0)],
        ]

    def test__output_coordinates_to_file(self):
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


class TestGridCoordinates:
    def test__from_yx_1d(self):
        coordinates = aa.GridCoordinates.from_yx_1d(y=[1.0, 1.0], x=[-1.0, 1.0])

        assert coordinates.in_list == [[(1.0, -1.0), (1.0, 1.0)]]

        coordinates = aa.GridCoordinates.from_yx_1d(
            y=[1.0, 1.0, 2.0, 4.0], x=[-1.0, 1.0, 3.0, 5.0]
        )

        assert coordinates.in_list == [
            [(1.0, -1.0), (1.0, 1.0), (2.0, 3.0), (4.0, 5.0)]
        ]

    def test__grid_from_deflection_grid(self):
        coordinates = aa.GridCoordinates(coordinates=[[(1.0, 1.0), (2.0, 2.0)]])

        coordinates = coordinates.grid_from_deflection_grid(
            deflection_grid=np.array([[1.0, 0.0], [1.0, 1.0]])
        )

        assert coordinates.in_list == [[(0.0, 1.0), (1.0, 1.0)]]

    def test__coordinates_from_grid_1d(self):

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

    def test__structure_from_result__maps_numpy_array_to__auto_array_or_grid(self):
        coordinates = aa.GridCoordinates(coordinates=[(1.0, -1.0), (1.0, 1.0)])

        result = coordinates.structure_from_result(result=np.array([1.0, 2.0]))

        assert isinstance(result, aa.Values)
        assert result.in_list == [[1.0, 2.0]]

        result = coordinates.structure_from_result(
            result=np.array([[1.0, 1.0], [2.0, 2.0]])
        )

        assert isinstance(result, aa.GridCoordinates)
        assert result.in_list == [[(1.0, 1.0), (2.0, 2.0)]]

    def test__structure_list_from_result_list__maps_list_to_auto_arrays_or_grids(self):
        coordinates = aa.GridCoordinates(coordinates=[(1.0, -1.0), (1.0, 1.0)])

        result = coordinates.structure_from_result(result=[np.array([1.0, 2.0])])

        assert isinstance(result[0], aa.Values)
        assert result[0].in_list == [[1.0, 2.0]]

        result = coordinates.structure_from_result(
            result=[np.array([[1.0, 1.0], [2.0, 2.0]])]
        )

        assert isinstance(result[0], aa.GridCoordinates)
        assert result[0].in_list == [[(1.0, 1.0), (2.0, 2.0)]]


class TestGridCoordinatesUniform:
    def test__from_sparse_uniform_upscale(self):

        grid_sparse_uniform = aa.GridCoordinatesUniform(
            coordinates=[[(1.0, 1.0), (1.0, 3.0)]], pixel_scales=2.0
        )

        grid_upscale = grids.GridCoordinatesUniform.from_grid_sparse_uniform_upscale(
            grid_sparse_uniform=grid_sparse_uniform, upscale_factor=2, pixel_scales=2.0
        )

        assert (
            grid_upscale
            == np.array(
                [
                    [1.5, 0.5],
                    [1.5, 1.5],
                    [0.5, 0.5],
                    [0.5, 1.5],
                    [1.5, 2.5],
                    [1.5, 3.5],
                    [0.5, 2.5],
                    [0.5, 3.5],
                ]
            )
        ).all()

        grid_sparse_uniform = aa.GridCoordinatesUniform(
            coordinates=[[(1.0, 1.0), (1.0, 3.0), (1.0, 5.0), (3.0, 3.0)]],
            pixel_scales=2.0,
        )

        grid_upscale = grids.GridCoordinatesUniform.from_grid_sparse_uniform_upscale(
            grid_sparse_uniform=grid_sparse_uniform, upscale_factor=4, pixel_scales=2.0
        )

        grid_upscale_util = aa.util.grid.grid_upscaled_1d_from(
            grid_1d=grid_sparse_uniform, upscale_factor=4, pixel_scales=(2.0, 2.0)
        )

        assert (grid_upscale == grid_upscale_util).all()

    def test__grid_from_deflection_grid(self):
        coordinates = aa.GridCoordinatesUniform(
            coordinates=[[(1.0, 1.0), (2.0, 2.0)]],
            pixel_scales=(1.0, 1.0),
            shape_2d=(3, 3),
        )

        coordinates = coordinates.grid_from_deflection_grid(
            deflection_grid=np.array([[1.0, 0.0], [1.0, 1.0]])
        )

        assert type(coordinates) == grids.GridCoordinatesUniform
        assert coordinates.in_list == [[(0.0, 1.0), (1.0, 1.0)]]
        assert coordinates.pixel_scales == (1.0, 1.0)
        assert coordinates.shape_2d == (3, 3)

    def test__coordinates_from_grid_1d(self):

        coordinates = aa.GridCoordinatesUniform(
            coordinates=[[(1.0, 1.0), (2.0, 2.0)]],
            pixel_scales=(1.0, 1.0),
            shape_2d=(3, 3),
        )

        coordinates_from_1d = coordinates.coordinates_from_grid_1d(
            grid_1d=np.array([[1.0, 1.0], [2.0, 2.0]])
        )

        assert type(coordinates_from_1d) == grids.GridCoordinatesUniform
        assert coordinates_from_1d.in_list == [[(1.0, 1.0), (2.0, 2.0)]]
        assert coordinates.pixel_scales == (1.0, 1.0)
        assert coordinates.shape_2d == (3, 3)

        coordinates = aa.GridCoordinatesUniform(
            coordinates=[[(1.0, 1.0), (2.0, 2.0)], [(3.0, 3.0)]],
            pixel_scales=(1.0, 3.0),
            shape_2d=(5, 5),
        )

        coordinates_from_1d = coordinates.coordinates_from_grid_1d(
            grid_1d=np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        )

        assert type(coordinates_from_1d) == grids.GridCoordinatesUniform
        assert coordinates_from_1d.in_list == [[(1.0, 1.0), (2.0, 2.0)], [(3.0, 3.0)]]
        assert coordinates.pixel_scales == (1.0, 3.0)
        assert coordinates.shape_2d == (5, 5)
