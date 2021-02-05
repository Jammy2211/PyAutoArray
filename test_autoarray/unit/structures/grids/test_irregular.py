import os
from os import path
import shutil
import numpy as np
import pytest

import autoarray as aa
from autoarray.structures import grids

test_grid_dir = path.join(
    "{}".format(path.dirname(path.realpath(__file__))), "files", "grid"
)


class TestAbstractGrid2DIrregular:
    def test__structure_from_result__maps_numpy_array_to__auto_array_or_grid(self):

        grid = aa.Grid2DIrregular(grid=[(1.0, -1.0), (1.0, 1.0)])

        result = grid.structure_from_result(result=np.array([1.0, 2.0]))

        assert isinstance(result, aa.ValuesIrregularGrouped)
        assert result.in_list == [1.0, 2.0]

        result = grid.structure_from_result(result=np.array([[1.0, 1.0], [2.0, 2.0]]))

        assert isinstance(result, aa.Grid2DIrregular)
        assert result.in_list == [(1.0, 1.0), (2.0, 2.0)]

        grid = aa.Grid2DIrregularGrouped(grid=[(1.0, -1.0), (1.0, 1.0)])

        result = grid.structure_from_result(result=np.array([1.0, 2.0]))

        assert isinstance(result, aa.ValuesIrregularGrouped)
        assert result.in_grouped_list == [[1.0, 2.0]]

        result = grid.structure_from_result(result=np.array([[1.0, 1.0], [2.0, 2.0]]))

        assert isinstance(result, aa.Grid2DIrregularGrouped)
        assert result.in_grouped_list == [[(1.0, 1.0), (2.0, 2.0)]]

    def test__structure_list_from_result_list__maps_list_to_auto_arrays_or_grids(self):

        grid = aa.Grid2DIrregular(grid=[(1.0, -1.0), (1.0, 1.0)])

        result = grid.structure_from_result(result=[np.array([1.0, 2.0])])

        assert isinstance(result[0], aa.ValuesIrregularGrouped)
        assert result[0].in_list == [1.0, 2.0]

        result = grid.structure_from_result(result=[np.array([[1.0, 1.0], [2.0, 2.0]])])

        assert isinstance(result[0], aa.Grid2DIrregular)
        assert result[0].in_list == [(1.0, 1.0), (2.0, 2.0)]

        grid = aa.Grid2DIrregularGrouped(grid=[(1.0, -1.0), (1.0, 1.0)])

        result = grid.structure_from_result(result=[np.array([1.0, 2.0])])

        assert isinstance(result[0], aa.ValuesIrregularGrouped)
        assert result[0].in_grouped_list == [[1.0, 2.0]]

        result = grid.structure_from_result(result=[np.array([[1.0, 1.0], [2.0, 2.0]])])

        assert isinstance(result[0], aa.Grid2DIrregularGrouped)
        assert result[0].in_grouped_list == [[(1.0, 1.0), (2.0, 2.0)]]


class TestGrid2DIrregular:
    def test__input_as_list_or_list_of_other_types__all_convert_correctly(self):

        # Input tuples

        grid = aa.Grid2DIrregular(grid=[(1.0, -1.0), (1.0, 1.0)])

        assert type(grid) == grids.Grid2DIrregular
        assert (grid == np.array([[[1.0, -1.0], [1.0, 1.0]]])).all()
        assert grid.in_list == [(1.0, -1.0), (1.0, 1.0)]

        # Input np array

        grid = aa.Grid2DIrregular(grid=[np.array([1.0, -1.0]), np.array([1.0, 1.0])])

        assert type(grid) == grids.Grid2DIrregular
        assert (grid == np.array([[[1.0, -1.0], [1.0, 1.0]]])).all()
        assert grid.in_list == [(1.0, -1.0), (1.0, 1.0)]

        # Input list

        grid = aa.Grid2DIrregular(grid=[[1.0, -1.0], [1.0, 1.0]])

        assert type(grid) == grids.Grid2DIrregular
        assert (grid == np.array([[[1.0, -1.0], [1.0, 1.0]]])).all()
        assert grid.in_list == [(1.0, -1.0), (1.0, 1.0)]

    def test__values_from_array_slim(self):

        grid = aa.Grid2DIrregular(grid=[(1.0, 1.0), (2.0, 2.0)])

        values_from_1d = grid.values_from_array_slim(array_slim=np.array([1.0, 2.0]))

        assert isinstance(values_from_1d, aa.ValuesIrregularGrouped)
        assert values_from_1d.in_list == [1.0, 2.0]

        grid = aa.Grid2DIrregular(grid=[(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)])

        values_from_1d = grid.values_from_array_slim(
            array_slim=np.array([1.0, 2.0, 3.0])
        )

        assert isinstance(values_from_1d, aa.ValuesIrregularGrouped)
        assert values_from_1d.in_list == [1.0, 2.0, 3.0]

    def test__grid_from_grid_slim(self):

        grid = aa.Grid2DIrregular(grid=[(1.0, 1.0), (2.0, 2.0)])

        grid_from_1d = grid.grid_from_grid_slim(
            grid_slim=np.array([[1.0, 1.0], [2.0, 2.0]])
        )

        assert type(grid_from_1d) == grids.Grid2DIrregular
        assert grid_from_1d.in_list == [(1.0, 1.0), (2.0, 2.0)]

        grid = aa.Grid2DIrregular(grid=[(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)])

        grid_from_1d = grid.grid_from_grid_slim(
            grid_slim=np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        )

        assert type(grid_from_1d) == grids.Grid2DIrregular
        assert grid_from_1d.in_list == [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]

    def test__grid_from_deflection_grid(self):

        grid = aa.Grid2DIrregular(grid=[(1.0, 1.0), (2.0, 2.0)])

        grid = grid.grid_from_deflection_grid(
            deflection_grid=np.array([[1.0, 0.0], [1.0, 1.0]])
        )

        assert type(grid) == grids.Grid2DIrregular
        assert grid.in_list == [(0.0, 1.0), (1.0, 1.0)]

    def test__furthest_distances_from_other_coordinates(self):

        grid = aa.Grid2DIrregular(grid=[(0.0, 0.0), (0.0, 1.0)])

        assert grid.furthest_distances_from_other_coordinates.in_grouped_list == [
            [1.0, 1.0]
        ]

        grid = aa.Grid2DIrregular(grid=[(2.0, 4.0), (3.0, 6.0)])

        assert grid.furthest_distances_from_other_coordinates.in_grouped_list == [
            [np.sqrt(5), np.sqrt(5)]
        ]

        grid = aa.Grid2DIrregular(grid=[(0.0, 0.0), (0.0, 1.0), (0.0, 3.0)])

        assert grid.furthest_distances_from_other_coordinates.in_grouped_list == [
            [3.0, 2.0, 3.0]
        ]

    def test__grid_of_closest_from_grid(self):

        grid = aa.Grid2DIrregular(grid=[(0.0, 0.0), (0.0, 1.0)])

        grid_of_closest = grid.grid_of_closest_from_grid_pair(
            grid_pair=np.array([[0.0, 0.1]])
        )

        assert (grid_of_closest == np.array([[0.0, 0.0]])).all()

        grid_of_closest = grid.grid_of_closest_from_grid_pair(
            grid_pair=np.array([[0.0, 0.1], [0.0, 0.2], [0.0, 0.3]])
        )

        assert (grid_of_closest == np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])).all()

        grid_of_closest = grid.grid_of_closest_from_grid_pair(
            grid_pair=np.array([[0.0, 0.1], [0.0, 0.2], [0.0, 0.9], [0.0, -0.1]])
        )

        assert (
            grid_of_closest
            == np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
        ).all()


class TestGrid2DIrregularGrouped:
    def test__indexes_give_entries_where_list_begin_and_end(self):

        grid = aa.Grid2DIrregularGrouped(grid=[[(0.0, 0.0)]])

        assert grid.lower_indexes == [0]
        assert grid.upper_indexes == [1]

        grid = aa.Grid2DIrregularGrouped(grid=[[(0.0, 0.0), (0.0, 0.0)]])

        assert grid.lower_indexes == [0]
        assert grid.upper_indexes == [2]

        grid = aa.Grid2DIrregularGrouped(grid=[[(0.0, 0.0), (0.0, 0.0)], [(0.0, 0.0)]])

        assert grid.lower_indexes == [0, 2]
        assert grid.upper_indexes == [2, 3]

        grid = aa.Grid2DIrregularGrouped(
            grid=[
                [(0.0, 0.0), (0.0, 0.0)],
                [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
                [(0.0, 0.0), (0.0, 0.0)],
                [(0.0, 0.0)],
            ]
        )

        assert grid.lower_indexes == [0, 2, 5, 7]
        assert grid.upper_indexes == [2, 5, 7, 8]

    def test__input_as_list_or_list_of_other_types__all_convert_correctly(self):

        # Input tuples

        grid = aa.Grid2DIrregularGrouped(grid=[(1.0, -1.0), (1.0, 1.0)])

        assert type(grid) == grids.Grid2DIrregularGrouped
        assert (grid == np.array([[[1.0, -1.0], [1.0, 1.0]]])).all()
        assert grid.in_grouped_list == [[(1.0, -1.0), (1.0, 1.0)]]
        assert grid.in_list == [(1.0, -1.0), (1.0, 1.0)]

        grid = aa.Grid2DIrregularGrouped(grid=[[(1.0, -1.0), (1.0, 1.0)]])

        assert type(grid) == grids.Grid2DIrregularGrouped
        assert (grid == np.array([[[1.0, -1.0], [1.0, 1.0]]])).all()
        assert grid.in_grouped_list == [[(1.0, -1.0), (1.0, 1.0)]]
        assert grid.in_list == [(1.0, -1.0), (1.0, 1.0)]

        # Input np array

        grid = aa.Grid2DIrregularGrouped(
            grid=[np.array([1.0, -1.0]), np.array([1.0, 1.0])]
        )

        assert type(grid) == grids.Grid2DIrregularGrouped
        assert (grid == np.array([[[1.0, -1.0], [1.0, 1.0]]])).all()
        assert grid.in_grouped_list == [[(1.0, -1.0), (1.0, 1.0)]]
        assert grid.in_list == [(1.0, -1.0), (1.0, 1.0)]

        grid = aa.Grid2DIrregularGrouped(
            grid=[[np.array([1.0, -1.0]), np.array([1.0, 1.0])]]
        )

        assert type(grid) == grids.Grid2DIrregularGrouped
        assert (grid == np.array([[[1.0, -1.0], [1.0, 1.0]]])).all()
        assert grid.in_grouped_list == [[(1.0, -1.0), (1.0, 1.0)]]
        assert grid.in_list == [(1.0, -1.0), (1.0, 1.0)]

        grid = aa.Grid2DIrregularGrouped(grid=[np.array([[1.0, -1.0], [1.0, 1.0]])])

        assert type(grid) == grids.Grid2DIrregularGrouped
        assert (grid == np.array([[[1.0, -1.0], [1.0, 1.0]]])).all()
        assert grid.in_grouped_list == [[(1.0, -1.0), (1.0, 1.0)]]
        assert grid.in_list == [(1.0, -1.0), (1.0, 1.0)]

        # Input list

        grid = aa.Grid2DIrregularGrouped(grid=[[1.0, -1.0], [1.0, 1.0]])

        assert type(grid) == grids.Grid2DIrregularGrouped
        assert (grid == np.array([[[1.0, -1.0], [1.0, 1.0]]])).all()
        assert grid.in_grouped_list == [[(1.0, -1.0), (1.0, 1.0)]]
        assert grid.in_list == [(1.0, -1.0), (1.0, 1.0)]

        grid = aa.Grid2DIrregularGrouped(grid=[[[1.0, -1.0]], [[1.0, 1.0]]])

        assert type(grid) == grids.Grid2DIrregularGrouped
        assert (grid == np.array([[[1.0, -1.0], [1.0, 1.0]]])).all()
        assert grid.in_grouped_list == [[(1.0, -1.0)], [(1.0, 1.0)]]
        assert grid.in_list == [(1.0, -1.0), (1.0, 1.0)]

        grid = aa.Grid2DIrregularGrouped(grid=[[[1.0, -1.0], [1.0, 1.0]]])

        assert type(grid) == grids.Grid2DIrregularGrouped
        assert (grid == np.array([[[1.0, -1.0], [1.0, 1.0]]])).all()
        assert grid.in_grouped_list == [[(1.0, -1.0), (1.0, 1.0)]]
        assert grid.in_list == [(1.0, -1.0), (1.0, 1.0)]

    def test__input_as_dict__retains_dict_and_convert_correctly(self):

        # Input tuples

        grid = aa.Grid2DIrregularGrouped(
            grid=dict(source_0=[(1.0, -1.0), (1.0, 1.0)], source_1=[(2.0, 2.0)])
        )

        assert type(grid) == grids.Grid2DIrregularGrouped
        assert (grid == np.array([[[1.0, -1.0], [1.0, 1.0], [2.0, 2.0]]])).all()
        assert grid.in_grouped_list == [[(1.0, -1.0), (1.0, 1.0)], [(2.0, 2.0)]]
        assert grid.in_list == [(1.0, -1.0), (1.0, 1.0), (2.0, 2.0)]
        assert grid.as_dict["source_0"] == [(1.0, -1.0), (1.0, 1.0)]
        assert grid.as_dict["source_1"] == [(2.0, 2.0)]

    def test__from_yx_1d(self):
        grid = aa.Grid2DIrregularGrouped.from_yx_1d(y=[1.0, 1.0], x=[-1.0, 1.0])

        assert grid.in_grouped_list == [[(1.0, -1.0), (1.0, 1.0)]]

        grid = aa.Grid2DIrregularGrouped.from_yx_1d(
            y=[1.0, 1.0, 2.0, 4.0], x=[-1.0, 1.0, 3.0, 5.0]
        )

        assert grid.in_grouped_list == [
            [(1.0, -1.0), (1.0, 1.0), (2.0, 3.0), (4.0, 5.0)]
        ]

    def test__values_from_array_slim(self):

        grid = aa.Grid2DIrregularGrouped(grid=[[(1.0, 1.0), (2.0, 2.0)]])

        values_from_1d = grid.values_from_array_slim(array_slim=np.array([1.0, 2.0]))

        assert values_from_1d.in_grouped_list == [[1.0, 2.0]]

        grid = aa.Grid2DIrregularGrouped(grid=[[(1.0, 1.0), (2.0, 2.0)], [(3.0, 3.0)]])

        values_from_1d = grid.values_from_array_slim(
            array_slim=np.array([1.0, 2.0, 3.0])
        )

        assert values_from_1d.in_grouped_list == [[1.0, 2.0], [3.0]]

    def test__values_from_value(self):

        grid = aa.Grid2DIrregularGrouped(grid=[[(1.0, 1.0), (2.0, 2.0)]])

        values_from_value = grid.values_from_value(value=1.0)

        assert values_from_value.in_grouped_list == [[1.0, 1.0]]

        grid = aa.Grid2DIrregularGrouped(grid=[[(1.0, 1.0), (2.0, 2.0)], [(3.0, 3.0)]])

        values_from_value = grid.values_from_value(value=2.0)

        assert values_from_value.in_grouped_list == [[2.0, 2.0], [2.0]]

    def test__grid_from_grid_slim(self):

        grid = aa.Grid2DIrregularGrouped(grid=[[(1.0, 1.0), (2.0, 2.0)]])

        grid_from_1d = grid.grid_from_grid_slim(
            grid_slim=np.array([[1.0, 1.0], [2.0, 2.0]])
        )

        assert type(grid_from_1d) == grids.Grid2DIrregularGrouped
        assert grid_from_1d.in_grouped_list == [[(1.0, 1.0), (2.0, 2.0)]]

        grid = aa.Grid2DIrregularGrouped(grid=[[(1.0, 1.0), (2.0, 2.0)], [(3.0, 3.0)]])

        grid_from_1d = grid.grid_from_grid_slim(
            grid_slim=np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        )

        assert type(grid_from_1d) == grids.Grid2DIrregularGrouped
        assert grid_from_1d.in_grouped_list == [[(1.0, 1.0), (2.0, 2.0)], [(3.0, 3.0)]]

        grid_from_1d = grid.grid_from_grid_slim(
            grid_slim=np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        )

        assert type(grid_from_1d) == grids.Grid2DIrregularGrouped
        assert grid_from_1d.in_grouped_list == [[(1.0, 1.0), (2.0, 2.0)], [(3.0, 3.0)]]

    def test__grid_from_deflection_grid(self):
        grid = aa.Grid2DIrregularGrouped(grid=[[(1.0, 1.0), (2.0, 2.0)]])

        grid = grid.grid_from_deflection_grid(
            deflection_grid=np.array([[1.0, 0.0], [1.0, 1.0]])
        )

        assert grid.in_grouped_list == [[(0.0, 1.0), (1.0, 1.0)]]

    def test__furthest_distances_from_other_coordinates(self):

        grid = aa.Grid2DIrregularGrouped(grid=[[(0.0, 0.0), (0.0, 1.0)]])

        assert grid.furthest_distances_from_other_coordinates.in_grouped_list == [
            [1.0, 1.0]
        ]

        grid = aa.Grid2DIrregularGrouped(grid=[[(2.0, 4.0), (3.0, 6.0)]])

        assert grid.furthest_distances_from_other_coordinates.in_grouped_list == [
            [np.sqrt(5), np.sqrt(5)]
        ]

        grid = aa.Grid2DIrregularGrouped(grid=[[(0.0, 0.0), (0.0, 1.0), (0.0, 3.0)]])

        assert grid.furthest_distances_from_other_coordinates.in_grouped_list == [
            [3.0, 2.0, 3.0]
        ]

        grid = aa.Grid2DIrregularGrouped(
            grid=[[(0.0, 0.0), (0.0, 1.0), (0.0, 3.0)], [(0.0, 0.0)]]
        )

        assert grid.furthest_distances_from_other_coordinates.in_grouped_list == [
            [3.0, 2.0, 3.0],
            [0.0],
        ]

    def test__grid_of_closest_from_grid(self):

        grid = aa.Grid2DIrregularGrouped(grid=[[(0.0, 0.0), (0.0, 1.0)]])
        grid_pair = aa.Grid2DIrregularGrouped(grid=[[(0.0, 0.1)]])

        grid_of_closest = grid.grid_of_closest_from_grid_pair(grid_pair=grid_pair)

        assert grid_of_closest.in_grouped_list == [[(0.0, 0.0)]]

        grid_pair = aa.Grid2DIrregularGrouped(
            grid=[[(0.0, 0.1), (0.0, 0.2), (0.0, 0.3)]]
        )

        grid_of_closest = grid.grid_of_closest_from_grid_pair(grid_pair=grid_pair)

        assert grid_of_closest.in_grouped_list == [[(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)]]

        grid_pair = aa.Grid2DIrregularGrouped(
            grid=[[(0.0, 0.1), (0.0, 0.2), (0.0, 0.9), (0.0, -0.1)]]
        )

        grid_of_closest = grid.grid_of_closest_from_grid_pair(grid_pair=grid_pair)

        assert grid_of_closest.in_grouped_list == [
            [(0.0, 0.0), (0.0, 0.0), (0.0, 1.0), (0.0, 0.0)]
        ]

        grid = aa.Grid2DIrregularGrouped(
            grid=[[(0.0, 0.0), (0.0, 1.0)], [(0.0, 0.0), (0.0, 1.0)]]
        )
        grid_pair = aa.Grid2DIrregularGrouped(
            grid=[
                [(0.0, 0.1), (0.0, 0.2), (0.0, 0.3)],
                [(0.0, 0.1), (0.0, 0.2), (0.0, 0.9), (0.0, -0.1)],
            ]
        )

        grid_of_closest = grid.grid_of_closest_from_grid_pair(grid_pair=grid_pair)

        assert grid_of_closest.in_grouped_list == [
            [(0.0, 0.0), (0.0, 0.0), (0.0, 0.0)],
            [(0.0, 0.0), (0.0, 0.0), (0.0, 1.0), (0.0, 0.0)],
        ]

    def test__with_mask__converts_to_and_from_pixels(self):

        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(2, 2)), pixel_scales=(2.0, 2.0)
        )

        grid = aa.Grid2DIrregularGrouped(grid=[[(1.0, -1.0), (1.0, 1.0)]])

        assert grid.in_grouped_list == [[(1.0, -1.0), (1.0, 1.0)]]

        grid = aa.Grid2DIrregularGrouped.from_pixels_and_mask(
            pixels=[[(0, 0), (0, 1)]], mask=mask
        )

        assert grid.in_grouped_list == [[(1.0, -1.0), (1.0, 1.0)]]

    def test__load_grid__retains_list_structure(self):

        file_path = path.join(test_grid_dir, "output_test", "grid_test.dat")

        grid = aa.Grid2DIrregularGrouped.from_file(file_path=file_path)

        assert grid.in_grouped_list == [
            [(4.0, 4.0), (5.0, 5.0)],
            [(6.0, 6.0), (7.0, 7.0), (8.0, 8.0)],
        ]

    def test__output_grid_to_file(self):
        grid = aa.Grid2DIrregularGrouped(
            [[(4.0, 4.0), (5.0, 5.0)], [(6.0, 6.0), (7.0, 7.0), (8.0, 8.0)]]
        )

        output_grid_dir = path.join(
            "{}".format(os.path.dirname(os.path.realpath(__file__))),
            "files",
            "grid",
            "output_test",
        )

        file_path = path.join(output_grid_dir, "grid_test.dat")

        if os.path.exists(output_grid_dir):
            shutil.rmtree(output_grid_dir)

        os.makedirs(output_grid_dir)

        grid.output_to_file(file_path=file_path)

        grid = aa.Grid2DIrregularGrouped.from_file(file_path=file_path)

        assert grid.in_grouped_list == [
            [(4.0, 4.0), (5.0, 5.0)],
            [(6.0, 6.0), (7.0, 7.0), (8.0, 8.0)],
        ]

        with pytest.raises(FileExistsError):
            grid.output_to_file(file_path=file_path)

        grid.output_to_file(file_path=file_path, overwrite=True)


class TestGrid2DIrregularGroupedUniform:
    def test__from_sparse_uniform_upscale(self):

        grid_sparse_uniform = aa.Grid2DIrregularGroupedUniform(
            grid=[[(1.0, 1.0), (1.0, 3.0)]], pixel_scales=2.0
        )

        grid_upscale = grids.Grid2DIrregularGroupedUniform.from_grid_sparse_uniform_upscale(
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

        grid_sparse_uniform = aa.Grid2DIrregularGroupedUniform(
            grid=[[(1.0, 1.0), (1.0, 3.0), (1.0, 5.0), (3.0, 3.0)]], pixel_scales=2.0
        )

        grid_upscale = grids.Grid2DIrregularGroupedUniform.from_grid_sparse_uniform_upscale(
            grid_sparse_uniform=grid_sparse_uniform, upscale_factor=4, pixel_scales=2.0
        )

        grid_upscale_util = aa.util.grid.grid_2d_slim_upscaled_from(
            grid_slim=grid_sparse_uniform, upscale_factor=4, pixel_scales=(2.0, 2.0)
        )

        assert (grid_upscale == grid_upscale_util).all()

    def test__grid_from_deflection_grid(self):
        grid = aa.Grid2DIrregularGroupedUniform(
            grid=[[(1.0, 1.0), (2.0, 2.0)]],
            pixel_scales=(1.0, 1.0),
            shape_native=(3, 3),
        )

        grid = grid.grid_from_deflection_grid(
            deflection_grid=np.array([[1.0, 0.0], [1.0, 1.0]])
        )

        assert type(grid) == grids.Grid2DIrregularGroupedUniform
        assert grid.in_grouped_list == [[(0.0, 1.0), (1.0, 1.0)]]
        assert grid.pixel_scales == (1.0, 1.0)
        assert grid.shape_native == (3, 3)

    def test__grid_from_grid_slim(self):

        grid = aa.Grid2DIrregularGroupedUniform(
            grid=[[(1.0, 1.0), (2.0, 2.0)]],
            pixel_scales=(1.0, 1.0),
            shape_native=(3, 3),
        )

        grid_from_1d = grid.grid_from_grid_slim(
            grid_slim=np.array([[1.0, 1.0], [2.0, 2.0]])
        )

        assert type(grid_from_1d) == grids.Grid2DIrregularGroupedUniform
        assert grid_from_1d.in_grouped_list == [[(1.0, 1.0), (2.0, 2.0)]]
        assert grid.pixel_scales == (1.0, 1.0)
        assert grid.shape_native == (3, 3)

        grid = aa.Grid2DIrregularGroupedUniform(
            grid=[[(1.0, 1.0), (2.0, 2.0)], [(3.0, 3.0)]],
            pixel_scales=(1.0, 3.0),
            shape_native=(5, 5),
        )

        grid_from_1d = grid.grid_from_grid_slim(
            grid_slim=np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        )

        assert type(grid_from_1d) == grids.Grid2DIrregularGroupedUniform
        assert grid_from_1d.in_grouped_list == [[(1.0, 1.0), (2.0, 2.0)], [(3.0, 3.0)]]
        assert grid.pixel_scales == (1.0, 3.0)
        assert grid.shape_native == (5, 5)
