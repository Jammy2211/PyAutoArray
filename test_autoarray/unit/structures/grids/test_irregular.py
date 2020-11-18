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


class TestAbstractGridIrregular:
    def test__structure_from_result__maps_numpy_array_to__auto_array_or_grid(self):

        grid = aa.GridIrregular(grid=[(1.0, -1.0), (1.0, 1.0)])

        result = grid.structure_from_result(result=np.array([1.0, 2.0]))

        assert isinstance(result, aa.Values)
        assert result.in_1d_list == [1.0, 2.0]

        result = grid.structure_from_result(result=np.array([[1.0, 1.0], [2.0, 2.0]]))

        assert isinstance(result, aa.GridIrregular)
        assert result.in_1d_list == [(1.0, 1.0), (2.0, 2.0)]

        grid = aa.GridIrregularGrouped(grid=[(1.0, -1.0), (1.0, 1.0)])

        result = grid.structure_from_result(result=np.array([1.0, 2.0]))

        assert isinstance(result, aa.Values)
        assert result.in_grouped_list == [[1.0, 2.0]]

        result = grid.structure_from_result(result=np.array([[1.0, 1.0], [2.0, 2.0]]))

        assert isinstance(result, aa.GridIrregularGrouped)
        assert result.in_grouped_list == [[(1.0, 1.0), (2.0, 2.0)]]

    def test__structure_list_from_result_list__maps_list_to_auto_arrays_or_grids(self):

        grid = aa.GridIrregular(grid=[(1.0, -1.0), (1.0, 1.0)])

        result = grid.structure_from_result(result=[np.array([1.0, 2.0])])

        assert isinstance(result[0], aa.Values)
        assert result[0].in_1d_list == [1.0, 2.0]

        result = grid.structure_from_result(result=[np.array([[1.0, 1.0], [2.0, 2.0]])])

        assert isinstance(result[0], aa.GridIrregular)
        assert result[0].in_1d_list == [(1.0, 1.0), (2.0, 2.0)]

        grid = aa.GridIrregularGrouped(grid=[(1.0, -1.0), (1.0, 1.0)])

        result = grid.structure_from_result(result=[np.array([1.0, 2.0])])

        assert isinstance(result[0], aa.Values)
        assert result[0].in_grouped_list == [[1.0, 2.0]]

        result = grid.structure_from_result(result=[np.array([[1.0, 1.0], [2.0, 2.0]])])

        assert isinstance(result[0], aa.GridIrregularGrouped)
        assert result[0].in_grouped_list == [[(1.0, 1.0), (2.0, 2.0)]]


class TestGridIrregular:
    def test__input_as_list_or_list_of_other_types__all_convert_correctly(self):

        # Input tuples

        grid = aa.GridIrregular(grid=[(1.0, -1.0), (1.0, 1.0)])

        assert type(grid) == grids.GridIrregular
        assert (grid == np.array([[[1.0, -1.0], [1.0, 1.0]]])).all()
        assert grid.in_1d_list == [(1.0, -1.0), (1.0, 1.0)]

        # Input np array

        grid = aa.GridIrregular(grid=[np.array([1.0, -1.0]), np.array([1.0, 1.0])])

        assert type(grid) == grids.GridIrregular
        assert (grid == np.array([[[1.0, -1.0], [1.0, 1.0]]])).all()
        assert grid.in_1d_list == [(1.0, -1.0), (1.0, 1.0)]

        # Input list

        grid = aa.GridIrregular(grid=[[1.0, -1.0], [1.0, 1.0]])

        assert type(grid) == grids.GridIrregular
        assert (grid == np.array([[[1.0, -1.0], [1.0, 1.0]]])).all()
        assert grid.in_1d_list == [(1.0, -1.0), (1.0, 1.0)]

    def test__values_from_arr_1d(self):

        grid = aa.GridIrregular(grid=[(1.0, 1.0), (2.0, 2.0)])

        values_from_1d = grid.values_from_arr_1d(arr_1d=np.array([1.0, 2.0]))

        assert isinstance(values_from_1d, aa.Values)
        assert values_from_1d.in_1d_list == [1.0, 2.0]

        grid = aa.GridIrregular(grid=[(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)])

        values_from_1d = grid.values_from_arr_1d(arr_1d=np.array([1.0, 2.0, 3.0]))

        assert isinstance(values_from_1d, aa.Values)
        assert values_from_1d.in_1d_list == [1.0, 2.0, 3.0]

    def test__grid_from_grid_1d(self):

        grid = aa.GridIrregular(grid=[(1.0, 1.0), (2.0, 2.0)])

        grid_from_1d = grid.grid_from_grid_1d(
            grid_1d=np.array([[1.0, 1.0], [2.0, 2.0]])
        )

        assert type(grid_from_1d) == grids.GridIrregular
        assert grid_from_1d.in_1d_list == [(1.0, 1.0), (2.0, 2.0)]

        grid = aa.GridIrregular(grid=[(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)])

        grid_from_1d = grid.grid_from_grid_1d(
            grid_1d=np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        )

        assert type(grid_from_1d) == grids.GridIrregular
        assert grid_from_1d.in_1d_list == [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]

    def test__grid_from_deflection_grid(self):

        grid = aa.GridIrregular(grid=[(1.0, 1.0), (2.0, 2.0)])

        grid = grid.grid_from_deflection_grid(
            deflection_grid=np.array([[1.0, 0.0], [1.0, 1.0]])
        )

        assert type(grid) == grids.GridIrregular
        assert grid.in_1d_list == [(0.0, 1.0), (1.0, 1.0)]


class TestGridIrregularGrouped:
    def test__indexes_give_entries_where_list_begin_and_end(self):

        grid = aa.GridIrregularGrouped(grid=[[(0.0, 0.0)]])

        assert grid.lower_indexes == [0]
        assert grid.upper_indexes == [1]

        grid = aa.GridIrregularGrouped(grid=[[(0.0, 0.0), (0.0, 0.0)]])

        assert grid.lower_indexes == [0]
        assert grid.upper_indexes == [2]

        grid = aa.GridIrregularGrouped(grid=[[(0.0, 0.0), (0.0, 0.0)], [(0.0, 0.0)]])

        assert grid.lower_indexes == [0, 2]
        assert grid.upper_indexes == [2, 3]

        grid = aa.GridIrregularGrouped(
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

        grid = aa.GridIrregularGrouped(grid=[(1.0, -1.0), (1.0, 1.0)])

        assert type(grid) == grids.GridIrregularGrouped
        assert (grid == np.array([[[1.0, -1.0], [1.0, 1.0]]])).all()
        assert grid.in_grouped_list == [[(1.0, -1.0), (1.0, 1.0)]]
        assert grid.in_1d_list == [(1.0, -1.0), (1.0, 1.0)]

        grid = aa.GridIrregularGrouped(grid=[[(1.0, -1.0), (1.0, 1.0)]])

        assert type(grid) == grids.GridIrregularGrouped
        assert (grid == np.array([[[1.0, -1.0], [1.0, 1.0]]])).all()
        assert grid.in_grouped_list == [[(1.0, -1.0), (1.0, 1.0)]]
        assert grid.in_1d_list == [(1.0, -1.0), (1.0, 1.0)]

        # Input np array

        grid = aa.GridIrregularGrouped(
            grid=[np.array([1.0, -1.0]), np.array([1.0, 1.0])]
        )

        assert type(grid) == grids.GridIrregularGrouped
        assert (grid == np.array([[[1.0, -1.0], [1.0, 1.0]]])).all()
        assert grid.in_grouped_list == [[(1.0, -1.0), (1.0, 1.0)]]
        assert grid.in_1d_list == [(1.0, -1.0), (1.0, 1.0)]

        grid = aa.GridIrregularGrouped(
            grid=[[np.array([1.0, -1.0]), np.array([1.0, 1.0])]]
        )

        assert type(grid) == grids.GridIrregularGrouped
        assert (grid == np.array([[[1.0, -1.0], [1.0, 1.0]]])).all()
        assert grid.in_grouped_list == [[(1.0, -1.0), (1.0, 1.0)]]
        assert grid.in_1d_list == [(1.0, -1.0), (1.0, 1.0)]

        grid = aa.GridIrregularGrouped(grid=[np.array([[1.0, -1.0], [1.0, 1.0]])])

        assert type(grid) == grids.GridIrregularGrouped
        assert (grid == np.array([[[1.0, -1.0], [1.0, 1.0]]])).all()
        assert grid.in_grouped_list == [[(1.0, -1.0), (1.0, 1.0)]]
        assert grid.in_1d_list == [(1.0, -1.0), (1.0, 1.0)]

        # Input list

        grid = aa.GridIrregularGrouped(grid=[[1.0, -1.0], [1.0, 1.0]])

        assert type(grid) == grids.GridIrregularGrouped
        assert (grid == np.array([[[1.0, -1.0], [1.0, 1.0]]])).all()
        assert grid.in_grouped_list == [[(1.0, -1.0), (1.0, 1.0)]]
        assert grid.in_1d_list == [(1.0, -1.0), (1.0, 1.0)]

        grid = aa.GridIrregularGrouped(grid=[[[1.0, -1.0]], [[1.0, 1.0]]])

        assert type(grid) == grids.GridIrregularGrouped
        assert (grid == np.array([[[1.0, -1.0], [1.0, 1.0]]])).all()
        assert grid.in_grouped_list == [[(1.0, -1.0)], [(1.0, 1.0)]]
        assert grid.in_1d_list == [(1.0, -1.0), (1.0, 1.0)]

        grid = aa.GridIrregularGrouped(grid=[[[1.0, -1.0], [1.0, 1.0]]])

        assert type(grid) == grids.GridIrregularGrouped
        assert (grid == np.array([[[1.0, -1.0], [1.0, 1.0]]])).all()
        assert grid.in_grouped_list == [[(1.0, -1.0), (1.0, 1.0)]]
        assert grid.in_1d_list == [(1.0, -1.0), (1.0, 1.0)]

    def test__input_as_dict__retains_dict_and_convert_correctly(self):

        # Input tuples

        grid = aa.GridIrregularGrouped(
            grid=dict(source_0=[(1.0, -1.0), (1.0, 1.0)], source_1=[(2.0, 2.0)])
        )

        assert type(grid) == grids.GridIrregularGrouped
        assert (grid == np.array([[[1.0, -1.0], [1.0, 1.0], [2.0, 2.0]]])).all()
        assert grid.in_grouped_list == [[(1.0, -1.0), (1.0, 1.0)], [(2.0, 2.0)]]
        assert grid.in_1d_list == [(1.0, -1.0), (1.0, 1.0), (2.0, 2.0)]
        assert grid.as_dict["source_0"] == [(1.0, -1.0), (1.0, 1.0)]
        assert grid.as_dict["source_1"] == [(2.0, 2.0)]

    def test__from_yx_1d(self):
        grid = aa.GridIrregularGrouped.from_yx_1d(y=[1.0, 1.0], x=[-1.0, 1.0])

        assert grid.in_grouped_list == [[(1.0, -1.0), (1.0, 1.0)]]

        grid = aa.GridIrregularGrouped.from_yx_1d(
            y=[1.0, 1.0, 2.0, 4.0], x=[-1.0, 1.0, 3.0, 5.0]
        )

        assert grid.in_grouped_list == [
            [(1.0, -1.0), (1.0, 1.0), (2.0, 3.0), (4.0, 5.0)]
        ]

    def test__values_from_arr_1d(self):

        grid = aa.GridIrregularGrouped(grid=[[(1.0, 1.0), (2.0, 2.0)]])

        values_from_1d = grid.values_from_arr_1d(arr_1d=np.array([1.0, 2.0]))

        assert values_from_1d.in_grouped_list == [[1.0, 2.0]]

        grid = aa.GridIrregularGrouped(grid=[[(1.0, 1.0), (2.0, 2.0)], [(3.0, 3.0)]])

        values_from_1d = grid.values_from_arr_1d(arr_1d=np.array([1.0, 2.0, 3.0]))

        assert values_from_1d.in_grouped_list == [[1.0, 2.0], [3.0]]

    def test__grid_from_grid_1d(self):

        grid = aa.GridIrregularGrouped(grid=[[(1.0, 1.0), (2.0, 2.0)]])

        grid_from_1d = grid.grid_from_grid_1d(
            grid_1d=np.array([[1.0, 1.0], [2.0, 2.0]])
        )

        assert type(grid_from_1d) == grids.GridIrregularGrouped
        assert grid_from_1d.in_grouped_list == [[(1.0, 1.0), (2.0, 2.0)]]

        grid = aa.GridIrregularGrouped(grid=[[(1.0, 1.0), (2.0, 2.0)], [(3.0, 3.0)]])

        grid_from_1d = grid.grid_from_grid_1d(
            grid_1d=np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        )

        assert type(grid_from_1d) == grids.GridIrregularGrouped
        assert grid_from_1d.in_grouped_list == [[(1.0, 1.0), (2.0, 2.0)], [(3.0, 3.0)]]

        grid_from_1d = grid.grid_from_grid_1d(
            grid_1d=np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        )

        assert type(grid_from_1d) == grids.GridIrregularGrouped
        assert grid_from_1d.in_grouped_list == [[(1.0, 1.0), (2.0, 2.0)], [(3.0, 3.0)]]

    def test__grid_from_deflection_grid(self):
        grid = aa.GridIrregularGrouped(grid=[[(1.0, 1.0), (2.0, 2.0)]])

        grid = grid.grid_from_deflection_grid(
            deflection_grid=np.array([[1.0, 0.0], [1.0, 1.0]])
        )

        assert grid.in_grouped_list == [[(0.0, 1.0), (1.0, 1.0)]]

    def test__with_mask__converts_to_and_from_pixels(self):

        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(2, 2)), pixel_scales=(2.0, 2.0)
        )

        grid = aa.GridIrregularGrouped(grid=[[(1.0, -1.0), (1.0, 1.0)]])

        assert grid.in_grouped_list == [[(1.0, -1.0), (1.0, 1.0)]]

        grid = aa.GridIrregularGrouped.from_pixels_and_mask(
            pixels=[[(0, 0), (0, 1)]], mask=mask
        )

        assert grid.in_grouped_list == [[(1.0, -1.0), (1.0, 1.0)]]

    def test__load_grid__retains_list_structure(self):

        file_path = path.join(test_grid_dir, "output_test", "grid_test.dat")

        grid = aa.GridIrregularGrouped.from_file(file_path=file_path)

        assert grid.in_grouped_list == [
            [(4.0, 4.0), (5.0, 5.0)],
            [(6.0, 6.0), (7.0, 7.0), (8.0, 8.0)],
        ]

    def test__output_grid_to_file(self):
        grid = aa.GridIrregularGrouped(
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

        grid = aa.GridIrregularGrouped.from_file(file_path=file_path)

        assert grid.in_grouped_list == [
            [(4.0, 4.0), (5.0, 5.0)],
            [(6.0, 6.0), (7.0, 7.0), (8.0, 8.0)],
        ]

        with pytest.raises(FileExistsError):
            grid.output_to_file(file_path=file_path)

        grid.output_to_file(file_path=file_path, overwrite=True)


class TestGridIrregularGroupedUniform:
    def test__from_sparse_uniform_upscale(self):

        grid_sparse_uniform = aa.GridIrregularGroupedUniform(
            grid=[[(1.0, 1.0), (1.0, 3.0)]], pixel_scales=2.0
        )

        grid_upscale = grids.GridIrregularGroupedUniform.from_grid_sparse_uniform_upscale(
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

        grid_sparse_uniform = aa.GridIrregularGroupedUniform(
            grid=[[(1.0, 1.0), (1.0, 3.0), (1.0, 5.0), (3.0, 3.0)]], pixel_scales=2.0
        )

        grid_upscale = grids.GridIrregularGroupedUniform.from_grid_sparse_uniform_upscale(
            grid_sparse_uniform=grid_sparse_uniform, upscale_factor=4, pixel_scales=2.0
        )

        grid_upscale_util = aa.util.grid.grid_upscaled_1d_from(
            grid_1d=grid_sparse_uniform, upscale_factor=4, pixel_scales=(2.0, 2.0)
        )

        assert (grid_upscale == grid_upscale_util).all()

    def test__grid_from_deflection_grid(self):
        grid = aa.GridIrregularGroupedUniform(
            grid=[[(1.0, 1.0), (2.0, 2.0)]], pixel_scales=(1.0, 1.0), shape_2d=(3, 3)
        )

        grid = grid.grid_from_deflection_grid(
            deflection_grid=np.array([[1.0, 0.0], [1.0, 1.0]])
        )

        assert type(grid) == grids.GridIrregularGroupedUniform
        assert grid.in_grouped_list == [[(0.0, 1.0), (1.0, 1.0)]]
        assert grid.pixel_scales == (1.0, 1.0)
        assert grid.shape_2d == (3, 3)

    def test__grid_from_grid_1d(self):

        grid = aa.GridIrregularGroupedUniform(
            grid=[[(1.0, 1.0), (2.0, 2.0)]], pixel_scales=(1.0, 1.0), shape_2d=(3, 3)
        )

        grid_from_1d = grid.grid_from_grid_1d(
            grid_1d=np.array([[1.0, 1.0], [2.0, 2.0]])
        )

        assert type(grid_from_1d) == grids.GridIrregularGroupedUniform
        assert grid_from_1d.in_grouped_list == [[(1.0, 1.0), (2.0, 2.0)]]
        assert grid.pixel_scales == (1.0, 1.0)
        assert grid.shape_2d == (3, 3)

        grid = aa.GridIrregularGroupedUniform(
            grid=[[(1.0, 1.0), (2.0, 2.0)], [(3.0, 3.0)]],
            pixel_scales=(1.0, 3.0),
            shape_2d=(5, 5),
        )

        grid_from_1d = grid.grid_from_grid_1d(
            grid_1d=np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        )

        assert type(grid_from_1d) == grids.GridIrregularGroupedUniform
        assert grid_from_1d.in_grouped_list == [[(1.0, 1.0), (2.0, 2.0)], [(3.0, 3.0)]]
        assert grid.pixel_scales == (1.0, 3.0)
        assert grid.shape_2d == (5, 5)
