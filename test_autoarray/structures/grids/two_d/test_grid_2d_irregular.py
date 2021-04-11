import os
from os import path
import shutil
import numpy as np
import pytest

import autoarray as aa

test_grid_dir = path.join("{}".format(path.dirname(path.realpath(__file__))), "files")


class TestGrid2DIrregular:
    def test__input_as_list_or_list_of_other_types__all_convert_correctly(self):

        # Input tuple

        grid = aa.Grid2DIrregular(grid=[(1.0, -1.0)])

        assert type(grid) == aa.Grid2DIrregular
        assert (grid == np.array([[1.0, -1.0]])).all()
        assert grid.in_list == [(1.0, -1.0)]

        # Input tuples

        grid = aa.Grid2DIrregular(grid=[(1.0, -1.0), (1.0, 1.0)])

        assert type(grid) == aa.Grid2DIrregular
        assert (grid == np.array([[1.0, -1.0], [1.0, 1.0]])).all()
        assert grid.in_list == [(1.0, -1.0), (1.0, 1.0)]

        # Input np array

        grid = aa.Grid2DIrregular(grid=[np.array([1.0, -1.0]), np.array([1.0, 1.0])])

        assert type(grid) == aa.Grid2DIrregular
        assert (grid == np.array([[1.0, -1.0], [1.0, 1.0]])).all()
        assert grid.in_list == [(1.0, -1.0), (1.0, 1.0)]

        # Input list

        grid = aa.Grid2DIrregular(grid=[[1.0, -1.0], [1.0, 1.0]])

        assert type(grid) == aa.Grid2DIrregular
        assert (grid == np.array([[1.0, -1.0], [1.0, 1.0]])).all()
        assert grid.in_list == [(1.0, -1.0), (1.0, 1.0)]

    def test__from_yx_1d(self):

        grid = aa.Grid2DIrregular.from_yx_1d(y=[1.0, 1.0], x=[-1.0, 1.0])

        assert grid.in_list == [(1.0, -1.0), (1.0, 1.0)]

        grid = aa.Grid2DIrregular.from_yx_1d(
            y=[1.0, 1.0, 2.0, 4.0], x=[-1.0, 1.0, 3.0, 5.0]
        )

        assert grid.in_list == [(1.0, -1.0), (1.0, 1.0), (2.0, 3.0), (4.0, 5.0)]

    def test__with_mask__converts_to_and_from_pixels(self):

        mask = aa.Mask2D.manual(
            mask=np.full(fill_value=False, shape=(2, 2)), pixel_scales=(2.0, 2.0)
        )

        grid = aa.Grid2DIrregular(grid=[(1.0, -1.0), (1.0, 1.0)])

        assert grid.in_list == [(1.0, -1.0), (1.0, 1.0)]

        grid = aa.Grid2DIrregular.from_pixels_and_mask(
            pixels=[(0, 0), (0, 1)], mask=mask
        )

        assert grid.in_list == [(1.0, -1.0), (1.0, 1.0)]

    def test__values_from_array_slim(self):

        grid = aa.Grid2DIrregular(grid=[(1.0, 1.0), (2.0, 2.0)])

        values_from_1d = grid.values_from_array_slim(array_slim=np.array([1.0, 2.0]))

        assert isinstance(values_from_1d, aa.ValuesIrregular)
        assert values_from_1d.in_list == [1.0, 2.0]

        grid = aa.Grid2DIrregular(grid=[(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)])

        values_from_1d = grid.values_from_array_slim(
            array_slim=np.array([1.0, 2.0, 3.0])
        )

        assert isinstance(values_from_1d, aa.ValuesIrregular)
        assert values_from_1d.in_list == [1.0, 2.0, 3.0]

    def test__values_from_value(self):

        grid = aa.Grid2DIrregular(grid=[(1.0, 1.0), (2.0, 2.0)])

        values_from_value = grid.values_from_value(value=1.0)

        assert values_from_value.in_list == [1.0, 1.0]

    def test__grid_from_grid_slim(self):

        grid = aa.Grid2DIrregular(grid=[(1.0, 1.0), (2.0, 2.0)])

        grid_from_1d = grid.grid_from_grid_slim(
            grid_slim=np.array([[1.0, 1.0], [2.0, 2.0]])
        )

        assert type(grid_from_1d) == aa.Grid2DIrregular
        assert grid_from_1d.in_list == [(1.0, 1.0), (2.0, 2.0)]

        grid = aa.Grid2DIrregular(grid=[(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)])

        grid_from_1d = grid.grid_from_grid_slim(
            grid_slim=np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        )

        assert type(grid_from_1d) == aa.Grid2DIrregular
        assert grid_from_1d.in_list == [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]

    def test__grid_from_deflection_grid(self):

        grid = aa.Grid2DIrregular(grid=[(1.0, 1.0), (2.0, 2.0)])

        grid = grid.grid_from_deflection_grid(
            deflection_grid=np.array([[1.0, 0.0], [1.0, 1.0]])
        )

        assert type(grid) == aa.Grid2DIrregular
        assert grid.in_list == [(0.0, 1.0), (1.0, 1.0)]

    def test__furthest_distances_from_other_coordinates(self):

        grid = aa.Grid2DIrregular(grid=[(0.0, 0.0), (0.0, 1.0)])

        assert grid.furthest_distances_from_other_coordinates.in_list == [1.0, 1.0]

        grid = aa.Grid2DIrregular(grid=[(2.0, 4.0), (3.0, 6.0)])

        assert grid.furthest_distances_from_other_coordinates.in_list == [
            np.sqrt(5),
            np.sqrt(5),
        ]

        grid = aa.Grid2DIrregular(grid=[(0.0, 0.0), (0.0, 1.0), (0.0, 3.0)])

        assert grid.furthest_distances_from_other_coordinates.in_list == [3.0, 2.0, 3.0]

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

    def test__structure_2d_from_result__maps_numpy_array_to__auto_array_or_grid(self):

        grid = aa.Grid2DIrregular(grid=[(1.0, -1.0), (1.0, 1.0)])

        result = grid.structure_2d_from_result(result=np.array([1.0, 2.0]))

        assert isinstance(result, aa.ValuesIrregular)
        assert result.in_list == [1.0, 2.0]

        result = grid.structure_2d_from_result(
            result=np.array([[1.0, 1.0], [2.0, 2.0]])
        )

        assert isinstance(result, aa.Grid2DIrregular)
        assert result.in_list == [(1.0, 1.0), (2.0, 2.0)]

    def test__structure_2d_list_from_result_list__maps_list_to_auto_arrays_or_grids(
        self
    ):

        grid = aa.Grid2DIrregular(grid=[(1.0, -1.0), (1.0, 1.0)])

        result = grid.structure_2d_list_from_result_list(
            result_list=[np.array([1.0, 2.0])]
        )

        assert isinstance(result[0], aa.ValuesIrregular)
        assert result[0].in_list == [1.0, 2.0]

        result = grid.structure_2d_list_from_result_list(
            result_list=[np.array([[1.0, 1.0], [2.0, 2.0]])]
        )

        assert isinstance(result[0], aa.Grid2DIrregular)
        assert result[0].in_list == [(1.0, 1.0), (2.0, 2.0)]

    def test__from_and_to_file_json(self):

        grid = aa.Grid2DIrregular(grid=[(6.0, 6.0), (7.0, 7.0), (8.0, 8.0)])

        output_grid_dir = path.join(
            "{}".format(os.path.dirname(os.path.realpath(__file__))),
            "files",
            "grid",
            "output_test",
        )

        file_path = path.join(output_grid_dir, "grid_test.json")

        if os.path.exists(output_grid_dir):
            shutil.rmtree(output_grid_dir)

        os.makedirs(output_grid_dir)

        grid.output_to_json(file_path=file_path)

        grid = aa.Grid2DIrregular.from_json(file_path=file_path)

        assert grid.in_list == [(6.0, 6.0), (7.0, 7.0), (8.0, 8.0)]

        with pytest.raises(FileExistsError):
            grid.output_to_json(file_path=file_path)

        grid.output_to_json(file_path=file_path, overwrite=True)


class TestGrid2DIrregularUniform:
    def test__from_sparse_uniform_upscale(self):

        grid_sparse_uniform = aa.Grid2DIrregularUniform(
            grid=[[(1.0, 1.0), (1.0, 3.0)]], pixel_scales=2.0
        )

        grid_upscale = aa.Grid2DIrregularUniform.from_grid_sparse_uniform_upscale(
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

        grid_sparse_uniform = aa.Grid2DIrregularUniform(
            grid=[[(1.0, 1.0), (1.0, 3.0), (1.0, 5.0), (3.0, 3.0)]], pixel_scales=2.0
        )

        grid_upscale = aa.Grid2DIrregularUniform.from_grid_sparse_uniform_upscale(
            grid_sparse_uniform=grid_sparse_uniform, upscale_factor=4, pixel_scales=2.0
        )

        grid_upscale_util = aa.util.grid_2d.grid_2d_slim_upscaled_from(
            grid_slim=grid_sparse_uniform, upscale_factor=4, pixel_scales=(2.0, 2.0)
        )

        assert (grid_upscale == grid_upscale_util).all()

    def test__grid_from_deflection_grid(self):

        grid = aa.Grid2DIrregularUniform(
            grid=[(1.0, 1.0), (2.0, 2.0)], pixel_scales=(1.0, 1.0), shape_native=(3, 3)
        )

        grid = grid.grid_from_deflection_grid(
            deflection_grid=np.array([[1.0, 0.0], [1.0, 1.0]])
        )

        assert type(grid) == aa.Grid2DIrregularUniform
        assert grid.in_list == [(0.0, 1.0), (1.0, 1.0)]
        assert grid.pixel_scales == (1.0, 1.0)
        assert grid.shape_native == (3, 3)

    def test__grid_from_grid_slim(self):

        grid = aa.Grid2DIrregularUniform(
            grid=[(1.0, 1.0), (2.0, 2.0)], pixel_scales=(1.0, 1.0), shape_native=(3, 3)
        )

        grid_from_1d = grid.grid_from_grid_slim(
            grid_slim=np.array([[1.0, 1.0], [2.0, 2.0]])
        )

        assert type(grid_from_1d) == aa.Grid2DIrregularUniform
        assert grid_from_1d.in_list == [(1.0, 1.0), (2.0, 2.0)]
        assert grid.pixel_scales == (1.0, 1.0)
        assert grid.shape_native == (3, 3)

        grid = aa.Grid2DIrregularUniform(
            grid=[(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)],
            pixel_scales=(1.0, 3.0),
            shape_native=(5, 5),
        )

        grid_from_1d = grid.grid_from_grid_slim(
            grid_slim=np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
        )

        assert type(grid_from_1d) == aa.Grid2DIrregularUniform
        assert grid_from_1d.in_list == [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
        assert grid.pixel_scales == (1.0, 3.0)
        assert grid.shape_native == (5, 5)
