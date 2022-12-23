import os
from os import path
import shutil
import numpy as np
import pytest

import autoarray as aa

test_grid_dir = path.join("{}".format(path.dirname(path.realpath(__file__))), "files")


def test__constructor():

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


def test__from_yx_1d():

    grid = aa.Grid2DIrregular.from_yx_1d(y=[1.0, 1.0], x=[-1.0, 1.0])

    assert grid.in_list == [(1.0, -1.0), (1.0, 1.0)]

    grid = aa.Grid2DIrregular.from_yx_1d(
        y=[1.0, 1.0, 2.0, 4.0], x=[-1.0, 1.0, 3.0, 5.0]
    )

    assert grid.in_list == [(1.0, -1.0), (1.0, 1.0), (2.0, 3.0), (4.0, 5.0)]


def test__from_pixels_and_mask():

    mask = aa.Mask2D.manual(
        mask=np.full(fill_value=False, shape=(2, 2)), pixel_scales=(2.0, 2.0)
    )

    grid = aa.Grid2DIrregular(grid=[(1.0, -1.0), (1.0, 1.0)])
    grid = aa.Grid2DIrregular.from_pixels_and_mask(pixels=[(0, 0), (0, 1)], mask=mask)

    assert grid.in_list == [(1.0, -1.0), (1.0, 1.0)]


def test__values_from():

    grid = aa.Grid2DIrregular(grid=[(1.0, 1.0), (2.0, 2.0)])

    values_from_1d = grid.values_from(array_slim=np.array([1.0, 2.0]))

    assert isinstance(values_from_1d, aa.ValuesIrregular)
    assert values_from_1d.in_list == [1.0, 2.0]

    grid = aa.Grid2DIrregular(grid=[(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)])

    values_from_1d = grid.values_from(array_slim=np.array([1.0, 2.0, 3.0]))

    assert isinstance(values_from_1d, aa.ValuesIrregular)
    assert values_from_1d.in_list == [1.0, 2.0, 3.0]


def test__values_via_value_from():

    grid = aa.Grid2DIrregular(grid=[(1.0, 1.0), (2.0, 2.0)])

    values_via_value_from = grid.values_via_value_from(value=1.0)

    assert values_via_value_from.in_list == [1.0, 1.0]


def test__grid_from():

    grid = aa.Grid2DIrregular(grid=[(1.0, 1.0), (2.0, 2.0)])

    grid_from_1d = grid.grid_from(grid_slim=np.array([[1.0, 1.0], [2.0, 2.0]]))

    assert type(grid_from_1d) == aa.Grid2DIrregular
    assert grid_from_1d.in_list == [(1.0, 1.0), (2.0, 2.0)]

    grid = aa.Grid2DIrregular(grid=[(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)])

    grid_from_1d = grid.grid_from(
        grid_slim=np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    )

    assert type(grid_from_1d) == aa.Grid2DIrregular
    assert grid_from_1d.in_list == [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]


def test__grid_2d_via_deflection_grid_from():

    grid = aa.Grid2DIrregular(grid=[(1.0, 1.0), (2.0, 2.0)])

    grid = grid.grid_2d_via_deflection_grid_from(
        deflection_grid=np.array([[1.0, 0.0], [1.0, 1.0]])
    )

    assert type(grid) == aa.Grid2DIrregular
    assert grid.in_list == [(0.0, 1.0), (1.0, 1.0)]


def test__furthest_distances_to_other_coordinates():

    grid = aa.Grid2DIrregular(grid=[(0.0, 0.0), (0.0, 1.0)])

    assert grid.furthest_distances_to_other_coordinates.in_list == [1.0, 1.0]

    grid = aa.Grid2DIrregular(grid=[(2.0, 4.0), (3.0, 6.0)])

    assert grid.furthest_distances_to_other_coordinates.in_list == [
        np.sqrt(5),
        np.sqrt(5),
    ]

    grid = aa.Grid2DIrregular(grid=[(0.0, 0.0), (0.0, 1.0), (0.0, 3.0)])

    assert grid.furthest_distances_to_other_coordinates.in_list == [3.0, 2.0, 3.0]


def test__grid_of_closest_from():

    grid = aa.Grid2DIrregular(grid=[(0.0, 0.0), (0.0, 1.0)])

    grid_of_closest = grid.grid_of_closest_from(grid_pair=np.array([[0.0, 0.1]]))

    assert (grid_of_closest == np.array([[0.0, 0.0]])).all()

    grid_of_closest = grid.grid_of_closest_from(
        grid_pair=np.array([[0.0, 0.1], [0.0, 0.2], [0.0, 0.3]])
    )

    assert (grid_of_closest == np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])).all()

    grid_of_closest = grid.grid_of_closest_from(
        grid_pair=np.array([[0.0, 0.1], [0.0, 0.2], [0.0, 0.9], [0.0, -0.1]])
    )

    assert (
        grid_of_closest == np.array([[0.0, 0.0], [0.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    ).all()


def test__structure_2d_from():

    grid = aa.Grid2DIrregular(grid=[(1.0, -1.0), (1.0, 1.0)])

    result = grid.structure_2d_from(result=np.array([1.0, 2.0]))

    assert isinstance(result, aa.ValuesIrregular)
    assert result.in_list == [1.0, 2.0]

    result = grid.structure_2d_from(result=np.array([[1.0, 1.0], [2.0, 2.0]]))

    assert isinstance(result, aa.Grid2DIrregular)
    assert result.in_list == [(1.0, 1.0), (2.0, 2.0)]


def test__structure_2d_list_from():

    grid = aa.Grid2DIrregular(grid=[(1.0, -1.0), (1.0, 1.0)])

    result = grid.structure_2d_list_from(result_list=[np.array([1.0, 2.0])])

    assert isinstance(result[0], aa.ValuesIrregular)
    assert result[0].in_list == [1.0, 2.0]

    result = grid.structure_2d_list_from(
        result_list=[np.array([[1.0, 1.0], [2.0, 2.0]])]
    )

    assert isinstance(result[0], aa.Grid2DIrregular)
    assert result[0].in_list == [(1.0, 1.0), (2.0, 2.0)]


def test__from_and_to_file_json():

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


def test__uniform__from_grid_sparse_uniform_upscale():

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


def test__uniform__grid_2d_via_deflection_grid_from():

    grid = aa.Grid2DIrregularUniform(
        grid=[(1.0, 1.0), (2.0, 2.0)], pixel_scales=(1.0, 1.0), shape_native=(3, 3)
    )

    grid = grid.grid_2d_via_deflection_grid_from(
        deflection_grid=np.array([[1.0, 0.0], [1.0, 1.0]])
    )

    assert type(grid) == aa.Grid2DIrregularUniform
    assert grid.in_list == [(0.0, 1.0), (1.0, 1.0)]
    assert grid.pixel_scales == (1.0, 1.0)
    assert grid.shape_native == (3, 3)


def test__uniform__grid_from():

    grid = aa.Grid2DIrregularUniform(
        grid=[(1.0, 1.0), (2.0, 2.0)], pixel_scales=(1.0, 1.0), shape_native=(3, 3)
    )

    grid_from_1d = grid.grid_from(grid_slim=np.array([[1.0, 1.0], [2.0, 2.0]]))

    assert type(grid_from_1d) == aa.Grid2DIrregularUniform
    assert grid_from_1d.in_list == [(1.0, 1.0), (2.0, 2.0)]
    assert grid.pixel_scales == (1.0, 1.0)
    assert grid.shape_native == (3, 3)

    grid = aa.Grid2DIrregularUniform(
        grid=[(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)],
        pixel_scales=(1.0, 3.0),
        shape_native=(5, 5),
    )

    grid_from_1d = grid.grid_from(
        grid_slim=np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    )

    assert type(grid_from_1d) == aa.Grid2DIrregularUniform
    assert grid_from_1d.in_list == [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]
    assert grid.pixel_scales == (1.0, 3.0)
    assert grid.shape_native == (5, 5)
