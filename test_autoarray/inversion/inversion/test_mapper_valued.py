import numpy as np
from os import path
import pytest

import autoarray as aa

from autoarray import exc



def test__brightest_reconstruction_pixel_and_centre():
    mapper = aa.m.MockMapper(
        source_plane_mesh_grid=aa.Mesh2DVoronoi(
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [5.0, 0.0]]
        )
    )

    mapper_valued = aa.MapperValued(
        values=np.array([2.0, 3.0, 5.0, 0.0]), mapper=mapper
    )

    assert mapper_valued.max_pixel_list_from(total_pixels=2)[0] == [
        2,
        1,
    ]

    assert mapper_valued.max_pixel_centre.in_list == [(5.0, 6.0)]


def test__brightest_reconstruction_pixel__filter_neighbors():
    mapper = aa.m.MockMapper(
        source_plane_mesh_grid=aa.Mesh2DVoronoi(
            [
                [1.0, 1.0],
                [1.0, 2.0],
                [1.0, 3.0],
                [2.0, 1.0],
                [2.0, 2.0],
                [2.0, 3.0],
                [3.0, 1.0],
                [3.0, 2.0],
                [3.0, 3.0],
            ]
        )
    )

    inversion = aa.m.MockInversion(
        linear_obj_list=[mapper],
        reconstruction=np.array([5.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]),
    )

    pixel_list = inversion.brightest_pixel_list_from(
        total_pixels=9, filter_neighbors=True
    )

    assert pixel_list[0] == [
        0,
        8,
    ]


def test__interpolated_reconstruction_list_from():
    interpolated_array = np.array([0.0, 1.0, 1.0, 1.0])

    mapper = aa.m.MockMapper(parameters=3, interpolated_array=interpolated_array)

    inversion = aa.m.MockInversion(
        linear_obj_list=[mapper], reconstruction=interpolated_array
    )

    interpolated_reconstruction_list = inversion.interpolated_reconstruction_list_from(
        shape_native=(3, 3), extent=(-0.2, 0.2, -0.3, 0.3)
    )

    assert (interpolated_reconstruction_list[0] == np.array([0.0, 1.0, 1.0, 1.0])).all()


def test__interpolated_errors_list_from():
    interpolated_array = np.array([0.0, 1.0, 1.0, 1.0])

    mapper = aa.m.MockMapper(parameters=3, interpolated_array=interpolated_array)

    inversion = aa.m.MockInversion(linear_obj_list=[mapper], errors=interpolated_array)

    interpolated_errors_list = inversion.interpolated_errors_list_from(
        shape_native=(3, 3), extent=(-0.2, 0.2, -0.3, 0.3)
    )

    assert (interpolated_errors_list[0] == np.array([0.0, 1.0, 1.0, 1.0])).all()