import numpy as np
import pytest

import autoarray as aa


def test__max_pixel_list_from_and_centre():
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


def test__max_pixel_list_from__filter_neighbors():
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

    mapper_valued = aa.MapperValued(
        values=np.array([5.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]), mapper=mapper
    )

    pixel_list = mapper_valued.max_pixel_list_from(
        total_pixels=9, filter_neighbors=True
    )

    assert pixel_list[0] == [
        0,
        8,
    ]


def test__interpolated_array_from():
    values = np.array([0.0, 1.0, 1.0, 1.0])

    mapper = aa.m.MockMapper(parameters=4, interpolated_array=values)

    mapper_valued = aa.MapperValued(values=values, mapper=mapper)

    values = mapper_valued.interpolated_array_from(
        shape_native=(3, 3), extent=(-0.2, 0.2, -0.3, 0.3)
    )

    assert (values == np.array([0.0, 1.0, 1.0, 1.0])).all()


def test__interpolated_array_from__with_pixel_mask():
    values = np.array([0.0, 1.0, 1.0, 1.0])

    mapper = aa.m.MockMapper(parameters=4, interpolated_array=values)

    mesh_pixel_mask = np.array([True, False, False, True])

    mapper_valued = aa.MapperValued(
        values=values, mapper=mapper, mesh_pixel_mask=mesh_pixel_mask
    )

    values = mapper_valued.interpolated_array_from(
        shape_native=(3, 3), extent=(-0.2, 0.2, -0.3, 0.3)
    )

    assert values == pytest.approx(np.array([0.0, 1.0, 1.0, 1.0]), 1.0e-4)


def test__magnification_via_mesh_from():
    mask = aa.Mask2D(
        mask=np.array(
            [
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, False, False],
            ]
        ),
        pixel_scales=(0.5, 0.5),
    )

    magnification = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    source_plane_mesh_grid = aa.Mesh2DVoronoi(
        values=np.array(
            [
                [0.0, 0.0],
                [1.0, 0.5],
                [0.0, 1.0],
                [1.0, 2.0],
                [2.0, 1.0],
                [2.0, 2.0],
                [2.0, 0.0],
                [0.0, 2.0],
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        ),
    )

    mapper = aa.m.MockMapper(
        parameters=3,
        source_plane_mesh_grid=source_plane_mesh_grid,
        mask=mask,
        mapping_matrix=np.ones((12, 10)),
    )

    mapper_valued = aa.MapperValued(values=magnification, mapper=mapper)

    magnification = mapper_valued.magnification_via_mesh_from()

    assert magnification == pytest.approx(11.7073170731, 1.0e-4)


def test__magnification_via_mesh_from__with_pixel_mask():
    mask = aa.Mask2D(
        mask=np.array(
            [
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [False, False, False],
            ]
        ),
        pixel_scales=(0.5, 0.5),
    )

    magnification = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    source_plane_mesh_grid = aa.Mesh2DVoronoi(
        values=np.array(
            [
                [0.0, 0.0],
                [1.0, 0.5],
                [0.0, 1.0],
                [1.0, 2.0],
                [2.0, 1.0],
                [2.0, 2.0],
                [2.0, 0.0],
                [0.0, 2.0],
                [1.0, 1.0],
                [1.0, 1.0],
            ]
        ),
    )

    mapper = aa.m.MockMapper(
        parameters=3,
        source_plane_mesh_grid=source_plane_mesh_grid,
        mask=mask,
        mapping_matrix=np.ones((12, 10)),
    )

    mesh_pixel_mask = np.array(
        [True, True, True, True, True, True, True, True, False, False]
    )

    mapper_valued = aa.MapperValued(
        values=magnification, mapper=mapper, mesh_pixel_mask=mesh_pixel_mask
    )

    magnification = mapper_valued.magnification_via_mesh_from()

    assert magnification == pytest.approx(4.0, 1.0e-4)


def test__magnification_via_interpolation_from():
    mask = aa.Mask2D(
        mask=np.array([[False, False], [False, False]]),
        pixel_scales=(0.5, 0.5),
    )

    magnification = aa.Array2D(
        values=[0.0, 1.0, 1.0, 1.0],
        mask=mask,
    )

    mapper = aa.m.MockMapper(
        parameters=4,
        mask=mask,
        interpolated_array=magnification,
        mapping_matrix=np.ones((4, 4)),
    )

    mapper_valued = aa.MapperValued(values=np.array(magnification), mapper=mapper)

    magnification = mapper_valued.magnification_via_interpolation_from()

    assert magnification == pytest.approx(4.0, 1.0e-4)

    magnification = mapper_valued.magnification_via_interpolation_from(
        shape_native=(3, 3), extent=(-1.0, 1.0, -1.0, 1.0)
    )

    assert magnification == pytest.approx(4.0, 1.0e-4)
