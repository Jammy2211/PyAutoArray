import numpy as np
import pytest

import autoarray as aa


def test__rectangular_mapper():
    mask = aa.Mask2D(
        mask=[
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, False, True, True, True],
            [True, True, False, False, False, True, True],
            [True, True, True, False, True, True, True],
            [True, True, True, True, True, True, True],
            [True, True, True, True, True, True, True],
        ],
        pixel_scales=2.0,
        origin=(0.5, 0.5),
    )

    # Slightly manipulate input grid so sub gridding is evidence in first source pixel.
    over_sampler = aa.OverSamplerUniform(mask=mask, sub_size=2)
    over_sampled_grid = over_sampler.over_sampled_grid
    over_sampled_grid[0, 0] = -2.0
    over_sampled_grid[0, 1] = 2.0

    mesh = aa.mesh.Rectangular(shape=(3, 3))

    mapper_grids = mesh.mapper_grids_from(
        mask=mask,
        border_relocator=None,
        source_plane_data_grid=over_sampled_grid,
        source_plane_mesh_grid=None,
    )

    mapper = aa.Mapper(
        mapper_grids=mapper_grids, over_sampler=over_sampler, regularization=None
    )

    assert isinstance(mapper, aa.MapperRectangularNoInterp)
    assert mapper.image_plane_mesh_grid == None

    assert mapper.source_plane_mesh_grid.geometry.shape_native_scaled == pytest.approx(
        (5.0, 5.0), 1.0e-4
    )
    assert mapper.source_plane_mesh_grid.origin == pytest.approx((0.5, 0.5), 1.0e-4)
    assert (
        mapper.mapping_matrix
        == np.array(
            [
                [0.0, 0.75, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            ]
        )
    ).all()
    assert mapper.shape_native == (3, 3)


def test__delaunay_mapper():
    mask = aa.Mask2D(
        mask=[
            [True, True, True, True, True],
            [True, True, False, True, True],
            [True, False, False, False, True],
            [True, True, False, True, True],
            [True, True, True, True, True],
        ],
        pixel_scales=1.0,
    )

    # Slightly manipulate input grid so sub gridding is evidence in first source pixel.
    over_sampler = aa.OverSamplerUniform(mask=mask, sub_size=2)
    over_sampled_grid = over_sampler.over_sampled_grid
    over_sampled_grid[0, 0] = -2.0
    over_sampled_grid[0, 1] = 2.0

    mesh = aa.mesh.Delaunay()
    image_mesh = aa.image_mesh.Overlay(shape=(3, 3))
    image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
        mask=mask, adapt_data=None
    )

    mapper_grids = mesh.mapper_grids_from(
        mask=mask,
        border_relocator=None,
        source_plane_data_grid=over_sampled_grid,
        source_plane_mesh_grid=image_plane_mesh_grid,
    )

    mapper = aa.Mapper(
        mapper_grids=mapper_grids, over_sampler=over_sampler, regularization=None
    )

    assert isinstance(mapper, aa.MapperDelaunay)
    assert mapper.source_plane_data_grid.shape_native_scaled_interior == pytest.approx(
        (3.25, 3.25), 1.0e-4
    )
    assert (mapper.source_plane_mesh_grid == image_plane_mesh_grid).all()
    assert mapper.source_plane_mesh_grid.origin == pytest.approx((0.0, 0.0), 1.0e-4)

    assert mapper.mapping_matrix == pytest.approx(
        np.array(
            [
                [0.625, 0.0625, 0.0, 0.3125, 0.0],
                [0.0625, 0.875, 0.0, 0.0, 0.0625],
                [0.125, 0.125, 0.5, 0.125, 0.125],
                [0.0625, 0.0, 0.0, 0.875, 0.0625],
                [0.0, 0.0625, 0.0, 0.0625, 0.875],
            ]
        ),
        1.0e-2,
    )


def test__voronoi_mapper():
    mask = aa.Mask2D(
        mask=[
            [True, True, True, True, True],
            [True, True, False, True, True],
            [True, False, False, False, True],
            [True, True, False, True, True],
            [True, True, True, True, True],
        ],
        pixel_scales=1.0,
    )

    # Slightly manipulate input grid so sub gridding is evidence in first source pixel.
    over_sampler = aa.OverSamplerUniform(mask=mask, sub_size=2)
    over_sampled_grid = over_sampler.over_sampled_grid
    over_sampled_grid[0, 0] = -2.0
    over_sampled_grid[0, 1] = 2.0

    mesh = aa.mesh.Voronoi()
    image_mesh = aa.image_mesh.Overlay(shape=(3, 3))
    image_plane_mesh_grid = image_mesh.image_plane_mesh_grid_from(
        mask=mask, adapt_data=None
    )

    mapper_grids = mesh.mapper_grids_from(
        mask=mask,
        border_relocator=None,
        source_plane_data_grid=over_sampled_grid,
        source_plane_mesh_grid=image_plane_mesh_grid,
    )

    mapper = aa.Mapper(
        mapper_grids=mapper_grids, over_sampler=over_sampler, regularization=None
    )

    assert isinstance(mapper, aa.MapperVoronoiNoInterp)
    assert mapper.source_plane_data_grid.shape_native_scaled_interior == pytest.approx(
        (3.25, 3.25), 1.0e-4
    )
    assert (mapper.source_plane_mesh_grid == image_plane_mesh_grid).all()
    assert mapper.source_plane_mesh_grid.origin == pytest.approx((0.0, 0.0), 1.0e-4)
    assert (
        mapper.mapping_matrix
        == np.array(
            [
                [0.75, 0.0, 0.0, 0.25, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )
    ).all()
