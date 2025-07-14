from os import path
import pytest

import autoarray.plot as aplt


@pytest.fixture(name="plot_path")
def make_plot_path_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plots", "imaging"
    )


def test__via_mask_from(mask_2d_7x7):
    visuals_2d = aplt.Visuals2D(origin=(1.0, 1.0), vectors=2)

    get_visuals = aplt.GetVisuals2D(visuals=visuals_2d)

    visuals_2d_via = get_visuals.via_mask_from(mask=mask_2d_7x7)

    assert visuals_2d_via.origin == (1.0, 1.0)
    assert (visuals_2d_via.mask == mask_2d_7x7).all()
    assert (visuals_2d_via.border == mask_2d_7x7.derive_grid.border).all()
    assert visuals_2d_via.vectors == 2


def test__via_grid_from(grid_2d_7x7):
    visuals_2d = aplt.Visuals2D()

    get_visuals = aplt.GetVisuals2D(visuals=visuals_2d)

    visuals_2d_via = get_visuals.via_grid_from(grid=grid_2d_7x7)

    assert (visuals_2d_via.origin == grid_2d_7x7.origin).all()


def test__via_mapper_for_data_from(voronoi_mapper_9_3x3):
    visuals_2d = aplt.Visuals2D(origin=(1.0, 1.0))

    get_visuals = aplt.GetVisuals2D(visuals=visuals_2d)

    visuals_2d_via = get_visuals.via_mapper_for_data_from(mapper=voronoi_mapper_9_3x3)

    assert visuals_2d.origin == (1.0, 1.0)
    assert (visuals_2d_via.mask == voronoi_mapper_9_3x3.mapper_grids.mask).all()
    assert (
        visuals_2d_via.border
        == voronoi_mapper_9_3x3.mapper_grids.mask.derive_grid.border
    ).all()

    assert (
        visuals_2d_via.mesh_grid == voronoi_mapper_9_3x3.image_plane_mesh_grid
    ).all()


def test__via_mapper_for_source_from(rectangular_mapper_7x7_3x3):
    visuals_2d = aplt.Visuals2D(origin=(1.0, 1.0))

    get_visuals = aplt.GetVisuals2D(visuals=visuals_2d)

    visuals_2d_via = get_visuals.via_mapper_for_source_from(
        mapper=rectangular_mapper_7x7_3x3
    )

    assert visuals_2d.origin == (1.0, 1.0)
    assert (
        visuals_2d_via.grid
        == rectangular_mapper_7x7_3x3.source_plane_data_grid.over_sampled
    ).all()
    border_grid = (
        rectangular_mapper_7x7_3x3.mapper_grids.source_plane_data_grid.over_sampled[
            rectangular_mapper_7x7_3x3.border_relocator.sub_border_slim
        ]
    )
    assert (visuals_2d_via.border == border_grid).all()
    assert (
        visuals_2d_via.mesh_grid == rectangular_mapper_7x7_3x3.source_plane_mesh_grid
    ).all()


def test__via_fit_imaging_from(fit_imaging_7x7):
    visuals_2d = aplt.Visuals2D(origin=(1.0, 1.0), vectors=2)

    get_visuals = aplt.GetVisuals2D(visuals=visuals_2d)

    visuals_2d_via = get_visuals.via_fit_imaging_from(fit=fit_imaging_7x7)

    assert visuals_2d_via.origin == (1.0, 1.0)
    assert (visuals_2d_via.mask == fit_imaging_7x7.mask).all()
    assert (visuals_2d_via.border == fit_imaging_7x7.mask.derive_grid.border).all()
    assert visuals_2d_via.vectors == 2
