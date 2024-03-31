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
    include_2d = aplt.Include2D(origin=True, mask=True, border=True)

    get_visuals = aplt.GetVisuals2D(include=include_2d, visuals=visuals_2d)

    visuals_2d_via = get_visuals.via_mask_from(mask=mask_2d_7x7)

    assert visuals_2d_via.origin == (1.0, 1.0)
    assert (visuals_2d_via.mask == mask_2d_7x7).all()
    assert (visuals_2d_via.border == mask_2d_7x7.derive_grid.border).all()
    assert visuals_2d_via.vectors == 2

    include_2d = aplt.Include2D(origin=False, mask=False, border=False)

    get_visuals = aplt.GetVisuals2D(include=include_2d, visuals=visuals_2d)

    visuals_2d_via = get_visuals.via_mask_from(mask=mask_2d_7x7)

    assert visuals_2d_via.origin == (1.0, 1.0)
    assert visuals_2d_via.mask == None
    assert visuals_2d_via.border == None
    assert visuals_2d_via.vectors == 2


def test__via_grid_from(grid_2d_7x7):
    visuals_2d = aplt.Visuals2D()
    include_2d = aplt.Include2D(origin=True)

    get_visuals = aplt.GetVisuals2D(include=include_2d, visuals=visuals_2d)

    visuals_2d_via = get_visuals.via_grid_from(grid=grid_2d_7x7)

    assert (visuals_2d_via.origin == grid_2d_7x7.origin).all()

    include_2d = aplt.Include2D(origin=False)

    get_visuals = aplt.GetVisuals2D(include=include_2d, visuals=visuals_2d)

    visuals_2d_via = get_visuals.via_grid_from(grid=grid_2d_7x7)

    assert visuals_2d_via.origin == None


def test__via_mapper_for_data_from(voronoi_mapper_9_3x3):
    visuals_2d = aplt.Visuals2D(origin=(1.0, 1.0))
    include_2d = aplt.Include2D(
        origin=True, mask=True, border=True, mapper_image_plane_mesh_grid=True
    )

    get_visuals = aplt.GetVisuals2D(include=include_2d, visuals=visuals_2d)

    visuals_2d_via = get_visuals.via_mapper_for_data_from(mapper=voronoi_mapper_9_3x3)

    assert visuals_2d.origin == (1.0, 1.0)
    assert (
        visuals_2d_via.mask == voronoi_mapper_9_3x3.source_plane_data_grid.mask
    ).all()
    assert (
        visuals_2d_via.border
        == voronoi_mapper_9_3x3.source_plane_data_grid.mask.derive_grid.border
    ).all()

    assert (
        visuals_2d_via.mesh_grid == voronoi_mapper_9_3x3.image_plane_mesh_grid
    ).all()

    include_2d = aplt.Include2D(
        origin=False, mask=False, border=False, mapper_image_plane_mesh_grid=False
    )

    get_visuals = aplt.GetVisuals2D(include=include_2d, visuals=visuals_2d)

    visuals_2d_via = get_visuals.via_mapper_for_data_from(mapper=voronoi_mapper_9_3x3)

    assert visuals_2d.origin == (1.0, 1.0)
    assert visuals_2d_via.mask == None
    assert visuals_2d_via.border == None
    assert visuals_2d_via.mesh_grid == None


def test__via_mapper_for_source_from(rectangular_mapper_7x7_3x3):
    visuals_2d = aplt.Visuals2D(origin=(1.0, 1.0))
    include_2d = aplt.Include2D(
        origin=True,
        border=True,
        mapper_source_plane_data_grid=True,
        mapper_source_plane_mesh_grid=True,
    )

    get_visuals = aplt.GetVisuals2D(include=include_2d, visuals=visuals_2d)

    visuals_2d_via = get_visuals.via_mapper_for_source_from(
        mapper=rectangular_mapper_7x7_3x3
    )

    assert visuals_2d.origin == (1.0, 1.0)
    assert (
        visuals_2d_via.grid == rectangular_mapper_7x7_3x3.source_plane_data_grid
    ).all()
    assert (
        visuals_2d_via.border
        == rectangular_mapper_7x7_3x3.source_plane_data_grid.sub_border_grid
    ).all()
    assert (
        visuals_2d_via.mesh_grid == rectangular_mapper_7x7_3x3.source_plane_mesh_grid
    ).all()

    include_2d = aplt.Include2D(
        origin=False,
        border=False,
        mapper_source_plane_data_grid=False,
        mapper_source_plane_mesh_grid=False,
    )

    get_visuals = aplt.GetVisuals2D(include=include_2d, visuals=visuals_2d)

    visuals_2d_via = get_visuals.via_mapper_for_source_from(
        mapper=rectangular_mapper_7x7_3x3
    )

    assert visuals_2d.origin == (1.0, 1.0)
    assert visuals_2d_via.grid == None
    assert visuals_2d_via.border == None
    assert visuals_2d_via.mesh_grid == None


def test__via_fit_imaging_from(fit_imaging_7x7):
    visuals_2d = aplt.Visuals2D(origin=(1.0, 1.0), vectors=2)
    include_2d = aplt.Include2D(origin=True, mask=True, border=True)

    get_visuals = aplt.GetVisuals2D(include=include_2d, visuals=visuals_2d)

    visuals_2d_via = get_visuals.via_fit_imaging_from(fit=fit_imaging_7x7)

    assert visuals_2d_via.origin == (1.0, 1.0)
    assert (visuals_2d_via.mask == fit_imaging_7x7.mask).all()
    assert (
        visuals_2d_via.border == fit_imaging_7x7.mask.derive_grid.border
    ).all()
    assert visuals_2d_via.vectors == 2

    include_2d = aplt.Include2D(origin=False, mask=False, border=False)

    get_visuals = aplt.GetVisuals2D(include=include_2d, visuals=visuals_2d)

    visuals_2d_via = get_visuals.via_fit_imaging_from(fit=fit_imaging_7x7)

    assert visuals_2d_via.origin == (1.0, 1.0)
    assert visuals_2d_via.mask == None
    assert visuals_2d_via.border == None
    assert visuals_2d_via.vectors == 2
