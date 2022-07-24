from os import path
import pytest

import autoarray as aa
import autoarray.plot as aplt

from autoarray.plot.mat_wrap.get_visuals import GetVisuals1D
from autoarray.plot.mat_wrap.get_visuals import GetVisuals2D


directory = path.dirname(path.realpath(__file__))


@pytest.fixture(name="plot_path")
def make_plot_path_setup():
    return path.join(
        "{}".format(path.dirname(path.realpath(__file__))), "files", "plots", "imaging"
    )


def test__via_array_1d_from(array_1d_7):

    visuals_1d = aplt.Visuals1D(origin=(1.0, 1.0))
    include_1d = aplt.Include1D(origin=True, mask=True)

    get_visuals = GetVisuals1D(include=include_1d, visuals=visuals_1d)

    visuals_1d_via = get_visuals.via_array_1d_from(array_1d=array_1d_7)

    assert visuals_1d_via.origin == (1.0, 1.0)
    assert (visuals_1d_via.mask == array_1d_7.mask).all()

    include_1d = aplt.Include1D(origin=False, mask=False)

    get_visuals = GetVisuals1D(include=include_1d, visuals=visuals_1d)

    visuals_1d_via = get_visuals.via_array_1d_from(array_1d=array_1d_7)

    assert visuals_1d_via.origin == (1.0, 1.0)
    assert visuals_1d_via.mask == None


def test__via_mask_from(mask_2d_7x7):

    visuals_2d = aplt.Visuals2D(origin=(1.0, 1.0), vectors=2)
    include_2d = aplt.Include2D(origin=True, mask=True, border=True)

    get_visuals = GetVisuals2D(include=include_2d, visuals=visuals_2d)

    visuals_2d_via = get_visuals.via_mask_from(mask=mask_2d_7x7)

    assert visuals_2d_via.origin == (1.0, 1.0)
    assert (visuals_2d_via.mask == mask_2d_7x7).all()
    assert (visuals_2d_via.border == mask_2d_7x7.border_grid_sub_1.binned).all()
    assert visuals_2d_via.vectors == 2

    include_2d = aplt.Include2D(origin=False, mask=False, border=False)

    get_visuals = GetVisuals2D(include=include_2d, visuals=visuals_2d)

    visuals_2d_via = get_visuals.via_mask_from(mask=mask_2d_7x7)

    assert visuals_2d_via.origin == (1.0, 1.0)
    assert visuals_2d_via.mask == None
    assert visuals_2d_via.border == None
    assert visuals_2d_via.vectors == 2


def test__via_grid_from(grid_2d_7x7):

    visuals_2d = aplt.Visuals2D()
    include_2d = aplt.Include2D(origin=True)

    get_visuals = GetVisuals2D(include=include_2d, visuals=visuals_2d)

    visuals_2d_via = get_visuals.via_grid_from(grid=grid_2d_7x7)

    assert (visuals_2d_via.origin == grid_2d_7x7.origin).all()

    include_2d = aplt.Include2D(origin=False)

    get_visuals = GetVisuals2D(include=include_2d, visuals=visuals_2d)

    visuals_2d_via = get_visuals.via_grid_from(grid=grid_2d_7x7)

    assert visuals_2d_via.origin == None


def test__via_mapper_for_data_from(voronoi_mapper_9_3x3):

    visuals_2d = aplt.Visuals2D(origin=(1.0, 1.0))
    include_2d = aplt.Include2D(
        origin=True, mask=True, border=True, mapper_data_pixelization_grid=True
    )

    get_visuals = GetVisuals2D(include=include_2d, visuals=visuals_2d)

    visuals_2d_via = get_visuals.via_mapper_for_data_from(mapper=voronoi_mapper_9_3x3)

    assert visuals_2d.origin == (1.0, 1.0)
    assert (visuals_2d_via.mask == voronoi_mapper_9_3x3.source_grid_slim.mask).all()
    assert (
        visuals_2d_via.border
        == voronoi_mapper_9_3x3.source_grid_slim.mask.border_grid_sub_1.binned
    ).all()

    print(visuals_2d.pixelization_grid)
    print(voronoi_mapper_9_3x3.source_mesh_grid)
    assert (
        visuals_2d_via.pixelization_grid == voronoi_mapper_9_3x3.data_mesh_grid
    ).all()

    include_2d = aplt.Include2D(
        origin=False, mask=False, border=False, mapper_data_pixelization_grid=False
    )

    get_visuals = GetVisuals2D(include=include_2d, visuals=visuals_2d)

    visuals_2d_via = get_visuals.via_mapper_for_data_from(mapper=voronoi_mapper_9_3x3)

    assert visuals_2d.origin == (1.0, 1.0)
    assert visuals_2d_via.mask == None
    assert visuals_2d_via.border == None
    assert visuals_2d_via.pixelization_grid == None


def test__via_mapper_for_source_from(rectangular_mapper_7x7_3x3):

    visuals_2d = aplt.Visuals2D(origin=(1.0, 1.0))
    include_2d = aplt.Include2D(
        origin=True,
        border=True,
        mapper_source_grid_slim=True,
        mapper_source_pixelization_grid=True,
    )

    get_visuals = GetVisuals2D(include=include_2d, visuals=visuals_2d)

    visuals_2d_via = get_visuals.via_mapper_for_source_from(
        mapper=rectangular_mapper_7x7_3x3
    )

    assert visuals_2d.origin == (1.0, 1.0)
    assert (visuals_2d_via.grid == rectangular_mapper_7x7_3x3.source_grid_slim).all()
    assert (
        visuals_2d_via.border
        == rectangular_mapper_7x7_3x3.source_grid_slim.sub_border_grid
    ).all()
    assert (
        visuals_2d_via.pixelization_grid == rectangular_mapper_7x7_3x3.source_mesh_grid
    ).all()

    include_2d = aplt.Include2D(
        origin=False,
        border=False,
        mapper_source_grid_slim=False,
        mapper_source_pixelization_grid=False,
    )

    get_visuals = GetVisuals2D(include=include_2d, visuals=visuals_2d)

    visuals_2d_via = get_visuals.via_mapper_for_source_from(
        mapper=rectangular_mapper_7x7_3x3
    )

    assert visuals_2d.origin == (1.0, 1.0)
    assert visuals_2d_via.grid == None
    assert visuals_2d_via.border == None
    assert visuals_2d_via.pixelization_grid == None


def test__via_fit_imaging_from(fit_imaging_7x7):

    visuals_2d = aplt.Visuals2D(origin=(1.0, 1.0), vectors=2)
    include_2d = aplt.Include2D(origin=True, mask=True, border=True)

    get_visuals = GetVisuals2D(include=include_2d, visuals=visuals_2d)

    visuals_2d_via = get_visuals.via_fit_imaging_from(fit=fit_imaging_7x7)

    assert visuals_2d_via.origin == (1.0, 1.0)
    assert (visuals_2d_via.mask == fit_imaging_7x7.mask).all()
    assert (
        visuals_2d_via.border == fit_imaging_7x7.mask.border_grid_sub_1.binned
    ).all()
    assert visuals_2d_via.vectors == 2

    include_2d = aplt.Include2D(origin=False, mask=False, border=False)

    get_visuals = GetVisuals2D(include=include_2d, visuals=visuals_2d)

    visuals_2d_via = get_visuals.via_fit_imaging_from(fit=fit_imaging_7x7)

    assert visuals_2d_via.origin == (1.0, 1.0)
    assert visuals_2d_via.mask == None
    assert visuals_2d_via.border == None
    assert visuals_2d_via.vectors == 2
