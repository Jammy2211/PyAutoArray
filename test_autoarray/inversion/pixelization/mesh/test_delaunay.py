import autoarray as aa
import numpy as np


def test__magnification__sparse_grid_from__returns_same_as_computed_from_grids_module(
    sub_grid_2d_7x7,
):

    pixelization = aa.mesh.DelaunayMagnification(shape=(3, 3))

    sparse_grid = pixelization.image_plane_mesh_grid_from(image_plane_data_grid=sub_grid_2d_7x7)

    mesh_grid = aa.Mesh2DDelaunay(
        grid=sparse_grid,
        nearest_pixelization_index_for_slim_index=sparse_grid.sparse_index_for_slim_index,
    )

    assert (mesh_grid == sparse_grid).all()
    assert (
        mesh_grid.nearest_pixelization_index_for_slim_index
        == sparse_grid.sparse_index_for_slim_index
    ).all()


def test__magnification__preloads_used_for_relocated_grid(sub_grid_2d_7x7):

    pixelization = aa.mesh.DelaunayMagnification(shape=(3, 3))

    relocated_grid = aa.Grid2D.uniform(shape_native=(3, 3), pixel_scales=1.0)

    sparse_grid = pixelization.image_plane_mesh_grid_from(image_plane_data_grid=sub_grid_2d_7x7)

    mapper_grids = pixelization.mapper_grids_from(
        source_plane_data_grid=relocated_grid,
        source_plane_mesh_grid=sparse_grid,
        settings=aa.SettingsPixelization(use_border=True),
        preloads=aa.Preloads(relocated_grid=relocated_grid),
    )

    assert (mapper_grids.source_plane_data_grid == relocated_grid).all()


def test__brightness__weight_map_from():

    hyper_data = np.array([0.0, 1.0, 0.0])

    pixelization = aa.mesh.DelaunayBrightnessImage(
        pixels=5, weight_floor=0.0, weight_power=0.0
    )

    weight_map = pixelization.weight_map_from(hyper_data=hyper_data)

    assert (weight_map == np.ones(3)).all()

    pixelization = aa.mesh.DelaunayBrightnessImage(
        pixels=5, weight_floor=0.0, weight_power=1.0
    )

    weight_map = pixelization.weight_map_from(hyper_data=hyper_data)

    assert (weight_map == np.array([0.0, 1.0, 0.0])).all()

    pixelization = aa.mesh.DelaunayBrightnessImage(
        pixels=5, weight_floor=1.0, weight_power=1.0
    )

    weight_map = pixelization.weight_map_from(hyper_data=hyper_data)

    assert (weight_map == np.array([1.0, 2.0, 1.0])).all()

    pixelization = aa.mesh.DelaunayBrightnessImage(
        pixels=5, weight_floor=1.0, weight_power=2.0
    )

    weight_map = pixelization.weight_map_from(hyper_data=hyper_data)

    assert (weight_map == np.array([1.0, 4.0, 1.0])).all()

    hyper_data = np.array([-1.0, 1.0, 3.0])

    pixelization = aa.mesh.DelaunayBrightnessImage(
        pixels=5, weight_floor=0.0, weight_power=1.0
    )

    weight_map = pixelization.weight_map_from(hyper_data=hyper_data)

    assert (weight_map == np.array([0.0, 0.5, 1.0])).all()

    pixelization = aa.mesh.DelaunayBrightnessImage(
        pixels=5, weight_floor=0.0, weight_power=2.0
    )

    weight_map = pixelization.weight_map_from(hyper_data=hyper_data)

    assert (weight_map == np.array([0.0, 0.25, 1.0])).all()

    pixelization = aa.mesh.DelaunayBrightnessImage(
        pixels=5, weight_floor=1.0, weight_power=1.0
    )

    weight_map = pixelization.weight_map_from(hyper_data=hyper_data)

    assert (weight_map == np.array([3.0, 3.5, 4.0])).all()


def test__brightness_mesh_grid__matches_manual_comparison_to_grids_module(
    sub_grid_2d_7x7,
):

    pixelization = aa.mesh.DelaunayBrightnessImage(
        pixels=6, weight_floor=0.1, weight_power=2.0
    )

    hyper_data = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

    weight_map = pixelization.weight_map_from(hyper_data=hyper_data)

    sparse_grid = aa.Grid2DSparse.from_total_pixels_grid_and_weight_map(
        total_pixels=pixelization.pixels,
        grid=sub_grid_2d_7x7,
        weight_map=weight_map,
        seed=1,
    )

    mesh_grid = aa.Mesh2DDelaunay(
        grid=sparse_grid,
        nearest_pixelization_index_for_slim_index=sparse_grid.sparse_index_for_slim_index,
    )

    assert (mesh_grid == sparse_grid).all()
    assert (
        mesh_grid.nearest_pixelization_index_for_slim_index
        == sparse_grid.sparse_index_for_slim_index
    ).all()
