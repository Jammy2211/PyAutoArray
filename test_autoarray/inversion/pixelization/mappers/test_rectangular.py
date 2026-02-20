import autoarray as aa

from autoarray.inversion.pixelization.mesh.rectangular_adapt_density import (
    overlay_grid_from,
)
from autoarray.inversion.pixelization.interpolator.rectangular_uniform import (
    rectangular_mappings_weights_via_interpolation_from,
)


def test__pix_indexes_for_sub_slim_index__matches_util():
    grid = aa.Grid2D.no_mask(
        values=[
            [1.5, -1.0],
            [1.3, 0.0],
            [1.0, 1.9],
            [-0.20, -1.0],
            [-5.0, 0.32],
            [6.5, 1.0],
            [-0.34, -7.34],
            [-0.34, 0.75],
            [-6.0, 8.0],
        ],
        pixel_scales=1.0,
        shape_native=(3, 3),
        over_sample_size=1,
    )

    mesh_grid = overlay_grid_from(
        shape_native=(3, 3), grid=grid.over_sampled, buffer=1e-8
    )

    mesh = aa.mesh.RectangularUniform(shape=(3, 3))

    mapper = mesh.mapper_from(
        mask=grid.mask,
        source_plane_data_grid=grid,
        source_plane_mesh_grid=aa.Grid2DIrregular(mesh_grid),
    )

    mappings, weights = rectangular_mappings_weights_via_interpolation_from(
        shape_native=(3, 3),
        mesh_grid=aa.Grid2DIrregular(mesh_grid),
        data_grid=aa.Grid2DIrregular(grid.over_sampled).array,
    )

    assert (mapper.pix_sub_weights.mappings == mappings).all()
    assert (mapper.pix_sub_weights.weights == weights).all()


def test__pixel_signals_from__matches_util(grid_2d_sub_1_7x7, image_7x7):

    mesh_grid = overlay_grid_from(
        shape_native=(3, 3), grid=grid_2d_sub_1_7x7.over_sampled, buffer=1e-8
    )

    mesh = aa.mesh.RectangularAdaptDensity(shape=(3, 3))

    mapper = mesh.mapper_from(
        mask=grid_2d_sub_1_7x7.mask,
        source_plane_data_grid=grid_2d_sub_1_7x7,
        source_plane_mesh_grid=mesh_grid,
        adapt_data=image_7x7,
    )

    pixel_signals = mapper.pixel_signals_from(signal_scale=2.0)

    pixel_signals_util = aa.util.mapper.adaptive_pixel_signals_from(
        pixels=9,
        signal_scale=2.0,
        pix_indexes_for_sub_slim_index=mapper.pix_indexes_for_sub_slim_index,
        pix_size_for_sub_slim_index=mapper.pix_sizes_for_sub_slim_index,
        pixel_weights=mapper.pix_weights_for_sub_slim_index,
        slim_index_for_sub_slim_index=grid_2d_sub_1_7x7.over_sampler.slim_for_sub_slim,
        adapt_data=image_7x7,
    )

    assert (pixel_signals == pixel_signals_util).all()
