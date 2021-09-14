import numpy as np
import pytest

import autoarray as aa

from autoarray.mock.mock import MockMapper


class TestAbstractMapper:
    def test__slim_indexes_from_pixelization_indexes__different_types_of_lists_input(
        self,
    ):

        mapper = MockMapper(
            pixelization_index_for_sub_slim_index=[0, 0, 0, 0, 0, 0, 0, 0], pixels=9
        )

        full_indexes = mapper.slim_indexes_from_pixelization_indexes(
            pixelization_indexes=[0, 1]
        )

        assert full_indexes == [0, 1, 2, 3, 4, 5, 6, 7]

        mapper = MockMapper(
            pixelization_index_for_sub_slim_index=[0, 0, 0, 0, 3, 4, 4, 7], pixels=9
        )

        full_indexes = mapper.slim_indexes_from_pixelization_indexes(
            pixelization_indexes=[[0], [4]]
        )

        assert full_indexes == [[0, 1, 2, 3], [5, 6]]

    def test__adaptive_pixel_signals_from___matches_util(self, grid_2d_7x7, image_7x7):

        pixels = 6
        signal_scale = 2.0
        pixelization_index_for_sub_slim_index = np.array([1, 1, 4, 0, 0, 3, 0, 0, 3])

        mapper = MockMapper(
            source_grid_slim=grid_2d_7x7,
            pixelization_index_for_sub_slim_index=pixelization_index_for_sub_slim_index,
            hyper_image=image_7x7,
            pixels=pixels,
        )

        pixel_signals = mapper.pixel_signals_from(signal_scale=2.0)

        pixel_signals_util = aa.util.mapper.adaptive_pixel_signals_from(
            pixels=pixels,
            signal_scale=signal_scale,
            pixelization_index_for_sub_slim_index=pixelization_index_for_sub_slim_index,
            slim_index_for_sub_slim_index=grid_2d_7x7.mask.slim_index_for_sub_slim_index,
            hyper_image=image_7x7,
        )

        assert (pixel_signals == pixel_signals_util).all()


class TestRectangularMapper:
    def test__pixelization_index_for_sub_slim_index__matches_util(self,):

        grid = aa.Grid2D.manual_slim(
            [
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
        )

        pixelization_grid = aa.Grid2DRectangular.overlay_grid(
            shape_native=(3, 3), grid=grid
        )

        mapper = aa.Mapper(
            source_grid_slim=grid, source_pixelization_grid=pixelization_grid
        )

        pixelization_index_for_sub_slim_index_util = aa.util.grid_2d.grid_pixel_indexes_2d_slim_from(
            grid_scaled_2d_slim=grid,
            shape_native=pixelization_grid.shape_native,
            pixel_scales=pixelization_grid.pixel_scales,
            origin=pixelization_grid.origin,
        ).astype(
            "int"
        )

        assert (
            mapper.pixelization_index_for_sub_slim_index
            == pixelization_index_for_sub_slim_index_util
        ).all()

    def test__reconstruction_from__matches_util(self):

        grid = aa.Grid2D.manual_slim(
            [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            pixel_scales=1.0,
            shape_native=(2, 2),
        )

        pixelization_grid = aa.Grid2DRectangular.overlay_grid(
            shape_native=(4, 3), grid=grid
        )

        mapper = aa.Mapper(
            source_grid_slim=grid, source_pixelization_grid=pixelization_grid
        )

        solution = np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 2.0, 3.0]
        )
        recon_pix = mapper.reconstruction_from(solution_vector=solution)
        recon_pix_util = aa.util.array_2d.array_2d_native_from(
            array_2d_slim=solution,
            mask_2d=np.full(fill_value=False, shape=(4, 3)),
            sub_size=1,
        )
        assert (recon_pix.native == recon_pix_util).all()
        assert recon_pix.shape_native == (4, 3)

        pixelization_grid = aa.Grid2DRectangular.overlay_grid(
            shape_native=(3, 4), grid=grid
        )

        mapper = aa.Mapper(
            source_grid_slim=grid, source_pixelization_grid=pixelization_grid
        )

        solution = np.array(
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 1.0, 2.0, 3.0]
        )
        recon_pix = mapper.reconstruction_from(solution_vector=solution)
        recon_pix_util = aa.util.array_2d.array_2d_native_from(
            array_2d_slim=solution,
            mask_2d=np.full(fill_value=False, shape=(3, 4)),
            sub_size=1,
        )
        assert (recon_pix.native == recon_pix_util).all()
        assert recon_pix.shape_native == (3, 4)

    def test__pixel_signals_from__matches_util(self, grid_2d_7x7, image_7x7):

        pixelization_grid = aa.Grid2DRectangular.overlay_grid(
            shape_native=(3, 3), grid=grid_2d_7x7
        )

        mapper = aa.Mapper(
            source_grid_slim=grid_2d_7x7,
            source_pixelization_grid=pixelization_grid,
            hyper_data=image_7x7,
        )

        pixel_signals = mapper.pixel_signals_from(signal_scale=2.0)

        pixel_signals_util = aa.util.mapper.adaptive_pixel_signals_from(
            pixels=9,
            signal_scale=2.0,
            pixelization_index_for_sub_slim_index=mapper.pixelization_index_for_sub_slim_index,
            slim_index_for_sub_slim_index=grid_2d_7x7.mask.slim_index_for_sub_slim_index,
            hyper_image=image_7x7,
        )

        assert (pixel_signals == pixel_signals_util).all()


class TestVoronoiMapper:
    def test__pixelization_index_for_sub_slim_index__matches_util(self, grid_2d_7x7):
        pixelization_grid = aa.Grid2D.manual_slim(
            [[0.1, 0.1], [1.1, 0.6], [2.1, 0.1], [0.4, 1.1], [1.1, 7.1], [2.1, 1.1]],
            shape_native=(3, 2),
            pixel_scales=1.0,
        )

        nearest_pixelization_index_for_slim_index = np.array(
            [0, 0, 1, 0, 0, 1, 2, 2, 3]
        )

        pixelization_grid = aa.Grid2DVoronoi(
            grid=pixelization_grid,
            nearest_pixelization_index_for_slim_index=nearest_pixelization_index_for_slim_index,
        )

        mapper = aa.Mapper(
            source_grid_slim=grid_2d_7x7, source_pixelization_grid=pixelization_grid
        )

        pixelization_index_for_sub_slim_index_util = aa.util.mapper.pixelization_index_for_voronoi_sub_slim_index_from(
            grid=grid_2d_7x7,
            nearest_pixelization_index_for_slim_index=pixelization_grid.nearest_pixelization_index_for_slim_index,
            slim_index_for_sub_slim_index=grid_2d_7x7.mask.slim_index_for_sub_slim_index,
            pixelization_grid=pixelization_grid,
            pixel_neighbors=pixelization_grid.pixel_neighbors,
            pixel_neighbors_sizes=pixelization_grid.pixel_neighbors.sizes,
        ).astype(
            "int"
        )

        assert (
            mapper.pixelization_index_for_sub_slim_index
            == pixelization_index_for_sub_slim_index_util
        ).all()
