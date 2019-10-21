import autoarray as aa
from autoarray.structures import grids
import numpy as np
import pytest

from test_autoarray.mock.mock_grids import MockPixelizationGrid


class TestRectangular:
    def test__5_simple_grid__no_sub_grid(self):

        mask = np.array(
            [
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, False, True, True, True],
                [True, True, False, False, False, True, True],
                [True, True, True, False, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
            ]
        )

        mask = aa.mask.manual(mask_2d=mask, pixel_scales=1.0, sub_size=1)

        # Source-plane comprises 5 grid, so 5 masked_image pixels traced to the pix-plane.

        grid = aa.masked_grid.manual_1d(
            grid=np.array(
                [[1.0, -1.0], [1.0, 1.0], [0.0, 0.0], [-1.0, -1.0], [-1.0, 1.0]]
            ),
            mask=mask,
        )

        pixelization_grid = MockPixelizationGrid(arr=grid)

        # There is no sub-grid, so our grid are just the masked_image grid (note the NumPy weighted_data structure
        # ensures this has no sub-gridding)

        pix = aa.pix.Rectangular(shape=(3, 3))

        mapper = pix.mapper_from_grid_and_pixelization_grid(
            grid=grid,
            pixelization_grid=pixelization_grid,
            inversion_uses_border=False,
            hyper_image=np.ones((2, 2)),
        )

        assert mapper.is_image_plane_pixelization == False
        assert mapper.geometry.shape_2d_arcsec == pytest.approx((2.0, 2.0), 1.0e-4)
        assert mapper.geometry.origin == pytest.approx((0.0, 0.0), 1.0e-4)

        assert (
            mapper.mapping_matrix
            == np.array(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ]
            )
        ).all()
        assert mapper.shape_2d == (3, 3)
        assert (mapper.hyper_image == np.ones((2, 2))).all()

        reg = aa.reg.Constant(coefficient=1.0)
        regularization_matrix = reg.regularization_matrix_from_mapper(
        mapper=mapper
        )

        assert (
            regularization_matrix
            == np.array(
                [
                    [2.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [-1.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 2.00000001, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, -1.0, 4.00000001, -1.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, 0.0, 0.0, -1.0],
                    [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 2.00000001, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, -1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 2.00000001],
                ]
            )
        ).all()

    def test__15_grid__no_sub_grid(self):

        mask = np.array(
            [
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, False, False, False, False, False, True],
                [True, False, False, False, False, False, True],
                [True, False, False, False, False, False, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
            ]
        )

        mask = aa.mask.manual(mask_2d=mask, pixel_scales=1.0, sub_size=1)

        # There is no sub-grid, so our grid are just the masked_image grid (note the NumPy weighted_data structure
        # ensures this has no sub-gridding)
        grid = aa.masked_grid.manual_1d(
            grid=np.array(
                [
                    [0.9, -0.9],
                    [1.0, -1.0],
                    [1.1, -1.1],
                    [0.9, 0.9],
                    [1.0, 1.0],
                    [1.1, 1.1],
                    [-0.01, 0.01],
                    [0.0, 0.0],
                    [0.01, 0.01],
                    [-0.9, -0.9],
                    [-1.0, -1.0],
                    [-1.1, -1.1],
                    [-0.9, 0.9],
                    [-1.0, 1.0],
                    [-1.1, 1.1],
                ]
            ),
            mask=mask,
        )

        pixelization_grid = MockPixelizationGrid(arr=grid)

        pix = aa.pix.Rectangular(shape=(3, 3))

        mapper = pix.mapper_from_grid_and_pixelization_grid(
            grid=grid, pixelization_grid=pixelization_grid, inversion_uses_border=False
        )

        assert mapper.is_image_plane_pixelization == False
        assert mapper.geometry.shape_2d_arcsec == pytest.approx((2.2, 2.2), 1.0e-4)
        assert mapper.geometry.origin == pytest.approx((0.0, 0.0), 1.0e-4)

        assert (
            mapper.mapping_matrix
            == np.array(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ]
            )
        ).all()
        assert mapper.shape_2d == (3, 3)

        reg = aa.reg.Constant(coefficient=1.0)
        regularization_matrix = reg.regularization_matrix_from_mapper(
            mapper=mapper
        )

        assert (
            regularization_matrix
            == np.array(
                [
                    [2.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [-1.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 2.00000001, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, -1.0, 4.00000001, -1.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, 0.0, 0.0, -1.0],
                    [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 2.00000001, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, -1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 2.00000001],
                ]
            )
        ).all()

    def test__5_simple_grid__include_sub_grid(self):

        mask = np.array(
            [
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, False, True, True, True],
                [True, True, False, False, False, True, True],
                [True, True, True, False, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
            ]
        )

        mask = aa.mask.manual(mask_2d=mask, pixel_scales=2.0, sub_size=2)

        # Assume a 2x2 sub-grid, so each of our 5 masked_image-pixels are split into 4.
        # The grid below is unphysical in that the (0.0, 0.0) terms on the end of each sub-grid probably couldn't
        # happen for a real lensing calculation. This is to make a mapping_matrix matrix which explicitly tests the
        # sub-grid.
        grid = aa.masked_grid.manual_1d(
            grid=np.array(
                [
                    [1.0, -1.0],
                    [1.0, -1.0],
                    [1.0, -1.0],
                    [1.0, 1.0],
                    [1.0, 1.0],
                    [1.0, 1.0],
                    [-1.0, -1.0],
                    [-1.0, -1.0],
                    [-1.0, -1.0],
                    [-1.0, 1.0],
                    [-1.0, 1.0],
                    [-1.0, 1.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
            ),
            mask=mask,
        )

        pixelization_grid = MockPixelizationGrid(arr=grid)

        pix = aa.pix.Rectangular(shape=(3, 3))

        mapper = pix.mapper_from_grid_and_pixelization_grid(
            grid=grid, pixelization_grid=pixelization_grid, inversion_uses_border=False
        )

        assert mapper.is_image_plane_pixelization == False
        assert mapper.geometry.shape_2d_arcsec == pytest.approx((2.0, 2.0), 1.0e-4)
        assert mapper.geometry.origin == pytest.approx((0.0, 0.0), 1.0e-4)

        assert (
            mapper.mapping_matrix
            == np.array(
                [
                    [0.75, 0.0, 0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.75],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()
        assert mapper.shape_2d == (3, 3)

        reg = aa.reg.Constant(coefficient=1.0)
        regularization_matrix = reg.regularization_matrix_from_mapper(
            mapper=mapper
        )

        assert (
            regularization_matrix
            == np.array(
                [
                    [2.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [-1.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 2.00000001, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, -1.0, 4.00000001, -1.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, 0.0, 0.0, -1.0],
                    [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 2.00000001, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, -1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 2.00000001],
                ]
            )
        ).all()

    def test__grid__requires_border_relocation(self):

        mask = np.array(
            [
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, False, True, True, True],
                [True, True, False, False, False, True, True],
                [True, True, True, False, True, True, True],
                [True, True, True, True, True, True, True],
                [True, True, True, True, True, True, True],
            ]
        )

        mask = aa.mask.manual(mask_2d=mask, pixel_scales=1.0, sub_size=1)

        grid = aa.masked_grid.manual_1d(
            grid=np.array(
                [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [-1.0, -1.0]]
            ),
            mask=mask,
        )

        pixelization_grid = MockPixelizationGrid(grid)

        pix = aa.pix.Rectangular(shape=(3, 3))

        mapper = pix.mapper_from_grid_and_pixelization_grid(
            grid=grid, pixelization_grid=pixelization_grid, inversion_uses_border=True
        )

        assert mapper.is_image_plane_pixelization == False
        assert mapper.geometry.shape_2d_arcsec == pytest.approx((2.0, 2.0), 1.0e-4)
        assert mapper.geometry.origin == pytest.approx((0.0, 0.0), 1.0e-4)

        assert (
            mapper.mapping_matrix
            == np.array(
                [
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                ]
            )
        ).all()
        assert mapper.shape_2d == (3, 3)

        reg = aa.reg.Constant(coefficient=1.0)
        regularization_matrix = reg.regularization_matrix_from_mapper(
            mapper=mapper
        )

        assert (
            regularization_matrix
            == np.array(
                [
                    [2.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [-1.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 2.00000001, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, -1.0, 4.00000001, -1.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, 0.0, 0.0, -1.0],
                    [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 2.00000001, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, -1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 2.00000001],
                ]
            )
        ).all()


class TestVoronoiMagnification:
    def test__3x3_simple_grid(self):

        mask = aa.mask.manual(
            mask_2d=np.array(
                [[False, False, False], [False, False, False], [False, False, False]]
            ),
            pixel_scales=1.0,
            sub_size=1,
        )

        grid = np.array(
            [
                [1.0, -1.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, -1.0],
                [0.0, 0.0],
                [0.0, 1.0],
                [-1.0, -1.0],
                [-1.0, 0.0],
                [-1.0, 1.0],
            ]
        )

        grid = aa.masked_grid.manual_1d(grid=grid, mask=mask)

        pix = aa.pix.VoronoiMagnification(shape=(3, 3))
        sparse_to_grid = grids.SparseToGrid.from_grid_and_unmasked_2d_grid_shape(
            grid=grid, unmasked_sparse_shape=pix.shape,
        )

        pixelization_grid = MockPixelizationGrid(
            arr=sparse_to_grid.sparse,
            nearest_pixelization_1d_index_for_mask_1d_index=sparse_to_grid.sparse_1d_index_for_mask_1d_index,
        )

        mapper = pix.mapper_from_grid_and_pixelization_grid(
            grid=grid,
            pixelization_grid=pixelization_grid,
            inversion_uses_border=False,
            hyper_image=np.ones((2, 2)),
        )

        assert mapper.is_image_plane_pixelization == True
        assert mapper.geometry.shape_2d_arcsec == pytest.approx((2.0, 2.0), 1.0e-4)
        assert (mapper.pixel_centres == sparse_to_grid.sparse).all()
        assert mapper.geometry.origin == pytest.approx((0.0, 0.0), 1.0e-4)
        assert (mapper.hyper_image == np.ones((2, 2))).all()

        assert isinstance(mapper, aa.mappers.VoronoiMapper)

        assert (
            mapper.mapping_matrix
            == np.array(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                ]
            )
        ).all()

        reg = aa.reg.Constant(coefficient=1.0)
        regularization_matrix = reg.regularization_matrix_from_mapper(
            mapper=mapper
        )

        assert (
            regularization_matrix
            == np.array(
                [
                    [2.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    [-1.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, -1.0, 2.00000001, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 3.00000001, -1.0, 0.0, -1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0, -1.0, 4.00000001, -1.0, 0.0, -1.0, 0.0],
                    [0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, 0.0, 0.0, -1.0],
                    [0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 2.00000001, -1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 3.00000001, -1.0],
                    [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, -1.0, 2.00000001],
                ]
            )
        ).all()

    def test__3x3_simple_grid__include_mask(self):

        mask = aa.mask.manual(
            mask_2d=np.array(
                [[True, False, True], [False, False, False], [True, False, True]]
            ),
            pixel_scales=1.0,
            sub_size=1,
        )

        grid = np.array([[1.0, 0.0], [0.0, -1.0], [0.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

        grid = aa.masked_grid.manual_1d(grid=grid, mask=mask)

        pix = aa.pix.VoronoiMagnification(shape=(3, 3))
        sparse_to_grid = grids.SparseToGrid.from_grid_and_unmasked_2d_grid_shape(
            grid=grid, unmasked_sparse_shape=pix.shape,
        )

        pixelization_grid = MockPixelizationGrid(
            arr=grid,
            nearest_pixelization_1d_index_for_mask_1d_index=sparse_to_grid.sparse_1d_index_for_mask_1d_index,
        )

        mapper = pix.mapper_from_grid_and_pixelization_grid(
            grid=grid, pixelization_grid=pixelization_grid, inversion_uses_border=False
        )

        assert mapper.is_image_plane_pixelization == True
        assert mapper.geometry.shape_2d_arcsec == pytest.approx((2.0, 2.0), 1.0e-4)
        assert (mapper.pixel_centres == sparse_to_grid.sparse).all()
        assert mapper.geometry.origin == pytest.approx((0.0, 0.0), 1.0e-4)

        assert isinstance(mapper, aa.mappers.VoronoiMapper)

        assert (
            mapper.mapping_matrix
            == np.array(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                ]
            )
        ).all()

        reg = aa.reg.Constant(coefficient=1.0)
        regularization_matrix = reg.regularization_matrix_from_mapper(
            mapper=mapper
        )

        assert (
            regularization_matrix
            == np.array(
                [
                    [3.00000001, -1.0, -1.0, -1.0, 0.0],
                    [-1.0, 3.00000001, -1.0, 0.0, -1.0],
                    [-1.0, -1.0, 4.00000001, -1.0, -1.0],
                    [-1.0, 0.0, -1.0, 3.00000001, -1.0],
                    [0.0, -1.0, -1.0, -1.0, 3.00000001],
                ]
            )
        ).all()

    def test__3x3_simple_grid__include_mask_and_sub_grid(self):

        mask = aa.mask.manual(
            mask_2d=np.array(
                [[True, False, True], [False, False, False], [True, False, True]]
            ),
            pixel_scales=1.0,
            sub_size=2,
        )

        grid = np.array(
            [
                [1.01, 0.0],
                [1.01, 0.0],
                [1.01, 0.0],
                [0.01, 0.0],
                [0.0, -1.0],
                [0.0, -1.0],
                [0.0, -1.0],
                [0.01, 0.0],
                [0.01, 0.0],
                [0.01, 0.0],
                [0.01, 0.0],
                [0.01, 0.0],
                [0.0, 1.01],
                [0.0, 1.01],
                [0.0, 1.01],
                [0.01, 0.0],
                [-1.01, 0.0],
                [-1.01, 0.0],
                [-1.01, 0.0],
                [0.01, 0.0],
            ]
        )

        grid = aa.masked_grid.manual_1d(grid=grid, mask=mask)

        pix = aa.pix.VoronoiMagnification(shape=(3, 3))
        sparse_to_grid = grids.SparseToGrid.from_grid_and_unmasked_2d_grid_shape(
            grid=grid, unmasked_sparse_shape=pix.shape,
        )

        pixelization_grid = MockPixelizationGrid(
            arr=sparse_to_grid.sparse,
            nearest_pixelization_1d_index_for_mask_1d_index=sparse_to_grid.sparse_1d_index_for_mask_1d_index,
        )

        mapper = pix.mapper_from_grid_and_pixelization_grid(
            grid=grid, pixelization_grid=pixelization_grid, inversion_uses_border=False
        )

        assert mapper.is_image_plane_pixelization == True
        assert mapper.geometry.shape_2d_arcsec == pytest.approx((2.02, 2.01), 1.0e-4)
        assert (mapper.pixel_centres == sparse_to_grid.sparse).all()
        assert mapper.geometry.origin == pytest.approx((0.0, 0.005), 1.0e-4)

        assert isinstance(mapper, aa.mappers.VoronoiMapper)

        assert (
            mapper.mapping_matrix
            == np.array(
                [
                    [0.75, 0.0, 0.25, 0.0, 0.0],
                    [0.0, 0.75, 0.25, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.25, 0.75, 0.0],
                    [0.0, 0.0, 0.25, 0.0, 0.75],
                ]
            )
        ).all()

        reg = aa.reg.Constant(coefficient=1.0)
        regularization_matrix = reg.regularization_matrix_from_mapper(
            mapper=mapper
        )

        assert (
            regularization_matrix
            == np.array(
                [
                    [3.00000001, -1.0, -1.0, -1.0, 0.0],
                    [-1.0, 3.00000001, -1.0, 0.0, -1.0],
                    [-1.0, -1.0, 4.00000001, -1.0, -1.0],
                    [-1.0, 0.0, -1.0, 3.00000001, -1.0],
                    [0.0, -1.0, -1.0, -1.0, 3.00000001],
                ]
            )
        ).all()

    def test__3x3_simple_grid__include_mask_with_offset_centre(self):

        mask = aa.mask.manual(
            mask_2d=np.array(
                [
                    [True, True, True, False, True],
                    [True, True, False, False, False],
                    [True, True, True, False, True],
                    [True, True, True, True, True],
                    [True, True, True, True, True],
                ]
            ),
            pixel_scales=1.0,
            sub_size=1,
        )

        grid = np.array([[2.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [0.0, 1.0]])

        grid = aa.masked_grid.manual_1d(grid=grid, mask=mask)

        pix = aa.pix.VoronoiMagnification(shape=(3, 3))
        sparse_to_grid = grids.SparseToGrid.from_grid_and_unmasked_2d_grid_shape(
            grid=grid, unmasked_sparse_shape=pix.shape,
        )

        pixelization_grid = MockPixelizationGrid(
            arr=sparse_to_grid.sparse,
            nearest_pixelization_1d_index_for_mask_1d_index=sparse_to_grid.sparse_1d_index_for_mask_1d_index,
        )

        mapper = pix.mapper_from_grid_and_pixelization_grid(
            grid=grid, pixelization_grid=pixelization_grid, inversion_uses_border=False
        )

        assert mapper.is_image_plane_pixelization == True
        assert mapper.geometry.shape_2d_arcsec == pytest.approx((2.0, 2.0), 1.0e-4)
        assert (mapper.pixel_centres == sparse_to_grid.sparse).all()
        assert mapper.geometry.origin == pytest.approx((1.0, 1.0), 1.0e-4)

        assert isinstance(mapper, aa.mappers.VoronoiMapper)

        assert (
            mapper.mapping_matrix
            == np.array(
                [
                    [1.0, 0.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0, 1.0],
                ]
            )
        ).all()

        reg = aa.reg.Constant(coefficient=1.0)
        regularization_matrix = reg.regularization_matrix_from_mapper(
            mapper=mapper
        )

        assert (
            regularization_matrix
            == np.array(
                [
                    [3.00000001, -1.0, -1.0, -1.0, 0.0],
                    [-1.0, 3.00000001, -1.0, 0.0, -1.0],
                    [-1.0, -1.0, 4.00000001, -1.0, -1.0],
                    [-1.0, 0.0, -1.0, 3.00000001, -1.0],
                    [0.0, -1.0, -1.0, -1.0, 3.00000001],
                ]
            )
        ).all()
