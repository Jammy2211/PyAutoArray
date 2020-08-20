import autoarray as aa
from autoarray.structures import grids
from autoarray.inversion import mappers
import numpy as np
import pytest


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

        mask = aa.Mask.manual(mask=mask, pixel_scales=1.0, sub_size=1)

        # Source-plane comprises 5 grid, so 5 masked_image pixels traced to the pix-plane.

        grid = aa.Grid.manual_mask(
            grid=np.array(
                [[1.0, -1.0], [1.0, 1.0], [0.0, 0.0], [-1.0, -1.0], [-1.0, 1.0]]
            ),
            mask=mask,
        )

        # There is no sub-grid, so our grid are just the masked_image grid (note the NumPy weighted_data structure
        # ensures this has no sub-gridding)

        pix = aa.pix.Rectangular(shape=(3, 3))

        mapper = pix.mapper_from_grid_and_sparse_grid(
            grid=grid,
            sparse_grid=None,
            settings=aa.SettingsPixelization(use_border=False),
            hyper_image=np.ones((2, 2)),
        )

        assert mapper.is_image_plane_pixelization == False
        assert mapper.grid.shape_2d_scaled == pytest.approx((2.0, 2.0), 1.0e-4)
        assert mapper.grid.origin == pytest.approx((0.0, 0.0), 1.0e-4)

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
        regularization_matrix = reg.regularization_matrix_from_mapper(mapper=mapper)

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

        image = aa.Array.ones(shape_2d=(7, 7))
        noise_map = aa.Array.ones(shape_2d=(7, 7))
        psf = aa.Kernel.no_blur()

        imaging = aa.Imaging(image=image, noise_map=noise_map, psf=psf)

        masked_data = aa.MaskedImaging(imaging=imaging, mask=mask)

        inversion = aa.Inversion(
            masked_dataset=masked_data,
            mapper=mapper,
            regularization=reg,
            settings=aa.SettingsInversion(check_solution=False),
        )

        assert (inversion.blurred_mapping_matrix == mapper.mapping_matrix).all()
        assert (inversion.regularization_matrix == regularization_matrix).all()
        assert inversion.mapped_reconstructed_image == pytest.approx(np.ones(5), 1.0e-4)

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

        mask = aa.Mask.manual(mask=mask, pixel_scales=1.0, sub_size=1)

        # There is no sub-grid, so our grid are just the masked_image grid (note the NumPy weighted_data structure
        # ensures this has no sub-gridding)
        grid = aa.Grid.manual_mask(
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

        pix = aa.pix.Rectangular(shape=(3, 3))

        mapper = pix.mapper_from_grid_and_sparse_grid(
            grid=grid,
            sparse_grid=None,
            settings=aa.SettingsPixelization(use_border=False),
        )

        assert mapper.is_image_plane_pixelization == False
        assert mapper.pixelization_grid.shape_2d_scaled == pytest.approx(
            (2.2, 2.2), 1.0e-4
        )
        assert mapper.pixelization_grid.origin == pytest.approx((0.0, 0.0), 1.0e-4)

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
        regularization_matrix = reg.regularization_matrix_from_mapper(mapper=mapper)

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

        image = aa.Array.ones(shape_2d=(7, 7))
        noise_map = aa.Array.ones(shape_2d=(7, 7))
        psf = aa.Kernel.no_blur()

        imaging = aa.Imaging(image=image, noise_map=noise_map, psf=psf)

        masked_data = aa.MaskedImaging(imaging=imaging, mask=mask)

        inversion = aa.Inversion(
            masked_dataset=masked_data,
            mapper=mapper,
            regularization=reg,
            settings=aa.SettingsInversion(check_solution=False),
        )

        assert (inversion.blurred_mapping_matrix == mapper.mapping_matrix).all()
        assert (inversion.regularization_matrix == regularization_matrix).all()
        assert inversion.mapped_reconstructed_image == pytest.approx(
            np.ones(15), 1.0e-4
        )

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

        mask = aa.Mask.manual(mask=mask, pixel_scales=2.0, sub_size=2)

        # Assume a 2x2 sub-grid, so each of our 5 masked_image-pixels are split into 4.
        # The grid below is unphysical in that the (0.0, 0.0) terms on the end of each sub-grid probably couldn't
        # happen for a real lens calculation. This is to make a mapping_matrix matrix which explicitly tests the
        # sub-grid.
        grid = aa.Grid.manual_mask(
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

        pix = aa.pix.Rectangular(shape=(3, 3))

        mapper = pix.mapper_from_grid_and_sparse_grid(
            grid=grid,
            sparse_grid=None,
            settings=aa.SettingsPixelization(use_border=False),
        )

        assert mapper.is_image_plane_pixelization == False
        assert mapper.pixelization_grid.shape_2d_scaled == pytest.approx(
            (2.0, 2.0), 1.0e-4
        )
        assert mapper.pixelization_grid.origin == pytest.approx((0.0, 0.0), 1.0e-4)

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
        regularization_matrix = reg.regularization_matrix_from_mapper(mapper=mapper)

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

        image = aa.Array.ones(shape_2d=(7, 7))
        noise_map = aa.Array.ones(shape_2d=(7, 7))
        psf = aa.Kernel.no_blur()

        imaging = aa.Imaging(image=image, noise_map=noise_map, psf=psf)

        masked_data = aa.MaskedImaging(imaging=imaging, mask=mask)

        inversion = aa.Inversion(
            masked_dataset=masked_data,
            mapper=mapper,
            regularization=reg,
            settings=aa.SettingsInversion(check_solution=False),
        )

        assert (inversion.blurred_mapping_matrix == mapper.mapping_matrix).all()
        assert (inversion.regularization_matrix == regularization_matrix).all()
        assert inversion.mapped_reconstructed_image == pytest.approx(np.ones(5), 1.0e-4)

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

        mask = aa.Mask.manual(mask=mask, pixel_scales=1.0, sub_size=1)

        grid = aa.Grid.manual_mask(
            grid=np.array(
                [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [-1.0, -1.0]]
            ),
            mask=mask,
        )

        pix = aa.pix.Rectangular(shape=(3, 3))

        mapper = pix.mapper_from_grid_and_sparse_grid(
            grid=grid,
            sparse_grid=None,
            settings=aa.SettingsPixelization(use_border=False),
        )

        assert mapper.is_image_plane_pixelization == False
        assert mapper.pixelization_grid.shape_2d_scaled == pytest.approx(
            (2.0, 2.0), 1.0e-4
        )
        assert mapper.pixelization_grid.origin == pytest.approx((0.0, 0.0), 1.0e-4)

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
        regularization_matrix = reg.regularization_matrix_from_mapper(mapper=mapper)

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

        image = aa.Array.ones(shape_2d=(7, 7))
        noise_map = aa.Array.ones(shape_2d=(7, 7))
        psf = aa.Kernel.no_blur()

        imaging = aa.Imaging(image=image, noise_map=noise_map, psf=psf)

        masked_data = aa.MaskedImaging(imaging=imaging, mask=mask)

        inversion = aa.Inversion(
            masked_dataset=masked_data,
            mapper=mapper,
            regularization=reg,
            settings=aa.SettingsInversion(check_solution=False),
        )

        assert (inversion.blurred_mapping_matrix == mapper.mapping_matrix).all()
        assert (inversion.regularization_matrix == regularization_matrix).all()
        assert inversion.mapped_reconstructed_image == pytest.approx(np.ones(5), 1.0e-4)

    def test__interferometer_matrices(self):

        visibilities_mask = np.full(fill_value=False, shape=(7, 2))

        real_space_mask = np.array(
            [
                [False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False],
            ]
        )

        real_space_mask = aa.Mask.manual(
            mask=real_space_mask, pixel_scales=0.1, sub_size=1
        )

        grid = aa.Grid.from_mask(mask=real_space_mask)

        pix = aa.pix.Rectangular(shape=(7, 7))

        mapper = pix.mapper_from_grid_and_sparse_grid(
            grid=grid,
            sparse_grid=None,
            settings=aa.SettingsPixelization(use_border=False),
        )

        reg = aa.reg.Constant(coefficient=0.0)

        visibilities = aa.Visibilities.manual_1d(
            visibilities=[
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ]
        )
        noise_map = aa.VisibilitiesNoiseMap.ones(shape_1d=(7,))
        uv_wavelengths = np.ones(shape=(7, 2))

        interferometer = aa.Interferometer(
            visibilities=visibilities,
            noise_map=noise_map,
            uv_wavelengths=uv_wavelengths,
        )

        masked_data = aa.MaskedInterferometer(
            interferometer=interferometer,
            visibilities_mask=visibilities_mask,
            real_space_mask=real_space_mask,
        )

        inversion = aa.Inversion(
            masked_dataset=masked_data,
            mapper=mapper,
            regularization=reg,
            settings=aa.SettingsInversion(check_solution=False),
        )

        assert inversion.mapped_reconstructed_visibilities[:, 0] == pytest.approx(
            np.ones(shape=(7,)), 1.0e-4
        )
        assert inversion.mapped_reconstructed_visibilities[:, 1] == pytest.approx(
            np.zeros(shape=(7,)), 1.0e-4
        )

    def test__interferometer_linear_operator(self):

        visibilities_mask = np.full(fill_value=False, shape=(7, 2))

        real_space_mask = np.array(
            [
                [False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False],
            ]
        )

        real_space_mask = aa.Mask.manual(
            mask=real_space_mask, pixel_scales=0.1, sub_size=1
        )

        grid = aa.Grid.from_mask(mask=real_space_mask)

        pix = aa.pix.Rectangular(shape=(7, 7))

        mapper = pix.mapper_from_grid_and_sparse_grid(
            grid=grid,
            sparse_grid=None,
            settings=aa.SettingsPixelization(use_border=False),
        )

        reg = aa.reg.Constant(coefficient=0.0)

        visibilities = aa.Visibilities.manual_1d(
            visibilities=[
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ]
        )
        noise_map = aa.VisibilitiesNoiseMap.ones(shape_1d=(7,))
        uv_wavelengths = np.ones(shape=(7, 2))

        interferometer = aa.Interferometer(
            visibilities=visibilities,
            noise_map=noise_map,
            uv_wavelengths=uv_wavelengths,
        )

        masked_data = aa.MaskedInterferometer(
            interferometer=interferometer,
            visibilities_mask=visibilities_mask,
            real_space_mask=real_space_mask,
            settings=aa.SettingsMaskedInterferometer(
                transformer_class=aa.TransformerNUFFT
            ),
        )

        inversion = aa.Inversion(
            masked_dataset=masked_data,
            mapper=mapper,
            regularization=reg,
            settings=aa.SettingsInversion(
                use_linear_operators=True, check_solution=False
            ),
        )

        assert inversion.mapped_reconstructed_visibilities[:, 0] == pytest.approx(
            0.56233 * np.ones(shape=(7,)), 1.0e-4
        )
        assert inversion.mapped_reconstructed_visibilities[:, 1] == pytest.approx(
            np.zeros(shape=(7,)), 1.0e-4
        )


class TestVoronoiMagnification:
    def test__3x3_simple_grid(self):

        mask = aa.Mask.manual(
            mask=np.array(
                [
                    [True, True, True, True, True],
                    [True, False, False, False, True],
                    [True, False, False, False, True],
                    [True, False, False, False, True],
                    [True, True, True, True, True],
                ]
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

        grid = aa.Grid.manual_mask(grid=grid, mask=mask)

        pix = aa.pix.VoronoiMagnification(shape=(3, 3))
        sparse_grid = grids.GridSparse.from_grid_and_unmasked_2d_grid_shape(
            grid=grid, unmasked_sparse_shape=pix.shape
        )

        pixelization_grid = aa.GridVoronoi(
            grid=sparse_grid.sparse,
            nearest_pixelization_1d_index_for_mask_1d_index=sparse_grid.sparse_1d_index_for_mask_1d_index,
        )

        mapper = pix.mapper_from_grid_and_sparse_grid(
            grid=grid,
            sparse_grid=pixelization_grid,
            settings=aa.SettingsPixelization(use_border=False),
            hyper_image=np.ones((2, 2)),
        )

        assert mapper.is_image_plane_pixelization == True
        assert mapper.pixelization_grid.shape_2d_scaled == pytest.approx(
            (2.0, 2.0), 1.0e-4
        )
        assert (mapper.pixelization_grid == sparse_grid.sparse).all()
        #     assert mapper.pixelization_grid.origin == pytest.approx((0.0, 0.0), 1.0e-4)
        assert (mapper.hyper_image == np.ones((2, 2))).all()

        assert isinstance(mapper, mappers.MapperVoronoi)

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
        regularization_matrix = reg.regularization_matrix_from_mapper(mapper=mapper)

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

        image = aa.Array.ones(shape_2d=(5, 5))
        noise_map = aa.Array.ones(shape_2d=(5, 5))
        psf = aa.Kernel.no_blur()

        imaging = aa.Imaging(image=image, noise_map=noise_map, psf=psf)

        masked_data = aa.MaskedImaging(imaging=imaging, mask=mask)

        inversion = aa.Inversion(
            masked_dataset=masked_data,
            mapper=mapper,
            regularization=reg,
            settings=aa.SettingsInversion(check_solution=False),
        )

        assert (inversion.blurred_mapping_matrix == mapper.mapping_matrix).all()
        assert (inversion.regularization_matrix == regularization_matrix).all()
        assert inversion.mapped_reconstructed_image == pytest.approx(np.ones(9), 1.0e-4)

    def test__3x3_simple_grid__include_mask(self):

        mask = aa.Mask.manual(
            mask=np.array(
                [
                    [True, True, True, True, True],
                    [True, True, False, True, True],
                    [True, False, False, False, True],
                    [True, True, False, True, True],
                    [True, True, True, True, True],
                ]
            ),
            pixel_scales=1.0,
            sub_size=1,
        )

        grid = np.array([[1.0, 0.0], [0.0, -1.0], [0.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])

        grid = aa.Grid.manual_mask(grid=grid, mask=mask)

        pix = aa.pix.VoronoiMagnification(shape=(3, 3))
        sparse_grid = grids.GridSparse.from_grid_and_unmasked_2d_grid_shape(
            grid=grid, unmasked_sparse_shape=pix.shape
        )

        pixelization_grid = aa.GridVoronoi(
            grid=sparse_grid.sparse,
            nearest_pixelization_1d_index_for_mask_1d_index=sparse_grid.sparse_1d_index_for_mask_1d_index,
        )

        mapper = pix.mapper_from_grid_and_sparse_grid(
            grid=grid,
            sparse_grid=pixelization_grid,
            settings=aa.SettingsPixelization(use_border=False),
        )

        assert mapper.is_image_plane_pixelization == True
        assert mapper.pixelization_grid.shape_2d_scaled == pytest.approx(
            (2.0, 2.0), 1.0e-4
        )
        assert (mapper.pixelization_grid == sparse_grid.sparse).all()
        #   assert mapper.pixelization_grid.origin == pytest.approx((0.0, 0.0), 1.0e-4)

        assert isinstance(mapper, mappers.MapperVoronoi)

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
        regularization_matrix = reg.regularization_matrix_from_mapper(mapper=mapper)

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

        image = aa.Array.ones(shape_2d=(5, 5))
        noise_map = aa.Array.ones(shape_2d=(5, 5))
        psf = aa.Kernel.no_blur()

        imaging = aa.Imaging(image=image, noise_map=noise_map, psf=psf)

        masked_data = aa.MaskedImaging(imaging=imaging, mask=mask)

        inversion = aa.Inversion(
            masked_dataset=masked_data,
            mapper=mapper,
            regularization=reg,
            settings=aa.SettingsInversion(check_solution=False),
        )

        assert (inversion.blurred_mapping_matrix == mapper.mapping_matrix).all()
        assert (inversion.regularization_matrix == regularization_matrix).all()
        assert inversion.mapped_reconstructed_image == pytest.approx(np.ones(5), 1.0e-4)

    def test__3x3_simple_grid__include_mask_and_sub_grid(self):

        mask = aa.Mask.manual(
            mask=np.array(
                [
                    [True, True, True, True, True],
                    [True, True, False, True, True],
                    [True, False, False, False, True],
                    [True, True, False, True, True],
                    [True, True, True, True, True],
                ]
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

        grid = aa.Grid.manual_mask(grid=grid, mask=mask)

        pix = aa.pix.VoronoiMagnification(shape=(3, 3))
        sparse_grid = grids.GridSparse.from_grid_and_unmasked_2d_grid_shape(
            grid=grid, unmasked_sparse_shape=pix.shape
        )

        pixelization_grid = aa.GridVoronoi(
            grid=sparse_grid.sparse,
            nearest_pixelization_1d_index_for_mask_1d_index=sparse_grid.sparse_1d_index_for_mask_1d_index,
        )

        mapper = pix.mapper_from_grid_and_sparse_grid(
            grid=grid,
            sparse_grid=pixelization_grid,
            settings=aa.SettingsPixelization(use_border=False),
        )

        assert mapper.is_image_plane_pixelization == True
        assert mapper.grid.shape_2d_scaled == pytest.approx((2.02, 2.01), 1.0e-4)
        assert (mapper.pixelization_grid == sparse_grid.sparse).all()
        #    assert mapper.pixelization_grid.origin == pytest.approx((0.0, 0.005), 1.0e-4)

        assert isinstance(mapper, mappers.MapperVoronoi)

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
        regularization_matrix = reg.regularization_matrix_from_mapper(mapper=mapper)

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

        image = aa.Array.ones(shape_2d=(5, 5))
        noise_map = aa.Array.ones(shape_2d=(5, 5))
        psf = aa.Kernel.no_blur()

        imaging = aa.Imaging(image=image, noise_map=noise_map, psf=psf)

        masked_data = aa.MaskedImaging(imaging=imaging, mask=mask)

        inversion = aa.Inversion(
            masked_dataset=masked_data,
            mapper=mapper,
            regularization=reg,
            settings=aa.SettingsInversion(check_solution=False),
        )

        assert (inversion.blurred_mapping_matrix == mapper.mapping_matrix).all()
        assert (inversion.regularization_matrix == regularization_matrix).all()
        assert inversion.mapped_reconstructed_image == pytest.approx(np.ones(5), 1.0e-4)

    def test__3x3_simple_grid__include_mask_with_offset_centre(self):

        mask = aa.Mask.manual(
            mask=np.array(
                [
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, False, True, True],
                    [True, True, True, False, False, False, True],
                    [True, True, True, True, False, True, True],
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True],
                    [True, True, True, True, True, True, True],
                ]
            ),
            pixel_scales=1.0,
            sub_size=1,
        )

        grid = np.array([[2.0, 1.0], [1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [0.0, 1.0]])

        grid = aa.Grid.manual_mask(grid=grid, mask=mask)

        pix = aa.pix.VoronoiMagnification(shape=(3, 3))
        sparse_grid = grids.GridSparse.from_grid_and_unmasked_2d_grid_shape(
            grid=grid, unmasked_sparse_shape=pix.shape
        )

        pixelization_grid = aa.GridVoronoi(
            grid=sparse_grid.sparse,
            nearest_pixelization_1d_index_for_mask_1d_index=sparse_grid.sparse_1d_index_for_mask_1d_index,
        )

        mapper = pix.mapper_from_grid_and_sparse_grid(
            grid=grid,
            sparse_grid=pixelization_grid,
            settings=aa.SettingsPixelization(use_border=False),
        )

        assert mapper.is_image_plane_pixelization == True
        assert mapper.pixelization_grid.shape_2d_scaled == pytest.approx(
            (2.0, 2.0), 1.0e-4
        )
        assert (mapper.pixelization_grid == sparse_grid.sparse).all()
        #   assert mapper.pixelization_grid.origin == pytest.approx((1.0, 1.0), 1.0e-4)

        assert isinstance(mapper, mappers.MapperVoronoi)

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
        regularization_matrix = reg.regularization_matrix_from_mapper(mapper=mapper)

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

        image = aa.Array.ones(shape_2d=(7, 7))
        noise_map = aa.Array.ones(shape_2d=(7, 7))
        psf = aa.Kernel.no_blur()

        imaging = aa.Imaging(image=image, noise_map=noise_map, psf=psf)

        masked_data = aa.MaskedImaging(imaging=imaging, mask=mask)

        inversion = aa.Inversion(
            masked_dataset=masked_data,
            mapper=mapper,
            regularization=reg,
            settings=aa.SettingsInversion(check_solution=False),
        )

        assert (inversion.blurred_mapping_matrix == mapper.mapping_matrix).all()
        assert (inversion.regularization_matrix == regularization_matrix).all()
        assert inversion.mapped_reconstructed_image == pytest.approx(np.ones(5), 1.0e-4)

    def test__interferometer(self):

        visibilities_mask = np.full(fill_value=False, shape=(7, 2))

        real_space_mask = np.array(
            [
                [False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False],
                [False, False, False, False, False, False, False],
            ]
        )

        real_space_mask = aa.Mask.manual(
            mask=real_space_mask, pixel_scales=0.1, sub_size=1
        )

        grid = aa.Grid.from_mask(mask=real_space_mask)

        pix = aa.pix.VoronoiMagnification(shape=(7, 7))

        sparse_grid = pix.sparse_grid_from_grid(grid=grid)

        mapper = pix.mapper_from_grid_and_sparse_grid(
            grid=grid,
            sparse_grid=sparse_grid,
            settings=aa.SettingsPixelization(use_border=False),
        )

        reg = aa.reg.Constant(coefficient=0.0)

        visibilities = aa.Visibilities.manual_1d(
            visibilities=[
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
            ]
        )
        noise_map = aa.VisibilitiesNoiseMap.ones(shape_1d=(7,))
        uv_wavelengths = np.ones(shape=(7, 2))

        interferometer = aa.Interferometer(
            visibilities=visibilities,
            noise_map=noise_map,
            uv_wavelengths=uv_wavelengths,
        )

        masked_data = aa.MaskedInterferometer(
            interferometer=interferometer,
            visibilities_mask=visibilities_mask,
            real_space_mask=real_space_mask,
        )

        inversion = aa.Inversion(
            masked_dataset=masked_data,
            mapper=mapper,
            regularization=reg,
            settings=aa.SettingsInversion(check_solution=False),
        )

        assert inversion.mapped_reconstructed_visibilities[:, 0] == pytest.approx(
            np.ones(shape=(7,)), 1.0e-4
        )
        assert inversion.mapped_reconstructed_visibilities[:, 1] == pytest.approx(
            np.zeros(shape=(7,)), 1.0e-4
        )
