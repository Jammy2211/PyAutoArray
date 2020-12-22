import autoarray as aa

import numpy as np
import pytest


class MockRealSpaceMask:
    def __init__(self, grid):

        self.grid = grid
        self.geometry = MockGeometry(grid=grid)

    @property
    def mask_sub_1(self):
        return self

    @property
    def pixels_in_mask(self):
        return self.geometry.masked_grid_sub_1.in_1d_binned.in_radians.shape[0]

    @property
    def pixel_scales(self):
        return self.grid.pixel_scales

    @property
    def sub_size(self):
        return self.grid.sub_size

    @property
    def origin(self):
        return self.grid.origin


class MockGeometry:
    def __init__(self, grid):

        self.masked_grid_sub_1 = MockMaskedGrid(grid=grid)


class MockMaskedGrid:
    def __init__(self, grid):
        self.in_1d_binned = MockMaskedGrid2(grid=grid)


class MockMaskedGrid2:
    def __init__(self, grid):
        self.in_radians = grid


class TestVisiblities:
    def test__visibilities__intensity_image_all_ones__simple_cases(self):

        uv_wavelengths = np.ones(shape=(4, 2))

        grid_radians = aa.Grid.manual_2d(grid=[[[1.0, 1.0]]], pixel_scales=1.0)

        real_space_mask = MockRealSpaceMask(grid=grid_radians)

        transformer = aa.TransformerDFT(
            uv_wavelengths=uv_wavelengths,
            real_space_mask=real_space_mask,
            preload_transform=False,
        )

        image = aa.Array.ones(shape_2d=(1, 1), pixel_scales=1.0)

        visibilities = transformer.visibilities_from_image(image=image)

        assert visibilities == pytest.approx(
            np.array([1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j]), 1.0e-4
        )

        uv_wavelengths = np.array([[0.2, 1.0], [0.5, 1.1], [0.8, 1.2]])

        grid_radians = aa.Grid.manual_2d(
            grid=[[[0.1, 0.2], [0.3, 0.4]]], pixel_scales=1.0
        )

        real_space_mask = MockRealSpaceMask(grid=grid_radians)

        transformer = aa.TransformerDFT(
            uv_wavelengths=uv_wavelengths,
            real_space_mask=real_space_mask,
            preload_transform=False,
        )

        image = aa.Array.ones(shape_2d=(1, 2), pixel_scales=1.0)

        visibilities = transformer.visibilities_from_image(image=image)

        assert visibilities == pytest.approx(
            np.array(
                [-0.091544 - 1.45506j, -0.73359736 - 0.781201j, -0.613160 - 0.077460j]
            ),
            1.0e-4,
        )

    def test__visibilities__intensity_image_varies__simple_cases(self):

        uv_wavelengths = np.ones(shape=(4, 2))
        grid_radians = aa.Grid.manual_2d(grid=[[[1.0, 1.0]]], pixel_scales=1.0)
        real_space_mask = MockRealSpaceMask(grid=grid_radians)

        transformer = aa.TransformerDFT(
            uv_wavelengths=uv_wavelengths,
            real_space_mask=real_space_mask,
            preload_transform=False,
        )

        image = aa.Array.manual_2d([[2.0]], pixel_scales=1.0)

        visibilities = transformer.visibilities_from_image(image=image)

        assert visibilities == pytest.approx(
            np.array([2.0 + 0.0j, 2.0 + 0.0j, 2.0 + 0.0j, 2.0 + 0.0j]), 1.0e-4
        )

        uv_wavelengths = np.array([[0.2, 1.0], [0.5, 1.1], [0.8, 1.2]])

        grid_radians = aa.Grid.manual_2d(
            grid=[[[0.1, 0.2], [0.3, 0.4]]], pixel_scales=1.0
        )
        real_space_mask = MockRealSpaceMask(grid=grid_radians)

        transformer = aa.TransformerDFT(
            uv_wavelengths=uv_wavelengths,
            real_space_mask=real_space_mask,
            preload_transform=False,
        )

        image = aa.Array.manual_2d([[3.0, 6.0]], pixel_scales=1.0)

        visibilities = transformer.visibilities_from_image(image=image)

        assert visibilities == pytest.approx(
            np.array([-2.46153 - 6.418822j, -5.14765 - 1.78146j, -3.11681 + 2.48210j]),
            1.0e-4,
        )

    def test__visibilities__preload_and_non_preload_give_same_answer(self):

        uv_wavelengths = np.array([[0.2, 1.0], [0.5, 1.1], [0.8, 1.2]])
        grid_radians = aa.Grid.manual_2d(
            grid=[[[0.1, 0.2], [0.3, 0.4]]], pixel_scales=1.0
        )
        real_space_mask = MockRealSpaceMask(grid=grid_radians)

        transformer_preload = aa.TransformerDFT(
            uv_wavelengths=uv_wavelengths,
            real_space_mask=real_space_mask,
            preload_transform=True,
        )
        transformer = aa.TransformerDFT(
            uv_wavelengths=uv_wavelengths,
            real_space_mask=real_space_mask,
            preload_transform=False,
        )

        image = aa.Array.manual_2d([[2.0, 6.0]], pixel_scales=1.0)

        visibilities_via_preload = transformer_preload.visibilities_from_image(
            image=image
        )
        visibilities = transformer.visibilities_from_image(image=image)

        assert (visibilities_via_preload == visibilities).all()


class TestVisiblitiesMappingMatrix:
    def test__visibilities__mapping_matrix_all_ones__simple_cases(self):

        uv_wavelengths = np.ones(shape=(4, 2))
        grid_radians = aa.Grid.manual_2d(grid=[[[1.0, 1.0]]], pixel_scales=1.0)
        real_space_mask = MockRealSpaceMask(grid=grid_radians)

        transformer = aa.TransformerDFT(
            uv_wavelengths=uv_wavelengths,
            real_space_mask=real_space_mask,
            preload_transform=False,
        )

        mapping_matrix = np.ones(shape=(1, 1))

        transformed_mapping_matrix = transformer.transformed_mapping_matrix_from_mapping_matrix(
            mapping_matrix=mapping_matrix
        )

        assert transformed_mapping_matrix == pytest.approx(
            np.array([[1.0 + 0.0j], [1.0 + 0.0j], [1.0 + 0.0j], [1.0 + 0.0j]]), 1.0e-4
        )

        uv_wavelengths = np.array([[0.2, 1.0], [0.5, 1.1], [0.8, 1.2]])

        grid_radians = aa.Grid.manual_2d(
            grid=[[[0.1, 0.2], [0.3, 0.4]]], pixel_scales=1.0
        )
        real_space_mask = MockRealSpaceMask(grid=grid_radians)

        transformer = aa.TransformerDFT(
            uv_wavelengths=uv_wavelengths,
            real_space_mask=real_space_mask,
            preload_transform=False,
        )

        mapping_matrix = np.ones(shape=(2, 1))

        transformed_mapping_matrix = transformer.transformed_mapping_matrix_from_mapping_matrix(
            mapping_matrix=mapping_matrix
        )

        assert transformed_mapping_matrix == pytest.approx(
            np.array(
                [
                    [-0.091544 - 1.455060j],
                    [-0.733597 - 0.78120j],
                    [-0.613160 - 0.07746j],
                ]
            ),
            1.0e-4,
        )

        mapping_matrix = np.ones(shape=(2, 2))

        transformed_mapping_matrix = transformer.transformed_mapping_matrix_from_mapping_matrix(
            mapping_matrix=mapping_matrix
        )

        assert transformed_mapping_matrix == pytest.approx(
            np.array(
                [
                    [-0.091544 - 1.45506j, -0.091544 - 1.45506j],
                    [-0.733597 - 0.78120j, -0.733597 - 0.78120j],
                    [-0.61316 - 0.07746j, -0.61316 - 0.07746j],
                ]
            ),
            1.0e-4,
        )

    def test__visibilities__more_complex_mapping_matrix(self):

        grid_radians = aa.Grid.manual_2d(
            [[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]], pixel_scales=1.0
        )
        real_space_mask = MockRealSpaceMask(grid=grid_radians)

        uv_wavelengths = np.array([[0.7, 0.8], [0.9, 1.0]])

        transformer = aa.TransformerDFT(
            uv_wavelengths=uv_wavelengths,
            real_space_mask=real_space_mask,
            preload_transform=False,
        )

        mapping_matrix = np.array([[1.0], [0.0], [0.0]])

        transformed_mapping_matrix = transformer.transformed_mapping_matrix_from_mapping_matrix(
            mapping_matrix=mapping_matrix
        )

        assert transformed_mapping_matrix == pytest.approx(
            np.array([[0.18738 - 0.982287j], [-0.18738 - 0.982287j]]), 1.0e-4
        )

        mapping_matrix = np.array([[0.0], [1.0], [0.0]])

        transformed_mapping_matrix = transformer.transformed_mapping_matrix_from_mapping_matrix(
            mapping_matrix=mapping_matrix
        )

        assert transformed_mapping_matrix == pytest.approx(
            np.array([[-0.992111 + 0.12533j], [-0.53582 + 0.84432j]]), 1.0e-4
        )

        mapping_matrix = np.array([[0.0, 0.5], [0.0, 0.2], [1.0, 0.0]])

        transformed_mapping_matrix = transformer.transformed_mapping_matrix_from_mapping_matrix(
            mapping_matrix=mapping_matrix
        )

        assert transformed_mapping_matrix == pytest.approx(
            np.array(
                [
                    [0.42577 + 0.90482j, -0.10473 - 0.46607j],
                    [0.968583 - 0.24868j, -0.20085 - 0.32227j],
                ]
            ),
            1.0e-4,
        )

    def test__transformed_mapping_matrix__preload_and_non_preload_give_same_answer(
        self,
    ):

        uv_wavelengths = np.array([[0.2, 1.0], [0.5, 1.1], [0.8, 1.2]])
        grid_radians = aa.Grid.manual_2d(
            grid=[[[0.1, 0.2], [0.3, 0.4]]], pixel_scales=1.0
        )
        real_space_mask = MockRealSpaceMask(grid=grid_radians)

        transformer_preload = aa.TransformerDFT(
            uv_wavelengths=uv_wavelengths,
            real_space_mask=real_space_mask,
            preload_transform=True,
        )

        transformer = aa.TransformerDFT(
            uv_wavelengths=uv_wavelengths,
            real_space_mask=real_space_mask,
            preload_transform=False,
        )

        mapping_matrix = np.array([[3.0, 5.0], [1.0, 2.0]])

        transformed_mapping_matrix_preload = transformer_preload.transformed_mapping_matrix_from_mapping_matrix(
            mapping_matrix=mapping_matrix
        )

        transformed_mapping_matrix = transformer.transformed_mapping_matrix_from_mapping_matrix(
            mapping_matrix=mapping_matrix
        )

        assert (transformed_mapping_matrix_preload == transformed_mapping_matrix).all()


class TestTransformerNUFFT:
    def test__image_from_visibilities__same_as_direct__include_numerics(self):

        uv_wavelengths = np.array([[0.2, 1.0], [0.5, 1.1], [0.8, 1.2]])

        grid_radians = aa.Grid.uniform(shape_2d=(5, 5), pixel_scales=0.005).in_radians
        real_space_mask = MockRealSpaceMask(grid=grid_radians)

        image = aa.Array.ones(
            shape_2d=grid_radians.shape_2d, pixel_scales=grid_radians.pixel_scales
        )

        transformer_dft = aa.TransformerDFT(
            uv_wavelengths=uv_wavelengths,
            real_space_mask=real_space_mask,
            preload_transform=False,
        )

        visibilities_dft = transformer_dft.visibilities_from_image(image=image.in_2d)

        real_space_mask = aa.Mask2D.unmasked(shape_2d=(5, 5), pixel_scales=0.005)

        transformer_nufft = aa.TransformerNUFFT(
            uv_wavelengths=uv_wavelengths, real_space_mask=real_space_mask
        )

        visibilities_nufft = transformer_nufft.visibilities_from_image(
            image=image.in_2d
        )

        assert visibilities_dft == pytest.approx(visibilities_nufft, 2.0)
        assert visibilities_nufft[0] == pytest.approx(25.02317617953263 + 0.0j, 1.0e-7)

    def test__mapping_matix_from_visibilities__same_as_direct__include_numerics(self):

        uv_wavelengths = np.array([[0.2, 1.0], [0.5, 1.1], [0.8, 1.2]])

        grid_radians = aa.Grid.uniform(shape_2d=(5, 5), pixel_scales=0.005)
        real_space_mask = MockRealSpaceMask(grid=grid_radians)

        mapping_matrix = np.ones(shape=(25, 3))

        transformer_dft = aa.TransformerDFT(
            uv_wavelengths=uv_wavelengths,
            real_space_mask=real_space_mask,
            preload_transform=False,
        )

        transformed_mapping_matrix_dft = transformer_dft.transformed_mapping_matrix_from_mapping_matrix(
            mapping_matrix=mapping_matrix
        )

        real_space_mask = aa.Mask2D.unmasked(shape_2d=(5, 5), pixel_scales=0.005)

        transformer_nufft = aa.TransformerNUFFT(
            uv_wavelengths=uv_wavelengths, real_space_mask=real_space_mask
        )

        transformed_mapping_matrix_nufft = transformer_nufft.transformed_mapping_matrix_from_mapping_matrix(
            mapping_matrix=mapping_matrix
        )

        assert transformed_mapping_matrix_dft == pytest.approx(
            transformed_mapping_matrix_nufft, 2.0
        )
        assert transformed_mapping_matrix_dft == pytest.approx(
            transformed_mapping_matrix_nufft, 2.0
        )

        assert transformed_mapping_matrix_nufft[0, 0] == pytest.approx(
            25.02317 + 0.0j, 1.0e-4
        )
