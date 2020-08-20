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
    def test__real_visibilities__intensity_image_all_ones__simple_cases(self):

        uv_wavelengths = np.ones(shape=(4, 2))

        grid_radians = aa.Grid.manual_2d([[[1.0, 1.0]]], pixel_scales=1.0)

        real_space_mask = MockRealSpaceMask(grid=grid_radians)

        transformer = aa.TransformerDFT(
            uv_wavelengths=uv_wavelengths,
            real_space_mask=real_space_mask,
            preload_transform=False,
        )

        image = aa.Array.ones(shape_2d=(1, 1))

        real_visibilities = transformer.real_visibilities_from_image(image=image)

        assert (real_visibilities == np.ones(shape=4)).all()

        uv_wavelengths = np.array([[0.2, 1.0], [0.5, 1.1], [0.8, 1.2]])

        grid_radians = aa.Grid.manual_2d([[[0.1, 0.2], [0.3, 0.4]]], pixel_scales=1.0)

        real_space_mask = MockRealSpaceMask(grid=grid_radians)

        transformer = aa.TransformerDFT(
            uv_wavelengths=uv_wavelengths,
            real_space_mask=real_space_mask,
            preload_transform=False,
        )

        image = aa.Array.ones(shape_2d=(1, 2))

        real_visibilities = transformer.real_visibilities_from_image(image=image)

        assert real_visibilities == pytest.approx(
            np.array([-0.091544, -0.73359736, -0.613160]), 1.0e-4
        )

    def test__real_visibilities__intensity_image_varies__simple_cases(self):

        uv_wavelengths = np.ones(shape=(4, 2))
        grid_radians = aa.Grid.manual_2d([[[1.0, 1.0]]], pixel_scales=1.0)
        real_space_mask = MockRealSpaceMask(grid=grid_radians)

        transformer = aa.TransformerDFT(
            uv_wavelengths=uv_wavelengths,
            real_space_mask=real_space_mask,
            preload_transform=False,
        )

        image = aa.Array.manual_2d([[2.0]])

        real_visibilities = transformer.real_visibilities_from_image(image=image)

        assert (real_visibilities == np.array([2.0])).all()

        uv_wavelengths = np.array([[0.2, 1.0], [0.5, 1.1], [0.8, 1.2]])

        grid_radians = aa.Grid.manual_2d([[[0.1, 0.2], [0.3, 0.4]]], pixel_scales=1.0)
        real_space_mask = MockRealSpaceMask(grid=grid_radians)

        transformer = aa.TransformerDFT(
            uv_wavelengths=uv_wavelengths,
            real_space_mask=real_space_mask,
            preload_transform=False,
        )

        image = aa.Array.manual_2d([[3.0, 6.0]])

        real_visibilities = transformer.real_visibilities_from_image(image=image)

        assert real_visibilities == pytest.approx(
            np.array([-2.46153, -5.14765, -3.11681]), 1.0e-4
        )

    def test__real_visibilities__preload_and_non_preload_give_same_answer(self):

        uv_wavelengths = np.array([[0.2, 1.0], [0.5, 1.1], [0.8, 1.2]])
        grid_radians = aa.Grid.manual_2d([[[0.1, 0.2], [0.3, 0.4]]], pixel_scales=1.0)
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

        image = aa.Array.manual_2d([[2.0, 6.0]])

        real_visibilities_via_preload = transformer_preload.real_visibilities_from_image(
            image=image
        )
        real_visibilities = transformer.real_visibilities_from_image(image=image)

        assert (real_visibilities_via_preload == real_visibilities).all()

    def test__imag_visibilities__intensity_image_all_ones__simple_cases(self):

        uv_wavelengths = np.ones(shape=(4, 2))
        grid_radians = aa.Grid.manual_2d([[[1.0, 1.0]]], pixel_scales=1.0)
        real_space_mask = MockRealSpaceMask(grid=grid_radians)

        transformer = aa.TransformerDFT(
            uv_wavelengths=uv_wavelengths,
            real_space_mask=real_space_mask,
            preload_transform=False,
        )

        image = aa.Array.ones(shape_2d=(1, 1))

        imag_visibilities = transformer.imag_visibilities_from_image(image=image)

        assert imag_visibilities == pytest.approx(np.zeros(shape=4), 1.0e-4)

        uv_wavelengths = np.array([[0.2, 1.0], [0.5, 1.1], [0.8, 1.2]])

        grid_radians = aa.Grid.manual_2d([[[0.1, 0.2], [0.3, 0.4]]], pixel_scales=1.0)
        real_space_mask = MockRealSpaceMask(grid=grid_radians)

        transformer = aa.TransformerDFT(
            uv_wavelengths=uv_wavelengths,
            real_space_mask=real_space_mask,
            preload_transform=False,
        )

        image = aa.Array.ones(shape_2d=(2, 1))

        imag_visibilities = transformer.imag_visibilities_from_image(image=image)

        assert imag_visibilities == pytest.approx(
            np.array([-1.45506, -0.781201, -0.077460]), 1.0e-4
        )

    def test__imag_visibilities__intensity_image_varies__simple_cases(self):

        uv_wavelengths = np.ones(shape=(4, 2))
        grid_radians = aa.Grid.manual_2d([[[1.0, 1.0]]], pixel_scales=1.0)
        real_space_mask = MockRealSpaceMask(grid=grid_radians)

        transformer = aa.TransformerDFT(
            uv_wavelengths=uv_wavelengths,
            real_space_mask=real_space_mask,
            preload_transform=False,
        )

        image = aa.Array.manual_2d([[2.0]])

        imag_visibilities = transformer.imag_visibilities_from_image(image=image)

        assert imag_visibilities == pytest.approx(np.zeros((4,)), 1.0e-4)

        uv_wavelengths = np.array([[0.2, 1.0], [0.5, 1.1], [0.8, 1.2]])

        grid_radians = aa.Grid.manual_2d([[[0.1, 0.2], [0.3, 0.4]]], pixel_scales=1.0)
        real_space_mask = MockRealSpaceMask(grid=grid_radians)

        transformer = aa.TransformerDFT(
            uv_wavelengths=uv_wavelengths,
            real_space_mask=real_space_mask,
            preload_transform=False,
        )

        image = aa.Array.manual_2d([[3.0, 6.0]])

        imag_visibilities = transformer.imag_visibilities_from_image(image=image)

        assert imag_visibilities == pytest.approx(
            np.array([-6.418822, -1.78146, 2.48210]), 1.0e-4
        )

    def test__imag_visibilities__preload_and_non_preload_give_same_answer(self):

        uv_wavelengths = np.array([[0.2, 1.0], [0.5, 1.1], [0.8, 1.2]])
        grid_radians = aa.Grid.manual_2d([[[0.1, 0.2], [0.3, 0.4]]], pixel_scales=1.0)
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

        image = aa.Array.manual_2d([[2.0, 6.0]])

        imag_visibilities_via_preload = transformer_preload.imag_visibilities_from_image(
            image=image
        )
        imag_visibilities = transformer.imag_visibilities_from_image(image=image)

        assert (imag_visibilities_via_preload == imag_visibilities).all()

    def test__visiblities_from_image__same_as_individual_calculations_above(self):

        uv_wavelengths = np.array([[0.2, 1.0], [0.5, 1.1], [0.8, 1.2]])

        grid_radians = aa.Grid.manual_2d([[[0.1, 0.2], [0.3, 0.4]]], pixel_scales=1.0)
        real_space_mask = MockRealSpaceMask(grid=grid_radians)

        transformer = aa.TransformerDFT(
            uv_wavelengths=uv_wavelengths,
            real_space_mask=real_space_mask,
            preload_transform=False,
        )

        image = aa.Array.manual_2d([[3.0, 6.0]])

        visibilities = transformer.visibilities_from_image(image=image)

        assert visibilities[:, 0] == pytest.approx(
            np.array([-2.46153, -5.14765, -3.11681]), 1.0e-4
        )
        assert visibilities[:, 1] == pytest.approx(
            np.array([-6.418822, -1.78146, 2.48210]), 1.0e-4
        )

        real_visibilities = transformer.real_visibilities_from_image(image=image)
        imag_visibilities = transformer.imag_visibilities_from_image(image=image)

        assert (visibilities[:, 0] == real_visibilities).all()
        assert (visibilities[:, 1] == imag_visibilities).all()


class TestVisiblitiesMappingMatrix:
    def test__real_visibilities__mapping_matrix_all_ones__simple_cases(self):

        uv_wavelengths = np.ones(shape=(4, 2))
        grid_radians = aa.Grid.manual_2d([[[1.0, 1.0]]], pixel_scales=1.0)
        real_space_mask = MockRealSpaceMask(grid=grid_radians)

        transformer = aa.TransformerDFT(
            uv_wavelengths=uv_wavelengths,
            real_space_mask=real_space_mask,
            preload_transform=False,
        )

        mapping_matrix = np.ones(shape=(1, 1))

        transformed_mapping_matrix = transformer.real_transformed_mapping_matrix_from_mapping_matrix(
            mapping_matrix=mapping_matrix
        )

        assert (transformed_mapping_matrix == np.ones(shape=(4, 1))).all()

        uv_wavelengths = np.array([[0.2, 1.0], [0.5, 1.1], [0.8, 1.2]])

        grid_radians = aa.Grid.manual_2d([[[0.1, 0.2], [0.3, 0.4]]], pixel_scales=1.0)
        real_space_mask = MockRealSpaceMask(grid=grid_radians)

        transformer = aa.TransformerDFT(
            uv_wavelengths=uv_wavelengths,
            real_space_mask=real_space_mask,
            preload_transform=False,
        )

        mapping_matrix = np.ones(shape=(2, 1))

        transformed_mapping_matrix = transformer.real_transformed_mapping_matrix_from_mapping_matrix(
            mapping_matrix=mapping_matrix
        )

        assert transformed_mapping_matrix == pytest.approx(
            np.array([[-0.091544], [-0.733597], [-0.613160]]), 1.0e-4
        )

        mapping_matrix = np.ones(shape=(2, 2))

        transformed_mapping_matrix = transformer.real_transformed_mapping_matrix_from_mapping_matrix(
            mapping_matrix=mapping_matrix
        )

        assert transformed_mapping_matrix == pytest.approx(
            np.array(
                [[-0.091544, -0.091544], [-0.733597, -0.733597], [-0.61316, -0.61316]]
            ),
            1.0e-4,
        )

    def test__real_visibilities__more_complex_mapping_matrix(self):

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

        transformed_mapping_matrix = transformer.real_transformed_mapping_matrix_from_mapping_matrix(
            mapping_matrix=mapping_matrix
        )

        assert transformed_mapping_matrix == pytest.approx(
            np.array([[0.18738], [-0.18738]]), 1.0e-4
        )

        mapping_matrix = np.array([[0.0], [1.0], [0.0]])

        transformed_mapping_matrix = transformer.real_transformed_mapping_matrix_from_mapping_matrix(
            mapping_matrix=mapping_matrix
        )

        assert transformed_mapping_matrix == pytest.approx(
            np.array([[-0.992111], [-0.53582]]), 1.0e-4
        )

        mapping_matrix = np.array([[0.0, 0.5], [0.0, 0.2], [1.0, 0.0]])

        transformed_mapping_matrix = transformer.real_transformed_mapping_matrix_from_mapping_matrix(
            mapping_matrix=mapping_matrix
        )

        assert transformed_mapping_matrix == pytest.approx(
            np.array([[0.42577, -0.10473], [0.968583, -0.20085]]), 1.0e-4
        )

    def test__real_visibilities__preload_and_non_preload_give_same_answer(self):

        uv_wavelengths = np.array([[0.2, 1.0], [0.5, 1.1], [0.8, 1.2]])
        grid_radians = aa.Grid.manual_2d([[[0.1, 0.2], [0.3, 0.4]]], pixel_scales=1.0)
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

        transformed_mapping_matrix_preload = transformer_preload.real_transformed_mapping_matrix_from_mapping_matrix(
            mapping_matrix=mapping_matrix
        )

        transformed_mapping_matrix = transformer.real_transformed_mapping_matrix_from_mapping_matrix(
            mapping_matrix=mapping_matrix
        )

        assert (transformed_mapping_matrix_preload == transformed_mapping_matrix).all()

    def test__imag_visibilities__mapping_matrix_all_ones__simple_cases(self):

        uv_wavelengths = np.ones(shape=(4, 2))
        grid_radians = aa.Grid.manual_2d([[[1.0, 1.0]]], pixel_scales=1.0)
        real_space_mask = MockRealSpaceMask(grid=grid_radians)

        transformer = aa.TransformerDFT(
            uv_wavelengths=uv_wavelengths,
            real_space_mask=real_space_mask,
            preload_transform=False,
        )

        mapping_matrix = np.ones(shape=(1, 1))

        transformed_mapping_matrix = transformer.imag_transformed_mapping_matrix_from_mapping_matrix(
            mapping_matrix=mapping_matrix
        )

        assert transformed_mapping_matrix == pytest.approx(
            np.zeros(shape=(4, 1)), 1.0e-4
        )

        uv_wavelengths = np.array([[0.2, 1.0], [0.5, 1.1], [0.8, 1.2]])
        grid_radians = aa.Grid.manual_2d([[[0.1, 0.2], [0.3, 0.4]]], pixel_scales=1.0)
        real_space_mask = MockRealSpaceMask(grid=grid_radians)

        transformer = aa.TransformerDFT(
            uv_wavelengths=uv_wavelengths,
            real_space_mask=real_space_mask,
            preload_transform=False,
        )

        mapping_matrix = np.ones(shape=(2, 1))

        transformed_mapping_matrix = transformer.imag_transformed_mapping_matrix_from_mapping_matrix(
            mapping_matrix=mapping_matrix
        )

        assert transformed_mapping_matrix == pytest.approx(
            np.array([[-1.455060], [-0.78120], [-0.07746]]), 1.0e-4
        )

        mapping_matrix = np.ones(shape=(2, 2))

        transformed_mapping_matrix = transformer.imag_transformed_mapping_matrix_from_mapping_matrix(
            mapping_matrix=mapping_matrix
        )

        assert transformed_mapping_matrix == pytest.approx(
            np.array(
                [[-1.45506, -1.45506], [-0.78120, -0.78120], [-0.07746, -0.07746]]
            ),
            1.0e-4,
        )

    def test__imag_visibilities__more_complex_mapping_matrix(self):

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

        transformed_mapping_matrix = transformer.imag_transformed_mapping_matrix_from_mapping_matrix(
            mapping_matrix=mapping_matrix
        )

        assert transformed_mapping_matrix == pytest.approx(
            np.array([[-0.982287], [-0.982287]]), 1.0e-4
        )

        mapping_matrix = np.array([[0.0], [1.0], [0.0]])

        transformed_mapping_matrix = transformer.imag_transformed_mapping_matrix_from_mapping_matrix(
            mapping_matrix=mapping_matrix
        )

        assert transformed_mapping_matrix == pytest.approx(
            np.array([[0.12533], [0.84432]]), 1.0e-4
        )

        mapping_matrix = np.array([[0.0, 0.5], [0.0, 0.2], [1.0, 0.0]])

        transformed_mapping_matrix = transformer.imag_transformed_mapping_matrix_from_mapping_matrix(
            mapping_matrix=mapping_matrix
        )

        assert transformed_mapping_matrix == pytest.approx(
            np.array([[0.90482, -0.46607], [-0.24868, -0.32227]]), 1.0e-4
        )

    def test__imag_visibilities__preload_and_non_preload_give_same_answer(self):

        uv_wavelengths = np.array([[0.2, 1.0], [0.5, 1.1], [0.8, 1.2]])
        grid_radians = aa.Grid.manual_2d([[[0.1, 0.2], [0.3, 0.4]]], pixel_scales=1.0)
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

        transformed_mapping_matrix_preload = transformer_preload.imag_transformed_mapping_matrix_from_mapping_matrix(
            mapping_matrix=mapping_matrix
        )

        transformed_mapping_matrix = transformer.imag_transformed_mapping_matrix_from_mapping_matrix(
            mapping_matrix=mapping_matrix
        )

        assert (transformed_mapping_matrix_preload == transformed_mapping_matrix).all()

    def test__transformed_mapping_matrices_from_mapping_matrix__same_as_individual_calculations_above(
        self
    ):

        uv_wavelengths = np.array([[0.2, 1.0], [0.5, 1.1], [0.8, 1.2]])
        grid_radians = aa.Grid.manual_2d([[[0.1, 0.2], [0.3, 0.4]]], pixel_scales=1.0)
        real_space_mask = MockRealSpaceMask(grid=grid_radians)

        transformer = aa.TransformerDFT(
            uv_wavelengths=uv_wavelengths,
            real_space_mask=real_space_mask,
            preload_transform=False,
        )

        mapping_matrix = np.ones(shape=(2, 1))

        transformed_mapping_matrices = transformer.transformed_mapping_matrices_from_mapping_matrix(
            mapping_matrix=mapping_matrix
        )

        assert transformed_mapping_matrices[0] == pytest.approx(
            np.array([[-0.09154], [-0.73359], [-0.61316]]), 1.0e-4
        )
        assert transformed_mapping_matrices[1] == pytest.approx(
            np.array([[-1.45506], [-0.78120], [-0.07746]]), 1.0e-4
        )


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

        real_space_mask = aa.Mask.unmasked(shape_2d=(5, 5), pixel_scales=0.005)

        transformer_nufft = aa.TransformerNUFFT(
            uv_wavelengths=uv_wavelengths, real_space_mask=real_space_mask
        )

        visibilities_nufft = transformer_nufft.visibilities_from_image(
            image=image.in_2d
        )

        assert visibilities_dft == pytest.approx(visibilities_nufft, 1.0e-2)

        assert visibilities_nufft[0, 0] == pytest.approx(25.02317617953263, 1.0e-7)

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

        transformed_mapping_matrices_dft = transformer_dft.transformed_mapping_matrices_from_mapping_matrix(
            mapping_matrix=mapping_matrix
        )

        real_space_mask = aa.Mask.unmasked(shape_2d=(5, 5), pixel_scales=0.005)

        transformer_nufft = aa.TransformerNUFFT(
            uv_wavelengths=uv_wavelengths, real_space_mask=real_space_mask
        )

        transformed_mapping_matrices_nufft = transformer_nufft.transformed_mapping_matrices_from_mapping_matrix(
            mapping_matrix=mapping_matrix
        )

        assert transformed_mapping_matrices_dft[0] == pytest.approx(
            transformed_mapping_matrices_nufft[0], 1.0e-2
        )
        assert transformed_mapping_matrices_dft[1] == pytest.approx(
            transformed_mapping_matrices_nufft[1], 1.0e-2
        )

        assert transformed_mapping_matrices_nufft[0][0, 0] == pytest.approx(
            24.99999636, 1.0e-2
        )
        assert transformed_mapping_matrices_nufft[1][0, 0] == pytest.approx(0.0, 1.0e-2)
