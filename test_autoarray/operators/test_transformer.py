import autoarray as aa

import numpy as np
import pytest


class MockDeriveMask2D:
    def __init__(self, grid):
        self.mask = grid.derive_mask.all_false
        self.grid = grid


    @property
    def sub_1(self):
        return self

    @property
    def derive_grid(self):
        return MockDeriveGrid2D(
            grid=self.grid,
        )


class MockDeriveGrid2D:
    def __init__(self, grid):
        self.unmasked = MockMaskedGrid(grid=grid)


class MockRealSpaceMask:
    def __init__(self, grid):
        self.grid = grid
        self.unmasked = MockMaskedGrid(grid=grid)

    @property
    def pixels_in_mask(self):
        return self.unmasked.slim.in_radians.shape[0]

    @property
    def derive_mask(self):
        return MockDeriveMask2D(
            grid=self.grid,
        )

    @property
    def derive_grid(self):
        return MockDeriveGrid2D(
            grid=self.grid,
        )

    @property
    def pixel_scales(self):
        return self.grid.pixel_scales

    @property
    def origin(self):
        return self.grid.origin


class MockMaskedGrid:
    def __init__(self, grid):

        self.in_radians = grid
        self.slim = grid



def test__dft__visibilities_from():
    uv_wavelengths = np.ones(shape=(4, 2))

    grid_radians = aa.Grid2D.no_mask(values=[[[1.0, 1.0]]], pixel_scales=1.0)

    real_space_mask = MockRealSpaceMask(grid=grid_radians)

    transformer = aa.TransformerDFT(
        uv_wavelengths=uv_wavelengths,
        real_space_mask=real_space_mask,
        preload_transform=False,
    )

    image = aa.Array2D.ones(shape_native=(1, 1), pixel_scales=1.0)

    visibilities = transformer.visibilities_from(image=image)

    assert visibilities == pytest.approx(
        np.array([1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j, 1.0 + 0.0j]), 1.0e-4
    )

    uv_wavelengths = np.array([[0.2, 1.0], [0.5, 1.1], [0.8, 1.2]])

    grid_radians = aa.Grid2D.no_mask(
        values=[[[0.1, 0.2], [0.3, 0.4]]], pixel_scales=1.0
    )

    real_space_mask = MockRealSpaceMask(grid=grid_radians)

    transformer = aa.TransformerDFT(
        uv_wavelengths=uv_wavelengths,
        real_space_mask=real_space_mask,
        preload_transform=False,
    )

    image = aa.Array2D.ones(shape_native=(1, 2), pixel_scales=1.0)

    visibilities = transformer.visibilities_from(image=image)

    assert visibilities == pytest.approx(
        np.array(
            [-0.091544 - 1.45506j, -0.73359736 - 0.781201j, -0.613160 - 0.077460j]
        ),
        1.0e-4,
    )

    uv_wavelengths = np.array([[0.2, 1.0], [0.5, 1.1], [0.8, 1.2]])

    grid_radians = aa.Grid2D.no_mask(
        values=[[[0.1, 0.2], [0.3, 0.4]]], pixel_scales=1.0
    )
    real_space_mask = MockRealSpaceMask(grid=grid_radians)

    transformer = aa.TransformerDFT(
        uv_wavelengths=uv_wavelengths,
        real_space_mask=real_space_mask,
        preload_transform=False,
    )

    image = aa.Array2D.no_mask([[3.0, 6.0]], pixel_scales=1.0)

    visibilities = transformer.visibilities_from(image=image)

    assert visibilities == pytest.approx(
        np.array([-2.46153 - 6.418822j, -5.14765 - 1.78146j, -3.11681 + 2.48210j]),
        1.0e-4,
    )


def test__dft__visibilities_from__preload_and_non_preload_give_same_answer():
    uv_wavelengths = np.array([[0.2, 1.0], [0.5, 1.1], [0.8, 1.2]])
    grid_radians = aa.Grid2D.no_mask(
        values=[[[0.1, 0.2], [0.3, 0.4]]], pixel_scales=1.0
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

    image = aa.Array2D.no_mask([[2.0, 6.0]], pixel_scales=1.0)

    visibilities_via_preload = transformer_preload.visibilities_from(image=image)
    visibilities = transformer.visibilities_from(image=image)

    assert (visibilities_via_preload == visibilities).all()


def test__dft__transform_mapping_matrix():
    uv_wavelengths = np.ones(shape=(4, 2))
    grid_radians = aa.Grid2D.no_mask(values=[[[1.0, 1.0]]], pixel_scales=1.0)
    real_space_mask = MockRealSpaceMask(grid=grid_radians)

    transformer = aa.TransformerDFT(
        uv_wavelengths=uv_wavelengths,
        real_space_mask=real_space_mask,
        preload_transform=False,
    )

    mapping_matrix = np.ones(shape=(1, 1))

    transformed_mapping_matrix = transformer.transform_mapping_matrix(
        mapping_matrix=mapping_matrix
    )

    assert transformed_mapping_matrix == pytest.approx(
        np.array([[1.0 + 0.0j], [1.0 + 0.0j], [1.0 + 0.0j], [1.0 + 0.0j]]), 1.0e-4
    )

    uv_wavelengths = np.array([[0.2, 1.0], [0.5, 1.1], [0.8, 1.2]])

    grid_radians = aa.Grid2D.no_mask(
        values=[[[0.1, 0.2], [0.3, 0.4]]], pixel_scales=1.0
    )
    real_space_mask = MockRealSpaceMask(grid=grid_radians)

    transformer = aa.TransformerDFT(
        uv_wavelengths=uv_wavelengths,
        real_space_mask=real_space_mask,
        preload_transform=False,
    )

    mapping_matrix = np.ones(shape=(2, 2))

    transformed_mapping_matrix = transformer.transform_mapping_matrix(
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

    grid_radians = aa.Grid2D.no_mask(
        [[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]]], pixel_scales=1.0
    )
    real_space_mask = MockRealSpaceMask(grid=grid_radians)

    uv_wavelengths = np.array([[0.7, 0.8], [0.9, 1.0]])

    transformer = aa.TransformerDFT(
        uv_wavelengths=uv_wavelengths,
        real_space_mask=real_space_mask,
        preload_transform=False,
    )

    mapping_matrix = np.array([[0.0, 0.5], [0.0, 0.2], [1.0, 0.0]])

    transformed_mapping_matrix = transformer.transform_mapping_matrix(
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


def test__dft__transformed_mapping_matrix__preload_and_non_preload_give_same_answer():
    uv_wavelengths = np.array([[0.2, 1.0], [0.5, 1.1], [0.8, 1.2]])
    grid_radians = aa.Grid2D.no_mask(
        values=[[[0.1, 0.2], [0.3, 0.4]]], pixel_scales=1.0
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

    transformed_mapping_matrix_preload = transformer_preload.transform_mapping_matrix(
        mapping_matrix=mapping_matrix
    )

    transformed_mapping_matrix = transformer.transform_mapping_matrix(
        mapping_matrix=mapping_matrix
    )

    assert (transformed_mapping_matrix_preload == transformed_mapping_matrix).all()


def test__nufft__visibilities_from():
    uv_wavelengths = np.array([[0.2, 1.0], [0.5, 1.1], [0.8, 1.2]])

    grid_radians = aa.Grid2D.uniform(shape_native=(5, 5), pixel_scales=0.005).in_radians
    real_space_mask = MockRealSpaceMask(grid=grid_radians)

    image = aa.Array2D.ones(
        shape_native=grid_radians.shape_native,
        pixel_scales=grid_radians.pixel_scales,
    )

    transformer_dft = aa.TransformerDFT(
        uv_wavelengths=uv_wavelengths,
        real_space_mask=real_space_mask,
        preload_transform=False,
    )

    visibilities_dft = transformer_dft.visibilities_from(image=image.native)

    real_space_mask = aa.Mask2D.all_false(shape_native=(5, 5), pixel_scales=0.005)

    transformer_nufft = aa.TransformerNUFFT(
        uv_wavelengths=uv_wavelengths, real_space_mask=real_space_mask
    )

    visibilities_nufft = transformer_nufft.visibilities_from(image=image.native)

    assert visibilities_dft == pytest.approx(visibilities_nufft, 2.0)
    assert visibilities_nufft[0] == pytest.approx(25.02317617953263 + 0.0j, 1.0e-7)


def test__nufft__transform_mapping_matrix():
    uv_wavelengths = np.array([[0.2, 1.0], [0.5, 1.1], [0.8, 1.2]])

    grid_radians = aa.Grid2D.uniform(shape_native=(5, 5), pixel_scales=0.005)
    real_space_mask = MockRealSpaceMask(grid=grid_radians)

    mapping_matrix = np.ones(shape=(25, 3))

    transformer_dft = aa.TransformerDFT(
        uv_wavelengths=uv_wavelengths,
        real_space_mask=real_space_mask,
        preload_transform=False,
    )

    transformed_mapping_matrix_dft = transformer_dft.transform_mapping_matrix(
        mapping_matrix=mapping_matrix
    )

    real_space_mask = aa.Mask2D.all_false(shape_native=(5, 5), pixel_scales=0.005)

    transformer_nufft = aa.TransformerNUFFT(
        uv_wavelengths=uv_wavelengths, real_space_mask=real_space_mask
    )

    transformed_mapping_matrix_nufft = transformer_nufft.transform_mapping_matrix(
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
