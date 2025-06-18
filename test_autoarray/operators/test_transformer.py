import autoarray as aa

import numpy as np
import pytest


def test__dft__visibilities_from(visibilities_7, uv_wavelengths_7x2, mask_2d_7x7):

    transformer = aa.TransformerDFT(
        uv_wavelengths=uv_wavelengths_7x2,
        real_space_mask=mask_2d_7x7,
        preload_transform=False,
    )

    image = aa.Array2D(
        values=[
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        mask=mask_2d_7x7,
    )

    visibilities = transformer.visibilities_from(image=image)

    assert visibilities[0:3] == pytest.approx(
        np.array(
            [
                -0.06434514 - 0.61763293j,
                1.71143349 - 1.184022j,
                0.90200541 + 0.03726693j,
            ]
        ),
        1.0e-4,
    )


def test__dft__image_from(visibilities_7, uv_wavelengths_7x2, mask_2d_7x7):

    transformer = aa.TransformerDFT(
        uv_wavelengths=uv_wavelengths_7x2,
        real_space_mask=mask_2d_7x7,
        preload_transform=False,
    )

    image = transformer.image_from(visibilities=visibilities_7)

    assert image[0:3] == pytest.approx([-1.49022481, -0.22395855, -0.45588535], 1.0e-4)


def test__dft__visibilities_from__preload_and_non_preload_give_same_answer(
    visibilities_7, uv_wavelengths_7x2, mask_2d_7x7
):

    transformer_preload = aa.TransformerDFT(
        uv_wavelengths=uv_wavelengths_7x2,
        real_space_mask=mask_2d_7x7,
        preload_transform=True,
    )
    transformer = aa.TransformerDFT(
        uv_wavelengths=uv_wavelengths_7x2,
        real_space_mask=mask_2d_7x7,
        preload_transform=False,
    )

    image = aa.Array2D(
        values=[
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.5, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        mask=mask_2d_7x7,
    )

    visibilities_via_preload = transformer_preload.visibilities_from(image=image)
    visibilities = transformer.visibilities_from(image=image)

    assert visibilities_via_preload == pytest.approx(visibilities, 1.0e-4)


def test__dft__transform_mapping_matrix(
    visibilities_7, uv_wavelengths_7x2, mask_2d_7x7
):

    transformer = aa.TransformerDFT(
        uv_wavelengths=uv_wavelengths_7x2,
        real_space_mask=mask_2d_7x7,
        preload_transform=False,
    )

    mapping_matrix = np.ones(shape=(1, 1))

    transformed_mapping_matrix = transformer.transform_mapping_matrix(
        mapping_matrix=mapping_matrix
    )

    assert transformed_mapping_matrix[0:3, :] == pytest.approx(
        np.array(
            [
                [0.80682556 - 0.59078974j],
                [-0.19648896 - 0.98050604j],
                [-0.47002763 - 0.8826517j],
            ]
        ),
        1.0e-4,
    )


def test__dft__transformed_mapping_matrix__preload_and_non_preload_give_same_answer(
    visibilities_7, uv_wavelengths_7x2, mask_2d_7x7
):

    transformer_preload = aa.TransformerDFT(
        uv_wavelengths=uv_wavelengths_7x2,
        real_space_mask=mask_2d_7x7,
        preload_transform=True,
    )

    transformer = aa.TransformerDFT(
        uv_wavelengths=uv_wavelengths_7x2,
        real_space_mask=mask_2d_7x7,
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
    real_space_mask = aa.Mask2D.all_false(shape_native=(5, 5), pixel_scales=0.005)

    image = aa.Array2D.ones(
        shape_native=(5, 5),
        pixel_scales=0.005,
    )

    transformer_nufft = aa.TransformerNUFFT(
        uv_wavelengths=uv_wavelengths, real_space_mask=real_space_mask
    )

    visibilities_nufft = transformer_nufft.visibilities_from(image=image.native)

    assert visibilities_nufft[0] == pytest.approx(25.02317617953263 + 0.0j, 1.0e-7)


def test__nufft__image_from(visibilities_7, uv_wavelengths_7x2, mask_2d_7x7):

    transformer = aa.TransformerNUFFT(
        uv_wavelengths=uv_wavelengths_7x2,
        real_space_mask=mask_2d_7x7,
    )

    image = transformer.image_from(visibilities=visibilities_7)

    assert image[0:3] == pytest.approx([0.00726546, 0.01149121, 0.01421022], 1.0e-4)


def test__nufft__transform_mapping_matrix():
    uv_wavelengths = np.array([[0.2, 1.0], [0.5, 1.1], [0.8, 1.2]])

    mapping_matrix = np.ones(shape=(25, 3))

    real_space_mask = aa.Mask2D.all_false(shape_native=(5, 5), pixel_scales=0.005)

    transformer_nufft = aa.TransformerNUFFT(
        uv_wavelengths=uv_wavelengths, real_space_mask=real_space_mask
    )

    transformed_mapping_matrix_nufft = transformer_nufft.transform_mapping_matrix(
        mapping_matrix=mapping_matrix
    )

    assert transformed_mapping_matrix_nufft[0, 0] == pytest.approx(
        25.02317 + 0.0j, 1.0e-4
    )
