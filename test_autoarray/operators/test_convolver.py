from astropy import units
from astropy.modeling import functional_models
from astropy.coordinates import Angle
import numpy as np
import pytest
from os import path

import autoarray as aa

test_data_path = path.join("{}".format(path.dirname(path.realpath(__file__))), "files")


def test__no_blur():
    convolver = aa.Convolver.no_blur()

    assert (
        convolver.kernel
        == np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    ).all()
    assert convolver.kernelpixel_scales == (1.0, 1.0)

    convolver = aa.Convolver.no_blur(pixel_scales=2.0)

    assert (
        convolver.kernel
        == np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    ).all()
    assert convolver.kernelpixel_scales == (2.0, 2.0)


def test__from_gaussian():
    
    convolver = aa.Convolver.from_gaussian(
        shape_native=(3, 3),
        pixel_scales=1.0,
        centre=(0.1, 0.1),
        axis_ratio=0.9,
        angle=45.0,
        sigma=1.0,
        normalize=True,
    )

    assert convolver.kernel == pytest.approx(
        np.array(
            [
                [0.06281, 0.13647, 0.0970],
                [0.11173, 0.21589, 0.136477],
                [0.065026, 0.11173, 0.06281],
            ]
        ),
        1.0e-3,
    )


def test__manual__normalize():

    kernel_data = np.ones((3, 3)) / 9.0
    convolver = aa.Convolver(
        kernel=kernel_data, normalize=True
    )

    assert convolver.kernel == pytest.approx(kernel_data, 1e-3)

    kernel_data = np.ones((3, 3))

    convolver = aa.Convolver(
        kernel=kernel_data, normalize=True
    )

    assert convolver.kernel == pytest.approx(np.ones((3, 3)) / 9.0, 1e-3)

    convolver = aa.Convolver(
        kernel=kernel_data, normalize=False
    )

    convolver = convolver.kernelnormalized

    assert convolver.kernel == pytest.approx(np.ones((3, 3)) / 9.0, 1e-3)

    kernel_data = np.ones((3, 3))

    convolver = aa.Convolver(
        kernel=kernel_data, normalize=False
    )

    assert convolver.kernel == pytest.approx(np.ones((3, 3)), 1e-3)


def test__convolved_image_from():

    mask = aa.Mask2D.circular(
        shape_native=(30, 30), pixel_scales=(1.0, 1.0), radius=4.0
    )

    import scipy.signal

    kernel = np.arange(49).reshape(7, 7)
    image = np.arange(900).reshape(30, 30)

    blurred_image_via_scipy = scipy.signal.convolve2d(image, kernel, mode="same")
    blurred_image_via_scipy = aa.Array2D.no_mask(
        kernel=blurred_image_via_scipy
    )
    blurred_masked_image_via_scipy = aa.Array2D(
        kernel=blurred_image_via_scipy.native, mask=mask
    )

    # Now reproduce this data using the convolved_image_from function

    image = aa.Array2D.no_mask(kernel=np.arange(900).reshape(30, 30))
    kernel = aa.Convolver(kernel=np.arange(49).reshape(7, 7))

    masked_image = aa.Array2D(kernel=image.native, mask=mask)

    blurring_mask = mask.derive_mask.blurring_from(
        kernel_shape_native=kernel.shape_native
    )

    blurring_image = aa.Array2D(kernel=image.native, mask=blurring_mask)

    blurred_masked_im_1 = kernel.convolved_image_from(
        image=masked_image, blurring_image=blurring_image
    )

    assert blurred_masked_image_via_scipy == pytest.approx(
        blurred_masked_im_1.array, 1e-4
    )


def test__convolve_imaged_from__no_blurring():
    # Setup a blurred data, using the PSF to perform the convolution in 2D, then masks it to make a 1d array.

    mask = aa.Mask2D.circular(
        shape_native=(30, 30), pixel_scales=(1.0, 1.0), radius=4.0
    )

    import scipy.signal

    kernel = np.arange(49).reshape(7, 7)
    image = np.arange(900).reshape(30, 30)

    blurring_mask = mask.derive_mask.blurring_from(kernel_shape_native=kernel.shape)
    blurred_image_via_scipy = scipy.signal.convolve2d(
        image * blurring_mask, kernel, mode="same"
    )
    blurred_image_via_scipy = aa.Array2D.no_mask(
        kernel=blurred_image_via_scipy
    )
    blurred_masked_image_via_scipy = aa.Array2D(
        kernel=blurred_image_via_scipy.native, mask=mask
    )

    # Now reproduce this data using the frame convolver_image

    kernel = aa.Convolver(kernel=np.arange(49).reshape(7, 7))
    image = aa.Array2D.no_mask(kernel=np.arange(900).reshape(30, 30))

    masked_image = aa.Array2D(kernel=image.native, mask=mask)

    blurred_masked_im_1 = kernel.convolved_image_from(
        image=masked_image, blurring_image=None
    )

    assert blurred_masked_image_via_scipy == pytest.approx(
        blurred_masked_im_1.array, 1e-4
    )


def test__convolved_mapping_matrix_from():
    mask = aa.Mask2D(
        mask=np.array(
            [
                [True, True, True, True, True, True],
                [True, False, False, False, False, True],
                [True, False, False, False, False, True],
                [True, False, False, False, False, True],
                [True, False, False, False, False, True],
                [True, True, True, True, True, True],
            ]
        ),
        pixel_scales=1.0,
    )

    kernel = aa.Convolver(
        kernel=[[0, 0.0, 0], [0.4, 0.2, 0.3], [0, 0.1, 0]]
    )

    mapping = np.array(
        [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [
                0,
                1,
                0,
            ],  # The 0.3 should be 'chopped' from this pixel as it is on the right-most edge
            [0, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )

    blurred_mapping = kernel.convolved_mapping_matrix_from(mapping, mask)

    assert (
        blurred_mapping
        == pytest.approx(
            np.array(
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0.4, 0],
                    [0, 0.2, 0],
                    [0.4, 0, 0],
                    [0.2, 0, 0.4],
                    [0.3, 0, 0.2],
                    [0, 0.1, 0.3],
                    [0, 0, 0],
                    [0.1, 0, 0],
                    [0, 0, 0.1],
                    [0, 0, 0],
                ]
            )
        ),
        1.0e-4,
    )

    kernel = aa.Convolver(
        kernel=[[0, 0.0, 0], [0.4, 0.2, 0.3], [0, 0.1, 0]]
    )

    mapping = np.array(
        [
            [0, 1, 0],
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [
                0,
                1,
                0,
            ],  # The 0.3 should be 'chopped' from this pixel as it is on the right-most edge
            [1, 0, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
        ]
    )

    blurred_mapping = kernel.convolved_mapping_matrix_from(mapping, mask)

    assert blurred_mapping == pytest.approx(
        np.array(
            [
                [0, 0.6, 0],
                [0, 0.9, 0],
                [0, 0.5, 0],
                [0, 0.3, 0],
                [0, 0.1, 0],
                [0, 0.1, 0],
                [0, 0.5, 0],
                [0, 0.2, 0],
                [0.6, 0, 0],
                [0.5, 0, 0.4],
                [0.3, 0, 0.2],
                [0, 0.1, 0.3],
                [0.1, 0, 0],
                [0.1, 0, 0],
                [0, 0, 0.1],
                [0, 0, 0],
            ]
        ),
        abs=1e-4,
    )


def test__convolve_imaged_from__via_fft__sizes_not_precomputed__compare_numerical_value():

    # -------------------------------
    # Case 1: direct image convolution
    # -------------------------------
    mask = aa.Mask2D.circular(
        shape_native=(20, 20), pixel_scales=(1.0, 1.0), radius=5.0
    )

    image = aa.Array2D.no_mask(kernel=np.arange(400).reshape(20, 20))
    masked_image = aa.Array2D(kernel=image.native, mask=mask)

    kernel_fft = aa.Convolver(
        kernel=np.arange(49).reshape(7, 7),
        pixel_scales=1.0,
        use_fft=True,
        normalize=True,
    )

    blurring_mask = mask.derive_mask.blurring_from(
        kernel_shape_native=kernel_fft.shape_native
    )
    blurring_image = aa.Array2D(kernel=image.native, mask=blurring_mask)

    blurred_fft = kernel_fft.convolved_image_from(
        image=masked_image, blurring_image=blurring_image
    )

    assert blurred_fft.native.array[13, 13] == pytest.approx(249.5, abs=1e-6)
