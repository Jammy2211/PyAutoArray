import os

import numpy as np
import pytest

from astropy import units
from astropy.modeling import functional_models
from astropy.coordinates import Angle
import autoarray as aa
from autoarray import exc

test_data_dir = "{}/arrays/files/array/".format(
    os.path.dirname(os.path.realpath(__file__))
)


class TestAPI:
    class TestManual:
        def test__init__input_kernel__all_attributes_correct_including_data_inheritance(
            self
        ):
            kernel = aa.Kernel.manual_2d(
                array=np.ones((3, 3)), pixel_scales=1.0, renormalize=False
            )

            assert kernel.shape_2d == (3, 3)
            assert (kernel.in_2d == np.ones((3, 3))).all()
            assert kernel.pixel_scales == (1.0, 1.0)
            assert kernel.origin == (0.0, 0.0)

            kernel = aa.Kernel.manual_1d(
                array=np.ones((12,)),
                shape_2d=(4, 3),
                pixel_scales=1.0,
                renormalize=False,
            )

            assert kernel.shape_2d == (4, 3)
            assert (kernel.in_2d == np.ones((4, 3))).all()
            assert kernel.pixel_scales == (1.0, 1.0)
            assert kernel.origin == (0.0, 0.0)

    class TestFull:
        def test__kernel_is_set_of_full_values(self):
            kernel = aa.Kernel.full(fill_value=3.0, shape_2d=(3, 3), pixel_scales=1.0)

            assert kernel.shape_2d == (3, 3)
            assert (kernel.in_2d == 3.0 * np.ones((3, 3))).all()
            assert kernel.pixel_scales == (1.0, 1.0)
            assert kernel.origin == (0.0, 0.0)

    class TestOnesZeros:
        def test__kernel_is_set_of_full_values(self):
            kernel = aa.Kernel.ones(shape_2d=(3, 3), pixel_scales=1.0)

            assert kernel.shape_2d == (3, 3)
            assert (kernel.in_2d == np.ones((3, 3))).all()
            assert kernel.pixel_scales == (1.0, 1.0)
            assert kernel.origin == (0.0, 0.0)

            kernel = aa.Kernel.zeros(shape_2d=(3, 3), pixel_scales=1.0)

            assert kernel.shape_2d == (3, 3)
            assert (kernel.in_2d == np.zeros((3, 3))).all()
            assert kernel.pixel_scales == (1.0, 1.0)
            assert kernel.origin == (0.0, 0.0)

    class TestFromFits:
        def test__from_fits__input_kernel_3x3__all_attributes_correct_including_data_inheritance(
            self
        ):
            kernel = aa.Kernel.from_fits(
                file_path=test_data_dir + "3x3_ones.fits", hdu=0, pixel_scales=1.0
            )

            assert (kernel.in_2d == np.ones((3, 3))).all()

            kernel = aa.Kernel.from_fits(
                file_path=test_data_dir + "4x3_ones.fits", hdu=0, pixel_scales=1.0
            )

            assert (kernel.in_2d == np.ones((4, 3))).all()

    class TestFromKernelNoBlurring:
        def test__correct_kernel(self):
            kernel = aa.Kernel.no_blur(pixel_scales=1.0)

            assert (kernel.in_2d == np.array([[1.0]])).all()
            assert kernel.pixel_scales == (1.0, 1.0)

            kernel = aa.Kernel.no_blur(pixel_scales=2.0)

            assert (kernel.in_2d == np.array([[1.0]])).all()
            assert kernel.pixel_scales == (2.0, 2.0)


class TestRenormalize:
    def test__input_is_already_normalized__no_change(self):

        kernel_data = np.ones((3, 3)) / 9.0
        kernel = aa.Kernel.manual_2d(
            array=kernel_data, pixel_scales=1.0, renormalize=True
        )

        assert kernel.in_2d == pytest.approx(kernel_data, 1e-3)

    def test__input_is_above_normalization_so_is_normalized(self):

        kernel_data = np.ones((3, 3))

        kernel = aa.Kernel.manual_2d(
            array=kernel_data, pixel_scales=1.0, renormalize=True
        )

        assert kernel.in_2d == pytest.approx(np.ones((3, 3)) / 9.0, 1e-3)

        kernel = aa.Kernel.manual_2d(
            array=kernel_data, pixel_scales=1.0, renormalize=False
        )

        kernel = kernel.renormalized

        assert kernel.in_2d == pytest.approx(np.ones((3, 3)) / 9.0, 1e-3)

    def test__same_as_above__renomalized_false_does_not_renormalize(self):
        kernel_data = np.ones((3, 3))

        kernel = aa.Kernel.manual_2d(
            array=kernel_data, pixel_scales=1.0, renormalize=False
        )

        assert kernel.in_2d == pytest.approx(np.ones((3, 3)), 1e-3)


class TestBinnedUp:
    def test__kernel_is_even_x_even__rescaled_to_odd_x_odd__no_use_of_dimension_trimming(
        self
    ):
        array_2d = np.ones((6, 6))
        kernel = aa.Kernel.manual_2d(
            array=array_2d, pixel_scales=1.0, renormalize=False
        )
        kernel = kernel.rescaled_with_odd_dimensions_from_rescale_factor(
            rescale_factor=0.5, renormalize=True
        )
        assert kernel.pixel_scales == (2.0, 2.0)
        assert (kernel.in_2d == (1.0 / 9.0) * np.ones((3, 3))).all()

        array_2d = np.ones((9, 9))
        kernel = aa.Kernel.manual_2d(
            array=array_2d, pixel_scales=1.0, renormalize=False
        )
        kernel = kernel.rescaled_with_odd_dimensions_from_rescale_factor(
            rescale_factor=0.333333333333333, renormalize=True
        )
        assert kernel.pixel_scales == (3.0, 3.0)
        assert (kernel.in_2d == (1.0 / 9.0) * np.ones((3, 3))).all()

        array_2d = np.ones((18, 6))
        kernel = aa.Kernel.manual_2d(
            array=array_2d, pixel_scales=1.0, renormalize=False
        )
        kernel = kernel.rescaled_with_odd_dimensions_from_rescale_factor(
            rescale_factor=0.5, renormalize=True
        )
        assert kernel.pixel_scales == (2.0, 2.0)
        assert (kernel.in_2d == (1.0 / 27.0) * np.ones((9, 3))).all()

        array_2d = np.ones((6, 18))
        kernel = aa.Kernel.manual_2d(
            array=array_2d, pixel_scales=1.0, renormalize=False
        )
        kernel = kernel.rescaled_with_odd_dimensions_from_rescale_factor(
            rescale_factor=0.5, renormalize=True
        )
        assert kernel.pixel_scales == (2.0, 2.0)
        assert (kernel.in_2d == (1.0 / 27.0) * np.ones((3, 9))).all()

    def test__kernel_is_even_x_even_after_binning_up__resized_to_odd_x_odd_with_shape_plus_one(
        self
    ):
        array_2d = np.array(np.ones((2, 2)))
        kernel = aa.Kernel.manual_2d(
            array=array_2d, pixel_scales=1.0, renormalize=False
        )
        kernel = kernel.rescaled_with_odd_dimensions_from_rescale_factor(
            rescale_factor=2.0, renormalize=True
        )
        assert kernel.pixel_scales == (0.4, 0.4)
        assert (kernel.in_2d == (1.0 / 25.0) * np.ones((5, 5))).all()

        array_2d = np.ones((40, 40))
        kernel = aa.Kernel.manual_2d(
            array=array_2d, pixel_scales=1.0, renormalize=False
        )
        kernel = kernel.rescaled_with_odd_dimensions_from_rescale_factor(
            rescale_factor=0.1, renormalize=True
        )
        assert kernel.pixel_scales == (8.0, 8.0)
        assert (kernel.in_2d == (1.0 / 25.0) * np.ones((5, 5))).all()

        array_2d = np.ones((2, 4))
        kernel = aa.Kernel.manual_2d(
            array=array_2d, pixel_scales=1.0, renormalize=False
        )
        kernel = kernel.rescaled_with_odd_dimensions_from_rescale_factor(
            rescale_factor=2.0, renormalize=True
        )

        assert kernel.pixel_scales[0] == pytest.approx(0.4, 1.0e-4)
        assert kernel.pixel_scales[1] == pytest.approx(0.4444444, 1.0e-4)
        assert (kernel.in_2d == (1.0 / 45.0) * np.ones((5, 9))).all()

        array_2d = np.array(np.ones((4, 2)))
        kernel = aa.Kernel.manual_2d(
            array=array_2d, pixel_scales=1.0, renormalize=False
        )
        kernel = kernel.rescaled_with_odd_dimensions_from_rescale_factor(
            rescale_factor=2.0, renormalize=True
        )
        assert kernel.pixel_scales[0] == pytest.approx(0.4444444, 1.0e-4)
        assert kernel.pixel_scales[1] == pytest.approx(0.4, 1.0e-4)
        assert (kernel.in_2d == (1.0 / 45.0) * np.ones((9, 5))).all()

    def test__kernel_is_odd_and_even_after_binning_up__resized_to_odd_and_odd_with_shape_plus_one(
        self
    ):
        array_2d = np.array(np.ones((6, 4)))
        kernel = aa.Kernel.manual_2d(
            array=array_2d, pixel_scales=1.0, renormalize=False
        )
        kernel = kernel.rescaled_with_odd_dimensions_from_rescale_factor(
            rescale_factor=0.5, renormalize=True
        )

        assert kernel.pixel_scales == pytest.approx((2.0, 1.3333333333), 1.0e-4)
        assert (kernel.in_2d == (1.0 / 9.0) * np.ones((3, 3))).all()

        array_2d = np.ones((9, 12))
        kernel = aa.Kernel.manual_2d(
            array=array_2d, pixel_scales=1.0, renormalize=False
        )
        kernel = kernel.rescaled_with_odd_dimensions_from_rescale_factor(
            rescale_factor=0.33333333333, renormalize=True
        )

        assert kernel.pixel_scales == pytest.approx((3.0, 2.4), 1.0e-4)
        assert (kernel.in_2d == (1.0 / 15.0) * np.ones((3, 5))).all()

        array_2d = np.array(np.ones((4, 6)))
        kernel = aa.Kernel.manual_2d(
            array=array_2d, pixel_scales=1.0, renormalize=False
        )
        kernel = kernel.rescaled_with_odd_dimensions_from_rescale_factor(
            rescale_factor=0.5, renormalize=True
        )

        assert kernel.pixel_scales == pytest.approx((1.33333333333, 2.0), 1.0e-4)
        assert (kernel.in_2d == (1.0 / 9.0) * np.ones((3, 3))).all()

        array_2d = np.ones((12, 9))
        kernel = aa.Kernel.manual_2d(
            array=array_2d, pixel_scales=1.0, renormalize=False
        )
        kernel = kernel.rescaled_with_odd_dimensions_from_rescale_factor(
            rescale_factor=0.33333333333, renormalize=True
        )
        assert kernel.pixel_scales == pytest.approx((2.4, 3.0), 1.0e-4)
        assert (kernel.in_2d == (1.0 / 15.0) * np.ones((5, 3))).all()


class TestConvolve:
    def test__kernel_is_not_odd_x_odd__raises_error(self):
        kernel = np.array([[0.0, 1.0], [1.0, 2.0]])

        kernel = aa.Kernel.manual_2d(array=kernel, pixel_scales=1.0)

        with pytest.raises(exc.KernelException):
            kernel.convolved_array_from_array(np.ones((5, 5)))

    def test__image_is_3x3_central_value_of_one__kernel_is_cross__blurred_image_becomes_cross(
        self
    ):

        image = aa.Array.manual_2d([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
        kernel = np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]])

        kernel = aa.Kernel.manual_2d(array=kernel, pixel_scales=1.0)

        blurred_image = kernel.convolved_array_from_array(image)

        assert (blurred_image == kernel).all()

    def test__image_is_4x4_central_value_of_one__kernel_is_cross__blurred_image_becomes_cross(
        self
    ):
        image = aa.Array.manual_2d(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )

        kernel = np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]])

        kernel = aa.Kernel.manual_2d(array=kernel, pixel_scales=1.0)

        blurred_image = kernel.convolved_array_from_array(image)

        assert (
            blurred_image.in_2d
            == np.array(
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 2.0, 1.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            )
        ).all()

    def test__image_is_4x3_central_value_of_one__kernel_is_cross__blurred_image_becomes_cross(
        self
    ):
        image = aa.Array.manual_2d(
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )

        kernel = np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]])

        kernel = aa.Kernel.manual_2d(array=kernel, pixel_scales=1.0)

        blurred_image = kernel.convolved_array_from_array(image)

        assert (
            blurred_image.in_2d
            == np.array(
                [[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
            )
        ).all()

    def test__image_is_3x4_central_value_of_one__kernel_is_cross__blurred_image_becomes_cross(
        self
    ):
        image = aa.Array.manual_2d(
            [[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
        )

        kernel = np.array([[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]])

        kernel = aa.Kernel.manual_2d(array=kernel, pixel_scales=1.0)

        blurred_image = kernel.convolved_array_from_array(image)

        assert (
            blurred_image.in_2d
            == np.array(
                [[0.0, 1.0, 0.0, 0.0], [1.0, 2.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
            )
        ).all()

    def test__image_is_4x4_has_two_central_values__kernel_is_asymmetric__blurred_image_follows_convolution(
        self
    ):
        image = aa.Array.manual_2d(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )

        kernel = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 1.0], [1.0, 3.0, 3.0]])

        kernel = aa.Kernel.manual_2d(array=kernel, pixel_scales=1.0)

        blurred_image = kernel.convolved_array_from_array(image)

        assert (
            blurred_image.in_2d
            == np.array(
                [
                    [1.0, 1.0, 1.0, 0.0],
                    [2.0, 3.0, 2.0, 1.0],
                    [1.0, 5.0, 5.0, 1.0],
                    [0.0, 1.0, 3.0, 3.0],
                ]
            )
        ).all()

    def test__image_is_4x4_values_are_on_edge__kernel_is_asymmetric__blurring_does_not_account_for_edge_effects(
        self
    ):
        image = aa.Array.manual_2d(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )

        kernel = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 1.0], [1.0, 3.0, 3.0]])

        kernel = aa.Kernel.manual_2d(array=kernel, pixel_scales=1.0)

        blurred_image = kernel.convolved_array_from_array(image)

        assert (
            blurred_image.in_2d
            == np.array(
                [
                    [1.0, 1.0, 0.0, 0.0],
                    [2.0, 1.0, 1.0, 1.0],
                    [3.0, 3.0, 2.0, 2.0],
                    [0.0, 0.0, 1.0, 3.0],
                ]
            )
        ).all()

    def test__image_is_4x4_values_are_on_corner__kernel_is_asymmetric__blurring_does_not_account_for_edge_effects(
        self
    ):
        image = aa.Array.manual_2d(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        kernel = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 1.0], [1.0, 3.0, 3.0]])

        kernel = aa.Kernel.manual_2d(array=kernel, pixel_scales=1.0)

        blurred_image = kernel.convolved_array_from_array(image)

        assert (
            blurred_image.in_2d
            == np.array(
                [
                    [2.0, 1.0, 0.0, 0.0],
                    [3.0, 3.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 1.0],
                    [0.0, 0.0, 2.0, 2.0],
                ]
            )
        ).all()


class TestFromGaussian:
    def test__identical_to_gaussian_light_profile(self):

        kernel = aa.Kernel.from_gaussian(
            shape_2d=(3, 3),
            pixel_scales=1.0,
            centre=(0.1, 0.1),
            axis_ratio=0.9,
            phi=45.0,
            sigma=1.0,
            renormalize=True,
        )

        assert kernel.in_2d == pytest.approx(
            np.array(
                [
                    [0.06281, 0.13647, 0.0970],
                    [0.11173, 0.21589, 0.136477],
                    [0.065026, 0.11173, 0.06281],
                ]
            ),
            1.0e-3,
        )


class TestFromAlmaGaussian:
    def test__identical_to_astropy_gaussian_model__circular_no_rotation(self):
        pixel_scales = 0.1

        x_stddev = (
            2.0e-5
            * (units.deg).to(units.arcsec)
            / pixel_scales
            / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        )
        y_stddev = (
            2.0e-5
            * (units.deg).to(units.arcsec)
            / pixel_scales
            / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        )

        gaussian_astropy = functional_models.Gaussian2D(
            amplitude=1.0,
            x_mean=2.0,
            y_mean=2.0,
            x_stddev=x_stddev,
            y_stddev=y_stddev,
            theta=0.0,
        )

        shape = (5, 5)
        y, x = np.mgrid[0 : shape[1], 0 : shape[0]]
        kernel_astropy = gaussian_astropy(x, y)
        kernel_astropy /= np.sum(kernel_astropy)

        kernel = aa.Kernel.from_as_gaussian_via_alma_fits_header_parameters(
            shape_2d=shape,
            pixel_scales=pixel_scales,
            y_stddev=2.0e-5,
            x_stddev=2.0e-5,
            theta=0.0,
            renormalize=True,
        )

        assert kernel_astropy == pytest.approx(kernel.in_2d, 1e-4)

    def test__identical_to_astropy_gaussian_model__circular_no_rotation_different_pixel_scale(
        self
    ):
        pixel_scales = 0.02

        x_stddev = (
            2.0e-5
            * (units.deg).to(units.arcsec)
            / pixel_scales
            / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        )
        y_stddev = (
            2.0e-5
            * (units.deg).to(units.arcsec)
            / pixel_scales
            / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        )

        gaussian_astropy = functional_models.Gaussian2D(
            amplitude=1.0,
            x_mean=2.0,
            y_mean=2.0,
            x_stddev=x_stddev,
            y_stddev=y_stddev,
            theta=0.0,
        )

        shape = (5, 5)
        y, x = np.mgrid[0 : shape[1], 0 : shape[0]]
        kernel_astropy = gaussian_astropy(x, y)
        kernel_astropy /= np.sum(kernel_astropy)

        kernel = aa.Kernel.from_as_gaussian_via_alma_fits_header_parameters(
            shape_2d=shape,
            pixel_scales=pixel_scales,
            y_stddev=2.0e-5,
            x_stddev=2.0e-5,
            theta=0.0,
            renormalize=True,
        )

        assert kernel_astropy == pytest.approx(kernel.in_2d, 1e-4)

    def test__identical_to_astropy_gaussian_model__include_ellipticity_from_x_and_y_stddev(
        self
    ):
        pixel_scales = 0.1

        x_stddev = (
            1.0e-5
            * (units.deg).to(units.arcsec)
            / pixel_scales
            / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        )
        y_stddev = (
            2.0e-5
            * (units.deg).to(units.arcsec)
            / pixel_scales
            / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        )

        theta_deg = 0.0
        theta = Angle(theta_deg, "deg").radian

        gaussian_astropy = functional_models.Gaussian2D(
            amplitude=1.0,
            x_mean=2.0,
            y_mean=2.0,
            x_stddev=x_stddev,
            y_stddev=y_stddev,
            theta=theta,
        )

        shape = (5, 5)
        y, x = np.mgrid[0 : shape[1], 0 : shape[0]]
        kernel_astropy = gaussian_astropy(x, y)
        kernel_astropy /= np.sum(kernel_astropy)

        kernel = aa.Kernel.from_as_gaussian_via_alma_fits_header_parameters(
            shape_2d=shape,
            pixel_scales=pixel_scales,
            y_stddev=2.0e-5,
            x_stddev=1.0e-5,
            theta=theta_deg,
            renormalize=True,
        )

        assert kernel_astropy == pytest.approx(kernel.in_2d, 1e-4)

    def test__identical_to_astropy_gaussian_model__include_different_ellipticity_from_x_and_y_stddev(
        self
    ):
        pixel_scales = 0.1

        x_stddev = (
            3.0e-5
            * (units.deg).to(units.arcsec)
            / pixel_scales
            / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        )
        y_stddev = (
            2.0e-5
            * (units.deg).to(units.arcsec)
            / pixel_scales
            / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        )

        theta_deg = 0.0
        theta = Angle(theta_deg, "deg").radian

        gaussian_astropy = functional_models.Gaussian2D(
            amplitude=1.0,
            x_mean=2.0,
            y_mean=2.0,
            x_stddev=x_stddev,
            y_stddev=y_stddev,
            theta=theta,
        )

        shape = (5, 5)
        y, x = np.mgrid[0 : shape[1], 0 : shape[0]]
        kernel_astropy = gaussian_astropy(x, y)
        kernel_astropy /= np.sum(kernel_astropy)

        kernel = aa.Kernel.from_as_gaussian_via_alma_fits_header_parameters(
            shape_2d=shape,
            pixel_scales=pixel_scales,
            y_stddev=2.0e-5,
            x_stddev=3.0e-5,
            theta=theta_deg,
            renormalize=True,
        )

        assert kernel_astropy == pytest.approx(kernel.in_2d, 1e-4)

    def test__identical_to_astropy_gaussian_model__include_rotation_angle_30(self):
        pixel_scales = 0.1

        x_stddev = (
            1.0e-5
            * (units.deg).to(units.arcsec)
            / pixel_scales
            / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        )
        y_stddev = (
            2.0e-5
            * (units.deg).to(units.arcsec)
            / pixel_scales
            / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        )

        theta_deg = 30.0
        theta = Angle(theta_deg, "deg").radian

        gaussian_astropy = functional_models.Gaussian2D(
            amplitude=1.0,
            x_mean=1.0,
            y_mean=1.0,
            x_stddev=x_stddev,
            y_stddev=y_stddev,
            theta=theta,
        )

        shape = (3, 3)
        y, x = np.mgrid[0 : shape[1], 0 : shape[0]]
        kernel_astropy = gaussian_astropy(x, y)
        kernel_astropy /= np.sum(kernel_astropy)

        kernel = aa.Kernel.from_as_gaussian_via_alma_fits_header_parameters(
            shape_2d=shape,
            pixel_scales=pixel_scales,
            y_stddev=2.0e-5,
            x_stddev=1.0e-5,
            theta=theta_deg,
            renormalize=True,
        )

        assert kernel_astropy == pytest.approx(kernel.in_2d, 1e-4)

    def test__identical_to_astropy_gaussian_model__include_rotation_angle_230(self):
        pixel_scales = 0.1

        x_stddev = (
            1.0e-5
            * (units.deg).to(units.arcsec)
            / pixel_scales
            / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        )
        y_stddev = (
            2.0e-5
            * (units.deg).to(units.arcsec)
            / pixel_scales
            / (2.0 * np.sqrt(2.0 * np.log(2.0)))
        )

        theta_deg = 230.0
        theta = Angle(theta_deg, "deg").radian

        gaussian_astropy = functional_models.Gaussian2D(
            amplitude=1.0,
            x_mean=1.0,
            y_mean=1.0,
            x_stddev=x_stddev,
            y_stddev=y_stddev,
            theta=theta,
        )

        shape = (3, 3)
        y, x = np.mgrid[0 : shape[1], 0 : shape[0]]
        kernel_astropy = gaussian_astropy(x, y)
        kernel_astropy /= np.sum(kernel_astropy)

        kernel = aa.Kernel.from_as_gaussian_via_alma_fits_header_parameters(
            shape_2d=shape,
            pixel_scales=pixel_scales,
            y_stddev=2.0e-5,
            x_stddev=1.0e-5,
            theta=theta_deg,
            renormalize=True,
        )

        assert kernel_astropy == pytest.approx(kernel.in_2d, 1e-4)
