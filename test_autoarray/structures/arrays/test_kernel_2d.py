from os import path
import numpy as np
import pytest

from astropy import units
from astropy.modeling import functional_models
from astropy.coordinates import Angle
import autoarray as aa
from autoarray import exc

test_data_dir = path.join("{}".format(path.dirname(path.realpath(__file__))), "files")


def test__full():

    kernel_2d = aa.Kernel2D.full(fill_value=3.0, shape_native=(3, 3), pixel_scales=1.0)

    assert kernel_2d.shape_native == (3, 3)
    assert (kernel_2d.native == 3.0 * np.ones((3, 3))).all()
    assert kernel_2d.pixel_scales == (1.0, 1.0)
    assert kernel_2d.origin == (0.0, 0.0)


def test__ones():
    kernel_2d = aa.Kernel2D.ones(shape_native=(3, 3), pixel_scales=1.0, normalize=False)

    assert kernel_2d.shape_native == (3, 3)
    assert (kernel_2d.native == np.ones((3, 3))).all()
    assert kernel_2d.pixel_scales == (1.0, 1.0)
    assert kernel_2d.origin == (0.0, 0.0)


def test__zeros():

    kernel_2d = aa.Kernel2D.zeros(shape_native=(3, 3), pixel_scales=1.0)

    assert kernel_2d.shape_native == (3, 3)
    assert (kernel_2d.native == np.zeros((3, 3))).all()
    assert kernel_2d.pixel_scales == (1.0, 1.0)
    assert kernel_2d.origin == (0.0, 0.0)


def test__from_fits():
    kernel_2d = aa.Kernel2D.from_fits(
        file_path=path.join(test_data_dir, "3x2_ones.fits"), hdu=0, pixel_scales=1.0
    )

    assert (kernel_2d.native == np.ones((3, 2))).all()

    kernel_2d = aa.Kernel2D.from_fits(
        file_path=path.join(test_data_dir, "3x2_twos.fits"), hdu=0, pixel_scales=1.0
    )

    assert (kernel_2d.native == 2.0 * np.ones((3, 2))).all()


def test__from_fits__loads_and_stores_header_info():
    kernel_2d = aa.Kernel2D.from_fits(
        file_path=path.join(test_data_dir, "3x2_ones.fits"), hdu=0, pixel_scales=1.0
    )

    assert kernel_2d.header.header_sci_obj["BITPIX"] == -64
    assert kernel_2d.header.header_hdu_obj["BITPIX"] == -64

    kernel_2d = aa.Kernel2D.from_fits(
        file_path=path.join(test_data_dir, "3x2_twos.fits"), hdu=0, pixel_scales=1.0
    )

    assert kernel_2d.header.header_sci_obj["BITPIX"] == -64
    assert kernel_2d.header.header_hdu_obj["BITPIX"] == -64


def test__no_blur():
    kernel_2d = aa.Kernel2D.no_blur(pixel_scales=1.0)

    assert (
        kernel_2d.native
        == np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    ).all()
    assert kernel_2d.pixel_scales == (1.0, 1.0)

    kernel_2d = aa.Kernel2D.no_blur(pixel_scales=2.0)

    assert (
        kernel_2d.native
        == np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    ).all()
    assert kernel_2d.pixel_scales == (2.0, 2.0)


def test__from_gaussian():

    kernel_2d = aa.Kernel2D.from_gaussian(
        shape_native=(3, 3),
        pixel_scales=1.0,
        centre=(0.1, 0.1),
        axis_ratio=0.9,
        angle=45.0,
        sigma=1.0,
        normalize=True,
    )

    assert kernel_2d.native == pytest.approx(
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
    kernel_2d = aa.Kernel2D.no_mask(
        values=kernel_data, pixel_scales=1.0, normalize=True
    )

    assert kernel_2d.native == pytest.approx(kernel_data, 1e-3)

    kernel_data = np.ones((3, 3))

    kernel_2d = aa.Kernel2D.no_mask(
        values=kernel_data, pixel_scales=1.0, normalize=True
    )

    assert kernel_2d.native == pytest.approx(np.ones((3, 3)) / 9.0, 1e-3)

    kernel_2d = aa.Kernel2D.no_mask(
        values=kernel_data, pixel_scales=1.0, normalize=False
    )

    kernel_2d = kernel_2d.normalized

    assert kernel_2d.native == pytest.approx(np.ones((3, 3)) / 9.0, 1e-3)

    kernel_data = np.ones((3, 3))

    kernel_2d = aa.Kernel2D.no_mask(
        values=kernel_data, pixel_scales=1.0, normalize=False
    )

    assert kernel_2d.native == pytest.approx(np.ones((3, 3)), 1e-3)


def test__rescaled_with_odd_dimensions_from__evens_to_odds():
    array_2d = np.ones((6, 6))
    kernel_2d = aa.Kernel2D.no_mask(values=array_2d, pixel_scales=1.0, normalize=False)
    kernel_2d = kernel_2d.rescaled_with_odd_dimensions_from(
        rescale_factor=0.5, normalize=True
    )
    assert kernel_2d.pixel_scales == (2.0, 2.0)
    assert (kernel_2d.native == (1.0 / 9.0) * np.ones((3, 3))).all()

    array_2d = np.ones((9, 9))
    kernel_2d = aa.Kernel2D.no_mask(values=array_2d, pixel_scales=1.0, normalize=False)
    kernel_2d = kernel_2d.rescaled_with_odd_dimensions_from(
        rescale_factor=0.333333333333333, normalize=True
    )
    assert kernel_2d.pixel_scales == (3.0, 3.0)
    assert (kernel_2d.native == (1.0 / 9.0) * np.ones((3, 3))).all()

    array_2d = np.ones((18, 6))
    kernel_2d = aa.Kernel2D.no_mask(values=array_2d, pixel_scales=1.0, normalize=False)
    kernel_2d = kernel_2d.rescaled_with_odd_dimensions_from(
        rescale_factor=0.5, normalize=True
    )
    assert kernel_2d.pixel_scales == (2.0, 2.0)
    assert (kernel_2d.native == (1.0 / 27.0) * np.ones((9, 3))).all()

    array_2d = np.ones((6, 18))
    kernel_2d = aa.Kernel2D.no_mask(values=array_2d, pixel_scales=1.0, normalize=False)
    kernel_2d = kernel_2d.rescaled_with_odd_dimensions_from(
        rescale_factor=0.5, normalize=True
    )
    assert kernel_2d.pixel_scales == (2.0, 2.0)
    assert (kernel_2d.native == (1.0 / 27.0) * np.ones((3, 9))).all()


def test__rescaled_with_odd_dimensions_from__different_scalings():

    kernel_2d = aa.Kernel2D.ones(shape_native=(2, 2), pixel_scales=1.0, normalize=False)
    kernel_2d = kernel_2d.rescaled_with_odd_dimensions_from(
        rescale_factor=2.0, normalize=True
    )
    assert kernel_2d.pixel_scales == (0.4, 0.4)
    assert (kernel_2d.native == (1.0 / 25.0) * np.ones((5, 5))).all()

    kernel_2d = aa.Kernel2D.ones(
        shape_native=(40, 40), pixel_scales=1.0, normalize=False
    )
    kernel_2d = kernel_2d.rescaled_with_odd_dimensions_from(
        rescale_factor=0.1, normalize=True
    )
    assert kernel_2d.pixel_scales == (8.0, 8.0)
    assert (kernel_2d.native == (1.0 / 25.0) * np.ones((5, 5))).all()

    kernel_2d = aa.Kernel2D.ones(shape_native=(2, 4), pixel_scales=1.0, normalize=False)
    kernel_2d = kernel_2d.rescaled_with_odd_dimensions_from(
        rescale_factor=2.0, normalize=True
    )

    assert kernel_2d.pixel_scales[0] == pytest.approx(0.4, 1.0e-4)
    assert kernel_2d.pixel_scales[1] == pytest.approx(0.4444444, 1.0e-4)
    assert (kernel_2d.native == (1.0 / 45.0) * np.ones((5, 9))).all()

    kernel_2d = aa.Kernel2D.ones(shape_native=(4, 2), pixel_scales=1.0, normalize=False)
    kernel_2d = kernel_2d.rescaled_with_odd_dimensions_from(
        rescale_factor=2.0, normalize=True
    )
    assert kernel_2d.pixel_scales[0] == pytest.approx(0.4444444, 1.0e-4)
    assert kernel_2d.pixel_scales[1] == pytest.approx(0.4, 1.0e-4)
    assert (kernel_2d.native == (1.0 / 45.0) * np.ones((9, 5))).all()

    kernel_2d = aa.Kernel2D.ones(shape_native=(6, 4), pixel_scales=1.0, normalize=False)
    kernel_2d = kernel_2d.rescaled_with_odd_dimensions_from(
        rescale_factor=0.5, normalize=True
    )

    assert kernel_2d.pixel_scales == pytest.approx((2.0, 1.3333333333), 1.0e-4)
    assert (kernel_2d.native == (1.0 / 9.0) * np.ones((3, 3))).all()

    kernel_2d = aa.Kernel2D.ones(
        shape_native=(9, 12), pixel_scales=1.0, normalize=False
    )
    kernel_2d = kernel_2d.rescaled_with_odd_dimensions_from(
        rescale_factor=0.33333333333, normalize=True
    )

    assert kernel_2d.pixel_scales == pytest.approx((3.0, 2.4), 1.0e-4)
    assert (kernel_2d.native == (1.0 / 15.0) * np.ones((3, 5))).all()

    kernel_2d = aa.Kernel2D.ones(shape_native=(4, 6), pixel_scales=1.0, normalize=False)
    kernel_2d = kernel_2d.rescaled_with_odd_dimensions_from(
        rescale_factor=0.5, normalize=True
    )

    assert kernel_2d.pixel_scales == pytest.approx((1.33333333333, 2.0), 1.0e-4)
    assert (kernel_2d.native == (1.0 / 9.0) * np.ones((3, 3))).all()

    kernel_2d = aa.Kernel2D.ones(
        shape_native=(12, 9), pixel_scales=1.0, normalize=False
    )
    kernel_2d = kernel_2d.rescaled_with_odd_dimensions_from(
        rescale_factor=0.33333333333, normalize=True
    )
    assert kernel_2d.pixel_scales == pytest.approx((2.4, 3.0), 1.0e-4)
    assert (kernel_2d.native == (1.0 / 15.0) * np.ones((5, 3))).all()


def test__convolved_array_from():

    array_2d = aa.Array2D.no_mask(
        values=[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], pixel_scales=1.0
    )

    kernel_2d = aa.Kernel2D.no_mask(
        values=[[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]], pixel_scales=1.0
    )

    blurred_array_2d = kernel_2d.convolved_array_from(array_2d)

    assert (blurred_array_2d == kernel_2d).all()

    array_2d = aa.Array2D.no_mask(
        values=[
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        pixel_scales=1.0,
    )

    kernel_2d = aa.Kernel2D.no_mask(
        values=[[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]], pixel_scales=1.0
    )

    blurred_array_2d = kernel_2d.convolved_array_from(array=array_2d)

    assert (
        blurred_array_2d.native
        == np.array(
            [
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 2.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ]
        )
    ).all()

    array_2d = aa.Array2D.no_mask(
        values=[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        pixel_scales=1.0,
    )

    kernel_2d = aa.Kernel2D.no_mask(
        values=[[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]], pixel_scales=1.0
    )

    blurred_array_2d = kernel_2d.convolved_array_from(array_2d)

    assert (
        blurred_array_2d.native
        == np.array(
            [[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
        )
    ).all()

    array_2d = aa.Array2D.no_mask(
        values=[[0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
        pixel_scales=1.0,
    )

    kernel_2d = aa.Kernel2D.no_mask(
        values=[[0.0, 1.0, 0.0], [1.0, 2.0, 1.0], [0.0, 1.0, 0.0]], pixel_scales=1.0
    )

    blurred_array_2d = kernel_2d.convolved_array_from(array_2d)

    assert (
        blurred_array_2d.native
        == np.array([[0.0, 1.0, 0.0, 0.0], [1.0, 2.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    ).all()

    array_2d = aa.Array2D.no_mask(
        values=[
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        pixel_scales=1.0,
    )

    kernel_2d = aa.Kernel2D.no_mask(
        values=[[1.0, 1.0, 1.0], [2.0, 2.0, 1.0], [1.0, 3.0, 3.0]], pixel_scales=1.0
    )

    blurred_array_2d = kernel_2d.convolved_array_from(array_2d)

    assert (
        blurred_array_2d.native
        == np.array(
            [
                [1.0, 1.0, 1.0, 0.0],
                [2.0, 3.0, 2.0, 1.0],
                [1.0, 5.0, 5.0, 1.0],
                [0.0, 1.0, 3.0, 3.0],
            ]
        )
    ).all()

    array_2d = aa.Array2D.no_mask(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        pixel_scales=1.0,
    )

    kernel_2d = aa.Kernel2D.no_mask(
        values=[[1.0, 1.0, 1.0], [2.0, 2.0, 1.0], [1.0, 3.0, 3.0]], pixel_scales=1.0
    )

    blurred_array_2d = kernel_2d.convolved_array_from(array_2d)

    assert (
        blurred_array_2d.native
        == np.array(
            [
                [1.0, 1.0, 0.0, 0.0],
                [2.0, 1.0, 1.0, 1.0],
                [3.0, 3.0, 2.0, 2.0],
                [0.0, 0.0, 1.0, 3.0],
            ]
        )
    ).all()

    array_2d = aa.Array2D.no_mask(
        values=[
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        pixel_scales=1.0,
    )

    kernel_2d = aa.Kernel2D.no_mask(
        values=[[1.0, 1.0, 1.0], [2.0, 2.0, 1.0], [1.0, 3.0, 3.0]], pixel_scales=1.0
    )

    blurred_array_2d = kernel_2d.convolved_array_from(array_2d)

    assert (
        blurred_array_2d.native
        == np.array(
            [
                [2.0, 1.0, 0.0, 0.0],
                [3.0, 3.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 2.0, 2.0],
            ]
        )
    ).all()


def test__convolved_array_from__not_odd_x_odd_kernel__raises_error():

    kernel_2d = aa.Kernel2D.no_mask(values=[[0.0, 1.0], [1.0, 2.0]], pixel_scales=1.0)

    with pytest.raises(exc.KernelException):
        kernel_2d.convolved_array_from(np.ones((5, 5)))


def test__from_as_gaussian_via_alma_fits_header_parameters__identical_to_astropy_gaussian_model():
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

    kernel_2d = aa.Kernel2D.from_as_gaussian_via_alma_fits_header_parameters(
        shape_native=shape,
        pixel_scales=pixel_scales,
        y_stddev=2.0e-5,
        x_stddev=1.0e-5,
        theta=theta_deg,
        normalize=True,
    )

    assert kernel_astropy == pytest.approx(kernel_2d.native, 1e-4)
